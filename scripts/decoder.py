"""
Steganographic decoder with block-wise Reed-Solomon error correction.

Extracts hidden bits from image patches using arithmetic coding, but applies
RS error correction *incrementally* per RS block — not on the entire message
at once.  Each fixed-size RS block (10 payload bytes + 6 ECC bytes) is decoded
as soon as enough bits are received.  When RS corrects errors within a block,
the corresponding patches are re-selected to prevent context poisoning of
subsequent patches.

Constants (must match pipeline):
    RS_BLOCK_BYTES = 10      # payload bytes per RS block
    RS_NSYM = 6              # ECC parity symbols per block (30% correction)
    RS_PAYLOAD_BITS = 80     # 10 × 8
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, FrozenSet

import numpy as np
import torch
from torchvision.io import read_image
from tqdm.auto import tqdm

from main import DataModuleFromConfig
from scripts.input import Options
from scripts.utils import (
    DEFAULT_CONTEXT_ROWS, set_context, int2bits, build_context_from_patches,
    DEFAULT_PRECISION_BITS, DEFAULT_CODEBOOK_SIZE, PATCH_SIZE,
)
from scripts.logger import logger
from scripts.stats import DecodingStatistics

# We no longer import ErrorCorrectionFactory — use ReedSolomonCorrectionCode directly
from scripts.error_correction import ReedSolomonCorrectionCode
from scripts.pipeline import HEADER_LENGTH_BITS, XorMask

# Block-wise RS parameters (must match pipeline.py)
# nsym=6 → corrects up to 3 bytes out of 10 = 30% correction capability
RS_NSYM = 6
RS_BLOCK_BYTES = 10
RS_PAYLOAD_BITS = RS_BLOCK_BYTES * 8  # 80

# How many bits after XOR undo are needed for one RS block (variable due to 3-bit padding header)
# Maximum possible: 3 padding header bits + (16 bytes * 8) = 131 bits
# Minimum possible: 3 padding header bits + (1 byte * 8) = 11 bits (bad case)
RS_ENCODED_MAX_BITS = 3 + (RS_BLOCK_BYTES + RS_NSYM) * 8  # 131
RS_ENCODED_MIN_BITS = 3 + 8  # 11


# ======================================================================
#  Per-patch record for deferred correction
# ======================================================================

@dataclass
class PatchRecord:
    """Stores everything we need to re-select a patch after RS correction."""
    row: int
    col: int
    actual_token: int
    selected_idx: int
    token_bits: str
    range_bottom: int
    range_top: int


# ======================================================================
#  Bit ↔ byte helpers
# ======================================================================

def bits_to_bytes(bits: str) -> bytes:
    """Convert a string of '0'/'1' up to the last complete byte boundary."""
    valid_len = (len(bits) // 8) * 8
    if valid_len == 0:
        return b""
    return int(bits[:valid_len], 2).to_bytes(valid_len // 8, 'big')


def bytes_to_bits(data: bytes) -> str:
    """Convert bytes back to a string of '0'/'1'."""
    return "".join(format(b, "08b") for b in data)


# ======================================================================
#  Patch re-selection
# ======================================================================

def _reselect_patch(
    model: torch.nn.Module,
    reference_indices: torch.Tensor,
    building_indices: torch.Tensor,
    patch: PatchRecord,
    corrected_bits_for_patch: str,
    codebook_len: int = DEFAULT_CODEBOOK_SIZE,
) -> int:
    """
    Given the *corrected* bits for a patch, re-run the arithmetic coding
    selection to find which codebook index the *encoder* would have chosen.
    """
    # Save and clear this patch's index to rebuild original context
    original_val = building_indices[patch.row, patch.col].item()
    building_indices[patch.row, patch.col] = 0

    context, (local_row, local_col) = build_context_from_patches(
        reference_indices, building_indices, patch.row, patch.col,
        (building_indices.shape[0], building_indices.shape[1]),
    )

    # Restore (will be updated with the correct index below)
    building_indices[patch.row, patch.col] = original_val

    model.eval()
    with torch.no_grad():
        logits, _ = model.transformer(context[:-1].unsqueeze(0))
        logits = logits[:, -256:, :].squeeze()
        logits = logits.reshape(PATCH_SIZE, PATCH_SIZE, -1)
        logits = logits[local_row, local_col, :].double()
        logits, indices = logits.sort(descending=True)
        probs = torch.nn.functional.softmax(logits, dim=-1)

    prob_threshold = 1 / codebook_len
    k = min(max(2, torch.nonzero(probs < prob_threshold)[0].item()), codebook_len)
    probs_int = probs[:k]
    probs_int = (probs_int / probs_int.sum() * codebook_len).round().long()
    cumulative_probs = probs_int.cumsum(0)

    overfill = torch.nonzero(cumulative_probs > codebook_len)
    if len(overfill) > 0:
        cumulative_probs = cumulative_probs[:overfill[0]]
    cumulative_probs += codebook_len - cumulative_probs[-1]

    message_str = (corrected_bits_for_patch + "0" * DEFAULT_PRECISION_BITS)[:DEFAULT_PRECISION_BITS]
    message_value = int(message_str, 2)
    selection = torch.nonzero(cumulative_probs > message_value)[0].item()

    return indices[selection].item()


# ======================================================================
#  Block-wise RS management
# ======================================================================

class RsBlockAccumulator:
    """
    Accumulates bits for a single RS block and attempts decode when enough
    bits are received.  Tracks which patch(es) contributed to this block.

    Because RS-encoded blocks have a variable size (3-bit padding header +
    encoded bytes), we use a heuristic: we try to decode as soon as we
    have at least *RS_ENCODED_MIN_BITS* and then extend the candidate
    window until decode succeeds.
    """

    def __init__(self, ecc: ReedSolomonCorrectionCode) -> None:
        self.ecc = ecc
        self.accumulated_bits: List[str] = []       # XOR-undoed bits for this block
        self.patch_indices: List[int] = []           # global patch indices in this block
        self.bit_length: int = 0                     # bits accumulated so far

    def add_bits(self, bits: str, patch_global_idx: int) -> None:
        """Append bits (XOR already undone) and record which patch they came from."""
        self.accumulated_bits.append(bits)
        self.patch_indices.append(patch_global_idx)
        self.bit_length += len(bits)

    @property
    def raw_bits(self) -> str:
        """All accumulated bits concatenated."""
        return "".join(self.accumulated_bits)

    @property
    def is_complete(self) -> bool:
        """
        Check if we have enough bits for a complete RS block.
        We can decide based on the accumulated length.
        """
        return self.bit_length >= RS_ENCODED_MIN_BITS and self._try_decode() is not None

    def try_decode(self) -> Optional[str]:
        """
        Attempt to RS-decode the accumulated bits.  Returns the decoded
        payload (without RS overhead) on success, or None on failure.
        """
        return self._try_decode()

    def _try_decode(self) -> Optional[str]:
        """Internal: try to RS-decode, return decoded payload bits or None."""
        raw = self.raw_bits
        if len(raw) < RS_ENCODED_MIN_BITS:
            return None
        try:
            decoded = self.ecc.decode(raw)
            if decoded:
                return decoded
        except Exception:
            pass
        return None


# ======================================================================
#  Decoder
# ======================================================================

def load_image(path: str) -> torch.Tensor:
    img = read_image(path)
    img = (img.type(torch.float32) / 127.5) - 1.0
    img = img.to('cuda')
    return img


class SteganographyDecoder:
    """
    Handles the decoding of hidden messages from steganographic images using arithmetic coding.
    This class extracts embedded bits from image patches and reconstructs the original message.

    RS error correction is applied *block-wise* as bits are decoded from patches.
    Each 10-byte RS block is decoded independently.  If RS detects and corrects errors
    within a block, the affected patch indices in the ``building_tensor`` are re-selected,
    preventing context poisoning of future patches.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dsets: DataModuleFromConfig,
        context_fraction: float = DEFAULT_CONTEXT_ROWS,
    ):
        self.model = model
        self.dsets = dsets
        self.context_fraction = context_fraction

    @torch.no_grad()
    def decode_token_from_patch(
            self,
            context: torch.Tensor,
            actual_token: int,
            position: Tuple[int, int],
            *,
            codebook_len: int = DEFAULT_CODEBOOK_SIZE,
            top_k: Optional[int] = None,
    ) -> Tuple[int, str, int, int]:
        """Decode hidden bits from a token in a patch."""
        top_k = top_k or codebook_len
        self.model.eval()

        logits, _ = self.model.transformer(context[:-1].unsqueeze(0))
        logits = logits[:, -256:, :].squeeze()
        logits = logits.reshape(PATCH_SIZE, PATCH_SIZE, -1)
        logits = logits[position[0], position[1], :].double()

        logits, indices = logits.sort(descending=True)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        prob_threshold = 1 / codebook_len
        k = min(max(2, torch.nonzero(probs < prob_threshold)[0].item()), top_k)
        probs_int = probs[:k]

        probs_int = (probs_int / probs_int.sum() * codebook_len).round().long()
        cumulative_probs = probs_int.cumsum(0)

        overfill_index = torch.nonzero(cumulative_probs > codebook_len)
        if len(overfill_index) > 0:
            cumulative_probs = cumulative_probs[:overfill_index[0]]
        cumulative_probs += codebook_len - cumulative_probs[-1]

        matching = torch.nonzero(indices[:k] == actual_token).squeeze()

        if matching.numel() == 0:
            logger.warning(f"Actual token {actual_token} is not in the top_k distribution.")
            return -1, "", 0, 0

        if matching.dim() == 0:
            selection = matching.item()
        else:
            logger.warning(f"Multiple matches for actual_token {actual_token}.")
            return -1, "", 0, 0

        if selection >= len(cumulative_probs) or selection < 0:
            logger.warning(f"Selection out of range: {selection}.")
            return -1, "", 0, 0

        range_bottom = cumulative_probs[selection - 1].item() if selection > 0 else 0
        range_top = cumulative_probs[selection].item()

        bottom_bits = int2bits(range_bottom, DEFAULT_PRECISION_BITS)
        top_bits = int2bits(range_top - 1, DEFAULT_PRECISION_BITS)

        decoded_bits = ""
        for b, t in zip(bottom_bits, top_bits):
            if b == t:
                decoded_bits += str(b)
            else:
                break

        return indices[selection].item(), decoded_bits, range_bottom, range_top

    @torch.no_grad()
    def decode_patch(
            self,
            reference_indices: torch.Tensor,
            building_indices: torch.Tensor,
            current_row: int,
            current_col: int,
            grid_shape: Tuple[int, int],
            stego_indices: torch.Tensor,
    ) -> Tuple[int, int, str, int, int, int]:
        """Decode a single patch and advance position."""
        context, (local_row, local_col) = build_context_from_patches(
            reference_indices, building_indices, current_row, current_col, grid_shape
        )

        actual_token = stego_indices[current_row, current_col].item()
        selected_idx, decoded_bits, range_bottom, range_top = self.decode_token_from_patch(
            context, actual_token, (local_row, local_col)
        )

        if selected_idx < 0:
            selected_idx = actual_token

        building_indices[current_row, current_col] = selected_idx

        next_col = current_col + 1
        next_row = current_row
        if next_col >= grid_shape[1]:
            next_col = 0
            next_row += 1

        return next_row, next_col, decoded_bits, selected_idx, range_bottom, range_top

    @torch.no_grad()
    def decode_message(
            self,
            options: Options,
            image: torch.Tensor,
            xor_mask: Optional['XorMask'] = None,
    ) -> Tuple[str, str, List[int], DecodingStatistics]:
        """
        Decode the hidden message from a steganographic image.

        Block-wise RS error correction is applied incrementally:
        bits are accumulated until a complete RS block can be decoded
        (a maximum of 123 bits after XOR undo).  When a block is successfully
        decoded, any corrected bits trigger re-selection of the corresponding patches.

        Args:
            options: Configuration options.
            image: The stego-image tensor.
            xor_mask: Optional pre-built XorMask. If None, builds one from options.xor_key.

        Returns:
            Tuple of (raw_bits, raw_bits, selected_indices, stats)
            where raw_bits are the bits as they came out of arithmetic coding
            (with XOR still applied). The pipeline will undo XOR later.
            The second element is also raw_bits (kept for backward compatibility).
        """
        # ------------------------------------------------------------------
        #  Setup
        # ------------------------------------------------------------------
        base_tensor, cond_tensor = set_context(self.model, self.dsets, DEFAULT_CONTEXT_ROWS)
        image_translations, image_indices = self.model.encode_to_z(base_tensor.unsqueeze(0))
        cond_translations, cond_indices = self.model.encode_to_c(cond_tensor)

        grid_shape = (image_translations.shape[2], image_translations.shape[3])
        reference_tensor = cond_indices.reshape(
            cond_translations.shape[0], cond_translations.shape[2], cond_translations.shape[3]
        ).squeeze()

        building_tensor = image_indices.clone()
        building_tensor = building_tensor.reshape(grid_shape)
        # Start from (0, 0) — decode the entire grid
        current_row = 0
        current_col = 0

        height_crop = grid_shape[0] * PATCH_SIZE
        width_crop = grid_shape[1] * PATCH_SIZE
        image = image[:, :height_crop, :width_crop]
        stego_indices = self.model.encode_to_z(image.unsqueeze(0))[1].reshape(grid_shape)

        # Build XOR mask from options if not provided
        if xor_mask is None:
            xor_key = getattr(options, "xor_key", 123)
            xor_mask = XorMask(xor_key)

        # Fixed RS ECC — no user configuration
        ecc = ReedSolomonCorrectionCode(nsym=RS_NSYM)

        # ------------------------------------------------------------------
        #  Buffers
        # ------------------------------------------------------------------
        patch_records: List[PatchRecord] = []
        header_decoded = False
        message_bit_length: int = 0
        header_payload_bits: int = 0   # bits of header + message payload (before RS)
        corrections_locked = False
        seen_patch_fingerprints: Set[FrozenSet[int]] = set()
        ENOUGH_BIT_MARGIN = 16

        # Block-wise RS accumulation
        current_rs_block = RsBlockAccumulator(ecc)
        decoded_payload_blocks: List[str] = []  # decoded payload bits across all blocks
        total_accumulated_bits: List[str] = []  # all XOR-undoed bits (for header detection + full trace)

        total_remaining = (grid_shape[0] - current_row) * grid_shape[1] - current_col
        with tqdm(total=total_remaining, desc="Decoding message", disable=True) as pbar:
            decoded_bits = ""
            selected_indices_list = []
            stats = DecodingStatistics()
            patch_index = 0

            while current_row < grid_shape[0]:
                prev_row = current_row
                prev_col = current_col

                (current_row, current_col,
                 token_bits, selected_idx, range_bottom, range_top) = self.decode_patch(
                    reference_tensor, building_tensor,
                    current_row, current_col,
                    grid_shape, stego_indices,
                )

                actual_token = stego_indices[prev_row, prev_col].item()
                success = selected_idx != -1

                # Accumulate raw bits (with XOR still applied — returned at end)
                decoded_bits += token_bits

                # Undo XOR for RS processing
                bits_plain = xor_mask.apply(token_bits)

                # Store in global accumulation buffer
                total_accumulated_bits.append(bits_plain)

                # Add to current RS block accumulator
                current_rs_block.add_bits(bits_plain, patch_index)

                # Store patch record
                patch_rec = PatchRecord(
                    row=prev_row, col=prev_col,
                    actual_token=actual_token,
                    selected_idx=selected_idx,
                    token_bits=token_bits,
                    range_bottom=range_bottom,
                    range_top=range_top,
                )
                patch_records.append(patch_rec)

                if success:
                    selected_indices_list.append(selected_idx)
                stats.add_patch_stat(
                    patch_index, prev_row, prev_col,
                    actual_token, range_bottom, range_top,
                    token_bits, success,
                )
                pbar.update(1)
                patch_index += 1

                # ----------------------------------------------------------
                #  Phase 1 — Try to decode current RS block
                # ----------------------------------------------------------
                decoded_payload = current_rs_block.try_decode()
                if decoded_payload is not None:
                    encoded_bits = current_rs_block.raw_bits
                    logger.info(
                        f"[Decoder] RS block {len(decoded_payload_blocks)} decoded: "
                        f"{len(encoded_bits)}b → {len(decoded_payload)}b "
                        f"({len(current_rs_block.patch_indices)} patches)"
                    )

                    # Check if corrections were made
                    re_encoded = ecc.encode(decoded_payload)
                    if re_encoded != encoded_bits[:len(re_encoded)] and not corrections_locked:
                        n_diff = sum(
                            1 for a, b in zip(encoded_bits[:len(re_encoded)], re_encoded)
                            if a != b
                        )
                        logger.info(
                            f"[Decoder] RS block corrected {n_diff} bit errors — "
                            f"re-selecting patches"
                        )
                        self._fix_patches_from_rs_block(
                            reference_tensor, building_tensor,
                            patch_records[:patch_index],
                            current_rs_block.patch_indices,
                            encoded_bits, re_encoded,
                            selected_indices_list,
                            seen_patch_fingerprints,
                        )

                    # Store decoded payload and reset block accumulator
                    decoded_payload_blocks.append(decoded_payload)
                    current_rs_block = RsBlockAccumulator(ecc)

                # ----------------------------------------------------------
                #  Phase 2 — Extract header from first decoded block
                # ----------------------------------------------------------
                if not header_decoded and len(decoded_payload_blocks) > 0:
                    # Concatenate all decoded payload blocks so far
                    full_decoded = "".join(decoded_payload_blocks)
                    if len(full_decoded) >= HEADER_LENGTH_BITS:
                        decoded_len = int(full_decoded[:HEADER_LENGTH_BITS], 2)
                        max_bits = (1 << HEADER_LENGTH_BITS) - 1
                        if 0 < decoded_len <= max_bits:
                            message_bit_length = decoded_len
                            header_decoded = True
                            # Calculate how many body bits are needed (before RS)
                            # We need: header (13) + message bits + padding to byte boundary
                            needed_payload_bits = (
                                (decoded_len + HEADER_LENGTH_BITS + 7) // 8
                            ) * 8
                            # Each RS block contributes RS_PAYLOAD_BITS of decoded payload
                            # (last block may be smaller)
                            header_payload_bits = needed_payload_bits
                            logger.info(
                                f"[Decoder] Header decoded: message length = {decoded_len} bits, "
                                f"need {needed_payload_bits} payload bits"
                            )

                # ----------------------------------------------------------
                #  Phase 3 — Early termination
                # ----------------------------------------------------------
                if header_decoded and header_payload_bits > 0:
                    full_decoded = "".join(decoded_payload_blocks)
                    if len(full_decoded) >= header_payload_bits - ENOUGH_BIT_MARGIN:
                        logger.info(
                            f"[Decoder] Reached target payload "
                            f"({len(full_decoded)} >= {header_payload_bits} - "
                            f"{ENOUGH_BIT_MARGIN}) — stopping"
                        )
                        break

        # Return raw bits (with XOR) — pipeline will undo XOR later
        return decoded_bits, decoded_bits, selected_indices_list, stats

    # ------------------------------------------------------------------
    #  Helper: fix patches from RS block correction
    # ------------------------------------------------------------------

    def _fix_patches_from_rs_block(
        self,
        reference_tensor: torch.Tensor,
        building_tensor: torch.Tensor,
        patch_records: List[PatchRecord],
        block_patch_indices: List[int],
        original_encoded_bits: str,
        re_encoded_bits: str,
        selected_indices_list: List[int],
        seen_patch_fingerprints: Set[FrozenSet[int]],
    ) -> None:
        """
        Compare original (received but XOR-undoed) RS-encoded bits with the
        re-encoded version after RS decode+encode.  If bits differ, re-select
        the affected patches.
        """
        min_len = min(len(original_encoded_bits), len(re_encoded_bits))
        if min_len == 0:
            return

        diff_positions = [
            i for i in range(min_len)
            if original_encoded_bits[i] != re_encoded_bits[i]
        ]
        if not diff_positions:
            return

        logger.info(
            f"[Decoder] RS correction: {len(diff_positions)} bit differences "
            f"over {min_len} bits — fixing patches in block"
        )

        # Build cumulative bit offset map for patches within this block
        block_patch_map = []
        bit_offset = 0
        for pi in block_patch_indices:
            if pi < len(patch_records):
                pr = patch_records[pi]
                aligned_len = len(pr.token_bits)  # same as bits_plain
                block_patch_map.append((pi, bit_offset, bit_offset + aligned_len))
                bit_offset += aligned_len

        # Find which patches overlap with corrected regions within this block
        patches_to_fix = set()
        for pi, start, end in block_patch_map:
            for dp in diff_positions:
                if start <= dp < end:
                    patches_to_fix.add(pi)
                    break

        if not patches_to_fix:
            return

        # Anti-loop fingerprint
        patch_fingerprint = frozenset(sorted(patches_to_fix))
        if patch_fingerprint in seen_patch_fingerprints:
            logger.warning(
                f"[Decoder] Same patches {sorted(patches_to_fix)} "
                f"keep getting corrected — locking further corrections"
            )
            return
        seen_patch_fingerprints.add(patch_fingerprint)

        logger.info(
            f"[Decoder] Re-selecting {len(patches_to_fix)} patches "
            f"(indices {sorted(patches_to_fix)})"
        )

        for pi in sorted(patches_to_fix):
            if pi >= len(patch_records):
                continue
            pr = patch_records[pi]
            _, body_start, _ = block_patch_map[block_patch_indices.index(pi)]
            corrected_patch_bits = re_encoded_bits[
                body_start : body_start + len(pr.token_bits)
            ]

            if corrected_patch_bits == pr.token_bits:
                continue

            new_idx = _reselect_patch(
                self.model, reference_tensor, building_tensor,
                pr, corrected_patch_bits,
            )
            logger.info(
                f"[Decoder] Patch ({pr.row},{pr.col}): "
                f"old_idx={pr.selected_idx} → new_idx={new_idx}, "
                f"bits '{pr.token_bits}' → '{corrected_patch_bits}'"
            )

            if new_idx != pr.selected_idx:
                building_tensor[pr.row, pr.col] = new_idx
                pr.selected_idx = new_idx
                pr.token_bits = corrected_patch_bits
                if pi < len(selected_indices_list):
                    selected_indices_list[pi] = new_idx


# ======================================================================
#  Convenience function
# ======================================================================

def decode_message(
        options: Options,
        model: torch.nn.Module,
        dsets: DataModuleFromConfig,
        image: torch.Tensor
) -> Tuple[str, str, List[int], DecodingStatistics]:
    decoder = SteganographyDecoder(model, dsets, context_fraction=options.context_fraction)
    return decoder.decode_message(options, image)