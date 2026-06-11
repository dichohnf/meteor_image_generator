"""
Steganographic decoder with interleaved Reed-Solomon error correction.

Extracts hidden bits from image patches using arithmetic coding, but applies
RS error correction *incrementally* as bits are decoded — not after the fact. 
This prevents a single corrupted patch from poisoning the context for all
subsequent patches.
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
from scripts.error_correction import ErrorCorrectionFactory, ErrorCorrectionCode
from scripts.pipeline import HEADER_LENGTH_BITS, XorMask


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

    RS error correction is applied *incrementally* as bits are decoded from patches.
    If the RS decoder detects and corrects errors, the affected patch indices in the
    ``building_tensor`` are also corrected, preventing context poisoning of future patches.
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

        RS error correction is applied *incrementally* as bits are decoded.
        When a block of bytes is successfully RS-corrected, the corresponding
        patches are re-selected so that the context remains correct.

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

        # Build ECC decoder
        ecc: ErrorCorrectionCode = ErrorCorrectionFactory.create(
            getattr(options, "error_correction_method", "reed_solomon"),
            nsym=getattr(options, "rs_nsym", 30),
        )

        # ------------------------------------------------------------------
        #  Buffers
        # ------------------------------------------------------------------
        all_bits_aligned: List[str] = []       # bits after XOR undo = RS-encoded body (header inside)
        patch_records: List[PatchRecord] = []
        header_decoded = False
        message_bit_length: int = 0
        target_body_len: int = 0               # bits of body (before RS) needed to reach decoded_len
        corrections_locked = False             # Task 4: anti-loop fingerprint
        seen_patch_fingerprints: Set[FrozenSet[int]] = set()
        BIT_MARGIN = 16                        # slack for target_body_len calculation

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

                # Accumulate raw bits
                decoded_bits += token_bits

                # Undo XOR and accumulate aligned bits
                bits_plain = xor_mask.apply(token_bits)
                all_bits_aligned.append(bits_plain)

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
                #  Phase 1 — Extract header from RS-decoded body (once)
                # ----------------------------------------------------------
                if not header_decoded:
                    body_bits = "".join(all_bits_aligned)
                    # Need at least nsym+1 bytes + 3 bits RS header to attempt decode
                    if len(body_bits) >= 8 * (ecc.nsym + 1) + 3:
                        try:
                            decoded_body = ecc.decode(body_bits)
                            decoded_len = int(decoded_body[:HEADER_LENGTH_BITS], 2)
                            max_bits = (1 << HEADER_LENGTH_BITS) - 1
                            if 0 < decoded_len <= max_bits:
                                message_bit_length = decoded_len
                                header_decoded = True
                                # Calculate how many body bits (before RS) are needed
                                rs_payload_bits = ((decoded_len + HEADER_LENGTH_BITS + 7) // 8) * 8 + 3
                                target_body_len = len(ecc.encode("0" * rs_payload_bits))
                                logger.info(
                                    f"[Decoder] Header decoded from RS body: "
                                    f"message length = {decoded_len} bits, "
                                    f"target body = {target_body_len} bits"
                                )
                            else:
                                logger.warning(
                                    f"[Decoder] Invalid header length {decoded_len}"
                                )
                        except Exception:
                            pass  # keep accumulating bits
                    continue

                # ----------------------------------------------------------
                #  Phase 2 — Early termination if we have enough bits
                # ----------------------------------------------------------
                if header_decoded and target_body_len > 0:
                    bits_so_far = len("".join(all_bits_aligned))
                    if bits_so_far >= target_body_len - BIT_MARGIN:
                        logger.info(
                            f"[Decoder] Reached target body length "
                            f"({bits_so_far} >= {target_body_len} - {BIT_MARGIN}) — stopping"
                        )
                        break

                # ----------------------------------------------------------
                #  Phase 3 — RS correction interleaved
                #
                #  Try to ECC-decode the accumulated body bits.
                #  If it succeeds and errors were found, re-select patches.
                # ----------------------------------------------------------
                body_bits = "".join(all_bits_aligned)

                if len(body_bits) >= 8 * (ecc.nsym + 1) + 3 and not corrections_locked:
                    try:
                        decoded_body = ecc.decode(body_bits)
                        logger.info(
                            f"[Decoder] RS decode successful "
                            f"({len(body_bits)} body bits → {len(decoded_body)} msg bits)"
                        )
                        corrected_body = ecc.encode(decoded_body)
                        min_len = min(len(body_bits), len(corrected_body))
                        orig_slice = body_bits[:min_len]
                        corr_slice = corrected_body[:min_len]

                        if orig_slice != corr_slice:
                            n_errors = sum(1 for a, b in zip(orig_slice, corr_slice) if a != b)
                            logger.info(
                                f"[Decoder] RS corrected {n_errors} bit errors"
                            )
                            self._fix_patches_from_correction(
                                reference_tensor, building_tensor,
                                patch_records, 0,  # header_len_bits = 0 (no vote header)
                                orig_slice, corr_slice,
                                selected_indices_list,
                                seen_patch_fingerprints,
                            )
                        else:
                            logger.info("[Decoder] RS verified — no corrections needed")

                    except Exception as ex:
                        logger.info(
                            f"[Decoder] RS decode failed ({type(ex).__name__}) — "
                            f"need more bits (have {len(body_bits)})"
                        )

        # Return raw bits (with XOR) — pipeline will undo XOR later
        return decoded_bits, decoded_bits, selected_indices_list, stats

    # ------------------------------------------------------------------
    #  Helper: fix patches after RS correction
    # ------------------------------------------------------------------

    def _fix_patches_from_correction(
        self,
        reference_tensor: torch.Tensor,
        building_tensor: torch.Tensor,
        patch_records: List[PatchRecord],
        header_len_bits: int,
        original_bits: str,
        corrected_bits: str,
        selected_indices_list: List[int],
        seen_patch_fingerprints: Optional[Set[FrozenSet[int]]] = None,
    ) -> None:
        """
        Compare original (pre-RS) body bits with corrected body bits.
        For each bit that changed, find the corresponding patch(es)
        and re-select the correct codebook index.
        """
        min_len = min(len(original_bits), len(corrected_bits))
        if min_len == 0:
            return

        diff_positions = [
            i for i in range(min_len)
            if original_bits[i] != corrected_bits[i]
        ]
        if not diff_positions:
            return

        logger.info(
            f"[Decoder] RS correction: {len(diff_positions)} bit differences "
            f"over {min_len} bits — fixing patches"
        )

        # Build cumulative bit offset map: patch_index → (start_bit, end_bit) in body
        patch_map = []
        bit_offset = 0
        for pi, pr in enumerate(patch_records):
            pb = pr.token_bits
            # Undo XOR on these bits too for proper alignment
            aligned_len = len(pb)  # same length, XOR doesn't change length
            patch_map.append((pi, bit_offset, bit_offset + aligned_len))
            bit_offset += aligned_len

        # Find which patches overlap with corrected regions
        patches_to_fix = set()
        for pi, start, end in patch_map:
            for dp in diff_positions:
                if start <= dp < end:
                    patches_to_fix.add(pi)
                    break

        if not patches_to_fix:
            return

        # Task 4: Anti-loop fingerprint — if we've already corrected these
        # same patches before without any change, lock further corrections.
        if seen_patch_fingerprints is not None:
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
            pr = patch_records[pi]
            _, body_start, _ = patch_map[pi]
            corrected_patch_bits = corrected_bits[
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