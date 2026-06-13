#!/usr/bin/env python3
"""
Diagnostic script: comprehensive index/bits comparison between encode and decode paths.

Generates one steganographic image and produces a detailed structured report showing
EVERY intermediate step, making it easy to pinpoint where encode and decode diverge.

Outputs:
    1. JSON file  - structured data for programmatic analysis
    2. .txt file  - human-readable diff matrices and summaries

Usage:
    python scripts/diagnose_indices.py "Your message" -o diagnostics_output/
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure we can import from project root
system_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(system_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.logger import logger
from scripts.decoder import SteganographyDecoder, load_image, PatchRecord, bits_to_bytes
from scripts.encoder import SteganographyEncoder
from scripts.input import Options
from scripts.utils import (
    get_vqgan_sflckr, reset_seeds, save_image, build_context_from_patches,
    DEFAULT_PRECISION_BITS, DEFAULT_CODEBOOK_SIZE, PATCH_SIZE,
    int2bits, bits2int, count_matching_bits_from_start, set_context,
)
from scripts.stats import EncodingStatistics, DecodingStatistics
from scripts.pipeline import (
    build_pipeline_from_options, SteganoPipeline, XorMask,
    RS_BLOCK_BYTES, RS_NSYM, RS_PAYLOAD_BITS, RS_ENCODED_BITS, RS_BLOCK_BYTES,
    HEADER_LENGTH_BITS, decode_block_fixed_size, _decode_length_from_body,
)
from scripts.error_correction import ReedSolomonCorrectionCode


# ======================================================================
#  Helpers
# ======================================================================

def bits_diff_table(bits_a: str, bits_b: str, label_a: str = "A", label_b: str = "B",
                    max_show: int = 200) -> Tuple[str, int, List[int]]:
    """Compare two bit strings and return (table_str, mismatch_count, mismatch_indices)."""
    mismatches = []
    for i, (ba, bb) in enumerate(zip(bits_a, bits_b)):
        if ba != bb:
            mismatches.append(i)
    n = len(mismatches)
    lines = [
        f"  Total bits compared: {len(bits_a)} vs {len(bits_b)}",
        f"  Mismatches: {n} / {min(len(bits_a), len(bits_b))}",
    ]
    if n > 0:
        show = mismatches[:max_show]
        lines.append(f"  First {len(show)} mismatch positions: {show}")
        lines.append(f"  Mismatch detail (first {min(20, len(show))}):")
        for pos in show[:20]:
            a_bit = bits_a[pos] if pos < len(bits_a) else '?'
            b_bit = bits_b[pos] if pos < len(bits_b) else '?'
            lines.append(f"    bit[{pos:4d}]: {label_a}={a_bit}  {label_b}={b_bit}")
    return "\n".join(lines), n, mismatches


def compute_cumulative_probs(
    encoder: Any,
    context: torch.Tensor,
    local_row: int,
    local_col: int,
    codebook_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Given a context tensor, return the sorted logits, sorted indices,
    cumulative_probs, and k (number of selected candidates).
    """
    top_k = codebook_len
    encoder.model.eval()
    with torch.no_grad():
        logits, _ = encoder.model.transformer(context[:-1].unsqueeze(0))
        logits = logits[:, -256:, :].squeeze()
        logits = logits.reshape(PATCH_SIZE, PATCH_SIZE, -1)
        logits = logits[local_row, local_col, :].double()
        logits_sorted, indices_sorted = logits.sort(descending=True)
        probs = torch.nn.functional.softmax(logits_sorted, dim=-1)

    prob_threshold = 1 / codebook_len
    first_below = torch.nonzero(probs < prob_threshold)
    if len(first_below) > 0:
        k = min(max(2, first_below[0].item()), top_k)
    else:
        k = top_k

    probs_int = probs[:k]
    probs_int = (probs_int / probs_int.sum() * codebook_len).round().long()
    cumulative_probs = probs_int.cumsum(0)

    overfill = torch.nonzero(cumulative_probs > codebook_len)
    if len(overfill) > 0:
        cumulative_probs = cumulative_probs[:overfill[0]]
    cumulative_probs += codebook_len - cumulative_probs[-1]

    return logits_sorted, indices_sorted, cumulative_probs, k


# ======================================================================
#  Main diagnostic
# ======================================================================

def run_diagnostics(
    message: str,
    model_directory_path: str,
    output_directory: str,
    seed: int = 42,
    xor_key: int = 123,
    char_encoding: str = "ASCII",
    context_fraction: float = 0.1,
) -> Dict[str, Any]:
    """Run the full encode → decode pipeline with per-step capture."""

    logger.info("=" * 70)
    logger.info("DIAGNOSTIC: Full encode/decode comparison")
    logger.info(f"  message: {message!r}")
    logger.info(f"  seed: {seed}")
    logger.info(f"  xor_key: {xor_key}")
    logger.info(f"  encoding: {char_encoding}")
    logger.info("=" * 70)

    # ---- 1. Setup model and pipeline ----
    dsets, model = get_vqgan_sflckr(model_directory_path)
    pipeline = SteganoPipeline(xor_key=xor_key, char_encoding=char_encoding)
    encoder = SteganographyEncoder(model, dsets, context_fraction=context_fraction)
    decoder = SteganographyDecoder(model, dsets, context_fraction=context_fraction)

    # ---- 2. Pipeline encode (WITH trace) ----
    protected_bits, encode_trace = pipeline.encode_message_with_trace(message)
    logger.info(f"Pipeline encode: {len(protected_bits)} protected bits")

    # ---- 3. VQGAN encode setup ----
    reset_seeds(seed)
    img, cond = set_context(model, dsets, 0)  # DEFAULT_CONTEXT_ROWS = 0
    image_translations, image_indices = model.encode_to_z(img.unsqueeze(0))
    cond_translations, cond_indices = model.encode_to_c(cond)

    grid_shape = (image_translations.shape[2], image_translations.shape[3])
    reference_tensor = cond_indices.reshape(
        cond_translations.shape[0], cond_translations.shape[2], cond_translations.shape[3]
    ).squeeze()

    half_start = int(image_indices.shape[1] * context_fraction)
    building_tensor = image_indices.clone()
    building_tensor[:, half_start:] = 0
    building_tensor = building_tensor.reshape(grid_shape)

    current_row = half_start // grid_shape[1]
    current_col = half_start % grid_shape[1]

    codebook_len = DEFAULT_CODEBOOK_SIZE

    # ---- 4. Encode patches with diagnostic capture ----
    remaining_bits = protected_bits
    encode_patch_records: List[Dict[str, Any]] = []
    indices_sequence: List[int] = []
    patch_index = 0
    random_fill_count = 0

    logger.info(f"Encoding starts at row={current_row}, col={current_col}, grid={grid_shape}")
    logger.info(f"Total protected bits to encode: {len(remaining_bits)}")

    while current_row < grid_shape[0]:
        patch_row, patch_col = current_row, current_col
        random_sample = not bool(remaining_bits)

        # Build context
        context, (local_row, local_col) = build_context_from_patches(
            reference_tensor, building_tensor, current_row, current_col, grid_shape
        )
        logits_sorted, indices_sorted, cumulative_probs, k = compute_cumulative_probs(
            encoder, context, local_row, local_col, codebook_len
        )

        if random_sample:
            # Random fill: pick random token from top-k
            probs = torch.nn.functional.softmax(logits_sorted[:k], dim=-1)
            selected = torch.multinomial(probs, 1).item()
            selected_idx = indices_sorted[selected].item()
            building_tensor[current_row, current_col] = selected_idx
            used_bits = 0
            encoded_bits_str = ""
            bits_fed = "RANDOM_FILL"

            rec = {
                "patch_index": patch_index,
                "row": patch_row,
                "col": patch_col,
                "local_row": local_row,
                "local_col": local_col,
                "selected_idx": selected_idx,
                "bits_fed": bits_fed,
                "used_bits": used_bits,
                "encoded_bits_str": encoded_bits_str,
                "range_bottom": 0,
                "range_top": codebook_len,
                "top_k_indices": indices_sorted[:k].tolist(),
                "top_k_logits": logits_sorted[:k].tolist(),
                "top_k_probs": None,
                "context": context.tolist(),
                "building_tensor_state": building_tensor[patch_row, patch_col].item(),
                "is_random_fill": True,
            }
            encode_patch_records.append(rec)
            indices_sequence.append(selected_idx)
            random_fill_count += 1

        else:
            # Arithmetic encode
            feed_bits = remaining_bits[:DEFAULT_PRECISION_BITS]
            message_str = (feed_bits + "0" * DEFAULT_PRECISION_BITS)[:DEFAULT_PRECISION_BITS]
            message_value = bits2int(message_str)

            selection = torch.nonzero(cumulative_probs > message_value)[0].item()
            range_bottom = cumulative_probs[selection - 1].item() if selection > 0 else 0
            range_top = cumulative_probs[selection].item()

            bottom_bits = int2bits(range_bottom, DEFAULT_PRECISION_BITS)
            top_bits = int2bits(range_top - 1, DEFAULT_PRECISION_BITS)

            encoded_bits = count_matching_bits_from_start(bottom_bits, top_bits)
            encoded_bits_str = bottom_bits[:encoded_bits]
            selected_idx = indices_sorted[selection].item()
            building_tensor[current_row, current_col] = selected_idx

            rec = {
                "patch_index": patch_index,
                "row": patch_row,
                "col": patch_col,
                "local_row": local_row,
                "local_col": local_col,
                "selected_idx": selected_idx,
                "bits_fed": feed_bits,
                "used_bits": encoded_bits,
                "encoded_bits_str": encoded_bits_str,
                "range_bottom": range_bottom,
                "range_top": range_top,
                "top_k_indices": indices_sorted[:k].tolist(),
                "top_k_logits": logits_sorted[:k].tolist(),
                "top_k_probs": None,
                "context": context.tolist(),
                "building_tensor_state": building_tensor[patch_row, patch_col].item(),
                "is_random_fill": False,
            }
            encode_patch_records.append(rec)
            indices_sequence.append(selected_idx)
            remaining_bits = remaining_bits[encoded_bits:]

        # Advance position
        current_col += 1
        if current_col >= grid_shape[1]:
            current_col = 0
            current_row += 1
        patch_index += 1

    logger.info(f"Encoding complete: {len(encode_patch_records)} patches "
                f"({random_fill_count} random fill)")

    # ---- 5. Generate output image and reload to get stego indices ----
    output_image = model.decode_to_img(
        building_tensor.unsqueeze(0), image_translations.shape
    ).squeeze()

    os.makedirs(output_directory, exist_ok=True)
    image_path = os.path.join(output_directory, "diagnostic_image.png")
    save_image(output_image, image_path.replace(".png", ""))

    loaded_image = load_image(image_path)
    height_crop = grid_shape[0] * PATCH_SIZE
    width_crop = grid_shape[1] * PATCH_SIZE
    cropped = loaded_image[:, :height_crop, :width_crop]
    stego_indices = model.encode_to_z(cropped.unsqueeze(0))[1].reshape(grid_shape)

    # ---- 6. Compare building_tensor with stego_indices ----
    building_vs_stego = (building_tensor == stego_indices).all().item()
    logger.info(f"building_tensor == stego_indices after encode+decode: {building_vs_stego}")

    index_diff_positions: List[Dict[str, int]] = []
    if not building_vs_stego:
        diff_pos = torch.nonzero(building_tensor != stego_indices)
        logger.warning(
            f"Differences between building_tensor and stego_indices: "
            f"{len(diff_pos)} positions"
        )
        for dp in diff_pos[:20]:
            r, c = dp[0].item(), dp[1].item()
            index_diff_positions.append({
                "row": r, "col": c,
                "building": building_tensor[r, c].item(),
                "stego": stego_indices[r, c].item(),
            })

    # ---- 7. Decode with capture ----
    reset_seeds(seed)
    img2, cond2 = set_context(model, dsets, 0)
    _, img_indices2 = model.encode_to_z(img2.unsqueeze(0))
    cond_translations2, cond_indices2 = model.encode_to_c(cond2)
    ref_tensor2 = cond_indices2.reshape(
        cond_translations2.shape[0], cond_translations2.shape[2], cond_translations2.shape[3]
    ).squeeze()

    building_tensor_dec = img_indices2.clone().reshape(grid_shape)
    ecc = ReedSolomonCorrectionCode(nsym=RS_NSYM)

    current_row = 0
    current_col = 0
    decode_patch_records: List[Dict[str, Any]] = []
    decoded_bits_raw_list: List[str] = []  # bits with XOR still applied
    accumulated_xor_undone: List[str] = []

    while current_row < grid_shape[0]:
        patch_row, patch_col = current_row, current_col

        context2, (local_row2, local_col2) = build_context_from_patches(
            ref_tensor2, building_tensor_dec, current_row, current_col, grid_shape
        )
        actual_token = stego_indices[current_row, current_col].item()

        logits_sorted2, indices_sorted2, cum_probs2, k2 = compute_cumulative_probs(
            decoder, context2, local_row2, local_col2, codebook_len
        )

        # Find matching index
        matching = torch.nonzero(indices_sorted2[:k2] == actual_token).squeeze()
        if matching.numel() == 0:
            match_pos = -1  # not found
        elif matching.dim() == 0:
            match_pos = matching.item()
        else:
            match_pos = -1

        if match_pos >= 0:
            # Arithmetic decode
            rb = cum_probs2[match_pos - 1].item() if match_pos > 0 else 0
            rt = cum_probs2[match_pos].item()
            bb = int2bits(rb, DEFAULT_PRECISION_BITS)
            bt = int2bits(rt - 1, DEFAULT_PRECISION_BITS)
            token_bits = ""
            for b, t in zip(bb, bt):
                if b == t:
                    token_bits += str(b)
                else:
                    break
        else:
            token_bits = ""
            rb = 0
            rt = 0

        building_tensor_dec[current_row, current_col] = actual_token

        decoded_bits_raw_list.append(token_bits)
        bits_plain = pipeline.xor_mask.apply(token_bits)
        accumulated_xor_undone.append(bits_plain)

        rec = {
            "patch_index": len(decode_patch_records),
            "row": patch_row,
            "col": patch_col,
            "local_row": local_row2,
            "local_col": local_col2,
            "stego_token": actual_token,
            "decoded_bits_str": token_bits,
            "bits_plain": bits_plain,
            "range_bottom": rb,
            "range_top": rt,
            "match_in_topk": match_pos >= 0,
            "match_position_in_sorted": match_pos,
            "k_used": k2,
            "top_k_indices": indices_sorted2[:k2].tolist(),
            "top_k_logits": logits_sorted2[:k2].tolist(),
            "cumulative_probs": cum_probs2.tolist(),
            "context": context2.tolist(),
        }
        decode_patch_records.append(rec)

        current_col += 1
        if current_col >= grid_shape[1]:
            current_col = 0
            current_row += 1

    # ---- 8. Full RS decode from accumulated XOR-undoed bits ----
    decoded_bits_raw = "".join(decoded_bits_raw_list)
    all_xor_undone = "".join(accumulated_xor_undone)

    rs_blocks_info: List[Dict[str, Any]] = []
    decoded_parts: List[str] = []
    offset = 0

    while offset < len(all_xor_undone):
        chunk = all_xor_undone[offset:offset + RS_ENCODED_BITS]
        if len(chunk) < 8:
            break
        if len(chunk) == RS_ENCODED_BITS:
            decoded_block = decode_block_fixed_size(chunk, ecc, RS_PAYLOAD_BITS)
        else:
            decoded_block = ecc.decode("000" + chunk)
            if decoded_block:
                decoded_block = decoded_block[:RS_PAYLOAD_BITS]

        if decoded_block:
            rs_blocks_info.append({
                "block_idx": len(rs_blocks_info),
                "input_bits": chunk,
                "input_len": len(chunk),
                "decoded_len": len(decoded_block),
                "decoded_bits": decoded_block,
            })
            decoded_parts.append(decoded_block)
            offset += len(chunk)
        else:
            break

    body_decoded = "".join(decoded_parts)

    message_bit_length = _decode_length_from_body(body_decoded)
    if message_bit_length > 0:
        recovered_bits = body_decoded[
            HEADER_LENGTH_BITS:HEADER_LENGTH_BITS + message_bit_length
        ]
    else:
        recovered_bits = ""

    from scripts.bit_utils import bits2string
    recovered_text = bits2string(recovered_bits, code=char_encoding)

    # ---- 9. Comparison ----
    pip_protected = encode_trace.get("protected_bits", "")
    bit_diff_table_str, bit_mismatch_count, bit_mismatch_pos = bits_diff_table(
        pip_protected, decoded_bits_raw,
        "encode_protected", "decode_raw"
    )

    # Per-patch index comparison
    per_patch_diffs: List[Dict[str, Any]] = []
    max_patches = max(len(encode_patch_records), len(decode_patch_records))
    for pi in range(max_patches):
        enc = encode_patch_records[pi] if pi < len(encode_patch_records) else None
        dec = decode_patch_records[pi] if pi < len(decode_patch_records) else None

        enc_idx = enc["selected_idx"] if enc else None
        dec_idx = dec["stego_token"] if dec else None

        entry: Dict[str, Any] = {"patch_index": pi}
        if enc:
            entry["encode"] = {
                "row": enc["row"], "col": enc["col"],
                "selected_index": enc_idx,
                "bits_fed": enc["bits_fed"],
                "used_bits": enc["used_bits"],
                "encoded_bits": enc["encoded_bits_str"],
                "range": [enc["range_bottom"], enc["range_top"]],
                "is_random_fill": enc.get("is_random_fill", False),
            }
        if dec:
            entry["decode"] = {
                "row": dec["row"], "col": dec["col"],
                "stego_token": dec_idx,
                "decoded_bits": dec["decoded_bits_str"],
                "range": [dec["range_bottom"], dec["range_top"]],
                "match_in_topk": dec["match_in_topk"],
                "match_position_in_sorted": dec["match_position_in_sorted"],
            }
        if enc_idx is not None and dec_idx is not None:
            entry["index_match"] = enc_idx == dec_idx
        else:
            entry["index_match"] = None

        per_patch_diffs.append(entry)

    index_mismatches = sum(1 for p in per_patch_diffs if p.get("index_match") is False)
    index_total = sum(1 for p in per_patch_diffs if p.get("index_match") is not None)

    # ---- 10. Assemble final report ----
    report: Dict[str, Any] = {
        "summary": {
            "message": message,
            "message_bits": encode_trace.get("raw_bits", ""),
            "message_bits_len": len(encode_trace.get("raw_bits", "")),
            "seed": seed,
            "xor_key": xor_key,
            "char_encoding": char_encoding,
            "grid_shape": list(grid_shape),
            "rs_nsym": RS_NSYM,
            "rs_block_bytes": RS_BLOCK_BYTES,
        },
        "pipeline_trace": {
            "encode": encode_trace,
            "decode": {
                "xor_output_bits": all_xor_undone,
                "rs_blocks": rs_blocks_info,
                "body_decoded": body_decoded,
                "header_raw": format(message_bit_length, f"0{HEADER_LENGTH_BITS}b"),
                "recovered_bits": recovered_bits,
                "recovered_text": recovered_text,
            },
        },
        "vqgan_encode": {
            "total_patches_encoded": len(encode_patch_records),
            "patches_used_for_message": len(encode_patch_records) - random_fill_count,
            "patches_fill_random": random_fill_count,
            "encode_start_row": half_start // grid_shape[1],
            "encode_start_col": half_start % grid_shape[1],
            "patches": encode_patch_records,
        },
        "vqgan_stego_indices": {
            "shape": list(stego_indices.shape),
            "matrix": stego_indices.tolist(),
        },
        "vqgan_stego_index_diffs": {
            "building_matches_stego": building_vs_stego,
            "diff_positions": index_diff_positions,
        },
        "vqgan_decode": {
            "total_patches_decoded": len(decode_patch_records),
            "patches": decode_patch_records,
        },
        "comparisons": {
            "index_comparison": {
                "total_compared": index_total,
                "mismatches": index_mismatches,
                "match_ratio": (1.0 - index_mismatches / index_total)
                               if index_total > 0 else 0.0,
                "per_patch": per_patch_diffs,
            },
            "bit_comparison": {
                "encode_bits_length": len(pip_protected),
                "decode_bits_length": len(decoded_bits_raw),
                "total_compared": min(len(pip_protected), len(decoded_bits_raw)),
                "mismatches": bit_mismatch_count,
                "mismatch_positions": bit_mismatch_pos[:100],
                "detail": bit_diff_table_str,
            },
            "message_recovery": {
                "recovered_text": recovered_text,
                "original_message": message,
                "match": message == recovered_text,
            },
        },
    }

    return report


def format_report_as_text(report: Dict[str, Any]) -> str:
    """Format the diagnostic report as a readable text file."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("  STEGANOGRAPHY ENCODE/DECODE DIAGNOSTIC REPORT")
    lines.append("=" * 80)

    s = report["summary"]
    lines.append(f"\nMessage         : {s['message']!r}")
    lines.append(f"Message bits    : {s['message_bits']}")
    lines.append(f"Message bit len : {s['message_bits_len']}")
    lines.append(f"Seed            : {s['seed']}")
    lines.append(f"XOR key         : {s['xor_key']}")
    lines.append(f"Char encoding   : {s['char_encoding']}")
    lines.append(f"Grid shape      : {s['grid_shape']}")
    lines.append(f"RS nsym         : {s['rs_nsym']}")
    lines.append(f"RS block bytes  : {s['rs_block_bytes']}")

    # Pipeline trace
    lines.append("\n" + "-" * 80)
    lines.append("  PIPELINE TRACE")
    lines.append("-" * 80)
    pt = report["pipeline_trace"]
    lines.append(f"\nEncode:")
    for k, v in pt["encode"].items():
        vstr = str(v)[:80] if not isinstance(v, str) else v[:80] if len(v) > 80 else v
        lines.append(f"  {k}: {vstr}")
    lines.append(f"\nDecode:")
    for k, v in pt["decode"].items():
        if k == "rs_blocks":
            continue
        vstr = str(v)[:80] if not isinstance(v, str) else v[:80] if len(v) > 80 else v
        lines.append(f"  {k}: {vstr}")

    # RS blocks
    rbs = pt["decode"].get("rs_blocks", [])
    lines.append(f"\nRS Blocks ({len(rbs)}):")
    for rb in rbs:
        lines.append(f"  Block {rb['block_idx']}: input={rb['input_len']}b -> decoded={rb['decoded_len']}b")
        lines.append(f"    input bits (first 32): {rb['input_bits'][:32]}...")

    # VQGAN Encode patches
    lines.append("\n" + "-" * 80)
    lines.append("  VQGAN ENCODE PATCHES")
    lines.append("-" * 80)
    ve = report["vqgan_encode"]
    lines.append(f"  Patches for message  : {ve['patches_used_for_message']}")
    lines.append(f"  Fill (random) patches: {ve['patches_fill_random']}")
    lines.append(f"  Encode start position: row={ve['encode_start_row']}, col={ve['encode_start_col']}")
    lines.append(f"\n  Per-patch encode details:")
    lines.append(f"  {'Idx':>4} {'Row':>4} {'Col':>4} {'SelIdx':>6} {'UsedBits':>8} {'RangeBot':>8} {'RangeTop':>8} {'Fill?':>6}")
    lines.append(f"  {'-'*65}")
    for p in ve['patches']:
        fill = "YES" if p.get("is_random_fill", False) else "NO"
        lines.append(f"  {p['patch_index']:>4} {p['row']:>4} {p['col']:>4} {p['selected_idx']:>6} "
                     f"{p['used_bits']:>8} {p['range_bottom']:>8} {p['range_top']:>8} {fill:>6}")

    # VQGAN Decode patches
    lines.append("\n" + "-" * 80)
    lines.append("  VQGAN DECODE PATCHES")
    lines.append("-" * 80)
    vd = report["vqgan_decode"]
    lines.append(f"  Total patches decoded: {vd['total_patches_decoded']}")
    lines.append(f"\n  Per-patch decode details:")
    lines.append(f"  {'Idx':>4} {'Row':>4} {'Col':>4} {'StegoTkn':>8} {'DecBits':>10} {'RangeBtm':>8} {'RangeTop':>8} {'Match?':>6}")
    lines.append(f"  {'-'*65}")
    for p in vd['patches']:
        match_str = "OK" if p.get('match_in_topk') else "FAIL"
        lines.append(f"  {p['patch_index']:>4} {p['row']:>4} {p['col']:>4} {p['stego_token']:>8} "
                     f"{p['decoded_bits_str']:>10} {p['range_bottom']:>8} {p['range_top']:>8} {match_str:>6}")

    # Index comparison
    lines.append("\n" + "=" * 80)
    lines.append("  INDEX COMPARISON (encode selected vs decode recovered)")
    lines.append("=" * 80)
    ic = report["comparisons"]["index_comparison"]
    lines.append(f"  Total compared: {ic['total_compared']}")
    lines.append(f"  Mismatches    : {ic['mismatches']}")
    lines.append(f"  Match ratio   : {ic['match_ratio']:.4f}")

    mismatches_only = [p for p in ic["per_patch"] if p.get("index_match") is False]
    if mismatches_only:
        lines.append(f"\n  *** INDEX MISMATCHES ({len(mismatches_only)}): ***")
        lines.append(f"  {'Idx':>4} {'Row':>4} {'Col':>4} {'EncIdx':>6} {'DecIdx':>6} {'Fill?':>6}")
        lines.append(f"  {'-'*50}")
        for p in mismatches_only:
            enc = p.get("encode", {})
            dec = p.get("decode", {})
            fill_str = "YES" if enc.get("is_random_fill", False) else "NO"
            lines.append(f"  {p['patch_index']:>4} {enc.get('row','?'):>4} {enc.get('col','?'):>4} "
                         f"{enc.get('selected_index','?'):>6} {dec.get('stego_token','?'):>6} {fill_str:>6}")
    else:
        lines.append(f"\n  No index mismatches found!")

    # Stego index differences
    sid = report.get("vqgan_stego_index_diffs", {})
    if sid.get("diff_positions"):
        lines.append(f"\n  *** STEGO INDEX DIFFERENCES (building vs stego after round-trip): ***")
        lines.append(f"  {'Row':>4} {'Col':>4} {'Building':>10} {'Stego':>8}")
        lines.append(f"  {'-'*40}")
        for d in sid["diff_positions"]:
            lines.append(f"  {d['row']:>4} {d['col']:>4} {d['building']:>10} {d['stego']:>8}")
    else:
        lines.append(f"\n  No differences between building_tensor and stego_indices.")

    # Bit comparison
    lines.append("\n" + "=" * 80)
    lines.append("  BIT COMPARISON (encode protected bits vs decode raw bits)")
    lines.append("=" * 80)
    lines.append(f"\n{report['comparisons']['bit_comparison']['detail']}")

    # Message recovery
    lines.append("\n" + "-" * 80)
    lines.append("  MESSAGE RECOVERY")
    lines.append("-" * 80)
    mr = report["comparisons"]["message_recovery"]
    lines.append(f"  Original message : {mr['original_message']!r}")
    lines.append(f"  Recovered text   : {mr['recovered_text']!r}")
    lines.append(f"  Match            : {mr['match']}")

    lines.append("\n" + "=" * 80)
    lines.append("  END OF DIAGNOSTIC REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose encode/decode index discrepancies in the steganography pipeline."
    )
    parser.add_argument("message", nargs="?", default="Hello",
                        help="Message to encode (default: 'Hello')")
    parser.add_argument("-m", "--model_directory",
                        type=str, default="logs/2020-11-09T13-31-51_sflckr",
                        help="Path to VQGAN model directory")
    parser.add_argument("-o", "--output_directory",
                        type=str, default="diagnostics_output",
                        help="Output directory for diagnostic files")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--xor-key", type=int, default=123,
                        help="XOR key (0-255)")
    parser.add_argument("--char-encoding", type=str, default="ASCII",
                        choices=["ASCII", "UNICODE", "DECIMAL"],
                        help="Character encoding")
    parser.add_argument("-c", "--context_fraction",
                        type=float, default=0.1,
                        help="Context fraction for VQGAN")
    args = parser.parse_args()

    # Clean output directory
    if os.path.exists(args.output_directory):
        shutil.rmtree(args.output_directory)
    os.makedirs(args.output_directory)

    # Run diagnostics with reduced logging
    logger.enable_set(False)

    report = run_diagnostics(
        message=args.message,
        model_directory_path=args.model_directory,
        output_directory=args.output_directory,
        seed=args.seed,
        xor_key=args.xor_key,
        char_encoding=args.char_encoding,
        context_fraction=args.context_fraction,
    )

    # Write JSON
    json_path = os.path.join(args.output_directory, "diagnostics_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report written to: {json_path}")

    # Write text report
    txt_path = os.path.join(args.output_directory, "diagnostics_report.txt")
    text = format_report_as_text(report)
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"Text report written to: {txt_path}")

    # Print quick summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    ic = report["comparisons"]["index_comparison"]
    bc = report["comparisons"]["bit_comparison"]
    mr = report["comparisons"]["message_recovery"]
    print(f"  Index mismatches : {ic['mismatches']} / {ic['total_compared']}")
    print(f"  Bit mismatches   : {bc['mismatches']} / {bc['total_compared']}")
    print(f"  Message match    : {mr['match']}")
    print(f"  Recovered        : {mr['recovered_text']!r}")
    print(f"  Original         : {mr['original_message']!r}")
    print()


if __name__ == "__main__":
    main()