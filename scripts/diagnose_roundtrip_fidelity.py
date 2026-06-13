#!/usr/bin/env python3
"""
Diagnostic script: check VQGAN save/load round-trip fidelity.

Checks two things:
  1. INDEX FIDELITY: after encoding an image → saving to PNG → loading
     it back → re-encoding, are the VQGAN indices *identical* to the
     original indices?  If not, the model's encode_to_z is not
     deterministic for saved images.
  2. PIXEL FIDELITY: how close are the pixel values of the saved-then-
     reloaded image to the original decoded image?

If indices change after a simple save/load round-trip (no steganographic
manipulation), then the problem is systemic: PNG loss compresses the
latent-reconstructed pixels enough to push some patches into neighbouring
codebook entries.  If indices are identical, the problem lies elsewhere
(e.g. in the steganographic encoding/decoding logic itself).

Usage:
    python scripts/diagnose_roundtrip_fidelity.py -m <model_dir> -o <out_dir>
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

system_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(system_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.logger import logger
from scripts.decoder import load_image
from scripts.utils import (
    get_vqgan_sflckr, reset_seeds, save_image, set_context,
    PATCH_SIZE,
)


# ======================================================================
#  Main diagnostic
# ======================================================================

def run_roundtrip_diagnostics(
    model_directory_path: str,
    output_directory: str,
    seed: int = 42,
    context_rows: int = 0,
) -> Dict[str, Any]:
    """
    1. Generate a VQGAN image (with context, no message).
    2. Save to PNG.
    3. Reload PNG.
    4. Re-encode the reloaded image → get stego indices.
    5. Compare original indices vs stego indices.
    6. Compare pixel values before save vs after load.
    """

    logger.info("=" * 70)
    logger.info("ROUND-TRIP FIDELITY DIAGNOSTIC")
    logger.info(f"  seed         = {seed}")
    logger.info(f"  model_dir    = {model_directory_path}")
    logger.info(f"  context_rows = {context_rows}")
    logger.info("=" * 70)

    os.makedirs(output_directory, exist_ok=True)

    # ---- 1. Load model and generate base image ----
    dsets, model = get_vqgan_sflckr(model_directory_path)
    reset_seeds(seed)

    img, cond = set_context(model, dsets, context_rows)

    # ---- 2. Encode to get original indices ----
    image_translations, image_indices = model.encode_to_z(img.unsqueeze(0))
    cond_translations, cond_indices = model.encode_to_c(cond)

    grid_shape = (image_translations.shape[2], image_translations.shape[3])
    original_indices = image_indices.clone().reshape(grid_shape)

    # ---- 3. Decode to pixels ----
    original_pixels = model.decode_to_img(
        image_indices, image_translations.shape
    ).squeeze()  # shape: (C, H, W)

    # ---- 4. Save to PNG ----
    original_pixels_path = os.path.join(output_directory, "original.png")
    save_image(original_pixels, original_pixels_path.replace(".png", ""))

    # ---- 5. Load back from PNG ----
    loaded = load_image(original_pixels_path)
    height_crop = grid_shape[0] * PATCH_SIZE
    width_crop = grid_shape[1] * PATCH_SIZE
    cropped = loaded[:, :height_crop, :width_crop]

    # ---- 6. Re-encode the loaded image to get stego indices ----
    stego_indices = model.encode_to_z(cropped.unsqueeze(0))[1].reshape(grid_shape)

    # ---- 7. Index comparison ----
    indices_identical = torch.equal(original_indices, stego_indices)

    index_diff_positions: List[Dict[str, Any]] = []
    if not indices_identical:
        diff_mask = original_indices != stego_indices
        diff_pos = torch.nonzero(diff_mask)
        n_diff = diff_pos.shape[0]
        logger.warning(f"Index differences: {n_diff} / {original_indices.numel()} positions")
        for dp in diff_pos[:50]:
            r, c = dp[0].item(), dp[1].item()
            index_diff_positions.append({
                "row": r, "col": c,
                "original": original_indices[r, c].item(),
                "stego": stego_indices[r, c].item(),
            })
    else:
        n_diff = 0
        logger.info("Indices are IDENTICAL after round-trip. ✓")

    # ---- 8. Pixel comparison ----
    # original_pixels is the decoded image BEFORE save (range [-1, 1])
    # cropped is the loaded image (range [-1, 1])
    # They may differ in dimensions (the cropped may have extra context rows)
    min_h = min(original_pixels.shape[1], cropped.shape[1])
    min_w = min(original_pixels.shape[2], cropped.shape[2])

    pixel_diff = original_pixels[:, :min_h, :min_w] - cropped[:, :min_h, :min_w]
    pixel_abs_diff = pixel_diff.abs()

    pixel_stats = {
        "mean_abs_diff": pixel_abs_diff.mean().item(),
        "max_abs_diff": pixel_abs_diff.max().item(),
        "median_abs_diff": pixel_abs_diff.median().item(),
        "std_abs_diff": pixel_abs_diff.std().item(),
        "min_pixel_before": original_pixels[:, :min_h, :min_w].min().item(),
        "max_pixel_before": original_pixels[:, :min_h, :min_w].max().item(),
        "min_pixel_after": cropped[:, :min_h, :min_w].min().item(),
        "max_pixel_after": cropped[:, :min_h, :min_w].max().item(),
    }
    logger.info(f"Pixel abs diff: mean={pixel_stats['mean_abs_diff']:.6f}, "
                f"max={pixel_stats['max_abs_diff']:.6f}")

    # ---- 9. Per-channel pixel stats ----
    ch_stats = {}
    for ch_idx, ch_name in enumerate(["R", "G", "B"]):
        ch_diff = pixel_diff[ch_idx]
        ch_stats[ch_name] = {
            "mean_abs_diff": ch_diff.abs().mean().item(),
            "max_abs_diff": ch_diff.abs().max().item(),
        }

    # ---- 10. Quantify how many indices changed per row (hotspot analysis) ----
    row_diff_counts: List[int] = []
    if not indices_identical:
        for r in range(grid_shape[0]):
            row_diffs = (original_indices[r, :] != stego_indices[r, :]).sum().item()
            row_diff_counts.append(row_diffs)
    else:
        row_diff_counts = [0] * grid_shape[0]

    # ---- 11. Assemble report ----
    report: Dict[str, Any] = {
        "summary": {
            "seed": seed,
            "context_rows": context_rows,
            "grid_shape": list(grid_shape),
            "total_patches": original_indices.numel(),
        },
        "index_comparison": {
            "identical": indices_identical,
            "total_mismatches": n_diff,
            "mismatch_ratio": round(n_diff / original_indices.numel(), 6) if original_indices.numel() > 0 else 0.0,
            "sample_diffs": index_diff_positions[:50],
            "row_diff_counts": row_diff_counts,
        },
        "pixel_comparison": pixel_stats,
        "pixel_per_channel": ch_stats,
    }

    return report


def format_report_as_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("  VQGAN ROUND-TRIP FIDELITY REPORT")
    lines.append("=" * 80)

    s = report["summary"]
    lines.append(f"\nSeed            : {s['seed']}")
    lines.append(f"Context rows    : {s['context_rows']}")
    lines.append(f"Grid shape      : {s['grid_shape']}")
    lines.append(f"Total patches   : {s['total_patches']}")

    # Index comparison
    lines.append("\n" + "-" * 80)
    lines.append("  INDEX FIDELITY")
    lines.append("-" * 80)
    ic = report["index_comparison"]
    if ic["identical"]:
        lines.append("\n  ✓ Indices are IDENTICAL after save/load round-trip.\n")
    else:
        lines.append(f"\n  ✗ Indices CHANGED after round-trip!")
        lines.append(f"    Total mismatches : {ic['total_mismatches']} / {s['total_patches']}")
        lines.append(f"    Mismatch ratio   : {ic['mismatch_ratio']}")
        lines.append(f"\n  Sample differences (first 50):")
        lines.append(f"  {'Row':>4} {'Col':>4} {'Original':>10} {'Stego':>8}")
        lines.append(f"  {'-'*40}")
        for d in ic["sample_diffs"]:
            lines.append(f"  {d['row']:>4} {d['col']:>4} {d['original']:>10} {d['stego']:>8}")

        # Show rows with many diffs
        lines.append(f"\n  Patches changed per row (top 10 worst rows):")
        rows_with_counts = [
            (r, ic["row_diff_counts"][r])
            for r in range(len(ic["row_diff_counts"]))
            if ic["row_diff_counts"][r] > 0
        ]
        rows_with_counts.sort(key=lambda x: -x[1])
        lines.append(f"  {'Row':>4} {'Changes':>8}")
        lines.append(f"  {'-'*20}")
        for r, cnt in rows_with_counts[:10]:
            lines.append(f"  {r:>4} {cnt:>8}")

    # Pixel comparison
    lines.append("\n" + "-" * 80)
    lines.append("  PIXEL FIDELITY (before save vs after load)")
    lines.append("-" * 80)
    pc = report["pixel_comparison"]
    lines.append(f"\n  Mean absolute diff  : {pc['mean_abs_diff']:.6f}")
    lines.append(f"  Median absolute diff: {pc['median_abs_diff']:.6f}")
    lines.append(f"  Max absolute diff   : {pc['max_abs_diff']:.6f}")
    lines.append(f"  Std absolute diff   : {pc['std_abs_diff']:.6f}")
    lines.append(f"\n  Pixel range before  : [{pc['min_pixel_before']:.6f}, {pc['max_pixel_before']:.6f}]")
    lines.append(f"  Pixel range after   : [{pc['min_pixel_after']:.6f}, {pc['max_pixel_after']:.6f}]")

    ch = report["pixel_per_channel"]
    lines.append(f"\n  Per-channel:")
    for ch_name in ["R", "G", "B"]:
        lines.append(f"    {ch_name}: mean_abs_diff={ch[ch_name]['mean_abs_diff']:.6f}, "
                     f"max_abs_diff={ch[ch_name]['max_abs_diff']:.6f}")

    lines.append("\n" + "=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check VQGAN save/load round-trip index and pixel fidelity."
    )
    parser.add_argument("-m", "--model_directory",
                        type=str, default="logs/2020-11-09T13-31-51_sflckr",
                        help="Path to VQGAN model directory")
    parser.add_argument("-o", "--output_directory",
                        type=str, default="roundtrip_diagnostics_output",
                        help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--context-rows", type=int, default=0,
                        help="Number of context rows")
    args = parser.parse_args()

    if os.path.exists(args.output_directory):
        shutil.rmtree(args.output_directory)
    os.makedirs(args.output_directory)

    logger.enable_set(False)

    report = run_roundtrip_diagnostics(
        model_directory_path=args.model_directory,
        output_directory=args.output_directory,
        seed=args.seed,
        context_rows=args.context_rows,
    )

    # Write JSON
    json_path = os.path.join(args.output_directory, "roundtrip_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"JSON report: {json_path}")

    # Write text
    txt_path = os.path.join(args.output_directory, "roundtrip_report.txt")
    text = format_report_as_text(report)
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"Text report: {txt_path}")

    # Quick summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    ic = report["index_comparison"]
    pc = report["pixel_comparison"]
    if ic["identical"]:
        print("  INDEX ROUND-TRIP: ✓ IDENTICAL — no loss from PNG save/load")
    else:
        print(f"  INDEX ROUND-TRIP: ✗ {ic['total_mismatches']} mismatches "
              f"({ic['mismatch_ratio']:.4%})")
    print(f"  PIXEL MEAN ABS DIFF: {pc['mean_abs_diff']:.6f}")
    print(f"  PIXEL MAX  ABS DIFF: {pc['max_abs_diff']:.6f}")
    print()


if __name__ == "__main__":
    main()
