"""
Statistics collection and reporting for steganography encoding and decoding.
Tracks metrics for both encoder and decoder to enable analysis of the process.
All stats are consolidated into a single JSON file per image.
"""
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class EncodingStatistics:
    """Collects statistics during the encoding process."""
    
    def __init__(self):
        self.patches = []  # List of per-patch stats
        
    def add_patch_stat(self, patch_index: int, row: int, col: int, 
                       selected_codebook_index: int, range_bottom: int, range_top: int,
                       encoded_bits: str, message_bits_encoded: int):
        """Record statistics for a single encoded patch."""
        self.patches.append({
            "patch_index": patch_index,
            "row": row,
            "col": col,
            "selected_codebook_index": selected_codebook_index,
            "probability_range": {
                "bottom": range_bottom,
                "top": range_top
            },
            "encoded_bits": encoded_bits,
            "message_bits_encoded": message_bits_encoded
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_patches_encoded": len(self.patches),
            "patches": self.patches
        }
    
    def get_grid_shape(self) -> Optional[Tuple[int, int]]:
        """Infer grid shape from patch positions."""
        if not self.patches:
            return None
        rows = set()
        cols = set()
        for p in self.patches:
            rows.add(p["row"])
            cols.add(p["col"])
        return (max(rows) + 1, max(cols) + 1)


class DecodingStatistics:
    """Collects statistics during the decoding process."""
    
    def __init__(self):
        self.patches = []  # List of per-patch stats
        
    def add_patch_stat(self, patch_index: int, row: int, col: int,
                       actual_token: int, range_bottom: int, range_top: int,
                       decoded_bits: str, success: bool = True):
        """Record statistics for a single decoded patch."""
        self.patches.append({
            "patch_index": patch_index,
            "row": row,
            "col": col,
            "actual_token": actual_token,
            "probability_range": {
                "bottom": range_bottom,
                "top": range_top
            },
            "decoded_bits": decoded_bits,
            "success": success
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        successful_patches = sum(1 for p in self.patches if p["success"])
        return {
            "total_patches_decoded": len(self.patches),
            "successful_patches": successful_patches,
            "patches": self.patches
        }
    
    def get_grid_shape(self) -> Optional[Tuple[int, int]]:
        """Infer grid shape from patch positions."""
        if not self.patches:
            return None
        rows = set()
        cols = set()
        for p in self.patches:
            rows.add(p["row"])
            cols.add(p["col"])
        return (max(rows) + 1, max(cols) + 1)


def _merge_patches(
    encoding_patches: List[Dict[str, Any]],
    decoding_patches: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge encoding and decoding per-patch stats into a single list where each
    entry contains both the encoding and decoding info for that patch.

    Encoding and decoding both iterate patches in the same raster-scan order,
    so they are merged by position (row, col) rather than by index alone.

    Parameters:
        encoding_patches: List of per-patch dicts from EncodingStatistics.
        decoding_patches: List of per-patch dicts from DecodingStatistics.

    Returns:
        A list of merged dicts, each with:
          - ``patch_index``, ``row``, ``col``
          - ``encoding``: encoding-specific fields (or ``null`` if patch was
            not used for encoding — e.g. random fill patches)
          - ``decoding``: decoding-specific fields
    """
    # Build a lookup from (row, col) → encoding entry
    enc_by_pos: Dict[Tuple[int, int], Dict[str, Any]] = {
        (p["row"], p["col"]): p for p in encoding_patches
    }

    merged = []
    for dec_patch in decoding_patches:
        key = (dec_patch["row"], dec_patch["col"])
        enc_patch = enc_by_pos.pop(key, None)

        entry = {
            "patch_index": dec_patch.get("patch_index", dec_patch["patch_index"]),
            "row": dec_patch["row"],
            "col": dec_patch["col"],
        }

        if enc_patch is not None:
            entry["encoding"] = {
                "selected_codebook_index": enc_patch["selected_codebook_index"],
                "probability_range": {
                    "bottom": enc_patch["probability_range"]["bottom"],
                    "top": enc_patch["probability_range"]["top"],
                },
                "encoded_bits": enc_patch["encoded_bits"],
                "message_bits_encoded": enc_patch["message_bits_encoded"],
            }
        else:
            entry["encoding"] = None

        entry["decoding"] = {
            "actual_token": dec_patch["actual_token"],
            "probability_range": {
                "bottom": dec_patch["probability_range"]["bottom"],
                "top": dec_patch["probability_range"]["top"],
            },
            "decoded_bits": dec_patch["decoded_bits"],
            "success": dec_patch["success"],
        }

        merged.append(entry)

    # Add any remaining encoding-only patches (shouldn't happen, but be safe)
    for enc_patch in enc_by_pos.values():
        entry = {
            "patch_index": enc_patch["patch_index"],
            "row": enc_patch["row"],
            "col": enc_patch["col"],
            "encoding": {
                "selected_codebook_index": enc_patch["selected_codebook_index"],
                "probability_range": {
                    "bottom": enc_patch["probability_range"]["bottom"],
                    "top": enc_patch["probability_range"]["top"],
                },
                "encoded_bits": enc_patch["encoded_bits"],
                "message_bits_encoded": enc_patch["message_bits_encoded"],
            },
            "decoding": None,
        }
        merged.append(entry)

    # Sort by (row, col) for deterministic order
    merged.sort(key=lambda e: (e["row"], e["col"]))
    return merged


def _list_to_matrix(indices: List[int], grid_shape: Tuple[int, int]) -> List[List[Optional[int]]]:
    """
    Reshape a flat list of indices into a 2D matrix (list of lists) of the
    given grid shape. Indices are arranged in raster-scan order (row-major).
    
    If the list is shorter than ``h * w``, the remaining cells are filled
    with ``None``. If longer, it is truncated to ``h * w``.
    """
    h, w = grid_shape
    total_cells = h * w
    # Truncate if longer than the grid, pad with None if shorter
    flat: List[Optional[int]] = list(indices[:total_cells])
    while len(flat) < total_cells:
        flat.append(None)
    matrix: List[List[Optional[int]]] = []
    for r in range(h):
        row_start = r * w
        row_end = row_start + w
        matrix.append(flat[row_start:row_end])
    return matrix


def _build_bit_comparison(encoded_bits: str, decoded_bits: str) -> List[Dict[str, Any]]:
    """
    Build a single list pairing encoded and decoded bits for easy comparison.
    
    Each entry contains:
      - index: bit position
      - encoded: the encoded bit value ('0' or '1')
      - decoded: the decoded bit value ('0' or '1')
      - match: whether encoded and decoded match
    
    For bits beyond the shorter sequence, the missing side is marked as None.
    """
    max_len = max(len(encoded_bits), len(decoded_bits))
    comparison = []
    for i in range(max_len):
        enc = encoded_bits[i] if i < len(encoded_bits) else None
        dec = decoded_bits[i] if i < len(decoded_bits) else None
        comparison.append({
            "index": i,
            "encoded": enc,
            "decoded": dec,
            "match": enc == dec if (enc is not None and dec is not None) else None
        })
    return comparison


class StatsWriter:
    """Writes consolidated encoding and decoding statistics to a JSON file.
    
    All auxiliary data (bits, indices, texts) are stored in the single JSON
    file, eliminating separate .txt files.  Encoded and decoded bits are
    presented as a single paired list for easy visual comparison.
    """
    
    @staticmethod
    def write_consolidated(
        filepath: str,
        *,
        # Original message
        original_message: str,
        # Encoded data
        encoded_bits: str,
        encoded_indices: List[int],
        encoding_stats: EncodingStatistics,
        # Decoded data
        decoded_bits: str,
        decoded_indices: List[int],
        recovered_text: str,
        decoding_stats: DecodingStatistics,
        # Metadata
        # Pipeline trace
        encode_trace: Optional[Dict[str, str]] = None,
        decode_trace: Optional[Dict[str, str]] = None,
        # Metadata
        options_dict: Optional[Dict[str, Any]] = None,
        pipeline_info: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Write a single consolidated statistics JSON file for one image.
        
        Args:
            filepath: Path to output JSON file.
            original_message: The original plain-text message.
            encoded_bits: Bit string after pipeline encoding (protected bits).
            encoded_indices: List of codebook indices selected during encoding.
            encoding_stats: EncodingStatistics with per-patch data.
            decoded_bits: Bit string from the decoder (raw bits).
            decoded_indices: List of codebook indices recovered during decoding.
            recovered_text: Plain-text message recovered after pipeline decoding.
            decoding_stats: DecodingStatistics with per-patch data.
            options_dict: Optional dict of Options attributes for provenance.
            pipeline_info: Optional dict describing the pipeline configuration.
            seed: The random seed used for this run.
        """
        # Build the paired bit-comparison list
        bit_comparison = _build_bit_comparison(encoded_bits, decoded_bits)
        
        # Merge encoding and decoding per-patch data into a single list
        enc_patch_list = encoding_stats.to_dict()
        dec_patch_list = decoding_stats.to_dict()
        merged_patches = _merge_patches(
            enc_patch_list.get("patches", []),
            dec_patch_list.get("patches", []),
        )
        total_patches_for_encoding = enc_patch_list.get("total_patches_encoded", 0)
        total_patches_decoded = dec_patch_list.get("total_patches_decoded", 0)
        successful_patches = dec_patch_list.get("successful_patches", 0)
        
        # Count errors (where both sides exist and differ)
        mismatches = [b for b in bit_comparison if b["match"] is False]
        error_count = len(mismatches)
        total_compared = sum(1 for b in bit_comparison if b["match"] is not None)
        error_ratio = error_count / total_compared if total_compared > 0 else 0.0
        
        # Determine match result
        message_match = original_message == recovered_text if recovered_text else False
        
        # Determine grid shape from per-patch statistics
        grid_shape = encoding_stats.get_grid_shape()
        if grid_shape is None:
            grid_shape = decoding_stats.get_grid_shape()
        # Fallback: try to infer from merged patches
        if grid_shape is None and merged_patches:
            rows = set(p["row"] for p in merged_patches)
            cols = set(p["col"] for p in merged_patches)
            grid_shape = (max(rows) + 1, max(cols) + 1)
        
        # Reshape flat index lists into 2D matrices
        if grid_shape is not None:
            encoded_indices_grid = _list_to_matrix(encoded_indices, grid_shape)
            decoded_indices_grid = _list_to_matrix(decoded_indices, grid_shape)
        else:
            encoded_indices_grid = []
            decoded_indices_grid = []
        
        stats_data = {
            "grid_shape": list(grid_shape) if grid_shape else None,
            "message": {
                "original": original_message,
            },
            "options": options_dict or {},
            "pipeline": pipeline_info or {},
            "pipeline_trace": {
                "encode": encode_trace or {},
                "decode": decode_trace or {},
            },
            "seed": seed,
            "encoding": {
                "bits": encoded_bits,
                "total_bits": len(encoded_bits),
                "indices": encoded_indices,
                "indices_grid": encoded_indices_grid,
                "total_indices": len(encoded_indices),
                "total_patches_encoded": total_patches_for_encoding,
            },
            "decoding": {
                "bits": decoded_bits,
                "total_bits": len(decoded_bits),
                "indices": decoded_indices,
                "indices_grid": decoded_indices_grid,
                "total_indices": len(decoded_indices),
                "total_patches_decoded": total_patches_decoded,
                "successful_patches": successful_patches,
                "recovered_text": recovered_text,
            },
            "per_patch": merged_patches,
            "bit_comparison": {
                "total_bits_compared": total_compared,
                "error_count": error_count,
                "error_ratio": error_ratio,
                "pairs": bit_comparison,
            },
            "result": {
                "message_match": message_match,
            },
        }
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)