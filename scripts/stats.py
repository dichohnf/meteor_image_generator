"""
Statistics collection and reporting for steganography encoding and decoding.
Tracks metrics for both encoder and decoder to enable analysis of the process.
All stats are consolidated into a single JSON file per image.
"""
import json
from typing import Dict, List, Any, Optional
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
        
        # Count errors (where both sides exist and differ)
        mismatches = [b for b in bit_comparison if b["match"] is False]
        error_count = len(mismatches)
        total_compared = sum(1 for b in bit_comparison if b["match"] is not None)
        error_ratio = error_count / total_compared if total_compared > 0 else 0.0
        
        # Determine match result
        message_match = original_message == recovered_text if recovered_text else False
        
        stats_data = {
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
                "total_indices": len(encoded_indices),
                "per_patch": encoding_stats.to_dict(),
            },
            "decoding": {
                "bits": decoded_bits,
                "total_bits": len(decoded_bits),
                "indices": decoded_indices,
                "total_indices": len(decoded_indices),
                "recovered_text": recovered_text,
                "per_patch": decoding_stats.to_dict(),
            },
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