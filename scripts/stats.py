"""
Statistics collection and reporting for steganography encoding and decoding.
Tracks metrics for both encoder and decoder to enable analysis of the process.
"""
import json
from typing import Dict, List, Any
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


class StatsWriter:
    """Writes encoding and decoding statistics to a JSON file."""
    
    @staticmethod
    def write_stats(filepath: str, message: str, 
                   encoded_bits: str, decoded_text: str, decoded_bits: str,
                   encoding_stats: EncodingStatistics, 
                   decoding_stats: DecodingStatistics = None):
        """
        Write complete statistics to a JSON file.
        
        Args:
            filepath: Path to output JSON file
            message: Original message encoded
            encoded_bits: Bit representation of the message
            decoded_text: Decoded message from the image
            decoded_bits: Bits decoded from the image
            encoding_stats: EncodingStatistics object with patch-level data
            decoding_stats: Optional DecodingStatistics object with patch-level data
        """
        stats_data = {
            "message": {
                "original": message,
                "encoded_bits": encoded_bits,
                "total_bits_encoded": len(encoded_bits)
            },
            "encoding": encoding_stats.to_dict(),
            "decoding": decoding_stats.to_dict() if decoding_stats else None,
            "result": {
                "decoded_message": decoded_text,
                "decoded_bits": decoded_bits,
                "total_bits_decoded": len(decoded_bits),
                "match": message == decoded_text if decoded_text else False
            }
        }
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)
    
    @staticmethod
    def write_encoding_only(filepath: str, message: str,
                          encoded_bits: str,
                          encoding_stats: EncodingStatistics):
        """Write encoding statistics only (before decoding)."""
        stats_data = {
            "message": {
                "original": message,
                "encoded_bits": encoded_bits,
                "total_bits_encoded": len(encoded_bits)
            },
            "encoding": encoding_stats.to_dict(),
            "decoding": None,
            "result": None
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)
    
    @staticmethod
    def append_decoding_stats(filepath: str, decoded_text: str, decoded_bits: str,
                            decoding_stats: DecodingStatistics, 
                            original_message: str):
        """Update stats file with decoding results."""
        try:
            with open(filepath, 'r') as f:
                stats_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            stats_data = {}
        
        stats_data["decoding"] = decoding_stats.to_dict()
        stats_data["result"] = {
            "decoded_message": decoded_text,
            "decoded_bits": decoded_bits,
            "total_bits_decoded": len(decoded_bits),
            "match": original_message == decoded_text if decoded_text else False
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)
