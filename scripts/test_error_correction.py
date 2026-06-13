"""
Test script for the Reed-Solomon error correction algorithm.

RS is now the only ECC method (no Vote). Parameters are fixed:
    nsym=6, block_bytes=10  (≈30% correction capability per 10-byte block)

Tests:
1. Basic encode/decode with no errors
2. Encode/decode with bit errors (within correction capacity)
3. Block-level encoding (multiple 10-byte blocks)
4. Partial block (last block shorter than 10 bytes)
5. Empty message handling
6. overhead_ratio property
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a minimal logger module so error_correction.py can import it
import types
logger_module = types.ModuleType('scripts.logger')
logger_module.logger = type('Logger', (), {
    'info': lambda *a, **kw: None,
    'warning': lambda *a, **kw: None,
    'error': lambda *a, **kw: None,
    'enable_set': lambda *a, **kw: None,
})()
sys.modules['scripts.logger'] = logger_module

# RS is always nsym=6 (fixed). These constants match the pipeline's block scheme.
RS_NSYM = 6
RS_BLOCK_BYTES = 10

from scripts.error_correction import ReedSolomonCorrectionCode


def bits_from_string(s: str) -> str:
    """Convert a string to a binary string."""
    return "".join(format(ord(c), "08b") for c in s)


def string_from_bits(bits: str) -> str:
    """Convert a binary string back to a string."""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return "".join(chars)


def introduce_burst_error(encoded_bits: str, start_pos: int, length: int) -> str:
    """Flip `length` consecutive bits starting at `start_pos`."""
    bits_list = list(encoded_bits)
    for i in range(start_pos, min(start_pos + length, len(bits_list))):
        bits_list[i] = "1" if bits_list[i] == "0" else "0"
    return "".join(bits_list)


def test_basic_encode_decode():
    """Test basic RS encode/decode with no errors."""
    print("=" * 60)
    print("TEST 1: Basic encode/decode (no errors)")
    print("=" * 60)

    ecc = ReedSolomonCorrectionCode(nsym=RS_NSYM)
    original = "Hello RS!"
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)
    decoded_bits = ecc.decode(encoded)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    # Encoded should be longer due to ECC: bits * (15/10) + RS overhead
    # But block-based: each 80-bit block → 120 bits, so ratio ≤ 1.5
    print(f"  Original:     {original}")
    print(f"  Bits:         {len(bits)}")
    print(f"  Encoded bits: {len(encoded)} ({len(encoded)/len(bits):.2f}x overhead)")
    print(f"  Decoded:      {decoded}")
    print(f"  PASS")
    print()


def test_error_correction_within_capacity():
    """Test RS corrects errors within its capacity (nsym/2 = 2 byte errors per block)."""
    print("=" * 60)
    print("TEST 2: Error correction within capacity")
    print("=" * 60)

    ecc = ReedSolomonCorrectionCode(nsym=RS_NSYM)
    # 10-byte message → fits in one block
    original = "1234567890"
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)

    # Flip 4 bits across 2 bytes in the encoded data
    # (can correct up to nsym//2 = 2 byte errors → more than enough)
    corrupted = introduce_burst_error(encoded, 10, 4)

    decoded_bits = ecc.decode(corrupted)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    print(f"  Original:     {original}")
    print(f"  Encoded:      {len(encoded)} bits")
    print(f"  Flipped 4 bits at position 10")
    print(f"  Decoded:      {decoded}")
    print(f"  PASS")
    print()


def test_block_encoding_multiple_blocks():
    """Test encoding with multiple 10-byte RS blocks."""
    print("=" * 60)
    print("TEST 3: Multi-block encoding")
    print("=" * 60)

    ecc = ReedSolomonCorrectionCode(nsym=RS_NSYM)
    # 30-byte message → 30 bytes = 15 bytes padded = 15 encoded bytes = 120 bits
    # RS pads to unit of RS_BLOCK_BYTES (10 bytes), so 30 bytes → pads to 30 bytes
    # (already multiple of 10). Then nsym=6 → 30*8 + 6*8 = 288 bits payload+ECC.
    # With 3-bit header and byte padding, total is ~291 bits.
    original = "A" * 30
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)
    decoded_bits = ecc.decode(encoded)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    # reedsolo encodes the entire message as a single RS codeword,
    # adding nsym=6 parity bytes at the end.
    # 30 bytes → 30 + 6 = 36 bytes = 288 bits, plus 3-bit header = 291 bits.
    expected_min_len = len(bits) + RS_NSYM * 8 + 3  # bits + ECC + header
    assert len(encoded) >= expected_min_len, \
        f"Encoded too short: {len(encoded)} < {expected_min_len} (raw body {len(bits)}b)"
    print(f"  Original:     {len(original)} bytes = {len(bits)} bits")
    print(f"  Encoded bits: {len(encoded)} ({len(encoded)/len(bits):.2f}x overhead)")
    print(f"  Decoded:      {len(decoded)} chars")
    print(f"  PASS")
    print()


def test_partial_last_block():
    """Test encoding with last block shorter than 10 bytes."""
    print("=" * 60)
    print("TEST 4: Partial last block")
    print("=" * 60)

    ecc = ReedSolomonCorrectionCode(nsym=RS_NSYM)
    # 12 bytes → 1 full block (10B) + 1 partial block (2B padded to 10B)
    original = "abcdefghijkl"
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)
    decoded_bits = ecc.decode(encoded)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    print(f"  Original:     {original}")
    print(f"  Bits:         {len(bits)}")
    print(f"  Encoded bits: {len(encoded)}")
    print(f"  Decoded:      {decoded}")
    print(f"  PASS")
    print()


def test_empty_message():
    """Test empty message handling."""
    print("=" * 60)
    print("TEST 5: Empty message")
    print("=" * 60)

    ecc = ReedSolomonCorrectionCode(nsym=RS_NSYM)
    assert ecc.encode("") == ""
    assert ecc.decode("") == ""
    print("  PASS")
    print()


def test_overhead_ratio():
    """Test overhead_ratio property."""
    print("=" * 60)
    print("TEST 6: overhead_ratio")
    print("=" * 60)

    ecc = ReedSolomonCorrectionCode(nsym=RS_NSYM)
    ratio = ecc.overhead_ratio
    # Each 10-byte block → 15 bytes = 1.5x overhead in the optimal case
    # But with padding and byte alignment, actual ratio may be slightly higher
    assert ratio > 1.0, f"Expected overhead_ratio > 1.0, got {ratio}"
    print(f"  overhead_ratio: {ratio:.4f}")
    print(f"  nsym:           {ecc.nsym}")
    print("  PASS")
    print()


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("RS ERROR CORRECTION TESTS (fixed nsym=6, block_bytes=10)")
    print("=" * 60)
    print()

    all_tests = [
        test_basic_encode_decode,
        test_error_correction_within_capacity,
        test_block_encoding_multiple_blocks,
        test_partial_last_block,
        test_empty_message,
        test_overhead_ratio,
    ]

    passed = 0
    failed = 0
    for test in all_tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            print()
            failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")
            print()
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(all_tests)} tests")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)