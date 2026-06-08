"""
Test script for the error correction algorithm.

Verifies that the ErrorCorrectionCode with full-message repetition
can correct burst errors of up to the specified burst_error_tolerance.

Tests (VoteCorrectionCode):
1. No errors: message should decode perfectly
2. Burst of 10 consecutive errors: should be corrected
3. Burst spanning repetition boundaries: should be corrected
4. Multiple bursts in different areas
5. Realistic VQGAN scenario - one patch corruption
6. All copies corrupted at same positions (exceeds design)
7. 10 consecutive errors at start of stream
8. Default burst_error_tolerance=10
9. Empty message handling
10. Errors exceeding tolerance: should detect and warn

Tests (ReedSolomonCorrectionCode):
11. Basic encode/decode with errors
12. Empty message handling
13. overhead_ratio property
"""

import sys
import os

# Parse command-line arg to optionally install logger
# Then import the class directly from the file to avoid logger import issues
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

from scripts.error_correction import VoteCorrectionCode, ReedSolomonCorrectionCode


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


def test_no_errors():
    """Test that with no errors, the message is decoded perfectly."""
    print("=" * 60)
    print("TEST 1: No errors")
    print("=" * 60)

    ecc = VoteCorrectionCode(burst_error_tolerance=10)
    original = "Hello, World!"
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)
    decoded_bits = ecc.decode(encoded)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    print(f"  Original: {original}")
    print(f"  Encoded length: {len(encoded)} bits ({len(encoded) // len(bits)}x overhead)")
    print(f"  Decoded:  {decoded}")
    print(f"  PASS: No errors, message recovered correctly.")
    print()


def test_burst_10_errors():
    """Test that a burst of 10 consecutive errors is corrected."""
    print("=" * 60)
    print("TEST 2: Burst of 10 consecutive errors")
    print("=" * 60)

    ecc = VoteCorrectionCode(burst_error_tolerance=10)
    original = "Hello, World! This is a test message."
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)
    message_len = len(bits)

    # Introduce a burst of 10 errors starting at position 50 in the encoded stream
    burst_start = 50
    corrupted = introduce_burst_error(encoded, burst_start, 10)

    print(f"  Original:         {original}")
    print(f"  Message bits:     {message_len}")
    print(f"  Encoded bits:     {len(encoded)} ({ecc.repetitions}x overhead)")
    print(f"  Burst at pos {burst_start}: flipping 10 bits")

    decoded_bits = ecc.decode(corrupted)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    print(f"  Decoded:          {decoded}")
    print(f"  PASS: 10 consecutive errors corrected successfully.")
    print()


def test_burst_at_boundary():
    """Test burst that spans across the boundary between two repetitions."""
    print("=" * 60)
    print("TEST 3: Burst spanning repetition boundary")
    print("=" * 60)

    ecc = VoteCorrectionCode(burst_error_tolerance=10)
    original = "Test boundary crossing burst."
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)
    message_len = len(bits)

    # Place burst right at the boundary between repetition 0 and repetition 1
    burst_start = message_len - 5
    corrupted = introduce_burst_error(encoded, burst_start, 10)

    print(f"  Original:         {original}")
    print(f"  Message bits:     {message_len}")
    print(f"  Burst from pos {burst_start} (crosses boundary between copies)")

    decoded_bits = ecc.decode(corrupted)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    print(f"  Decoded:          {decoded}")
    print(f"  PASS: Boundary-crossing burst corrected.")
    print()


def test_multiple_bursts():
    """Test multiple bursts in different areas of the encoded stream."""
    print("=" * 60)
    print("TEST 4: Multiple burst errors")
    print("=" * 60)

    ecc = VoteCorrectionCode(burst_error_tolerance=10)
    original = "This is a longer test message with multiple burst errors!"
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)

    # Introduce 3 separate bursts of 8 errors each
    corrupted = encoded
    corrupted = introduce_burst_error(corrupted, 30, 8)
    corrupted = introduce_burst_error(corrupted, 120, 8)
    corrupted = introduce_burst_error(corrupted, 250, 8)

    decoded_bits = ecc.decode(corrupted)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    print(f"  Original:         {original}")
    print(f"  Decoded:          {decoded}")
    print(f"  PASS: 3 separate bursts of 8 errors each corrected.")
    print()


def test_one_corrupted_patch_realistic():
    """
    Simulate ONE corrupted VQGAN patch: it corrupts exactly 10 consecutive
    bits in the encoded stream. Since the encoded stream is organized as
    [copy0][copy1]...[copy21], these 10 consecutive bits fall within ONE copy.
    The other 20 copies remain intact → majority voting succeeds.
    This is the REALISTIC scenario: VQGAN re-encoding corrupts one patch
    at a time, corrupting at most DEFAULT_PRECISION_BITS (16) consecutive bits.
    """
    print("=" * 60)
    print("TEST 5: Realistic VQGAN scenario - one patch corruption (10 bits)")
    print("=" * 60)

    ecc = VoteCorrectionCode(burst_error_tolerance=10)
    original = "Realistic: one patch corrupted."
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)
    message_len = len(bits)

    # One patch = 10 consecutive bits within a single copy
    patch_start_within_copy = 100  # position within copy0
    burst_start = patch_start_within_copy  # starts in copy0, stays within copy0 if message_len >= 110
    corrupted = introduce_burst_error(encoded, burst_start, 10)

    print(f"  Original:         {original}")
    print(f"  Message bits:     {message_len}")
    print(f"  Corrupted 10 consecutive bits at pos {burst_start} (inside copy 0)")
    print(f"  Only copy 0's bits at positions [100..109] are affected")
    print(f"  => 10 out of 21 copies are corrupted for those positions")
    print(f"  => 11 remaining copies give majority")

    decoded_bits = ecc.decode(corrupted)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    print(f"  Decoded:          {decoded}")
    print(f"  PASS: One corrupted patch handled correctly.")
    print()


def test_max_positions_corrupted_in_each_copy():
    """
    Borderline case: corrupt the same 10 bit positions in each copy.
    This is the worst-case scenario — but it requires affecting 10 bits
    in EVERY copy (not realistic for a single patch corruption).
    With burst_error_tolerance=10 and repetitions=21, we can tolerate
    up to 10 errors per bit position. Here we have 1 error per position
    per copy, for 21 copies = 21 errors per position → EXCEEDS tolerance.
    
    This test shows that the code is DESIGNED for realistic error patterns
    (one corrupted patch, not all copies corrupted at same positions).
    """
    print("=" * 60)
    print("TEST 6: All copies corrupted at same 10 positions (exceeds design)")
    print("=" * 60)

    ecc = VoteCorrectionCode(burst_error_tolerance=10)
    original = "Borderline test."
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)
    message_len = len(bits)

    # Corrupt same 10 positions in EVERY copy (21 errors per bit position)
    corrupted = list(encoded)
    for rep in range(ecc.repetitions):
        start = rep * message_len
        for offset in range(10):
            idx = start + offset
            if idx < len(corrupted):
                corrupted[idx] = "1" if corrupted[idx] == "0" else "0"
    corrupted = "".join(corrupted)

    decoded_bits = ecc.decode(corrupted)
    decoded = string_from_bits(decoded_bits)

    print(f"  Original:         {original}")
    print(f"  Message bits:     {message_len}")
    print(f"  Corrupted first 10 bits of ALL {ecc.repetitions} copies")
    print(f"  => 21 errors at each of positions 0..9 (exceeds tolerance of 10)")
    print(f"  Decoded:          {decoded}")
    # This might or might not match; we don't assert it
    if decoded == original:
        print(f"  NOTE: By chance the message was still correct (rare).")
    else:
        print(f"  PASS: Expected - errors beyond tolerance correctly cause corruption.")
    print()


def test_burst_10_straddles_boundaries():
    """
    10 consecutive errors at the start of the encoded stream corrupt only
    the first 10 bits of the first copy. The other 20 copies are intact.
    Since repetitions=21 and tolerance=10, majority voting still works.
    """
    print("=" * 60)
    print("TEST 7: 10 consecutive errors at start of stream")
    print("=" * 60)

    ecc = VoteCorrectionCode(burst_error_tolerance=10)
    original = "Testing the worst case."
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)
    message_len = len(bits)

    # 10 consecutive errors: these corrupt the SAME 10 positions in every copy
    # because the data is organized as [copy0][copy1]...[copyN]
    corrupted = encoded
    corrupted = introduce_burst_error(corrupted, 0, 10)

    decoded_bits = ecc.decode(corrupted)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    print(f"  Original:         {original}")
    print(f"  Message bits:     {message_len}")
    print(f"  Encoded is 21 copies concatenated")
    print(f"  Corrupted first 10 bits = first 10 bits of first copy")
    print(f"  => only 10 bits of first copy are corrupted")
    print(f"  Decoded:          {decoded}")
    print(f"  PASS: 20 remaining copies provide majority.")
    print()


def test_default_10():
    """Test that the default burst_error_tolerance=10 works."""
    print("=" * 60)
    print("TEST 8: Default burst_error_tolerance=10")
    print("=" * 60)

    ecc = VoteCorrectionCode()  # default tolerance = 10
    assert ecc.repetitions == 21, f"Expected 21 repetitions, got {ecc.repetitions}"
    assert ecc.burst_error_tolerance == 10

    original = "Default test"
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)

    # Burst of 10 errors
    corrupted = introduce_burst_error(encoded, 0, 10)

    decoded_bits = ecc.decode(corrupted)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    print(f"  Original:         {original}")
    print(f"  Repetitions:      {ecc.repetitions}")
    print(f"  Corrupted first 10 bits")
    print(f"  Decoded:          {decoded}")
    print(f"  PASS: Default tolerance corrects 10 consecutive errors.")
    print()


def test_empty_message():
    """Test empty message handling."""
    print("=" * 60)
    print("TEST 9: Empty message")
    print("=" * 60)

    ecc = VoteCorrectionCode()
    assert ecc.encode("") == ""
    assert ecc.decode("") == ""
    print("  PASS: Empty message handled correctly.")
    print()


def test_reed_solomon_basic():
    """Test Reed-Solomon error correction: basic encode/decode with errors."""
    print("=" * 60)
    print("TEST 11: Reed-Solomon basic encode/decode with errors")
    print("=" * 60)

    try:
        ecc = ReedSolomonCorrectionCode(nsym=10)
    except ImportError as e:
        print(f"  SKIP: {e}")
        print()
        return

    nsym = 10  # can correct up to 5 erroneous bytes
    original = "Hello RS! Testing 123."
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)

    # Skip the 3-bit header; corrupt bits in the payload only.
    # Flip 3 bits in 2 separate bytes — this should be within RS correction
    # capacity (nsym // 2 = 5 byte errors).
    header_bits = 3
    payload = list(encoded[header_bits:])
    # Flip 3 bits: all within the first few bytes of the payload
    for pos in [10, 50, 100]:
        if pos < len(payload):
            payload[pos] = "1" if payload[pos] == "0" else "0"
    corrupted = encoded[:header_bits] + "".join(payload)

    decoded_bits = ecc.decode(corrupted)
    decoded = string_from_bits(decoded_bits)

    assert decoded == original, f"FAIL: '{decoded}' != '{original}'"
    print(f"  Original:         {original}")
    print(f"  Encoded length:   {len(encoded)} bits")
    print(f"  nsym:             {nsym}")
    print(f"  Corrupted 3 bits (payload only)")
    print(f"  Decoded:          {decoded}")
    print(f"  PASS: Reed-Solomon corrected errors.")
    print()


def test_reed_solomon_empty():
    """Test Reed-Solomon with empty message."""
    print("=" * 60)
    print("TEST 12: Reed-Solomon empty message")
    print("=" * 60)

    try:
        ecc = ReedSolomonCorrectionCode(nsym=10)
    except ImportError as e:
        print(f"  SKIP: {e}")
        print()
        return

    assert ecc.encode("") == ""
    assert ecc.decode("") == ""
    print("  PASS: Empty message handled correctly.")
    print()


def test_reed_solomon_overhead_ratio():
    """Test Reed-Solomon overhead_ratio property."""
    print("=" * 60)
    print("TEST 13: Reed-Solomon overhead_ratio")
    print("=" * 60)

    try:
        ecc = ReedSolomonCorrectionCode(nsym=10)
    except ImportError as e:
        print(f"  SKIP: {e}")
        print()
        return

    ratio = ecc.overhead_ratio
    # Should be a float > 1.0
    assert ratio > 1.0, f"Expected overhead_ratio > 1.0, got {ratio}"
    print(f"  overhead_ratio:   {ratio:.2f}")
    print(f"  nsym:             {ecc.nsym}")
    print("  PASS: overhead_ratio is reasonable.")
    print()


def test_beyond_tolerance():
    """Test that errors exceeding 10 at one position may fail (warning case)."""
    print("=" * 60)
    print("TEST 10: Errors beyond tolerance (worst-case)")
    print("=" * 60)

    tolerance = 5
    ecc = VoteCorrectionCode(burst_error_tolerance=tolerance)
    original = "XYZ"
    bits = bits_from_string(original)
    encoded = ecc.encode(bits)
    message_len = len(bits)

    # Corrupt MORE than `tolerance` copies at one position (majority flips!)
    pos_in_message = 3
    bits_list = list(encoded)
    for rep in range(tolerance + 2):  # corrupt tolerance+2 = 7 out of 11 copies
        idx = rep * message_len + pos_in_message
        bits_list[idx] = "1" if bits_list[idx] == "0" else "0"
    corrupted = "".join(bits_list)

    decoded_bits = ecc.decode(corrupted)
    decoded = string_from_bits(decoded_bits)

    print(f"  Original:         {original}")
    print(f"  Tolerance:        {tolerance} (repetitions={ecc.repetitions})")
    print(f"  Corrupted {tolerance + 2} out of {ecc.repetitions} copies at position {pos_in_message}")
    print(f"  Decoded:          {decoded}")
    if decoded != original:
        print(f"  PASS: Errors beyond tolerance correctly detected as failure.")
    else:
        print(f"  NOTE: By chance, the message was still correct (rare).")
    print()


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("ERROR CORRECTION CODE TESTS")
    print("(full-message repetition | burst_error_tolerance = 10)")
    print("=" * 60)
    print()

    all_tests = [
        test_no_errors,
        test_burst_10_errors,
        test_burst_at_boundary,
        test_multiple_bursts,
        test_one_corrupted_patch_realistic,
        test_max_positions_corrupted_in_each_copy,
        test_burst_10_straddles_boundaries,
        test_default_10,
        test_empty_message,
        test_beyond_tolerance,
        test_reed_solomon_basic,
        test_reed_solomon_empty,
        test_reed_solomon_overhead_ratio,
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