"""
Error correction module for the steganography system.

Provides a configurable error correction code that adds redundancy to message bits
before encoding into images, and corrects errors during decoding.

The correction scheme uses a full-message repetition code with majority voting:
- The entire message (as a bit string) is repeated N times (odd number)
- During decoding, the repetitions are aligned and bit-by-bit majority voting
  recovers the original message
- This is robust against burst errors because a single corrupted patch only
  corrupts bits at a fixed position in each repetition, leaving the majority
  of repetitions intact at those positions.
- The burst_error_tolerance parameter controls the number of consecutive
  bit errors the algorithm can withstand in the encoded stream.
"""

import math
from typing import Tuple

from scripts.logger import logger


class ErrorCorrectionCode:
    """
    Provides error correction encoding and decoding for binary message bits.

    Uses a full-message repetition code with majority voting. Unlike a
    per-bit repetition code (which repeats each individual bit N times
    consecutively), this repeats the ENTIRE message N times. This design
    is specifically chosen to handle burst errors typical in VQGAN
    steganography, where a single corrupted patch can corrupt up to
    DEFAULT_PRECISION_BITS (16) consecutive bits in the decoded stream.

    With full-message repetition, a burst of B consecutive errors corrupts
    at most one position in each repetition of the message. If the number
    of repetitions N is odd and B < N, majority voting can still recover
    the correct bit at every position.

    Attributes:
        burst_error_tolerance: Maximum number of consecutive bit errors
            the code can withstand in the encoded stream.
        repetitions: Number of times the entire message is repeated (odd).
        _correctable_per_position: Number of errors that can be tolerated
            per bit position (= repetitions // 2).
    """

    def __init__(self, burst_error_tolerance: int = 10) -> None:
        """
        Initializes the error correction code with the given burst error tolerance.

        Args:
            burst_error_tolerance: Maximum number of consecutive bit errors
                the code can withstand (default 10). The number of repetitions
                is set to burst_error_tolerance * 2 + 1, ensuring that even
                if all bits in a single patch (up to burst_error_tolerance)
                are corrupted for a given position across repetitions,
                majority voting can still recover the correct value.

        Raises:
            ValueError: If burst_error_tolerance is negative.
        """
        if burst_error_tolerance < 0:
            raise ValueError(
                f"burst_error_tolerance must be >= 0, got {burst_error_tolerance}"
            )
        self.burst_error_tolerance = burst_error_tolerance
        # N repetitions: if burst destroys up to burst_error_tolerance copies
        # of a given bit position, the remaining copies still win via majority
        self.repetitions = burst_error_tolerance * 2 + 1
        self._correctable_per_position = self.repetitions // 2

    def encode(self, bits: str) -> str:
        """
        Encodes a binary string by repeating the entire message N times.

        This creates N consecutive copies of the full message bit string,
        which are then embedded into the image. If a patch is corrupted
        during VQGAN re-encoding, only one copy of the affected bits is lost,
        and majority voting can recover them.

        Args:
            bits: The original binary string (e.g., "10110").

        Returns:
            The encoded binary string with redundancy added.
        """
        if not bits:
            return ""

        encoded = bits * self.repetitions
        logger.info(
            f"Error correction encoding: {len(bits)} bits -> {len(encoded)} bits "
            f"(repetitions={self.repetitions}, burst_error_tolerance={self.burst_error_tolerance})"
        )
        return encoded

    def decode(self, encoded_bits: str) -> str:
        """
        Decodes a binary string by applying majority voting across repetitions.

        The encoded bits consist of N consecutive copies of the original message.
        The method splits the encoded stream into N blocks, aligns them by
        position, and applies majority voting for each bit position.

        Args:
            encoded_bits: The encoded binary string (potentially with errors).

        Returns:
            The corrected original binary string.
        """
        if not encoded_bits:
            return ""

        # The message length is the total encoded length divided by repetitions
        message_len = len(encoded_bits) // self.repetitions

        if message_len == 0:
            return ""

        decoded_parts = []
        total_errors_corrected = 0
        positions_with_errors = 0

        for pos in range(message_len):
            # Collect the bit at position 'pos' from each repetition
            bits_at_position = []
            for rep in range(self.repetitions):
                idx = rep * message_len + pos
                if idx < len(encoded_bits):
                    bits_at_position.append(encoded_bits[idx])

            # Count 0s and 1s
            ones = bits_at_position.count("1")
            zeros = len(bits_at_position) - ones

            if ones > zeros:
                recovered_bit = "1"
                errors_at_position = zeros
            elif zeros > ones:
                recovered_bit = "0"
                errors_at_position = ones
            else:
                # Tie (shouldn't happen with odd repetitions, but be safe)
                recovered_bit = bits_at_position[0]
                errors_at_position = len(bits_at_position) // 2

            decoded_parts.append(recovered_bit)
            total_errors_corrected += errors_at_position
            if errors_at_position > 0:
                positions_with_errors += 1

        decoded = "".join(decoded_parts)

        # Compute actual error statistics
        total_processed_bits = message_len * self.repetitions
        actual_error_ratio = (
            total_errors_corrected / total_processed_bits
            if total_processed_bits > 0
            else 0.0
        )

        if positions_with_errors > 0:
            logger.info(
                f"Error correction decoding: corrected {total_errors_corrected} bit errors "
                f"across {positions_with_errors} positions "
                f"(actual error ratio={actual_error_ratio:.4f}, "
                f"burst_error_tolerance={self.burst_error_tolerance})"
            )
            if total_errors_corrected > positions_with_errors:
                # Some positions had multiple errors
                max_errors = max(
                    sum(
                        1
                        for rep in range(self.repetitions)
                        if rep * message_len + pos < len(encoded_bits)
                        and encoded_bits[rep * message_len + pos] != decoded_parts[pos]
                    )
                    for pos in range(message_len)
                )
                if max_errors > self._correctable_per_position:
                    logger.warning(
                        f"A position had {max_errors} errors (max correctable = "
                        f"{self._correctable_per_position}). Some bits may still be corrupted."
                    )
        else:
            logger.info("Error correction decoding: no errors detected.")

        return decoded

    @property
    def overhead_ratio(self) -> float:
        """
        Returns the overhead ratio introduced by error correction encoding.
        E.g., repetitions=3 means 3x the original bits (200% overhead).
        """
        return self.repetitions

    def __repr__(self) -> str:
        return (
            f"ErrorCorrectionCode(burst_error_tolerance={self.burst_error_tolerance}, "
            f"repetitions={self.repetitions})"
        )