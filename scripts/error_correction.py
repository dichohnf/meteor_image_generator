"""
Error correction module for the steganography system.

Provides a configurable error correction code that adds redundancy to message bits
before encoding into images, and corrects errors during decoding.

The correction scheme uses a repetition code with majority voting:
- Each message bit is repeated N times (where N is derived from max_error_ratio)
- During decoding, majority voting recovers the original bit
- The max_error_ratio hyper-parameter controls the trade-off between
  redundancy and the number of correctable bit errors.
"""

import math
from typing import Tuple

from scripts.logger import logger


def _repetition_factor(max_error_ratio: float) -> int:
    """
    Computes the repetition factor based on the maximum tolerable error ratio.

    The repetition factor is chosen such that the overhead is approximately
    max_error_ratio of the original bits, ensuring that up to half of the
    repetitions can be corrupted and the original bit can still be recovered
    via majority voting.

    Args:
        max_error_ratio: Maximum tolerable bit error ratio (0.0 to 1.0).

    Returns:
        The repetition factor (odd number >= 3).
    """
    # We need an odd repetition factor for unambiguous majority voting.
    # The error correction capability floor(k/2) bits per group of k repetitions
    # must be >= ceil(max_error_ratio * k).
    # For simplicity: factor = 2 * ceil(max_error_ratio * 5) + 1, clamped to a minimum of 3.
    raw_factor = max(3, 2 * math.ceil(max_error_ratio * 5) + 1)
    # Ensure odd
    if raw_factor % 2 == 0:
        raw_factor += 1
    return raw_factor


class ErrorCorrectionCode:
    """
    Provides error correction encoding and decoding for binary message bits.

    Uses a repetition code with majority voting. The repetition factor is
    derived from the configurable max_error_ratio hyper-parameter.

    Attributes:
        max_error_ratio: Maximum tolerable bit error ratio (0.0 to 1.0).
        repetition_factor: Number of times each message bit is repeated.
    """

    def __init__(self, max_error_ratio: float = 0.2) -> None:
        """
        Initializes the error correction code with the given max error ratio.

        Args:
            max_error_ratio: Maximum tolerable bit error ratio (default 0.2 = 20%).
                This controls how much redundancy is added. A value of 0.2 means
                the algorithm can tolerate up to ~20% of bits being corrupted.
                Must be in the range [0.0, 1.0].

        Raises:
            ValueError: If max_error_ratio is outside the valid range.
        """
        if not 0.0 <= max_error_ratio <= 1.0:
            raise ValueError(
                f"max_error_ratio must be between 0.0 and 1.0, got {max_error_ratio}"
            )
        self.max_error_ratio = max_error_ratio
        self.repetition_factor = _repetition_factor(max_error_ratio)
        self._correctable_bits_per_group = self.repetition_factor // 2

    def encode(self, bits: str) -> str:
        """
        Encodes a binary string by adding repetition-based error correction redundancy.

        Args:
            bits: The original binary string (e.g., "10110").

        Returns:
            The encoded binary string with redundancy added.
        """
        if not bits:
            return ""

        encoded_parts = []
        for bit in bits:
            encoded_parts.append(bit * self.repetition_factor)

        encoded = "".join(encoded_parts)
        logger.info(
            f"Error correction encoding: {len(bits)} bits -> {len(encoded)} bits "
            f"(repetition factor={self.repetition_factor})"
        )
        return encoded

    def decode(self, encoded_bits: str) -> str:
        """
        Decodes a binary string by applying majority voting error correction.

        Args:
            encoded_bits: The encoded binary string (potentially with errors).

        Returns:
            The corrected original binary string.
        """
        if not encoded_bits:
            return ""

        total_groups = len(encoded_bits) // self.repetition_factor
        decoded_parts = []
        error_count = 0
        total_bits_corrected = 0

        for i in range(total_groups):
            start = i * self.repetition_factor
            end = start + self.repetition_factor
            group = encoded_bits[start:end]

            # Majority voting: count 0s and 1s
            ones = group.count("1")
            zeros = self.repetition_factor - ones

            if ones > zeros:
                recovered_bit = "1"
                # Errors are the minority count
                errors_in_group = zeros
            elif zeros > ones:
                recovered_bit = "0"
                errors_in_group = ones
            else:
                # Tie (shouldn't happen with odd repetition factor, but be safe)
                recovered_bit = group[0]
                errors_in_group = self.repetition_factor // 2

            decoded_parts.append(recovered_bit)
            total_bits_corrected += errors_in_group
            if errors_in_group > 0:
                error_count += 1

        decoded = "".join(decoded_parts)

        # Compute actual error ratio
        total_decoded_bits = len(encoded_bits)
        actual_error_ratio = total_bits_corrected / total_decoded_bits if total_decoded_bits > 0 else 0.0

        if error_count > 0:
            logger.info(
                f"Error correction decoding: corrected {total_bits_corrected} bit errors "
                f"across {error_count} groups (actual error ratio={actual_error_ratio:.4f}, "
                f"max tolerance={self.max_error_ratio:.2f})"
            )
            if actual_error_ratio > self.max_error_ratio:
                logger.warning(
                    f"Actual error ratio {actual_error_ratio:.4f} exceeds max tolerance "
                    f"{self.max_error_ratio:.2f}. Some bits may still be corrupted."
                )
        else:
            logger.info("Error correction decoding: no errors detected.")

        return decoded

    @property
    def overhead_ratio(self) -> float:
        """
        Returns the overhead ratio introduced by error correction encoding.
        E.g., a repetition factor of 3 means 3x the original bits (200% overhead).
        """
        return self.repetition_factor

    def __repr__(self) -> str:
        return (
            f"ErrorCorrectionCode(max_error_ratio={self.max_error_ratio}, "
            f"repetition_factor={self.repetition_factor})"
        )