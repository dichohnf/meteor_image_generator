"""
Error correction module for the steganography system.

Provides Reed-Solomon error correction over GF(256) via the ``reedsolo``
library.  The voting / repetition code (VoteCorrectionCode) has been
removed — RS is the only supported algorithm.

All codecs implement the abstract ErrorCorrectionCode interface with
    encode(bits: str) -> str
    decode(encoded_bits: str) -> str
"""

from abc import ABC, abstractmethod
from typing import ClassVar
from scripts.logger import logger


# ======================================================================
#  Abstract base
# ======================================================================

class ErrorCorrectionCode(ABC):
    """
    Abstract base class for error correction codecs.

    All concrete implementations must provide encode() and decode()
    methods that operate on binary strings ("010101...").
    """

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def encode(self, bits: str) -> str:
        """Encode a binary string by adding error correction redundancy."""
        ...

    @abstractmethod
    def decode(self, encoded_bits: str) -> str:
        """Decode a binary string by detecting/correcting errors."""
        ...

    @property
    def nsym(self) -> int:
        """Number of ECC symbols (0 for non-block codes)."""
        return 0

    @property
    @abstractmethod
    def overhead_ratio(self) -> float:
        """Overhead factor introduced by the code (1.0 = no overhead)."""
        ...


# ======================================================================
#  Reed-Solomon code via the ``reedsolo`` library
#
#  This wrapper uses the public reedsolo.RSCodec interface.
#  Install with:  pip install reedsolo
# ======================================================================

class ReedSolomonCorrectionCode(ErrorCorrectionCode):
    """
    Reed-Solomon error correction over GF(256).

    Wraps the ``reedsolo.RSCodec`` class.  Operates on bytes internally.
    The message bit string is padded to a byte boundary, RS-encoded, then
    converted back to bits.  Decoding reverses the process.

    Requires the ``reedsolo`` package (``pip install reedsolo``).

    Attributes:
        nsym: Number of ECC symbols (bytes) to append.
              Can correct up to nsym // 2 erroneous bytes.
    """

    name: ClassVar[str] = "reed_solomon"

    def __init__(self, nsym: int = 5) -> None:
        if nsym < 1:
            raise ValueError(f"nsym must be >= 1, got {nsym}")

        # Try importing reedsolo; fail early if not installed.
        try:
            from reedsolo import RSCodec
        except ImportError:
            raise ImportError(
                "ReedSolomonCorrectionCode requires the 'reedsolo' package.\n"
                "Install with:  pip install reedsolo"
            ) from None

        self._nsym = nsym
        self._codec = RSCodec(nsym)

    @property
    def nsym(self) -> int:
        return self._nsym

    # ---- public interface -----------------------------------------------

    def encode(self, bits: str) -> str:
        if not bits:
            return ""

        # Pad bits to byte boundary
        pad_len = (8 - len(bits) % 8) % 8
        padded_bits = bits + "0" * pad_len

        # Convert to bytes
        msg_bytes = self._bits_to_bytes(padded_bits)

        # RS encode (reedsolo.RSCodec.encode returns bytearray)
        codeword = bytes(self._codec.encode(bytearray(msg_bytes)))

        # Convert back to bits
        encoded_bits = self._bytes_to_bits(codeword)

        # Prepend padding length as 3 bits (0-7) so decoder knows how to strip
        header = format(pad_len, "03b")
        result = header + encoded_bits

        logger.info(
            f"[RS] encode: {len(bits)} bits -> {len(result)} bits "
            f"(nsym={self.nsym}, pad={pad_len})"
        )
        return result

    def decode(self, encoded_bits: str) -> str:
        if not encoded_bits:
            return ""

        # Minimum length: 3 header bits + at least 1 byte
        if len(encoded_bits) < 3 + 8:
            logger.warning("[RS] decode: encoded bits too short")
            return ""

        # Extract header (padding length)
        pad_len = int(encoded_bits[:3], 2)

        # Convert to bytes
        body_bits = encoded_bits[3:]
        if len(body_bits) < 8:
            return ""
        rx_bytes = self._bits_to_bytes(body_bits)

        # RS decode (reedsolo.RSCodec.decode returns (decoded, ecc, errata_pos))
        errata: list = []  # type: ignore[no-redef]
        try:
            decoded_ba, ecc_ba, errata = self._codec.decode(bytearray(rx_bytes))
            success = True
            corrected_bytes = bytes(decoded_ba)
        except Exception as exc:
            logger.warning(
                f"[RS] decode: RS decoding failed ({exc}). "
                "Returning raw data (strip ECC)."
            )
            success = False
            # Log diagnostic info
            logger.warning(
                f"[RS] decode diagnostic: rx_bytes={len(rx_bytes)}, "
                f"nsym={self.nsym}, stripped={len(rx_bytes) - self.nsym}, "
                f"first_bytes={' '.join(f'{b:02x}' for b in rx_bytes[:12])}"
            )
            corrected_bytes = rx_bytes[:len(rx_bytes) - self.nsym]

        # Convert back to bits
        decoded_bits = self._bytes_to_bits(corrected_bytes)

        # Strip padding
        if pad_len > 0:
            decoded_bits = decoded_bits[:-pad_len]

        if not success:
            logger.info("[RS] decode: completed with fallback (no correction)")
        else:
            logger.info(
                f"[RS] decode: success ({len(decoded_bits)} bits recovered, "
                f"{len(errata)} errors corrected)"
            )

        return decoded_bits

    @property
    def overhead_ratio(self) -> float:
        """
        For RS, the overhead depends on message size.
        Returns worst-case overhead for small messages.
        """
        return 1.0 + (self.nsym + 1) / max(1, 1)  # header byte + nsym bytes

    # ---- internal helpers -----------------------------------------------

    @staticmethod
    def _bits_to_bytes(bits: str) -> bytes:
        """Convert a binary string (multiple of 8) to bytes."""
        return bytes(
            int(bits[i:i + 8], 2) for i in range(0, len(bits), 8)
        )

    @staticmethod
    def _bytes_to_bits(data: bytes) -> str:
        """Convert bytes to a binary string."""
        return "".join(format(b, "08b") for b in data)

    def __repr__(self) -> str:
        return f"ReedSolomonCorrectionCode(nsym={self.nsym})"