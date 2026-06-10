"""
Error correction module for the steganography system.

Provides multiple error correction algorithms:
1. VoteCorrectionCode — full-message repetition with majority voting (burst-resistant)
2. ReedSolomonCorrectionCode — Reed-Solomon block code over GF(256) via ``reedsolo``
3. ErrorCorrectionFactory — static factory to build the appropriate codec by name.

All codecs implement the abstract ErrorCorrectionCode interface with
    encode(bits: str) -> str
    decode(encoded_bits: str) -> str
"""

from abc import ABC, abstractmethod
from typing import ClassVar, List
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
#  1 — VoteCorrectionCode (full-message repetition + majority voting)
# ======================================================================

class VoteCorrectionCode(ErrorCorrectionCode):
    """
    Full-message repetition code with majority voting.

    The entire message bit string is repeated N times (odd number).
    During decoding, the repetitions are aligned and bit-by-bit majority
    voting recovers each original bit.

    This code is optimised for **burst errors** caused by VQGAN patch
    re-encoding.  A single patch corruption flips at most
    ``DEFAULT_PRECISION_BITS`` (16) consecutive bits within **one copy**
    of the message.  With ``repetitions = 2 * burst_error_tolerance + 1``,
    the algorithm can survive up to ``burst_error_tolerance`` such
    corrupted patches without losing data.

    The algorithm is:

        repetitions = 2 * burst_error_tolerance + 1
        encoded_bits = message_bits * repetitions
        decode:
            for each bit position i:
                vote = majority(copy_0[i], copy_1[i], ..., copy_N[i])
                recovered[i] = vote

    Attributes:
        burst_error_tolerance: Maximum number of patches that can be
            corrupted without losing data (default 10).
        repetitions: Number of copies of the message (always odd).
    """

    name: ClassVar[str] = "vote"

    def __init__(self, burst_error_tolerance: int = 10) -> None:
        if burst_error_tolerance < 1:
            raise ValueError(
                f"burst_error_tolerance must be >= 1, got {burst_error_tolerance}"
            )
        self.burst_error_tolerance = burst_error_tolerance
        self.repetitions = burst_error_tolerance * 2 + 1
        self._correctable_per_position = self.repetitions // 2

    def encode(self, bits: str) -> str:
        if not bits:
            return ""
        encoded = bits * self.repetitions
        logger.info(
            f"[Vote] encode: {len(bits)} bits -> {len(encoded)} bits "
            f"(repetitions={self.repetitions})"
        )
        return encoded

    def decode(self, encoded_bits: str) -> str:
        if not encoded_bits:
            return ""

        message_len = len(encoded_bits) // self.repetitions
        if message_len == 0:
            return ""

        decoded_parts: List[str] = []
        total_errors = 0
        error_positions = 0

        for pos in range(message_len):
            bits_at_pos = [
                encoded_bits[rep * message_len + pos]
                for rep in range(self.repetitions)
                if rep * message_len + pos < len(encoded_bits)
            ]
            ones = bits_at_pos.count("1")
            zeros = len(bits_at_pos) - ones

            if ones > zeros:
                recovered = "1"
                err = zeros
            elif zeros > ones:
                recovered = "0"
                err = ones
            else:
                recovered = bits_at_pos[0]
                err = len(bits_at_pos) // 2

            decoded_parts.append(recovered)
            total_errors += err
            if err > 0:
                error_positions += 1

        decoded = "".join(decoded_parts)

        total_bits = message_len * self.repetitions
        err_ratio = total_errors / total_bits if total_bits else 0.0

        if error_positions > 0:
            logger.info(
                f"[Vote] decode: corrected {total_errors} errors across "
                f"{error_positions} positions (err_ratio={err_ratio:.4f})"
            )
        else:
            logger.info("[Vote] decode: no errors detected.")

        return decoded

    @property
    def overhead_ratio(self) -> float:
        return float(self.repetitions)

    def __repr__(self) -> str:
        return (
            f"VoteCorrectionCode(burst_error_tolerance={self.burst_error_tolerance}, "
            f"repetitions={self.repetitions})"
        )


# ======================================================================
#  2 — Reed-Solomon code via the ``reedsolo`` library
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

    def __init__(self, nsym: int = 10, **kwargs) -> None:
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

        # Forward any compatible kwargs to RSCodec constructor
        c_primitive = kwargs.pop("c_primitive", None)

        self._nsym = nsym
        self._codec = (
            RSCodec(nsym) if c_primitive is None
            else RSCodec(nsym, c_primitive=c_primitive)
        )

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


# ======================================================================
#  3 — ErrorCorrectionFactory
# ======================================================================

class ErrorCorrectionFactory:
    """
    Static factory for building ErrorCorrectionCode instances by name.

    Usage::

        ecc = ErrorCorrectionFactory.create("vote", burst_error_tolerance=10)
        ecc = ErrorCorrectionFactory.create("reed_solomon", nsym=10)
    """

    _REGISTRY: ClassVar[dict] = {
        "vote": VoteCorrectionCode,
        "reed_solomon": ReedSolomonCorrectionCode,
    }

    @staticmethod
    def register(name: str, codec_class: type) -> None:
        """Register a custom codec class under *name*."""
        ErrorCorrectionFactory._REGISTRY[name] = codec_class

    @staticmethod
    def create(method: str, **kwargs) -> ErrorCorrectionCode:
        """
        Build and return an ErrorCorrectionCode instance.

        Args:
            method: Algorithm name.  Built-in choices are ``"vote"`` and
                    ``"reed_solomon"``.  Custom names can be added via
                    :meth:`register`.
            **kwargs: Keyword arguments forwarded to the concrete class
                      constructor.

        Returns:
            An :class:`ErrorCorrectionCode` instance.

        Raises:
            ValueError: If *method* is not recognised.
        """
        cls = ErrorCorrectionFactory._REGISTRY.get(method)
        if cls is None:
            raise ValueError(
                f"Unknown error correction method {method!r}. "
                f"Available: {list(ErrorCorrectionFactory._REGISTRY)}"
            )
        logger.info(
            f"Factory: building {cls.__name__} with kwargs {kwargs}"
        )
        return cls(**kwargs)