"""
Steganographic pipeline for message transformation.

Chains together all string-level transformations independently from the
VQGAN image encoding/decoding:

    encode:  message (str) → XOR + CharToBits + ErrorCorrection → bits (str)
    decode:  bits (str)    → ErrorCorrection + BitsToChar + XOR → message (str)

The pipeline is self-contained: the encoder/decoder receive/return raw bits
and never see the internals of the pipeline.
"""

from typing import Optional

from scripts.error_correction import ErrorCorrectionFactory, ErrorCorrectionCode
from scripts.logger import logger
from scripts.utils import bits2string, string2bits


# ======================================================================
#  XOR bit mask
# ======================================================================

class XorMask:
    """
    Applies/reverses a bitwise XOR mask on a binary string.

    The mask is derived from an integer key and repeated to match the
    length of the input.  XOR is its own inverse, so the same
    :meth:`apply` serves both directions.
    """

    def __init__(self, key: int) -> None:
        if key < 0 or key > 255:
            raise ValueError(f"XOR key must be in [0, 255], got {key}")
        self.key = key
        self._key_bits = format(key, "08b")
        logger.info(f"XorMask initialized with key={key} (0b{self._key_bits})")

    def apply(self, bits: str) -> str:
        """XOR *bits* with the repeated key pattern."""
        if not bits:
            return ""
        key_len = len(self._key_bits)
        result_chars = []
        for i, ch in enumerate(bits):
            k = self._key_bits[i % key_len]
            result_chars.append("1" if ch != k else "0")
        return "".join(result_chars)


# ======================================================================
#  Main pipeline
# ======================================================================

class SteganoPipeline:
    """
    Full string-transformation pipeline for steganography.

    Composes, in order:
        1. (optional) XOR bit mask  – obfuscation layer
        2. Character encoding        – str → binary via *char_encoding*
        3. Error correction coding   – redundancy against VQGAN burst errors

    Usage::

        pipe = SteganoPipeline(
            xor_key=123,
            char_encoding="ASCII",
            error_correction_method="vote",
            burst_error_tolerance=10,
        )
        protected_bits = pipe.encode_message("Hello")
        recovered_msg = pipe.decode_message(protected_bits)
    """

    def __init__(
        self,
        *,
        xor_key: Optional[int] = None,
        char_encoding: str = "ASCII",
        error_correction_method: str = "vote",
        **ecc_kwargs,
    ) -> None:
        """
        Args:
            xor_key: Optional integer key 0-255 for XOR obfuscation.
                     ``None`` (default) disables XOR masking.
            char_encoding: Character encoding to use for str↔bits conversion.
                           One of ``"ASCII"``, ``"UNICODE"``, ``"DECIMAL"``.
                           Default ``"ASCII"``.
            error_correction_method: Error correction algorithm.
                                     ``"vote"`` or ``"reed_solomon"``.
            **ecc_kwargs: Keyword arguments forwarded to the ECC constructor
                          (e.g. ``burst_error_tolerance=10``, ``nsym=10``).
        """
        # 1. XOR mask (optional)
        self.xor_mask: Optional[XorMask] = None
        if xor_key is not None:
            self.xor_mask = XorMask(xor_key)

        # 2. Character encoding
        supported = {"ASCII", "UNICODE", "DECIMAL"}
        if char_encoding.upper() not in supported:
            raise ValueError(
                f"Unsupported char_encoding {char_encoding!r}. "
                f"Choose from {sorted(supported)}"
            )
        self.char_encoding = char_encoding.upper()

        # 3. Error correction codec
        self.ecc: ErrorCorrectionCode = ErrorCorrectionFactory.create(
            error_correction_method, **ecc_kwargs,
        )

        logger.info(
            f"SteganoPipeline initialized: "
            f"xor={'ON' if self.xor_mask else 'OFF'}, "
            f"encoding={self.char_encoding}, "
            f"ecc={self.ecc}"
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def encode_message(self, message: str) -> str:
        """
        Full forward transformation: plain text → protected bit string.

        Order: XOR (optional) → str→bits → ECC encode.
        """
        if not message:
            return ""

        # XOR at the string level (before bits) for conceptual clarity,
        # but we implement it on bits so the same key works uniformly.
        # Actually, XOR on raw string characters is less robust; we do it
        # on the bit stream for consistent per-bit obfuscation.
        bits = string2bits(message, code=self.char_encoding)
        logger.info(
            f"[Pipeline] encode: message={len(message)} chars "
            f"→ {len(bits)} raw bits"
        )

        if self.xor_mask is not None:
            bits = self.xor_mask.apply(bits)
            logger.info(f"[Pipeline] XOR applied ({len(bits)} bits)")

        bits = self.ecc.encode(bits)
        logger.info(
            f"[Pipeline] ECC encode → {len(bits)} protected bits"
        )
        return bits

    def decode_message(self, bits: str) -> str:
        """
        Full reverse transformation: protected bit string → plain text.

        Order: ECC decode → bits→str → XOR (optional).
        """
        if not bits:
            return ""

        # Step 1: error correction
        corrected = self.ecc.decode(bits)
        logger.info(
            f"[Pipeline] ECC decode → {len(corrected)} bits"
        )

        # Step 2: undo XOR (if enabled)
        if self.xor_mask is not None:
            corrected = self.xor_mask.apply(corrected)
            logger.info(f"[Pipeline] XOR reversed ({len(corrected)} bits)")

        # Step 3: bits → string
        message = bits2string(corrected, code=self.char_encoding)
        logger.info(
            f"[Pipeline] decode → {len(message)} chars"
        )
        return message

    def __repr__(self) -> str:
        return (
            f"SteganoPipeline("
            f"xor={'ON' if self.xor_mask else 'OFF'}, "
            f"encoding={self.char_encoding!r}, "
            f"ecc={self.ecc})"
        )


# ======================================================================
#  Convenience helper (optional, used by tests / CLI scripts)
# ======================================================================

def build_pipeline_from_options(options) -> SteganoPipeline:
    """
    Build a :class:`SteganoPipeline` from an ``Options`` namespace.

    This is the bridge between CLI arguments and the pipeline object.
    """
    method = getattr(options, "error_correction_method", "vote")
    if method == "vote":
        ecc_kwargs = {
            "burst_error_tolerance": getattr(options, "burst_error_tolerance", 10),
        }
    elif method == "reed_solomon":
        ecc_kwargs = {
            "nsym": getattr(options, "rs_nsym", 10),
        }
    else:
        ecc_kwargs = {}

    return SteganoPipeline(
        xor_key=getattr(options, "xor_key", None),
        char_encoding=getattr(options, "char_encoding", "ASCII"),
        error_correction_method=method,
        **ecc_kwargs,
    )