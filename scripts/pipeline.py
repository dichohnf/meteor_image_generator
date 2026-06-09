"""
Steganographic pipeline for message transformation.

Chains together all string-level transformations independently from the
VQGAN image encoding/decoding::

    encode:  message (str) → XOR + CharToBits + ErrorCorrection → bits (str)
    decode:  bits (str)    → ErrorCorrection + BitsToChar + XOR → message (str)

The pipeline is self-contained: the encoder/decoder receive/return raw bits
and never see the internals of the pipeline.
"""

from typing import Optional, Dict, Any, Tuple

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
    #  Public API — standard
    # ------------------------------------------------------------------

    def encode_message(self, message: str) -> str:
        """
        Full forward transformation: plain text → protected bit string.

        Order: str→bits → XOR (optional) → ECC encode.
        """
        if not message:
            return ""
        trace, _ = self._encode_with_trace(message)
        return trace[-1][1] if trace else ""

    def decode_message(self, bits: str) -> str:
        """
        Full reverse transformation: protected bit string → plain text.

        Order: ECC decode → XOR (optional) → bits→str.
        """
        if not bits:
            return ""
        _, message = self._decode_with_trace(bits)
        return message

    # ------------------------------------------------------------------
    #  Public API — with trace (for statistics)
    # ------------------------------------------------------------------

    def encode_message_with_trace(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Like :meth:`encode_message` but also returns a trace dict with
        the intermediate bit strings at each pipeline step.

        Returns:
            Tuple of (protected_bits, trace_dict).
            The trace dict has keys:
              - ``"raw_bits"``       – output of str→bits conversion
              - ``"xor_bits"``       – after XOR masking (omitted if XOR is off)
              - ``"protected_bits"`` – after ECC encode (final output)
        """
        if not message:
            return "", {}
        trace, _ = self._encode_with_trace(message)
        protected = trace[-1][1] if trace else ""
        trace_dict = {label: bits for label, bits in trace}
        return protected, trace_dict

    def decode_message_with_trace(self, bits: str) -> Tuple[str, Dict[str, Any]]:
        """
        Like :meth:`decode_message` but also returns a trace dict with
        the intermediate bit strings at each pipeline step.

        Returns:
            Tuple of (recovered_message, trace_dict).
            The trace dict has keys:
              - ``"ecc_output_bits"``  – after ECC decode
              - ``"xor_output_bits"``  – after XOR undo (omitted if XOR is off)
              - ``"recovered_bits"``   – bits fed to bits→str conversion
              - ``"recovered_text"``   – final decoded string
        """
        if not bits:
            return "", {}
        trace, message = self._decode_with_trace(bits)
        trace_dict = {}
        for label, bits_val in trace:
            trace_dict[label] = bits_val
        trace_dict["recovered_text"] = message
        return message, trace_dict

    # ------------------------------------------------------------------
    #  Internal trace helpers
    # ------------------------------------------------------------------

    def _encode_with_trace(self, message: str) -> Tuple[list, str]:
        """
        Run encode_message step-by-step and record each intermediate.

        Returns:
            Tuple of (trace_list, final_bits).
            trace_list is a list of ``(step_label, bit_string)`` tuples.
        """
        trace = []

        # Step 1: str → bits
        bits = string2bits(message, code=self.char_encoding)
        logger.info(
            f"[Pipeline] encode: message={len(message)} chars "
            f"→ {len(bits)} raw bits"
        )
        trace.append(("raw_bits", bits))

        # Step 2: XOR (optional)
        if self.xor_mask is not None:
            bits = self.xor_mask.apply(bits)
            logger.info(f"[Pipeline] XOR applied ({len(bits)} bits)")
            trace.append(("xor_bits", bits))

        # Step 3: ECC encode
        bits = self.ecc.encode(bits)
        logger.info(
            f"[Pipeline] ECC encode → {len(bits)} protected bits"
        )
        trace.append(("protected_bits", bits))

        return trace, bits

    def _decode_with_trace(self, bits: str) -> Tuple[list, str]:
        """
        Run decode_message step-by-step and record each intermediate.

        Returns:
            Tuple of (trace_list, recovered_message).
        """
        trace = []

        # Step 1: ECC decode
        corrected = self.ecc.decode(bits)
        logger.info(
            f"[Pipeline] ECC decode → {len(corrected)} bits"
        )
        trace.append(("ecc_output_bits", corrected))

        # Step 2: XOR undo (optional)
        if self.xor_mask is not None:
            corrected = self.xor_mask.apply(corrected)
            logger.info(f"[Pipeline] XOR reversed ({len(corrected)} bits)")
            trace.append(("xor_output_bits", corrected))

        # Step 3: bits → string
        trace.append(("recovered_bits", corrected))
        message = bits2string(corrected, code=self.char_encoding)
        logger.info(
            f"[Pipeline] decode → {len(message)} chars"
        )

        return trace, message

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