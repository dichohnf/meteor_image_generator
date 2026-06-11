"""
Steganographic pipeline for message transformation.

Chains together all string-level transformations independently from the
VQGAN image encoding/decoding::

    encode:  message (str) → CharToBits → ECC(RS) on body → LengthHeader(Vote)
             → header_vote + body_RS → XOR → bits (str)
    decode:  bits (str)    → XOR undo → StripHeader(Vote) → body_RS → ECC(RS) on body
             → BitsToChar → message (str)

The pipeline is self-contained: the encoder/decoder receive/return raw bits
and never see the internals of the pipeline.
"""

from typing import Optional, Dict, Any, Tuple

from scripts.error_correction import ErrorCorrectionFactory, ErrorCorrectionCode
from scripts.bit_utils import bits2string, string2bits
from scripts.logger import logger

# Number of bits used for the length header (supports up to 2^13 - 1 = 8191 bits)
HEADER_LENGTH_BITS = 13


# ======================================================================
#  Length header helper (plain, embedded in RS body)
# ======================================================================

def _decode_length_from_body(body_decoded: str) -> int:
    """
    Extract the message length from the first *HEADER_LENGTH_BITS* bits
    of an RS-decoded body.

    Returns:
        The message length in bits, or 0 if the length is invalid.
    """
    if len(body_decoded) < HEADER_LENGTH_BITS:
        return 0
    num_bits = int(body_decoded[:HEADER_LENGTH_BITS], 2)
    max_bits = (1 << HEADER_LENGTH_BITS) - 1
    if num_bits <= 0 or num_bits > max_bits:
        logger.warning(
            f"[Pipeline] _decode_length_from_body: invalid decoded length "
            f"{num_bits} (max={max_bits})"
        )
        return 0
    return num_bits


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
        1. Character encoding       – str → binary via *char_encoding*
        2. Length header            – 13-bit message length in bits (prepended to body)
        3. Error correction coding  – Reed-Solomon applied over (header + body)
        4. XOR bit mask             – obfuscation layer (always active)

    On decode, the order is reversed: XOR undo → strip header → RS decode(body).

        Usage::

            pipe = SteganoPipeline(
                xor_key=123,
                char_encoding="ASCII",
                rs_nsym=30,
            )
            protected_bits = pipe.encode_message("Hello")
            recovered_msg = pipe.decode_message(protected_bits)
    """

    def __init__(
        self,
        *,
        xor_key: int,
        char_encoding: str = "ASCII",
        error_correction_method: str = "reed_solomon",
        **ecc_kwargs,
    ) -> None:
        """
        Args:
            xor_key: Integer key 0-255 for XOR obfuscation (required).
            char_encoding: Character encoding for str↔bits conversion.
                           One of ``"ASCII"``, ``"UNICODE"``, ``"DECIMAL"``.
                           Default ``"ASCII"``.
            error_correction_method: Error correction algorithm for the body.
                                     Default ``"reed_solomon"``.
            (The 13-bit length header is embedded inside the RS body;
             no separate vote protection is needed.)
            **ecc_kwargs: Keyword arguments forwarded to the ECC constructor
                          (e.g. ``nsym=10`` for Reed-Solomon).
        """
        # 1. Character encoding
        supported = {"ASCII", "UNICODE", "DECIMAL"}
        if char_encoding.upper() not in supported:
            raise ValueError(
                f"Unsupported char_encoding {char_encoding!r}. "
                f"Choose from {sorted(supported)}"
            )
        self.char_encoding = char_encoding.upper()

        # 2. XOR mask (always active)
        self.xor_mask = XorMask(xor_key)

        # 3. Body ECC
        self.ecc: ErrorCorrectionCode = ErrorCorrectionFactory.create(
            error_correction_method, **ecc_kwargs,
        )

        logger.info(
            f"SteganoPipeline initialized: "
            f"xor_key={xor_key}, "
            f"encoding={self.char_encoding}, "
            f"ecc={self.ecc}"
        )

    # ------------------------------------------------------------------
    #  Public API — standard
    # ------------------------------------------------------------------

    def encode_message(self, message: str) -> str:
        """
        Full forward transformation: plain text → protected bit string.

        New order: str→bits → RS encode(body) → vote-header → XOR.
        """
        if not message:
            return ""
        trace, _ = self._encode_with_trace(message)
        return trace[-1][1] if trace else ""

    def decode_message(self, bits: str) -> str:
        """
        Full reverse transformation: protected bit string → plain text.

        New order: XOR undo → strip header → RS decode(body) → bits→str.
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
              - ``"raw_bits"``           – output of str→bits conversion
              - ``"body_ecc_bits"``      – after RS encode on the body only
              - ``"header_bits"``        – raw 13-bit length header
              - ``"header_protected"``   – vote-protected header
              - ``"full_bits"``          – header + body_RS bits (before XOR)
              - ``"xor_bits"``           – after XOR masking
              - ``"protected_bits"``     – final output (same as xor_bits, no outer ECC)
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
              - ``"xor_output_bits"``    – after XOR undo
              - ``"xor_sample"``         – sample of first 40 bits after XOR
              - ``"header_protected"``   – vote-protected header extracted
              - ``"header_raw"``         – raw 13-bit header after vote decode
              - ``"body_ecc_bits"``      – body bits extracted (RS-encoded body)
              - ``"body_ecc_len"``       – length of body_ecc_bits
              - ``"ecc_output_bits"``    – after RS decode on the body
              - ``"recovered_bits"``     – bits fed to bits→str conversion
              - ``"recovered_text"``     – final decoded string
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

        # Step 2: Prepend raw length header to payload
        header_raw = format(len(bits), f"0{HEADER_LENGTH_BITS}b")
        trace.append(("header_bits", header_raw))

        body_input = header_raw + bits
        logger.info(
            f"[Pipeline] payload for RS: {len(body_input)}b = "
            f"{HEADER_LENGTH_BITS}b header + {len(bits)}b body"
        )

        # Step 3: RS-encode (header+body) together
        bits_ecc_body = self.ecc.encode(body_input)
        logger.info(
            f"[Pipeline] RS encode on (header+body): "
            f"{len(body_input)}b → {len(bits_ecc_body)}b"
        )
        trace.append(("body_ecc_bits", bits_ecc_body))
        trace.append(("full_bits", bits_ecc_body))

        # Step 4: XOR (always active) — THIS is the final output
        bits = self.xor_mask.apply(bits_ecc_body)
        logger.info(
            f"[Pipeline] XOR applied ({len(bits)} bits) — final output"
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

        # Step 1: XOR undo
        corrected = self.xor_mask.apply(bits)
        logger.info(
            f"[Pipeline] XOR reversed ({len(corrected)} bits)"
        )
        trace.append(("xor_output_bits", corrected))
        trace.append(("xor_sample", corrected[:40]))

        # Step 2: The entire bit stream after XOR undo is the RS-encoded body
        # (header is embedded inside the RS body)
        body_ecc_bits = corrected
        trace.append(("body_ecc_bits", body_ecc_bits))
        trace.append(("body_ecc_len", len(body_ecc_bits)))
        logger.info(
            f"[Pipeline] extracted body ECC: {len(body_ecc_bits)} bits"
        )

        # Step 3: RS decode on the body
        body_decoded = self.ecc.decode(body_ecc_bits)
        logger.info(
            f"[Pipeline] ECC decode on body: {len(body_ecc_bits)}b → {len(body_decoded)}b"
        )
        trace.append(("ecc_output_bits", body_decoded))

        # Step 4: Extract header from first HEADER_LENGTH_BITS bits of decoded body
        message_bit_length = _decode_length_from_body(body_decoded)
        trace.append(("header_raw", format(message_bit_length, f"0{HEADER_LENGTH_BITS}b")))
        logger.info(
            f"[Pipeline] header decoded from RS body: {message_bit_length} bits"
        )

        if message_bit_length == 0:
            logger.warning("[Pipeline] header decoded to 0 — aborting decode")
            return trace, ""

        # Step 5: Take only the expected message bits from decoded body
        recovered_bits = body_decoded[HEADER_LENGTH_BITS:HEADER_LENGTH_BITS + message_bit_length]
        trace.append(("recovered_bits", recovered_bits))

        # Step 6: bits → string
        message = bits2string(recovered_bits, code=self.char_encoding)
        logger.info(
            f"[Pipeline] decode → {len(message)} chars "
            f"(header said {message_bit_length}b)"
        )

        if len(recovered_bits) < message_bit_length:
            logger.warning(
                f"[Pipeline] decoded body too short: "
                f"have {len(recovered_bits)}b, expected {message_bit_length}b"
            )

        return trace, message

    def __repr__(self) -> str:
        return (
            f"SteganoPipeline("
            f"xor_key={self.xor_mask.key}, "
            f"encoding={self.char_encoding!r}, "
            f"ecc={self.ecc}, "
            f"header_tolerance=<embedded_in_rs>)"
        )


# ======================================================================
#  Convenience helper (optional, used by tests / CLI scripts)
# ======================================================================

def build_pipeline_from_options(options) -> SteganoPipeline:
    """
    Build a :class:`SteganoPipeline` from an ``Options`` namespace.

    This is the bridge between CLI arguments and the pipeline object.
    """
    method = getattr(options, "error_correction_method", "reed_solomon")
    if method == "vote":
        ecc_kwargs = {
            "burst_error_tolerance": getattr(options, "burst_error_tolerance", 10),
        }
    elif method == "reed_solomon":
        ecc_kwargs = {
            "nsym": getattr(options, "rs_nsym", 30),
        }
    else:
        ecc_kwargs = {}

    return SteganoPipeline(
        xor_key=getattr(options, "xor_key", 123),
        char_encoding=getattr(options, "char_encoding", "ASCII"),
        error_correction_method=method,
        **ecc_kwargs,
    )