"""
Steganographic pipeline for message transformation.

Chains together all string-level transformations independently from the
VQGAN image encoding/decoding::

    encode:  message (str) → CharToBits → LengthHeader(Vote) → XOR → ECC(RS) → bits (str)
    decode:  bits (str)    → ECC(RS) → XOR → StripHeader(Vote) → BitsToChar → message (str)

The pipeline is self-contained: the encoder/decoder receive/return raw bits
and never see the internals of the pipeline.
"""

from typing import Optional, Dict, Any, Tuple

from scripts.error_correction import ErrorCorrectionFactory, ErrorCorrectionCode, VoteCorrectionCode
from scripts.logger import logger
from scripts.utils import bits2string, string2bits

# Number of bits used for the length header (supports up to 2^13 - 1 = 8191 bits)
HEADER_LENGTH_BITS = 13


# ======================================================================
#  Length header helpers (vote-protected)
# ======================================================================

def _encode_length_header(num_bits: int, burst_error_tolerance: int) -> str:
    """
    Encode the message length (in bits) as a vote-protected header.

    The raw header is *HEADER_LENGTH_BITS* (13) bits wide, supporting
    message lengths up to 2^13 - 1 = 8191 bits.  It is then protected
    with :class:`VoteCorrectionCode` using the given tolerance.

    Returns:
        Vote-protected bit string for the header.
    """
    header_raw = format(num_bits, f"0{HEADER_LENGTH_BITS}b")
    vote = VoteCorrectionCode(burst_error_tolerance=burst_error_tolerance)
    return vote.encode(header_raw)


def _decode_length_header(protected_header: str, burst_error_tolerance: int) -> int:
    """
    Decode a vote-protected length header and return the message length in bits.

    Args:
        protected_header: The vote-protected bit string for the header.
        burst_error_tolerance: The burst error tolerance used during encoding.

    Returns:
        The message length in bits.
    """
    vote = VoteCorrectionCode(burst_error_tolerance=burst_error_tolerance)
    header_raw = vote.decode(protected_header)
    # In case decoding returns more bits than expected, take only the first 13
    return int(header_raw[:HEADER_LENGTH_BITS], 2)


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
        2. Length header            – vote-protected 13-bit header with message
                                      length in bits (uses VoteCorrectionCode
                                      with *header_burst_error_tolerance*)
        3. XOR bit mask             – obfuscation layer (always active)
        4. Error correction coding  – Reed-Solomon redundancy (body + header)

    Usage::

        pipe = SteganoPipeline(
            xor_key=123,
            char_encoding="ASCII",
            rs_nsym=10,
            header_burst_error_tolerance=10,
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
        header_burst_error_tolerance: int = 10,
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
            header_burst_error_tolerance: Burst error tolerance for the
                                          vote-protected length header.
                                          Default 10 (→ 21 repetitions).
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

        # 3. Header vote protection tolerance
        if header_burst_error_tolerance < 1:
            raise ValueError(
                f"header_burst_error_tolerance must be >= 1, "
                f"got {header_burst_error_tolerance}"
            )
        self.header_burst_error_tolerance = header_burst_error_tolerance

        # 4. Body ECC
        self.ecc: ErrorCorrectionCode = ErrorCorrectionFactory.create(
            error_correction_method, **ecc_kwargs,
        )

        logger.info(
            f"SteganoPipeline initialized: "
            f"xor_key={xor_key}, "
            f"encoding={self.char_encoding}, "
            f"ecc={self.ecc}, "
            f"header_tolerance={header_burst_error_tolerance}"
        )

    # ------------------------------------------------------------------
    #  Public API — standard
    # ------------------------------------------------------------------

    def encode_message(self, message: str) -> str:
        """
        Full forward transformation: plain text → protected bit string.

        Order: str→bits → length-header(Vote) → XOR → ECC encode.
        """
        if not message:
            return ""
        trace, _ = self._encode_with_trace(message)
        return trace[-1][1] if trace else ""

    def decode_message(self, bits: str) -> str:
        """
        Full reverse transformation: protected bit string → plain text.

        Order: ECC decode → XOR → strip header(Vote) → bits→str.
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
              - ``"raw_bits"``         – output of str→bits conversion
              - ``"header_bits"``      – raw 13-bit length header
              - ``"header_protected"`` – vote-protected header
              - ``"full_bits"``        – header + message bits (before XOR)
              - ``"xor_bits"``         – after XOR masking
              - ``"protected_bits"``   – after ECC encode (final output)
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
              - ``"ecc_output_bits"``    – after ECC decode
              - ``"xor_output_bits"``    – after XOR undo
              - ``"full_bits"``          – header + message bits (after XOR)
              - ``"header_protected"``   – vote-protected header extracted
              - ``"header_raw"``         – raw 13-bit header after vote decode
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

        # Step 2: Build vote-protected length header
        header_raw = format(len(bits), f"0{HEADER_LENGTH_BITS}b")
        trace.append(("header_bits", header_raw))

        header_protected = _encode_length_header(
            len(bits), self.header_burst_error_tolerance
        )
        logger.info(
            f"[Pipeline] header: {HEADER_LENGTH_BITS}b raw → "
            f"{len(header_protected)}b vote-protected "
            f"(tolerance={self.header_burst_error_tolerance})"
        )
        trace.append(("header_protected", header_protected))

        # Step 3: Concatenate header + message bits
        full_bits = header_protected + bits
        trace.append(("full_bits", full_bits))

        # Step 4: XOR (always active)
        bits = self.xor_mask.apply(full_bits)
        logger.info(f"[Pipeline] XOR applied ({len(bits)} bits)")
        trace.append(("xor_bits", bits))

        # Step 5: ECC encode
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

        # Step 2: XOR undo (always active)
        corrected = self.xor_mask.apply(corrected)
        logger.info(f"[Pipeline] XOR reversed ({len(corrected)} bits)")
        trace.append(("xor_output_bits", corrected))

        # Step 3: Extract and decode the vote-protected header
        header_len = HEADER_LENGTH_BITS * (
            2 * self.header_burst_error_tolerance + 1
        )
        if len(corrected) < header_len:
            logger.warning(
                f"[Pipeline] decoded bits too short for header "
                f"({len(corrected)} < {header_len})"
            )
            return trace, ""

        header_protected = corrected[:header_len]
        trace.append(("header_protected", header_protected))

        # Decode the vote-protected header
        header_raw_decoded = _decode_length_header(
            header_protected, self.header_burst_error_tolerance
        )
        trace.append(("header_raw", format(header_raw_decoded, f"0{HEADER_LENGTH_BITS}b")))

        message_bit_length = header_raw_decoded

        # Step 4: Extract message bits
        body_start = header_len
        body_end = body_start + message_bit_length
        recovered_bits = corrected[body_start:body_end]
        trace.append(("recovered_bits", recovered_bits))

        # Step 5: bits → string
        message = bits2string(recovered_bits, code=self.char_encoding)
        logger.info(
            f"[Pipeline] decode → {len(message)} chars "
            f"(header said {message_bit_length}b)"
        )

        if message_bit_length > 0 and len(recovered_bits) < message_bit_length:
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
            f"header_tolerance={self.header_burst_error_tolerance})"
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
            "nsym": getattr(options, "rs_nsym", 10),
        }
    else:
        ecc_kwargs = {}

    return SteganoPipeline(
        xor_key=getattr(options, "xor_key", 123),
        char_encoding=getattr(options, "char_encoding", "ASCII"),
        error_correction_method=method,
        header_burst_error_tolerance=getattr(options, "header_burst_error_tolerance", 10),
        **ecc_kwargs,
    )