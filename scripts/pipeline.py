"""
Steganographic pipeline for message transformation.

Chains together all string-level transformations independently from the
VQGAN image encoding/decoding::

    encode:  message (str) → CharToBits → split (header+body) into 10-byte
             blocks → RS-encode each block → strip RS padding header →
             concatenate → XOR
    decode:  bits (str)    → XOR undo → split into fixed 120-bit blocks →
             restore 3-bit pad header → RS-decode each block → trim to
             expected payload → extract length header → take body → BitsToChar

The pipeline is self-contained: the encoder/decoder return raw bits
and never see the internals of the pipeline.

RS:  Each 10-byte payload block is RS-encoded with nsym=5 → 15 bytes.
     The 3-bit padding header that ReedSolomonCorrectionCode.encode adds
     is stripped/restored internally by the pipeline so that each block
     occupies exactly (10 + 5) * 8 = 120 bits in the transmitted stream.
"""

from typing import Optional, Dict, Any, Tuple, List

from scripts.error_correction import ReedSolomonCorrectionCode
from scripts.bit_utils import bits2string, string2bits
from scripts.logger import logger

# Number of bits used for the length header (supports up to 2^13 - 1 = 8191 bits)
HEADER_LENGTH_BITS = 13

# Block-wise Reed-Solomon parameters
RS_BLOCK_BYTES = 10       # payload bytes per RS block
RS_NSYM = 5               # ECC parity symbols per block
RS_PAYLOAD_BITS = RS_BLOCK_BYTES * 8         # 80 bits

# Encoded size of a full block (10 payload bytes + 5 ECC bytes) = 15 bytes = 120 bits
# after stripping the 3-bit RS-internal padding header.
RS_ENCODED_BITS = (RS_BLOCK_BYTES + RS_NSYM) * 8  # 120 bits


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
#  Block-wise RS helpers  (fixed-size blocks, 120 bits each)
# ======================================================================

def _split_into_rs_blocks(bits: str) -> List[str]:
    """
    Split a bit string into blocks of *RS_PAYLOAD_BITS* bits (80 bits = 10 bytes).
    The last block may be smaller.
    """
    blocks = []
    for i in range(0, len(bits), RS_PAYLOAD_BITS):
        blocks.append(bits[i:i + RS_PAYLOAD_BITS])
    return blocks


def encode_block_fixed_size(block_bits: str, ecc: ReedSolomonCorrectionCode) -> str:
    """
    RS-encode a single block and return exactly RS_ENCODED_BITS (120 bits).

    1. Pad block_bits to RS_PAYLOAD_BITS (80 bits).
    2. RS-encode via ecc.encode() → 123 bits (3 pad header + 120 body).
    3. Strip the 3-bit padding header → 120 bits.

    This guarantees that every block (including the last) occupies exactly
    RS_ENCODED_BITS in the transmitted stream, making the decoder's job
    trivial: split by RS_ENCODED_BITS.
    """
    # Pad to 80 bits
    padded = block_bits.ljust(RS_PAYLOAD_BITS, "0")
    # RS-encode: returns 3-bit pad header + encoded body
    raw_encoded = ecc.encode(padded)
    # Strip the 3-bit padding header
    return raw_encoded[3:]


def decode_block_fixed_size(encoded_bits: str, ecc: ReedSolomonCorrectionCode,
                            expected_payload_bits: int) -> str:
    """
    RS-decode a single fixed-size block (RS_ENCODED_BITS = 120 bits).

    Restores the 3-bit padding header before calling ecc.decode(), then
    trims the result to expected_payload_bits.

    Args:
        encoded_bits: Exactly RS_ENCODED_BITS (120) bits.
        ecc: The RS codec instance.
        expected_payload_bits: Expected payload size (80 for full blocks, less for last).

    Returns:
        The decoded payload bits (trimmed to expected_payload_bits).
    """
    if len(encoded_bits) != RS_ENCODED_BITS:
        # Allow the decoder to handle edge cases
        pass
    # Restore the 3-bit padding header (pad_len=0 for full-payload blocks)
    restored = "000" + encoded_bits
    decoded = ecc.decode(restored)
    # Trim to expected payload bits
    if len(decoded) > expected_payload_bits:
        decoded = decoded[:expected_payload_bits]
    return decoded


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
        1. Character encoding       - str → binary via *char_encoding*
        2. Length header            - 13-bit message length in bits (prepended to body)
        3. Block-wise error correction  - Reed-Solomon applied per 10-byte block
        4. XOR bit mask             - obfuscation layer (always active)

    On decode, the order is reversed: XOR undo → split into blocks →
    RS decode each block → extract header → take body → bits→str.

        Usage::

            pipe = SteganoPipeline(
                xor_key=123,
                char_encoding="ASCII",
            )
            protected_bits = pipe.encode_message("Hello")
            recovered_msg = pipe.decode_message(protected_bits)
    """

    def __init__(
        self,
        *,
        xor_key: int,
        char_encoding: str = "ASCII",
    ) -> None:
        """
        Args:
            xor_key: Integer key 0-255 for XOR obfuscation (required).
            char_encoding: Character encoding for str↔bits conversion.
                           One of ``"ASCII"``, ``"UNICODE"``, ``"DECIMAL"``.
                           Default ``"ASCII"``.
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

        # 3. Block-wise RS ECC with fixed parameters
        self.ecc = ReedSolomonCorrectionCode(nsym=RS_NSYM)

        logger.info(
            f"SteganoPipeline initialized: "
            f"xor_key={xor_key}, "
            f"encoding={self.char_encoding}, "
            f"ecc={self.ecc}, "
            f"block={RS_BLOCK_BYTES}B + {RS_NSYM}B RS, "
            f"fixed_block_size={RS_ENCODED_BITS}b"
        )

    # ------------------------------------------------------------------
    #  Public API — standard
    # ------------------------------------------------------------------

    def encode_message(self, message: str) -> str:
        """
        Full forward transformation: plain text → protected bit string.
        """
        if not message:
            return ""
        trace, _ = self._encode_with_trace(message)
        return trace[-1][1] if trace else ""

    def decode_message(self, bits: str) -> str:
        """
        Full reverse transformation: protected bit string → plain text.
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
              - ``"raw_bits"``           - output of str→bits conversion
              - ``"header_bits"``        - raw 13-bit length header
              - ``"body_blocks"``        - list of (payload_bits, encoded_bits) per block
              - ``"full_bits"``          - concatenated all RS-encoded blocks (before XOR)
              - ``"protected_bits"``     - after XOR masking
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
              - ``"xor_output_bits"``    - after XOR undo
              - ``"blocks_info"``        - list of (encoded_size, decoded_size) per block
              - ``"body_decoded"``       - concatenated decoded bits from all blocks
              - ``"header_raw"``         - raw 13-bit header
              - ``"recovered_bits"``     - message bits extracted from body
              - ``"recovered_text"``     - final decoded string
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

        New order: str→bits → split into 10-byte RS blocks →
                   RS-encode each (fixed 120-bit output) → concatenate → XOR.
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

        # Combine header + body bits
        body_input = header_raw + bits
        logger.info(
            f"[Pipeline] payload for block-wise RS: {len(body_input)}b = "
            f"{HEADER_LENGTH_BITS}b header + {len(bits)}b body"
        )

        # Step 3: Split into blocks and RS-encode each
        payload_blocks = _split_into_rs_blocks(body_input)
        encoded_blocks: List[str] = []
        blocks_info = []

        for idx, block in enumerate(payload_blocks):
            encoded = encode_block_fixed_size(block, self.ecc)
            encoded_blocks.append(encoded)
            blocks_info.append((block, encoded))
            logger.info(
                f"[Pipeline] RS block {idx}: {len(block)}b payload → "
                f"{len(encoded)}b encoded"
            )

        trace.append(("body_blocks", str(blocks_info)))  # diagnostic string
        full_bits = "".join(encoded_blocks)
        trace.append(("full_bits", full_bits))
        logger.info(
            f"[Pipeline] block-wise RS: {len(body_input)}b → {len(full_bits)}b "
            f"({len(payload_blocks)} blocks, {RS_ENCODED_BITS}b/block)"
        )

        # Step 4: XOR (always active) — THIS is the final output
        bits = self.xor_mask.apply(full_bits)
        logger.info(
            f"[Pipeline] XOR applied ({len(bits)} bits, "
            f"{len(full_bits)} before XOR) — final output"
        )
        trace.append(("protected_bits", bits))

        return trace, bits

    def _decode_with_trace(self, bits: str) -> Tuple[list, str]:
        """
        Run decode_message step-by-step and record each intermediate.

        New order: XOR undo → split into fixed 120-bit RS blocks →
                   RS-decode each → concatenate decoded blocks →
                   extract header → take body → bits→str.
        """
        trace = []

        # Step 1: XOR undo
        corrected = self.xor_mask.apply(bits)
        logger.info(
            f"[Pipeline] XOR reversed ({len(corrected)} bits)"
        )
        trace.append(("xor_output_bits", corrected))
        trace.append(("xor_sample", corrected[:40]))

        # Step 2: Split into fixed-size blocks and decode each.
        # Each block is exactly RS_ENCODED_BITS (120 bits), except possibly
        # the last block if the stream was truncated.
        blocks_info = []
        decoded_parts: List[str] = []
        offset = 0

        while offset < len(corrected):
            chunk = corrected[offset:offset + RS_ENCODED_BITS]

            if len(chunk) < 8:  # too short to be useful
                logger.warning(
                    f"[Pipeline] RS decode: trailing {len(chunk)} bits ignored"
                )
                break

            if len(chunk) == RS_ENCODED_BITS:
                # Full-size block: expect 80 payload bits
                decoded_block = decode_block_fixed_size(chunk, self.ecc,
                                                        RS_PAYLOAD_BITS)
            else:
                # Last block (truncated): use the RS decode which handles it
                # by trying the actual size
                decoded_block = self.ecc.decode("000" + chunk)
                if decoded_block:
                    decoded_block = decoded_block[:RS_PAYLOAD_BITS]

            if decoded_block:
                blocks_info.append((len(chunk), len(decoded_block)))
                decoded_parts.append(decoded_block)
                offset += len(chunk)
            else:
                logger.warning(
                    f"[Pipeline] Cannot decode RS block at offset {offset}, "
                    f"size {len(chunk)}"
                )
                break

        body_decoded = "".join(decoded_parts)
        trace.append(("body_decoded", body_decoded))
        trace.append(("blocks_info", str(blocks_info)))
        logger.info(
            f"[Pipeline] block-wise RS decode: {len(corrected)}b → "
            f"{len(body_decoded)}b ({len(blocks_info)} blocks)"
        )

        # Step 3: Extract header from first HEADER_LENGTH_BITS bits of decoded body
        message_bit_length = _decode_length_from_body(body_decoded)
        trace.append(("header_raw", format(message_bit_length, f"0{HEADER_LENGTH_BITS}b")))
        logger.info(
            f"[Pipeline] header decoded from RS body: {message_bit_length} bits"
        )

        if message_bit_length == 0:
            logger.warning("[Pipeline] header decoded to 0 — aborting decode")
            return trace, ""

        # Step 4: Take only the expected message bits from decoded body
        recovered_bits = body_decoded[HEADER_LENGTH_BITS:HEADER_LENGTH_BITS + message_bit_length]
        trace.append(("recovered_bits", recovered_bits))

        # Step 5: bits → string
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
            f"ecc=RS(nsym={RS_NSYM},block={RS_BLOCK_BYTES}B), "
            f"fixed_block={RS_ENCODED_BITS}b)"
        )


# ======================================================================
#  Convenience helper (optional, used by tests / CLI scripts)
# ======================================================================

def build_pipeline_from_options(options) -> SteganoPipeline:
    """
    Build a :class:`SteganoPipeline` from an ``Options`` namespace.

    This is the bridge between CLI arguments and the pipeline object.
    Only uses xor_key and char_encoding from options (ECC is fixed).
    """
    return SteganoPipeline(
        xor_key=getattr(options, "xor_key", 123),
        char_encoding=getattr(options, "char_encoding", "ASCII"),
    )