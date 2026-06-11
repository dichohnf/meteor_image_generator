"""
Lightweight bit-string conversion utilities (no torch dependency).

These functions convert between Python strings and binary strings.
They are factored out of ``utils.py`` to allow the pipeline module
to import them without pulling in torch.
"""

from typing import List


def string2bits(text: str, code: str = "ASCII") -> str:
    """
    Convert a text string to a binary string using the specified encoding.

    Supported encodings:
        - ASCII:   8 bits per character (only ASCII-range chars allowed)
        - UNICODE: 28 bits per character (U+0000 – U+9FFF, i.e. CJK + BMP)
        - DECIMAL: variable-length encoding using decimal digit pairs

    Args:
        text: Input string to encode.
        code: Encoding scheme — ``"ASCII"``, ``"UNICODE"``, or ``"DECIMAL"``.

    Returns:
        Binary string (e.g. ``"0100100001101001"``).
    """
    code = code.upper()
    if code == "ASCII":
        return "".join(format(ord(c), "08b") for c in text)
    elif code == "UNICODE":
        return "".join(format(ord(c), "028b") for c in text)
    elif code == "DECIMAL":
        # Encode each character as two decimal digits (00–99)
        return "".join(format(ord(c), "07b") for c in text)
    else:
        raise ValueError(f"Unsupported code: {code}")


def bits2string(binary_str: str, code: str = "ASCII") -> str:
    """
    Convert a binary string back to text using the specified encoding.

    Args:
        binary_str: Binary string to decode.
        code: Encoding scheme — ``"ASCII"``, ``"UNICODE"``, or ``"DECIMAL"``.

    Returns:
        Decoded text string.
    """
    code = code.upper()
    if code == "ASCII":
        chars = []
        for i in range(0, len(binary_str) - len(binary_str) % 8, 8):
            byte = binary_str[i:i + 8]
            chars.append(chr(int(byte, 2)))
        return "".join(chars)
    elif code == "UNICODE":
        chars = []
        for i in range(0, len(binary_str) - len(binary_str) % 28, 28):
            code_point = binary_str[i:i + 28]
            chars.append(chr(int(code_point, 2)))
        return "".join(chars)
    elif code == "DECIMAL":
        chars = []
        for i in range(0, len(binary_str) - len(binary_str) % 7, 7):
            bits = binary_str[i:i + 7]
            val = int(bits, 2)
            if val != 0:  # skip null bytes
                chars.append(chr(val))
        return "".join(chars)
    else:
        raise ValueError(f"Unsupported code: {code}")