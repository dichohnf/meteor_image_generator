from abc import abstractmethod
from typing import Any, Generic, TypeVar
from reedsolo import RSCodec

T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')

class Transformer(Generic[T,S]):
    """Abstract base class for transformers that convert data from type T to type S and vice versa."""
    
    @abstractmethod
    def apply_transformation(self, input_data : T) -> S:
        raise NotImplementedError("Subclasses must implement this method.")
    @abstractmethod
    def reverse_transformation(self, transformed_data : S) -> T:
        raise NotImplementedError("Subclasses must implement this method.")

class XorBytesTransformer(Transformer[bytes, bytes]):
  """
Applies a bitwise XOR transformation to a binary string using a specified key. The transformation takes an input binary string, divides it into 8-bit segments (bytes), and applies the XOR operation between each byte and the corresponding byte of the key (repeated as necessary). The result is a transformed binary string. The reverse transformation applies the same XOR operation to the transformed string to recover the original input string, since XOR is its own inverse.
  """
  def __init__(self, key : int):
    self.key = key
    self.key_string = format(key, '08b')
    
  def apply_transformation(self, input_string : str) -> str:    
    if len(input_string) % 8 != 0:
        raise ValueError("input_string length must be a multiple of 8.")
    if not all(c in ['0', '1'] for c in input_string):
        raise ValueError("input_string must be a binary string containing only '0' and '1'.")
    
    transformed = []
    for i in range(0, len(input_string), 8):
        byte_str = input_string[i:i+8]
        xor_result = ''.join(str(int(byte_str[j]) ^ int(self.key_string[j])) for j in range(8))
        transformed.append(xor_result)
    
    return ''.join(transformed)
  
  def reverse_transformation(self, transformed_string : str) -> str:
    return self.apply_transformation(transformed_string)
  
class HammingTransformer(Transformer[bytes, bytes]):
    """
    Applies a Hamming(7,4) error correction code transformation to a binary string. The transformation takes an input binary string, divides it into 4-bit segments, and encodes each segment into a 7-bit codeword by adding 3 parity bits. The resulting transformed string is longer than the original input string due to the added parity bits. The reverse transformation takes the transformed string, divides it into 7-bit codewords, checks for and corrects any single-bit errors using the parity bits, and extracts the original 4-bit segments to recover the original binary string.
    """
    def apply_transformation(self, input_string):
        padded_input = input_string + "1"
        while len(padded_input) % 4 != 0:
            padded_input += "0"
        
        res = []
        for i in range(0, len(padded_input), 4):
            d = [int(b) for b in padded_input[i:i+4]]
            
            p1 = d[0] ^ d[1] ^ d[3]
            p2 = d[0] ^ d[2] ^ d[3]
            p3 = d[1] ^ d[2] ^ d[3]
            
            res.extend([p1, p2, d[0], p3, d[1], d[2], d[3]])
            
        return "".join(map(str, res))

    def reverse_transformation(self, transformed_string):
        
        extracted_bits = []
        for i in range(0, len(transformed_string), 7):
            if i + 7 > len(transformed_string):
                return "".join(map(str, extracted_bits))
            y = [int(b) for b in transformed_string[i:i+7]]
            
            s1 = y[0] ^ y[2] ^ y[4] ^ y[6]
            s2 = y[1] ^ y[2] ^ y[5] ^ y[6]
            s3 = y[3] ^ y[4] ^ y[5] ^ y[6]
            
            error_pos = s1 * 1 + s2 * 2 + s3 * 4
            if error_pos != 0:
                y[error_pos - 1] = 1 - y[error_pos - 1]
            
            extracted_bits.extend([y[2], y[4], y[5], y[6]])
        
        full_decoded_str = "".join(map(str, extracted_bits))

        last_one_index = full_decoded_str.rfind('1')
        return full_decoded_str[:last_one_index]
  
class CharWithBaseToBinaryTransformer(Transformer[str, bytes]):
    """Transforms a string into its binary representation based on a specified character encoding (base). The transformation converts each character in the input string into its corresponding byte representation according to the specified encoding (e.g., UTF-8, UTF-16, UTF-32) and then concatenates the binary representations of these bytes into a single binary string. The reverse transformation takes a binary string, splits it into groups of 8 bits (1 byte), converts each group back into its byte representation, and then decodes the resulting byte array back into the original string using the same character encoding.
    """
    def __init__(self, base: str):
        if base not in ['utf-8', 'utf-16', 'utf-32']:
            raise ValueError("Base must be one of 'utf-8', 'utf-16', or 'utf-32'.")
        self.base = base

    def apply_transformation(self, input_string : str) -> bytes:
        byte_data = input_string.encode(self.base)
        return ''.join(format(byte, '08b') for byte in byte_data)

    def reverse_transformation(self, transformed_string : bytes) -> str:
        if len(transformed_string) % 8 != 0:
            raise ValueError("transformed_string length must be a multiple of 8.")
        byte_array = bytearray(int(transformed_string[i:i+8], 2) for i in range(0, len(transformed_string), 8))
        return byte_array.decode(self.base)
  
class Pipeline(Transformer[T, S]):
    """A class to construct a pipeline of trnsformers to manage the transformations of some kind of object."""
    def __init__(self, transformers: list[Transformer[Any, Any]] = []):
        self.transformers = transformers

    @classmethod
    def start(cls, first_transformer: Transformer[T, S]) -> 'Pipeline[T, S]':
        """Initialize a pipeline with the first transformer, setting the initial input and output types."""
        return cls([first_transformer])

    def add(self, next_transformer: Transformer[S, U]) -> 'Pipeline[T, U]':
        """Add a new transformer to the pipeline, ensuring that the output type of the last transformer matches the input type of the new transformer. This method returns a new Pipeline instance with the added transformer, allowing for method chaining.
        """
        return Pipeline(self.transformers + [next_transformer])

    def apply_transformation(self, input_data: T) -> S:
        result: Any = input_data
        for transformer in self.transformers:
            result = transformer.apply_transformation(result)
        return result
        
    def reverse_transformation(self, transformed_data: S) -> T:
        result: Any = transformed_data
        for transformer in reversed(self.transformers):
            result = transformer.reverse_transformation(result)
        return result
    
        
class ReedSolomonTransformer(Transformer[bytes, str]):
    """
    Applies Reed-Solomon error correction code transformation to a binary string. The transformation takes an input binary string, encodes it using the Reed-Solomon algorithm to add error correction capabilities, and produces a transformed string that includes both the original data and the error correction code. The reverse transformation takes the transformed string, decodes it using the Reed-Solomon algorithm to correct any errors that may have occurred during transmission, and retrieves the original binary string.
    """
    def __init__(self, nsym: int):
        self.nsym = nsym
        self.rsc = RSCodec(nsym)

    def apply_transformation(self, input_string: bytes) -> str:
        encoded_data = self.rsc.encode(input_string)
        return ''.join(format(byte, '08b') for byte in encoded_data)

    def reverse_transformation(self, transformed_string: str) -> bytes:
        if len(transformed_string) % 8 != 0:
            raise ValueError("transformed_string length must be a multiple of 8.")
        byte_array = bytearray(int(transformed_string[i:i+8], 2) for i in range(0, len(transformed_string), 8))
        rmes  = self.rsc.decode(byte_array)[0]
        return rmes