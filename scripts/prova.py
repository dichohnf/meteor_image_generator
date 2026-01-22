import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from typing import Tuple, List, Optional

from tqdm import tqdm

from scripts.encode_message import get_vqgan_sflckr

# Constants
PATCH_SIZE = 16
DEFAULT_PRECISION_BITS = 10
DEFAULT_CODEBOOK_SIZE = 1024
DEFAULT_CONTEXT_ROWS = 512
LN2 = 0.69315  # ln(2) for entropy calculation


@torch.no_grad()
def set_context(model, dsets, num_rows: int) -> torch.Tensor:
    """
    Prepares and returns a context image from the given dataset.

    Extracts a random image from the dataset, processes it through the model,
    and crops it to align with 16-pixel patch boundaries.

    Args:
        model: The model instance with a `get_input` method.
        dsets: Data structure containing datasets.
        num_rows: Number of pixel rows to retain from the top of the image.

    Returns:
        torch.Tensor: A 3D tensor (channels, height, width) representing
            the prepared context image, cropped to be divisible by 16 pixels.

    Raises:
        ValueError: If num_rows is None.
    """
    if num_rows is None:
        raise ValueError("num_rows must be not None")

    # Select dataset
    if len(dsets.datasets) > 1:
        split = sorted(dsets.datasets.keys())[0]
        dset = dsets.datasets[split]
    else:
        dset = next(iter(dsets.datasets.values()))

    # Get random image
    context_idx = torch.randint(len(dset), size=(1,))
    example = default_collate([dset[context_idx.item()]])
    x = model.get_input("image", example).to(model.device).squeeze()

    # Validate and adjust num_rows
    if num_rows > x.shape[1]:
        num_rows = x.shape[1]
        print(f"WARNING: num_rows clamped to image height: {num_rows}")

    # Crop to align with PATCH_SIZE boundaries
    height_crop = x.shape[1] - ((x.shape[1] - num_rows) % PATCH_SIZE)
    width_crop = x.shape[2] - (x.shape[2] % PATCH_SIZE)
    x = x[:, :height_crop, :width_crop]

    return x


def string2bits(message: str, code: str = 'ASCII') -> str:
    """
    Converts a string message into its binary representation.

    Args:
        message: The input string to be converted.
        code: The encoding standard ('ASCII', 'UNICODE', or 'DECIMAL').

    Returns:
        str: A binary string representing the entire input message.

    Raises:
        ValueError: If encoding is unsupported or message is invalid.
    """
    code = code.upper()
    bits = ""

    encoding_config = {
        'ASCII': 8,
        'UNICODE': 21,
        'DECIMAL': 4
    }

    if code not in encoding_config:
        raise ValueError(f"Unsupported encoding: {code}. Use 'ASCII', 'UNICODE', or 'DECIMAL'.")

    if code == 'DECIMAL' and not message.isdigit():
        raise ValueError("Message must contain only decimal digits for 'DECIMAL' encoding")

    bit_width = encoding_config[code]

    for char in message:
        char_code = int(char) if code == 'DECIMAL' else ord(char)
        bits += bin(char_code)[2:].zfill(bit_width)

    return bits


def bits2int(bits: str, *, reversed: bool = False) -> int:
    """
    Converts a binary string to its integer representation.

    Args:
        bits: A string of binary digits ("0" and "1").
        reversed: If True, reverses the bit string before conversion.

    Returns:
        int: The integer representation of the binary string.
    """
    return int(bits[::-1] if reversed else bits, 2)


def int2bits(value: int, num_bits: int) -> List[int]:
    """
    Converts an integer to a reversed list of binary digits.

    Args:
        value: The integer value to convert.
        num_bits: The number of bits for the binary representation.

    Returns:
        List[int]: Binary digits in reversed order (LSB first).
    """
    if num_bits == 0:
        return []

    binary_str = format(value, f'0{num_bits}b')
    return [int(bit) for bit in reversed(binary_str)]


def local_indexes(index: int, max_len: int) -> Tuple[int, int, int]:
    """
    Calculates local position within a 16-element sliding window.

    Args:
        index: The current absolute index position.
        max_len: The total length of the range.

    Returns:
        Tuple[int, int, int]: (local_index, window_start, window_end)
    """
    if index <= 8:
        local_ind = index
    elif max_len - index < 8:
        local_ind = PATCH_SIZE - (max_len - index)
    else:
        local_ind = 8

    idx_start = index - local_ind
    idx_end = idx_start + PATCH_SIZE

    return local_ind, idx_start, idx_end


def entropy(q: Tensor, logq: Tensor) -> float:
    """
    Calculates Shannon entropy from probability distribution.

    Args:
        q: Tensor of probabilities.
        logq: Tensor of log probabilities (natural logarithm).

    Returns:
        float: The calculated entropy value in bits.
    """
    res = q * logq / LN2
    res[q == 0] = 0
    return -res.sum().item()


def count_matching_bits_from_start(bits1: List[int], bits2: List[int]) -> int:
    """
    Counts consecutive matching bits from the beginning of two sequences.

    Args:
        bits1: First sequence of bits.
        bits2: Second sequence of bits (must be same length).

    Returns:
        int: Number of consecutive matching bits from the beginning.
    """
    assert len(bits1) == len(bits2), "Bit sequences must have equal length"

    for i, (b1, b2) in enumerate(zip(bits1, bits2)):
        if b1 != b2:
            return i

    return len(bits1)


@torch.no_grad()
def next_patch(
        model: torch.nn.Module,
        context: torch.Tensor,
        message_bits: str,
        position: Tuple[int, int],
        *,
        codebook_len: int = DEFAULT_CODEBOOK_SIZE,
        random_sample: bool = False,
        top_k: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Generates the next patch token using arithmetic coding for steganography.

    Args:
        model: The transformer model for prediction.
        context: Context tensor containing previous patches.
        message_bits: Binary string of message bits to encode.
        position: The (row, col) position within the patch grid.
        codebook_len: Size of the codebook (vocabulary size).
        random_sample: If True, samples randomly without encoding.
        top_k: Maximum number of top probable tokens to consider.

    Returns:
        Tuple[int, int]: (selected_codebook_index, num_bits_encoded)
    """
    top_k = top_k or codebook_len
    model.eval()

    # Get logits from model
    logits, _ = model.transformer(context[:-1].unsqueeze(0))
    logits = logits[:, -256:, :].squeeze()
    logits = logits.reshape(PATCH_SIZE, PATCH_SIZE, -1)
    logits = logits[position[0], position[1], :].double()

    # Sort and get probabilities
    logits, indices = logits.sort(descending=True)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Apply probability threshold
    prob_threshold = 1 / codebook_len
    k = min(max(2, torch.nonzero(probs < prob_threshold)[0].item()), top_k)
    probs_int = probs[:k]

    if random_sample:
        selection = torch.multinomial(probs, 1).item()
        return indices[selection].item(), 0

    # Convert probabilities to integer representation for arithmetic coding
    probs_int = (probs_int / probs_int.sum() * codebook_len).round().long()
    cumulative_probs = probs_int.cumsum(0)

    # Adjust for rounding errors
    overfill_index = torch.nonzero(cumulative_probs > codebook_len)
    if len(overfill_index) > 0:
        cumulative_probs = cumulative_probs[:overfill_index[0]]
    cumulative_probs += codebook_len - cumulative_probs[-1]

    # Select based on message bits
    message_idx = bits2int(message_bits, reversed=True)
    selection = torch.nonzero(cumulative_probs > message_idx)[0].item()

    # Calculate encoded bit range
    range_bottom = cumulative_probs[selection - 1].item() if selection > 0 else 0
    range_top = cumulative_probs[selection].item()

    bottom_bits = list(reversed(int2bits(range_bottom, DEFAULT_PRECISION_BITS)))
    top_bits = list(reversed(int2bits(range_top - 1, DEFAULT_PRECISION_BITS)))

    num_bits_encoded = count_matching_bits_from_start(bottom_bits, top_bits)

    return indices[selection].item(), num_bits_encoded


@torch.no_grad()
def show_image(image: torch.Tensor, *, plot_title: str = "") -> None:
    """
    Displays an image tensor using Matplotlib.

    Args:
        image: Image tensor with shape (C, H, W) and values in [0, 1].
        plot_title: The title to display above the image.
    """
    image_np = image.clip(0, 1).cpu().numpy().transpose(1, 2, 0)
    plt.imshow(image_np)
    plt.title(plot_title)
    plt.show()


def build_context_from_patches(
        reference_indices: torch.Tensor,
        building_indices: torch.Tensor,
        current_row: int,
        current_col: int,
        image_shape: torch.Size
) -> Tuple[Tensor, Tuple[int, int]]:
    """
    Builds a context tensor by concatenating reference and current patch sequences.

    Args:
        reference_indices: Reference codebook indices from context image.
        building_indices: Current generated codebook indices.
        current_row: Current row position in the patch grid.
        current_col: Current column position in the patch grid.
        image_shape: Shape of the image tensor.

    Returns:
        Tuple[Tensor, Tuple[int, int]]: (context_tensor, (local_row, local_col))
    """
    local_row, row_start, row_end = local_indexes(current_row, image_shape[1])
    local_col, col_start, col_end = local_indexes(current_col, image_shape[2])

    ref_patch = reference_indices[row_start:row_end, col_start:col_end].reshape(-1)
    curr_patch = building_indices[row_start:row_end, col_start:col_end].reshape(-1)
    context = torch.cat((ref_patch, curr_patch), dim=0)

    return context, (local_row, local_col)


def _encode_single_patch(
        model: torch.nn.Module,
        reference_indices: torch.Tensor,
        building_indices: torch.Tensor,
        current_row: int,
        current_col: int,
        image_shape: torch.Size,
        grid_shape: Tuple[int, int],
        bits_to_encode: str = "",
        random_sample: bool = False
) -> Tuple[int, int, int, int]:
    """
    Encodes a single patch position and advances to the next position.

    Args:
        model: The transformer model for prediction.
        reference_indices: Reference codebook indices tensor.
        building_indices: Building codebook indices tensor (modified in-place).
        current_row: Current row position in the patch grid.
        current_col: Current column position in the patch grid.
        image_shape: Shape of the original image tensor.
        grid_shape: Grid dimensions (height, width) in patches.
        bits_to_encode: Binary string of message bits to encode.
        random_sample: If True, samples randomly without encoding.

    Returns:
        Tuple[int, int, int, int]: (new_row, new_col, codebook_idx, num_bits_encoded)
    """
    context, (local_row, local_col) = build_context_from_patches(
        reference_indices, building_indices, current_row, current_col, image_shape
    )

    selected_idx, encoded_bits = next_patch(
        model, context, bits_to_encode, (local_row, local_col), random_sample=random_sample
    )

    building_indices[current_row, current_col] = selected_idx

    # Advance to next position (raster order)
    new_col = current_col + 1
    new_row = current_row

    if new_col >= grid_shape[1]:
        new_col = 0
        new_row += 1

    return new_row, new_col, selected_idx, encoded_bits


def encode_message_to_image(message: str) -> torch.Tensor:
    """
    Encodes a text message into a generated image using steganography.

    Args:
        message: The text message to encode into the image.

    Returns:
        torch.Tensor: A generated image tensor (C, H, W) containing the encoded message.
    """
    print("=" * 30, "Setting up the model", "=" * 30)
    dsets, model = get_vqgan_sflckr()

    print("=" * 40, " ENCODING ", "=" * 40)
    image = set_context(model, dsets, DEFAULT_CONTEXT_ROWS)

    print(f"Original image dimension: {image.unsqueeze(0).shape}")
    codebook_translations, codebook_indices = model.encode_to_z(image.unsqueeze(0))

    grid_shape = (codebook_translations.shape[2], codebook_translations.shape[3])
    reference_tensor = codebook_indices.squeeze().reshape(grid_shape)
    building_tensor = torch.zeros_like(reference_tensor)

    message_bits = string2bits(message)
    remaining_bits = message_bits

    current_row = DEFAULT_CONTEXT_ROWS // PATCH_SIZE
    current_col = 0

    # Encode message bits
    with tqdm(total=len(remaining_bits), desc="Encoding message") as pbar:
        while remaining_bits:
            next_bits = remaining_bits[:DEFAULT_PRECISION_BITS]

            current_row, current_col, _, encoded_len = _encode_single_patch(
                model, reference_tensor, building_tensor,
                current_row, current_col, image.shape, grid_shape,
                bits_to_encode=next_bits, random_sample=False
            )

            remaining_bits = remaining_bits[encoded_len:]
            pbar.update(encoded_len)

            if current_row >= grid_shape[0]:
                print("Reached the end of the image!")
                break

    # Fill remaining patches with random samples
    if current_row < grid_shape[0]:
        total_remaining = (grid_shape[0] - current_row) * grid_shape[1] - current_col
        with tqdm(total=total_remaining, desc="Filling remaining patches") as pbar:
            while current_row < grid_shape[0]:
                current_row, current_col, _, _ = _encode_single_patch(
                    model, reference_tensor, building_tensor,
                    current_row, current_col, image.shape, grid_shape,
                    random_sample=True
                )
                pbar.update(1)

    # Restore context region
    context_patches = DEFAULT_CONTEXT_ROWS // PATCH_SIZE
    building_tensor[:context_patches, :] = reference_tensor[:context_patches, :]

    # Decode to image
    image = model.decode_to_img(
        building_tensor.unsqueeze(0),
        codebook_translations.shape
    ).squeeze()

    return image


def main():
    """Main entry point for message encoding demonstration."""
    message_to_encode = "Hello world!" * 15

    torch.manual_seed(42)
    generated_image = encode_message_to_image(message_to_encode)
    show_image(generated_image, plot_title="Generated Image with Encoded Message")


if __name__ == '__main__':
    main()
