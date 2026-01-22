import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn.functional import threshold
from torch.utils.data.dataloader import default_collate
from typing import Tuple, List, Union, Any

from tqdm import tqdm

from scripts.encode_message import get_vqgan_sflckr


@torch.no_grad()
def set_context(model, dsets, num_rows: int) -> torch.Tensor:
    """
    Prepares and returns a context image from the given dataset.

    Extracts a random image from the dataset, processes it through the model,
    and crops it to align with 16-pixel patch boundaries. The resulting image
    can be used as context for subsequent image generation tasks.

    Args:
        model: The model instance used to extract and process input data. Must
            implement a `get_input` method that accepts a key and example batch.
        dsets: A data structure containing datasets. Can be a dictionary with a
            'datasets' attribute containing one or more named datasets.
        num_rows: Number of pixel rows to retain from the top of the image.
            Must not be None. If larger than the image height, it will be
            clamped to the image height.

    Returns:
        torch.Tensor: A 3D tensor with shape (channels, height, width) representing
            the prepared context image, cropped to be divisible by 16 pixels.

    Raises:
        ValueError: If num_rows is None.
    """
    if num_rows is None:
        raise ValueError("num_rows must be not None")
    if len(dsets.datasets) > 1:
        split = sorted(dsets.datasets.keys())[0]
        dset = dsets.datasets[split]
    else:
        dset = next(iter(dsets.datasets.values()))

    context_idx = torch.randint(len(dset), size=(1,))
    example = default_collate([dset[context_idx.item()]])
    x = model.get_input("image", example).to(model.device).squeeze()
    if num_rows > x.shape[1]:
        num_rows = x.shape[1]
        print("WARNING: num_rows is greater than the image height. Setting num_rows to image height.")
    x = x[:,
    :x.shape[1] - ((x.shape[1] + num_rows) % 16),
    :x.shape[2] - (x.shape[2] % 16)]

    return x


def string2bits(message, code='ASCII') -> str:
    """
    Converts a string message into its binary representation.

    Each character in the input message is converted to its numeric code
    (according to the specified encoding), then to a binary string with
    appropriate bit padding. All character binary strings are concatenated
    into a single binary string.

    Args:
        message (str): The input string to be converted into binary representation.
        code (str, optional): The encoding standard to use for conversion.
            Defaults to 'ASCII'. Supported values:
            - 'ASCII': 8-bit ASCII encoding (0-255)
            - 'UNICODE': 21-bit Unicode encoding (up to U+10FFFF)
            - 'DECIMAL': 4-bit decimal encoding (0-9 only)

    Returns:
        str: A binary string representing the entire input message, where each
            character is encoded according to the specified encoding scheme.

    Raises:
        ValueError: If the encoding is 'DECIMAL' and the message contains
            non-digit characters.
        ValueError: If an unsupported encoding is specified.
    """
    bits = ""
    code = code.upper()

    if code == 'ASCII':
        for char in message:
            char_code = ord(char)
            char_bits = bin(char_code)[2:].zfill(8)
            bits += char_bits
    elif code == 'UNICODE':
        for char in message:
            char_code = ord(char)
            char_bits = bin(char_code)[2:].zfill(21)  # Unicode can go up to U+10FFFF (21 bits)
            bits += char_bits
    elif code == 'DECIMAL':
        if not message.isdigit():
            raise ValueError("Message must contain only decimal digits for 'decimal' encoding")
        for char in message:
            char_code = int(char)
            char_bits = bin(char_code)[2:].zfill(4)  # 0-9 needs 4 bits
            bits += char_bits
    else:
        raise ValueError(f"Unsupported encoding: {code}. Use 'ASCII', 'unicode', or 'decimal'.")

    return bits


def bits2int(bits: str, *, reversed: bool = False) -> int:
    """
    Converts a binary string to its integer representation.

    Interprets a string of binary digits ('0' and '1') as a base-2 number
    and returns the corresponding integer value. Optionally reverses the
    bit string before conversion.

    Args:
        bits (str): A string consisting of binary digits ("0" and "1")
            representing a binary number.
        reversed (bool, optional): If True, reverses the bit string before
            converting to integer. Defaults to False. Useful for little-endian
            bit ordering.

    Returns:
        int: The integer representation of the given binary string.

    Raises:
        ValueError: If the input string contains characters other than
            "0" or "1", or if it is not a valid binary string format.
    """
    return int(bits, 2) if not reversed else int(bits[::-1], 2)


def local_indexes(index: int, max_len: int) -> Tuple[int, int, int]:
    """
    Calculates local position within a 16-element sliding window.

    Computes the local index position and the start/end boundaries of a
    16-element window centered around the given index. The window is adjusted
    to stay within valid boundaries [0, max_len).

    Args:
        index (int): The current absolute index position within the range [0, max_len).
        max_len (int): The total length of the range.

    Returns:
        Tuple[int, int, int]: A tuple containing:
            - local_ind (int): The local position within the 16-element window (0-15)
            - idx_start (int): The starting index of the 16-element window
            - idx_end (int): The ending index of the 16-element window (exclusive)

    Note:
        - For indices 0-8: window starts at 0, local position equals index
        - For indices near the end (max_len-8 to max_len-1): window ends at max_len
        - For middle indices: window is centered at position 8
    """
    if index <= 8:
        local_ind = index
    elif max_len - index < 8:
        local_ind = 16 - (max_len - index)
    else:
        local_ind = 8

    idx_start = index - local_ind
    idx_end = idx_start + 16

    return local_ind, idx_start, idx_end


def entropy(q, logq) -> float:
    """
    Calculates Shannon entropy from probability distribution.

    Computes the entropy H = -sum(q * log(q)) using pre-computed log probabilities,
    with conversion from natural log to log base 2 via division by ln(2) ≈ 0.69315.

    Args:
        q: Tensor of probabilities.
        logq: Tensor of log probabilities (natural logarithm).

    Returns:
        float: The calculated entropy value in bits.
    """
    res = q * logq / 0.69315
    res[q == 0] = 0
    return -res.sum().item()


def int2bits(inp, num_bits) -> List[int]:
    """
    Converts an integer to a reversed list of binary digits.

    Converts the input integer to a binary representation with the specified
    number of bits, then returns the bits in reversed order as a list of integers.

    Args:
        inp: The integer value to convert to binary.
        num_bits: The number of bits to use for the binary representation.
            The result will be zero-padded to this length.

    Returns:
        List[int]: A list of integers (0 or 1) representing the binary digits
            in reversed order (least significant bit first). Returns an empty
            list if num_bits is 0.
    """
    if num_bits == 0:
        return []
    str_list = ('{0:0%db}' % num_bits).format(inp)
    return [int(str_val) for str_val in reversed(str_list)]


def num_same_from_beg(bits1, bits2) -> int:
    """
    Counts the number of matching bits from the beginning of two bit sequences.

    Compares two equal-length bit sequences element by element from the start
    and returns the count of consecutive matching bits before the first mismatch.

    Args:
        bits1: First sequence of bits (list or similar indexable sequence).
        bits2: Second sequence of bits (list or similar indexable sequence).
            Must have the same length as bits1.

    Returns:
        int: The number of consecutive matching bits from the beginning.
            Returns the full length if all bits match.

    Raises:
        AssertionError: If bits1 and bits2 have different lengths.
    """
    assert len(bits1) == len(bits2)
    i = 0
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break

    return i


@torch.no_grad()
def next_patch(model: torch.nn.Module,
               context: torch.Tensor,
               next_ten_bits: str,
               position: Tuple[int, int],
               *,
               codebook_len: int = 1024,
               random_sample: bool = False,
               top_k: int = None,
               ) -> Tuple[Union[int, float, bool], int]:
    """
    Generates the next patch token using arithmetic coding for steganography.

    Uses a transformer model to predict probability distribution over codebook
    entries, then either samples randomly or encodes message bits into the
    selection using arithmetic coding. This enables hiding data in generated images.

    Args:
        model (torch.nn.Module): The transformer model used for prediction.
            Must have a `transformer` attribute that outputs logits.
        context (torch.Tensor): Context tensor containing previous patches.
        next_ten_bits (str): Binary string of message bits to encode (typically 10 bits).
            Ignored if random_sample is True.
        position (Tuple[int, int]): The (row, col) position within the 16x16 patch grid.
        codebook_len (int, optional): Size of the codebook (vocabulary size).
            Defaults to 1024 (2^10).
        random_sample (bool, optional): If True, samples randomly without encoding
            message bits. Defaults to False.
        top_k (int, optional): Maximum number of top probable tokens to consider.
            Defaults to codebook_len if not specified.

    Returns:
        Tuple[Union[int, float, bool], int]: A tuple containing:
            - selected_index (int): The selected codebook index
            - num_bits_encoded (int): Number of message bits successfully encoded
              (0 if random_sample is True)
    """
    if top_k is None:
        top_k = codebook_len
    model.eval()

    logits, _ = model.transformer(context[:-1].unsqueeze(0))
    logits = logits[:, -256:, :].squeeze()
    logits = logits.reshape(16, 16, -1)
    logits = logits[position[0], position[1], :]

    # Sampling with Meteor
    logits, indices = logits.sort(descending=True)
    logits = logits.double()

    probs = torch.nn.functional.softmax(logits, dim=-1)

    probs_threshold = 1 / codebook_len
    k = min(max(2, torch.nonzero((probs < probs_threshold))[0].item()), top_k)
    probs_int = probs[:k]

    if random_sample:
        ix = torch.multinomial(probs, 1)
        return ix.item(), 0

    probs_int = probs_int / probs_int.sum() * codebook_len
    probs_int = probs_int.round().long()

    # Remove any elements from the bottom if rounding caused the total prob to be too large
    cumulative_probs = probs_int.cumsum(0)
    overfill_index = torch.nonzero((cumulative_probs > codebook_len))
    if len(overfill_index) > 0:
        cumulative_probs = cumulative_probs[:overfill_index[0]]
    # Add any mass to the top if removing/rounding causes the total prob to be too small
    cumulative_probs += codebook_len - cumulative_probs[-1]

    # Get out resulting probabilities
    probs_final = cumulative_probs.clone()
    probs_final[1:] = cumulative_probs[1:] - cumulative_probs[:-1]

    # Get selected index based on binary fraction from message bits
    message_idx = bits2int(next_ten_bits, reversed=True)
    selection = torch.nonzero((cumulative_probs > message_idx))[0].item()

    # Calculate new range as ints
    new_int_bottom = cumulative_probs[selection - 1] if selection > 0 else 0
    new_int_top = cumulative_probs[selection]

    # Convert range to bits
    new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, 10)))
    new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, 10)))  # -1 here because upper bound is exclusive

    # Consume most significant bits which are now fixed and update interval
    num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

    return selection, num_bits_encoded


@torch.no_grad()
def show_image(image: torch.Tensor, *, plot_title: str = "") -> None:
    """
    Displays an image tensor using Matplotlib.

    Converts a PyTorch tensor image from (C, H, W) format to (H, W, C) format,
    clips values to [0, 1] range, and displays it using matplotlib.

    Args:
        image (torch.Tensor): A tensor representation of the image with shape
            (channels, height, width). Values should be in the range [0, 1] or
            will be clipped to this range.
        plot_title (str, optional): The title to display above the image.
            Defaults to an empty string.
    """
    image = image.clip(0, 1).cpu().numpy().transpose((1, 2, 0))
    plt.imshow(image)
    plt.title(plot_title)
    plt.show()


@torch.no_grad()
def complete_image(model: torch.nn.Module,
                   image: torch.Tensor,
                   next_position: Tuple[int, int, int],
                   patch_size: Tuple[int, int, int], *,
                   color_combination: "None | Tuple[float, float, float]" = None
                   ) -> torch.Tensor:
    """
    Completes remaining portions of an image by filling patches.

    Starting from the specified position, fills all remaining patches in the image
    either with a solid color or random values. Processes patches row by row,
    left to right.

    Args:
        model (torch.nn.Module): The model (currently unused, placeholder for future
            token generation implementation).
        image (torch.Tensor): The image tensor to complete, with shape (C, H, W).
        next_position (Tuple[int, int, int]): Starting position as (channel, height, width).
            Only height and width are used.
        patch_size (Tuple[int, int, int]): Size of each patch as (channels, height, width).
        color_combination (None | Tuple[float, float, float], optional): RGB values
            for solid color fill. If None, uses random values. Defaults to None.

    Returns:
        torch.Tensor: The completed image tensor with all patches filled.
    """
    channels, next_h, next_w = next_position
    image_size = image.shape
    single_color_mode = color_combination is not None

    current_h = next_h
    current_w = next_w

    while current_h < image_size[1]:
        while current_w < image_size[2]:
            slices = slice(0, 3), slice(current_h, current_h + patch_size[1]), slice(current_w,
                                                                                     current_w + patch_size[2])
            if single_color_mode:
                image[slices] = torch.tensor(color_combination).view(3, 1, 1)
            else:
                # TODO: change the random method with the model token generation
                image[slices] = torch.rand((3, patch_size[1], patch_size[2]))
            current_w += patch_size[2]
        current_w = 0
        current_h += patch_size[1]

    return image


def pad_image_to_height(image: torch.Tensor,
                        height: int) -> torch.Tensor:
    """
    Pads an image tensor to a specified height by adding zero rows at the bottom.

    If the current height is less than the target height, appends rows of zeros
    to the bottom of the image. If already at or above target height, returns
    the image unchanged.

    Args:
        image (torch.Tensor): The image tensor to pad, with shape (C, H, W).
        height (int): The target height in pixels.

    Returns:
        torch.Tensor: The padded image tensor with height >= target height.
            Has the same number of channels and width as the input.
    """
    current_h = image.shape[1]
    if height > current_h:
        padding_size = height - current_h
        zeros_padding = torch.zeros((image.shape[0], padding_size, image.shape[2]),
                                    dtype=image.dtype,
                                    device=image.device)
        image = torch.cat((image, zeros_padding), dim=1)
    return image


def build_context_from_patches(cidx: torch.Tensor,
                               idx: torch.Tensor,
                               current_row: int,
                               current_col: int,
                               image_shape: torch.Size) -> Tuple[Tensor, Tuple[Any, Any]]:
    """
    Builds a context tensor by concatenating reference and current patch sequences.

    Extracts 16x16 windows from both reference (cidx) and generated (idx) codebook
    indices around the current position, flattens them, and concatenates them to
    form the context for the next prediction.

    Args:
        cidx (torch.Tensor): Reference codebook indices tensor (typically from
            the context/original image).
        idx (torch.Tensor): Current generated codebook indices tensor being constructed.
        current_row (int): Current row position in the patch grid.
        current_col (int): Current column position in the patch grid.
        image_shape (torch.Size): Shape of the image tensor (used for boundary calculation).

    Returns:
        Tuple[Tensor, Tuple[int, int]]: A tuple containing:
            - context (Tensor): Flattened and concatenated patch sequences
            - (local_row, local_col) (Tuple[int, int]): Local position within the 16x16 window
    """
    local_row, row_start, row_end = local_indexes(current_row, image_shape[1])
    local_col, col_start, col_end = local_indexes(current_col, image_shape[2])

    cpatch = cidx[row_start:row_end, col_start:col_end].reshape(-1)
    patch = idx[row_start:row_end, col_start:col_end].reshape(-1)
    context = torch.cat((cpatch, patch), dim=0)

    return context, (local_row, local_col)


def _encode_single_patch(model: torch.nn.Module,
                         cidx: torch.Tensor,
                         idx: torch.Tensor,
                         current_row: int,
                         current_col: int,
                         image_shape: torch.Size,
                         shape: Tuple[int, int],
                         bits_to_encode: str = "",
                         random_sample: bool = False
                         ) -> Tuple[int, int, int, int]:
    """
    Encodes a single patch position and advances to the next position.

    Builds context from surrounding patches, selects the next codebook index
    (either by encoding message bits or random sampling), updates the building
    tensor, and calculates the next position in raster order.

    Args:
        model (torch.nn.Module): The transformer model for prediction.
        cidx (torch.Tensor): Reference codebook indices tensor.
        idx (torch.Tensor): Current building codebook indices tensor (modified in-place).
        current_row (int): Current row position in the patch grid.
        current_col (int): Current column position in the patch grid.
        image_shape (torch.Size): Shape of the original image tensor.
        shape (Tuple[int, int]): Grid dimensions as (height, width) in patches.
        bits_to_encode (str, optional): Binary string of message bits to encode.
            Empty string for random sampling. Defaults to "".
        random_sample (bool, optional): If True, samples randomly without encoding.
            Defaults to False.

    Returns:
        Tuple[int, int, int, int]: A tuple containing:
            - new_row (int): Updated row position
            - new_col (int): Updated column position
            - selected_codebook_idx (int): The selected codebook index
            - encoded_bits_len (int): Number of message bits encoded
    """
    context, (local_row, local_col) = build_context_from_patches(
        cidx, idx, current_row, current_col, image_shape
    )

    selected_codebook_idx, encoded_bits_len = next_patch(
        model, context, bits_to_encode, (local_row, local_col), random_sample=random_sample
    )

    idx[current_row, current_col] = selected_codebook_idx

    # Update position
    new_col = current_col + 1
    new_row = current_row

    if new_col >= shape[1]:
        new_col = 0
        new_row = current_row + 1

    return new_row, new_col, selected_codebook_idx, encoded_bits_len


def encode_message_to_image(message: str) -> torch.Tensor:
    """
    Encodes a text message into a generated image using steganography.

    Converts the message to binary, sets up a VQGAN model with context from a
    dataset image, then generates patches that encode the message bits using
    arithmetic coding. Remaining image portions are filled with random patches.

    Args:
        message (str): The text message to encode into the image.

    Returns:
        torch.Tensor: A generated image tensor with shape (C, H, W) containing
            the encoded message. The image has a context region at the top from
            the original dataset and generated regions below that encode the message.

    Note:
        Uses precision of 10 bits per patch (codebook size 1024) and a context
        region of 512 pixels height from a random dataset image.
    """
    # Hyperparameters
    precision = 10
    codebook_len = 2 ** precision
    context_rows = 512

    print("=" * 30, "Setting up the model", "=" * 30)
    dsets, model = get_vqgan_sflckr()
    print("=" * 40, " ENCODING ", "=" * 40)

    image = set_context(model, dsets, context_rows)

    print(f"Original image dimension: {image.unsqueeze(dim=0).shape}")
    codebook_translations, codebook_indices = model.encode_to_z(image.unsqueeze(0))

    shape = (codebook_translations.shape[2], codebook_translations.shape[3])
    reconstructed_tensor = codebook_indices.squeeze().reshape(shape)
    building_tensor = torch.zeros_like(codebook_indices).squeeze().reshape(shape)

    message_bits = string2bits(message)
    to_encode_bits = message_bits

    current_row, current_col = context_rows // 16, 0
    image_completed = False

    tot = len(to_encode_bits)
    with tqdm(total=tot) as pbar:
        while to_encode_bits:
            next_encodable_bits = to_encode_bits[:precision]

            current_row, current_col, selected_codebook_idx, encoded_bits_len = _encode_single_patch(
                model, reconstructed_tensor, building_tensor, current_row, current_col, image.shape, shape,
                bits_to_encode=next_encodable_bits, random_sample=False
            )

            to_encode_bits = to_encode_bits[encoded_bits_len:]
            pbar.update(encoded_bits_len)

            if current_row >= shape[0]:
                image_completed = True
                print("Reached the end of the image!")
                break

    if image_completed:
        building_tensor[:context_rows // 16, :shape[1]] = reconstructed_tensor[:context_rows // 16, :shape[1]]
        image = model.decode_to_img(building_tensor.unsqueeze(0), codebook_translations.shape).squeeze()
        return image

    tot = (shape[0] - current_row - 1) * shape[1] + (shape[1] - current_col)
    with tqdm(total=tot) as pbar:
        while True:
            current_row, current_col, selected_codebook_idx, encoded_bits_len = _encode_single_patch(
                model, reconstructed_tensor, building_tensor, current_row, current_col, image.shape, shape,
                bits_to_encode="", random_sample=True
            )

            pbar.update(1)

            if current_row >= shape[0]:
                print("Image completed!")
                break

    building_tensor[:context_rows // 16, :shape[1]] = reconstructed_tensor[:context_rows // 16, :shape[1]]
    image = model.decode_to_img(building_tensor.unsqueeze(0), codebook_translations.shape).squeeze()
    return image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    Main entry point for message encoding demonstration.

    Encodes a sample message into an image and displays the result.
    Uses a fixed random seed for reproducibility.
    """
    message_to_encode = "Hello world!" * 15

    torch.manual_seed(42)
    generated_image = encode_message_to_image(message_to_encode)
    show_image(generated_image)


if __name__ == '__main__':
    main()