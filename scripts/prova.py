import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn.functional import threshold
from torch.utils.data.dataloader import default_collate
from typing import Tuple, List, Union

from scripts.encode_message import get_vqgan_sflckr

@torch.no_grad()
def set_context(model, dsets, num_rows: int) -> torch.Tensor:
    """
    Prepares and returns a context image from the given dataset, structured such that the
    upper portion contains real data and the lower portion contains zeros. The resulting
    image is adjusted to align with specific patch size constraints.

    Args:
        model: The model instance to extract input data. Expected to implement a `get_input`
            method for processing examples.
        dsets: A dictionary or data loader containing the datasets, with at least one dataset
            present under the "train" key or other accessible keys.
        num_rows: Number of rows to take from the dataset's image. If None, takes half of the
            image height. Default is None.

    Returns:
        tuple[torch.Tensor, tuple[int, int, int]]: A tensor representing the prepared context
        image and a tuple indicating coordinate adjustments for the generated image.
    """
    # Define dset
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

    # return torch.zeros(image_size), (0,512,0)


def string2bits(message, code='ASCII') -> str:
    """
    Converts a string message into its binary representation based on the specified encoding.

    This function takes each character from the input message, converts it to its ASCII code (or the code
    defined by the specified encoding), and then converts that code into an 8-bit binary representation.
    The binary strings for all characters are concatenated and returned as a single binary string.

    Args:
        message (str): The input string to be converted into binary representation.
        code (str): The encoding standard to use for conversion. The default value is 'ASCII'.
                   Supported values: 'ASCII', 'unicode', 'decimal'.

    Returns:
        str: A single binary string representing the entire input message.
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

def bits2int(bits:str, *, reversed:bool = False) -> int:
    """
    Converts a binary string to its integer representation.

    This function interprets a string of binary digits and converts it into
    its corresponding integer value. The input should only contain valid
    binary digits ("0" or "1") as per the base-2 numeral system.

    Args:
        bits (str): A string consisting of binary digits ("0" and "1")
            representing a binary number.
        reversed (bool): A boolean ...

    Returns:
        int: The integer representation of the given binary string.

    Raises:
        ValueError: If the input string contains characters other than
            "0" or "1", or if it is not a valid binary string.
    """
    return int(bits, 2) if not reversed else int(bits[::-1], 2)

def local_indexes(index:int, max_len:int):
    """
    Calculate local indices and range start and end based on the given index and maximum
    length. This function determines whether the local index is near the start, end, or
    middle of the range, ensuring the correct limits within the maximum length.

    Args:
        index (int): The current index within the range.
        max_len (int): The total length of the range.

    Returns:
        tuple[int, int, int]: A tuple containing the local index, the starting index
        of the range, and the ending index of the range.
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
    res = q*logq/0.69315
    res[q==0] = 0
    return -res.sum().item()


def int2bits(inp, num_bits) -> List[int]:
    if num_bits == 0:
        return []
    str_list = ('{0:0%db}'%num_bits).format(inp)
    return [int(str_val) for str_val in reversed(str_list)]


def num_same_from_beg(bits1, bits2) -> int:
    assert len(bits1) == len(bits2)
    i = 0
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break

    return i


@torch.no_grad()
def next_patch(model : torch.nn.Module,
               context : torch.Tensor,
               next_ten_bits : str,
               position : Tuple[int, int],
               *,
               codebook_len : int = 1024,
               random_sample : bool = False,
               top_k : int = None,
               ) -> Tuple[Union[int, float, bool], int]:

    # return torch.randint(0, 1024, (1,)).item(), 10
    if top_k is None:
        top_k = codebook_len
    model.eval()

    logits, _ = model.transformer(context[:-1].unsqueeze(0))
    # print(logits.shape)
    logits = logits[:, -256:, :].squeeze()
    # print(logits.shape)
    logits = logits.reshape(16, 16, -1)
    logits = logits[position[0], position[1], :]
    # print(logits.shape)
    # logits = logits.squeeze()

    # Sampling with Meteor
    logits, indices = logits.sort(descending=True)
    logits = logits.double()

    # print(logits.shape)
    # logits = logits.reshape(16, 16, -1)
    # print(logits.shape)
    # logits = logits[position[0], position[1], :]
    # print(logits.shape)
    # print(logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)

    # log_probs = logits.log_softmax(dim=-1)

    probs_threshold = 1 / codebook_len
    k = min(max(2, torch.nonzero((probs < probs_threshold))[0].item()), top_k)
    probs_int = probs[:k]

    if random_sample:
        ix = torch.multinomial(probs, 1)
        # _, ix = torch.topk(probs, k=1)
        return ix.item(), 0

    probs_int = probs_int / probs_int.sum() * codebook_len
    probs_int = probs_int.round().long()

    # current_entropy = entropy(probs, log_probs)

    # Remove any elements from the bottom if rounding caused the total prob to be too large
    cumulative_probs = probs_int.cumsum(0)
    overfill_index = torch.nonzero((cumulative_probs > codebook_len))
    if len(overfill_index) > 0:
        cumulative_probs = cumulative_probs[:overfill_index[0]]
    # Add any mass to the top if removing/rounding causes the total prob to be too small
    cumulative_probs += codebook_len - cumulative_probs[-1]  # add

    # Get out resulting probabilities
    probs_final = cumulative_probs.clone()
    probs_final[1:] = cumulative_probs[1:] - cumulative_probs[:-1]

    # Convert to position in range
    # cumulative_probs += 0

    # mask_bits = mask_generator.generate_bits(precision)
    # for b in range(0, len(message_bits)):
    #     message_bits[b] = message_bits[b] ^ mask_bits[b]

    # Get selected index based on binary fraction from message bits
    message_idx = bits2int(next_ten_bits, reversed=True)
    selection = torch.nonzero((cumulative_probs > message_idx))[0].item()

    # Calculate new range as ints
    new_int_bottom = cumulative_probs[selection - 1] if selection > 0 else 0
    new_int_top = cumulative_probs[selection]

    # Convert range to bits
    new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, 10)))
    new_int_top_bits_inc    = list(reversed(int2bits(new_int_top - 1, 10)))  # -1 here because upper bound is exclusive

    # Consume most significant bits which are now fixed and update interval
    num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

    return selection, num_bits_encoded

@torch.no_grad()
def show_image(image: torch.Tensor, *, plot_title:str = "") -> None:
    """
    Displays an image using Matplotlib.

    This function takes a tensor image, converts it to a NumPy array, transposes its dimensions
    to the appropriate format for visualization, and then displays it using Matplotlib's imshow.

    Args:
        image: A tensor representation of the image to be displayed. The tensor should have
            dimensions corresponding to (channels, height, width).
        plot_title: A string representing the title of the image to be displayed.
    """
    image = image.clip(0,1).cpu().numpy().transpose((1,2,0))
    plt.imshow(image)
    plt.title(plot_title)
    plt.show()

@torch.no_grad()
def complete_image(model : torch.nn.Module,
                   image: torch.Tensor,
                   next_position: Tuple[int, int, int],
                   patch_size: Tuple[int, int, int], *,
                   color_combination : "None | Tuple[float, float, float]" = None
                   ) -> torch.Tensor:
    channels, next_h, next_w = next_position
    image_size = image.shape
    single_color_mode = color_combination is not None

    current_h = next_h
    current_w = next_w

    while current_h < image_size[1]:
        while current_w < image_size[2]:
            slices = slice(0,3), slice(current_h, current_h + patch_size[1]), slice(current_w, current_w + patch_size[2])
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
    current_h = image.shape[1]
    if height > current_h:
        padding_size = height - current_h
        zeros_padding = torch.zeros((image.shape[0], padding_size, image.shape[2]),
                                    dtype=image.dtype,
                                    device=image.device)
        image = torch.cat((image, zeros_padding), dim=1)
    return image


def encode_message_to_image(message: str) -> torch.Tensor:
    bits_length = 10
    precision = 2 ** bits_length

    print("=" * 40, " ENCODING ", "=" * 40)
    # print(f"Message: {message}")

    dsets, model = get_vqgan_sflckr()
    num_rows = 512
    image = set_context(model, dsets, num_rows)

    print(f"Original image dimension: {image.unsqueeze(dim=0).shape}")
    codebook_translations, codebook_indices = model.encode_to_z(image.unsqueeze(0))
    # reconstructed_image = model.first_stage_model.decode(codebook_translations)
    # show_image(reconstructed_image.squeeze(), plot_title="Reconstructed image")

    shape = (codebook_translations.shape[2],codebook_translations.shape[3])
    cidx  = codebook_indices.squeeze().reshape(shape)
    idx   = torch.zeros_like(codebook_indices).squeeze().reshape(shape)

    # print(f"Idx shape: {idx.shape}")
    # print(f"Cidx shape: {cidx.shape}")

    image_size = image.shape
    patch_size = (3, 16, 16)

    message_bits = string2bits(message)
    remaining_bits = message_bits
    image_completed = False

    next_h, next_w = num_rows // 16, 0


    # print(f"Image size: {image_size}\tPatch size: {patch_size}")

    tot = len(remaining_bits)
    while remaining_bits:
        print(f"Remaining: {len(remaining_bits)}/{tot}")
        next_encodable_bits = remaining_bits[:bits_length]
        # print(f"Next ten bits: {next_ten_bits}\tNext position: {(channels, next_h, next_w)}")

        local_row, row_start, row_end = local_indexes(next_h, image.shape[1])
        local_col, col_start, col_end = local_indexes(next_w, image.shape[2])

        cpatch = cidx[row_start:row_end, col_start:col_end].reshape(-1)
        patch  =  idx[row_start:row_end, col_start:col_end].reshape(-1)
        context = torch.cat((cpatch, patch), dim=0)

        selected_idx, encoded_bits_len = next_patch(model, context,
                                                    next_encodable_bits,
                                                    (local_row,local_col))
        # print(f"Prev: pos:({local_row},{local_col}), idx:{idx[local_row,local_col]}")
        idx[next_h, next_w] = selected_idx
        # print(f"Succ: pos:({local_row},{local_col}), idx:{idx[local_row,local_col]}")

        # print(codebook_indices.shape)
        remaining_bits = remaining_bits[encoded_bits_len:]
        # print(len(remaining_bits))
        # show_image(image)
        next_w += 1
        if next_w >= shape[1]:
            next_w = 0
            next_h += 1
            if next_h >= shape[0]:
                image_completed = True
                print("Reached the end of the image!")
                break

    if image_completed:
        idx[:num_rows // 16, :shape[1]] = cidx[:num_rows // 16, :shape[1]]
        image = model.decode_to_img(idx.unsqueeze(0), codebook_translations.shape).squeeze()
        return image

    while True:
        print(f"Remaining second loop: {next_h}/{shape[0]}")

        local_row, row_start, row_end = local_indexes(next_h, image.shape[1])
        local_col, col_start, col_end = local_indexes(next_w, image.shape[2])

        cpatch = cidx[row_start:row_end, col_start:col_end].reshape(-1)
        patch  =  idx[row_start:row_end, col_start:col_end].reshape(-1)
        context = torch.cat((cpatch, patch), dim=0)

        selected_idx, encoded_bits_len = next_patch(model, context,
                                                    "",
                                                    (local_row, local_col),
                                                    random_sample=True)
        idx[next_h, next_w] = selected_idx
        next_w += 1
        if next_w >= shape[1]:
            next_w = 0
            next_h += 1
            if next_h >= shape[0]:
                print("Image completed!")
                break
    idx[:num_rows//16, :shape[1]] = cidx[:num_rows//16, :shape[1]]
    image = model.decode_to_img(idx.unsqueeze(0), codebook_translations.shape).squeeze()
    return image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    message_to_encode = "Hello world!"*15

    torch.manual_seed(42)
    generated_image = encode_message_to_image(message_to_encode)
    show_image(generated_image)


if __name__ == '__main__':
    main()