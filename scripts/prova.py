import torch
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import default_collate

from scripts.encode_message import get_vqgan_sflckr

@torch.no_grad()
def set_context(model, dsets, num_rows: int = 512) -> tuple[torch.Tensor, tuple[int, int, int]]:
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
    zeros = torch.zeros_like(x)

    # Merge the upper part of x and lower part of zeros
    image = torch.cat((x[:, :num_rows, :], zeros[:, num_rows:, :]), dim=1)

    return image, (0,num_rows,0)

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

def bits2int(bits:str) -> int:
    """
    Converts a binary string to its integer representation.

    This function interprets a string of binary digits and converts it into
    its corresponding integer value. The input should only contain valid
    binary digits ("0" or "1") as per the base-2 numeral system.

    Args:
        bits (str): A string consisting of binary digits ("0" and "1")
            representing a binary number.

    Returns:
        int: The integer representation of the given binary string.

    Raises:
        ValueError: If the input string contains characters other than
            "0" or "1", or if it is not a valid binary string.
    """
    return int(bits, 2)

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

@torch.no_grad()
def next_patch(model : torch.nn.Module,
               position : tuple[int, int, int],
               image:torch.Tensor,
               next_ten_bits : str, *,
               patch_size : tuple[int, int, int] = (3,16,16)
               ) -> tuple[torch.Tensor, int]:

    model.eval()
    local_row, row_start, row_end = local_indexes(position[1], image.shape[1])
    local_col, col_start, col_end = local_indexes(position[2], image.shape[2])

    patch = image[:,row_start:row_end,col_start:col_end]
    value = torch.rand((1,))
    return torch.ones(patch_size) * value, 10


def show_image(image: torch.Tensor) -> None:
    """
    Displays an image using Matplotlib.

    This function takes a tensor image, converts it to a NumPy array, transposes its dimensions
    to the appropriate format for visualization, and then displays it using Matplotlib's imshow.

    Args:
        image: A tensor representation of the image to be displayed. The tensor should have
            dimensions corresponding to (channels, height, width).
    """
    image = image.clip(0,1).numpy().transpose((1,2,0))
    plt.imshow(image)
    plt.show()

@torch.no_grad()
def complete_image(model : torch.nn.Module,
                   image: torch.Tensor,
                   next_position: tuple[int, int, int],
                   patch_size: tuple[int, int, int], *,
                   color_combination : None | tuple[float, float, float] = None
                   ) -> torch.Tensor:
    channels, next_h, next_w = next_position
    image_size = image.shape
    single_color_mode = color_combination is not None

    current_h = next_h
    current_w = next_w

    while current_h < image_size[1]:
        while current_w < image_size[2]:
            slices = ..., slice(current_h, current_h + patch_size[1]), slice(current_w, current_w + patch_size[2])
            if single_color_mode:
                image[slices] = torch.tensor(color_combination).view(3, 1, 1)
            else:
                 # TODO: change the next random method with the model token generation
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
    image, (channels, next_h, next_w) = set_context(model, dsets)

    quant_z, z_indices = model.encode_to_z(image.unsqueeze(0))
    show_image(model.first_stage_model.decode(quant_z))
    return


    image_size = image.shape
    patch_size = (3, 16, 16)

    message_bits = string2bits(message)
    remaining_bits = message_bits
    image_completed = False


    print(f"Image size: {image_size}\tPatch size: {patch_size}")

    while remaining_bits:
        next_encodable_bits = remaining_bits[:bits_length]
        # print(f"Next ten bits: {next_ten_bits}\tNext position: {(channels, next_h, next_w)}")

        slices = ..., slice(next_h, next_h+patch_size[1]), slice(next_w, next_w+patch_size[2])
        image[slices], encoded_bits_len = next_patch(model, image, next_encodable_bits, patch_size=patch_size)
        next_w = (next_w + patch_size[2]) % image_size[2]
        if next_w <= 0:
            next_h += patch_size[2]
            if next_h == image_size[1]:
                image_completed = True
                print("Reached the end of the image!")
                break
            elif next_h > image_size[1]:
                image = pad_image_to_height(image, height=next_h)
                print(f"Padded image to height {next_h}!")
        remaining_bits = remaining_bits[encoded_bits_len:]

    if not image_completed:
        print("WARNING: Image was not completed! Final position:", (channels, next_h, next_w))
        print("Completing image...")
        image = complete_image(model, image, (channels, next_h, next_w), patch_size, color_combination = (0,1,0))

    return image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    message_to_encode = "Hello world!\n"*20

    torch.manual_seed(0)
    generated_image = encode_message_to_image(message_to_encode)
    show_image(generated_image)






if __name__ == '__main__':
    main()