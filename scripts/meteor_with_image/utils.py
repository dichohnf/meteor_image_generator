import glob
import os
from pathlib import Path
from typing import List, Tuple, Any

import importlib
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data.dataloader import default_collate

# Constants
PATCH_SIZE = 16
DEFAULT_CODEBOOK_SIZE = 1024
DEFAULT_PRECISION_BITS = 10
DEFAULT_CONTEXT_ROWS = 0
LN2 = 0.69315  # ln(2) for entropy calculation


def bits2string(binary_str: str, code: str = 'ASCII') -> str:
    """
    Decodes a binary string back into its original message using the specified encoding.

    Args:
        binary_str (str): The binary string to decode.
        code (str, optional): The encoding scheme used for decoding. Can be 'ASCII',
                            'UNICODE', or 'DECIMAL'. Defaults to 'ASCII'.

    Returns:
        str: The decoded message as a string.

    Raises:
        ValueError: If the length of `binary_str` is not compatible with the specified `code`.
                     Or if an invalid encoding scheme is provided.
    """

    encoding_config = {
        'ASCII': 8,
        'UNICODE': 21,
        'DECIMAL': 4
    }

    if not binary_str.strip():
        return ''

    # Get the necessary bit width for the given code
    bit_width = encoding_config.get(code)
    if not bit_width:
        raise ValueError(f"Unsupported encoding: {code}. Use 'ASCII', 'UNICODE', or 'DECIMAL'.")

    total_bits = len(binary_str)
    if total_bits % bit_width != 0:
        raise ValueError("The binary string length must be divisible by the bit width for the given code.")

    # Split the binary string into chunks of size bit_width
    bits_per_char = [binary_str[i * bit_width:(i + 1) * bit_width] for i in range(total_bits // bit_width)]

    decoded_message = []
    for bits in bits_per_char:
        int_val = int(bits, 2)

        if code == 'ASCII':
            char = chr(int_val)
        elif code == 'UNICODE':
            try:
                char = chr(int_val)  # Converts binary to Unicode character
            except ValueError:
                raise ValueError(f"Invalid Unicode code point: {int_val}")
        elif code == 'DECIMAL':
            char = str(int_val)

        decoded_message.append(char)

    return ''.join(decoded_message)


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

def save_image(image: torch.Tensor, file_path : str):
    x_np = ((image.detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8").transpose(1, 2, 0)
    file_path = Path(file_path + ".png")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(x_np).save(file_path, "PNG")

def reset_seeds(seed : int):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data

def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}

def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data
    dsets = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model, global_step

def get_vqgan_sflckr(model_directory_path: str, *, quiet: bool = False) -> Tuple[Any, Any]:
    if not os.path.exists(model_directory_path):
        raise ValueError(f"Cannot find {model_directory_path}")
    if os.path.isfile(model_directory_path):
        paths = model_directory_path.split("/")
        try:
            idx = len(paths) - paths[::-1].index("logs") + 1
        except ValueError:
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
        logdir = "/".join(paths[:idx])
        ckpt = model_directory_path
    else:
        assert os.path.isdir(model_directory_path), model_directory_path
        logdir = model_directory_path.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
    if not quiet:
        print(f"logdir:{logdir}")
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = OmegaConf.merge(*configs)

    gpu = torch.cuda.is_available()
    eval_mode = True
    dsets, model, _ = load_model_and_dset(config, ckpt, gpu, eval_mode)

    return dsets, model


@torch.no_grad()
def set_context(model, dsets, num_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        raise ValueError("num_rows must be not \"None\"")

    # Select dataset
    if len(dsets.datasets) > 1:
        split = sorted(dsets.datasets.keys())[0]
        dset = dsets.datasets[split]
    else:
        dset = next(iter(dsets.datasets.values()))

    # Get random image
    context_idx = torch.randint(len(dset), size=(1,)).item()
    example = default_collate([dset[context_idx]])

    image = model.get_input("image", example).to(model.device).squeeze()
    cond_tensor = model.get_input(model.cond_stage_key, example).to(model.device)

    # Validate and adjust num_rows
    if num_rows > image.shape[1]:
        num_rows = image.shape[1]
        # print(f"WARNING: num_rows clamped to image height: {num_rows}")

    # Crop to align with PATCH_SIZE boundaries
    height_crop = image.shape[1] - ((image.shape[1] - num_rows) % PATCH_SIZE)
    width_crop = image.shape[2] - (image.shape[2] % PATCH_SIZE)
    image = image[:, :height_crop, :width_crop]

    return image, cond_tensor
