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

# Constants
PATCH_SIZE = 16
DEFAULT_CODEBOOK_SIZE = 1024
DEFAULT_PRECISION_BITS = 10
DEFAULT_CONTEXT_ROWS = 512
LN2 = 0.69315  # ln(2) for entropy calculation

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
    x_np = ((image.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype("uint8")
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

