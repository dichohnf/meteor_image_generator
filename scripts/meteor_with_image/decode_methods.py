import math
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from main import DataModuleFromConfig
from scripts.meteor_with_image.encode_methods import build_context_from_patches
from scripts.meteor_with_image.utils import DEFAULT_CONTEXT_ROWS, set_context, bits2string
from scripts.meteor_with_image.input import Options


def load_image(path) -> torch.Tensor:
    loaded_img = Image.open(path).convert("RGB")
    img_np = np.array(loaded_img)
    # Reverse the normalization: from [0, 255] back to [-1.0, 1.0]
    img_np = (img_np.astype(np.float32) / 127.5) - 1.0
    img_np = img_np.transpose(2, 0, 1)
    restored_tensor = torch.from_numpy(img_np)
    restored_tensor = restored_tensor.to('cuda')
    return restored_tensor


def decode_next_patch(
        model: torch.nn.Module,
        context: torch.Tensor,
        bits_to_encode: str,
        param: Tuple[int, int],
        random_sample: bool) -> Tuple[int, str]:
    pass


def _decode_single_patch(
        model: torch.nn.Module,
        reference_indices: torch.Tensor,
        building_indices: torch.Tensor,
        current_row: int,
        current_col: int,
        grid_shape: Tuple[int, int],
        bits_to_encode: str = "",
        random_sample: bool = False
) -> Tuple[int, int, str]:
    context, (local_row, local_col) = build_context_from_patches(
        reference_indices, building_indices, current_row, current_col, grid_shape
    )

    selected_idx, decoded_bits = decode_next_patch(
        model, context, bits_to_encode, (local_row, local_col), random_sample=random_sample
    )

    building_indices[current_row, current_col] = selected_idx

    # Advance to next position (raster order)
    new_col = current_col + 1
    new_row = current_row

    if new_col >= grid_shape[1]:
        new_col = 0
        new_row += 1

    return new_row, new_col, decoded_bits


def decode_message(
        options: Options,
        model: torch.nn.Module,
        dsets: DataModuleFromConfig,
        image: torch.Tensor,
        context_fraction: int) -> str:
    image, cond_tensor = set_context(model, dsets, DEFAULT_CONTEXT_ROWS)
    image_translations, image_indices = model.encode_to_z(image.unsqueeze(0))
    cond_translations, cond_indices = model.encode_to_c(cond_tensor)

    grid_shape = (image_translations.shape[2], image_translations.shape[3])

    reference_tensor = cond_indices.reshape(
        cond_translations.shape[0], cond_translations.shape[2], cond_translations.shape[3]
    ).squeeze()

    half_start = math.floor(image_indices.shape[1] * context_fraction)
    building_tensor = image_indices
    building_tensor[:, half_start:] = 0
    building_tensor = building_tensor.reshape(grid_shape)  # hw

    current_row = half_start // grid_shape[1]
    current_col = half_start % grid_shape[1]

    # Decoding message bits
    total_remaining = (grid_shape[0] - current_row) * grid_shape[1] - current_col
    pbar = tqdm(total=total_remaining, desc="Decoding message", disable=True)
    message_bits = ""
    with pbar:
        while current_row < grid_shape[0]:
            current_row, current_col, decoded_bits = _decode_single_patch(
                model, reference_tensor, building_tensor,
                current_row, current_col, grid_shape,
                random_sample=True
            )
            message_bits.join(decoded_bits)
            pbar.update(1)

    decoded_string = bits2string(message_bits)
    return decoded_string
