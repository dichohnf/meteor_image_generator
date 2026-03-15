import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
from tqdm.auto import tqdm

from main import DataModuleFromConfig
from scripts.encode_methods import build_context_from_patches
from scripts.input import Options
from scripts.utils import (DEFAULT_CONTEXT_ROWS, set_context, bits2string, int2bits,
                           DEFAULT_PRECISION_BITS, DEFAULT_CODEBOOK_SIZE, PATCH_SIZE)


def load_image(path : str) -> torch.Tensor:
    img = read_image(path)
    # loaded_img = Image.open(path).convert("RGB")
    # img_np = np.array(loaded_img)
    # Reverse the normalization: from [0, 255] back to [-1.0, 1.0]
    img = (img.type(torch.float32) / 127.5) - 1.0
    # img = img.permute(2, 0, 1)
    # restored_tensor = torch.from_numpy(img_np)
    img = img.to('cuda')
    return img


import torch
from typing import Tuple, Optional


@torch.no_grad()
def decode_next_patch(
        model: torch.nn.Module,
        context: torch.Tensor,
        actual_token: int,  # Replaces 'bits_to_encode'
        position: Tuple[int, int],
        *,
        codebook_len: int = DEFAULT_CODEBOOK_SIZE,  # Make sure to import this
        top_k: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Decodes the hidden message bits from a given image patch token.
    """

    top_k = top_k or codebook_len
    model.eval()

    # 1. Get logits from the model (Exactly the same as encoder)
    logits, _ = model.transformer(context[:-1].unsqueeze(0))
    logits = logits[:, -256:, :].squeeze()

    # Ensure PATCH_SIZE is defined/imported (assuming 16 based on your encoder)
    logits = logits.reshape(PATCH_SIZE, PATCH_SIZE, -1)
    logits = logits[position[0], position[1], :].double()

    # 2. Sort and get probabilities (Exactly the same as encoder)
    logits, indices = logits.sort(descending=True)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # 3. Apply probability threshold (Exactly the same as encoder)
    prob_threshold = 1 / codebook_len
    k = min(max(2, torch.nonzero(probs < prob_threshold)[0].item()), top_k)
    probs_int = probs[:k]

    # 4. Convert probabilities to integer representation
    probs_int = (probs_int / probs_int.sum() * codebook_len).round().long()
    cumulative_probs = probs_int.cumsum(0)

    # 5. Adjust for rounding errors
    overfill_index = torch.nonzero(cumulative_probs > codebook_len)
    if len(overfill_index) > 0:
        cumulative_probs = cumulative_probs[:overfill_index[0]]
    cumulative_probs += codebook_len - cumulative_probs[-1]

    # 6. Find the interval for the actual token
    matching_indices = torch.nonzero(indices == actual_token)

    # If the token isn't in our valid top_k distribution, something went wrong
    # during encoding/transmission, or the image was corrupted.
    if len(matching_indices) == 0 or matching_indices[0].item() >= len(cumulative_probs):
        return actual_token, ""

    selection = matching_indices[0].item()

    # 7. Calculate the probability boundaries
    range_bottom = cumulative_probs[selection - 1].item() if selection > 0 else 0
    range_top = cumulative_probs[selection].item()

    # 8. Reconstruct the bits (Find the matching prefix)
    bottom_bits = list(reversed(int2bits(range_bottom, DEFAULT_PRECISION_BITS)))
    top_bits = list(reversed(int2bits(range_top - 1, DEFAULT_PRECISION_BITS)))

    decoded_bits = ""
    for b, t in zip(bottom_bits, top_bits):
        if b == t:
            decoded_bits += str(b)
        else:
            break

    return actual_token, decoded_bits


def _decode_single_patch(
        model: torch.nn.Module,
        reference_indices: torch.Tensor,
        building_indices: torch.Tensor,
        current_row: int,
        current_col: int,
        grid_shape: Tuple[int, int],
) -> Tuple[int, int, str]:
    context, (local_row, local_col) = build_context_from_patches(
        reference_indices, building_indices, current_row, current_col, grid_shape
    )

    actual_token = building_indices[current_row, current_col].item()

    _, decoded_bits = decode_next_patch(
        model, context, actual_token, (local_row, local_col)
    )

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
        image: torch.Tensor
) -> str:
    # FIX: Do not overwrite 'image'. Use '_' for the first return value.
    _, cond_tensor = set_context(model, dsets, DEFAULT_CONTEXT_ROWS)

    # Now, model.encode_to_z actually processes your stego-image!
    image_translations, image_indices = model.encode_to_z(image.unsqueeze(0))
    cond_translations, cond_indices = model.encode_to_c(cond_tensor)

    grid_shape = (image_translations.shape[2], image_translations.shape[3])

    reference_tensor = cond_indices.reshape(
        cond_translations.shape[0], cond_translations.shape[2], cond_translations.shape[3]
    ).squeeze()

    half_start = math.floor(image_indices.shape[1] * options.context_fraction)

    # Reshape the actual stego-image indices into our grid
    building_tensor = image_indices.reshape(grid_shape)  # hw

    current_row = half_start // grid_shape[1]
    current_col = half_start % grid_shape[1]

    # Decoding message bits
    total_remaining = (grid_shape[0] - current_row) * grid_shape[1] - current_col
    pbar = tqdm(total=total_remaining, desc="Decoding message", disable=False)
    message_bits = ""

    with pbar:
        while current_row < grid_shape[0]:
            current_row, current_col, decoded_bits = _decode_single_patch(
                model, reference_tensor, building_tensor,
                current_row, current_col, grid_shape,
            )
            message_bits += decoded_bits
            pbar.update(1)
    decoded_string = bits2string(message_bits)
    return decoded_string
