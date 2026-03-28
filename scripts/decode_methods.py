import math
import os

import numpy as np
import torch
from torchvision.io import read_image
from tqdm.auto import tqdm

from main import DataModuleFromConfig
from scripts.input import Options
from scripts.utils import (DEFAULT_CONTEXT_ROWS, set_context, bits2string, int2bits, build_context_from_patches,
                           DEFAULT_PRECISION_BITS, DEFAULT_CODEBOOK_SIZE, PATCH_SIZE)
from scripts.logger import logger




def load_image(path : str) -> torch.Tensor:
    img = read_image(path)
    # Reverse the normalization: from [0, 255] back to [-1.0, 1.0]
    img = (img.type(torch.float32) / 127.5) - 1.0
    img = img.to('cuda')
    return img


import torch
from typing import List, Tuple, Optional


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
    Decodes the next patch token using arithmetic coding for steganography.
    
    Args:
        model: The transformer model for prediction.
        context: Context tensor containing previous patches.
        actual_token: The actual token from the image that we want to decode.
        position: The (row, col) position within the patch grid.
        codebook_len: Size of the codebook (vocabulary size).
        top_k: Maximum number of top probable tokens to consider.
    Returns:
        Tuple[int, str]: (selected_codebook_index, decoded_bits)
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
    matching_indices = torch.nonzero(indices[:k] == actual_token).squeeze()

    if matching_indices.numel() == 0:
        logger.warning(f"Actual token {actual_token} is not in the top_k distribution. This may indicate an error in encoding/transmission or image corruption.")
        return -1, ""

    # If there are multiple matches, take the first one (should be unique in this context)
    if matching_indices.dim() == 0:
        selection = matching_indices.item()
    else:
        logger.warning(f"Multiple matches found for actual token {actual_token} in top_k distribution. This is unexpected and may indicate an issue with the model's output.")
        return -1, ""
    
    if selection >= len(cumulative_probs) or selection < 0:
        logger.warning(f"Selected index {selection} out of bounds for cumulative_probs with length {len(cumulative_probs)}")
        return -1, ""

    # 7. Calculate the probability boundaries
    range_bottom = cumulative_probs[selection - 1].item() if selection > 0 else 0
    range_top = cumulative_probs[selection].item()

    # 8. Reconstruct the bits (Find the matching prefix)
    bottom_bits = int2bits(range_bottom, DEFAULT_PRECISION_BITS)
    top_bits = int2bits(range_top - 1, DEFAULT_PRECISION_BITS)

    decoded_bits = ""
    for b, t in zip(bottom_bits, top_bits):
        if b == t:
            decoded_bits += str(b)
        else:
            break
        
    logger.info(f"Decoded bits for token {actual_token}: {decoded_bits} (range: [{range_bottom}, {range_top}))")

    return indices[selection].item(), decoded_bits

def _decode_single_patch(
        model: torch.nn.Module,
        reference_indices: torch.Tensor,
        building_indices: torch.Tensor,
        current_row: int,
        current_col: int,
        grid_shape: Tuple[int, int],
        image_reshaped: torch.Tensor
) -> Tuple[int, int, str, int]:
    """
    Decodes a single patch position and advances to the next position.

    Args:
        model: The transformer model for prediction.
        reference_indices: Reference codebook indices tensor.
        building_indices: Building codebook indices tensor (modified in-place).
        current_row: Current row position in the patch grid.
        current_col: Current column position in the patch grid.
        grid_shape: Grid dimensions (height, width) in patches.
        image_reshaped: The reshaped stego-image tensor.

    Returns:
        Tuple[int, int, str, int]: (new_row, new_col, decoded_bits, selected_idx)
    """

    context, (local_row, local_col) = build_context_from_patches(
        reference_indices, building_indices, current_row, current_col, grid_shape
    )

    actual_token = image_reshaped[current_row, current_col].item()

    selected_codebook_idx, decoded_bits = decode_next_patch(
        model, context, actual_token, (local_row, local_col),
    )
        
    if selected_codebook_idx < 0:
        selected_codebook_idx = actual_token  # Fallback to actual token if decoding fails
        
    building_indices[current_row, current_col] = selected_codebook_idx

    new_col = current_col + 1
    new_row = current_row

    if new_col >= grid_shape[1]:
        new_col = 0
        new_row += 1

    return new_row, new_col, decoded_bits, selected_codebook_idx


def decode_message(
        options: Options,
        model: torch.nn.Module,
        dsets: DataModuleFromConfig,
        image: torch.Tensor
) -> Tuple[str, str, List[int]]:
    # FIX: Do not overwrite 'image'. Use '_' for the first return value.
    base_tensor, cond_tensor = set_context(model, dsets, DEFAULT_CONTEXT_ROWS)

    # Now, model.encode_to_z actually processes your stego-image!
    image_translations, image_indices = model.encode_to_z(base_tensor.unsqueeze(0))
    cond_translations, cond_indices = model.encode_to_c(cond_tensor)

    grid_shape = (image_translations.shape[2], image_translations.shape[3])
    reference_tensor = cond_indices.reshape(
        cond_translations.shape[0], cond_translations.shape[2], cond_translations.shape[3]
    ).squeeze()

    half_start = math.floor(image_indices.shape[1] * options.context_fraction)

    # Reshape the actual stego-image indices into our grid
    building_tensor = image_indices
    building_tensor[:, half_start:] = 0
    building_tensor = building_tensor.reshape(grid_shape) # hw

    current_row = half_start // grid_shape[1]
    current_col = half_start % grid_shape[1]

    # Decoding message bits
    total_remaining = (grid_shape[0] - current_row) * grid_shape[1] - current_col
    pbar = tqdm(total=total_remaining, desc="Decoding message", disable=True)
    message_bits = ""   
    selected_indices = []

    with pbar:
        while current_row < grid_shape[0]:
            logger.info(f"Decoding patch at row {current_row}, col {current_col}")
            current_row, current_col, decoded_bits, selected_idx = _decode_single_patch(
                model, reference_tensor, building_tensor,
                current_row, current_col, grid_shape, model.encode_to_z(image.unsqueeze(0))[1].reshape(grid_shape)
            )
            if selected_idx != -1:
                selected_indices.append(selected_idx)
            if current_row == 10:
                return bits2string(message_bits), message_bits, selected_indices
            message_bits += decoded_bits
            pbar.update(1)
            
    return bits2string(message_bits), message_bits, selected_indices