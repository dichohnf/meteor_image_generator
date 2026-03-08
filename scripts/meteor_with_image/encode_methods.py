import math
import os
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm, trange

from main import DataModuleFromConfig
from scripts.meteor_with_image.input import Options
from scripts.meteor_with_image.utils import bits2int, int2bits, count_matching_bits_from_start, local_indexes, \
    reset_seeds, save_image, string2bits
from scripts.meteor_with_image.utils import PATCH_SIZE, DEFAULT_CODEBOOK_SIZE, DEFAULT_PRECISION_BITS, DEFAULT_CONTEXT_ROWS


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
        raise ValueError("num_rows must be not None")

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

    if random_sample:
        selection = torch.multinomial(probs, 1).item()
        return indices[selection].item(), 10

    # Apply probability threshold
    prob_threshold = 1 / codebook_len
    k = min(max(2, torch.nonzero(probs < prob_threshold)[0].item()), top_k)
    probs_int = probs[:k]

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

def build_context_from_patches(
        reference_indices: torch.Tensor,
        building_indices: torch.Tensor,
        current_row: int,
        current_col: int,
        grid_shape: Tuple[int, int]
) -> Tuple[Tensor, Tuple[int, int]]:
    """
    Builds a context tensor by concatenating reference and current patch sequences.

    Args:
        reference_indices: Reference codebook indices from context image.
        building_indices: Current generated codebook indices.
        current_row: Current row position in the patch grid.
        current_col: Current column position in the patch grid.
        grid_shape: Shape of the patch translation.

    Returns:
        Tuple[Tensor, Tuple[int, int]]: (context_tensor, (local_row, local_col))
    """
    local_row, row_start, row_end = local_indexes(current_row, grid_shape[0])
    local_col, col_start, col_end = local_indexes(current_col, grid_shape[1])

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
        reference_indices, building_indices, current_row, current_col, grid_shape
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


def encode_message_to_image(
        message: str,
        model: torch.nn.Module,
        dsets : DataModuleFromConfig,
        random_sample: bool,
        context_fraction : float
    ) -> torch.Tensor:
    """
    Encodes a text message into a generated image using steganography.

    Args:
        message: The text message to encode into the image.
        model: The transformer model for prediction.
        dsets: Dataset object containing the reference image.
        # quiet (Optional): Impose to remove all the console outputs.
        random_sample (Optional): If True, samples randomly without encoding.

    Returns:
        torch.Tensor: A generated image tensor (C, H, W) containing the encoded message.
    """
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
    building_tensor = building_tensor.reshape(grid_shape) # hw

    current_row = half_start // grid_shape[1]
    current_col = half_start % grid_shape[1]

    message_bits = string2bits(message)
    remaining_bits = message_bits

    # Encode message bits
    pbar = tqdm(total=len(remaining_bits), desc="Encoding message", disable=True)
    with pbar:
        while remaining_bits:
            next_bits = remaining_bits[:DEFAULT_PRECISION_BITS]

            current_row, current_col, _, encoded_len = _encode_single_patch(
                model, reference_tensor, building_tensor,
                current_row, current_col, grid_shape,
                bits_to_encode=next_bits, random_sample=random_sample
            )

            remaining_bits = remaining_bits[encoded_len:]
            pbar.update(encoded_len)

            if current_row >= grid_shape[0]:
                break

    # Fill remaining patches with random samples
    if current_row < grid_shape[0]:
        total_remaining = (grid_shape[0] - current_row) * grid_shape[1] - current_col
        pbar = tqdm(total=total_remaining, desc="Filling remaining patches", disable=True)
        with pbar:
            while current_row < grid_shape[0]:
                current_row, current_col, _, _ = _encode_single_patch(
                    model, reference_tensor, building_tensor,
                    current_row, current_col, grid_shape,
                    random_sample=True
                )
                pbar.update(1)

    # Decode to image
    image = model.decode_to_img(
        building_tensor[:grid_shape[0], :grid_shape[1]].unsqueeze(0),
        image_translations.shape,
    ).squeeze()

    return image

def images_generation(
        options: Options,
        model: torch.nn.Module,
        dsets: DataModuleFromConfig,
        random_generation: bool
) -> None:
    reset_seeds(options.seed)
    for i in trange(options.to_gen_number, position=0, leave=True, desc="Random generation" if random_generation else "Meteor generation", disable=options.quiet):
        generated_image = encode_message_to_image(
            options.message,
            model, dsets,
            random_sample=True,
            context_fraction=options.context_fraction
        )
        path = os.path.join(options.output_directory, "random" if random_generation else "meteor", f"rand_{i:03}")
        save_image(generated_image, path)