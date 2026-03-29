import math
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from main import DataModuleFromConfig
from scripts.input import Options
from scripts.utils import bits2int, build_context_from_patches, int2bits, count_matching_bits_from_start, \
    reset_seeds, save_image, string2bits, set_context
from scripts.utils import PATCH_SIZE, DEFAULT_CODEBOOK_SIZE, DEFAULT_PRECISION_BITS, DEFAULT_CONTEXT_ROWS
from scripts.logger import logger
from scripts.stats import EncodingStatistics


class SteganographyEncoder:
    """
    Handles the encoding of secret messages into images using arithmetic coding and VQGAN.
    This class manages the process of selecting appropriate codebook tokens for each image patch
    to embed message bits while maintaining visual fidelity.
    """

    def __init__(self, model: torch.nn.Module, dsets: DataModuleFromConfig, context_fraction: float = DEFAULT_CONTEXT_ROWS):
        """
        Initializes the encoder with the VQGAN model, dataset, and context fraction.

        Args:
            model: The trained VQGAN transformer model for generating image patches.
            dsets: The dataset configuration containing reference images.
            context_fraction: Fraction of the image used as context for generation.
        """
        self.model = model
        self.dsets = dsets
        self.context_fraction = context_fraction

    @torch.no_grad()
    def select_token_for_patch(
            self,
            context: torch.Tensor,
            message_bits: str,
            position: Tuple[int, int],
            *,
            codebook_len: int = DEFAULT_CODEBOOK_SIZE,
            random_sample: bool = False,
            top_k: Optional[int] = None,
    ) -> Tuple[int, int, int, int, str]:
        """
        Selects the optimal codebook token for a specific patch position to encode message bits
        using arithmetic coding. If random sampling is enabled, selects randomly without encoding.

        Args:
            context: Tensor representing the context from previous patches.
            message_bits: Binary string of bits to encode in this patch.
            position: (row, col) position within the patch grid.
            codebook_len: Size of the codebook (vocabulary size).
            random_sample: If True, samples randomly without embedding message.
            top_k: Maximum number of top probable tokens to consider.

        Returns:
            Tuple of (selected_codebook_index, number_of_bits_encoded, range_bottom, range_top, encoded_bits_binary).
        """
        top_k = top_k or codebook_len
        self.model.eval()

        logits, _ = self.model.transformer(context[:-1].unsqueeze(0))
        logits = logits[:, -256:, :].squeeze()
        logits = logits.reshape(PATCH_SIZE, PATCH_SIZE, -1)
        logits = logits[position[0], position[1], :].double()

        logits, indices = logits.sort(descending=True)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        if random_sample:
            selected = torch.multinomial(probs, 1).item()
            encoded_bits_str = format(0, f'0{DEFAULT_PRECISION_BITS}b')
            # For random sampling, use full range [0, codebook_len)
            return indices[selected].item(), DEFAULT_PRECISION_BITS, 0, codebook_len, encoded_bits_str

        prob_threshold = 1 / codebook_len
        k = min(max(2, torch.nonzero(probs < prob_threshold)[0].item()), top_k)
        probs_int = probs[:k]

        probs_int = (probs_int / probs_int.sum() * codebook_len).round().long()
        cumulative_probs = probs_int.cumsum(0)

        overfill_indices = torch.nonzero(cumulative_probs > codebook_len)
        if len(overfill_indices) > 0:
            cumulative_probs = cumulative_probs[:overfill_indices[0]]
        cumulative_probs += codebook_len - cumulative_probs[-1]

        message_str = (message_bits + "0" * DEFAULT_PRECISION_BITS)[:DEFAULT_PRECISION_BITS]
        message_value = bits2int(message_str)
        selection = torch.nonzero(cumulative_probs > message_value)[0].item()

        range_bottom = cumulative_probs[selection - 1].item() if selection > 0 else 0
        range_top = cumulative_probs[selection].item()

        bottom_bits = int2bits(range_bottom, DEFAULT_PRECISION_BITS)
        top_bits = int2bits(range_top - 1, DEFAULT_PRECISION_BITS)

        encoded_bits = count_matching_bits_from_start(bottom_bits, top_bits)
        encoded_bits_str = bottom_bits[:encoded_bits]

        return indices[selection].item(), encoded_bits, range_bottom, range_top, encoded_bits_str

    @torch.no_grad()
    def encode_patch(
            self,
            reference_indices: torch.Tensor,
            building_indices: torch.Tensor,
            current_row: int,
            current_col: int,
            grid_shape: Tuple[int, int],
            image_translations_shape: Tuple[int, int, int, int],
            bits_to_encode: str = "",
            random_sample: bool = False,
    ) -> Tuple[int, int, int, int, int, int, str]:
        """
        Encodes a single patch at the given position by selecting an appropriate token
        and updating the building indices tensor. Advances to the next patch position.

        Args:
            reference_indices: Reference codebook indices from the context image.
            building_indices: Current indices being built for the output image.
            current_row: Current row in the patch grid.
            current_col: Current column in the patch grid.
            grid_shape: Dimensions of the patch grid (height, width).
            image_translations_shape: Shape of the image translations tensor for decoding.
            bits_to_encode: Binary string to embed in this patch.
            random_sample: If True, selects token randomly without encoding.

        Returns:
            Tuple of (next_row, next_col, selected_index, bits_encoded, range_bottom, range_top, encoded_bits_str).
        """
        context, (local_row, local_col) = build_context_from_patches(
            reference_indices, building_indices, current_row, current_col, grid_shape
        )

        count = 0
        while True:
            count += 1
            if count > 100:  # Prevent infinite loops
                logger.warning(f"Exceeded maximum attempts to encode patch at ({current_row}, {current_col}). Proceeding with last selection.")
                selected_idx = building_indices[current_row, current_col].item()
                encoded_length = 0
                range_bottom, range_top = 0, DEFAULT_CODEBOOK_SIZE
                encoded_bits_str = ""
                break

            selected_idx, encoded_length, range_bottom, range_top, encoded_bits_str = self.select_token_for_patch(
                context,
                bits_to_encode,
                (local_row, local_col),
                random_sample=random_sample,
            )
            building_indices[current_row, current_col] = selected_idx

            exit_condition = random_sample or \
                self.model.encode_to_z(
                    self.model.decode_to_img(
                        building_indices.unsqueeze(0),
                        image_translations_shape)
                )[1].squeeze().reshape(grid_shape)[current_row, current_col].item() \
                == selected_idx
        
            if exit_condition:
                break

        next_col = current_col + 1
        next_row = current_row
        if next_col >= grid_shape[1]:
            next_col = 0
            next_row += 1

        return next_row, next_col, selected_idx, encoded_length, range_bottom, range_top, encoded_bits_str

    @torch.no_grad()
    def encode_message_to_image(self, message: str, random_sample: bool = False) -> Tuple[torch.Tensor, str, List[int], torch.Tensor, EncodingStatistics]:
        """
        Encodes a text message into a generated image tensor using steganography.
        Processes the message bit by bit, embedding it into the image patches.
        Also collects and returns detailed statistics about the encoding process.

        Args:
            message: The text message to hide in the image.
            random_sample: If True, generates a random image without embedding the message.

        Returns:
            Tuple of (generated_image, encoded_bits_string, indices_sequence, building_tensor, encoding_statistics).
        """
        source_image, cond_tensor = set_context(self.model, self.dsets, DEFAULT_CONTEXT_ROWS)
        image_translations, image_indices = self.model.encode_to_z(source_image.unsqueeze(0))
        cond_translations, cond_indices = self.model.encode_to_c(cond_tensor)

        grid_shape = (image_translations.shape[2], image_translations.shape[3])
        reference_tensor = cond_indices.reshape(cond_translations.shape[0], cond_translations.shape[2], cond_translations.shape[3]).squeeze()

        half_start = math.floor(image_indices.shape[1] * self.context_fraction)
        building_tensor = image_indices.clone()
        building_tensor[:, half_start:] = 0
        building_tensor = building_tensor.reshape(grid_shape)

        current_row = half_start // grid_shape[1]
        current_col = half_start % grid_shape[1]

        remaining_bits = string2bits(message)
        indices_sequence = []
        stats = EncodingStatistics()
        patch_index = 0

        with tqdm(total=len(remaining_bits), desc="Encoding message", disable=True) as progress:
            while remaining_bits and current_row < grid_shape[0]:
                feed_bits = remaining_bits[:DEFAULT_PRECISION_BITS]
                # Record position before encoding
                patch_row, patch_col = current_row, current_col
                current_row, current_col, selected_idx, used_bits, range_bottom, range_top, encoded_bits_str = self.encode_patch(
                    reference_tensor,
                    building_tensor,
                    current_row,
                    current_col,
                    grid_shape,
                    image_translations.shape,
                    bits_to_encode=feed_bits,
                    random_sample=random_sample,
                )
                        
                indices_sequence.append(selected_idx)
                stats.add_patch_stat(patch_index, patch_row, patch_col, selected_idx, range_bottom, range_top, encoded_bits_str, used_bits)
                remaining_bits = remaining_bits[used_bits:]
                progress.update(used_bits)
                patch_index += 1

        if current_row < grid_shape[0]:
            total_remaining = (grid_shape[0] - current_row) * grid_shape[1] - current_col
            with tqdm(total=total_remaining, desc="Filling remaining patches", disable=True) as fill_progress:
                while current_row < grid_shape[0]:
                    patch_row, patch_col = current_row, current_col
                    current_row, current_col, selected_idx, _, range_bottom, range_top, encoded_bits_str = self.encode_patch(
                        reference_tensor,
                        building_tensor,
                        current_row,
                        current_col,
                        grid_shape,
                        image_translations.shape,
                        random_sample=True,
                    )
                    indices_sequence.append(selected_idx)
                    stats.add_patch_stat(patch_index, patch_row, patch_col, selected_idx, range_bottom, range_top, encoded_bits_str, 0)
                    fill_progress.update(1)
                    patch_index += 1

        output_image = self.model.decode_to_img(building_tensor.unsqueeze(0), image_translations.shape).squeeze()
        return output_image, string2bits(message), indices_sequence, building_tensor, stats


def encode_message_to_image(
        message: str,
        model: torch.nn.Module,
        dsets: DataModuleFromConfig,
        random_sample: bool,
        context_fraction: float
) -> Tuple[torch.Tensor, str, List[int], torch.Tensor]:
    """
    Convenience function to encode a message into an image using the SteganographyEncoder class.

    Args:
        message: The message to encode.
        model: The VQGAN model.
        dsets: The dataset configuration.
        random_sample: Whether to sample randomly.
        context_fraction: Fraction of context to use.

    Returns:
        Tuple of (image, bits, indices, tensor).
    """
    encoder = SteganographyEncoder(model, dsets, context_fraction=context_fraction)
    return encoder.encode_message_to_image(message, random_sample=random_sample)