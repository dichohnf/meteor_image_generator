import math

import numpy as np
import torch
from typing import List, Tuple, Optional
from torchvision.io import read_image
from tqdm.auto import tqdm

from main import DataModuleFromConfig
from scripts.input import Options
from scripts.utils import (DEFAULT_CONTEXT_ROWS, set_context, bits2string, int2bits, build_context_from_patches,
                           DEFAULT_PRECISION_BITS, DEFAULT_CODEBOOK_SIZE, PATCH_SIZE)
from scripts.logger import logger
from scripts.stats import DecodingStatistics


def load_image(path: str) -> torch.Tensor:
    img = read_image(path)
    img = (img.type(torch.float32) / 127.5) - 1.0
    img = img.to('cuda')
    return img


class SteganographyDecoder:
    """
    Handles the decoding of hidden messages from steganographic images using arithmetic coding.
    This class extracts embedded bits from image patches and reconstructs the original message.
    """

    def __init__(self, model: torch.nn.Module, dsets: DataModuleFromConfig, context_fraction: float = DEFAULT_CONTEXT_ROWS):
        """
        Initializes the decoder with the VQGAN model, dataset, and context fraction.

        Args:
            model: The trained VQGAN transformer model for analyzing image patches.
            dsets: The dataset configuration containing reference images.
            context_fraction: Fraction of the image used as context for decoding.
        """
        self.model = model
        self.dsets = dsets
        self.context_fraction = context_fraction

    @torch.no_grad()
    def decode_token_from_patch(
            self,
            context: torch.Tensor,
            actual_token: int,
            position: Tuple[int, int],
            *,
            codebook_len: int = DEFAULT_CODEBOOK_SIZE,
            top_k: Optional[int] = None,
    ) -> Tuple[int, str, int, int]:
        """
        Decodes the hidden bits from a specific token in a patch by analyzing the probability distribution.

        Args:
            context: Tensor representing the context from previous patches.
            actual_token: The codebook index found in the stego-image.
            position: (row, col) position within the patch grid.
            codebook_len: Size of the codebook.
            top_k: Maximum number of top probable tokens to consider.

        Returns:
            Tuple of (selected_codebook_index, decoded_bits_string, range_bottom, range_top).
        """
        top_k = top_k or codebook_len
        self.model.eval()

        logits, _ = self.model.transformer(context[:-1].unsqueeze(0))
        logits = logits[:, -256:, :].squeeze()
        logits = logits.reshape(PATCH_SIZE, PATCH_SIZE, -1)
        logits = logits[position[0], position[1], :].double()

        logits, indices = logits.sort(descending=True)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        prob_threshold = 1 / codebook_len
        k = min(max(2, torch.nonzero(probs < prob_threshold)[0].item()), top_k)
        probs_int = probs[:k]

        probs_int = (probs_int / probs_int.sum() * codebook_len).round().long()
        cumulative_probs = probs_int.cumsum(0)

        overfill_index = torch.nonzero(cumulative_probs > codebook_len)
        if len(overfill_index) > 0:
            cumulative_probs = cumulative_probs[:overfill_index[0]]
        cumulative_probs += codebook_len - cumulative_probs[-1]

        matching_indices = torch.nonzero(indices[:k] == actual_token).squeeze()

        if matching_indices.numel() == 0:
            logger.warning(f"Actual token {actual_token} is not in the top_k distribution.")
            return -1, "", 0, 0

        if matching_indices.dim() == 0:
            selection = matching_indices.item()
        else:
            logger.warning(f"Multiple matches for actual_token {actual_token}.")
            return -1, "", 0, 0

        if selection >= len(cumulative_probs) or selection < 0:
            logger.warning(f"Selection out of range: {selection}.")
            return -1, "", 0, 0

        range_bottom = cumulative_probs[selection - 1].item() if selection > 0 else 0
        range_top = cumulative_probs[selection].item()

        bottom_bits = int2bits(range_bottom, DEFAULT_PRECISION_BITS)
        top_bits = int2bits(range_top - 1, DEFAULT_PRECISION_BITS)

        decoded_bits = ""
        for b, t in zip(bottom_bits, top_bits):
            if b == t:
                decoded_bits += str(b)
            else:
                break

        logger.info(f"Decoded bits for token {actual_token}: {decoded_bits} range=[{range_bottom},{range_top})")

        return indices[selection].item(), decoded_bits, range_bottom, range_top

    @torch.no_grad()
    def decode_patch(
            self,
            reference_indices: torch.Tensor,
            building_indices: torch.Tensor,
            current_row: int,
            current_col: int,
            grid_shape: Tuple[int, int],
            stego_indices: torch.Tensor,
    ) -> Tuple[int, int, str, int, int, int]:
        """
        Decodes a single patch from the stego-image and advances to the next position.

        Args:
            reference_indices: Reference codebook indices.
            building_indices: Indices being reconstructed.
            current_row: Current row in the grid.
            current_col: Current column in the grid.
            grid_shape: Grid dimensions.
            stego_indices: Indices from the stego-image.

        Returns:
            Tuple of (next_row, next_col, decoded_bits, selected_index, range_bottom, range_top).
        """
        context, (local_row, local_col) = build_context_from_patches(
            reference_indices, building_indices, current_row, current_col, grid_shape
        )

        actual_token = stego_indices[current_row, current_col].item()
        selected_idx, decoded_bits, range_bottom, range_top = self.decode_token_from_patch(
            context,
            actual_token,
            (local_row, local_col)
        )

        if selected_idx < 0:
            selected_idx = actual_token

        building_indices[current_row, current_col] = selected_idx

        next_col = current_col + 1
        next_row = current_row
        if next_col >= grid_shape[1]:
            next_col = 0
            next_row += 1

        return next_row, next_col, decoded_bits, selected_idx, range_bottom, range_top

    @torch.no_grad()
    def decode_message(
            self,
            options: Options,
            image: torch.Tensor
    ) -> Tuple[str, str, List[int], DecodingStatistics]:
        """
        Decodes the hidden message from a steganographic image tensor.
        Also collects and returns detailed statistics about the decoding process.

        Args:
            options: Configuration options.
            image: The stego-image tensor to decode.

        Returns:
            Tuple of (decoded_text, decoded_bits, selected_indices, decoding_statistics).
        """
        base_tensor, cond_tensor = set_context(self.model, self.dsets, DEFAULT_CONTEXT_ROWS)

        image_translations, image_indices = self.model.encode_to_z(base_tensor.unsqueeze(0))
        cond_translations, cond_indices = self.model.encode_to_c(cond_tensor)

        grid_shape = (image_translations.shape[2], image_translations.shape[3])
        reference_tensor = cond_indices.reshape(cond_translations.shape[0], cond_translations.shape[2], cond_translations.shape[3]).squeeze()

        half_start = math.floor(image_indices.shape[1] * self.context_fraction)
        building_tensor = image_indices.clone()
        building_tensor[:, half_start:] = 0
        building_tensor = building_tensor.reshape(grid_shape)

        current_row = half_start // grid_shape[1]
        current_col = half_start % grid_shape[1]

        height_crop = grid_shape[0] * PATCH_SIZE
        width_crop = grid_shape[1] * PATCH_SIZE
        image = image[:, :height_crop, :width_crop]

        stego_indices = self.model.encode_to_z(image.unsqueeze(0))[1].reshape(grid_shape)

        total_remaining = (grid_shape[0] - current_row) * grid_shape[1] - current_col
        with tqdm(total=total_remaining, desc="Decoding message", disable=True) as pbar:
            decoded_bits = ""
            selected_indices = []
            stats = DecodingStatistics()
            patch_index = 0

            while current_row < grid_shape[0]:
                logger.info(f"Decoding patch row={current_row} col={current_col}")
                current_row, current_col, token_bits, selected_idx, range_bottom, range_top = self.decode_patch(
                    reference_tensor,
                    building_tensor,
                    current_row,
                    current_col,
                    grid_shape,
                    stego_indices
                )
                prev_row = current_row - (1 if current_col == 0 else 0)
                prev_col = (current_col - 1) % grid_shape[1] if current_col > 0 else grid_shape[1] - 1
                actual_token = stego_indices[prev_row, prev_col].item()
                success = selected_idx != -1
                if success:
                    selected_indices.append(selected_idx)
                stats.add_patch_stat(patch_index, prev_row, prev_col, actual_token, range_bottom, range_top, token_bits, success)
                decoded_bits += token_bits
                pbar.update(1)
                patch_index += 1

        return bits2string(decoded_bits), decoded_bits, selected_indices, stats


def decode_message(
        options: Options,
        model: torch.nn.Module,
        dsets: DataModuleFromConfig,
        image: torch.Tensor
) -> Tuple[str, str, List[int]]:
    """
    Convenience function to decode a message from an image using the SteganographyDecoder class.

    Args:
        options: Configuration options.
        model: The VQGAN model.
        dsets: The dataset configuration.
        image: The image to decode.

    Returns:
        Tuple of (decoded_text, decoded_bits, indices).
    """
    decoder = SteganographyDecoder(model, dsets, context_fraction=options.context_fraction)
    return decoder.decode_message(options, image)
