import os
import shutil

import numpy as np
import torch

system_path = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.dirname(system_path)
if scripts_path not in os.sys.path:
    os.sys.path.append(scripts_path)

from scripts.logger import logger
from scripts.decoder import SteganographyDecoder, load_image
from scripts.encoder import SteganographyEncoder
from scripts.input import Options, initialized_parser
from scripts.utils import get_vqgan_sflckr, reset_seeds, save_image
from scripts.stats import StatsWriter
from scripts.pipeline import build_pipeline_from_options


def main():
    """
    Orchestrates the steganography workflow: sets up the model, encodes messages into images,
    and decodes them back to verify the process. Handles both random image generation and
    message embedding based on user options.

    The pipeline (XOR → char encoding → error correction) is applied externally from the
    VQGAN encoder/decoder, which operate on raw bits only.
    """
    parser = initialized_parser()
    args = parser.parse_args()

    options = Options(
        args.message,
        args.model_directory,
        args.context_fraction,
        quiet=args.quiet,
        to_gen_number=args.to_gen_number,
        output_directory_path=args.output_directory,
        relative_options_file_path=args.relative_options_file,
        seed=args.seed,
        random_generation=args.random_generation,
        max_error_ratio=args.max_error_ratio,
        burst_error_tolerance=args.burst_error_tolerance,
        error_correction_method=args.error_correction_method,
        rs_nsym=args.rs_nsym,
    )

    if os.path.exists(options.output_directory):
        shutil.rmtree(options.output_directory)
    os.makedirs(options.output_directory, exist_ok=True)

    options.save_as_file()
    logger.enable_set(not options.quiet)

    logger.info("Options saved to file.")
    logger.info("SETTING UP VQGAN MODEL AND DATASETS...")
    dsets, model = get_vqgan_sflckr(options.model_directory_path)
    logger.info("VQGAN MODEL AND DATASETS SETUP COMPLETED.")

    # Build the string-transformation pipeline from CLI arguments
    pipeline = build_pipeline_from_options(options)
    logger.info(f"SteganoPipeline initialized: {pipeline}")

    # Encoder/decoder now operate on raw bits — no error_correction argument
    encoder = SteganographyEncoder(model, dsets, context_fraction=options.context_fraction)
    decoder = SteganographyDecoder(model, dsets, context_fraction=options.context_fraction)

    # Pre-compute the protected bit string once for all encoding runs
    protected_bits = pipeline.encode_message(options.message)
    logger.info(
        f"Protected bits prepared: {len(protected_bits)} bits "
        f"(from {len(options.message)} chars)"
    )

    # ============================================================
    #  Random generation (no message embedded)
    # ============================================================
    reset_seeds(options.seed)
    if options.random_generation:
        logger.info("Random generation enabled. Generating random images without encoding messages.")
        for i in range(options.to_gen_number):
            generated_image, bits_string, indices_sequence, _, encoding_stats = encoder.encode_message_to_image(
                protected_bits, random_sample=True
            )
            path = os.path.join(options.output_directory, "random", f"rand_{i:03}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(generated_image, path)
            with open(path + "_encoded_bits.txt", "x") as f:
                f.write(bits_string)
            with open(path + "_encoded_indices.txt", "x") as f:
                f.write(np.array2string(np.array(indices_sequence), suppress_small=True))
            stats_file = path + "_stats.json"
            StatsWriter.write_encoding_only(stats_file, options.message, bits_string, encoding_stats)

    # ============================================================
    #  Encode message into images (deterministic, seed-controlled)
    # ============================================================
    reset_seeds(options.seed)
    logger.info("Starting message encoding into images...")
    for i in range(options.to_gen_number):
        generated_image, bits_string, indices_sequence, building_tensor, encoding_stats = encoder.encode_message_to_image(
            protected_bits, random_sample=False
        )
        path = os.path.join(options.output_directory, "meteor", f"rand_{i:03}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_image(generated_image, path)
        with open(path + "_encoded_bits.txt", "x") as f:
            f.write(bits_string)
        with open(path + "_encoded_indices.txt", "x") as f:
            f.write(np.array2string(np.array(indices_sequence), suppress_small=True))
        stats_file = path + "_stats.json"
        StatsWriter.write_encoding_only(stats_file, options.message, bits_string, encoding_stats)

    # ============================================================
    #  Decode all generated images and recover the message
    # ============================================================
    meteor_path = os.path.join(options.output_directory, "meteor")
    if not os.path.isdir(meteor_path):
        logger.error(f"Meteor path '{meteor_path}' does not exist or is not a directory.")
        return

    reset_seeds(options.seed)
    logger.info("Starting message decoding from generated images...")
    for root, _, files in os.walk(meteor_path):
        for filename in files:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(root, filename)
            image = load_image(image_path)

            # Decoder returns raw bits — run them through the pipeline to recover the string
            raw_bits, _, decoded_indices, decoding_stats = decoder.decode_message(options, image)
            recovered_text = pipeline.decode_message(raw_bits)

            image_name, _ = os.path.splitext(filename)
            with open(os.path.join(root, f"{image_name}_decoded_bits.txt"), "x") as f:
                f.write(raw_bits)
            with open(os.path.join(root, f"{image_name}_decoded_indices.txt"), "x") as f:
                f.write(np.array2string(np.array(decoded_indices), suppress_small=True))

            # Append decoding stats to the existing stats file
            stats_file = image_path.rsplit(".", 1)[0] + "_stats.json"
            StatsWriter.append_decoding_stats(
                stats_file, recovered_text, raw_bits, decoding_stats, options.message
            )
            with open(os.path.join(root, f"{image_name}_decoded_text.txt"), "x") as f:
                f.write(recovered_text)


if __name__ == '__main__':
    main()