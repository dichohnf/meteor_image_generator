import os
import shutil

import numpy as np
import torch

system_path = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.dirname(system_path)
if scripts_path not in os.sys.path:
    os.sys.path.append(scripts_path)

from scripts.logger import logger
from scripts.decode_methods import SteganographyDecoder, load_image
from scripts.encode_methods import SteganographyEncoder
from scripts.input import Options, initialized_parser
from scripts.utils import get_vqgan_sflckr, reset_seeds, save_image
from scripts.stats import StatsWriter


def main():
    """
    Orchestrates the steganography workflow: sets up the model, encodes messages into images,
    and decodes them back to verify the process. Handles both random image generation and
    message embedding based on user options.
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

    encoder = SteganographyEncoder(model, dsets, context_fraction=options.context_fraction)
    decoder = SteganographyDecoder(model, dsets, context_fraction=options.context_fraction)

    reset_seeds(options.seed)
    if options.random_generation:
        logger.info("Random generation enabled. Generating random images without encoding messages.")
        for i in range(options.to_gen_number):
            generated_image, bits_string, indices_sequence, _, encoding_stats = encoder.encode_message_to_image(options.message, random_sample=True)
            path = os.path.join(options.output_directory, "random", f"rand_{i:03}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(generated_image, path)
            with open(path + "_encoded_bits.txt", "x") as f:
                f.write(bits_string)
            with open(path + "_encoded_indices.txt", "x") as f:
                f.write(np.array2string(np.array(indices_sequence), suppress_small=True))
            stats_file = path + "_stats.json"
            StatsWriter.write_encoding_only(stats_file, options.message, bits_string, encoding_stats)

    reset_seeds(options.seed)
    logger.info("Starting message encoding into images...")
    for i in range(options.to_gen_number):
        generated_image, bits_string, indices_sequence, building_tensor, encoding_stats = encoder.encode_message_to_image(options.message, random_sample=False)
        path = os.path.join(options.output_directory, "meteor", f"rand_{i:03}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_image(generated_image, path)
        with open(path + "_encoded_bits.txt", "x") as f:
            f.write(bits_string)
        with open(path + "_encoded_indices.txt", "x") as f:
            f.write(np.array2string(np.array(indices_sequence), suppress_small=True))
        stats_file = path + "_stats.json"
        StatsWriter.write_encoding_only(stats_file, options.message, bits_string, encoding_stats)

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
            decoded_text, decoded_bits, decoded_indices, decoding_stats = decoder.decode_message(options, image)

            image_name, _ = os.path.splitext(filename)
            with open(os.path.join(root, f"{image_name}_decoded_bits.txt"), "x") as f:
                f.write(decoded_bits)
            with open(os.path.join(root, f"{image_name}_decoded_indices.txt"), "x") as f:
                f.write(np.array2string(np.array(decoded_indices), suppress_small=True))
            
            # Append decoding stats to the existing stats file
            stats_file = image_path.rsplit(".", 1)[0] + "_stats.json"
            StatsWriter.append_decoding_stats(stats_file, decoded_text, decoded_bits, decoding_stats, options.message)
            with open(os.path.join(root, f"{image_name}_decoded_text.txt"), "x") as f:
                f.write(decoded_text)


if __name__ == '__main__':
    main()
