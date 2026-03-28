import os
import shutil

import numpy as np
import torch

from scripts.logger import logger
from scripts.decode_methods import decode_message, load_image
from scripts.encode_methods import images_generation
from scripts.input import Options, initialized_parser
from scripts.utils import get_vqgan_sflckr, reset_seeds


def main():
    """Main entry point for message encoding demonstration."""

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
    os.makedirs(options.output_directory)

    options.save_as_file()
    logger.enable_set(not options.quiet)

    logger.info("Options saved to file.")
    logger.info("SETTING UP VQGAN MODEL AND DATASETS...")
    dsets, model = get_vqgan_sflckr(options.model_directory_path)
    logger.info("VQGAN MODEL AND DATASETS SETUP COMPLETED.")

    logger.info("Starting image generation...")
    if options.random_generation:
        logger.info("Random generation enabled. Generating random images without encoding messages.")
        images_generation(options, model, dsets, True)
        
    logger.info("Starting message encoding into images...")
    images_generation(options, model, dsets, False)

    meteor_path = os.path.join(options.output_directory, "meteor")
    if not os.path.exists(meteor_path) or not os.path.isdir(meteor_path):
        logger.error(f"Meteor path '{meteor_path}' does not exist or is not a directory.")
        return

    reset_seeds(options.seed)
    logger.info("Starting message decoding from generated images...")
    for root, _, filenames in os.walk(meteor_path):
        for filename in filenames:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image_path = os.path.join(root, filename)
            image = load_image(image_path)
            # arr = image.detach().cpu().numpy()
            _, bits_string, selected_indices = decode_message(options, model, dsets, image)
            
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            with open(root + "/" + image_name + "_decoded_bits.txt", "x") as f:
                f.write(bits_string)
            with open(root + "/" + image_name + "_decoded_indices.txt", "x") as f:
                f.write(np.array2string(np.array(selected_indices), suppress_small=True))
                
if __name__ == '__main__':
    torch.set_printoptions(profile="full")
    main()