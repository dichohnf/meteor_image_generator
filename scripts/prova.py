import os

from scripts.meteor_with_image.decode_methods import load_image, decode_message
from scripts.meteor_with_image.encode_methods import images_generation
from scripts.meteor_with_image.input import initialized_parser, Options
from scripts.meteor_with_image.utils import reset_seeds, get_vqgan_sflckr

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
        os.rmdir(options.output_directory)
    os.makedirs(options.output_directory)

    options.save_as_file()

    if not options.quiet:
        print("=" * 30, "Setting up the model", "=" * 30, flush=True)
    dsets, model = get_vqgan_sflckr(options.model_directory_path)

    if options.random_generation:
        images_generation(options, model, dsets, True)

    images_generation(options, model, dsets, False)

    meteor_path = os.path.join(options.output_directory, "meteor")
    if not os.path.exists(meteor_path) or not os.path.isdir(meteor_path):
        raise Exception("No meteor generated image founded")

    reset_seeds(options.seed)
    for _,_,image_path in os.walk(meteor_path):
        image = load_image(image_path)
        decoded_message = decode_message(options, model, dsets, image)


if __name__ == '__main__':
    main()
