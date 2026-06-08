import argparse
import datetime
import json
import os
import random
from argparse import ArgumentError, ArgumentParser
from typing import Optional


def initialized_parser() -> ArgumentParser:
    """
    Creates and configures the command-line argument parser for the steganography application.
    Defines all necessary arguments for message encoding, model configuration, and output settings.

    Returns:
        Configured ArgumentParser instance ready to parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Encode a message into an image using steganography',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('message', help='The message to encrypt within the generated image')
    parser.add_argument('-m', '--model_directory',
                        type=str, required=False, default='logs/2020-11-09T13-31-51_sflckr',
                        help='Directory where are present the subdirectories \'checkpoint\', \'configs\' and \'samples\' of the model, generally under \'logs\'')
    parser.add_argument('-c', '--context_fraction',
                        type=float, required=False, default=0.1,
                        help='Fraction of the contextual image to use as context')
    parser.add_argument('-q', '--quiet',
                        action='store_true', required=False, default=False,
                        help='Impose to remove all the console outputs')
    parser.add_argument('-o', '--output_directory',
                        type=str, required=False, default=None,
                        help='Directory where to save the generated image')
    parser.add_argument('-s', '--seed',
                        type=int, required=False, default=random.randint(1, 10000),
                        help='Seed for the random number generator')
    parser.add_argument('-n', '--to_gen_number',
                        type=int, required=False, default=1,
                        help='Number of generated images to generate')
    parser.add_argument('-r', '--relative_options_file',
                        type=str, required=False, default=None,
                        help='Relative path to the options file')
    # Random generation
    parser.add_argument('--random-generation',
                        action='store_true',
                        help='Whatever the code should generate even images with random sampling (top_k)')
    parser.add_argument('--without-random-generation',
                        dest='random_generation', action='store_false')
    parser.add_argument('--burst-error-tolerance',
                        type=int, required=False, default=10,
                        help='Maximum number of consecutive bit errors the error correction can withstand. '
                             'Default 10. The overhead is burst_error_tolerance * 2 + 1 times the original message size. '
                             'Higher values protect against longer burst errors but add more redundancy.')
    parser.add_argument('--max-error-ratio',
                        type=float, required=False, default=0.2,
                        help='(DEPRECATED) Use --burst-error-tolerance instead. '
                             'Maximum tolerable bit error ratio for error correction (0.0 to 1.0). '
                             'Default 0.2 means up to 20%% of bits can be corrected. '
                             'Higher values add more redundancy but allow more error recovery.')
    parser.add_argument('--error-correction-method',
                        type=str, required=False, default='vote',
                        choices=['vote', 'reed_solomon'],
                        help='Error correction algorithm to use. "vote" uses full-message '
                             'repetition with majority voting (burst-resistant). '
                             '"reed_solomon" uses Reed-Solomon block coding over GF(256). '
                             'Default: "vote".')
    parser.add_argument('--rs-nsym',
                        type=int, required=False, default=10,
                        help='Number of ECC symbols for Reed-Solomon code. '
                             'Can correct up to nsym // 2 erroneous bytes. '
                             'Only used when --error-correction-method=reed_solomon. '
                             'Default: 10.')
    parser.add_argument('--xor-key',
                        type=int, required=False, default=None,
                        help='Optional integer key (0-255) for XOR obfuscation of the '
                             'bit stream. If not set, no XOR masking is applied.')
    parser.add_argument('--char-encoding',
                        type=str, required=False, default='ASCII',
                        choices=['ASCII', 'UNICODE', 'DECIMAL'],
                        help='Character encoding for str↔bits conversion. '
                             'Default: "ASCII".')
    parser.set_defaults(random_generation=False)
    return parser

class Options:
    """
    Configuration class holding all settings for the steganography encoding and decoding process.
    Manages message, model paths, generation parameters, and output configurations.
    """

    def __init__(
        self, message : str,
        model_directory_path : str,
        /,
        context_fraction : float = 1/20.,
        * ,
        quiet : bool = False,
        seed : Optional[int] = None,
        to_gen_number : int = 5,
        output_directory_path : Optional[str] = None,
        relative_options_file_path : Optional[str] = None,
        random_generation : bool = False,
        max_error_ratio : float = 0.2,
        burst_error_tolerance : int = 10,
        error_correction_method : str = "vote",
        rs_nsym : int = 10,
    ) -> None:
        """
        Initializes the Options object with user-specified or default parameters.

        Args:
            message: The secret message to embed in images.
            model_directory_path: Path to the VQGAN model directory.
            context_fraction: Fraction of image used as context.
            quiet: If True, suppresses console output.
            seed: Random seed for reproducibility.
            to_gen_number: Number of images to generate.
            output_directory_path: Directory to save outputs.
            relative_options_file_path: Path for options JSON file.
            random_generation: If True, generates random images without messages.
            max_error_ratio: (DEPRECATED) Maximum tolerable bit error ratio for error correction (0.0 to 1.0).
                             Default 0.2 means up to 20% of bits can be corrected.
            burst_error_tolerance: Maximum number of consecutive bit errors the error correction
                                   can withstand. Default 10.

        Raises:
            ArgumentError: If required parameters are invalid.
        """
        if message is None:
            raise ArgumentError(message, "Message cannot be None")
        self.message = message
        if model_directory_path is None:
            raise ArgumentError(model_directory_path, "Model directory cannot be None")
        self.model_directory_path = model_directory_path

        self.context_fraction = context_fraction

        self.quiet = quiet
        self.seed = seed if seed is not None else random.randint(1, 10000)
        self.to_gen_number = to_gen_number
        self.output_directory = output_directory_path or os.path.join("examples", str(datetime.datetime.now()))
        self.relative_options_file_path = relative_options_file_path or "options.json"
        self.random_generation = random_generation
        self.max_error_ratio = max_error_ratio
        # Convert max_error_ratio to burst_error_tolerance if the legacy parameter is used
        # and burst_error_tolerance wasn't explicitly provided
        self.burst_error_tolerance = burst_error_tolerance
        self.error_correction_method = error_correction_method
        self.rs_nsym = rs_nsym

    def save_as_file(self) -> None:
        """
        Saves the current options configuration to a JSON file in the output directory.
        Creates the output directory if it doesn't exist.
        """
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        path = os.path.join(self.output_directory, self.relative_options_file_path)
        with open(path, "x") as file:
            json.dump(self.__dict__, file, indent=2)