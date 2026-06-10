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
from scripts.stats import StatsWriter, EncodingStatistics, DecodingStatistics
from scripts.pipeline import build_pipeline_from_options, HEADER_LENGTH_BITS


def _options_to_dict(options: Options) -> dict:
    """Serialize relevant Options fields to a plain dict for JSON."""
    return {
        "message": options.message,
        "model_directory_path": options.model_directory_path,
        "context_fraction": options.context_fraction,
        "quiet": options.quiet,
        "seed": options.seed,
        "to_gen_number": options.to_gen_number,
        "output_directory": options.output_directory,
        "random_generation": options.random_generation,
        "max_error_ratio": options.max_error_ratio,
        "burst_error_tolerance": options.burst_error_tolerance,
        "error_correction_method": options.error_correction_method,
        "rs_nsym": options.rs_nsym,
    }


def _pipeline_to_dict(pipeline) -> dict:
    """Serialize pipeline configuration to a plain dict for JSON."""
    info = {
        "char_encoding": pipeline.char_encoding,
        "ecc_method": type(pipeline.ecc).__name__,
        "xor_key": pipeline.xor_mask.key,
        "header_burst_error_tolerance": pipeline.header_burst_error_tolerance,
    }

    # Include ECC-specific parameters
    ecc = pipeline.ecc
    if hasattr(ecc, "burst_error_tolerance"):
        info["burst_error_tolerance"] = ecc.burst_error_tolerance
    if hasattr(ecc, "nsym"):
        info["rs_nsym"] = ecc.nsym

    return info


def main():
    """
    Orchestrates the steganography workflow: sets up the model, encodes messages into images,
    and decodes them back to verify the process. Handles both random image generation and
    message embedding based on user options.

    The pipeline (XOR → char encoding → error correction) is applied externally from the
    VQGAN encoder/decoder, which operate on raw bits only.

    All statistics (bits, indices, per-patch data) are written into a single consolidated
    JSON file per image — no auxiliary .txt files are produced.
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

    # Pre-compute the pipeline trace and protected bits once for all encoding runs
    protected_bits, encode_trace = pipeline.encode_message_with_trace(options.message)
    logger.info(
        f"Protected bits prepared: {len(protected_bits)} bits "
        f"(from {len(options.message)} chars)"
    )

    # Pre-compute metadata dicts (same for all images in this run)
    options_dict = _options_to_dict(options)
    pipeline_info = _pipeline_to_dict(pipeline)

    logger.info(
        f"Protected bits prepared: {len(protected_bits)} bits "
        f"(from {len(options.message)} chars). "
        f"Vote-protected {HEADER_LENGTH_BITS}-bit header with "
        f"tolerance={getattr(options, 'header_burst_error_tolerance', 10)}"
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

            StatsWriter.write_consolidated(
                path + "_stats.json",
                original_message=options.message,
                encoded_bits=bits_string,
                encoded_indices=list(indices_sequence),
                encoding_stats=encoding_stats,
                decoded_bits="",
                decoded_indices=[],
                recovered_text="",
                decoding_stats=DecodingStatistics(),
                encode_trace=encode_trace,
                decode_trace=None,
                options_dict=options_dict,
                pipeline_info=pipeline_info,
                seed=options.seed,
            )

    # ============================================================
    #  Encode message into images (deterministic, seed-controlled)
    #  Store per-image encoding data for later consolidation
    # ============================================================
    reset_seeds(options.seed)
    logger.info("Starting message encoding into images...")
    encoding_data_list = []
    for i in range(options.to_gen_number):
        generated_image, bits_string, indices_sequence, building_tensor, encoding_stats = encoder.encode_message_to_image(
            protected_bits, random_sample=False
        )
        path = os.path.join(options.output_directory, "meteor", f"rand_{i:03}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_image(generated_image, path)
        encoding_data_list.append({
            "bits_string": bits_string,
            "indices_sequence": list(indices_sequence),
            "encoding_stats": encoding_stats,
            "stats_file": path + "_stats.json",
        })

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
        for filename in sorted(files):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(root, filename)
            image = load_image(image_path)

            # Decoder returns (raw_bits, raw_bits, decoded_indices, stats)
            # raw_bits have XOR still applied — pipeline will undo it
            raw_bits, _, decoded_indices, decoding_stats = decoder.decode_message(options, image)
            recovered_text, decode_trace = pipeline.decode_message_with_trace(raw_bits)

            # Build stats file path and retrieve the matching encoding data
            stats_file = image_path.rsplit(".", 1)[0] + "_stats.json"

            # Extract run index from filename (e.g. "rand_002" -> 2)
            stem = os.path.splitext(filename)[0]
            gen_index = int(stem.split("_")[-1])

            enc_data = encoding_data_list[gen_index]

            StatsWriter.write_consolidated(
                stats_file,
                original_message=options.message,
                encoded_bits=enc_data["bits_string"],
                encoded_indices=enc_data["indices_sequence"],
                encoding_stats=enc_data["encoding_stats"],
                decoded_bits=raw_bits,
                decoded_indices=list(decoded_indices),
                recovered_text=recovered_text,
                decoding_stats=decoding_stats,
                encode_trace=encode_trace,
                decode_trace=decode_trace,
                options_dict=options_dict,
                pipeline_info=pipeline_info,
                seed=options.seed,
            )


if __name__ == '__main__':
    main()