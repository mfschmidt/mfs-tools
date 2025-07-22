#!/usr/bin/env python3

# mfs_decoder_from_mask_and_weights.py

import sys
from pathlib import Path
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from nilearn.signal import clean
from nilearn.image import resample_img


"""
Decoders are typically a 3D Nifti1Image of weights. And non-zero weights
would be applied to BOLD data while zero-weights would naturally multiply
to zero and have no effect. But Noam's training
process generates one mask representing the volume of interest, and a
separate list of weights that correspond to the voxels in the mask.

This script loads the mask and weights, joins them into a single volume,
and saves it as a Nifti1Image for use as a decoder.
"""

# Trigger printing in red to highlight problems
red_on = '\033[91m'
green_on = '\033[92m'
color_off = '\033[0m'
err = f"{red_on}ERROR: {color_off}"


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Convert a mask and a list of weights into a 3D weights file.",
    )
    parser.add_argument(
        "mask_file",
        help="A 3D mask, probably True/False or 1/0, but anything non-zero/zero",
    )
    parser.add_argument(
        "weight_file",
        help="The weights corresponding to non-zero voxels in the mask",
    )
    parser.add_argument(
        "output_file",
        help="The full path to the output decoder file to be written",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="set to trigger output of some extra values for comparisons",
    )

    args = parser.parse_args()

    return args


def validated(args):
    """ Validate arguments """

    if args.verbose:
        print(f"Running from {str(Path.cwd())}")

    we_have_a_fatal_error = False

    for p, desc, optional in [
        (args.bold_file, 'mask_file', False),
        (args.decoder_file, 'weight_file', False),
    ]:
        if p is not None and Path(p).exists():
            if args.verbose:
                print(f"{green_on}Path '{p}' exists for '{desc}'.{color_off}")
            setattr(args, desc, Path(p).resolve())
        elif p is not None and not Path(p).exists():
            if args.verbose:
                print(f"{red_on}Path '{p}' for '{desc}' does not exist.{color_off}")
            we_have_a_fatal_error = True
        else:
            if args.verbose:
                print(f"{red_on}Path for '{desc}' is required, but not provided.{color_off}")
            we_have_a_fatal_error = True

    # Store paths as Path objects rather than strings
    setattr(args, "output_file", Path(args.output_file).resolve())
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    if we_have_a_fatal_error:
        sys.exit(1)

    return args


def combine_mask_and_weights(args, verbose=False):
    """ Load the 4D bold data, and mask it with the decoder mask. """

    mask_img = nib.load(args.mask_file)
    one_hot_mask = mask_img.get_fdata().astype("bool").astype("uint8")

    weights = pd.read_csv(
        args.decoder_weights, sep="\t", header=None
    ).values

    # TODO: Apply the weights to the mask in fortran order
    weight_data = one_hot_mask * weights

    weight_image = nib.Nifti1Image(weight_data, mask_img.affine)

    return weight_image


def main(args):
    """ Entry point """

    print("Combining '{}' with '{}' to make decoder '{}'.".format(
        str(args.mask_file), str(args.weight_file), str(args.output_file)
    ))

    weight_image = combine_mask_and_weights(args)
    if args.verbose:
        print(f"Shape of weights: {weight_image.shape}")

    weight_image.to_filename(args.output_file)

    return 0


if __name__ == "__main__":
    main(validated(get_arguments()))
