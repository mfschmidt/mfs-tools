#!/usr/bin/env python3

# decode.py

import sys
from pathlib import Path
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from nilearn.signal import clean
from nilearn.image import resample_to_img, smooth_img, index_img


"""
The main kernel of decoding is on one run with one mask/weights pair.
So the 'for each subject, for each run' stuff should happen in the shell,
where each iteration should execute this file. The current implementation
uses the 'decode_everything_with_python.sh' script alongside this one.
Note that this script uses several python libraries that must be
installed for successful execution, too. It would be best to activate
a virtual environment before running the shell script.

The original matlab decoder also spends a lot of lines of code on picking
out the TRs of interest. This decoder doesn't care. It will just decode
the entire BOLD file start to finish. The user can pick out their own
blocks/periods/trials however they like.

"""

# Trigger printing in red to highlight problems
red_on = '\033[91m'
green_on = '\033[92m'
yellow_on = '\033[33m'
color_off = '\033[0m'
err = f"{red_on}ERROR: {color_off}"


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description="Apply one or more decoder(s) (mask and weights) to a 4D BOLD file.",
    )
    parser.add_argument(
        "bold_file",
        help="The 4D data, already pre-processed, cropped, smoothed, filtered",
        # Future upgrade would be to accept an fMRIPrep output file, then
        # do the crop, smooth, filter here to save the Feat step
        # and the space from the extra files Feat writes.
        # But for right now, this simply replaces the matlab version
        # in a way that saves me time writing debug info for visuals.
    )
    parser.add_argument(
        "decoder_files",
        nargs="+",
        help="One or more files containing 3D decoder weights",
    )
    parser.add_argument(
        "--decoder-mask",
        help="Optionally, apply a mask to the decoder weights, so that only "
             "weights within the mask are applied to the BOLD data.",
    )
    parser.add_argument(
        "--confounds",
        help="Optionally, mfs_decode.py can regress out vectors in each "
             "column of a tsv file, w/ header. This file can be created by "
             "extracting selected columns from fMRIPrep confounds",
        # A potential upgrade is accepting full fMRIPrep confounds and a spec
        # for which confounds to pull out of it and how many TRs to crop off
        # the top of it (which would match the non-steady-state TRs in BOLD).
        # This script would then do the crop/extract without the extra file.
    )
    parser.add_argument(
        "--confound-strategy", type=str, default="",
        help="Optionally, extract a collection of confounds from the "
             "confounds file for regression \n"
             "'motion_6' regresses out 3 translation and 3 rotation confounds."
             "'motion_7' adds csf_wm to the six motion confounds.",
    )
    parser.add_argument(
        "--clip", type=int, default=0,
        help="Optionally, clip the first N volumes as non-steady-state"
             "outliers, before smoothing or analysis. The confounds will "
             "be clipped to match.",
    )
    parser.add_argument(
        "--smooth", type=float, default=0.0,
        help="Optionally, apply smoothing to the BOLD data with "
             "full width at half maximum (fwhm) specified",
    )
    parser.add_argument(
        "--output-path", default=".",
        help="write output files here, rather than in the current directory",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="set to force overwriting of existing data",
    )
    parser.add_argument(
        "--save-intermediates", action="store_true",
        help="set to save out resampled decoders and masks",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="set to trigger verbose output",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="set to trigger output of some raw BOLD data for review",
    )

    args = parser.parse_args()

    return args


def validated(args):
    """ Validate arguments """

    if args.verbose:
        print(f"Running from {str(Path.cwd())}")

    we_have_a_fatal_error = False

    for p, desc, optional in [
        (args.bold_file, 'bold_file', False),
        (args.decoder_mask, 'decoder_mask', True),
        (args.confounds, 'confounds', True),
    ] + [
        (decoder_file, f'decoder_file {i}', False)
        for i, decoder_file in enumerate(args.decoder_files)
    ]:
        if p is not None and Path(p).exists():
            if args.verbose:
                print(f"{green_on}Path '{p}' exists for '{desc}'.{color_off}")
            setattr(args, desc, Path(p).resolve())
        elif p is not None and not Path(p).exists():
            if args.verbose:
                print(f"{red_on}Path '{p}' for '{desc}' does not exist.{color_off}")
            we_have_a_fatal_error = True
        elif p is None and not optional:
            if args.verbose:
                print(f"{red_on}Path for '{desc}' is required, but not provided.{color_off}")
            we_have_a_fatal_error = True
        else:
            print(f"No optional '{desc}' will be applied.")

    # Store paths as Path objects rather than strings
    setattr(args, "output_path", Path(args.output_path).resolve())
    if args.output_path.exists():
        existing_score_files = list(args.output_path.glob("*.tsv"))
        if len(existing_score_files) > (len(args.decoder_files) * 2):
            if args.force:
                print(f"{red_on}{len(existing_score_files)}/{len(args.decoder_files) * 2} "
                      f"decoder scores already exist at {str(args.output_path)}. "
                      f"We will overwrite them because --force was set.{color_off}")
            else:
                print(f"{red_on}{len(existing_score_files)}/{len(args.decoder_files) * 2} "
                      f"decoder scores already exist at {str(args.output_path)}. "
                      f"To overwrite them, run with --force{color_off}")
                we_have_a_fatal_error = True
        elif len(existing_score_files) > 0:
            if args.force:
                print(f"{red_on}{len(existing_score_files)}/{len(args.decoder_files) * 2} "
                      f"decoder scores already exist at {str(args.output_path)}. "
                      f"We will overwrite them because --force was set.{color_off}")
            else:
                print(f"{red_on}{len(existing_score_files)}/{len(args.decoder_files) * 2} "
                      f"decoder scores already exist at {str(args.output_path)}. "
                      f"Running to fill in the missing scores.{color_off}")
    args.output_path.mkdir(parents=True, exist_ok=True)

    if we_have_a_fatal_error:
        sys.exit(1)

    return args


def load_bold_image(bold_file, smoothing=None, clipping=None, verbose=False):
    """ Load the BOLD data file, smoothing if requested. """

    bold_img = nib.load(bold_file)
    if verbose:
        print(f"Loading a {bold_img.shape} BOLD image")

    if (clipping is not None) and (clipping != 0):
        if verbose:
            print(f"  clipping the first {clipping} volumes")
        bold_img = index_img(bold_img, slice(clipping, bold_img.shape[-1]))

    if (smoothing is not None) and (smoothing != 0.0):
        if verbose:
            print(f"  smoothing the BOLD image with a "
                  f"{smoothing:0.1f}mm Gaussian kernel")
        bold_img = smooth_img(bold_img, smoothing)

    return bold_img


def remove_motion(bold_img, confound_file, scale='zscore', strategy='', method='manual', verbose=False):
    """ Regress out motion confounds, return scaled residuals. """

    if confound_file.name.endswith(".tsv"):
        # fMRIPrep prepares a tab-separated table, with a header row
        confounds = pd.read_csv(confound_file, sep='\t', header=0)
        print(f"loaded confounds for {len(confounds)} time points, "
              f"to match data with {bold_img.shape[-1]} time points")
    elif confound_file.name.endswith(".par"):
        # If motion correction was done by FSL Feat, double-spaces
        confounds = pd.read_csv(confound_file, sep=r'\s+', header=None)
        print(f"loaded confounds for {len(confounds)} time points, "
              f"to match data with {bold_img.shape[-1]} time points")
    else:
        print(f"{yellow_on}WARNING : No confound file, not removing motion"
              f"{color_off}")
        raise FileNotFoundError(f"Could not find '{confound_file}'")

    confound_clip_num = len(confounds) - bold_img.shape[-1]
    if confound_clip_num > 0:
        print(f"Clipping first {confound_clip_num} of {len(confounds)} "
              f"confounds values to match BOLD length")
        confounds = confounds.iloc[confound_clip_num:, :]

    # If a specific strategy was requested, extract the appropriate columns
    motion_6_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    if strategy == 'motion_6':
        print(f"Extracting six motion columns from confounds file")
        confounds = confounds[motion_6_cols]
    elif strategy == 'motion_7':
        print(f"Extracting 'csf_wm' and six motion columns from confounds file")
        confounds = confounds[motion_6_cols + ['csf_wm', ]]

    # Ensure the y-intercept, or arbitrary mean BOLD, doesn't make a difference.
    confounds['bias'] = 1.0

    # One way is to do this with nilearn, in one line:
    if method == 'nilearn':
        # Nilearn insists we should de-trend or standardize.
        # For now, I prevent it to ensure these results are identical to matlab.
        return clean(bold_img.get_fdata().T, confounds=confounds.values, detrend=False,
                     standardize=scale, standardize_confounds=False).T

    # Another way is to replicate matlab exactly and do all of this manually:
    beta_motion = np.dot(bold_img.get_fdata(), np.linalg.pinv(confounds.values).T)
    motion_residuals = bold_img.get_fdata() - np.dot(beta_motion, confounds.values.T)

    if scale == 'zscore':
        # Compute z scores across voxel rows, NOT time columns
        # with population degrees of freedom, not sample

        # Python returns a row of NaN z-scores for a row of zero data.
        # Matlab returns a row of zeros, which is more useful.
        # Here, we zero out the NaNs to allow scoring via the other voxels.
        raw_z = zscore(motion_residuals, axis=1, ddof=0)
        safe_z = np.nan_to_num(raw_z, nan=0.0)
        return nib.Nifti1Image(safe_z, bold_img.affine)

    return nib.Nifti1Image(motion_residuals, bold_img.affine)


def load_and_mask_data(
        bold_img, decoder_file, mask_file=None, output_path=None, verbose=False
):
    """ Load 4D bold data, mask it, and return 2D matrix. """

    # Name the decoder, without the .nii.gz
    loc_underscore = Path(decoder_file).name.find(".")
    _name = Path(decoder_file).name[0:loc_underscore]

    # The BOLD image is the standard; weights and masks must be resampled
    # to match it, not the other way around.
    decoder_img = nib.load(decoder_file)
    decoder_weights = decoder_img.get_fdata()
    if verbose:
        print(f"A {decoder_weights.shape} decoder was loaded with "
              f"{np.sum(decoder_weights.astype('bool')):,} hot voxels")

    # We regularly deal with decoders that are in LAS+ (FSL's MNI152)
    # rather than RAS+ (templateflow's MNI152s) or in a different
    # space, like Phil Kragel's occipital lobe weights, etc.
    # We anticipate this, and resample the decoder weights into BOLD
    # space. We use nearest-neighbor resampling because expanding the
    # edges into the zeros can significantly increase computation of all
    # the nearly-zero weights that ought to simply be masked out.
    if (    (decoder_img.shape != bold_img.shape[:3]) or
            (not np.allclose(decoder_img.affine, bold_img.affine))
    ):
        if verbose:
            print(f"  {yellow_on}WARNING : The decoder is not in the same "
                  f"space as the BOLD data. Resampling decoder from "
                  f"{decoder_img.shape} to "
                  f"{bold_img.shape[:3]} to match BOLD image.{color_off}")
        decoder_img = resample_to_img(
            decoder_img, bold_img,
            interpolation='nearest', force_resample=True,
        )
        decoder_weights = decoder_img.get_fdata()
        if verbose:
            print(f"  the {decoder_img.shape} decoder now has "
                  f"{np.sum(decoder_weights.astype('bool')):,} "
                  f"{tuple([float(_) for _ in decoder_img.header.get_zooms()])} hot voxels")

        if output_path is not None:
            decoder_img.to_filename(output_path / f"decoder_{_name}_2.nii.gz")

    # Now that we have the decoder in BOLD space, should we also mask it?
    if mask_file is None:
        print(f"  there is no mask, using non-zero decoder weights.")
        x_res, y_res, z_res = decoder_img.header.get_zooms()
        voxel_volume = x_res * y_res * z_res
        decoder_vol = np.sum((decoder_weights != 0.0).astype('bool')) * voxel_volume
        print(f"  the decoder's non-zero weights {decoder_img.shape}, "
              f"{decoder_img.header.get_zooms()}, {decoder_vol:0.1f}mm3")
    else:
        # Assuming the mask and weights are in the same space
        mask_img = nib.load(mask_file)
        if verbose:
            print(f"  a {mask_img.shape} mask was loaded with "
                  f"{np.sum(mask_img.get_fdata().astype('bool')):,} hot voxels")
        if (    (not np.allclose(bold_img.affine, mask_img.affine)) or
                (bold_img.shape != mask_img.shape)
        ):
            print(f"  {yellow_on}WARNING : The decoder weights and the mask "
                  f"are not in the same space. Resampling...{color_off}")
            x_res, y_res, z_res = mask_img.header.get_zooms()
            voxel_volume = x_res * y_res * z_res
            mask_vol = np.sum((mask_img.get_fdata() != 0.0).astype('bool')) * voxel_volume
            print(f"  the mask started as {mask_img.shape}, "
                  f"{mask_img.header.get_zooms()}, {mask_vol:0.1f}mm3")
            resampled_mask = resample_to_img(
                mask_img, bold_img,
                interpolation='nearest', force_resample=True
            )
            x_res, y_res, z_res = resampled_mask.header.get_zooms()
            voxel_volume = x_res * y_res * z_res
            mask_vol = np.sum((resampled_mask.get_fdata() != 0.0).astype('bool')) * voxel_volume
            print(f"  it was resampled to {resampled_mask.shape}, "
                  f"{resampled_mask.header.get_zooms()}, {mask_vol:0.1f}mm3")
            # print(f"{red_on}The mask must be in the same space as the decoder.{color_off}")
            # raise ValueError("Decoder/Mask mismatch")
        else:
            resampled_mask = mask_img

        # Binarize the mask and filter the decoder weights by it.
        one_hot_mask = resampled_mask.get_fdata().astype("bool").astype("uint8")
        decoder_weights = decoder_weights * one_hot_mask
        decoder_img = nib.Nifti1Image(decoder_weights, decoder_img.affine)
        if verbose:
            print(f"  the {decoder_weights.shape} decoder was masked down to "
                  f"{np.sum(decoder_weights.astype('bool')):,} hot voxels."
                  f"{color_off}")

        if output_path:
            decoder_img.to_filename(output_path / f"decoder_{_name}_3.nii.gz")

    # The BOLD data were previously loaded, smoothed, and residualized
    # Handle 4D data as [all_voxels x time] 2D matrix.
    # To match matlab and the weights, this MUST be done in fortran order.
    bold_4d_data = bold_img.get_fdata()
    dims = bold_img.shape
    voxels_per_volume = dims[0] * dims[1] * dims[2]
    bold_full_2d_data = np.reshape(
        bold_4d_data, (voxels_per_volume, dims[3]), order='F'
    )
    decoder_2d_data = np.reshape(
        decoder_weights, voxels_per_volume, order='F'
    )
    masked_bold_data = bold_full_2d_data[decoder_2d_data != 0]

    if verbose:
        print(f"  masked BOLD data are now shaped {masked_bold_data.shape} "
              f"and has {np.sum(masked_bold_data != 0.0):,} values.")
        print(f"  weights are now shaped {decoder_2d_data.shape} "
              f"and has {np.sum(decoder_2d_data != 0.0):,} values.")

    # Add a bias, for the intercept. This is never zero or one in Noam's
    # decoders, though. :( I am using a 0-intercept, and a brief
    # investigation looked like putting it AFTER the data fits best.
    weights = np.append(decoder_2d_data[decoder_2d_data != 0.0], 0.0)
    print(f"  mean weight value: {np.mean(weights):.3f} "
          f"({np.min(weights):.3f} to {np.max(weights):.3f})")

    return masked_bold_data, weights, _name


def predict_y(data, weights, verbose=False):
    """ Use measured BOLD data (cleaned) to predict y """

    # Normally, we use a decoder, which is a vector of weights.
    # But we may also want to use all ones for the decoder as a null comparison.
    # This allows us to see if the mask itself is responsible for any
    # significant decoder effect, without the weights.
    if np.array_equal(np.ones(weights.shape), weights):
        words = "created", "as ones"
    else:
        words = "extracted", "from decoder volume"

    if verbose:
        print(f"  {words[0]} {len(weights)} weights {words[1]}")

    if data.shape[0] == weights.shape[0]:
        # No intercept, use as-is
        x = data
    else:
        # The weights have an intercept, add ones to the data
        x = np.append(data, np.ones((1, data.shape[1])), axis=0)
    y_hat = np.dot(weights.T, x).T

    # This is the decoder score for each t
    return y_hat


def write_some_matrices(data):
    if len(data.shape) != 3:
        print(f"not debugging {data.shape}-shaped matrix; expecting 3D.")
    # Select a 3D patch that contains different mask labels
    samp_x = 20  # int(data.shape[0] / 3)
    samp_y = 72  # int(data.shape[1] * 2 / 3)
    samp_z = 32  # int(data.shape[2] / 2)
    for _z in range(samp_z, samp_z + 3):
        print(
            f"z slice {_z}; "
            f"x = {samp_x} to {samp_x + 6}, "
            f"y = {samp_y} to {samp_y + 4}:"
        )
        for _y in range(samp_y, samp_y + 5):
            print(
                f"y={_y:>3}:  " + ", ".join([
                    f"{data[_x, _y, _z]:0.4f}"
                    for _x in range(samp_x, samp_x + 7)
                ])
            )


def main(args):
    """ Entry point """

    if len(args.decoder_files) == 1:
        print("Decoding {} with a '{}' decoder.".format(
            str(args.bold_file), Path(args.decoder_files[0]).name
        ))
    else:
        print("Decoding {} with {} decoders.".format(
            str(args.bold_file), len(args.decoder_files)
        ))

    # Load the BOLD fMRI 4D data
    bold_img = load_bold_image(
        args.bold_file, smoothing=args.smooth, clipping=args.clip, verbose=args.verbose
    )

    # Remove motion confounds from BOLD, if requested
    if args.confounds:
        bold_img = remove_motion(
            bold_img, args.confounds, scale='zscore',
            strategy=args.confound_strategy, verbose=args.verbose
        )

    if args.save_intermediates:
        bold_img.to_filename(args.output_path / "final_bold.nii.gz")

    for decoder_file in [Path(df) for df in args.decoder_files]:
        split_idx = max(
            decoder_file.name.find("_ones"),
            decoder_file.name.find("_weights")
        )
        decoder_name = decoder_file.name[8:split_idx]
        existing_score_files = [
            sf for sf in list(args.output_path.glob("*.tsv"))
            if decoder_name in sf.name
        ]
        if len(existing_score_files) > 1 and not args.force:
            print(f"Scores for {decoder_file.name} already exist. "
                  f"Skipping this decoder. Use --force to overwrite, "
                  f"or delete files you'd like to replace and run again.")
            continue
        elif len(existing_score_files) > 0 and not args.force:
            print(f"One score file for {str(decoder_file)} already exists, "
                  f"but there should be two. Trying again to generate "
                  f"both scores files for {decoder_file.name}.")

        if args.save_intermediates:
            intermediate_output_path = args.output_path
        else:
            intermediate_output_path = None
        bold_data, weight_data, decoder_name = load_and_mask_data(
            bold_img, decoder_file, args.decoder_mask,
            output_path=intermediate_output_path, verbose=args.verbose
        )
        if args.verbose:
            print(f"  shape of loaded data : {bold_data.shape}")
            print(f"  shape of weights     : {weight_data.shape}")
            print(f"  shape of final data  : {bold_data.shape}")

        if args.debug:
            # Write out values from a specific region in each piece of data.
            write_some_matrices(nib.load(args.bold_file).get_fdata()[:, :, :, 1])

        for label, weights in [
            ("ones", np.ones((weight_data.shape[0], 1))),
            ("weights", weight_data),
        ]:
            print(f"Shape of weights: {weights.shape}")
            predicted_y = predict_y(bold_data, weights, args.verbose)
            if np.sum(np.isnan(predicted_y)) > 0:
                print("NaN values in predicted y, no scores!")
            pd.DataFrame(predicted_y).to_csv(
                args.output_path / f"all_trs_{decoder_name}_{label}_scores.tsv",
                sep='\t', header=False, index=False,
            )

    return 0


if __name__ == "__main__":
    main(validated(get_arguments()))
