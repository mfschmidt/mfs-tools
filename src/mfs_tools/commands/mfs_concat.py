#!/usr/bin/env python3
import argparse
import mfs_tools as mt
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from datetime import datetime

from mfs_tools.library import yellow_on, red_on, color_off
from mfs_tools.library.bids_stuff import get_bids_key_pairs, glob_and_bids_match_files


def get_arguments():
    """ Parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Concatenate neuroimages along the time axis."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Files to concatenate, may be Nifti2 or Cifti2.",
    )
    parser.add_argument(
        "--save-as",
        default=None,
        help="File to write complete long neuroimage into",
    )
    parser.add_argument(
        "--fd-threshold",
        default=0.3,
        type=float,
        help="If kept frames are specified with _tmask.txt files, or if "
             "outlier frames are specified in xcp_d's outliers.tsv files, "
             "this fd-threshold is irrelevant. But if we fall back to "
             "fMRIPrep's confounds file, this threshold will determine "
             "which frames are kept and which are discarded.",
    )
    parser.add_argument(
        "--tr-len",
        default=np.nan,
        type=float,
        help="Force the tr length to this value, in seconds. If this is "
             "specified, it will be used. If not, json files will be sought "
             "to find the length of each TR. If TR length cannot be found "
             "anywhere, it will be set to 1.0s.",
    )
    parser.add_argument(
        "--set-mean",
        default=0.0,
        type=float,
        help="Each locus (voxel or vertex) is normalized within each run. "
             "By default, normalization de-means BOLD to 0.0 with a standard  "
             "deviation of 1.0, but you can change the mean here if desired.",
    )
    parser.add_argument(
        "--set-sd",
        default=1.0,
        type=float,
        help="Each locus (voxel or vertex) is normalized within each run. "
             "By default, normalization de-means BOLD to 0.0 with a standard  "
             "deviation of 1.0, but you can change the SD here if desired.",
    )
    parser.add_argument(
        "--crop-initial-frames",
        default=0,
        type=int,
        help="Some scanners acquire frames while the scanner is warming up. "
             "If those frames weren't already cropped during pre-processing, "
             "you can specify the number of frames to crop here, and they "
             "will be removed from each frame before normalization and "
             "concatenation. The same number of frames will be removed "
             "from every run.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Use --sort to re-order the files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Use --force to overwrite previously saved output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use --dry-run to list the files in order, "
             "but not actually concatenate them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use --verbose for more verbose output. This is terribly "
             "useful to review included files, dropped frames, etc.",
    )

    args = parser.parse_args()

    setattr(args, "input_files", [Path(f) for f in args.input_files])
    if args.save_as is not None:
        setattr(args, "save_as", Path(args.save_as))

    return args


def get_time_mask(bold_file, fd_threshold=0.3, verbose=False):
    """ From a bold file, find motion data and return a mask of good time points.
    """

    # Before we start, get some info on our BOLD file
    bids_key_pairs = get_bids_key_pairs(bold_file.name)
    if verbose:
        print(f"bold_file {bold_file.name} has BIDS key pairs: {bids_key_pairs}")

    # First, see if there's a matching _tmask.txt file (like Lynch data)
    # If there is, the threshold doesn't matter; the mask was already built.
    if verbose:
        print(f"  looking for {str(bold_file.parent)}/*tmask.txt...")
    for tmask_file in glob_and_bids_match_files(
            bold_file.parent, "*tmask.txt", bids_key_pairs
    ):
        data = pd.read_csv(tmask_file, header=None, index_col=False)
        mask = data.values.astype(np.bool)
        if verbose:
            print(f"  found a _tmask.txt file, '{tmask_file.name}' "
                  f"masking in {np.sum(mask)}/{len(mask)} good frames.")
        return mask

    # There was no appropriate _tmask, so lets look for xcp_d outliers
    if verbose:
        print(f"  looking for {str(bold_file.parent)}/*outliers.tsv...")
    for outlier_file in glob_and_bids_match_files(
            bold_file.parent, "*outliers.tsv", bids_key_pairs
    ):
        data = pd.read_csv(outlier_file, header=0, index_col=False, sep='\t')
        mask = ~data['framewise_displacement'].values.astype(np.bool)
        if verbose:
            print(f"  found an outliers.tsv file, '{outlier_file.name}' "
                  f"masking in {np.sum(mask)}/{len(mask)} frames.")
        return mask

    # Or xcp_d motion
    if verbose:
        print(f"  looking for {str(bold_file.parent)}/*motion.tsv...")
    for motion_file in glob_and_bids_match_files(
            bold_file.parent, "*motion.tsv", bids_key_pairs
    ):
        data = pd.read_csv(motion_file, header=0, index_col=False, sep='\t')
        mask = (data['framewise_displacement'].values <= fd_threshold).astype(np.bool)
        if verbose:
            print(f"  found a motion.tsv file, '{motion_file.name}'; "
                  f"creating a mask with {np.sum(mask)}/{len(mask)} "
                  f"(good <={fd_threshold:0.2f}) good frames.")
        return mask

    # As a last resort, lets look for fMRIPrep outliers
    for confound_file in glob_and_bids_match_files(
            bold_file.parent, "*confounds_timeseries.tsv", bids_key_pairs
    ):
        data = pd.read_csv(confound_file, header=0, index_col=False, sep='\t')
        mask = (data['framewise_displacement'].values <= fd_threshold).astype(np.bool)
        if verbose:
            print(f"  found a confounds_timeseries.tsv file, "
                  f"'{confound_file.name}'; creating a mask "
                  f"with {np.sum(mask)}/{len(mask)} "
                  f"(good <={fd_threshold:0.2f}) good frames.")
        return mask

    # If nothing was found and returned yet...
    if verbose:
        print(f"No FD data found; I tried _tmask.txt, outliers.tsv, "
              "confounds_timeseries.tsv, but none match.")
    return None


def concat(input_files, temporal_mean=0.0, temporal_sd=1.0,
        tr_len=None, verbose=False):
    """ Concatenate input_files along the time axis
    """

    images = [nib.load(file_path) for file_path in input_files]
    is_cifti2 = [isinstance(img, nib.cifti2.Cifti2Image) for img in images]
    is_nifti2 = [isinstance(img, nib.nifti2.Nifti2Image) for img in images]
    is_nifti1 = [isinstance(img, nib.nifti1.Nifti1Image) for img in images]

    if np.all(is_cifti2) or np.all(is_nifti1) or np.all(is_nifti2):
        # Motion outliers can be stored several ways, load them consistently.
        good_frame_masks = [
            get_time_mask(f, verbose=verbose) for f in input_files
        ]
        return mt.concat_images(
            input_files, temporal_masks=good_frame_masks,
            temporal_mean=temporal_mean, temporal_sd=temporal_sd,
            tr_len=tr_len, verbose=verbose
        )
    elif np.any(is_cifti2):
        print(f"{red_on}Some files are Cifti2, some are not. "
              f"I can only concatenate files of the same type.")
        return None
    elif np.any(is_nifti2):
        print(f"{red_on}"
              f"Some files are Nifti, some are not. "
              f"I can only concatenate files of the same type. "
              f"And I haven't coded nifti concatenation yet anyway."
              f"{color_off}")
        return None
    return None


def main():
    """ main entry point

        The main function wraps 'concat' for command line usage.
    """

    args = get_arguments()
    warnings = list()
    errors = list()

    dt_start = datetime.now()
    if args.verbose:
        print(f"Starting mfs_concat at {dt_start}")

    for f in args.input_files:
        if not f.exists():
            errors.append(f"{red_on}File '{str(f)}' does not exist."
                          f"{color_off}")

    if args.verbose or args.dry_run:
        print(f"Running mfs_concat on these {len(args.input_files)} files, "
              "with verbose output")
        for f in args.input_files:
            if f.exists():
                print(f" - '{str(f)}' : {mt.file_info(f)[1]}")
            else:
                print(f"{red_on} - '{str(f)}' : "
                      f"(doesn't exist){color_off}")
        if args.sort:
            print(f"Sorting requested, so they'll be concatenated in order:")
            for f in sorted(args.input_files):
                if f.exists():
                    print(f" - '{str(f)}' : {mt.file_info(f)[1]}")
                else:
                    print(f"{red_on} - '{str(f)}' : "
                          f"(doesn't exist){color_off}")

    if args.save_as is None:
        errors.append(f"--save-as is required. Please provide an output file.")
    else:
        print(f"Saving to '{str(args.save_as)}'")
        if args.save_as.is_file() and not args.force:
            errors.append(f"The suggested output file, '{str(args.save_as)}'"
                          f" already exists. mfs_concat will not overwrite "
                          f" it unless you use --force.")

    for warning in warnings:
        print(f"{yellow_on}Warning: {warning}{color_off}")
    for error in errors:
        print(f"{red_on}Error: {error}{color_off}")

    # Take any opportunity to exit before doing any real work.
    if len(errors) > 0:
        return 1
    if args.dry_run:
        return 0

    # There are no errors; this isn't a dry run; and warnings have been issued.
    # Execute the code
    concatenated_img = concat(
        sorted(args.input_files) if args.sort else args.input_files,
        temporal_mean=args.set_mean, temporal_sd=args.set_sd,
        tr_len=args.tr_len,
        verbose=args.verbose
    )
    if concatenated_img is not None:
        concatenated_img.to_filename(args.save_as)

    dt_end = datetime.now()
    if args.verbose:
        print(f"Final file '{str(args.save_as)}' "
              f"{mt.file_info(args.save_as)[1]}")
        print(f"Finished mfs_concat at {dt_end} (elapsed {dt_end - dt_start})")


if __name__ == "__main__":
    main()
