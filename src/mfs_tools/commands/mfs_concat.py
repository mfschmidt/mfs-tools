import argparse
import mfs_tools as mt
from mfs_tools.library import yellow_on, red_on, cyan_on, color_off
from pathlib import Path
import nibabel as nib
import numpy as np


def get_arguments():
    """ Parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Concatenate neuroimages along the time axis."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Files to concatenate",
    )
    parser.add_argument(
        "--save-as",
        default=None,
        help="File to write complete long neuroimage into",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Use --sort to re-order the files.",
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
        help="Use --verbose for more verbose output.",
    )

    args = parser.parse_args()

    setattr(args, "input_files", [Path(f) for f in args.input_files])
    if args.save_as is not None:
        setattr(args, "save_as", Path(args.save_as))

    return args


def concat(input_files, verbose):
    """ Concatenate input_files along the time axis
    """

    images = [nib.load(file_path) for file_path in input_files]
    is_cifti2 = [isinstance(img, nib.cifti2.Cifti2Image) for img in images]
    is_nifti2 = [isinstance(img, nib.nifti2.Nifti2Image) for img in images]
    if np.all(is_cifti2):
        return mt.concat_dtseries(input_files, tr_len=1.0)
        # TODO: Figure out tr_len
    elif np.any(is_cifti2):
        print(f"{red_on}Some files are Cifti2, some are not. "
              f"I can only concatenate files of the same type.")
        return None
    elif np.all(is_nifti2):
        print(f"{cyan_on}"
              f"Nifti concatenation is coming, but not yet coded."
              f"{color_off}")
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
        print(f"Saving them all to '{str(args.save_as)}'")
        if args.save_as.is_file():
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
        verbose=args.verbose
    )
    if concatenated_img is not None:
        concatenated_img.to_filename(args.save_as)

    if args.verbose:
        print(f"Final file '{str(args.save_as)}' "
              f"{mt.file_info(args.save_as)[1]}")


if __name__ == "__main__":
    main()
