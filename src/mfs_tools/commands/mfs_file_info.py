import argparse
import mfs_tools as mt
from pathlib import Path


def get_arguments():
    """ Parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Describe the file at the path specified."
    )
    parser.add_argument(
        "input_file",
        help="A file to investigate",
    )
    parser.add_argument(
        "--just-the-dims",
        action="store_true",
        help="Output only the comma-separated dimensions.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use --verbose for more verbose output.",
    )

    args = parser.parse_args()

    return args


def file_info(input_file, just_the_dims=False, verbose=False):
    """ Return a string with information about the file_path.
    """

    return mt.get_img_and_desc(
        Path(input_file),
        only_dims=just_the_dims,
        verbose=verbose
    )


def main():
    """ main entry point

        The main function wraps 'file_info' for command line usage.
    """

    args = get_arguments()

    if args.verbose:
        print(f"Running mfs_file_info on '{args.file}', with verbose output")

    if Path(args.file).is_file():
        img, desc = file_info(
            args.file,
            just_the_dims=args.just_the_dims,
            verbose=args.verbose
        )
        print(desc)
    else:
        print(f"No file named '{args.file}' exists.")


if __name__ == "__main__":
    main()
