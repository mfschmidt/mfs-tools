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
        "file",
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


def main():
    """ main entry point
    """

    args = get_arguments()

    if args.verbose:
        print(f"Running mfs_file_info on '{args.file}', with verbose output")

    if Path(args.file).is_file():
        img, desc = mt.get_img_and_desc(
            args.file,
            only_dims=args.just_the_dims,
            verbose=args.verbose
        )
        print(desc)
    else:
        print(f"No file named '{args.file}' exists.")


if __name__ == "__main__":
    main()
