import re


def get_bids_key_pairs(bids_filename):
    """
    Extracts key-value pairs from a BIDS-compliant filename.

    This function processes a provided filename and extracts key-value pairs
    associated with 'sub', 'ses', 'task', 'run' BIDS keys
    to match BOLD files to their confounds and motion files.
    Other keys are ignored because they prevent matching BOLD files
    to their corresponding confounds or motion files.

    :param bids_filename: A string representing a BIDS-compliant filename to
        extract key-value pairs from.
    :type bids_filename: str
    :return: A dictionary where keys are BIDS entities (e.g., 'sub', 'ses', etc.)
        and values are their corresponding extracted values from the filename.
    :rtype: Dict[str, str]
    """

    bids_key_pairs = dict()
    bids_pattern_template = ".*{key}-([A-Z0-9a-z]+)_.*"
    for bids_key in ('sub', 'ses', 'task', 'run', ):
        match = re.match(
            bids_pattern_template.format(key=bids_key), bids_filename,
        )
        if match:
            bids_key_pairs[bids_key] = match.group(1)
    return bids_key_pairs


def glob_and_bids_match_files(glob_path, glob_str, bids_key_pairs):
    """
    Find and return files that match the provided glob pattern and BIDS key-value pairs.

    This function performs a glob search on a specified path with a given glob string,
    and additionally filters the resulting files based on BIDS key-value pairs. The
    resulting list contains only files that satisfy all BIDS key-value pair matches in
    their filenames.

    :param glob_path: The pathlib.Path object representing the directory path to search.
    :param glob_str: The glob pattern string to match files.
    :param bids_key_pairs: A dictionary where the keys are BIDS attribute names and the
        values are the corresponding desired values. Files must contain these key-value
        pairs in their names to be included in the result list.
    :return: A list of pathlib.Path objects representing the files that match the glob
        pattern and satisfy the BIDS key-value conditions.
    :rtype: list[pathlib.Path]
    """

    matched_files = list()
    for file in glob_path.glob(glob_str):
        file_match = True  # Assume it's good unless we disprove it.
        for bids_key, bids_val in bids_key_pairs.items():
            if f"{bids_key}-{bids_val}" not in file.name:
                file_match = False
        if file_match:
            matched_files.append(file)
    return matched_files


