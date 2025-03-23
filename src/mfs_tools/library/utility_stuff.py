import os
import pathlib

import psutil
import numpy as np
import pandas as pd
from datetime import datetime
import math
import nibabel as nib
import json
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

from mfs_tools.library import red_on, green_on, color_off
from .bids_stuff import get_bids_key_pairs, glob_and_bids_match_files
from .cifti_stuff import get_repetition_time


def compare_mats(
        a, b,
        a_name="a", b_name="b", verbose=True, preview=True, tolerance=0.00001
):
    """
    Compares two matrices element-by-element with a specified tolerance.
    This function checks if the matrices are equal within the given tolerance.
    It also offers a preview of mismatched matrix sections and summarizes
    the discrepancies.

    :param a: First matrix for comparison
    :type a: numpy.array
    :param b: Second matrix for comparison
    :type b: numpy.array
    :param a_name: Name label for the first matrix, used in output
    :type a_name: str, optional
    :param b_name: Name label for the second matrix, used in output
    :type b_name: str, optional
    :param verbose: Whether to print detailed output about the comparison process
    :type verbose: bool, optional
    :param preview: Whether to output a small preview of matrix differences
    :type preview: bool, optional
    :param tolerance: The acceptable difference between corresponding elements in the matrices
    :type tolerance: float, optional
    :return: Boolean indicating whether the matrices are equal within the specified tolerance
    :rtype: bool
    """

    assert a.shape == b.shape, "Matrices must be the same size and shape."

    # Track memory usage
    mem_before = psutil.Process(os.getpid()).memory_info().rss

    # This comparison may consume a lot of RAM,
    # I assume because it compares floats.
    # So ensure there's memory available for it.
    if np.allclose(a, b, atol=tolerance):
        if verbose:
            print(green_on +
                  f"  The matrices '{a_name}' and '{b_name}' are equal, "
                  f"with tolerance of {tolerance}." +
                  color_off)
        return_val = True
    else:
        if verbose:
            print(f"  There are mismatches between '{a_name}' ({a.dtype}) "
                  f" and '{b_name}' ({b.dtype}).")
            if preview:
                print(f"  Top left corners, for a small preview:")
                for row in range(min([5, a.shape[0], b.shape[0], ])):
                    mid_str = "vs" if row == 2 else "  "
                    a_vals = ",".join([
                        f"{v:0.4f}".rjust(9, " ") for v in a[row, :5]
                    ])
                    b_vals = ",".join([
                        f"{v:0.4f}".rjust(9, " ") for v in b[row, :5]
                    ])
                    print("| " + a_vals + f" | {mid_str} | " + b_vals + " |")

            # Extract just values that differ between matrices and compare them.
            eq = np.array(a == b, dtype=np.bool)
            different_a_vals = a[~eq]
            different_b_vals = b[~eq]

            diff_vals = pd.DataFrame({
                a_name: np.astype(different_a_vals, np.float32),
                b_name: np.astype(different_b_vals, np.float32),
            })
            diff_vals['delta'] = np.abs(diff_vals[a_name] - diff_vals[b_name])

            if (np.sum(~eq) == 0) or ((len(eq) / np.sum(~eq)) < 10000):
                print(red_on +
                      f"  {np.sum(~eq):,} of {len(eq.ravel()):,} values differ. " +
                      f"The mean difference, where there are differences, "
                      f" is {np.mean(np.abs(diff_vals['delta'])):0.9f}." +
                      color_off)
            else:
                print(green_on +
                      f"  Only 1 in {int(len(eq) / np.sum(~eq))} values differ" +
                      f" ({np.sum(~eq):,} of {len(eq):,}). " +
                      color_off)

            if diff_vals['delta'].max() >= 1.0:
                print(red_on, end="")
            else:
                print(green_on, end="")
            print(f"  The largest difference is {diff_vals['delta'].max()} "
                  f"{color_off}")
        return_val = False

    mem_after = psutil.Process(os.getpid()).memory_info().rss

    if verbose:
        print(f"  Mem before {mem_before / 1024 / 1024:0,.1f}MB; "
              f"Mem after {mem_after / 1024 / 1024:0,.1f}MB; "
              f"delta {(mem_after - mem_before) / 1024 / 1024:0,.1f}")

    return return_val


def correlate_columns(a, b):
    """
    Calculate the Pearson correlation coefficient between two input arrays.

    The input arrays must have the same number of rows for correlations
    to work, but they may differ in column number. A correlation matrix
    from a [100 x 14] 'a' and a [100 x 30] 'b' would be [14 x 30], where
    return_matrix[4, 6] would be the correlation between a[:,4] and b[:,6].

    :param a: The first input array. Each column represents a variable.
    :type a: numpy.ndarray
    :param b: The second input array. Each column represents a variable.
    :type b: numpy.ndarray
    :return: The Pearson correlation matrix, where each entry represents
             the correlation between a column of the first input array and a column
             of the second input array.
    :rtype: numpy.ndarray
    """
    # Demean input arrays and calculate covariance
    a_demeaned = a - np.mean(a, axis=0)[np.newaxis, :]
    b_demeaned = b - np.mean(b, axis=0)[np.newaxis, :]
    numerator = np.dot(a_demeaned.T, b_demeaned)

    # Calculate deviations
    sum_squares_a = np.sum(a_demeaned**2, axis=0)[:, np.newaxis]
    sum_squares_b = np.sum(b_demeaned**2, axis=0)[np.newaxis, :]
    denominator = np.sqrt(np.dot(sum_squares_a, sum_squares_b))

    # Covariance over variance is the Pearson correlation
    return np.divide(numerator, denominator, where=denominator != 0.0)


def correlate_bold(
        bold_data, strip_size=2048, dtype=np.float32,
        force_diagonals_to=None, zero_nans=False, verbose=False
):
    """
    Correlate BOLD data time series to compute a correlation matrix.

    This method divides the data into manageable strips,
    computes the correlation for each strip, and assembles these into the final
    correlation matrix. The results from this method should match results
    from numpy's corrcoef function, but with lower memory usage and longer
    processing time. This operation is computationally intensive and can take
    considerable time depending on the size of the input data and the strip
    size. It is also single-process because each additional process requires
    a copy of the huge matrix, costing more memory.

    To create a dense connectivity matrix for a full 92k Cifti file, the
    correlation matrix will contain 8.5 billion (92k**2) entries,
    which will take 34GB memory as floats, or 68GB as doubles. This is
    a bare minimum. Adding the size of your strip would approximate your
    memory usage.
    strip_size=1024 would add 400MB.
    strip_size=2048 (the default) would add 800MB.
    strip_size=46000 (half the matrix) would add 17GB.

    The output is a correlation matrix representing pairwise correlations
    between locations.

    :param bold_data: A numpy array of BOLD data where rows represent time
        points and columns represent locations.
    :param strip_size: Size of the data strip for processing. Default is 2048.
        The larger the strip size, the faster the computation, but at a
        cost of higher memory usage.
    :param dtype: Data type of the output correlation matrix. Default is
        np.float32.
    :param force_diagonals_to: Default None does nothing;
        If anything else is provided, all diagonals in the correlation
        matrix will be set to this value.
    :param zero_nans: Default is False;
        If set to True, any NaN values will be set to 0.0.
    :param verbose: If set to True, prints details about the computation.
    :return: A numpy array representing the correlation matrix.
    :rtype: numpy.array
    """

    start_dt = datetime.now()

    m = np.empty((bold_data.shape[1], bold_data.shape[1]), dtype=dtype)
    if verbose:
        print(f"Input BOLD has {bold_data.shape[1]} columns (locations) and "
              f"{bold_data.shape[0]} rows (time points).")
        print(f"Correlating time series "
              f"will yield a {m.shape}-shaped correlation matrix.",
              flush=True)
    # Create indices into strips of the matrix to do strip-wise correlations.
    slice_bounds = list(np.arange(strip_size, bold_data.shape[1], strip_size))
    slice_bounds = [((b - strip_size), b) for b in slice_bounds]
    slice_bounds += [(slice_bounds[-1][1], bold_data.shape[1])]

    # "Manually" correlate the strips, filling the correlation matrix as we go
    # These are single-threaded and serial, so very slow, but using
    # multi-processing would duplicate memory also.
    # TODO: Look into one process, multiple threads if possible.
    for idx_start, idx_end in slice_bounds:
        iter_start_dt = datetime.now()
        if verbose:
            print(f"Loci {idx_start} to {idx_end}: ["
                  f"{bold_data[:, idx_start:idx_end].shape} x "
                  f"{bold_data.shape}]",
                  end="", flush=True)
        sub_m = correlate_columns(
            bold_data[:, idx_start:idx_end], bold_data
        )
        m[idx_start:idx_end, :] = sub_m
        iter_end_dt = datetime.now()
        elapsed_time = (iter_end_dt - iter_start_dt).total_seconds()
        if verbose:
            print(f" ... {sub_m.shape} in {elapsed_time:0.0f}s", flush=True)

    if force_diagonals_to is not None:
        if verbose:
            print(f"Forcing all values on the diagonal to {force_diagonals_to:0.1f}")
        np.fill_diagonal(m, force_diagonals_to)

    if zero_nans:
        if verbose:
            print(f"Removing {np.sum(np.isnan(m)):,} NaN values from BOLD connectivity")
        m[np.isnan(m)] = 0.0

    print(f"Overall, correlation took "
          f"{(datetime.now() - start_dt).total_seconds():0.0f}s")
    return m


def generate_colormap(num_distinct_colors: int = 80):
    """ Generate distinguishable colors

        Copied and modified from https://stackoverflow.com/questions/42697933/colormap-with-maximum-distinguishable-colours
    """

    if num_distinct_colors == 0:
        num_distinct_colors = 80

    number_of_shades = 7
    num_distinct_colors_with_multiply_of_shades = int(
        math.ceil(num_distinct_colors / number_of_shades) * number_of_shades
    )

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = (
        np.arange(num_distinct_colors_with_multiply_of_shades) /
        num_distinct_colors_with_multiply_of_shades
    )

    # We are going to reorganise monotonically growing numbers in such way
    # that there will be single array with saw-like pattern
    # but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays
    # containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(
        number_of_shades,
        num_distinct_colors_with_multiply_of_shades // number_of_shades
    )

    # Transpose the above matrix (columns become rows) -
    # as a result each row contains saw tooth with values
    # slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic
    # (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic),
    # we'll use this property
    initial_cm = cm.get_cmap('hsv')(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of
    # partition are darker
    # First colours are affected more,
    # colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8 / lower_half)

    # Modify second half in such way that colours towards end of
    # partition are less intense and brighter
    # Colours closer to the middle are affected less,
    # colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            idx_start = lower_half + j * number_of_shades
            idx_end = lower_half + (j + 1) * number_of_shades
            modifier = (
                    np.ones(number_of_shades) -
                    initial_cm[idx_start: idx_end, i]
            )
            modifier = j * modifier / upper_partitions_half
            initial_cm[idx_start: idx_end, i] += modifier

    return ListedColormap(initial_cm)


def get_tr_len(bold_file, verbose=False):
    """ Find a json file with a TR length for this BOLD file.
    """

    # The best bet is to get it directly from the Nifti/Cifti2 file.
    tr_len = np.nan
    if (
        isinstance(bold_file, nib.nifti1.Nifti1Image) or
        isinstance(bold_file, nib.nifti2.Nifti2Image) or
        isinstance(bold_file, nib.cifti2.Cifti2Image)
    ):
        img = bold_file
    elif (
        isinstance(bold_file, str) or
        isinstance(bold_file, pathlib.Path) or
        isinstance(bold_file, os.PathLike)
    ):
        img = nib.load(bold_file)
        bold_file = pathlib.Path(bold_file)
    else:
        raise TypeError(f"Expected a path or nibabel image, got {type(bold_file)}")

    if (    isinstance(img, nib.nifti1.Nifti1Image) or
            isinstance(img, nib.nifti2.Nifti2Image)
    ):
        if len(img.shape) > 3:
            dims = img.header.get('pixdim', None)
            if dims is not None and len(dims) > 4:
                tr_len = dims[4]
    elif isinstance(img, nib.cifti2.Cifti2Image):
        tr_len = get_repetition_time(img)

    if np.isfinite(tr_len):
        return tr_len

    # If that didn't work, get some info on our BOLD file to look for sidecars
    bids_key_pairs = get_bids_key_pairs(bold_file.name)

    # Look for a .json sidecar with the same features as the BOLD file
    for sidecar_file in glob_and_bids_match_files(
            bold_file.parent, "*_bold.json", bids_key_pairs
    ):
        with open(sidecar_file, 'r') as f:
            json_data = json.load(f)
            tr_len = json_data.get('RepetitionTime', np.nan)
        if verbose:
            print(f"  found a sidecar file, '{sidecar_file.name}'; "
                  f"using its 'RepetitionTime' value of '{tr_len:0.2f}s'.")
        if not np.isnan(tr_len):
            return tr_len

    # If nothing worked, just return np.nan
    return tr_len
