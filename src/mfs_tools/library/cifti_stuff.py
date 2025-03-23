import nibabel as nib
import numpy as np
from datetime import datetime
import subprocess

from mfs_tools.library import red_on, color_off
from mfs_tools.library.file_stuff import find_wb_command_path


# This file contains functions extending nibabel functionality for doing
# common tasks with Cifti2 images.


cortical_region_names = [
    'CIFTI_STRUCTURE_CORTEX_LEFT',
    'CIFTI_STRUCTURE_CORTEX_RIGHT',
]


def get_axes_by_type(img, label_type, verbose=False):
    """ From a Cifti2 image, extract all axes matching label_type.

        If one axis is found that matches, it is returned. If more
        than one axis matches the label_type, all are returned in a list.
        If no axes match, None is returned.

        :param img: A Cifti2Image
        :type img: nibabel.Cifti2Image

        :param label_type: The type of axis sought
            This is specified as an actual type, not a string, for example:
            nib.cifti2.BrainModelAxis or nib.cifti2.ScalarAxis
        :type label_type: type  # nibabel.cifti2.cifti2_axes.Axis

        :param verbose: Pass True to get printed output for debugging
        :type verbose: bool
    """

    good_axes = list()
    for i, axis_index in enumerate(img.header.mapped_indices):
        ax = img.header.get_axis(axis_index)
        if verbose:
            print(f"  discovered axis {i}: {str(type(ax))}")
        if isinstance(ax, label_type):
            good_axes.append(ax)

    if len(good_axes) == 1:
        return good_axes[0]
    elif len(good_axes) > 1:
        return good_axes
    else:
        return None


def get_label_axes(img, verbose=False):
    """ From a Cifti2 image, extract a LabelAxis.

        This is a wrapper function to simplify :py:func:`get_axes_by_type`

        :param img: A Cifti2Image
        :type img: nibabel.Cifti2Image

        :param verbose: Pass True to get printed output for debugging
        :type verbose: bool
    """

    return get_axes_by_type(img, nib.cifti2.LabelAxis, verbose=verbose)


def get_scalar_axes(img, verbose=False):
    """ From a Cifti2 image, extract a ScalarAxis.

        This is a wrapper function to simplify :py:func:`get_axes_by_type`

        :param img: A Cifti2Image
        :type img: nibabel.Cifti2Image

        :param verbose: Pass True to get printed output for debugging
        :type verbose: bool
    """

    return get_axes_by_type(img, nib.cifti2.ScalarAxis, verbose=verbose)


def get_repetition_time(img, verbose=False):
    """ From a Cifti2 image, extract the Repetition Time (TR).

        This function extracts and returns the step size from the series axis
        in the Cifti2 image header.

        :param img: A Cifti2Image
        :type img: nibabel.Cifti2Image

        :param verbose: Pass True to get printed output for debugging
        :type verbose: bool
    """

    return get_axes_by_type(img, nib.cifti2.SeriesAxis, verbose=verbose).step


def get_series_axes(img, verbose=False):
    """ From a Cifti2 image, extract a SeriesAxis.

        This is a wrapper function to simplify :py:func:`get_axes_by_type`

        :param img: A Cifti2Image
        :type img: nibabel.Cifti2Image

        :param verbose: Pass True to get printed output for debugging
        :type verbose: bool
    """

    return get_axes_by_type(img, nib.cifti2.SeriesAxis, verbose=verbose)


def get_brain_model_axes(img, verbose=False):
    """ From a Cifti2 image, extract a BrainModelAxis.

        This is a wrapper function to simplify :py:func:`get_axes_by_type`

        :param img: A Cifti2Image
        :type img: nibabel.Cifti2Image

        :param verbose: Pass True to get printed output for debugging
        :type verbose: bool
    """

    return get_axes_by_type(img, nib.cifti2.BrainModelAxis, verbose=verbose)


def get_cortical_indices(img, verbose=False):
    """ Find the indices into img data pertaining to only cortical vertices

        Extract only cortical data from img and return them as a numpy array.

        :param img: A Cifti2Image
        :type img: nibabel.Cifti2Image

        :param verbose: Pass True to get printed output for debugging
        :type verbose: bool
    """

    anat_axis = get_brain_model_axes(img, verbose=verbose)
    region_names = [str(name) for name in anat_axis.name]
    cort_idx = np.nonzero(
        [ax in cortical_region_names for ax in region_names]
    )[0]
    if verbose:
        print(f"  found {len(cort_idx)} cortical vertices")
    return cort_idx


def get_subcortical_indices(img, verbose=False):
    """ Find the indices into img data pertaining to only subcortical vertices

        Extract only cortical data from img and return them as a numpy array.

        :param img: A Cifti2Image
        :type img: nibabel.Cifti2Image

        :param verbose: Pass True to get printed output for debugging
        :type verbose: bool
    """

    anat_axis = get_brain_model_axes(img, verbose=verbose)
    region_names = [str(name) for name in anat_axis.name]
    subcort_idx = np.nonzero(
        [ax not in cortical_region_names for ax in region_names]
    )[0]
    if verbose:
        print(f"  found {len(subcort_idx)} subcortical voxels")
    return subcort_idx


def get_cortical_data(img, verbose=False):
    """ Retain only cortical data from img

        Extract only cortical data from img and return them as a numpy array.

        :param img: A Cifti2Image
        :type img: nibabel.Cifti2Image

        :param verbose: Pass True to get printed output for debugging
        :type verbose: bool
    """

    all_data = img.get_fdata()
    anat_axis = get_brain_model_axes(img, verbose=verbose)
    cort_idx = get_cortical_indices(img, verbose=verbose)
    if len(anat_axis) == all_data.shape[0]:
        if verbose:
            print(f"  [v={len(cort_idx):,} x t={all_data.shape[1]:,}]")
        return all_data[cort_idx, :]
    elif len(anat_axis) == all_data.shape[1]:
        if verbose:
            print(f"  [t={all_data.shape[1]:,} x v={len(cort_idx):,}]")
        return all_data[:, cort_idx]
    else:
        raise ValueError(f"Cifti data are shaped {img.shape}, but I'm getting "
                         f"a {len(anat_axis)}-long axis. Not sure what to do.")


def get_subcortical_data(img, verbose=False):
    """ Retain only subcortical data from img

        Extract only subcortical data from img and return them as a numpy array

        :param img: A Cifti2Image
        :type img: nibabel.Cifti2Image

        :param verbose: Pass True to get printed output for debugging
        :type verbose: bool
    """

    all_data = img.get_fdata()
    anat_axis = get_brain_model_axes(img, verbose=verbose)
    subcort_idx = get_subcortical_indices(img, verbose=verbose)
    if len(anat_axis) == all_data.shape[0]:
        if verbose:
            print(f"  [v={len(subcort_idx):,} x t={all_data.shape[1]:,}]")
        return all_data[subcort_idx, :]
    elif len(anat_axis) == all_data.shape[1]:
        if verbose:
            print(f"  [t={all_data.shape[1]:,} x v={len(subcort_idx):,}]")
        return all_data[:, subcort_idx]
    else:
        raise ValueError(f"Cifti data are shaped {img.shape}, but I'm getting "
                         f"a {len(anat_axis)}-long axis. Not sure what to do.")


def correlate_with_workbench(bold_file, dconn_file, verbose=False):
    """
    Correlates time series in a Cifti2 BOLD file using Connectome Workbench

    This function uses a subprocess to invoke the
    Workbench command-line utility and logs the process if verbose mode is enabled.
    Time series are read from bold_file, and the connectivity matrix is written to
    dconn_file.

    :param bold_file: The path to the input BOLD file.
    :type bold_file: Union[str, pathlib.Path]
    :param dconn_file: The path to the output dense connectivity (dconn) file.
    :type dconn_file: Union[str, pathlib.Path]
    :param verbose: Flag to enable detailed print information.
    :type verbose: bool
    :return: Boolean value indicating whether the correlation process was successful.
    :rtype: bool
    """

    if verbose:
        print(f"Starting correlation at {datetime.now()}")

    correlation_command = [
        find_wb_command_path(),
        "-cifti-correlation",
        str(bold_file),
        str(dconn_file),
    ]
    corr_proc = subprocess.run(correlation_command, capture_output=True)

    if verbose:
        print(datetime.now())
        print(corr_proc.stdout.decode("utf-8"))
    if corr_proc.returncode != 0:
        print(f"{red_on}{corr_proc.stderr.decode('utf8')}{color_off}")

    return True if corr_proc.returncode == 0 else False
