import numpy as np
import nibabel as nib
from mfs_tools.library import red_on, color_off

from .utility_stuff import get_tr_len


def concat_niftis(
        arrays_to_stack,
        all_func_images,
        tr_len=None,
        verbose=False,
):
    """ Concatenate data from multiple Nifti files together

        :param arrays_to_stack: list of arrays to concatenate
        :type arrays_to_stack: list of numpy.array objects

        :param all_func_images: The original images to be concatenated.
        :type all_func_images: iterable of :nib.Cifti2Image: images

        :param tr_len: length of a single TR, in seconds. If not provided,
            this function will attempt to find it in the json sidecar's
            RepetitionTime field. If that fails, it will just assume 1.0s,
            which is unlikely to be correct.
        :type tr_len: float

        :param verbose: Boolean flag indicating whether to output detailed logs
            during execution.
        :type verbose: bool

        :return: A Nifti2 image containing data from all files,
            concatenated along the time axis, in the order of the list
            provided.
        :rtype: nibabel.Nifti2Image

    """

    # Nifti files are 4D, so are hstacked over the 4th dimension
    all_run_data = np.hstack(arrays_to_stack)
    if verbose:
        print(f"Final Nifti data shaped {all_run_data.shape}")

    # Use the original Nifti file's header to build a new Nifti
    all_run_img = nib.nifti2.Nifti2Image(
        all_run_data,
        all_func_images[0].affine, header=all_func_images[0].header
    )

    # Return the new Nifti2Image
    return all_run_img


def concat_ciftis(
        arrays_to_stack,
        all_func_images,
        tr_len=None,
        verbose=False,
):
    """ Concatenate multiple Cifti2 files together

        :param arrays_to_stack: list of arrays to concatenate
        :type arrays_to_stack: list of :np.array: objects

        :param all_func_images: The original images to be concatenated.
        :type all_func_images: iterable of :nib.Cifti2Image: images

        :param tr_len: length of a single TR, in seconds. If not provided,
            this function will attempt to find it in the json sidecar's
            RepetitionTime field. If that fails, it will just assume 1.0s,
            which is unlikely to be correct.
        :type tr_len: float

        :param verbose: Boolean flag indicating whether to output detailed logs
            during execution.
        :type verbose: bool

        :return: A Cifti2 dtseries image containing data from all files,
            concatenated along the time axis, in the order of the list
            provided.
        :rtype: nibabel.Cifti2Image

    """

    # Cifti2 dtseries images are [tr x locus], are vstacked over the time axis.
    all_run_data = np.vstack(arrays_to_stack)
    if verbose:
        print(f"Final Nifti data shaped {all_run_data.shape}")

    # Axis 0 is the time axis, in seconds; we'll build our own
    # This axis ignores the fact that some frames were removed.
    # It just pretends everything smoothly progresses across time.
    all_run_cifti_axis_0 = nib.cifti2.SeriesAxis(
        start=0, step=tr_len, size=all_run_data.shape[0]
    )
    # Axis 1 is the region axis, we need to copy it for our matching image
    first_cifti_axis_1 = all_func_images[0].header.get_axis(1)
    all_run_cifti_axis_1 = first_cifti_axis_1

    # Build a new Cifti2Image from concatenated data
    all_run_img = nib.cifti2.Cifti2Image(
        all_run_data, (all_run_cifti_axis_0, all_run_cifti_axis_1)
    )
    all_run_img.update_headers()

    # Return the new dtseries image
    return all_run_img


def concat_images(
        files,
        temporal_masks=None,
        temporal_mean=0.0,
        temporal_sd=1.0,
        tr_len=None,
        verbose=False,
):
    """ Concatenate multiple Nifti/Cifti files together along the time axis

        :param files: list of files to concatenate
        :type files: list of :pathlib.Path: objects

        :param temporal_masks: A mask for each BOLD file, indicating which
            frames to use (True) and which high motion frames to exclude
            (False). The length must match the length of the file list,
            and the length of each mask must match the file's 4th dimension.
        :type temporal_masks: iterable

        :param temporal_mean: By default, we normalize each voxel or vertex
            to a mean of 0.0 with a standard deviation of 1.0. This parameter
            overrides the mean.
        :type temporal_mean: float

        :param temporal_sd: By default, we normalize each voxel or vertex
            to a mean of 0.0 with a standard deviation of 1.0. This parameter
            overrides the SD.
        :type temporal_sd: float

        :param tr_len: length of a single TR, in seconds. If not provided,
            this function will attempt to find it in the json sidecar's
            RepetitionTime field. If that fails, it will just assume 1.0s,
            which is unlikely to be correct.
        :type tr_len: float

        :param verbose: Boolean flag indicating whether to output detailed logs
            during execution.
        :type verbose: bool

        :return: An image containing data from all files,
            concatenated along the time axis, in the order of the list
            provided.
        :rtype: nibabel.cifti2.Cifti2Image or nibabel.Nifti1Image

    """

    # Load all dtseries images
    nii_files = sorted(files)
    all_func_images = [nib.load(f) for f in nii_files]


    # Report on problems and their resolutions
    discovered_tr_len = None
    if not np.isfinite(tr_len):
        # Make sure the TRs are all the same for these images
        tr_lens = set([get_tr_len(img) for img in all_func_images])
        if len(tr_lens) > 1:
            tr_str = ', '.join(f"{tr_len:0.1f}" for tr_len in list(tr_lens))
            print(f"{red_on}Warning: TRs differ!!! [{tr_str}]{color_off}")
        elif len(tr_lens) == 0:
            print(f"{red_on}Warning:{color_off} no RepetitionTime in images or json files.")
        else:  # len(json_tr_lens) == 1
            discovered_tr_len = tr_lens.pop()

        if discovered_tr_len is None:  # still
            print(f"{red_on}Warning:{color_off} no RepetitionTime in images, "
                  f"and no tr_len was provided. Just making up TR=1.0.")
            discovered_tr_len = 1.0
        tr_len = discovered_tr_len
    else:
        print(f"Using argument-provided TR={tr_len:0.2f}.")

    # Concatenate data
    arrays_to_stack = list()
    for i, img in enumerate(all_func_images):
        # Requiring further documentation and thought:
        #     Chuck Lynch and Evan Gordons' code mean-centered each image to
        #     zero, then removed frames with motion FD > 0.3mm. But since we
        #     don't like or don't trust the high-motion frames, it doesn't
        #     seem wise to let them affect our mean-centering, so I reversed
        #     that order. This function first removes motion outliers, then
        #     mean-centers the data.
        array_to_stack = img.get_fdata()
        original_shape = array_to_stack.shape
        if (
                np.sum(temporal_masks[i]) == array_to_stack.shape[-1] or
                np.sum(temporal_masks[i]) == array_to_stack.shape[0]
        ):
            if verbose:
                print(f"  data appear already masked, ignoring the mask.")
            # leave the data alone, no need to mask anything
        elif (
                len(temporal_masks[i]) == array_to_stack.shape[-1] or
                len(temporal_masks[i]) == array_to_stack.shape[0]
        ):
            array_to_stack = array_to_stack[temporal_masks[i]]

        if np.sum(temporal_masks[i]) == array_to_stack.shape[0]:
            time_axis = 0
        elif np.sum(temporal_masks[i]) == array_to_stack.shape[-1]:
            time_axis = len(array_to_stack.shape) - 1
        else:
            raise ValueError(
                f"Temporal mask ({temporal_masks[i].shape}-shaped, "
                f"{np.sum(temporal_masks[i])} good) does not match "
                f"{img.shape}-shaped data."
            )

        # Whatever the desired mean, first shift to zero for scaling
        array_to_stack = array_to_stack - np.mean(array_to_stack, axis=time_axis)
        # if verbose:
        #     zero_denom_mask = np.array(
        #         np.std(array_to_stack, axis=time_axis) == 0.0,
        #         dtype=bool
        #     )
        #     print(f"In SD of array, {np.sum(zero_denom_mask)}"
        #           f" denominators are zero. Numerators in those same positions"
        #           f" sum to {np.sum(array_to_stack[:, zero_denom_mask]):0.4f}.")
        std_dev = np.std(array_to_stack, axis=time_axis)
        array_to_stack = np.divide(
            array_to_stack, std_dev, where=std_dev != 0.0,
        )
        # if verbose:
        #     print(f"Final normalized array has "
        #           f"{np.sum(np.isnan(array_to_stack))} nans, "
        #           f"{np.sum(array_to_stack == 0.0)} zeros, "
        #           f"{np.sum(np.isinf(array_to_stack))} infs.")

        # Now we can scale and shift locus-wise normalized data
        array_to_stack = array_to_stack * temporal_sd + temporal_mean
        arrays_to_stack.append(array_to_stack)
        if verbose:
            print(f"  image data went from {original_shape} to {array_to_stack.shape}")

    if (
        isinstance(all_func_images[0], nib.Nifti1Image) or
        isinstance(all_func_images[0], nib.Nifti2Image)
    ):
        return concat_niftis(arrays_to_stack, all_func_images, tr_len)
    elif (
        isinstance(all_func_images[0], nib.Cifti2Image)
    ):
        return concat_ciftis(arrays_to_stack, all_func_images, tr_len)
