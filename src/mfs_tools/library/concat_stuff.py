import json
import numpy as np
import nibabel as nib
from mfs_tools.library import red_on, color_off
from pathlib import Path


def concat_dtseries(files, tr_len=None):
    """ Concatenate multiple Cifti2 files together

        Requiring further documentation and thought:
        Chuck Lynch and Evan Gordons' code mean-centered each image to
        zero, then removed frames with motion FD > 0.3mm. But since we
        don't like or don't trust the high-motion frames, it doesn't
        seem wise to let them affect our mean-centering, so I reversed
        that order. This function first removes motion outliers, then
        mean-centers the data.

        This should eventually be optional (it's not yet) and configurable.
        But the current iteration is simply replicating the `Lynch 2024
        code <https://github.com/cjl2007/PFM-Depression>`_

        :param files: list of files to concatenate
        :type files: list of :pathlib.Path: objects

        :param tr_len: length of a single TR, in seconds. If not provided,
            this function will attempt to find it in the json sidecar's
            RepetitionTime field. If that fails, it will just assume 2.0s,
            which is unlikely to be correct.
        :type tr_len: float

        :return: A Cifti2 dtseries image containing data from all files,
            concatenated along the time axis, in the order of the list
            provided.
        :rtype: nibabel.Cifti2Image

    """

    # Make sure the TRs are all the same for these images
    # TODO: The smoothed json files are subsets of the unsmoothed; need to work this out
    #       to get TR length from either version of the dtseries.
    json_tr_lens = set()
    json_tr_len = None
    json_files = sorted([
        Path(str(f).replace(".dtseries.nii", ".json")) for f in files
    ])
    for json_file in json_files:
        if json_file.exists():
            with open(json_file) as fp:
                json_data = json.load(fp)
            if "RepetitionTime" in json_data:
                json_tr_lens.add(np.float32(json_data["RepetitionTime"]))

    # Report on problems and their resolutions
    if len(json_tr_lens) > 1:
        tr_str = ', '.join(f"{tr_len:0.1f}" for tr_len in list(json_tr_lens))
        print(f"{red_on}Warning: TRs differ!!! [{tr_str}]{color_off}")
    elif len(json_tr_lens) == 0:
        print(f"{red_on}Warning:{color_off} no RepetitionTime in json files.")
    else:  # len(json_tr_lens) == 1
        json_tr_len = json_tr_lens.pop()

    if json_tr_len is None and tr_len is None:
        print(f"{red_on}Warning:{color_off} no RepetitionTime in json files, "
              f"and no tr_len was provided. Just making up TR=2.0.")
        tr_len = 2.0
    elif json_tr_len is None:
        print(f"Using provided TR={tr_len:0.1f}. No RepetitionTime in jsons.")
    elif tr_len is None:
        print(f"Using TR={json_tr_len:0.1f} from jsons. No tr_len provided.")
        tr_len = json_tr_len
    elif json_tr_len == tr_len:
        print(f"tr_len '{tr_len:0.1f}' matches jsons, '{json_tr_len:0.1f}'")
    else:
        print(f"{red_on}Warning:{color_off} tr_len '{tr_len:0.1f}' does not "
              f"match jsons, and will be overridden by '{json_tr_len:0.1f}', "
              "the value in them.")
        tr_len = json_tr_len

    # Load all dtseries images
    nii_files = sorted(files)
    all_func_images = [nib.Cifti2Image.from_filename(f) for f in nii_files]
    # Axis 0 is the time axis, in seconds; we'll build our own
    # Axis 1 is the region axis, we need to copy it for our matching image
    first_cifti_axis_1 = all_func_images[0].header.get_axis(1)

    # Concatenate data
    all_run_data = np.vstack([f.get_fdata() for f in all_func_images])

    # Build a new Cifti2Image from concatenated data
    all_run_cifti_axis_0 = nib.cifti2.SeriesAxis(
        start=0, step=tr_len, size=all_run_data.shape[0]
    )
    all_run_cifti_axis_1 = first_cifti_axis_1
    all_run_img = nib.cifti2.Cifti2Image(
        all_run_data, (all_run_cifti_axis_0, all_run_cifti_axis_1)
    )
    all_run_img.update_headers()

    # Return the new dtseries image
    return all_run_img


