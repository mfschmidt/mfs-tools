from pathlib import Path
import multiprocessing as mp
import subprocess
import nibabel as nib
from datetime import datetime as dt
import numpy as np
import os
from scipy import stats

from mfs_tools.library import red_on, color_off
from mfs_tools.library.cifti_stuff import get_cortical_indices, get_subcortical_indices
from mfs_tools.library.file_stuff import find_wb_command_path


# Define the logic for calculating distances with wb_command
def worker(cmd_path, gifti_path, tmp_path, ord_idx, vert_idx):
    """ execute wb_command on one vertex"""

    _p = subprocess.run([
        str(cmd_path), '-surface-geodesic-distance',
        str(gifti_path), str(vert_idx), str(tmp_path)
    ])
    if _p.returncode == 0:
        tmp_img = nib.GiftiImage.from_filename(str(tmp_path))
        # Use uint8 rather than float to keep memory usage reasonable.
        # Numpy truncates floats to cast to uint8, but matlab rounds.
        # We would like to match matlab so result verification is easier.
        # Numpy also overflows negatives to 255, so we need to clip them.
        # Note that even after doing it this way, the distance values from
        # python differ from matlab's in 7611 of the 441 million lh distances.
        # That's only about 1 difference in 50,000, but it's finite.
        dist_data = np.uint8(np.clip(tmp_img.darrays[0].data + 0.5, 0, 255))
        Path(tmp_path).unlink(missing_ok=True)
        return ord_idx, vert_idx, dist_data
    else:
        print(f"ERROR: {ord_idx}/{vert_idx} returned {_p.returncode}")
        _p = subprocess.run(["touch", str(tmp_path)])
        return ord_idx, vert_idx, None


def make_distance_matrix(
        reference_cifti_img,
        surface_file,
        save_to,
        num_procs,
        wb_command_path=None,
        work_dir=None,
        verbose=False,
):
    """ Make a distance matrix from a reference image to one surface file.

        This function uses wb_command at each vertex and voxel in a cifti2
        image to all locations in surface_file. It gathers each vector of
        distances into a single distance matrix. Because there are so many
        values, it saves distances in uint8 format as whole integer
        millimeters. To build a full distance matrix, this must be run on
        each hemisphere, and those matrices must be combined with
        subcortical distances.

        :param reference_cifti_img: The reference cifti image.
            This is most likely one of the BOLD dtseries images to be
            filtered by distance. It doesn't contain any location
            information per vertex, but contains a BrainModelAxis
            we use to determine which vertices are being used.
        :type reference_cifti_img: nibabel.Cifti2Image

        :param surface_file: The path to a Cifti2 surface file.
        :type surface_file: pathlib.Path

        :param save_to: The path to save the distance matrix to.
        :type save_to: pathlib.Path

        :param num_procs: The number of processes to use. Default is 1.
            This function will use the smaller value of num_procs and
            the value it gets back from multiprocessing.cpu_count().
        :type num_procs: int

        :param wb_command_path: The path to a Connectome Workbench
            wb_command executable. This function relies on wb_commend
            to determine the distance from each location to all other
            locations.
        :type wb_command_path: pathlib.Path

        :param work_dir: The working directory. Intermediate files
            from wb_command are written, read, and deleted from
            this path. Ideally, it should be fast and local.
        :type work_dir: pathlib.Path

        :param verbose: Whether to print detailed output about the process
        :type verbose: bool, optional

        :return: The distance matrix. This is a symmetrical matrix
            with ones along the diagonal. Each edge contains the
            distance in mm between its row location and column location.
            Subcortical-to-subcortical and Subcortical-to-cortical
            distances are Euclidean. Cortical-to-cortical distances
            are geodesic along the surface in surface_file.
            With around 92k locations, depending on the reference
            image, there are about 8.5 billion values. We return
            them as numpy.uint8 values to fit them into about 8GB
            rather than floats, which would consume about 33GB for
            singles or 66 GB for doubles.
        :rtype: numpy.ndarray

    """

    # Ensure wb_command is good
    wb_command = find_wb_command_path(wb_command_path)
    if wb_command is None:
        print(f"Could not find wb_command; try providing it explicitly.")
        return None

    # Load reference data (just for header) and surface geometry
    surf = nib.gifti.GiftiImage.from_filename(surface_file)

    # Only calculate distance to real cortical vertices that may get used.
    brain_anat_ax = reference_cifti_img.header.get_axis(1)
    surf_anat = surf.darrays[0].metadata.get('AnatomicalStructurePrimary', '')
    anat_map = {
        'CortexLeft': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'CortexRight': 'CIFTI_STRUCTURE_CORTEX_RIGHT',
    }
    try:
        # These are the indices we actually want to calculate from
        surf_idx = brain_anat_ax[brain_anat_ax.name == anat_map[surf_anat]]
    except KeyError:
        print(f"{red_on}Error: The surface file contains '{surf_anat}' "
              f"anatomy, which does not match 'CortexLeft' or 'CortexRight'."
              f"{color_off}")
        return None

    # Set up a temporary working directory
    if work_dir is None:
        work_dir = Path(save_to) / "wb_vec_tmp"
        work_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(work_dir / "temp_deletable_file_.nothing", "w") as f:
            f.write("")
    except PermissionError:
        print(f"The work path {work_dir} doesn't allow me to write to it.")
        return None
    except FileNotFoundError:
        print(f"The work path {work_dir} doesn't exist and I can't create it.")
        return None
    finally:
        Path(work_dir / "temp_deletable_file_.nothing").unlink(missing_ok=True)

    # Report some debuggable info
    num_procs = np.min([
        mp.cpu_count(), os.cpu_count(), num_procs,
    ])
    print(f"The system has {os.cpu_count()} CPUs, "
          f"and we're going to use {num_procs}.")

    # Define the callback handler for storing results from the worker
    vectors_modified = list()
    pcts_written = list()
    distance_matrix = np.zeros(
        (len(surf_idx.vertex), len(surf_idx.vertex)),
        dtype=np.uint8
    )
    def result_receiver(result):
        """ Mask the distances vector and integrate it into our matrix.
        """
        if result is None:
            print(f"ERROR: Something went wrong with a result.")
        elif result[2] is None:
            print(f"ERROR: Something went wrong with result {result[0]}/{result[1]}.")
        else:
            distance_matrix[:, result[0]] = result[2][surf_idx.vertex]
            vectors_modified.append(result[1])
            pct_done = int(100 * len(vectors_modified) / len(surf_idx.vertex))
            if (pct_done % 5 == 0) and (pct_done not in pcts_written):
                if pct_done % 10 == 0:
                    print(f" {(pct_done / 100):0.0%}", end="", flush=True)
                else:
                    print(" .", end="", flush=True)
                pcts_written.append(pct_done)

    # Execute wb_command distance calculators in num_threads parallel processes
    start_dt = dt.now()

    print(f"  creating {len(surf_idx.vertex):,} processes ({dt.now() - start_dt})")
    with mp.Pool(num_procs) as pool:
        for i, loc_idx in enumerate(surf_idx.vertex):
            # if (i < 5) or (i % 3000 == 0):
            #     print(f"apply_async on {i}/{loc_idx}")
            tmp_file_path = work_dir / f"dist_{loc_idx:07d}.func.gii"
            pool.apply_async(
                func=worker,
                args=(wb_command, surface_file, tmp_file_path, i, loc_idx, ),
                callback=result_receiver,
            )

        pool.close()
        pool.join()

    print()
    print(f"  processing complete ({len(vectors_modified)} vectors added)"
          f" ({dt.now() - start_dt})")

    end_dt = dt.now()
    print(f"Ran from {start_dt} to {end_dt} ({end_dt - start_dt})")

    return distance_matrix


def regress_adjacent_cortex(
        bold_cifti, distance_matrix, distance_threshold, verbose=False
):
    """ Regress out the cortical signal from subcortical voxels within a range.

        :param bold_cifti: The cifti image containing 2D BOLD data, loci x time
            This is most likely one of the BOLD dtseries files to be
            filtered by distance. It doesn't contain any location
            information per vertex, but contains a BrainModelAxis
            we use to determine which vertices are being used.
            likely [85059ish loci x t]
        :type bold_cifti: nibabel.Cifti2Image

        :param distance_matrix: A huge distance matrix
            This contains distances between every locus in the bold_cifti
            and every other distance, likely [85059ish x 85059ish]
        :type distance_matrix: numpy.ndarray

        :param distance_threshold: The path to save the distance matrix to.
        :type distance_threshold: int

        :param verbose: Whether to print detailed output about the process
        :type verbose: bool, optional
    """

    # Make a copy of BOLD data to modify with adjusted values
    adjusted_data = bold_cifti.get_fdata().copy()

    # Extract just distances between subcortical voxels and cortical vertices
    # distance_matrix is symmetrical, and we only need one side.
    # Only calculate distance to real cortical vertices that may get used.
    cort_idx = get_cortical_indices(bold_cifti)
    subcort_idx = get_subcortical_indices(bold_cifti)
    relevant_distances = distance_matrix[subcort_idx, :][:, cort_idx]
    if verbose:
        print(f"  filtered distance matrix down to "
              f"{relevant_distances.shape[0]:,} "
              f"sub-cortical voxels by {relevant_distances.shape[1]:,} "
              f"cortical vertices")

    # Determine which subcortical voxels are within 20mm of a cortical vertex.
    smallest_distances = np.min(relevant_distances, axis=1)
    outer_voxel_indices = np.where(smallest_distances <= distance_threshold)[0]
    if verbose:
        print(f"  found {len(outer_voxel_indices):,} voxels within "
              f"{distance_threshold}mm of a cortical vertex.")

    # Regress surrounding signal from each voxel near cortex
    for cifti_locus_index in outer_voxel_indices:
        # Extract all BOLD data within 20mm of this voxel
        dist_mask = distance_matrix[cifti_locus_index, :] <= distance_threshold
        nearby_bold = bold_cifti.get_fdata()[:, dist_mask]
        if nearby_bold.shape[1] > 1:
            nearby_bold = np.mean(nearby_bold, axis=1)

        # Regress surrounding BOLD from this voxel's BOLD
        voxel_index = subcort_idx[cifti_locus_index]
        y = bold_cifti.get_fdata()[:, voxel_index]
        results = stats.linregress(nearby_bold, y)
        predicted_y = results.intercept + results.slope * nearby_bold
        residuals = y - predicted_y

        # Replace the BOLD data with residuals
        adjusted_data[:, voxel_index] = residuals

    adjusted_img = nib.Cifti2Image(adjusted_data, header=bold_cifti.header)
    if verbose:
        print(f"Adjustments to {len(outer_voxel_indices):,} subcortical voxels"
              f" near cortex complete. New Cifti2 image {adjusted_img.shape}.")
    return adjusted_img
