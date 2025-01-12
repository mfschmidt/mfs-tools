from pathlib import Path
import multiprocessing as mp
import subprocess
import nibabel as nib
from datetime import datetime as dt
import numpy as np
import pandas as pd
import os
import psutil

from mfs_tools.library import red_on, green_on, color_off


def find_wb_command_path(suggested_path=None):
    """ Find wb_command in some likely places.
    """

    if (
            (suggested_path is not None) and
            Path(suggested_path).is_file() and
            str(suggested_path).endswith("wb_command")
    ):
        return suggested_path

    path_suggestions = [
        Path("/opt/workbench/bin_linux64/wb_command"),
        Path("/opt/workbench/2.0.1/bin_linux64/wb_command"),
        Path("/usr/local/workbench/bin_linux64/wb_command"),
        Path("/usr/local/workbench/2.0.1/bin_linux64/wb_command"),
    ]
    for p in path_suggestions:
        if p.is_file():
            return p

    return None


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
        reference_cifti_path,
        surface_file,
        save_to,
        num_procs,
        wb_command_path=None,
        work_dir=None,
):
    """ Make a distance matrix from surface files to match reference.

        :param reference_cifti_path: The path to a reference cifti file.
            This is most likely one of the BOLD dtseries files to be
            filtered by distance. It doesn't contain any location
            information per vertex, but contains a BrainModelAxis
            we use to determine which vertices are being used.
        :type reference_cifti_path: pathlib.Path

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
    ref_cifti = nib.cifti2.Cifti2Image.from_filename(reference_cifti_path)
    surf = nib.gifti.GiftiImage.from_filename(surface_file)

    # Only calculate distance to real cortical vertices that may get used.
    brain_anat_ax = ref_cifti.header.get_axis(1)
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


def compare_mats(a, b, a_name="a", b_name="b", verbose=True, preview=True):
    """ Wrap numpy allclose() with analysis of differences and commentary. """

    assert a.shape == b.shape, "Matrices must be the same size and shape."

    # Track memory usage
    mem_before = psutil.Process(os.getpid()).memory_info().rss

    # This comparison may consume a lot of RAM,
    # I assume because it compares floats.
    # So ensure there's memory available for it.
    if np.allclose(a, b):
        if verbose:
            print(green_on +
                  f"  The matrices '{a_name}' and '{b_name}' are equal." +
                  color_off)
        return_val = True
    else:
        if verbose:
            print(f"  There are mismatches between '{a_name}' and '{b_name}'.")
            if preview:
                print(f"  Top left corners, for a small preview:")
                print(np.hstack([a[:6, :6], b[:6, :6]]))

            # Extract just the values that differ between methods and compare them.
            eq = np.array(a == b, dtype=np.bool)[np.tril_indices_from(a)]
            different_a_vals = a[np.tril_indices_from(a)][~eq]
            different_b_vals = b[np.tril_indices_from(b)][~eq]

            if (len(eq) / np.sum(~eq)) < 10000:
                print(red_on +
                      f"  {np.sum(~eq):,} of {len(eq):,} values differ." +
                      color_off)
            else:
                print(green_on +
                      f"  Only 1 in {int(len(eq) / np.sum(~eq))} values differ" +
                      f" ({np.sum(~eq):,} of {len(eq):,}). " +
                      color_off)

            diff_vals = pd.DataFrame({
                a_name: np.astype(different_a_vals, np.float32),
                b_name: np.astype(different_b_vals, np.float32),
            })
            diff_vals['delta'] = diff_vals[a_name] - diff_vals[b_name]

            if (    (diff_vals['delta'].min() < -1.0) or
                    (diff_vals['delta'].max() > 1.0)
            ):
                print(red_on)
            else:
                print(green_on)
            print("  The largest difference is "
                  f"{diff_vals['delta'].min():0.2f} or "
                  f"{diff_vals['delta'].max():0.2f}{color_off}")
        return_val = False

    mem_after = psutil.Process(os.getpid()).memory_info().rss

    if verbose:
        print(f"  Mem before {mem_before / 1024 / 1024:0,.1f}MB; "
              f"Mem after {mem_after / 1024 / 1024:0,.1f}MB; "
              f"delta {(mem_after - mem_before) / 1024 / 1024:0,.1f}")

    return return_val


def regress_adjacent_cortex(bold_cifti, distance_matrix, distance_threshold):
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

    """

    # Extract just the distance between subcortical voxels and cortical vertices
    # distance_matrix is symmetrical, and we only need one side.
    # Only calculate distance to real cortical vertices that may get used.
    brain_anat_ax = bold_cifti.header.get_axis(1)
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
