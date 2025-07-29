#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from datetime import datetime

from debugpy.adapter.servers import dont_wait_for_first_connection
from templateflow import api as tflow
from scipy import stats
from scipy.spatial.distance import cdist
from nibabel.affines import apply_affine
import subprocess

import mfs_tools as mt
from mfs_tools.library import yellow_on, red_on, green_on, color_off
from mfs_tools.library.bids_stuff import get_bids_key_pairs


"""
This script reads in pre-processed BOLD data from xcp-d,
builds Evan Gordon-style dense connectomes,
and saves them out as network files, ready for infomap.
This script will consume about 170GB RAM, which is why
it is saved as its own module. The following steps, like
running infomap on these outputs, can be run in far less
memory and distributed differently across computing
resources.
"""


def get_arguments():
    """ Parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="From a clean BOLD Cifti2 dtseries, "
                    "create a set of Pajek networks, ready for infomap."
    )
    parser.add_argument(
        "input_file",
        help="Cifti2 dtseries file containing clean BOLD data",
    )
    parser.add_argument(
        "--output-path",
        default=".",
        help="A path for writing multiple directories and files into. "
             "If not provided, files will be written to the current directory.",
    )
    parser.add_argument(
        "--distance-matrix-file",
        default="na",
        help="The distance matrix will be used to regress cortical signal "
             "from near-by subcortical voxels, and also to zero out "
             "connectivity from near-by nodes. If not provided, a distance "
             "matrix will be created from templateflow templates.",
    )
    parser.add_argument(
        "--exclude-structures",
        nargs="+",
        type=str,
        help="We can exclude anatomical structures from connectivity and "
             "infomap. Lynch excludes 'BRAIN_STEM'. You can add structures "
             "from Cifti2 brain_model_axes here to avoid their influence "
             "on the networks.",
    )
    parser.add_argument(
        "--smoothing-kernel-sizes",
        nargs="+",
        type=float,
        help="If this is not specified, smoothing kernels will be created "
             "with full-width-at-half-maxima of [0.85, 1.70, 2.55] "
             "which were used in the Lynch paper. All specified "
             "sizes in this argument will be run, resulting in multiple "
             "output files and multiple downstream network atlases.",
    )
    parser.add_argument(
        "--graph-densities",
        nargs="+",
        type=float,
        help="Before running infomap, most connectivity matrix edges are set "
             "to zero. By default, we'll run infomap on nine network densities"
             ", (0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001). "
             "These specify the proportion of edges to keep at each locus. "
             "You can provide your own list with --graph-densities.",
    )
    parser.add_argument(
        "--distance-to-cortex-threshold",
        default=10,
        type=int,
        help="Specify the distance whereby all subcortical voxels within "
             "this distance of a cortical vertex will have the cortical "
             "signal regressed out.",
    )
    parser.add_argument(
        "--distance-threshold-exclusion",
        default=10,
        type=int,
        help="Specify the distance whereby all network edges with nodes "
             "nearer than this will be set to zero. ",
    )
    parser.add_argument(
        "--infomap-reps",
        default=50,
        type=int,
        help="How many iterations of infomap should be run? ",
    )
    parser.add_argument(
        "--random-seed",
        default=44,
        type=int,
        help="Optionally, set a seed for random number generation.",
    )
    parser.add_argument(
        "--min-network-size",
        default=0,
        type=int,
        help="After generating networks, we can optionally remove any "
             "network clusters smaller than this size. The space they "
             "filled will then be re-labeled with neighboring networks' "
             "labels. Lynch, et al., used 50 as a lower threshold for "
             "network size.",
    )
    parser.add_argument(
        "--wb-command-path",
        default=None,
        help="This script attempts to locate wb_command in expected "
             "paths. If you've installed it somewhere else, or if you get "
             "errors about it not being found, you can specify its path "
             "here to avoid the trouble.",
    )
    parser.add_argument(
        "--infomap-path",
        default=None,
        help="Infomap is installed as a python package, with pip, and is a "
             "dependency of mfs_tools, so it should already be available. "
             "If you're getting errors about it not being found, you can "
             "specify the path here to avoid the trouble.",
    )
    parser.add_argument(
        "--num-procs",
        default=1,
        type=int,
        help="Use --num-procs to specify cores for multiprocessing.",
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
        help="Use --verbose for more verbose output. This is terribly "
             "useful to review included files, dropped frames, etc.",
    )

    args = parser.parse_args()

    setattr(args, "input_file", Path(args.input_file))
    setattr(args, "output_path", Path(args.output_path))

    setattr(args, "wb_command_path",
            mt.find_wb_command_path(args.wb_command_path))
    setattr(args, "infomap_path",
            mt.find_infomap_path(args.infomap_path))
    if args.smoothing_kernel_sizes is None:
        setattr(args, "smoothing_kernel_sizes", [0.85, 1.70, 2.55, ])
    if args.graph_densities is None:
        setattr(args, "graph_densities",
                [0.0001, 0.0002, 0.0005, 0.001,
                 0.002, 0.005, 0.01, 0.02, 0.05, ],
        )
    if args.exclude_structures is None:
        setattr(args, "exclude_structures", list())

    return args


def build_distance_matrix(
        bold_img, lh_surf_file, rh_surf_file,
        num_procs=1, wb_command_path=None, output_path=".", dtype=np.uint8,
        verbose=False,
):
    """ """

    # Fetch some prerequisites

    # Build individualized distance matrices with an individual's anatomical
    # surface files.
    surface_files = {
        'lh': lh_surf_file, 'rh': rh_surf_file,
    }
    """
    surface_files = {
        'lh': tflow.get(
            'fsLR', suffix='midthickness', extension='surf.gii', hemi='L'
        ),
        'rh': tflow.get(
            'fsLR', suffix='midthickness', extension='surf.gii', hemi='R'
        ),
    }
    """
    brain_ax = mt.get_brain_model_axes(bold_img)
    print(f"Length of cifti2 brain_axis: {len(brain_ax)}")


    # Build a distance matrix for each cortical hemisphere.
    distance_matrices = dict()
    (Path(output_path) / "tmp").mkdir(exist_ok=True, parents=True)
    for (hemi) in ('lh', 'rh',):
        mat_file = Path(output_path) / "tmp" / f"distance_matrix_{hemi}.npy"
        if mat_file.exists():
            distance_matrices[hemi] = np.load(mat_file)
            print(f"  loaded {distance_matrices[hemi].shape}-shaped "
                  f"'{hemi}' distance matrix")
        else:
            distance_matrices[hemi] = mt.make_distance_matrix(
                bold_img,
                surface_files[hemi],
                Path(output_path),
                dtype=dtype,
                num_procs=num_procs,
                wb_command_path=mt.find_wb_command_path(wb_command_path),
                work_dir=Path(output_path) / "tmp",
            )
            print(f"built {distance_matrices[hemi].shape}-shaped "
                  f"'{hemi}' distance matrix")
            np.save(mat_file, distance_matrices[hemi])

    # Build a subcortical-to-cortical distance matrix for each hemisphere
    # Extract the 3D Cartesian coordinates of all surface vertices
    lh_surface_img = nib.gifti.gifti.GiftiImage.from_filename(surface_files['lh'])
    rh_surface_img = nib.gifti.gifti.GiftiImage.from_filename(surface_files['rh'])
    print("Gifti Surface coordinates: "
          f"[{lh_surface_img.darrays[0].data.shape} + "
          f"{rh_surface_img.darrays[0].data.shape}]")

    # Extract the vertex indices into the mapped BOLD data
    anat_map = {
        'CortexLeft': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'CortexRight': 'CIFTI_STRUCTURE_CORTEX_RIGHT',
    }
    lh_surf_anat = lh_surface_img.darrays[0].metadata.get('AnatomicalStructurePrimary', '')
    lh_surf_idx = brain_ax[brain_ax.name == anat_map[lh_surf_anat]]
    lh_surf_coords = lh_surface_img.darrays[0].data[lh_surf_idx.vertex, :]
    print(f"Just vertices in {str(type(lh_surf_idx))} {lh_surf_anat}: {len(lh_surf_idx)}")
    rh_surf_anat = rh_surface_img.darrays[0].metadata.get('AnatomicalStructurePrimary', '')
    rh_surf_idx = brain_ax[brain_ax.name == anat_map[rh_surf_anat]]
    rh_surf_coords = rh_surface_img.darrays[0].data[rh_surf_idx.vertex, :]
    print(f"Just vertices in {str(type(rh_surf_idx))} {rh_surf_anat}: {len(rh_surf_idx)}")

    # Get the subcortical voxels, too, from a volumetric grid rather than vertices.
    # Note that python's voxel locations are consistently shifted relative to
    # matlab's. Python's x values are ml+2mm, y=ml+2mm, z=ml-2mm.
    # Maybe 0-based vs 1-based indexing, then multiplied by the affine?
    # Maybe it's start of voxel vs end of voxel, not center?
    # It's all relative, so the subcortex-to-subcortex distances are identical,
    # and distance differences are only between subcortical and cortical.
    # I've added a 1 below to ensure these data match Lynch's for now.
    # I also checked for left-right biases in another notebook to see
    # whether the Lynch matlab distance matrix or the Schmidt python distance
    # matrix have more left/right bias. Lynch's is less biased, so I think
    # adding 1 is the correct approach.
    ctx_labels = list(anat_map.values())
    subcortical_coordinates = apply_affine(
        brain_ax.affine,
        brain_ax.voxel[~np.isin(brain_ax.name, ctx_labels)] + 1,
    )
    print("Nifti subcortical coordinates: "
          f" = {subcortical_coordinates.shape}")
    print("Cifti cortical coordinates: "
          f" = {lh_surf_coords.shape} & {rh_surf_coords.shape}")

    whole_brain_coordinates = np.vstack([
        lh_surf_coords, rh_surf_coords, subcortical_coordinates
    ])
    print("Whole brain coordinates: "
          f" = {whole_brain_coordinates.shape}")

    # (2) Now, calculate the Euclidean distances between subcortical voxels
    #     and everywhere. Doing this in three chunks avoids duplication of
    #     memory, but takes a little longer (seconds, not minutes).

    sc_to_lh_dist = dtype(np.clip(
        cdist(subcortical_coordinates, lh_surf_coords) + 0.5,
        0, 255
    ))
    sc_to_rh_dist = dtype(np.clip(
        cdist(subcortical_coordinates, rh_surf_coords) + 0.5,
        0, 255
    ))
    sc_to_sc_dist = dtype(np.clip(
        cdist(subcortical_coordinates, subcortical_coordinates) + 0.5,
        0, 255
    ))
    print(f"Euclidean distance matrices: {sc_to_lh_dist.shape}, {sc_to_rh_dist.shape}, {sc_to_sc_dist.shape}")

    # We've calculated most of the distance matrix piece-meal, and need to
    # fill in some missing sections with fake large values that will exceed
    # the distance threshold and cause removal of those connectivity values later.
    # Start pasting lh-lh and rh-rh distances into a complete distance matrix
    # where anything between lh and rh is "large".
    # The largest uint8 is 2^8 == 256, which is big enough to get masked later.
    # [ [ lh  ] [255s ] [lh-sc] ]
    # [ [255s ] [ rh  ] [rh-sc] ]
    # [ [sc-lh] [sc-rh] [ sc  ] ]

    # (3) Create two filler blocks of distances safely larger than threshold
    top_mid_lh_rh = 255 * np.ones(
        (distance_matrices['lh'].shape[0], distance_matrices['rh'].shape[1]),
        dtype=dtype
    )
    mid_mid_rh_lh = 255 * np.ones(
        (distance_matrices['rh'].shape[0], distance_matrices['lh'].shape[1]),
        dtype=dtype
    )

    # Put complete distance matrix together with real and filler data
    py_complete = np.vstack([
        np.hstack([distance_matrices['lh'], top_mid_lh_rh, sc_to_lh_dist.T, ]),
        np.hstack([mid_mid_rh_lh, distance_matrices['rh'], sc_to_rh_dist.T, ]),
        np.hstack([sc_to_lh_dist, sc_to_rh_dist, sc_to_sc_dist, ])  # bottom row of blocks
    ])

    return py_complete


def remove_cortical_influence_from_subcortex(
    bold_img, distance_img, distance_threshold=20, verbose=False,
):
    """ """

    # Only calculate distance to real cortical vertices that may get used.
    cort_idx = mt.get_cortical_indices(bold_img)
    subcort_idx = mt.get_subcortical_indices(bold_img)
    distance_matrix = distance_img.get_fdata().astype(np.uint8)
    relevant_distances = distance_matrix[subcort_idx, :][:, cort_idx]
    if verbose:
        print(f"  filtered distance matrix to {relevant_distances.shape[0]:,} "
          f"sub-cortical voxels by {relevant_distances.shape[1]:,} "
          f"cortical vertices")

    # Determine which subcortical voxels are within distance_threshold of a cortical vertex.
    smallest_distances = np.min(relevant_distances, axis=1)
    outer_voxel_indices = np.where(smallest_distances <= distance_threshold)[0]
    if verbose:
        print(f"  {len(outer_voxel_indices)} voxels are within "
              f"{distance_threshold}mm of a cortical vertex.")

    adjusted_data = bold_img.get_fdata().copy()
    for cifti_locus_index in outer_voxel_indices:
        # Extract all BOLD data within distance_threshold of this voxel
        near_ctx_mask = distance_matrix[cifti_locus_index, :] <= distance_threshold
        near_ctx_bold = bold_img.get_fdata()[:, near_ctx_mask]
        if near_ctx_bold.shape[1] > 1:
            near_ctx_bold = np.mean(near_ctx_bold, axis=1)

        # Regress surrounding BOLD from this voxel's BOLD
        voxel_index = subcort_idx[cifti_locus_index]
        y = bold_img.get_fdata()[:, voxel_index]
        results = stats.linregress(near_ctx_bold, y)
        predicted_y = results.intercept + results.slope * near_ctx_bold
        residuals = y - predicted_y
        if np.sum(np.isnan(residuals)) > 0:
            print(f"{np.sum(np.isnan(residuals))} nan values in residuals.")
        # Replace the BOLD data with residuals
        adjusted_data[:, voxel_index] = residuals

    # Return the data, packaged into a Cifti2Image to match bold_img
    return nib.Cifti2Image(
        adjusted_data, header=bold_img.header,
    )


def build_connectivity_matrix(
        bold_img, dist_img, good_loci, dist_threshold, dtype=np.float32, verbose=False
):
    """ """

    # bold_data = bold_img.get_fdata()[:, good_loci]
    # dist_data = dist_img.get_fdata()[good_loci, :][:, good_loci]
    bold_brain_axis = mt.get_brain_model_axes(bold_img)
    connectivity = mt.correlate_bold(
        bold_img.get_fdata(dtype=dtype), strip_size=8*1024, verbose=verbose
    )

    # Remove the diagonal (set to zero)
    connectivity[np.diag_indices_from(connectivity)] = 0.0
    if verbose:
        print(f"  {np.sum(connectivity > 0.0):,} edges remain "
              "after removing diagonal")

    # Remove edges where nodes are nearer than the distance threshold
    connectivity[dist_img.get_fdata(dtype=np.float32) <= dist_threshold] = 0.0
    if verbose:
        print(f"  {np.sum(connectivity > 0.0):,} edges remain "
              "after removing local edges")

    # Remove any NaN values
    connectivity[np.isnan(connectivity)] = 0.0
    if verbose:
        print(f"  {np.sum(connectivity > 0.0):,} edges remain "
              "after removing NaNs")

    dconn_img = nib.Cifti2Image(
        connectivity, (bold_brain_axis, bold_brain_axis)
    )
    return dconn_img


def build_pajek_networks(
        bold_img, dist_img, connectivity_img, good_loci, dist_threshold,
        graph_density=0.002, verbose=False
):
    """ """

    # Normally, we would load data with nibabel's get_fdata(), but for these
    # matrices, with 8.3 billion values, it would load with the default
    # float64 data type, consuming 67GB of RAM. In practice, I watched RAM
    # usage climb from 8GB to 192GB loading a single distance matrix.
    # We need to carefully control data types to avoid crashing the
    # systems running this code. By using asanyarray(dataobj), both
    # distance_matrix and connectivity fully loaded in under 80GB.
    distance_matrix = np.asanyarray(dist_img.dataobj)
    dist_img.uncache()  # use extracted matrix, no longer need dist_img's cache
    if verbose:
        print(f"  Original distance matrix is shaped {distance_matrix.shape}")
        print(f"  removing subcort-subcort edges")
    # Set subcort-subcort distances to zero.
    subcort_indices = [idx for idx in mt.get_subcortical_indices(bold_img)
                       if idx in good_loci]
    distance_matrix[np.ix_(subcort_indices, subcort_indices)] = 0.0
    if verbose:
        print(f"  D1 (sans subcortical edges) is shaped {distance_matrix.shape}")
        print(f"  D1 has {np.sum(distance_matrix > 0.0):,} non-zero edges.")
        print(f"  D1 has {np.sum(distance_matrix == 0.0):,} zero edges.")
        print(f"  D1 has {np.sum(distance_matrix <= dist_threshold):,} local edges.")
    # These numbers, too, match matlab exactly (but only after using the np.ix_ function)

    # Trim the distance matrix to match our filtered vertices.
    distance_matrix = distance_matrix[good_loci, :][:, good_loci]
    if verbose:
        print(f"  filtering D1 by acceptable structures")
        print(f"  D2 (with only 'good' loci) is shaped {distance_matrix.shape}")
        print(f"  D2 has {np.sum(distance_matrix > 0.0):,} non-zero edges.")
        print(f"  D2 has {np.sum(distance_matrix == 0.0):,} zero edges.")
        print(f"  D2 has {np.sum(distance_matrix <= dist_threshold):,} local edges.")


    # Next, go through the connectivity matrix, with local edges, diagonals,
    # and NaNs removed, and sort each column to find
    # the highest connectivity edges for each locus.
    # For each column, set the top edges (thresholded by graph density) to True,
    # leaving everything else False.
    # This ensures that even weakly connected loci maintain some
    # connectivity to their hub-like partners.
    # Remove all but the very most significant edges, by graph density value
    print(f"Starting infomap with graph density {graph_density:0.4f} "
          f"at {datetime.now()}...")
    total_edges_kept = 0  # not accurate, some overlap, but useful for comparisons
    log_notes = list()

    connectivity = np.asanyarray(connectivity_img.dataobj)[good_loci, :][:, good_loci]
    connectivity_img.uncache()  # clear memory from original img cache
    if verbose:
        print(f"  connectivity has {np.sum(connectivity > 0.0):,} positive edges.")

    # Create a mask with all False until we decide what to keep.
    hi_conn_mask = np.zeros(connectivity.shape, dtype=bool)
    if verbose:
        print(f"  matching boolean mask has {len(hi_conn_mask.ravel()):,} zero edges.")
    # for each column in the connectivity matrix, find the highest
    # correlations and add those edges to the mask for that location's
    # column AND row.
    for i_n in range(connectivity.shape[1]):
        if np.any(connectivity[:, i_n]):
            ordered_connectivity = np.flip(np.argsort(connectivity[:, i_n]))
            num_to_keep = int(np.ceil(graph_density * len(ordered_connectivity)))
            total_edges_kept += num_to_keep
            log_notes.append(
                f"Keeping {num_to_keep:,} edges for density "
                f"{graph_density:0.4f}, col {i_n}"
            )
            hi_conn_mask[ordered_connectivity[:num_to_keep], i_n] = True
            hi_conn_mask[i_n, ordered_connectivity[:num_to_keep]] = True

    # We built the matrix up symmetrically, so now that it's complete,
    # we only need half of it. Delete the lower triangle, then
    # find the indices of the masked edges.
    hi_conn_mask[np.tril_indices_from(hi_conn_mask)] = False
    hi_conn_idx = np.argwhere(hi_conn_mask)
    print(f"Finished density {graph_density:0.4f} at {datetime.now()}...")

    return hi_conn_idx, log_notes


def build_infomap_networks():
    """ """


def main():
    """ main entry point

        The main function orchestrates all network construction processes
        for command line usage.
    """

    args = get_arguments()
    warnings = list()
    errors = list()

    dt_start = datetime.now()
    if args.verbose:
        print(f"Starting mfs_make_infomap_networks.py at {dt_start}")

    if not args.input_file.exists():
        errors.append(f"File '{str(args.input_file)}' does not exist.")

    if args.verbose or args.dry_run:
        print(f"Running mfs_make_infomap_networks on '{str(args.input_file)}', "
              "with verbose output")
        if args.input_file.exists():
            print(f" - '{str(args.input_file)}' : "
                  f"{mt.file_info(args.input_file)[1]}")

    if not args.output_path.exists():
        errors.append(f"Output path '{str(args.output_path)}' doesn't "
                      f"exist. Please create it before running.")
    elif str(args.output_path) == ".":
        warnings.append(f"Output path has not been set. "
                        f"This is fine, and will write everything to your "
                        f"working directory, but probably isn't what you want."
                        f" You can use --output-path to specify another path.")
    else:
        print(f"Saving matrices and networks to '{str(args.output_path)}'")
        if args.output_path.is_file():
            errors.append(f"The output path, '{str(args.output_path)}'"
                          f" is a file. It should be a directory.")

    for warning in warnings:
        print(f"{yellow_on}Warning: {warning}{color_off}")
    for error in errors:
        print(f"{red_on}Error: {error}{color_off}")

    # Take any opportunity to exit before doing any real work.
    if len(errors) > 0:
        return 1
    if args.dry_run:
        return 0

    # Fetch some prerequisites
    template_surface_files = {
        'lh': tflow.get(
            'fsLR', suffix='midthickness', extension='surf.gii', hemi='L'
        ),
        'rh': tflow.get(
            'fsLR', suffix='midthickness', extension='surf.gii', hemi='R'
        ),
    }
    # The BOLD file from xcp_d is like "sub-P10002_ses-10002_task-rest_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    # The surface file is like "sub-P10003_ses-10003_space-fsLR_den-32k_hemi-L_desc-hcp_midthickness.surf.gii"
    surf_file_template = "sub-{0}_ses-{1}_space-fsLR_den-32k_hemi-{2}_desc-hcp_midthickness.surf.gii"
    key_pairs = get_bids_key_pairs(args.input_file.name)
    surface_files = {
        'lh': (args.input_file.parent / ".." / "anat" /
               surf_file_template.format(key_pairs['sub'], key_pairs['ses'], 'L')),
        'rh': (args.input_file.parent / ".." / "anat" /
               surf_file_template.format(key_pairs['sub'], key_pairs['ses'], 'R')),
    }

    np.random.seed(args.random_seed)

    ## Step 1. Load the concatenated and cleaned BOLD data.
    if args.verbose:
        print("=" * 80)
        print(f"| Step 1. Starting mfs_make_infomap_networks.py at {dt_start}")
        print("=" * 80)
    bold_img = nib.cifti2.Cifti2Image.from_filename(args.input_file)
    bold_brain_axis = mt.get_brain_model_axes(bold_img)
    bold_series_axis = mt.get_series_axes(bold_img)
    if args.verbose:
        print(f"  loaded BOLD with {len(bold_brain_axis)} loci, "
              f"{len(bold_series_axis)} time points. "
              f"({len(bold_brain_axis) * len(bold_series_axis) / 1000000:0,.0f}M values)")

    ## Step 2. Load the distance matrix (or build it).
    if args.verbose:
        print("=" * 80)
        print(f"| Step 2. Load or build individual distance matrix at {dt_start}")
        print("=" * 80)
    if args.distance_matrix_file == "na":
        setattr(args, "distance_matrix_file",
                args.output_path / "fsLR_distance.dconn.nii")
    if Path(args.distance_matrix_file).exists():
        # Load distance matrix and validate it.
        distance_img = nib.cifti2.Cifti2Image.from_filename(
            args.distance_matrix_file
        )
        # distance_matrix = distance_img.get_fdata()
        if args.verbose:
            print(f"  loaded distance matrix with {distance_img.shape[0]} "
                  f"loci."
                  f"({distance_img.shape[0] * distance_img.shape[0] / 1000000:0,.0f}M values)")
    else:
        # Generate and save a distance matrix.
        # I checked and verified that xcp_d's surfaces in the anat directory
        # are unique to each subject and different from the fsLR template.
        # That tells me that they are each individual's cortical surface,
        # not truly "fsLR" or "hcp" suggested by the named file.
        # If we look into the fmriprep anat, the midthickness.surf.gii
        # files are also unique to each subject, but with 140k vertices.

        if args.verbose:
            print(f"  building distance matrix from BOLD and {surface_files['lh'].name} and {surface_files['rh'].name}...")
        distance_matrix = build_distance_matrix(
            bold_img,
            surface_files['lh'],
            surface_files['rh'],
            num_procs=args.num_procs,
            wb_command_path=args.wb_command_path,
            output_path=args.output_path,
            verbose=args.verbose,
        )
        distance_img = nib.cifti2.Cifti2Image(
            distance_matrix,
            (bold_brain_axis, bold_brain_axis,)
        )
        distance_img.to_filename(args.distance_matrix_file)
        if args.verbose:
            print(f"  built distance matrix with {distance_img.shape[0]} "
                  f"loci. ({sys.getsizeof(distance_img.dataobj) / 1000000:0,.0f}M bytes)")

    ## Step 3. Regress Cortical signal from Subcortical voxels.
    if args.verbose:
        print("=" * 80)
        print(f"| Step 3. Remove cortical signal from subcortical voxels at {dt_start}")
        print("=" * 80)
    if args.verbose:
        print(f"Starting to regress cortical signal from subcortical voxels.")
    bold_regr_file = (
            args.output_path /
            args.input_file.name.replace(".dtseries.nii", "_regr.dtseries.nii")
    )
    if bold_regr_file.exists():
        bold_regr_img = nib.cifti2.Cifti2Image.from_filename(bold_regr_file)
        print(f"  loaded post-regression BOLD from '{bold_regr_file.name}'. "
                  f"({bold_regr_img.shape[0] * bold_regr_img.shape[0] / 1000000:0,.0f}M values)")
    else:
        bold_regr_img = remove_cortical_influence_from_subcortex(
            bold_img, distance_img, args.distance_to_cortex_threshold,
            verbose=args.verbose,
        )
        print(f"  regressed cortical from subcortical BOLD.")
        bold_regr_img.to_filename(bold_regr_file)

    # Report on whether regression changed anything
    if args.verbose:
        # We expect some regions to change, but most to remain identical
        mt.compare_mats(bold_img.get_fdata(), bold_regr_img.get_fdata(),
                        "Original BOLD", "Regressed BOLD",
                        verbose=args.verbose)

    # We are done with the original bold_img, using bold_regr_img from here
    bold_img.uncache()
    del bold_img

    ## Step 4. Smoothing
    if args.verbose:
        print("=" * 80)
        print(f"| Step 4. Smooth BOLD residuals at {dt_start}")
        print("=" * 80)
    smoothed_images = dict()
    for k in args.smoothing_kernel_sizes:
        output_file = str(bold_regr_file).replace(
            ".dtseries.nii", f"_sm-{k:0.2f}.dtseries.nii"
        )
        if not Path(output_file).exists():
            command_list = [
                str(args.wb_command_path), '-cifti-smoothing',
                str(bold_regr_file), f"{k:0.2f}", f"{k:0.2f}", "COLUMN",
                output_file,
                "-left-surface", str(surface_files['lh']),
                "-right-surface", str(surface_files['rh']),
                "-merged-volume"
            ]
            print("  running command: '" + " ".join(command_list) + "'")
            _p = subprocess.run(command_list)
            if _p.returncode != 0:
                print("    failed")
            else:
                print("    success")

        smoothed_images[k] = {
            'img': nib.Cifti2Image.from_filename(output_file),
            'path': Path(output_file),
            'k': k,
        }

    # Step 5. Infomap (and connectivity)
    if args.verbose:
        print("=" * 80)
        print(f"| Step 5. Build InfoMap Networks at {dt_start}")
        print("=" * 80)
    # get a list of loci indices to filter connectivity and infomap anatomy
    idx_good_loci = mt.get_good_loci_indices(
        bold_regr_img, args.exclude_structures, verbose=args.verbose
    )

    # We're done with bold_regr_img, using smoothed versions from here
    bold_regr_img.uncache()
    del bold_regr_img

    for k, img_dict in smoothed_images.items():
        print(f"Starting smoothing kernel {k:0.2f}...")
        # First, build a dense connectivity matrix.
        # This requires AT LEAST 33GB of RAM.
        dconn_file = (
                args.output_path /
                str(img_dict['path'].name).replace(".dtseries.nii", ".dconn.nii")
        )
        if dconn_file.exists():
            dconn_img = nib.Cifti2Image.from_filename(dconn_file)
            print(f"  loaded connectivity from {str(dconn_file)}")
        else:
            dconn_img = build_connectivity_matrix(
                img_dict['img'], distance_img, idx_good_loci,
                args.distance_threshold_exclusion, verbose=args.verbose
            )
            dconn_img.to_filename(dconn_file)
            print(f"  built {dconn_img.shape} connectivity")
            # Save to disk, then clear memory.
            # We need it later, but can't hold multiple dconns at once
            dconn_img.uncache()

        # Once we have connectivity, we no longer need BOLD
        img_dict['img'].uncache()

        # Then, build networks from the connectivity
        for graph_density in args.graph_densities:
            print(f"  Starting graph density {graph_density:0.4f}...")
            network_file = (
                    args.output_path /
                    f"hi_conn_sm-{k:0.2f}_dens-{graph_density:0.4f}_pajek.net"
            )
            log_file = (
                args.output_path / network_file.name.replace(".net", ".log")
            )

            if not network_file.exists():
                hi_conn_idx, log_notes = build_pajek_networks(
                    img_dict['img'], distance_img, dconn_img, idx_good_loci,
                    args.distance_threshold_exclusion, graph_density,
                    verbose=args.verbose
                )
                mt.write_pajek_file(
                    hi_conn_idx, dconn_img.get_fdata(dtype=np.float32),
                    network_file,
                    verbose=True
                )
                dconn_img.uncache()
                with open(log_file, 'w') as f:
                    f.write("\n".join(log_notes))

    dt_end = datetime.now()
    if args.verbose:
        print(f"Finished mfs_make_infomap_networks at {dt_end} "
              f"(elapsed {dt_end - dt_start})")

    return 0


if __name__ == "__main__":
    main()
