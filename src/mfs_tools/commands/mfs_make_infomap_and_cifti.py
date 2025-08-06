#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from datetime import datetime
import re

from debugpy.adapter.servers import dont_wait_for_first_connection
from templateflow import api as tflow
from scipy import stats
from scipy.spatial.distance import cdist
from nibabel.affines import apply_affine
import subprocess

import mfs_tools as mt
from mfs_tools.library import yellow_on, red_on, green_on, color_off
from mfs_tools.library.bids_stuff import get_bids_key_pairs


""" This script reads in a Pajek-style network file,
    runs infomap on it, and saves out the infomap network.
    It then converts that network into a cifti file
    to view the networks on the cortical surface.
"""


def get_arguments():
    """ Parse command line arguments
    """

    parser = argparse.ArgumentParser(
        description="From a Pajek network definition, "
                    "create an infomap network with a cifti2 representation."
    )
    parser.add_argument(
        "input_file",
        help="Pajek-formatted network definition file",
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
        "--source-bold-file",
        default=None,
        help="Without this argument, this script attempts to infer which "
             "BOLD file the network was built from. If that doesn't work, "
             "or if you just want to self-document your process, you can "
             "specify the dtseries.nii file here.",
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
        "--force",
        action="store_true",
        help="Use --force to build new network and cifti files, "
             "even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use --dry-run to go through input files, "
             "reporting what it finds, but avoid running anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use --verbose for more verbose output. This is terribly "
             "useful to review included files, dropped frames, etc.",
    )

    args = parser.parse_args()

    setattr(args, "input_file", Path(args.input_file))
    setattr(args, "output_path", args.input_file.parent)

    setattr(args, "wb_command_path",
            mt.find_wb_command_path(args.wb_command_path))
    setattr(args, "infomap_path",
            mt.find_infomap_path(args.infomap_path))
    # if args.smoothing_kernel_sizes is None:
    #     setattr(args, "smoothing_kernel_sizes", [0.85, 1.70, 2.55, ])
    # if args.graph_densities is None:
    #     setattr(args, "graph_densities",
    #             [0.0001, 0.0002, 0.0005, 0.001,
    #              0.002, 0.005, 0.01, 0.02, 0.05, ],
    #     )
    # if args.exclude_structures is None:
    #     setattr(args, "exclude_structures", list())

    if args.source_bold_file is not None:
        setattr(args, "source_bold_file", Path(args.source_bold_file))

    return args


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
        print(f"Starting mfs_make_infomap_and_cifti.py at {dt_start}")

    if not args.input_file.exists():
        errors.append(f"File '{str(args.input_file)}' does not exist.")

    if args.verbose or args.dry_run:
        print(f"Running mfs_make_infomap_and_cifti.py on '{str(args.input_file)}', "
              "with verbose output")
    else:
        print(f"Saving matrices and networks to '{str(args.output_path)}'")
        if args.output_path.is_file():
            errors.append(f"The output path, '{str(args.output_path)}'"
                          f" is a file. It should be a directory.")

    input_file_pattern = re.compile(
        r"hi_conn_sm-([0-9.]+)_dens-([0-9.]+)_pajek.net"
    )
    match = input_file_pattern.match(args.input_file.name)
    smoothing_fwhm = 0.0
    graph_density = 0.0
    if match:
        smoothing_fwhm = float(match.group(1))
        graph_density = float(match.group(2))
    else:
        errors.append(f"The Pajek network file, '{str(args.input_file)}' "
                      f"does not match the expected naming convention, "
                      f"'hi_conn_sm-(smoothing)_dens-(density_threshold)_pajek.net'.")

    if args.source_bold_file is None:
        if args.verbose:
            print(f"The source BOLD file was not specified, looking for options...")
        likely_bold_path = args.input_file.parent / "_".join([
            args.input_file.parent.parent.parent.name,
            args.input_file.parent.parent.name,
            "task-rest",
            args.input_file.parent.name,
            "space-fsLR_den-91k_desc-denoised_bold_regr",
            f"sm-{match.group(1) if match else 'doesnt_matter'}.dtseries.nii"
        ])
        if likely_bold_path.exists():
            if args.verbose:
                print(f"...found it at '{str(likely_bold_path)}'")
            setattr(args, "source_bold_file", likely_bold_path)
        else:
            errors.append(f"No BOLD file was specified, and it could not be "
                          f"found at '{str(likely_bold_path)}'.")

    for warning in warnings:
        print(f"{yellow_on}Warning: {warning}{color_off}")
    for error in errors:
        print(f"{red_on}Error: {error}{color_off}")

    # Take any opportunity to exit before doing any real work.
    if len(errors) > 0:
        return 1
    if args.dry_run:
        return 0

    """
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
    idx_good_loci = get_good_loci_indices(
        bold_regr_img, args.exclude_structures, verbose=args.verbose
    )

    # We're done with bold_regr_img, using smoothed versions from here
    bold_regr_img.uncache()
    del bold_regr_img
    """




    log_file = (
        args.output_path / args.input_file.name.replace(".net", ".log")
    )
    clu_file = (
        args.output_path / args.input_file.name.replace(".net", ".clu")
    )
    cifti_label_file = (
        args.output_path / args.input_file.name.replace(".net", ".dlabel.nii")
    )

    """
    if not args.input_file.exists():
        hi_conn_idx, log_notes = build_pajek_networks(
            img_dict['img'], distance_img, dconn_img, idx_good_loci,
            args.distance_threshold_exclusion, graph_density,
            verbose=args.verbose
        )
        mt.write_pajek_file(
            hi_conn_idx, dconn_img.get_fdata(dtype=np.float32),
            args.input_file,
            verbose=True
        )
        dconn_img.uncache()
        with open(log_file, 'w') as f:
            f.write("\n".join(log_notes))
    """
    if args.force or not clu_file.exists():
        if args.verbose:
            print(f"Starting infomap ({smoothing_fwhm}, {graph_density}) "
                  f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ...")
        infomap_proc = subprocess.run(
            [
                mt.find_infomap_path(args.infomap_path),
                str(args.input_file),
                str(args.output_path),
                "--clu", "-2", "-s", "42", "-N", str(args.infomap_reps),
                "--no-self-links",
            ],
            capture_output=True,
        )
        if args.verbose:
            print(f"...finished infomap ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ...")
            print(f"...logging to ({log_file}).")
        with open(log_file, 'a') as f:
            f.write(infomap_proc.stdout.decode("utf-8"))
            f.write(infomap_proc.stderr.decode("utf-8"))

    # Load infomap's parcellation and convert it to Cifti2
    infomap_output = mt.load_infomap_clu_file(clu_file, verbose=True)

    bold_img = nib.cifti2.Cifti2Image.from_filename(args.source_bold_file)
    bold_brain_axis = mt.get_brain_model_axes(bold_img)
    # idx_good_loci = mt.get_good_loci_indices(
    #     bold_img, args.exclude_structures, verbose=args.verbose
    # )
    # I never filtered connectivity or distance matrices by --exclude-structures,
    # so everything is full 91282, for better or worse.
    # If I implement --exclude-structures properly, I'll need to coordinate
    # it between mfs_make_fconn_networks and mfs_make_infomap_and_cifti.
    idx_good_loci = [i for i in range(bold_brain_axis.size)]
    infomap_data = np.zeros(
        (len(bold_brain_axis), 1), dtype=np.uint32
    )
    infomap_data[idx_good_loci, 0] = infomap_output.sort_values(
        ['node_id', ]
    ).reset_index(drop=True)['module'][idx_good_loci]

    unique_communities = np.unique(infomap_data[:, 0])
    print(graph_density, unique_communities)
    for comm_idx, community_id in enumerate(unique_communities):
        if community_id != 0:
            community_idx = np.where(
                infomap_data[:, 0] == community_id
            )[0]
            if len(community_idx) < 10:
                # if args.verbose:
                #     print(f"    removing density {graph_density:0.4f}'s "
                #           f"community {community_id} with only "
                #           f"{len(community_idx)} members.")
                infomap_data[community_idx, 0] = 0

    # Package labels into Cifti2
    my_cm = mt.generate_colormap(64)
    first_label = nib.cifti2.Cifti2Label(0, "0", 0.0, 0.0, 0.0, 1.0)
    rest_of_labels = [
        nib.cifti2.Cifti2Label(n + 1, f"{n + 1}", *my_cm.colors[min(63, n)])
        for n in range(len(np.unique(infomap_data)))
    ]
    all_labels = [first_label, ] + rest_of_labels
    packageable_labels = dict([
        (lbl.key, (lbl.label, (lbl.red, lbl.green, lbl.blue, lbl.alpha,)))
        for lbl in all_labels
    ])
    label_axis = nib.cifti2.LabelAxis(
        [f"density {graph_density:0.04f}", ],
        packageable_labels,
    )
    # wb_view wants the brain_models along the columns, and the
    # labels along the rows. So here we transpose our data and
    # create axes to match.
    if args.verbose:
        print(f"    transposing infomap_data from {infomap_data.shape} "
              f"to {infomap_data.T.shape}")
    network_label_img = nib.cifti2.Cifti2Image(
        infomap_data.T, (label_axis, bold_brain_axis)
    )
    network_label_img.update_headers()
    network_label_img.to_filename(cifti_label_file)
    if args.verbose:
        print(f"    saved {str(cifti_label_file)}; "
              f"done with d={graph_density:0.4f}")

    dt_end = datetime.now()
    if args.verbose:
        print(f"Finished mfs_make_infomap_networks at {dt_end} "
              f"(elapsed {dt_end - dt_start})")

    return 0


if __name__ == "__main__":
    main()
