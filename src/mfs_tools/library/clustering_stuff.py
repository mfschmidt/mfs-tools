from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import nibabel as nib
import subprocess

from mfs_tools.library.file_stuff import (
    find_infomap_path, find_wb_command_path,
    write_pajek_file, load_infomap_clu_file
)
from mfs_tools.library.cifti_stuff import (
    get_brain_model_axes, get_label_axes,
    get_subcortical_indices, get_cortical_indices,
    get_cortical_data
)
from mfs_tools.library.utility_stuff import (
    correlate_bold, generate_colormap
)
from mfs_tools.library.similarity_scores import SimilarityScores


def find_good_loci(bold_img, bad_vertices=None, verbose=False):
    """ Figure out which loci will be used for connectivity.
    """

    # Ensure we have more than zero valid structures to work with
    # I don't know why, but Lynch's code only includes the 10 regions
    # with distinct LEFT or RIGHT, but this excludes BRAIN_STEM.
    # I'll exclude it here, too, so code matches, but maybe we can
    # revisit if we have brainstem-specific hypotheses.
    # Lynch also put ACCUMBENS in his code twice, one of them
    # probably replacing 'DIENCEPHALON_VENTRAL', so only
    # 18 regions remain out of the original 21. For debugging, we'll try to
    # match exactly, but in practice, using all regions seems better.
    anat_axis = get_brain_model_axes(bold_img)

    # The 'anat_axis' already has 85,059 loci,
    # just like matlab's BrainStructure and BrainStructureLabel,
    # so we don't need to do anything else with it here.
    structures = np.unique([
        str(name) for name in anat_axis.name
        if ("BRAIN_STEM" not in name) & ('DIENCEPHALON' not in name)
    ])
    # There are 20 structures in 'structures'

    # We need to get indices into each region, so gather all potentials
    # and cross reference with what we have in our data
    potential_structures = np.unique([str(name) for name in anat_axis.name])
    print(f"Found {len(potential_structures)} potential structures, "
          f"But we only want {len(structures)}.")
    structure_mask = [s in structures for s in potential_structures]
    structure_indices = np.where(structure_mask)[0]
    good_structures = potential_structures[structure_indices]
    print(f"Indexing leaves us with {len(structure_indices)} structures")

    # Now go through each locus to determine if it's in our list of structures.
    locus_mask = [str(s) in good_structures for s in anat_axis.name]
    locus_indices = np.where(locus_mask)[0]
    print(f"Filtering structures leaves {len(locus_indices)} loci")

    # Further, remove any pre-designated bad vertices
    if bad_vertices is not None:
        locus_indices = [li for li in locus_indices if li not in bad_vertices]
        print(f"Removing bad verts leaves {len(locus_indices)} loci")

    if verbose:
        print(f"Using {len(good_structures)} structures: ")
        for i, s in enumerate(good_structures):
            print(f"{i + 1:>2}. {str(s)}")

    return locus_indices


def infomap(bold_img, distance_matrix, distance_threshold=10,
            graph_densities=None, num_reps=50, bad_vertices=None,
            structures=None, num_cores=1, working_path="/tmp", verbose=False):
    """

    :param bold_img: A cifti2 image (or a path to one on disk) containing BOLD data.
    :param distance_matrix:
    :param distance_threshold: How far away (in mm) two nodes must be to use their edge.
    :param graph_densities: An iterable of densities for focusing connectivity
    :param num_reps: How many times to repeat infomap at each density, default 50.
    :param bad_vertices:
    :param structures:
    :param num_cores:
    :param working_path: A location on disk for saving intermediate files.
    :param verbose: Set to True for more verbose output.
    :return:
    """

    # Handle defaults, setup, etc, before actually starting.
    np.random.seed(44)
    working_path = Path(working_path)

    infomap_binary = find_infomap_path()
    if infomap_binary is None:
        raise FileNotFoundError("Could not find Infomap binary.")

    if graph_densities is None:
        graph_densities = sorted(
            [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, ],
            reverse=True,
        )

    # We only want connectivity from some regions, so we need to
    # create a list of indices of the loci we're interested in.
    # Note that Lynch may have errors in his matlab-based list.
    good_indices = find_good_loci(
        bold_img, bad_vertices=bad_vertices, verbose=verbose
    )

    # Trim the distance matrix.
    # Assuming the distance matrix was already computed, here we
    # modify it several ways.
    # First, avoid subcortical regions. They have strong connectivity
    # that dominates density-thresholded matrices, and we need to see
    # cortical connectivity.
    # Trim the distance matrix to match our filtered vertices.
    # While we're at it, set subcort-subcort distances to zero.
    if verbose:
        print(f"Original distance matrix is shaped {distance_matrix.shape}")
    subcort_indices = get_subcortical_indices(bold_img)
    distance_matrix[np.ix_(subcort_indices, subcort_indices)] = 0.0
    if verbose:
        print(f"D1 is shaped {distance_matrix.shape} with ")
        print(f"  {np.sum(distance_matrix > 0.0):,} non-zero edges,")
        print(f"  {np.sum(distance_matrix == 0.0):,} zero edges,")
        print(f"  {np.sum(distance_matrix <= distance_threshold):,} local edges.")
    # Next, restrict distances to the same loci of interest as connectivity.
    distance_matrix = distance_matrix[good_indices, :][:, good_indices]
    if verbose:
        print(f"D2 is shaped {distance_matrix.shape} with ")
        print(f"  {np.sum(distance_matrix > 0.0):,} non-zero edges,")
        print(f"  {np.sum(distance_matrix == 0.0):,} zero edges,")
        print(f"  {np.sum(distance_matrix <= distance_threshold):,} local edges.")

    # If connectivity data exists, use it. Otherwise, compute it.
    # Trim the BOLD data (loaded from matlab's directory earlier) to match
    # locations in distance matrix.
    full_conn_path = working_path / "full_connectivity.npy"
    final_conn_path = working_path / "final_connectivity.npy"
    bold_data = bold_img.get_fdata()[:, good_indices]
    if full_conn_path.exists():
        print(f"Loading full connectivity from {str(full_conn_path)}")
        full_connectivity = np.load(full_conn_path)
    else:
        full_connectivity = correlate_bold(bold_data, strip_size=4096, verbose=True)
        np.save(full_conn_path, full_connectivity)

    # Prepare the connectivity for infomap
    if final_conn_path.exists():
        print(f"Loading final connectivity from {str(final_conn_path)}")
        final_connectivity = np.load(final_conn_path)
        print(f"{np.sum(final_connectivity > 0.0):,} usable edges in the loaded final connectivity.")
    else:
        final_connectivity = full_connectivity.copy()
        print(f"Starting with {np.sum(final_connectivity > 0.0):,} edges.")

        # Remove the diagonal (set to zero)
        final_connectivity[np.diag_indices_from(final_connectivity)] = 0.0
        print(f"{np.sum(final_connectivity > 0.0):,} edges remain after removing diagonal")

        # Remove local edges
        # (because we set all subcortical-subcortical edges to zero,
        #  this also removes them all.)
        final_connectivity[distance_matrix <= distance_threshold] = 0.0
        print(f"{np.sum(final_connectivity > 0.0):,} edges remain after removing local edges")

        # Remove any NaN values
        final_connectivity[np.isnan(final_connectivity)] = 0.0
        print(f"{np.sum(final_connectivity > 0.0):,} edges remain after removing NaNs")

        # Save connectivity for comparison with matlab's filtered connectivity.
        np.save(final_conn_path, final_connectivity)

    # Next, go through the connectivity matrix, with local edges, diagonals,
    # and NaNs removed, and sort each column to find
    # the highest connectivity edges for each locus.
    # For each column, set the top edges (thresholded by graph density) to True,
    # leaving everything else False.
    # This ensures that even weakly connected loci maintain some
    # connectivity to their hub-like partners.

    total_edges_kept = 0  # not accurate, some overlap, but useful for comparisons

    for i_d, d in enumerate(graph_densities):
        print(f"Starting density {d} at {datetime.now()}...")
        log_notes = list()

        # Create a mask with all False until we decide what to keep.
        hi_conn_mask = np.zeros(final_connectivity.shape, dtype=bool)
        # for each column in the connectivity matrix, find the highest
        # correlations and add those edges to the mask for that location's
        # column AND row.
        for i_n in range(final_connectivity.shape[1]):
            if np.any(final_connectivity[:, i_n]):
                ordered_connectivity = np.flip(np.argsort(final_connectivity[:, i_n]))
                num_to_keep = int(np.ceil(d * len(ordered_connectivity)))
                total_edges_kept += num_to_keep
                log_notes.append(
                    f"Keeping {num_to_keep:,} edges for density {d}, col {i_n}"
                )
                # for v in ordered_connectivity[:num_to_keep]:
                #     print(f"Index {v} == {m[v, i_n]:0.3f}")
                hi_conn_mask[ordered_connectivity[:num_to_keep], i_n] = True
                hi_conn_mask[i_n, ordered_connectivity[:num_to_keep]] = True

        # We built the matrix up symmetrically, so now that it's complete,
        # we only need half of it. Delete the lower triangle, then
        # find the indices of the masked edges.
        hi_conn_mask[np.tril_indices_from(hi_conn_mask)] = False
        hi_conn_idx = np.argwhere(hi_conn_mask)
        write_pajek_file(
            hi_conn_idx, final_connectivity, working_path / f"hi_conn_d-{d:0.4f}.net",
            verbose=verbose
        )
        with open(working_path / f"hi_conn_d-{d:0.4f}.log", 'w') as f:
            f.write("\n".join(log_notes))

        print(f"Created network for d={d} at {datetime.now()}...")

        # These networks are independent of each other and could run
        # in parallel, theoretically, but the larger d=0.05 network can
        # consume enough memory to crash a large system, so I'm just
        # running in series for now. Perhaps an upgrade would be to
        # get smart enough to run smaller networks in parallel.
        infomap_start_time = datetime.now()
        proc = subprocess.run(
            [
                str(infomap_binary),
                str(working_path / f"hi_conn_d-{d:0.4f}.net"),
                str(working_path),
                "--clu", "-2",
                "-s", "42",
                "-N", str(num_reps),
                "--no-self-links"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_path,
            env={
                "PATH": infomap_binary.parent.as_posix(),
                "LD_LIBRARY_PATH": infomap_binary.parent.as_posix(),
            },
            check=True,
            shell=True,
        )
        with open(working_path / f"hi_conn_d-{d:0.4f}.log", 'w') as f:
            f.write(f"Began at {infomap_start_time}")
            f.write(proc.stdout.decode())
            f.write(proc.stderr.decode())
            f.write(f"Complete at {datetime.now()}")
        if proc.returncode != 0:
            print(f"ERROR: Infomap did not complete for d={d} at {datetime.now()}...")
        print(f"Finished infomap for d={d} at {datetime.now()}...")

    # Pre-allocate data for a new Cifti file of Infomap data
    infomap_data = np.zeros(
        (len(bold_img.shape[1]), len(graph_densities)),
        dtype=np.int8
    )
    total_edges_removed = 0
    most_communities = 0
    density_with_most_communities = 0.0
    # Extract community membership, matched by node_id, and
    # save it into a single matrix, created just above.
    for i_d, d in enumerate(graph_densities):
        # For each density, extract the community labels for our atlas.
        infomap_output = load_infomap_clu_file(
            working_path / f"hi_conn_d-{d:0.4f}.clu", verbose=True
        )
        infomap_data[good_indices, i_d] = infomap_output.sort_values(
            ['node_id', ]
        )['module'].values

        unique_communities = np.unique(infomap_data[:, i_d])
        if verbose:
            print(d, unique_communities)
        if len(unique_communities) > most_communities:
            most_communities = len(unique_communities)
            density_with_most_communities = d
        for comm_idx, community_id in enumerate(unique_communities):
            if community_id != 0:
                community_idx = np.where(
                    infomap_data[:, i_d] == community_id
                )[0]
                if len(community_idx) < 10:
                    if verbose:
                        print(f"  Removing density {d}'s community "
                              f"{community_id} with only {len(community_idx)} "
                              "members.")
                    infomap_data[community_idx, i_d] = 0
                    total_edges_removed += len(community_idx)

    if verbose:
        print(f"Removed {total_edges_removed:,} total edges due to "
              "communities having fewer than 10 members.")
        print(f"The largest set of communities was {most_communities}, "
              f"from d={density_with_most_communities:0.4f}.")

    # Package the community data as a Cifti2 Label file with random colors
    my_cm = generate_colormap(24)
    first_label = nib.cifti2.Cifti2Label(0, "0", 0.0, 0.0, 0.0, 1.0)
    rest_of_labels = [
        nib.cifti2.Cifti2Label(n + 1, f"{n + 1}", *my_cm.colors[n])
        for n in range(most_communities)
    ]
    all_labels = [first_label, ] + rest_of_labels
    packageable_labels = dict([
        (lbl.key, (lbl.label, (lbl.red, lbl.green, lbl.blue, lbl.alpha,)))
        for lbl in all_labels
    ])
    label_axis = nib.cifti2.LabelAxis(
        [f"density {d:0.04f}" for d in graph_densities],
        packageable_labels,
    )
    # wb_view wants the brain_models along the columns, and the
    # labels along the rows. So here we transpose our data and
    # create axes to match.
    network_label_img = nib.cifti2.Cifti2Image(
        infomap_data.T,
        (label_axis, get_brain_model_axes(bold_img))
    )
    # Save these labels as a Cifti file, then return the image to the caller.
    network_label_img.update_headers()
    network_label_img.to_filename(
        working_path /
        f"infomap_calculated_network_atlases_over_9_densities.dlabel.nii"
    )

    return network_label_img


def spatial_filter(atlas_img, lh_surf_path, rh_surf_path,
                   min_area=50, work_path="/tmp", verbose=False):
    """ Remove clusters below an areal threshold.
    """

    wb_command = find_wb_command_path()

    label_axis = get_label_axes(atlas_img)
    anat_axis = get_brain_model_axes(atlas_img)
    atlas_data = atlas_img.get_fdata()

    filtered_atlas_data = np.zeros(atlas_data.shape)

    # For each of the graph densities in the clusters,
    for d_i, d in enumerate(label_axis.name):
        if verbose:
            print(f"Removing small islets from '{d}' communities.")
        _labels = atlas_data[d_i, :]
        community_ids, community_sizes = np.unique(_labels, return_counts=True)
        # Do we need to drop NaNs? Matlab does, but it never applies.
        for c_i, c in enumerate([c for c in community_ids if c != 0]):
            c = int(c)
            # if verbose:
            #     print(f"  community {c}:")
            # Generate a binary mask with ones for this community's voxels and vertices.
            _mask_data = np.array(_labels == c).astype(np.uint8).reshape(1, -1)
            # Save this community-specific mask to disk.
            filename = f"tmp_{d.replace(' ', '_')}_c-{c:02d}_mask.dscalar.nii"
            community_scalar_axis = nib.cifti2.ScalarAxis(
                [f"community {c}", ]
            )
            community_img = nib.cifti2.Cifti2Image(
                _mask_data, (community_scalar_axis, anat_axis)
            )
            community_img.to_filename(Path(work_path) / filename)
            # Ask workbench to label islands within the community,
            # but only if they're larger than our threshold, leaving
            # small islets as zero.
            proc = subprocess.run([
                wb_command,
                "-cifti-find-clusters",
                str(Path(work_path) / filename),
                "0", str(min_area), "0", str(min_area), "COLUMN",
                str(Path(work_path) / filename),
                "-left-surface", lh_surf_path,
                "-right-surface", rh_surf_path,
                "-merged-volume"
            ])
            if proc.returncode == 0:
                # Use the wb_command-generated mask to eliminate small islets
                # from our original infomap atlas.
                clean_mask = nib.cifti2.Cifti2Image.from_filename(
                    Path(work_path) / filename
                ).get_fdata().astype(np.bool).ravel()
                if verbose:
                    print(f"    saving {np.sum(clean_mask):,} of "
                          f"{np.sum(_mask_data):,} members of '{c}'")
                filtered_atlas_data[d_i, clean_mask] = c
            else:
                print(f"    FAILED to detect small islets in community '{c}'")
        if verbose:
            print("Before:")
            print(np.unique(_labels, return_counts=True))
            print("After:")
            print(np.unique(filtered_atlas_data[d_i, :], return_counts=True))

    density_scalar_axis = nib.cifti2.ScalarAxis(
        [f"community {d}" for d in label_axis.name ]
    )
    filtered_img = nib.cifti2.Cifti2Image(
        filtered_atlas_data, (density_scalar_axis, anat_axis)
    )
    filtered_img.update_headers()
    filtered_img.to_filename(
        Path(work_path) / f"infomap_atlas_filtered_unfilled.dscalar.nii"
    )

    proc = subprocess.run([
        wb_command,
        "-cifti-dilate",
        str(Path(work_path) / f"infomap_atlas_filtered_unfilled.dscalar.nii"),
        "COLUMN", "50", "50",
        "-left-surface", lh_surf_path,
        "-right-surface", rh_surf_path,
        str(Path(work_path) / f"infomap_atlas_filtered_and_filled.dscalar.nii"),
        "-nearest"
    ])
    if proc.returncode != 0:
        print(f"ERROR in filling deleted small islets with nearest labels.")

    return nib.cifti2.Cifti2Image.from_filename(
        Path(work_path) / f"infomap_atlas_filtered_and_filled.dscalar.nii"
    )


def identify_networks(
        bold_img,
        community_labels,
        surface_files,
        network_priors,
        output_path,
        pfm_dir,
        workbench_binary=None,
        verbose=False
):
    """
    Identifies and associates communities in a brain network with predefined
    network priors, generating similarity scores and network assignment based
    on functional and spatial connectivity patterns derived from brain activity.

    :param bold_img: 4D array representing the BOLD functional time-series data.
    :type bold_img: np.ndarray
    :param community_labels: 1D or 2D array mapping vertex indices to their respective
        initial community label identifiers.
    :type community_labels: np.ndarray
    :param surface_files: File path(s) to brain surface data, utilized in connectivity
        calculations.
    :type surface_files: str or list[str]
    :param network_priors: Network priors object encapsulating functional
        connectivity (fc), spatial probability maps, and network metadata.
    :type network_priors: NetworkPriors
    :param output_path: Directory path where computed results and intermediate
        outputs will be saved.
    :type output_path: str
    :param pfm_dir: File path to the directory containing probabilistic fine
        mapping resources.
    :type pfm_dir: str
    :param workbench_binary: Optional path to the Connectome Workbench executable binary.
        If not provided, an available binary will be searched for automatically.
    :type workbench_binary: str or None
    :param verbose: Boolean flag indicating whether to output detailed logs
        during execution.
    :type verbose: bool
    :return: The function does not explicitly return a value. Instead, it processes
        data to generate network associations and saves results to the specified
        output path.
    :rtype: None
    """

    # Use provided binary, if provided and valid, otherwise find one
    wb_command = find_wb_command_path(workbench_binary)

    ctx_idx = get_cortical_indices(bold_img)

    # Generate cortical connectivity matrix from BOLD.
    # This will include all edges, not just those above a threshold.
    bold_conn = correlate_bold(
        get_cortical_data(bold_img),
        force_diagonals_to=0.0,
        zero_nans=True
    )

    # Create a place for new community connectivity
    # We need all cortical vertices for each label in the community map
    unique_community_labels = [
        int(lbl) for lbl in np.unique(community_labels) if int(lbl) != 0
    ]
    num_communities = len(unique_community_labels)
    new_conn = np.zeros((len(ctx_idx), len(unique_community_labels)),
                        dtype=np.float32)

    for i, lbl in enumerate(unique_community_labels):
        # Average connectivity, but only in cortical vertices (avoid subcortex)
        new_conn[:, i] = np.mean(
            bold_conn[:, community_labels[ctx_idx] == lbl],
            axis=1
        )

    # From a [vertices x labels] new_conn and a [vertices x prior_labels] prior_fc,
    # We'd like [labels x prior_labels] connectivity matrices
    # numpy.corrcoef is row-wise, so we transpose the matrices.
    real_v_atlas_functional = np.corrcoef(
        new_conn.T, network_priors.fc.T
    )[:new_conn.shape[1], new_conn.shape[1]:]
    real_v_atlas_functional[np.isnan(real_v_atlas_functional)] = 0.0

    # prior_spatial is a 59412 x 20 probability map of how likely
    # each cortical vertex is to be part of that label.
    real_v_atlas_spatial = np.zeros(real_v_atlas_functional.shape)
    for i, lbl in enumerate(unique_community_labels):
        for j in range(network_priors.num_networks):
            real_v_atlas_spatial[i, j] = np.mean(network_priors.spatial[community_labels[ctx_idx] == lbl, j])
    real_v_atlas_spatial[np.isnan(real_v_atlas_spatial)] = 0.0

    # Create the structure to hold all the probabilities
    s = SimilarityScores(num_communities)

    for com_idx in range(num_communities):
        prob_combo = real_v_atlas_functional[com_idx, :] * real_v_atlas_spatial[com_idx, :]
        sorted_indices = np.argsort(prob_combo)[::-1]

        s.community[com_idx] = unique_community_labels[com_idx]
        s.r[com_idx] = network_priors.labels.loc[sorted_indices[0], 'r']
        s.g[com_idx] = network_priors.labels.loc[sorted_indices[0], 'g']
        s.b[com_idx] = network_priors.labels.loc[sorted_indices[0], 'b']
        s.network[com_idx] = network_priors.labels.loc[sorted_indices[0], 'id']
        s.func_conn[com_idx] = real_v_atlas_functional[com_idx, sorted_indices[0]]
        s.spatial_score[com_idx] = real_v_atlas_spatial[com_idx, sorted_indices[0]]
        delta_first_second = prob_combo[sorted_indices[0]] - prob_combo[sorted_indices[1]]
        s.confidence[com_idx] = delta_first_second / prob_combo[sorted_indices[1]]

        # These are offset to store the next-best choices, sequentially
        # My #1 matches matlab #1, and my #19 matches matlab #19. but my #0 is unnecessary?
        for net_idx in range(network_priors.num_networks - 1):
            s.alt_networks[net_idx][(com_idx,0)] = network_priors.labels.loc[
                sorted_indices[net_idx + 1], 'label'
            ]
            s.alt_func_sims[net_idx][com_idx] = real_v_atlas_functional[
                com_idx, sorted_indices[net_idx + 1]
            ]
            s.alt_spatial_scores[net_idx][com_idx] = real_v_atlas_spatial[
                com_idx, sorted_indices[net_idx + 1]
            ]
    return None


def map_community_labels(
        community_labels, network_map, label_df,
        collapse=False, verbose=False
):
    """ Map arbitrary integer labels from community_label_array
        into meaningful label_df ids through network_dict. """

    # Extract an ordered list of non-zero community labels (arbitrary)
    unique_community_labels = sorted([
        int(lbl) for lbl in np.unique(community_labels)
        if int(lbl) != 0
    ])

    # Create an empty array to hold translated ids
    _data = np.zeros(
        (len(unique_community_labels), len(community_labels)),
        dtype=np.uint8
    )
    _header_names = list()
    for com_idx, com_lbl in enumerate(unique_community_labels):  # 6, not 9
        # The mapping from arbitrary to actual is via network_dict.
        network_name = network_map[com_idx]
        _header_names.append(network_name)
        # With a network_name, we can look up the id from network priors.
        label_index = pd.Index(label_df.label).get_loc(network_name)
        label_id = label_df.loc[label_index, 'id']
        # Wherever the old label was in community_label_array,
        # write the new meaningful label into _data
        loci_mask = community_labels == com_lbl
        _data[com_idx, loci_mask] = label_id
        if verbose:
            print(f"Community #{com_idx}: {np.sum(loci_mask):,} loci "
                  f"get id {label_id} ({network_name}).")

    # Each column contains all zeros with one id. We can collapse these
    # to make one atlas rather than 'num_rows' masks.
    if collapse:
        _data = np.sum(_data, axis=0).reshape(1, -1)

    return _data
