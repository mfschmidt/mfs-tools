from pathlib import Path
import re
import json
from datetime import datetime
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io as sio
from collections import namedtuple
from nibabel.filebasedimages import ImageFileError


def get_img_and_desc(file_path, only_dims=False, verbose=False):
    """ Load neuroimaging file and build a descriptive string.

        :param file_path: path to image file
        :param bool only_dims: Set to output only the dimensions
        :param verbose: activate additional output
    """

    desc = "N/A"  # default, should get replaced

    # Let nibabel load and interpret the file
    try:
        img = nib.load(file_path)
    except ImageFileError():
        return None, f"'{str(file_path)}' is not a supported neuroimage"

    tr_str = ""
    if (    isinstance(img, nib.nifti1.Nifti1Image) or
            isinstance(img, nib.nifti2.Nifti2Image)
    ):
        if len(img.shape) > 3:
            dims = img.header.get('pixdim', None)
            if dims is not None and len(dims) > 4:
                tr_str = f", TR={dims[4]:.2f}s"
    if isinstance(img, nib.nifti1.Nifti1Image):
        desc = f"Nifti1 file: {img.shape}{tr_str}"
        if only_dims:
            desc = ",".join([str(dim) for dim in img.shape])
    elif isinstance(img, nib.nifti2.Nifti2Image):
        desc = f"Nifti2 file: {img.shape}{tr_str}"
        if only_dims:
            desc = ",".join([str(dim) for dim in img.shape])
    elif isinstance(img, nib.cifti2.Cifti2Image):
        if str(file_path).endswith(".dtseries.nii"):
            file_type = "dtseries"
        elif str(file_path).endswith("dscalar.nii"):
            file_type = "dscalar"
        elif str(file_path).endswith("dlabel.nii"):
            file_type = "dlabel"
        else:
            file_type = "file"

        axes_strs = list()
        for axis in img.header.mapped_indices:
            ax = img.header.get_axis(axis)
            if verbose:
                print(f"  found cifti2 '{str(type(ax))}' axis")
            if isinstance(ax, nib.cifti2.SeriesAxis):
                tr_str = f" {ax.step:.2f} {getattr(ax, 'unit', '')} TRs"
                if only_dims:
                    axes_strs.append(str(ax.size))
                else:
                    axes_strs.append(f"{ax.size}{tr_str}")
            if isinstance(ax, nib.cifti2.BrainModelAxis):
                if only_dims:
                    axes_strs.append(str(ax.size))
                else:
                    axes_strs.append(f"{ax.size} grayordinates: "
                                     f"{np.sum(ax.volume_mask):,} voxels & "
                                     f"{np.sum(ax.surface_mask):,} vertices")
            if isinstance(ax, nib.cifti2.LabelAxis):
                if only_dims:
                    axes_strs.append(str(len(ax.label[0].keys())))
                else:
                    axes_strs.append(f"{len(ax.label[0].keys())} labels")
            if isinstance(ax, nib.cifti2.ScalarAxis):
                if only_dims:
                    axes_strs.append(str(ax.size))
                else:
                    axes_strs.append(f"{ax.size} scalars")
        if len(axes_strs) > 0:
            if only_dims:
                desc = ",".join(axes_strs)
            else:
                desc = f"Cifti2 {file_type}: ({' * '.join(axes_strs)})"
        else:
            if only_dims:
                desc = ",".join([str(dim) for dim in img.shape])
            else:
                desc = f"Cifti2 {file_type}: {img.shape}"
    elif isinstance(img, nib.gifti.gifti.GiftiImage):
        axes_strs = list()
        for arr in img.darrays:
            if isinstance(arr, nib.gifti.gifti.GiftiDataArray):
                if only_dims:
                    axes_strs.append(str(arr.dims[0]))
                else:
                    axes_strs.append(f"{arr.dims[0]:,} of intent {arr.intent}")
        if len(axes_strs) > 0:
            if only_dims:
                desc = ",".join(axes_strs)
            else:
                desc = f"Gifti file: ({' * '.join(axes_strs)})"
        else:
            if only_dims:
                desc = "0"
            else:
                desc = f"Gifti file: no data"


    return img, desc


def get_cifti_desc(file_path):
    """ Load cifti file and build a descriptive string.
    """

    desc = "Cifti file:"
    img = nib.load(file_path)
    if isinstance(img, nib.cifti2.cifti2.Cifti2Image):
        desc = " ".join([
            desc,
            f"{img.shape}"
        ])
    return img, desc


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


def find_infomap_path(suggested_path=None):
    """ Find infomap in some likely places.
    """

    if (
            (suggested_path is not None) and
            Path(suggested_path).is_file() and
            Path(suggested_path).name == "infomap"
    ):
        return Path(suggested_path)

    path_suggestions = list(Path("/home").glob("*/.virtualenvs/*/bin/infomap"))
    for p in path_suggestions:
        if p.is_file():
            return p

    return None


def write_pajek_file(indices, connectivity_matrix, filename, verbose=False):
    """
    Writes a Pajek format file based on the given connectivity matrix and indices of edges.

    This function generates a Pajek file, which can be used with infomap.
    The Pajek file includes vertices and edges defined by the input connectivity matrix
    and edge indices. If `verbose` is set to True, it also prints a summary of the data written
    to the file.

    :param indices: List of tuples representing index pairs of edges in the connectivity
        matrix.
    :type indices: list[tuple[int, int]] or array-like

    :param connectivity_matrix: A 2D array representing the connectivity
        matrix. The values indicate the weight of the connections between nodes.
    :type connectivity_matrix: numpy.array

    :param filename: The path to the output file where the Pajek network data will be
        written. This can include a full or relative file path.
    :type filename: str or Path

    :param verbose: When set to True, the function will print details about the number
        of vertices and edges written to the specified output file. Defaults to False.
    :type verbose: bool

    :return: True if the file was written successfully, False otherwise.
    """
    with open(filename, "w") as f:
        f.write(f"*Vertices {connectivity_matrix.shape[0]}\n")
        for i in range(connectivity_matrix.shape[0]):
            f.write(f"{i + 1} \"{i + 1}\"\n")
        f.write(f"*Edges {len(indices)}\n")
        for i, idx in enumerate(indices):
            f.write(f"{idx[0] + 1} {idx[1] + 1} "
                    f"{connectivity_matrix[idx[0], idx[1]]:0.6f}\n")
    if verbose:
        print(f"Wrote {connectivity_matrix.shape[0]} vertices, "
              f"and {len(indices)} edges to '{str(filename)}'")

    return True


def read_pajek_file(pajek_file, verbose=False, sort_vertices=False, sort_edges=False):
    """
    Parses a Pajek format file (.net file) to extract graph data, including vertices and edges.
    The file is processed based on specific patterns for vertices and edges, and relevant
    data is extracted into separate lists. An optional verbose mode provides progress output
    and error details. The extracted vertices and edges lists can optionally be sorted.

    :param pajek_file: The path to the Pajek file to be parsed as a string or file-like object.
    :param verbose: A boolean flag. If True, outputs processing details to the console.
    :param sort_vertices: A boolean flag. If True, sorts the vertices list by vertex id.
    :param sort_edges: A boolean flag. If True, sorts the edges list by source and target ids.
    :return: A dictionary with three keys: "vertices", "edges", and "errors". The "vertices" and
        "edges" keys store lists of parsed vertex and edge data respectively, while the
        "errors" key stores a list of encountered errors.
    """

    vertex_mode_pattern = re.compile(r"\*Vertices (\d+)")
    vertex_data_pattern = re.compile(r"(\d+) \"(.*?)\"")
    edge_mode_pattern = re.compile(r"\*Edges (\d+)")
    edge_data_pattern = re.compile(r"(\d+) (\d+) (\d+\.\d+)")
    mode = None
    vertices = dict()
    edges = dict()
    errors = list()
    edges_per_col = dict()
    edges_per_row = dict()
    if verbose:
        print(f"Reading from '{str(pajek_file)}'...")
    with open(pajek_file, "r") as f:
        for line in f:
            if vertex_mode_pattern.match(line):
                mode = "vertices"
            elif edge_mode_pattern.match(line):
                mode = "edges"
            elif mode == "vertices":
                if vertex_data_pattern.match(line):
                    vid, label = vertex_data_pattern.match(line).groups()
                    vertices[int(vid)] = label
                else:
                    errors.append(f"Error parsing vertex data: {line}")
            elif mode == "edges":
                if edge_data_pattern.match(line):
                    col, row, weight = edge_data_pattern.match(line).groups()
                    col, row, weight = int(col), int(row), float(weight)
                    edges[(col, row)] = weight
                    if col in edges_per_col.keys():
                        edges_per_col[col] += 1
                    else:
                        edges_per_col[col] = 1
                    if row in edges_per_row.keys():
                        edges_per_row[row] += 1
                    else:
                        edges_per_row[row] = 1
                else:
                    errors.append(f"Error parsing edge data: {line}")
            else:
                errors.append(f"Error parsing file: {line}")
    if verbose:
        print(f"Read {len(vertices)} vertices and {len(edges)} edges.")
        print(f"Encountered {len(errors)} errors.")

    # if sort_vertices:
    #     vertices = sorted(vertices, key=lambda v: v[0])
    # if sort_edges:
    #     edges = sorted(edges, key=lambda e: (e[0], e[1]))

    return {
        "vertices": vertices,
        "edges": edges,
        "errors": errors,
        "edges_per_col": edges_per_col,
        "edges_per_row": edges_per_row,
    }


def compare_pajek_data(pajek_data_a, pajek_data_b, verbose=False):
    """
    Compares two dictionaries representing Pajek graph data, identifying discrepancies
    in vertices and edges. Outputs information about mismatches and returns detailed
    results for edges unique to each dataset.

    :param pajek_data_a: A dictionary representing Pajek data containing vertices
        and edges. Expected keys are "vertices" and "edges".
    :type pajek_data_a: dict
    :param pajek_data_b: A dictionary representing Pajek data containing vertices
        and edges. Expected keys are "vertices" and "edges".
    :type pajek_data_b: dict
    :param verbose: A boolean flag. If True, outputs processing details to the console.
    :return: A dictionary with two keys, "a_only_edges" and "b_only_edges", each
        containing lists of edges unique to `pajek_data_a` and `pajek_data_b`,
        respectively.
    :rtype: dict
    """

    if len(pajek_data_a["vertices"]) != len(pajek_data_b["vertices"]):
        print(f"Number of vertices differ: {len(pajek_data_a['vertices'])} != "
              f"{len(pajek_data_b['vertices'])}")
    if len(pajek_data_a["edges"]) != len(pajek_data_b["edges"]):
        print(f"Number of edges differ: {len(pajek_data_a['edges'])} != "
              f"{len(pajek_data_b['edges'])}")

    # Traverse edges to quantify differences
    if verbose:
        print(f"Traversing edges... ({datetime.now()})")
    a_only_edges = dict()
    b_only_edges = dict()
    diffs = list()
    matches = dict()
    for a_key, a_val in pajek_data_a["edges"].items():
        if a_key not in pajek_data_b["edges"].keys():
            a_only_edges[a_key] = a_val
        else:
            if a_val != pajek_data_b["edges"][a_key]:
                diffs.append((a_key, a_val, pajek_data_b["edges"][a_key]))
            else:
                matches[a_key] = a_val
    for b_key, b_val in pajek_data_b["edges"].items():
        if b_key not in pajek_data_a["edges"].keys():
            b_only_edges[b_key] = b_val
        else:
            if b_val != pajek_data_a["edges"][b_key]:
                diffs.append((b_key, b_val, pajek_data_a["edges"][b_key]))
            else:
                matches[b_key] = b_val
    if len(a_only_edges) > 0:
        print(f"Edges in A but not B: {len(a_only_edges):,}")
    if len(b_only_edges) > 0:
        print(f"Edges in B but not A: {len(b_only_edges):,}")

    # Count edges per row and column
    if verbose:
        print(f"Counting edges per column and row... ({datetime.now()})")
    col_mismatches = list()
    for col in pajek_data_a["edges_per_col"].keys():
        if col not in pajek_data_b["edges_per_col"].keys():
            print(f"Column {col} only in A")
        elif pajek_data_a["edges_per_col"][col] != pajek_data_b["edges_per_col"][col]:
            col_mismatches.append((
                col,
                pajek_data_a["edges_per_col"][col],
                pajek_data_b["edges_per_col"][col]
            ))
    for col in pajek_data_b["edges_per_col"].keys():
        if col not in pajek_data_a["edges_per_col"].keys():
            print(f"Column {col} only in B")

    return {
        "a_only_edges": a_only_edges,
        "b_only_edges": b_only_edges,
        "diffs": diffs,
        "matches": matches,
        "col_mismatches": col_mismatches,
    }


def load_infomap_clu_file(file_path, verbose=False):
    """ Read a .clu file as a pandas dataframe.

        :param file_path: the file location
        :param verbose: optionally, set to True for additional output.

        The file probably has 10 header rows to start, followed by a
        long 3-column matrix of (node_id, module, flow) values. To be safe,
        this function counts the header rows before skipping them, then
        lets pandas read_csv handle the data import, then manually
        formats the dataframe column names before returning it.
    """

    lines_to_skip = 0
    col_names = ("node_id", "module", "flow")

    col_name_pattern = re.compile(r"^#\s+(node_id)\s+([a-z_]+)\s+([a-z_]+)")
    header_pattern = re.compile(r"^#.*")
    if verbose:
        print(f"Reading from '{str(file_path)}'...")

    with open(file_path, "r") as f:
        for line in f:
            match = col_name_pattern.match(line)
            if match:
                col_names = (match.group(1), match.group(2), match.group(3))
                if verbose:
                    print(f"  found column names as {col_names}")
                lines_to_skip += 1
            elif header_pattern.match(line):
                # This would also match col_name_pattern,
                # so must be in elif clause
                lines_to_skip += 1
            else:
                if verbose:
                    print(f"  after {lines_to_skip} header lines, "
                          "assuming the rest are data")
                break

    clu_data = pd.read_csv(
        file_path, skiprows=lines_to_skip, header=None, sep=" "
    )
    clu_data = clu_data.rename(columns=dict(zip(range(3), col_names)))
    if verbose:
        print(f"  file contained {len(clu_data):,} nodes.")
    return clu_data


def load_lynch_network_priors(file_path, verbose=False):
    """ """

    # Load Lynch's network priors from matlab
    priors = sio.loadmat(file_path).get('Priors')

    # And organize it for use in python
    prior_labels = pd.Series(
        [cell[0][0] for cell in priors[0, 0]['NetworkLabels']],
        name='label',
        index=range(0, len(priors[0, 0]['NetworkLabels']))
    )
    prior_colors = pd.DataFrame(
        priors[0, 0]['NetworkColors'],
        columns=['r', 'g', 'b'],
    )
    prior_colors['label'] = prior_labels
    prior_colors['id'] = pd.Series(range(1, len(prior_labels) + 1))

    # Generate a cifti-appropriate label dict
    cifti_labels = {0: ('background', (1.0, 1.0, 1.0, 0.0))}
    for idx, row in prior_colors.iterrows():
        cifti_labels.update(
            {int(row.id): (str(row.label), (row.r, row.g, row.b, 1.0))}
        )

    # Create a structure to store Lynch's data pythonically
    Priors = namedtuple(
        'Priors',
        'fc spatial labels cifti_labels num_networks'
    )
    return Priors(
        priors[0, 0]['FC'],
        priors[0, 0]['Spatial'],
        prior_colors[['id', 'label', 'r', 'g', 'b']].reset_index(drop=True),
        cifti_labels,
        priors[0, 0]['FC'].shape[1],
    )

