import nibabel as nib
from nibabel.filebasedimages import ImageFileError
from numpy import sum


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

    if isinstance(img, nib.nifti1.Nifti1Image):
        desc = f"Nifti1 file: {img.shape}"
        if only_dims:
            desc = ",".join([str(dim) for dim in img.shape])
    elif isinstance(img, nib.nifti2.Nifti2Image):
        desc = f"Nifti2 file: {img.shape}"
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
                if only_dims:
                    axes_strs.append(str(ax.size))
                else:
                    axes_strs.append(f"{ax.size} {getattr(ax, 'unit', '')}s")
            if isinstance(ax, nib.cifti2.BrainModelAxis):
                if only_dims:
                    axes_strs.append(str(ax.size))
                else:
                    axes_strs.append(f"{ax.size} grayordinates: "
                                     f"{sum(ax.volume_mask):,} volumes & "
                                     f"{sum(ax.surface_mask):,} vertices")
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


