#!/usr/bin/env python3

# mfs_decode.py

import sys
from pathlib import Path
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from nilearn.signal import clean
from nilearn.image import resample_to_img, smooth_img  # , index_img
from rich import print

from mfs_tools.library.file_stuff import index_img  # Uses nilearn index_img for nifti, but also handles cifti


if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))
# for p in sys.path:
#     print(f"  {p} in path")
from library.file_stuff import get_img_and_desc


"""
The main kernel of decoding is on one run with one mask/weights pair.
So the 'for each subject, for each run' stuff should happen in the shell,
where each iteration should execute this file. The current implementation
uses the 'decode_everything_with_python.sh' script alongside this one.
Note that this script uses several python libraries that must be
installed for successful execution, too. It usually requires activating
a virtual environment before running the shell script.

The original matlab decoder also spends a lot of lines of code on picking
out the TRs of interest. This decoder doesn't care. It will just decode
the entire BOLD file start to finish. The user can pick out their own
blocks/periods/trials however they like.

"""

# Trigger printing in red to highlight problems
err = f"[red]ERROR: [/red]"

def get_env(args):
    """ Integrate environment variables into our args. """

    # No environment variables are useful, so just return args unmolested.
    return args


class App:
    def __init__(self):
        self.get_arguments()
        self.args = get_env(self.args)
        self.validate_args()

        # App globals to be set later
        self.bold_img = None
        self.bold_data = None
        self.bold_residuals = None

    def get_arguments(self):
        """ Parse command line arguments """

        parser = argparse.ArgumentParser(
            description="Apply one or more decoder(s) (mask and weights) to a 4D BOLD file.",
        )
        parser.add_argument(
            "bold_file",
            help="The file containing 4D BOLD data",
            # If not yet smoothed, smoothing can be applied.
            # If not yet cropped, cropping can be applied.
            # If not yet residualized for motion, motion can be removed, too.
        )
        parser.add_argument(
            "decoder_files",
            nargs="+",
            help="One or more files containing 3D decoder weights. "
                 "Each decoder will get its own output file. ",
        )
        parser.add_argument(
            "--decoder-mask",
            help="Optionally, apply a mask to the decoder weights, so that only "
                 "weights within the mask are applied to the BOLD data.",
        )
        parser.add_argument(
            "--confounds",
            help="Optionally, mfs_decode.py can regress out vectors in each "
                 "column of a tsv file, w/ header. This file can be created by "
                 "extracting selected columns from fMRIPrep confounds",
            # A potential upgrade is accepting full fMRIPrep confounds and a spec
            # for which confounds to pull out of it and how many TRs to crop off
            # the top of it (which would match the non-steady-state TRs in BOLD).
            # This script would then do the crop/extract without the extra file.
        )
        parser.add_argument(
            "--ignore-motion-outliers", action="store_true",
            help="by default, motion outliers will be added to the confounds "
                 "matrix (assuming fMRIPrep confounds); you can turn that off"
                 "with --ignore-motion-outliers.",
        )
        parser.add_argument(
            "--confound-strategy", type=str, default="",
            help="Optionally, extract a collection of confounds from the "
                 "confounds file for regression \n"
                 "'motion_6' regresses out 3 translation and 3 rotation confounds."
                 "'motion_7' adds csf_wm to the six motion confounds."
                 "'motion_25' adds derivatives and powers to motion confounds.",
        )
        parser.add_argument(
            "--normalize", type=str, default="each",
            help="By default, --normalize each, the decoder weights are "
                 "normalized to mean 0.0 and sd 1.0, and each voxel of BOLD activity "
                 "is normalized to mean 0.0 and sd 1.0 separately.\n"
                 "  'none' leaves both BOLD and decoder weights as they come.\n"
                 "  'bold' normalizes the BOLD, temporally by voxel, but not the decoder.\n"
                 "  'decoder' normalizes decoder weights, but not the BOLD.\n"
                 "  'each' normalizes decoder weights and BOLD separately (default, recommended).\n"
                 "  'result' normalizes the final scores to 0+-1.\n",
        )
        parser.add_argument(
            "--clip", type=int, default=0,
            help="Optionally, clip the first N volumes as non-steady-state"
                 "outliers, before smoothing or analysis. The confounds will "
                 "be clipped to match.",
        )
        parser.add_argument(
            "--smooth", type=float, default=0.0,
            help="Optionally, apply smoothing to the BOLD data with "
                 "full width at half maximum (fwhm) specified",
        )
        parser.add_argument(
            "--output-path", default=".",
            help="write output files here, rather than in the current directory",
        )
        parser.add_argument(
            "--force", action="store_true",
            help="set to force overwriting of existing data",
        )
        parser.add_argument(
            "--save-intermediates", action="store_true",
            help="set to save out resampled decoders and masks",
        )
        parser.add_argument(
            "--verbose", action="store_true",
            help="set to trigger verbose output",
        )
        parser.add_argument(
            "--debug", action="store_true",
            help="set to trigger output of some raw BOLD data for review",
        )

        self.args = parser.parse_args()

    def validate_args(self):
        """ Validate arguments """

        if self.args.verbose:
            print(f"Running from {str(Path.cwd())}")

        we_have_a_fatal_error = False

        for p, desc, optional in [
            (self.args.bold_file, 'bold_file', False),
            (self.args.decoder_mask, 'decoder_mask', True),
            (self.args.confounds, 'confounds', True),
        ] + [
            (decoder_file, f'decoder_file {i}', False)
            for i, decoder_file in enumerate(self.args.decoder_files)
        ]:
            if p is not None and Path(p).exists():
                if self.args.verbose:
                    print(f"[green]Path '{p}' exists for '{desc}'.[/green]")
                setattr(self.args, desc, Path(p).resolve())
            elif p is not None and not Path(p).exists():
                if self.args.verbose:
                    print(f"[red]Path '{p}' for '{desc}' does not exist.[/red]")
                we_have_a_fatal_error = True
            elif p is None and not optional:
                if self.args.verbose:
                    print(f"[red]Path for '{desc}' is required, but not provided.[/red]")
                we_have_a_fatal_error = True
            else:
                print(f"No optional '{desc}' will be applied.")

        # Store paths as Path objects rather than strings
        setattr(self.args, "output_path", Path(self.args.output_path).resolve())
        if self.args.output_path.exists():
            existing_score_files = list(self.args.output_path.glob("*.tsv"))
            if len(existing_score_files) > (len(self.args.decoder_files) * 2):
                if self.args.force:
                    print(f"[red]{len(existing_score_files)}/{len(self.args.decoder_files) * 2} "
                          f"decoder scores already exist at {str(self.args.output_path)}. "
                          f"We will overwrite them because --force was set.[/red]")
                else:
                    print(f"[red]{len(existing_score_files)}/{len(self.args.decoder_files) * 2} "
                          f"decoder scores already exist at {str(self.args.output_path)}. "
                          f"To overwrite them, run with --force[/red]")
                    we_have_a_fatal_error = True
            elif len(existing_score_files) > 0:
                if self.args.force:
                    print(f"[red]{len(existing_score_files)}/{len(self.args.decoder_files) * 2} "
                          f"decoder scores already exist at {str(self.args.output_path)}. "
                          f"We will overwrite them because --force was set.[/red]")
                else:
                    print(f"[red]{len(existing_score_files)}/{len(self.args.decoder_files) * 2} "
                          f"decoder scores already exist at {str(self.args.output_path)}. "
                          f"Running to fill in the missing scores.[/red]")
        self.args.output_path.mkdir(parents=True, exist_ok=True)

        if we_have_a_fatal_error:
            sys.exit(1)

    def get_data_from_image(self, img):
        """ Extract the data from a nibabel image. """

        _data = np.array([])

        if isinstance(img, nib.Nifti1Image):
            _data = img.get_fdata()
        elif isinstance(img, nib.Nifti2Image):
            _data = img.get_fdata()
        elif isinstance(img, nib.cifti2.Cifti2Image):
            if len(img.header.mapped_indices) == 2:
                if (
                    (
                        isinstance(
                            img.header.get_axis(img.header.mapped_indices[0]),
                            nib.cifti2.cifti2_axes.SeriesAxis
                        ) or
                        isinstance(
                            img.header.get_axis(img.header.mapped_indices[0]),
                            nib.cifti2.cifti2_axes.ScalarAxis
                        )
                    ) and
                    isinstance(
                        img.header.get_axis(img.header.mapped_indices[1]),
                        nib.cifti2.cifti2_axes.BrainModelAxis
                    )
                ):
                    # Extract the [time x locus] data and transpose to [locus x time]
                    _data = img.get_fdata().T
                elif (
                    isinstance(
                        img.header.get_axis(img.header.mapped_indices[0]),
                        nib.cifti2.cifti2_axes.BrainModelAxis
                    ) and
                    (
                        isinstance(
                            img.header.get_axis(img.header.mapped_indices[1]),
                            nib.cifti2.cifti2_axes.SeriesAxis
                        ) or
                        isinstance(
                            img.header.get_axis(img.header.mapped_indices[1]),
                            nib.cifti2.cifti2_axes.ScalarAxis
                        )
                    )
                ):
                    # Extract the [locus x time] data as it is
                    _data = img.get_fdata()
            else:
                raise ValueError(f"Unsupported CIFTI2 image with {len(img.header.mapped_indices)} ")

        return _data

    def load_bold_image(self):
        """ Load the BOLD data file, smoothing and clipping as requested. """

        # Load the BOLD data whether from nifti or cifti
        _bold_img, _bold_desc = get_img_and_desc(
            self.args.bold_file,
            verbose=self.args.verbose
        )
        if self.args.verbose:
            print(f"Loaded a {_bold_img.shape} BOLD image")

        if (self.args.clip is not None) and (self.args.clip != 0):
            # Remove the initial volumes from the BOLD data
            if self.args.verbose:
                print(f"  clip the first {self.args.clip} volumes")
            _bold_img = index_img(
                _bold_img, slice(self.args.clip, None), self.args.verbose
            )
        else:
            if self.args.verbose:
                if self.args.clip is None:
                    print(f"  not clipping any volumes, --clip was not set.")
                if self.args.clip == 0:
                    print(f"  not clipping any volumes, --clip was set to 0.")

        if (self.args.smooth is not None) and (self.args.smooth != 0.0):
            if self.args.verbose:
                print(f"  smoothing the BOLD image with a "
                      f"{self.args.smooth:0.1f}mm Gaussian kernel")
            _bold_img = smooth_img(_bold_img, self.args.smooth)

        if self.args.verbose:
            print(f"Extracted {_bold_desc} BOLD data with shape {_bold_img.shape}")

        return _bold_img

    def load_decoder_weights(self, decoder_file):
        """ Load the decoder data file. """

        decoder_img, decoder_desc = get_img_and_desc(decoder_file, verbose=self.args.verbose)
        decoder_weights = self.get_data_from_image(decoder_img)
        if self.args.verbose:
            print(f"Loaded a {decoder_img.shape} decoder image")
            print(f"Extracted decoder weights with shape {decoder_weights.shape}")
            print(f"Decoder {decoder_file.stem}, {decoder_desc}, "
                  f"loaded with {np.sum(decoder_weights.astype('bool')):,} hot voxels")

        # We regularly deal with decoders that are in LAS+ (FSL's MNI152)
        # rather than RAS+ (templateflow's MNI152s) or in a different
        # space, like Phil Kragel's occipital lobe weights, etc.
        # We anticipate this, and resample the decoder weights into BOLD
        # space. We use nearest-neighbor resampling because expanding the
        # edges into the zeros can significantly increase computation of all
        # the nearly-zero weights that ought to simply be masked out.
        # This resampling does NOT guarantee or even imply that the
        # brains will overlap or be co-registered. All decoders we've
        # encountered thus far are aligned with one of the MNI spaces,
        # so this code works, but will emit invalid results if we ever
        # find a decoder in its own space.
        if (
                isinstance(self.bold_img, nib.Nifti1Image) or
                isinstance(self.bold_img, nib.Nifti2Image)
        ) and (
                (decoder_img.shape != self.bold_img.shape[:3]) or
                (not np.allclose(decoder_img.affine, self.bold_img.affine))
        ):
            if self.args.verbose:
                print(f"  [yellow]WARNING : The decoder is not in the same "
                      f"space as the BOLD data. Resampling decoder from "
                      f"{decoder_img.shape} to "
                      f"{self.bold_img.shape[:3]} to match BOLD image. "
                      f"If BOLD data are in subject-space, images will "
                      f"be misaligned and results will not be valid.[/yellow]")
            decoder_img = resample_to_img(
                decoder_img, self.bold_img,
                interpolation='nearest', force_resample=True, copy_header=True,
            )
            decoder_weights = decoder_img.get_fdata()
            if self.args.verbose:
                print(f"  the {decoder_img.shape} decoder now has "
                      f"{np.sum(decoder_weights.astype('bool')):,} "
                      f"{tuple([float(_) for _ in decoder_img.header.get_zooms()])}"
                      f" hot voxels")

        return decoder_img, decoder_weights

    def unused_save_bold_img(self, bold_data, img_path):
        """ Save data to same image type as BOLD Image. """

        if isinstance(self.bold_img, nib.Nifti1Image) or isinstance(self.bold_img, nib.Nifti2Image):
            save_img = self.bold_img.__class__(bold_data, self.bold_img.affine, self.bold_img.header)
            save_img.to_filename(img_path)
            return
        if isinstance(self.bold_img, nib.Cifti2Image):
            series_axis, brain_axis, new_data = None, None, None
            for ax_idx in self.bold_img.header.mapped_indices:
                ax = self.bold_img.header.get_axis(ax_idx)
                if isinstance(ax, nib.cifti2.cifti2_axes.SeriesAxis):
                    tr_len = ax.step
                    series_axis = nib.cifti2.SeriesAxis(
                        start=0, step=tr_len, size=len(ax)
                    )
                elif isinstance(ax, nib.cifti2.cifti2_axes.BrainModelAxis):
                    brain_axis = ax

            if series_axis and brain_axis:
                if (
                    (self.bold_data.shape[0] == len(series_axis)) and
                    (self.bold_data.shape[1] == len(brain_axis))
                ):
                    new_data = bold_data
                elif (
                    (self.bold_data.shape[1] == len(series_axis)) and
                    (self.bold_data.shape[0] == len(brain_axis))
                ):
                    new_data = bold_data.T
                else:
                    raise ValueError(f"Bold image {self.bold_img.shape} does "
                                     f"not match data shape {bold_data.shape} ")
            if new_data is not None:
                new_img = nib.cifti2.Cifti2Image(
                    new_data, (series_axis, brain_axis)
                )
                new_img.update_headers()
                new_img.to_filename(img_path)
            else:
                raise ValueError(f"Bold image {self.bold_img.shape} does not "
                                 f"have expected SeriesAxis and BrainModelAxis.")
        else:
            raise TypeError(f"Unsupported image type: {type(self.bold_img)}")

    def remove_motion(
            self,
            strategy='',
            remove_spikes=True,
            method='manual'
    ):
        """ Regress out motion confounds, return scaled residuals. """

        if self.args.confounds.name.endswith(".tsv"):
            # fMRIPrep prepares a tab-separated table, with a header row
            confounds = pd.read_csv(self.args.confounds, sep='\t', header=0)
            if self.args.verbose:
                print(f"loaded confounds for {len(confounds)} time points, "
                      f"to match data with {self.bold_data.shape[-1]} time points")
        elif self.args.confounds.name.endswith(".par"):
            # If motion correction was done by FSL Feat, double-spaces
            confounds = pd.read_csv(self.args.confounds, sep=r'\s+', header=None)
            if self.args.verbose:
                print(f"loaded confounds for {len(confounds)} time points, "
                      f"to match data with {self.bold_data.shape[-1]} time points")
        elif self.args.confounds.name == "Movement_Regressors.txt":
            # If motion correction was done by the HCP, variable space
            confounds = pd.read_csv(self.args.confounds, sep=r'\s+', header=None)
            confounds.rename(columns={
                0: 'trans_x', 1: 'trans_y', 2: 'trans_z', 3: 'rot_x', 4: 'rot_y', 5: 'rot_z',
                6: 'trans_x_derivative1', 7: 'trans_y_derivative1', 8: 'trans_z_derivative1',
                9: 'rot_x_derivative1', 10: 'rot_y_derivative1', 11: 'rot_z_derivative1',
            }, inplace=True)
            if self.args.verbose:
                print(f"loaded confounds for {len(confounds)} time points, "
                      f"to match data with {self.bold_data.shape[-1]} time points")
        else:
            if self.args.verbose:
                print(f"[yellow]WARNING : No confound file, not removing motion"
                      f"[/yellow]")
            raise FileNotFoundError(f"Could not find '{self.args.confounds}'")

        confound_clip_num = len(confounds) - self.bold_data.shape[-1]
        if confound_clip_num > 0:
            if self.args.verbose:
                print(f"Clipping first {confound_clip_num} of {len(confounds)} "
                      f"confounds values to match BOLD length")
            confounds = confounds.iloc[confound_clip_num:, :]

        # Find motion outliers
        if remove_spikes:
            # We need to check for sum() > 0 because we just clipped some rows
            # above, and don't need to include columns representing a spike there.
            spike_cols = [col for col in confounds.columns
                          if (col.startswith('motion_outlier') and
                              confounds[col].sum() > 0)]
            if self.args.verbose:
                print(f"Including {len(spike_cols)} motion outlier (spike) columns")
        else:
            spike_cols = []


        # If a specific strategy was requested, extract the appropriate columns
        cols_to_use = confounds.columns
        motion_6_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        deriv_6_cols = [f"{motion}_derivative1" for motion in motion_6_cols]
        power_6_cols = [f"{motion}_power2" for motion in motion_6_cols]
        power_deriv_6_cols = [f"{motion}_derivative1_power2" for motion in motion_6_cols]
        if strategy == 'motion_6':
            cols_to_use = [col for col in motion_6_cols + spike_cols
                           if col in confounds.columns]
            if self.args.verbose:
                print(f"Extracting six motion columns from confounds file "
                      f"(actual {len(cols_to_use)} columns)")
        elif strategy == 'motion_7':
            cols_to_use = [col for col in motion_6_cols + ['csf_wm', ] + spike_cols
                           if col in confounds.columns]
            if self.args.verbose:
                print(f"Extracting 'csf_wm' and six motion columns from confounds "
                      f"file (actual {len(cols_to_use)} columns)")
        elif strategy == "motion_25":
            motion_24_cols = motion_6_cols + deriv_6_cols + power_6_cols + power_deriv_6_cols
            cols_to_use = [col for col in motion_24_cols + ['csf_wm', ] + spike_cols
                           if col in confounds.columns]
            if self.args.verbose:
                print(f"Extracting 'csf_wm' and 24 motion columns from confounds "
                      f"file (actual {len(cols_to_use)} columns)")
        confounds = confounds[cols_to_use]

        # Ensure the y-intercept, or arbitrary mean BOLD, doesn't make a difference.
        confounds['bias'] = 1.0

        # One way is to do this with nilearn, in one line:
        if method == 'nilearn':
            if self.args.normalize in ["bold", "each", ]:
                scale = "zscore"
            else:
                scale = None
            # Nilearn insists we should de-trend or standardize.
            # For now, I prevent it to ensure these results are identical to matlab.
            return clean(self.bold_data.T, confounds=confounds.values, detrend=False,
                         standardize=scale, standardize_confounds=False).T

        # Another way is to replicate Noam's matlab exactly and do all of this manually:
        beta_motion = np.dot(
            self.bold_data,
            np.linalg.pinv(np.nan_to_num(confounds.values)).T
        )
        _bold_residuals = (
            self.bold_data -
            np.dot(
                beta_motion,
                np.nan_to_num(confounds.values).T
            )
        )

        if self.args.normalize in ["bold", "each", ]:
            # Compute z scores temporally, across voxel rows, NOT time columns
            # with population degrees of freedom, not sample

            # Python returns a row of NaN z-scores for a row of zero data.
            # Matlab returns a row of zeros, which is more useful.
            # Here, we zero out the NaNs to allow scoring via the other voxels.
            raw_z = zscore(_bold_residuals, axis=1, ddof=0)
            _bold_residuals = np.nan_to_num(raw_z, nan=0.0)
            if self.args.verbose:
                print(f"  After z-scoring, {np.sum(np.isnan(raw_z)):,} NaNs "
                      f"were changed to 0.0; {np.sum(np.isfinite(raw_z)):,} "
                      f"were already finite. Z-scores shaped {_bold_residuals.shape}.")

        return _bold_residuals

    def load_and_mask_data(
            self, decoder_file, mask_file=None, output_path=None
    ):
        """ Load 4D bold data, mask it, and return 2D matrix. """

        # Name the decoder, without the .nii.gz
        decoder_stem = Path(decoder_file).name.split(".")[0]
        decoder_extension = "".join(Path(decoder_file).suffixes)

        # The BOLD image is the standard; weights and masks must be resampled
        # to match it, not the other way around.
        decoder_img, decoder_weights = self.load_decoder_weights(decoder_file)
        if self.args.output_path is not None:
            decoder_img.to_filename(self.args.output_path / f"decoder_{decoder_stem}_2{decoder_extension}")

        # Now that we have the decoder in BOLD space, should we also mask it?
        if mask_file is None:
            print(f"  there is no mask, using non-zero decoder weights.")
            if isinstance(self.bold_img, nib.Nifti1Image) or isinstance(self.bold_img, nib.Nifti2Image):
                x_res, y_res, z_res = decoder_img.header.get_zooms()
                voxel_volume = x_res * y_res * z_res
                decoder_vol = np.sum((decoder_weights != 0.0).astype('bool')) * voxel_volume
                zooms = ", ".join([f"{z:0.2f}" for z in decoder_img.header.get_zooms()])
                print(f"  the decoder's non-zero weights {decoder_img.shape}, "
                      f"({zooms}), {decoder_vol:0.1f}mm3")
            elif isinstance(self.bold_img, nib.Cifti2Image):
                decoder_vol = np.sum((decoder_weights != 0.0).astype('bool'))
                print(f"  the decoder's non-zero weights {decoder_weights.shape}, "
                      f"{decoder_vol:,} voxels")
        else:
            # Assuming the mask and weights are in the same space
            # And everything's Nifti1 or Nifti2
            # (I haven't implemented masks in Cifti2)
            # They could be binary or only use a subset of voxels and vertices
            mask_img = nib.load(mask_file)
            if self.args.verbose:
                print(f"  a {mask_img.shape} mask was loaded with "
                      f"{np.sum(mask_img.get_fdata().astype('bool')):,} hot voxels")
            if (    (not np.allclose(self.bold_img.affine, mask_img.affine)) or
                    (self.bold_data.shape != mask_img.shape)
            ):
                print(f"  [yellow]WARNING : The decoder weights and the mask "
                      f"are not in the same space. Resampling...[/yellow]")
                x_res, y_res, z_res = mask_img.header.get_zooms()
                voxel_volume = x_res * y_res * z_res
                mask_vol = np.sum((mask_img.get_fdata() != 0.0).astype('bool')) * voxel_volume
                print(f"  the mask started as {mask_img.shape}, "
                      f"{mask_img.header.get_zooms()}, {mask_vol:0.1f}mm3")
                resampled_mask = resample_to_img(
                    mask_img, self.bold_img,
                    interpolation='nearest', force_resample=True
                )
                x_res, y_res, z_res = resampled_mask.header.get_zooms()
                voxel_volume = x_res * y_res * z_res
                mask_vol = np.sum((resampled_mask.get_fdata() != 0.0).astype('bool')) * voxel_volume
                print(f"  it was resampled to {resampled_mask.shape}, "
                      f"{resampled_mask.header.get_zooms()}, {mask_vol:0.1f}mm3")
                # print(f"[red]The mask must be in the same space as the decoder.[/red]")
                # raise ValueError("Decoder/Mask mismatch")
            else:
                resampled_mask = mask_img

            # Binarize the mask and filter the decoder weights by it.
            one_hot_mask = resampled_mask.get_fdata().astype("bool").astype("uint8")
            decoder_weights = decoder_weights * one_hot_mask
            decoder_img = nib.Nifti1Image(decoder_weights, decoder_img.affine)
            if self.args.verbose:
                print(f"  the {decoder_weights.shape} decoder was masked down to "
                      f"{np.sum(decoder_weights.astype('bool')):,} hot voxels.")

            if output_path and self.args.save_intermediates:
                decoder_img.to_filename(output_path / f"decoder_{decoder_stem}_3{decoder_extension}")

        # The BOLD data were previously loaded, smoothed, and residualized
        # Handle 4D data as [all_voxels x time] 2D matrix.
        if isinstance(self.bold_img, nib.Nifti1Image) or isinstance(self.bold_img, nib.Nifti2Image):
            # To match matlab and the weights, this MUST be done in fortran order.
            dims = self.bold_img.shape
            voxels_per_volume = dims[0] * dims[1] * dims[2]
            bold_full_2d_data = np.reshape(
                self.bold_residuals, (voxels_per_volume, dims[3]), order='F'
            )
            decoder_2d_data = np.reshape(
                decoder_weights, voxels_per_volume, order='F'
            )
            masked_bold_residuals = bold_full_2d_data[decoder_2d_data != 0]
            decoder_2d_data = decoder_2d_data[decoder_2d_data != 0]
        elif isinstance(self.bold_img, nib.Cifti2Image):
            # bold_data is [loci, time]
            masked_bold_residuals = self.bold_residuals[decoder_weights.ravel() != 0, :]
            decoder_2d_data = decoder_weights[decoder_weights != 0]
        else:
            raise ValueError(f"Unsupported image type: {type(self.bold_img)}")
        if self.args.verbose:
            print(f"  masked BOLD data are now shaped {masked_bold_residuals.shape} "
                  f"and have {np.sum(masked_bold_residuals != 0.0):,} values.")
            print(f"  weights are now shaped {decoder_2d_data.shape} "
                  f"and have {np.sum(decoder_2d_data != 0.0):,} values.")

        # Add a bias, for the intercept. This is never zero or one in Noam's
        # decoders, though. :( I am using a 0-intercept, and a brief
        # investigation looked like putting it AFTER the data fits best.
        weights = np.append(decoder_2d_data, 0.0)
        print(f"  mean weight value: {np.mean(weights):.3f} "
              f"({np.min(weights):.3f} to {np.max(weights):.3f})")

        return masked_bold_residuals, weights, decoder_stem

    def predict_y(self, data, weights):
        """ Use measured BOLD data (cleaned) to predict y """

        # Normally, we use a decoder, which is a vector of weights.
        # But we may also want to use all ones for the decoder as a null comparison.
        # This allows us to see if the mask itself is responsible for any
        # significant decoder effect, without the weights.
        if np.array_equal(np.ones(weights.shape), weights):
            words = "created", "as ones"
        else:
            words = "extracted", "from decoder volume"
            # Normalize decoder weights before decoding
            if self.args.normalize in ["decoder", "each", ]:
                weights = (weights - weights.mean()) / weights.std()

        if self.args.verbose:
            print(f"  - {words[0]} {len(weights)} weights {words[1]}")

        # Normalize BOLD data before decoding
        if self.args.normalize in ["bold", "each", ]:
            data = data - data.mean(axis=-1, keepdims=True)
            data = data / data.std(axis=-1, keepdims=True)

        if data.shape[0] == weights.shape[0]:
            # No intercept, use as-is
            x = data
        else:
            # The weights have an intercept, add ones to the data
            x = np.append(data, np.ones((1, data.shape[1])), axis=0)
        y_hat = np.dot(weights.T, x).T

        if self.args.normalize == "result":
            y_hat = (y_hat - y_hat.mean()) / y_hat.std()

        # This is the decoder score for each t
        return y_hat

    @staticmethod
    def write_some_matrices(data):
        if len(data.shape) != 3:
            print(f"not debugging {data.shape}-shaped matrix; expecting 3D.")
        # Select a 3D patch that contains different mask labels
        samp_x = 20  # int(data.shape[0] / 3)
        samp_y = 72  # int(data.shape[1] * 2 / 3)
        samp_z = 32  # int(data.shape[2] / 2)
        for _z in range(samp_z, samp_z + 3):
            print(
                f"z slice {_z}; "
                f"x = {samp_x} to {samp_x + 6}, "
                f"y = {samp_y} to {samp_y + 4}:"
            )
            for _y in range(samp_y, samp_y + 5):
                print(
                    f"y={_y:>3}:  " + ", ".join([
                        f"{data[_x, _y, _z]:0.4f}"
                        for _x in range(samp_x, samp_x + 7)
                    ])
                )


    def run(self):
        """ Entry point """

        if len(self.args.decoder_files) == 1:
            print("Decoding {} with a '{}' decoder.".format(
                str(self.args.bold_file), Path(self.args.decoder_files[0]).name
            ))
        else:
            print("Decoding {} with {} decoders.".format(
                str(self.args.bold_file), len(self.args.decoder_files)
            ))

        # 1. Load the BOLD fMRI data
        self.bold_img = self.load_bold_image()
        self.bold_data = self.get_data_from_image(self.bold_img)
        # <state:> self.bold_img contains BOLD fMRI image, clipped and smoothed
        # <state:> self.bold_data contains BOLD fMRI data from self.bold_img

        # Remove motion confounds from BOLD, if requested
        if self.args.confounds:
            self.bold_residuals = self.remove_motion(
                strategy=self.args.confound_strategy,
                remove_spikes=(not self.args.ignore_motion_outliers)
            )
        else:
            self.bold_residuals = self.bold_data
        # <state:> self.bold_residuals contains BOLD fMRI data with motion confounds removed
        # <state:> self.bold_data is not changed

        # TODO: Save self.bold_residuals as a cifti file or nifti file depending on context.
        # if self.args.save_intermediates:
        #     filename = f"final_bold{''.join(self.args.bold_file.suffixes)}"
        #     self.bold_img.to_filename(self.args.output_path / filename)

        for decoder_file in [Path(df) for df in self.args.decoder_files]:
            split_idx = max(
                decoder_file.name.find("_ones"),
                decoder_file.name.find("_weights")
            )
            decoder_name = decoder_file.name[0:split_idx]
            existing_score_files = [
                sf for sf in list(self.args.output_path.glob("*.tsv"))
                if decoder_name in sf.name
            ]
            if len(existing_score_files) > 1 and not self.args.force:
                print(f"Scores for {decoder_file.name} already exist. "
                      f"Skipping this decoder. Use --force to overwrite, "
                      f"or delete files you'd like to replace and run again.")
                continue
            elif len(existing_score_files) > 0 and not self.args.force:
                print(f"One score file for {str(decoder_file)} already exists, "
                      f"but there should be two. Trying again to generate "
                      f"both scores files for {decoder_file.name}.")

            if self.args.save_intermediates:
                intermediate_output_path = self.args.output_path
            else:
                intermediate_output_path = None
            masked_bold_residuals, weight_data, decoder_name = self.load_and_mask_data(
                decoder_file, self.args.decoder_mask,
                output_path=intermediate_output_path,
            )
            if self.args.verbose:
                print(f"  shape of loaded data    : {self.bold_data.shape}")
                print(f"  shape of residual data  : {self.bold_residuals.shape}")
                print(f"  shape of weights        : {weight_data.shape}")
                print(f"  shape of final residuals: {masked_bold_residuals.shape}")

            if self.args.debug:
                # Write out values from a specific region in each piece of data.
                self.write_some_matrices(self.bold_img.get_fdata()[:, :, :, 1])

            for label, weights in [
                ("ones", np.ones((weight_data.shape[0], 1))),
                ("weights", weight_data),
            ]:
                print(f"  - shape of weights: {weights.shape}")
                predicted_y = self.predict_y(masked_bold_residuals, weights)
                if np.sum(np.isnan(predicted_y)) > 0:
                    print("NaN values in predicted y, no scores!")
                pd.DataFrame(predicted_y).to_csv(
                    self.args.output_path / f"all_trs_{decoder_name}_{label}_scores.tsv",
                    sep='\t', header=False, index=False,
                )

        return 0


def main():
    """ Entry point """
    app = App()
    app.run()


if __name__ == "__main__":
    main()
