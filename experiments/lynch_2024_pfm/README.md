This area implements Charles Lynch's methods from his paper,
[Frontostriatal salience network expansion in individuals in depression](https://doi.org//s41586-024-07805-2)
with [matlab code](https://github.com/cjl2007/PFM-Depression) in python rather than matlab.

Lynch's code is organized in pfm_tutorial.m, which calls several functions
to package up blocks of functionality. Those blocks are implemented here
similarly. Individual jupyter notebooks were used while developing the scripts
to run through processing one step at a time. Generally speaking, the python 
notebooks can be a little sloppy because they check steps,
sometimes do things multiple times, and can have old code for reference and
experimentation. The scripts are cleaner, further developed, and more
efficient to do the job once without fuss.
Obviously, you'll need to `pip install mfs_tools` or 
`git clone https://github.com/mfschmidt/mfs_tools.git` in your local python
environment to do any of this.

Some differences are that Lynch's example data came from multi-echo fMRI
pre-processed by their [pre-processing pipeline](https://github.com/cjl2007/Liston-Laboratory-MultiEchofMRI-Pipeline).
I've tried to make the python code flexible, but it was written to analyze
single-echo BOLD data pre-processed by [fMRIPrep](https://fmriprep.org/en/stable/),
followed by [xcp_d](https://xcp-d.readthedocs.io/en/latest/). As with everything,
your mileage may vary depending on your use case.


Step 1. Temporal concatenation is implemented haphazardly in
`01_concat_check.ipynb`. It is flexible and robust in the command-line python script 
`mfs_concat.py`. Either approach will call the same underlying python functions
to concatenate input files into a single output file. Know your data, though.
Typically, xcp_d output has already removed motion outlier frames and preliminary
non-steady-state frames from fMRI, so you don't need to do that again.
For usage and options, just run the script without arguments.

Step 2. A distance matrix can be created if necessary, and used to regress
cortical signal out of subcortical voxels that may be influenced by
the stronger cortical signal. Examples here use template surfaces to
generate distance matrices, but for individual network atlases, each
individual's surface geometry will be more precise.

Step 3. 