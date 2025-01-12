Mike's mfs-tools
================

``mfs-tools`` is a library of functions I use for analyzing fMRI data.
It also contains executable scripts that package up the functionality
for use from the command-line, similar to what you'd find with FSL,
AFNI, or FreeSurfer packages.

Recursive auto-documentation
----------------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   mfs_tools

Documenting the library, not the commands
-----------------------------------------

.. automodule:: mfs_tools.library

Documenting the commands, not the library
-----------------------------------------

.. automodule:: mfs_tools.commands

Documenting two functions, one at a time
----------------------------------------

.. autofunction:: mfs_tools.library.concat_stuff.concat_dtseries
.. autofunction:: mfs_tools.library.distance_stuff.make_distance_matrix

The end
