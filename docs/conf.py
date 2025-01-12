import os
import sys
sys.path.insert(0, os.path.abspath('../src'))  # Source code dir relative to this file


project = "mfs-tools"
html_theme = "sphinx_rtd_theme"
copyright = "2024, Mike Schmidt"
version = "0.0.1"

extensions = [
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
