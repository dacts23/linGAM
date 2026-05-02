# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Allow autodoc to discover the package
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'linGAM'
copyright = '2026, David Ávila Cortés'
author = 'David Ávila Cortés'
release = '0.2.0'
version = '0.2'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "navigation_depth": 3,
    "show_prev_next": True,
    "pygments_dark_style": "monokai",
}
html_static_path = ['_static']
html_title = 'linGAM Documentation'
html_short_title = 'linGAM'
html_css_files = ['custom.css']

# -- Autodoc -----------------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}
autoclass_content = 'both'

# -- Napoleon ----------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_ivar = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Viewcode ----------------------------------------------------------------
viewcode_follow_imported_members = True
html_show_sourcelink = True

# -- MathJax -----------------------------------------------------------------
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}
