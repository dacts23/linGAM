# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'magli'
copyright = '2026, David Ávila Cortés'
author = 'David Ávila Cortés'
release = '0.3.1'
version = '0.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_title = 'magli: Generalized Additive Models in Python'
html_short_title = 'magli'
html_baseurl = 'https://dacts23.github.io/magli/'

html_meta = {
    'description': 'Modern Pythonic Generalized Additive Models (GAMs) in pure Python with Wood (mgcv REML) and Eilers-Marx (P-splines) backends.',
    'keywords': 'python, gam, generalized additive models, splines, mgcv, pygam, reml, p-splines, interpretable machine learning, regression',
}


pygments_style = 'github-dark'
pygments_dark_style = 'github-dark'

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}
autoclass_content = 'both'

autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_ivar = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

viewcode_follow_imported_members = True
html_show_sourcelink = True

html_static_path = ['_static']
html_css_files = ['custom.css']

mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}
