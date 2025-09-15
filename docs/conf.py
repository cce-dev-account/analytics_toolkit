# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Add the project root and src directory to Python path
docs_root = Path(__file__).parent
project_root = docs_root.parent
src_root = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_root))

# -- Project information -----------------------------------------------------

project = "Analytics Toolkit"
copyright = "2024, Analytics Team"
author = "Analytics Team"

# The full version, including alpha/beta/rc tags
release = "0.1.0"
version = "0.1"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core autodoc functionality
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.intersphinx",  # Link to other project's documentation
    "sphinx.ext.mathjax",  # Render math via JavaScript
    "sphinx.ext.githubpages",  # GitHub Pages support
    "sphinx_rtd_theme",  # Read the Docs theme
    "myst_parser",  # Markdown support
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  See the documentation for
# further information on this.
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "analytics_anonymize_ip": False,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom CSS files
html_css_files = [
    "custom.css",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -------------------------------------------

# This value selects what content will be inserted into the main body of an autoclass directive.
autoclass_content = "both"

# This value selects if automatically documented members are sorted alphabetical (value 'alphabetical'),
# by member type (value 'groupwise') or by source order (value 'bysource').
autodoc_member_order = "bysource"

# This value is a list of autodoc directive flags that should be automatically applied to all autodoc directives.
autodoc_default_flags = ["members", "undoc-members", "show-inheritance"]

# Mock imports for modules that might not be available during doc building
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "statsmodels",
    "sklearn",
    "scipy",
    "plotly",
]

# -- Options for autosummary extension ---------------------------------------

# Boolean indicating whether to scan all found documents for autosummary directives,
# and to generate stub pages for each.
autosummary_generate = True

# If true, autosummary overwrites existing files by generated stub pages.
autosummary_generate_overwrite = True

# -- Options for intersphinx extension ---------------------------------------

# This config value contains the locations and names of other projects that should be linked to in this documentation.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}

# -- Options for Napoleon extension ------------------------------------------

# True to parse NumPy style docstrings. False to disable NumPy style docstrings.
napoleon_numpy_docstring = True

# True to parse Google style docstrings. False to disable Google style docstrings.
napoleon_google_docstring = True

# True to include private members (like _membername) with docstrings in the documentation.
napoleon_include_private_with_doc = False

# True to include special members (like __membername__) with docstrings in the documentation.
napoleon_include_special_with_doc = True

# True to use the .. admonition:: directive for the Example and Examples sections.
napoleon_use_admonition_for_examples = False

# True to use the .. admonition:: directive for Notes sections.
napoleon_use_admonition_for_notes = False

# True to use the .. admonition:: directive for References sections.
napoleon_use_admonition_for_references = False

# True to use the :ivar: role for instance variables.
napoleon_use_ivar = False

# True to use a :param: role for each function parameter.
napoleon_use_param = True

# True to use a :rtype: role for the return type.
napoleon_use_rtype = True

# -- Options for MyST parser ---------------------------------------------

# Enable some useful MyST syntax extensions
myst_enable_extensions = [
    "colon_fence",  # ::: fences
    "deflist",  # Definition lists
    "html_admonition",  # HTML-style admonitions
    "html_image",  # HTML images
    "linkify",  # Auto-convert URLs to links
    "replacements",  # Text replacements
    "smartquotes",  # Smart quotes
    "substitution",  # Variable substitutions
    "tasklist",  # Task lists
]

# -- Custom configuration ------------------------------------------------


def setup(app):
    """Custom setup function for Sphinx."""
    app.add_css_file("custom.css")


# Version information
def get_version():
    """Get version from pyproject.toml or package."""
    try:
        import analytics_toolkit

        return analytics_toolkit.__version__
    except ImportError:
        try:
            import tomllib

            with open(project_root / "pyproject.toml", "rb") as f:
                data = tomllib.load(f)
                return data["tool"]["poetry"]["version"]
        except (ImportError, FileNotFoundError, KeyError):
            return "0.1.0"


# Update version dynamically
try:
    release = get_version()
    version = ".".join(release.split(".")[:2])
except Exception:
    pass  # Use default values

# GitHub Pages configuration
html_baseurl = "https://your-username.github.io/analytics-toolkit/"

# Add edit on GitHub link
html_context = {
    "display_github": True,
    "github_user": "your-username",
    "github_repo": "analytics-toolkit",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# LaTeX output configuration
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files.
latex_documents = [
    (
        master_doc,
        "AnalyticsToolkit.tex",
        "Analytics Toolkit Documentation",
        "Analytics Team",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "analytics-toolkit", "Analytics Toolkit Documentation", [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files.
texinfo_documents = [
    (
        master_doc,
        "AnalyticsToolkit",
        "Analytics Toolkit Documentation",
        author,
        "AnalyticsToolkit",
        "Python Analytics Toolkit with PyTorch.",
        "Miscellaneous",
    ),
]
