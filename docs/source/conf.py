import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Add project root and src folder to sys.path so autodoc can find modules
# project_root = Path(__file__).resolve().parent.parent.parent  # adjust if conf.py is in docs/source
# src_root = project_root / "src"
# sys.path.insert(0, str(src_root))  # only add src, not project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------
project = "VisionPipe"
copyright = "2026, Nathaniel Yeo, Muhammad Tufail"
author = "Nathaniel Yeo, Muhammad Tufail, Jake Tensen"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",   # generate docs from docstrings
    "sphinx.ext.napoleon",  # support Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # add links to source code
    "sphinx.ext.todo",      # handle .. todo:: directives
    'sphinx_simplepdf',
]

# Show todos in the documentation
todo_include_todos = True

# Mock heavy external dependencies so autodoc works without them
autodoc_mock_imports = [
    "cv2",
    "torch",
    "ultralytics",
    "numpy",
    "pandas",
    "onvif",
    "pymongo",
    "PIL",
    "zeep",
    "pydantic",
    "piexif",
    "fastapi",
    "flask",
    "deepstream", 
    "gi", 
    "motor", 
    "streamlit", 
    "client",
    "bson",
    "httpx",
    "aiofiles"
]


pdf_use_toc = True
pdf_toc_depth = 3
pdf_use_numbered_links = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "**/__pycache__", "**/.ipynb_checkpoints"]

# -- HTML output -------------------------------------------------------------
html_theme = "alabaster"  # change to 'sphinx_rtd_theme' if installed
html_static_path = ["_static"]
# html_css_files = [
#     'custom.css',
# ]

# -- Autodoc options to reduce duplicate warnings -----------------------------
# Ignore duplicates caused by __init__.py re-exports
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
    "imported-members": False,  # don't document imported/re-exported members
}

# This will supress the warnings on startup to help narrow down issues.
suppress_warnings = ['ref.duplicate', "ref.python","autosectionlabel.*"]

nitpick_ignore = [
    ('py:class', 'Any'),
    ('py:class', 'BytesIO'),
    ('py:class', 'datetime.datetime'),
    ('py:class', '_io.BytesIO'),
]

