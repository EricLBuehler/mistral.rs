# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import logging
import os
import sys

import requests
from sphinx.ext import autodoc

logger = logging.getLogger(__name__)
sys.path.append(os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "mistral.rs"
copyright = f"{datetime.datetime.now().year}, mistral.rs"
author = "Eric Buehler"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinxarg.ext",
    "sphinx_design",
    "sphinx_togglebutton",
]
myst_enable_extensions = [
    "colon_fence",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = ["**/*.template.md", "**/*.inc.md"]

# Exclude the prompt "$" when copying code
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_title = project
html_theme = "sphinx_book_theme"
html_logo = "assets/logos/mistralrs-logo-text-light.png"
html_favicon = "assets/logos/mistralrs-logo-only-light.ico"
html_theme_options = {
    "path_to_docs": "docs/source",
    "repository_url": "https://github.com/EricLBuehler/mistral.rs",
    "use_repository_button": True,
    "use_edit_page_button": True,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_js_files = ["custom.js"]
html_css_files = ["custom.css"]

myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "ftp": None,
    "gh-issue": {
        "url": "https://github.com/EricLBuehler/mistral.rs/issues/{{path}}#{{fragment}}",
        "title": "Issue #{{path}}",
        "classes": ["github"],
    },
    "gh-pr": {
        "url": "https://github.com/EricLBuehler/mistral.rs/pull/{{path}}#{{fragment}}",
        "title": "Pull Request #{{path}}",
        "classes": ["github"],
    },
    "gh-dir": {
        "url": "https://github.com/EricLBuehler/mistral.rs/tree/main/{{path}}",
        "title": "{{path}}",
        "classes": ["github"],
    },
    "gh-file": {
        "url": "https://github.com/EricLBuehler/mistral.rs/blob/main/{{path}}",
        "title": "{{path}}",
        "classes": ["github"],
    },
}

# see https://docs.readthedocs.io/en/stable/reference/environment-variables.html # noqa
READTHEDOCS_VERSION_TYPE = os.environ.get("READTHEDOCS_VERSION_TYPE")
if READTHEDOCS_VERSION_TYPE == "tag":
    # remove the warning banner if the version is a tagged release
    header_file = os.path.join(
        os.path.dirname(__file__), "_templates/sections/header.html"
    )
    # The file might be removed already if the build is triggered multiple times
    # (readthedocs build both HTML and PDF versions separately)
    if os.path.exists(header_file):
        os.remove(header_file)

_cached_base: str = ""
_cached_branch: str = ""


def get_repo_base_and_branch(pr_number):
    global _cached_base, _cached_branch
    if _cached_base and _cached_branch:
        return _cached_base, _cached_branch

    url = f"https://github.com/EricLBuehler/mistral.rs/pulls/{pr_number}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        _cached_base = data["head"]["repo"]["full_name"]
        _cached_branch = data["head"]["ref"]
        return _cached_base, _cached_branch
    else:
        logger.error("Failed to fetch PR details: %s", response)
        return None, None


# Mock out external dependencies here, otherwise the autodoc pages may be blank.
autodoc_mock_imports = [
    "blake3",
    "compressed_tensors",
    "cpuinfo",
    "cv2",
    "torch",
    "transformers",
    "psutil",
    "prometheus_client",
    "sentencepiece",
    "PIL",
    "numpy",
    "triton",
    "tqdm",
    "tensorizer",
    "pynvml",
    "outlines",
    "xgrammar",
    "librosa",
    "soundfile",
    "gguf",
    "lark",
    "decord",
]

for mock_target in autodoc_mock_imports:
    if mock_target in sys.modules:
        logger.info(
            "Potentially problematic mock target (%s) found; "
            "autodoc_mock_imports cannot mock modules that have already "
            "been loaded into sys.modules when the sphinx build starts.",
            mock_target,
        )


class MockedClassDocumenter(autodoc.ClassDocumenter):
    """Remove note about base class when a class is derived from object."""

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)


autodoc.ClassDocumenter = MockedClassDocumenter

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest", None),
    "aiohttp": ("https://docs.aiohttp.org/en/stable", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "psutil": ("https://psutil.readthedocs.io/en/stable", None),
}

autodoc_preserve_defaults = True
autodoc_warningiserror = True

navigation_with_keys = False
