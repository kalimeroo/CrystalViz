"""Shared bootstrap for the CrystalViz headless tests.

Blender runs these with its own bundled CPython, which does not see the add-on
package or gemmi unless they are placed on ``sys.path``.  Import this module
*before* ``import CrystalViz``: it puts the add-on parent directory and a gemmi
build on ``sys.path`` and exposes the repo-relative paths.

gemmi resolution order:
  1. ``CRYSTALVIZ_GEMMI_DIR`` environment variable (a directory containing the
     ``gemmi`` package).
  2. ``<repo>/tests/_gemmi`` (drop an extracted gemmi there for a local run).
  3. the legacy scratch extraction ``%TEMP%/crystalviz_gemmi``.
  4. the newest ``<repo>/wheels/gemmi-*.whl``, unpacked once into
     ``<repo>/tests/_gemmi`` (gitignored).
"""
import glob
import os
import sys
import zipfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
ADDON_PARENT = os.path.dirname(REPO_DIR)
CIF_DIR = os.path.join(REPO_DIR, "CIF")
GEMMI_LOCAL = os.path.join(TESTS_DIR, "_gemmi")


def _has_gemmi(directory):
    return bool(directory) and os.path.isdir(os.path.join(directory, "gemmi"))


def _gemmi_dir():
    override = os.environ.get("CRYSTALVIZ_GEMMI_DIR")
    if _has_gemmi(override):
        return override
    if _has_gemmi(GEMMI_LOCAL):
        return GEMMI_LOCAL
    legacy = os.path.join(
        os.environ.get("TEMP", os.path.expanduser("~")), "crystalviz_gemmi"
    )
    if _has_gemmi(legacy):
        return legacy
    wheels = sorted(glob.glob(os.path.join(REPO_DIR, "wheels", "gemmi-*.whl")))
    if wheels:
        os.makedirs(GEMMI_LOCAL, exist_ok=True)
        with zipfile.ZipFile(wheels[-1]) as archive:
            archive.extractall(GEMMI_LOCAL)
        if _has_gemmi(GEMMI_LOCAL):
            return GEMMI_LOCAL
    raise RuntimeError(
        "gemmi was not found. Point CRYSTALVIZ_GEMMI_DIR at a directory that "
        "contains the gemmi package, or run "
        "`python build_extension.py download` to populate wheels/."
    )


GEMMI_DIR = _gemmi_dir()
for _path in (GEMMI_DIR, ADDON_PARENT):
    if _path and _path not in sys.path:
        sys.path.insert(0, _path)
