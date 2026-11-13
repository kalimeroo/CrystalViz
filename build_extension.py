#!/usr/bin/env python3
"""Build the CrystalViz extension with the gemmi wheel bundled.

The add-on depends on ``gemmi``, a compiled C extension. Blender only ships
pure-Python and a small vendored set of wheels, so gemmi has to be bundled with
the extension as a per-platform ``.whl``. This script does exactly that:

* ``download`` fetches the correct gemmi CPython-3.13 wheel for every target
  platform from PyPI into ``wheels/`` and keeps the ``wheels`` array in
  ``blender_manifest.toml`` in sync.
* ``build`` produces one installable extension ``.zip`` per platform in
  ``dist/`` (the same output Blender's own
  ``blender --command extension build --split-platforms`` produces), with only
  the wheel matching that platform included.

It uses only the Python standard library, so it runs on any machine with
Python 3.8+ and no dependencies. The wheels are downloaded unmodified from
PyPI as required by Blender's extension guidelines.

Example
-------
    python build_extension.py download --gemmi-version 0.7.5
    python build_extension.py build

Build for a single platform::

    python build_extension.py build --platforms macos-arm64
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
import zipfile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Blender 5.x ships CPython 3.13, so we need cp313 wheels.
_PY_TAG = "cp313"

# Blender platform identifier -> the PEP 425 / pip platform tag (or tag
# family) we match against the wheel filename. "linux-x64" accepts any
# manylinux x86_64 wheel because gemmi is manylinux2014+-compatible.
_PLATFORM_TAG = {
    "windows-x64": "win_amd64",
    "linux-x64": "manylinux_",
    "macos-arm64": "macosx_",
    "macos-x64": "macosx_",
}

# A wheel's "arm" is derived from its filename platform tag.
_PLATFORM_ARCH = {
    "windows-x64": "amd64",
    "linux-x64": "x86_64",
    "macos-arm64": "arm64",
    "macos-x64": "x86_64",
}

# Officially supported target platforms (the `platforms` key in the manifest).
TARGET_PLATFORMS = ["windows-x64", "linux-x64", "macos-arm64", "macos-x64"]

# Distribution name on PyPI.
_DIST = "gemmi"

# Files from the add-on directory that go into every extension zip.
_PACKAGE_FILES = [
    "blender_manifest.toml",
    "__init__.py",
    "gemmi_worker.py",
    "README.md",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_WHEELS_DIR = os.path.join(_HERE, "wheels")
_DIST_DIR = os.path.join(_HERE, "dist")
_MANIFEST = os.path.join(_HERE, "blender_manifest.toml")


# ---------------------------------------------------------------------------
# Wheel discovery
# ---------------------------------------------------------------------------


def _wheel_platform_tags(filename):
    """Return the pip platform tag a wheel was built for.

    ``<name>-<version>-<py>-<abi>-<platform>.whl``; the platform tag is the
    final ``-``-separated field once ``.whl`` is removed. A platform tag may
    itself contain dots (e.g. ``manylinux_2_27_x86_64.manylinux_2_28_x86_64``).
    """
    base = filename[:-4]  # strip ".whl"
    return base.split("-")[-1]


def wheel_matches_platform(filename, platform):
    """Return True if *filename* (a ``*.whl``) can run on *platform*."""
    if _PY_TAG not in filename:
        return False
    arch = _PLATFORM_ARCH[platform]
    if arch not in filename:
        return False
    tags = _wheel_platform_tags(filename)
    expected = _PLATFORM_TAG[platform]
    return expected in tags


def _fetch(dist):
    url = f"https://pypi.org/pypi/{dist}/json"
    try:
        # The PyPI JSON API only ever serves https:// URLs.
        with urllib.request.urlopen(url, timeout=60) as resp:  # nosec B310
            return json.load(resp)
    except Exception as exc:  # network or JSON errors
        raise SystemExit(f"error: could not reach PyPI for '{dist}': {exc}") from exc


def _select_wheels(version):
    """Return {platform: (filename, url)} for a gemmi release.

    The newest prepublished release whose version equals *version* is used.
    """
    data = _fetch(_DIST)
    release = data["releases"].get(str(version))
    if not release:
        raise SystemExit(
            f"error: gemmi {version} not found on PyPI "
            f"(current release is {data['info']['version']})"
        )

    wheels = [u for u in release if u["packagetype"] == "bdist_wheel"]
    chosen = {}
    for platform in _PLATFORM_TAG:
        match = [u for u in wheels if wheel_matches_platform(u["filename"], platform)]
        if not match:
            raise SystemExit(f"error: no gemmi {version} cp313 wheel for {platform}")
        chosen[platform] = (match[0]["filename"], match[0]["url"])
    return chosen


def _latest_release():
    data = _fetch(_DIST)
    return data["info"]["version"]


# ---------------------------------------------------------------------------
# Manifest sync (we own the whole file, so we rewrite it cleanly)
# ---------------------------------------------------------------------------


_MANIFEST_TEMPLATE = (
    """schema_version = "1.0.0"

id = "crystalviz"
version = "0.9.9"
name = "CrystalViz"
tagline = "Import CIF atoms and render with Geometry Nodes"
maintainer = "Erwann Jeanneau"
type = "add-on"
license = [
  "SPDX:MPL-2.0",
]

tags = ["Import-Export", "Geometry Nodes", "Material"]

blender_version_min = "5.2.0"

platforms = [{platforms}]

wheels = [
{wheels}
]

[permissions]
files = "Read CIF files from disk and store imported structures"
""".rstrip()
    + "\n"
)


def _wheels_array(chosen):
    lines = []
    for platform in TARGET_PLATFORMS:
        if platform not in chosen:
            continue
        filename = chosen[platform][0]
        lines.append(f'    "./wheels/{filename}",')
    # Keep the same order as TARGET_PLATFORMS.
    return "\n".join(lines)


def write_manifest(chosen):
    """Regenerate blender_manifest.toml with the current wheel filenames."""
    platforms = ", ".join(f'"{p}"' for p in TARGET_PLATFORMS)
    text = _MANIFEST_TEMPLATE.format(platforms=platforms, wheels=_wheels_array(chosen))
    with open(_MANIFEST, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"updated {os.path.relpath(_MANIFEST, _HERE)}")


# ---------------------------------------------------------------------------
# Build (per-platform zips)
# ---------------------------------------------------------------------------


def _read_manifest_text(platform):
    """Return the manifest text filtered to the wheels that run on *platform*."""
    with open(_MANIFEST, encoding="utf-8") as handle:
        text = handle.read()

    keep = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('"./wheels/'):
            continue
        filename = line.strip('",').split("/")[-1]
        if wheel_matches_platform(filename, platform):
            keep.append(filename)

    if not keep:
        raise SystemExit(f"error: the manifest references no wheel that runs on {platform}.")

    # Replace every path in the wheels array with the matching subset.
    lines = text.splitlines()
    out = []
    inside_wheels = False
    wrote_open = False
    for line in lines:
        if line.strip() == "wheels = [":
            inside_wheels = True
            out.append("wheels = [")
            if keep:
                for name in keep:
                    out.append(f'    "./wheels/{name}",')
            wrote_open = True
            continue
        if inside_wheels:
            if line.strip() == "]":
                if wrote_open:
                    out.append("]")
                inside_wheels = False
                wrote_open = False
            # Drop the original entries; we already wrote the new ones.
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def build(args):
    os.makedirs(_DIST_DIR, exist_ok=True)
    if not os.path.isdir(_WHEELS_DIR):
        raise SystemExit(
            "error: no wheels/ directory. Run 'python build_extension.py download' first."
        )

    # Cache the wheel filenames per platform.
    by_platform = {}
    for platform in args.platforms:
        if platform not in _PLATFORM_TAG:
            raise SystemExit(f"error: unknown platform '{platform}'")
        matches = [
            f
            for f in os.listdir(_WHEELS_DIR)
            if f.endswith(".whl") and wheel_matches_platform(f, platform)
        ]
        if not matches:
            raise SystemExit(
                f"error: no bundleable gemmi wheel for {platform}. Run "
                "'python build_extension.py download' first."
            )
        by_platform[platform] = matches

    for platform in args.platforms:
        manifest_text = _read_manifest_text(platform)
        version_re = re.search(r'^version = "([^"]+)"', manifest_text, re.M)
        version = version_re.group(1) if version_re else "0.0.0"

        zip_name = f"crystalviz-{version}-{platform.replace('-', '_')}.zip"
        zip_path = os.path.join(_DIST_DIR, zip_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("blender_manifest.toml", manifest_text)
            for name in _PACKAGE_FILES:
                if name == "blender_manifest.toml":
                    continue
                path = os.path.join(_HERE, name)
                if os.path.isfile(path):
                    zf.write(path, name)
            for wheel in by_platform[platform]:
                zf.write(os.path.join(_WHEELS_DIR, wheel), f"wheels/{wheel}")
        print(f"built {os.path.relpath(zip_path, _HERE)}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download(args):
    os.makedirs(_WHEELS_DIR, exist_ok=True)
    if args.gemmi_version:
        version = args.gemmi_version
    else:
        version = _latest_release()
        print(f"using latest gemmi release: {version}")
    chosen = _select_wheels(version)

    for platform in TARGET_PLATFORMS:
        filename, url = chosen[platform]
        dest = os.path.join(_WHEELS_DIR, filename)
        if os.path.isfile(dest) and not args.force:
            print(f"already present: {filename}")
            continue
        print(f"downloading {filename}")
        try:
            # The wheel URL comes straight from the PyPI JSON payload (https).
            with urllib.request.urlopen(url, timeout=120) as resp:  # nosec B310
                blob = resp.read()
        except Exception as exc:
            raise SystemExit(f"error downloading {filename}: {exc}") from exc
        with open(dest, "wb") as handle:
            handle.write(blob)
        print(f"  -> {os.path.relpath(dest, _HERE)}")

    write_manifest(chosen)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="build_extension",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="fetch gemmi wheels from PyPI")
    dl.add_argument(
        "--gemmi-version",
        help="gemmi release to fetch (default: latest on PyPI)",
    )
    dl.add_argument(
        "--force",
        action="store_true",
        help="re-download wheels even if already present",
    )
    dl.set_defaults(func=download)

    bd = sub.add_parser("build", help="build per-platform extension zips")
    bd.add_argument(
        "--platforms",
        nargs="+",
        default=TARGET_PLATFORMS,
        help="which platforms to build (default: all)",
    )
    bd.set_defaults(func=build)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
