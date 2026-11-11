# CrystalViz tests

Headless Blender checks for the add-on. They import `CrystalViz`, register it,
and drive the `cgn.*` operators the same way the UI does.

There is no CI: run them locally against the Blender version you ship for
(5.2+ / CPython 3.13).

## Layout

```
tests/
  _env.py                      shared bootstrap (gemmi + the add-on parent on sys.path)
  run_all.ps1                  runs every test_*.py and prints a PASS/FAIL summary
  test_*.py                    the suites run by default
  tools/                       development-time probes/benchmarks (not run automatically)
  legacy_baked_polyhedra/      pre-GeoNodes baked-polyhedra tests, superseded by test_gn_polyhedra*.py
  external_cctbx/              tests for the separate cctbx_geometry_nodes_addon (not CrystalViz)
  _logs/                       per-test logs written by run_all.ps1 (gitignored)
  _gemmi/                      gemmi unpacked from wheels/ on first run (gitignored)
```

## Running

```powershell
# whole suite
pwsh tests/run_all.ps1

# one suite
pwsh tests/run_all.ps1 -Only "*polyhedra*"

# a single test, directly
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background --factory-startup --python tests\test_gn_polyhedra.py

# archived groups, run by path
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background --factory-startup --python tests\legacy_baked_polyhedra\test_polyhedra_fixes.py
```

`run_all.ps1` accepts `-Blender <path>` (or set `$env:BLENDER`) and
`-Timeout <seconds>`.

## gemmi

`_env.py` finds gemmi in this order:

1. `$env:CRYSTALVIZ_GEMMI_DIR` — a directory containing the `gemmi` package.
2. `tests/_gemmi/` — drop an extracted gemmi here for a local run.
3. `%TEMP%\crystalviz_gemmi/` — the legacy scratch extraction.
4. `wheels/gemmi-*.whl` — unpacked once into `tests/_gemmi/`.

So `python build_extension.py download` (which populates `wheels/`) is enough
for a first run.

## Conventions

- Each test prints a `RESULT: ALL PASS` / `RESULT: FAIL` line (or is a
  `unittest` module); `run_all.ps1` keys off that, falling back to the exit
  code.
- New tests belong in `tests/` and should start with:
  ```python
  import sys
  sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
  import _env  # noqa: F401
  import CrystalViz
  CrystalViz.register()
  ```
- Import CIFs from `_env.CIF_DIR` (the repo `CIF/` folder) rather than
  hard-coding an absolute path.

## Current status

`pwsh tests/run_all.ps1` → 36/36 pass.

- `test_optionb_multi.py` used to run on `aromatic.cif`, whose imported ASU is
  already the complete molecule, so no connectivity pair could grow atoms and
  the preservation assertions were vacuous. It now runs on `Ag2Cr2O7.cif`
  (11 → 20 atoms, pairs `AG|O` / `CR|O`), so both Option-B checks are real.

## Ordered suites

`test_gn_persist_build.py` → `test_gn_persist_check.py` → `test_gn_persist_export.py`
form one chain: the build saves a `.blend` in the OS temp directory, and the
other two reopen it. Alphabetical order (what `run_all.ps1` uses) gives the
right sequence, so a `-Only` filter must select all three, e.g.
`-Only "*persist*"`.

## Archived groups

- `legacy_baked_polyhedra/` — exercises the pre-GeoNodes baked design
  (`cgn_role == "polyhedron_instance"` empties, prototype collections,
  per-image face objects). It fails on the GN branch by design and is kept for
  historical reference; `test_gn_polyhedra*.py` is the current coverage.
- `external_cctbx/` — targets `cctbx_geometry_nodes_addon`; without that add-on
  installed the tests abort at add-on enable.
- `tools/` — probes/benchmarks used during development, several with
  machine-specific paths (e.g. an extracted `gemmi` or a pre-GN reference
  checkout). They are references, not part of the suite.
