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

`pwsh tests/run_all.ps1` → 42/42 pass.

- `test_selection_highlight.py` pins the growth-selection highlight fix: the
  selection modal keeps the live set in memory and `_apply_selection` now calls
  `_refresh_growth_selection` (visual-only, no re-serialization) instead of
  `_update_growth_selection` on every click/box release. The latter reuses the
  single `growth_selection` object in place (`clear_geometry` + `from_pydata`)
  rather than tearing down and recreating the object/material/modifier graph
  each update, and persists exactly once, at the modal boundary in `_finish`.
  The test verifies one reused object across updates, correct vertex counts,
  `_refresh_growth_selection` never writes `cgn_selected_indices`, empty
  selection removes the object and persists `[]`, and recreation after a clear
  is still a single object.

- `test_bond_per_pair_reach.py` pins the per-pair early rejection in the bond
  builder (`_bond_objects`): a centre now rejects far candidates against its
  own element's widest enabled window (`reach[element]`) instead of the largest
  window in the whole table, so one permissive pair no longer inflates every
  centre's distance tests.  On several arrangements (disparate reaches, a
  permissive pair for one element, an element with no enabled pair, asymmetric
  reach, the grow/append path, and a non-zero `min_distance`) the built bond set
  must be byte-identical to an exhaustive O(N^2) reference that tests each pair
  against its own window.

- `test_polyhedron_search_parity.py` pins the shared coordination search
  (`_PolyhedronSearch`): on every example CIF it must return exactly the ligand
  lists of the old per-centre brute force (so the gemmi `NeighborSearch` index
  can never drop a ligand), plus a synthetic small cell where several periodic
  images of one atom fall inside the window, the pool-scan fallback, and the
  memo/window-change contract the panel draw relies on.

- `test_optionb_multi.py` used to run on `aromatic.cif`, whose imported ASU is
  already the complete molecule, so no connectivity pair could grow atoms and
  the preservation assertions were vacuous. It now runs on `Ag2Cr2O7.cif`
  (11 → 20 atoms, pairs `AG|O` / `CR|O`), so both Option-B checks are real.
- `test_grow_duplicate_sites.py` builds a synthetic P1 table with same-element
  duplicate sites 2e-6 Å apart (above the 1e-12 self filter, below the 1e-5
  canonical grouping resolution) and asserts that growth never walks into the
  duplicate copy, with `min_distance` at both 0 and the 0.1 default.
- `test_bond_duplicate_sites.py` covers the Blender-side counterpart: the bond
  builder bonds a disordered site once (no zero-length bond, no doubled bonds
  to its neighbours) for exact and near duplicates, leaves coincident
  different-element disorder unbonded, and does the same on the grow/append
  path.

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
