# CrystalViz tests

Headless Blender checks for the add-on. They import `CrystalViz`, register it,
and drive the `cgn.*` operators the same way the UI does.

CI runs only the pure-Python quality gates (ruff, formatting, `compileall`,
bandit); there is no headless Blender in CI. Run these suites locally against
the Blender version you ship for (5.2+ / CPython 3.13).

## Layout

```
tests/
  _env.py                      shared bootstrap (gemmi + the add-on parent on sys.path)
  run_all.ps1                  runs every test_*.py and prints a PASS/FAIL summary
  test_*.py                    the suites run by default
  _logs/                       per-test logs written by run_all.ps1 (gitignored)
  _gemmi/                      gemmi unpacked from wheels/ on first run (gitignored)
```

Local-only groups (`tools/`, `legacy_baked_polyhedra/`, `external_cctbx/`) are
gitignored and not published; see [Archived groups](#archived-groups).

## Running

```powershell
# whole suite
pwsh tests/run_all.ps1

# one suite
pwsh tests/run_all.ps1 -Only "*polyhedra*"

# a single test, directly
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background --factory-startup --python tests\test_gn_polyhedra.py
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

`pwsh tests/run_all.ps1` → 51/51 pass.

- `test_chemistry.py` covers the pure-Python perception module (`chemistry.py`,
  standard library only — it also runs under plain `python tests/test_chemistry.py`,
  no Blender or gemmi). It pins Kekulé repair on synthetic benzene, pyridine,
  pyrrole (nitrogen lone pair, no N=C), naphthalene (two aromatic rings),
  indolizine (a bridgehead nitrogen donates its lone pair to neither ring, so the
  fused 5+6 system is drawn Kekulé) and indole (a nitrogen in a single ring still
  donates, so both fused rings stay aromatic), cyclohexane/piperidine (nothing
  aromatic), a hydrogen-less benzene ring and a hydrogen-less graphene patch
  (fusion bonds must not veto a ring, and the bounded searches must stay
  bounded), acetylene/acetonitrile triples, allene, formaldehyde, the "long
  contacts are never promoted" gate, unknown elements staying single-bonded,
  orientation independence of the edge keys, and the ring enumerator (4-rings
  skipped, hydrogens/metals excluded). It finishes on the real fixture: the
  `CIF/multiple_bonds.cif` `_geom_bond` loop must yield exactly the reference
  Kekulé structure (7 C=C + 3 C=O, no triple) and exactly one aromatic ring (the
  benzene; the annulated-indolizine five-ring is a bridgehead case), with every
  atom at a valid valence.

- `test_chemdraw_depiction.py` is the end-to-end counterpart: it imports
  `multiple_bonds.cif`, builds the bond set with ChemDraw on and asserts the
  drawn stroke counts (4 non-aromatic C=C doubles, 3 C=O doubles, 14 single C-C
  plus the 3 ring doubles suppressed inside the aromatic benzene ring), no
  invented triple bond, and one grouped aromatic object holding exactly one ring.
  The previous length-cutoff heuristic found only 3 C=C + 3 C=O and drew the
  aromatic rings as Kekulé strokes.

- `test_grow_fill_noadd_bonds.py` pins that Fill ASU → Grow bonds the initial
  atoms even when grow adds no shell — small finite molecules (Ferrocene.cif,
  multiple_bonds.cif) do not pack again within the connectivity window, and grow used to
  return early at "no additional atoms matched" without calling the bond
  builder. It now builds the full set when starting bond-less after a fill; a
  second atom-less grow preserves those bonds.

- `test_reset_default_pairs.py` pins that **Reset Default** restores the exact
  import-time connectivity windows. The reset recomputed defaults from covalent
  radii alone, but import derives them from the CIF's formal charges plus Shannon
  ionic radii for ionically bonded pairs (e.g. Cu-S in Chalco: (1.54, 3.08) vs
  the covalent (1.42, 2.77)). The import now persists that chemistry on the
  structure root and the reset reads it, with a covalent fallback for roots that
  predate it. It also pins that reset keeps every element pair after atoms are
  deleted (only Ag left in Ag2Cr2O7 — the list is rebuilt from the persistent P1
  site table, not the displayed atoms), and that **Update Bonds** rebuilds the
  displayed atoms' bonds from the current windows. Finally it pins the
  metal-metal cap: a compound without declared oxidation numbers (Brookite)
  caps Ti-Ti at 0.85 x the covalent sum, while elemental copper keeps the wide
  Cu-Cu window.

- `test_bond_clear_on_rebuild.py` pins that a full `_bond_objects` rebuild
  (`append_only=False`) drops the previous bond set when nothing can be
  bonded — the reported sequence Fill → Grow → Delete (box-select every atom but
  one) leaves no bond object behind, and the same early-return cleanup applies
  to an empty/single-atom structure, every connectivity pair disabled, a zero
  max-distance window, and the space-filling model. Append-only growth still
  preserves existing bonds.

- `test_import_no_bonds.py` pins that importing a CIF behaves like Fill →
  Asymmetric Unit: atoms, unit cell and the default connectivity windows are
  stored, but no bond objects (nor chemdraw tori) are created. Re-importing
  replaces a bonded structure with a bond-less one (no stray bond object
  survives), and bonds come back through the existing actions — grow (full set,
  initial–initial bonds included), Create Polyhedra and a rendering-model
  switch.

- `test_fill_no_bonds.py` pins that the three fill modes (Asymmetric Unit,
  Unit-Cell, Pack Unit-cells) display atoms only: every pre-existing bond object
  is dropped, chemdraw aromatic tori included, no bond is rebuilt, and the atom
  counts/vertices stay consistent. It then proves bonds come back through the
  existing actions — the `_bond_objects` call used by polyhedra/delete and a
  rendering-model switch — and that a second fill clears them again. Finally it
  pins that a grow right after a fill builds the bonds among the _initial_
  atoms as well as those to the new atoms (the grow path falls back from
  append-only to a full build when the structure has no bonds to preserve).

- `test_polyhedra_bond_visibility.py` pins the "Hide Bonds Inside Polyhedra"
  option (scene boolean `cgn_hide_polyhedron_bonds`): the centre→ligand bonds
  are dropped from the per-pair bond objects and come back unticked. It also
  covers the mask edge cases — a "Create from Selected" subset hides only its
  own spokes, clearing polyhedra restores every bond, polyhedra created while
  the option is already on are masked immediately, grown Ti images without a
  polyhedron keep their bonds, and deleting a ligand re-derives the mask from
  the rebuilt polyhedra (the delete path now remaps polyhedra _before_ the bond
  rebuild so stale centre indices can never hide the wrong bonds).

- `test_polyhedra_ligand_delete.py` pins that deleting a ligand atom sitting on a
  polyhedron vertex removes that image's polyhedron — the same outcome as
  deleting the central atom — while unaffected polyhedra survive. It drives the
  delete data path (`_remap_polyhedra` with `removed_quads`) against Brookite
  Ti-centred polyhedra, covers the centre-deletion parity case, and exercises a
  grown structure so a ligand of a _translated_ image is matched per periodic
  copy (the pre-fix check only inspected the home-cell representative).

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
  centre's distance tests. On several arrangements (disparate reaches, a
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

- `test_optionb_multi.py` used to run on an earlier aromatic fixture whose
  imported ASU was already the complete molecule, so no connectivity pair could
  grow atoms and the preservation assertions were vacuous. It now runs on
  `Ag2Cr2O7.cif`
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

Local-only material: gitignored, kept on the author's disk, not published.

- `legacy_baked_polyhedra/` — exercises the pre-GeoNodes baked design
  (`cgn_role == "polyhedron_instance"` empties, prototype collections,
  per-image face objects). It fails on the GN branch by design and is kept for
  historical reference; `test_gn_polyhedra*.py` is the current coverage.
- `external_cctbx/` — targets `cctbx_geometry_nodes_addon`; without that add-on
  installed the tests abort at add-on enable.
- `tools/` — probes/benchmarks used during development, several with
  machine-specific paths (e.g. an extracted `gemmi` or a pre-GN reference
  checkout). They are references, not part of the suite.
