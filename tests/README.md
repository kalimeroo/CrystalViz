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

`pwsh tests/run_all.ps1` → 64/64 pass.

- `test_chemistry.py` covers the pure-Python perception module (`chemistry.py`,
  standard library only — it also runs under plain `python tests/test_chemistry.py`,
  no Blender or gemmi). It pins Kekulé repair on synthetic benzene, pyridine,
  pyrrole (nitrogen lone pair, no N=C), naphthalene (two aromatic rings),
  indolizine (a bridgehead nitrogen donates its lone pair to neither ring, so the
  fused 5+6 system is drawn Kekulé) and indole (a nitrogen in a single ring still
  donates, so both fused rings stay aromatic), cyclohexane/piperidine (nothing
  aromatic), a hydrogen-less benzene ring and a hydrogen-less graphene patch
  (fusion bonds must not veto a ring, the bounded searches must stay bounded,
  and the dangling carbons at the patch edge — which the Kekulé structure
  doubles — do not veto their ring; a real exocyclic =CH2 still does, while a
  hydrogen-less p-quinodimethane is the documented exception), acetylene/acetonitrile triples, allene, formaldehyde, the "long
  contacts are never promoted" gate, unknown elements staying single-bonded,
  orientation independence of the edge keys, the N-oxide rule (a three-connected
  N carrying one terminal O is four-valent: its dative N-O bond stays single and
  its fourth order goes into the ring, so pyridine N-oxide and pyrazine
  N,N′-dioxide become aromatic; a nitro group is drawn N(=O)₂ with two
  genuine doubles, a two-connected nitroso N=O is untouched, and a
  hydroxylamine invents nothing), and the ring enumerator (4-rings
  skipped, hydrogens/metals excluded). The aromatic records also pin the
  depiction split: pyrrole reports 2 lone-pair electrons of its sextet, benzene
  and pyridine report 0, a ferrocene-shaped Cp + Fe fragment is a 6 π haptic
  ligand with no double bond, a Cp–C≡C–anthracene fragment keeps its alkyne
  triple (the Ferrocene.cif allene failure mode), and an η6-coordinated quinone
  is still vetoed by its exocyclic C=O. An η8-COT + Sm fragment is the 10 π
  haptic ligand (an eight-ring is enumerated), a bare seven-ring carbon with no
  double is aromatic only when its short bonds make it the tropylium empty-p
  centre, and a plain eight-ring carbon without the metal stays non-aromatic.
  The sulfur rules are pinned on synthetic
  six-rings: a thiopyrylium sulfur takes one ring S=C double (three alternating
  doubles, no lone-pair share), a thiabenzene 1-oxide sulfur keeps its terminal
  S=O double and still counts as a conjugated ring atom, while thiophene (a
  five-ring, where an all-double-bond Kekulé has an odd number of orders) keeps
  its lone pair and a saturated thiacyclohexane stays single-bonded. A sulfone
  sulfur (two terminal oxygens) is not walked as a ring atom and is hexavalent:
  its two spare orders become genuine S=O doubles. A terminal oxygen keeps its
  oxide role even under cationic coordination — a metal salt contacts every S=O
  oxygen (graph degree two: covalent S-O plus an ionic M-O), so the metal bond
  must not hide the sulfone (the metal-coordinated sulfone test pins exactly
  that: K/Ca on both S=O oxygens still yields two S=O doubles, S-C single, and
  the M-O contacts single). The same gates are
  pinned for the heavier main-group elements: selenopyrylium/telluropyrylium
  take one ring C=Se/C=Te double, selenophene/tellurophene keep their lone pair,
  a selenoxide keeps its Se=O and still counts as a ring atom, and a saturated
  selenane stays single-bonded; arsabenzene draws an As=C (group 15 ladder),
  arsole/stibole keep their lone pair, and a B3N3H6 borazine / B3O3R3 boroxine is
  aromatic from the N/O lone pairs even though a three-connected boron has no
  double bond and contributes 0 π electrons (a saturated boracyclohexane is
  still rejected, so boron alone does not make a ring conjugated; an h-BN patch
  keeps the shared nitrogen donating, and its rings report `kekule` false
  because no B=N exists to draw). It finishes on
  the real fixture: the
  `CIF/multiple_bonds.cif` `_geom_bond` loop must yield exactly the reference
  Kekulé structure (7 C=C + 3 C=O, no triple) and exactly one aromatic ring (the
  benzene; the annulated-indolizine five-ring is a bridgehead case), with every
  atom at a valid valence. It also pins the delocalised-oxo detection
  (`delocalised_oxo_edges`): every terminal X-O bond of a
  carboxylate/carbonate, nitrate, phosphate/phosphonate and sulfate/sulfonate
  is marked — including a metal-coordinated oxygen and the Se/Te analogues —
  while nitro, sulfone, ketone/aldehyde/ester/amide, a carboxylic or nitric acid
  whose hydrogen is present, a phenoxide, an N-oxide and a metal carbonyl are
  left alone.

- `test_chemdraw_delocalised.py` is the end-to-end counterpart for the 1.5
  stroke: it calls `_chemdraw_segments()` on a synthetic carboxylate and checks
  that each C-O edge becomes one full-length order-2 stroke plus five short
  collinear dashes on the *inside* of the O-C-O angle (the dashed stroke sits
  toward the other oxygen and the solid stroke outside, the hand-drawn
  nitrite/carboxylate convention; twice the dash density of the first
  implementation), 12 strokes total, all order 2, that a ketone C=O is still
  two full-length strokes, and that the `_chemdraw_delocalised_edges()`
  adapter marks a carboxylate and a phosphate while leaving nitro, sulfone and
  ketone localized.

- `test_chemdraw_depiction.py` is the end-to-end counterpart: it imports
  `multiple_bonds.cif`, builds the bond set with ChemDraw on and asserts the
  drawn stroke counts (4 non-aromatic C=C doubles, 3 C=O doubles, 14 single C-C
  plus the 3 ring doubles suppressed inside the aromatic benzene ring), no
  invented triple bond, and one grouped aromatic object holding exactly one ring.
  The previous length-cutoff heuristic found only 3 C=C + 3 C=O and drew the
  aromatic rings as Kekulé strokes.

- `test_chemdraw_azole.py` is the depiction counterpart for five-membered rings.
  On `aromatic.cif` (COD 4025500: a methyl+phenyl 1H-1,2,4-triazole) it pins
  that the 1,2,4-triazole keeps its two ring C=N doubles as strokes (4 C|N
  double strokes) and that the phenyl is the only torus left (48 segments, its
  three doubles still suppressed); the ring reports 2 lone-pair electrons, so
  IUPAC GR-6.3's "one unique arrangement of alternating bonds" rule applies.
  The same perception is checked on hand-built fragments: N-methylpyrrole → two
  doubles and no torus, benzene → torus, and a Cp + Fe fragment → no double
  bond but a torus (the 6 π haptic ligand of GR-1.7.2). It closes on
  `Ferrocene.cif` (9-(ferrocenylethynyl)anthracene), which must draw five tori
  (2 × η5-Cp + the three anthracene rings, 240 segments), the alkyne as a
  C|C triple (3 strokes) and no C|C double at all: the Cp carbon that carries
  the alkyne is four-coordinate, so the alkyne takes both spare orders itself
  instead of borrowing one from the arene (the old allene depiction drew
  C11=C12=C13 and cost two of the three rings). Two COD fixtures close the
  heteroatom case: `Pyridine_N_oxide.cif` (2105865) must keep four tori (four
  molecules, 192 segments) with its dative N-O drawn single and no N=O at all,
  and `Nitrobenzene.cif` (5000155) must keep its four tori while drawing both
  N-O bonds as genuine doubles, N(=O)₂ (16 strokes, no single N-O left): the
  delocalised O-N-O is depicted as two localised double bonds.

- `test_chemdraw_sulfur.py` is the end-to-end counterpart for the two CSD
  fixtures. `S1.cif` (ATBOXM, thiabenzene 1-oxide) must draw the S=O as a
  genuine double (2 strokes, no single S-O left) and the sulfur ring as one
  aromatic torus (48 segments): its boat is 0.127 Å RMS out of plane, under the
  `_AROMATIC_PLANARITY_MAX` gate, so it reads as flat and keeps the torus, and
  the ring S=C and both ring C=C doubles are suppressed inside it (only the
  S=O and the exocyclic ketone C=O stay drawn). `S2.cif` (BIZTAF,
  thiopyrylium) must draw one aromatic object holding four tori (the sulfur
  ring + three phenyls, 192 segments), with the S=C suppressed inside the torus
  and both S-C bonds demoted to singles.

- `test_chemdraw_heterocycles.py` is the end-to-end counterpart for the heavy
  main-group fixtures. The two chalcogenylium salts (`Selenochromenylium.cif`,
  COD 7243860, and `Tellurochromenylium.cif`, COD 7243859) must draw their flat
  fused bicycle as two tori (96 segments) with both ring C-Se/C-Te bonds demoted
  to singles; `Selenophene.cif` (1553554) and `Tellurophene.cif` (1558536) must
  keep the five-ring chalcogen single (its lone pair, so Kekulé) with its two
  C=C doubles, and still give the two phenyls their tori. `Arsabenzene.cif`
  (1573224, arsabenzene fused to two thiophenes) must draw the arsenic ring as
  a torus (48 segments) with both As-C bonds single and one localized double in
  each thiophene, while `Arsole.cif` (1555319, 1,2,5-triphenylarsole) keeps the
  arsenic lone pair (three As-C singles, two C=C strokes) and its three phenyl
  tori (144 segments). `Diazastibolene.cif` (7111438) pins the negative case:
  the antimony ring is found but its 8 π count is not aromatic, so there is no
  torus and the single C=C and both Sb-N bonds are drawn as they are.
  `Borazine.cif` (2216154) pins the boron rule: the two B3N3 rings are aromatic
  in the chemistry API (the sextet is the three nitrogen lone pairs), and with
  no B=N double bond there is no alternating arrangement to draw, so both rings
  keep their delocalisation torus (one object, 96 segments) while all twelve
  B-N bonds are drawn single.

- `test_chemdraw_ring8.py` pins the η8-cyclooctatetraenyl ligand (ABIDUK, CSD;
  the CIF is local-only and the test skips when it is absent). The eight-ring
  is enumerated (ring sizes 5-8) and the haptic path counts it as 10 π, so the
  Cp* and the COT ring share one aromatic object with two 48-segment tori
  (96 edges) and both rings are recorded on it.

- `test_chemdraw_bn.py` pins the grown h-BN lattice. `BN.cif` (1010602) imports
  as four atoms and no torus; growing the sheet shell by shell must then give
  every closed hexagon a torus — 4 at grow ×2, 8 at ×3, 20 at ×4, 28 at ×5 and
  48 at ×6 (148 atoms) — each a full 48-segment six-ring. Every nitrogen belongs
  to three rings, so the indolizine rule would withhold all lone pairs; a ring
  containing boron is an aromatic network and keeps donating.

- `test_chemdraw_graphite.py` pins the grown-lattice case. `Graphite.cif`
  imports as a single C-C bond (two atoms, no torus); growing the sheet shell
  by shell must then give every closed hexagon a torus — 2 at grow ×2, 4 at ×3,
  10 at ×4 and 14 at ×5 (54 atoms) — and every torus is a full 48-segment
  six-ring. The patch edge ends in dangling carbons that the Kekulé structure
  doubles; that is a truncated lattice, not an exocyclic C=CH2, so it must not
  veto the ring. The last check keeps the exemption honest: `multiple_bonds.cif`
  still draws exactly one torus, its exocyclic C=O veto intact.

- `test_gemmi_worker.py` (bpy-free, plain gemmi) pins the P1 site and ASU
  instance counts of the example CIFs, the grow shell, the ADP conversion, and
  the dummy-site skip: `Cyanoferrocene.cif` (2005233) carries two `Cg` ring
  centroids with `_atom_site_calc_flag` = `dum` and no `_atom_site_type_symbol`,
  so the label fallback used to turn them into carbon and bond them to every Cp
  carbon. The loader now reads the calc flag from the CIF block (gemmi exposes
  no `calc_flag` on its sites) and drops those sites plus any explicit zero
  occupancy: 88 P1 sites / 22 ASU atoms, 44 C + 4 Fe + 4 N + 36 H, no `Cg`
  label. A CIF without the tag (Nitrobenzene) skips nothing.

- `test_grow_fill_noadd_bonds.py` pins that Fill ASU → Grow bonds the initial
  atoms even when grow adds no shell — small finite molecules (Ferrocene.cif,
  multiple_bonds.cif) do not pack again within the connectivity window, and grow used to
  return early at "no additional atoms matched" without calling the bond
  builder. It now builds the full set when starting bond-less after a fill; a
  second atom-less grow preserves those bonds.

- `test_bond_grow_chemdraw_union.py` pins the grow-append chemdraw fix: when
  bonds already exist, every grow must *append*, and the bond-order/aromatic
  perception now runs over the union of the bonds on screen plus the new ones
  instead of the freshly grown shell alone. With Pyrazine_dioxide.cif (half a
  molecule per asymmetric unit on an inversion centre) the append draw no
  longer promotes the lone half-molecule C-C to a spurious triple, and the
  first grow that closes the ring renders the aromatic torus (one object, 48
  edges) with the ring C-C bonds demoted to single strokes and the N-O bond
  kept single — no duplicate groups, no `.00x` names. A final full rebuild
  with the same windows needs no bond pair the appended grows lacked.

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
- `test_bond_grow_disorder_twin_aromatic.py` pins the disorder-twin growth fix.
  `ABILIL01.cif` (CSD CCDC 2530362; a local fixture like S1.cif/S2.cif)
  deposits *both* halves of an inversion-disordered molecule, so the P1
  expansion holds 848 same-element twin pairs, some separated by a single unit
  of the 5-decimal fractionals (~1.6e-4 Å, far beyond the old 1e-5 Å rounding).
  Growth used to pull the twin hydrogen of an already-displayed ring atom into
  the scene, the ring carbon gained a fourth non-metal neighbour, and the
  straddling benzo rings were no longer enumerated: Fill → Asymmetric Unit then
  Grow → ChemDraw drew two phenyl rings with Kekulé doubles instead of aromatic
  tori. `_canonical_base_indices` now merges twins within
  `_DUPLICATE_POSITION_TOLERANCE` (0.01 Å), grow adds only the clean atoms, and
  both the fill and the grown structures draw their 18 C6 rings as tori (the 6
  carbazole pyrrole rings are aromatic too — 24 aromatic rings in the chemistry
  API — but keep a unique alternating Kekulé arrangement, so GR-6.3 shows their
  doubles rather than a torus). `tests/test_gemmi_worker.py` carries the
  bpy-free counterpart: the canonical merge turns the 1992 P1 sites into 1144
  representatives (848 merged), each H36/H36B twin pair shares one
  representative, and a grown shell never materialises a same-element pair
  closer than 0.05 Å (the ASU being clean at ≥ 0.2 Å).
- `test_bond_duplicate_sites.py` covers the Blender-side counterpart: the bond
  builder bonds a disordered site once (no zero-length bond, no doubled bonds
  to its neighbours) for exact and near duplicates, leaves coincident
  different-element disorder unbonded, and does the same on the grow/append
  path.
- `test_import_disorder_primes.py` pins the load-time omission of alternate
  disorder components (`_disorder_minor_labels`). It embeds the AKUQAB
  thiophene fragment (the CSD export is gitignored) with its primed second
  orientation, imports it, fills the asymmetric unit and bonds it: only the
  eleven unprimed atoms survive (16 primed sites are omitted, here eight in the
  embedded subset), the shared carbon no longer gets a fourth neighbour, and
  the chemistry API reports the five-ring as aromatic with its two Kekulé
  doubles. With both components kept the ring is not perceived at all, which
  is the `S_ring5_no_aromatic` symptom. `tests/test_gemmi_worker.py` carries the
  bpy-free rule tests: primes dropped without occupancy, the more occupied
  component winning (even when primed), equal occupancies keeping the unprimed
  component, fixed hydrogen occupancies abstaining, lone primed labels kept,
  double-prime third components dropped, and letter suffixes (`H8A`/`H8B`) left
  alone. It also pins the formal `_atom_site_disorder_group` rule
  (`_disorder_group_labels`): group 2 is dropped per
  `_atom_site_disorder_assembly` even when its refined occupancy is the higher
  one (0.517 against 0.483), an assembly numbered only 2 stays put, a tag with
  no sibling removes nothing, and the local-only `CIF/disorder.cif` (four
  assemblies A-D, `C1A`/`C1B` ...) reports its 100 group-2 sites omitted while
  `keep_disorder=True` keeps both components (that check skips when the file is
  absent).
- `test_import_symmetry_disorder.py` pins the load-time resolver for
  space-group-generated disorder (`_resolve_symmetry_disorder`). The driving
  case `CIF/Ca.cif` (local-only) sits two groups — a *tert*-butyl and a methyl
  — on a mirror plane, with every atom written once at half occupancy, so the
  P1 expansion holds both orientations at once. The resolver keeps exactly one
  chemically valid orientation per anchored cluster (parents-first, identity-max
  solution, valence and bond-length gates) on a synthetic mirror-t-Bu fixture,
  promotes it to full occupancy and relocates the asymmetric unit onto it, and
  leaves half-occupied substitutional solvent on general positions untouched
  (a K/water pair stays four 0.5 images). The Blender suite drives import →
  Fill → bond → ChemDraw: one tetrahedral t-Bu (3 methyls, 9 H, no clash), the
  benzene ring still aromatic, solvent present; `keep_disorder=True` keeps both
  components as site `component` keys with no cross-component methyl–methyl
  bond. `tests/test_gemmi_worker.py` carries the bpy-free rule tests: counts,
  occupancy promotion, ASU relocation, keep-all components, the
  general-position-anchor rejection (the false positive that used to swallow
  the Ca.cif solvent region), and both suites share the fixture in
  `tests/_symmetry_disorder_fixture.py`.
- `test_view_projection.py` pins the View sub-panel's `cgn.view_along`
  operator: it points the 3D viewport down a crystallographic direction and
  snaps it to orthographic projection (always). On `multiple_bonds.cif` the
  view forward (-Z local) is asserted to be parallel to the cell **a**/**b**/**c**
  axes and the in-plane roll is pinned to the conventional layout (along **a**,
  **b** horizontal with **c** vertical; along **b**, **a** horizontal with
  **c** vertical; along **c**, **a** horizontal with **b** vertical), the [h k
  l] view-along projection reproduces the same directions for [1 0 0],
  [0 1 0] and [0 0 1] and the [1 1 1] = a+b+c combination, the (h k l) plane
  projection (the same three values read as Miller indices, via the panel's
  View dropdown) looks along the reciprocal normal h a* + k b* + l c* (seen
  face on), and its look sense follows the index signs — positive indices
  view from the origin along the normal, negated (-h -k -l) indices the
  opposite way (asserted as a vertical mirror; the horizontal axis stays
  exact in both), each
  projection re-centres the viewport on the structure, and zero [0 0 0]
  direction / (0 0 0) plane are rejected without moving the view.

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
