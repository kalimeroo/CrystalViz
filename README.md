# CrystalViz

This Blender extension allows the importation of Crystallographic Information
Files (CIF) and provides tools to display molecular strucures in Blender.
It uses the gemmi as a backend (https://github.com/project-gemmi/gemmi).

gemmi is a compiled C extension, so it is **bundled with the add-on** as a
per-platform wheel. **There is nothing to install** — no pip, no terminal, no
`--ignore-installed` hacks. Just install the extension archive for your
operating system.

## Installing the add-on

This add-on targets **Blender 5.2** and is distributed as a modern Blender
**Extension** (`*.zip` built with `build_extension.py`). Install it from
**Edit > Preferences > Get Extensions > (dropdown) > Install from Disk...**,
choose the `.zip` that matches your platform, then enable it:

| Platform      | Archive                        |
| ------------- | ------------------------------ |
| Windows x64   | `crystalviz-*-windows_x64.zip` |
| Linux x64     | `crystalviz-*-linux_x64.zip`   |
| macOS (Apple) | `crystalviz-*-macos_arm64.zip` |
| macOS (Intel) | `crystalviz-*-macos_x64.zip`   |

Each archive contains the add-on **and** the correct `gemmi` wheel for that
platform. Since gemmi is a per-platform wheel, use the archive for your own OS
and CPU architecture.

The **Add-on Preferences** and the **About** sub-panel display the detected,
bundled gemmi version. If gemmi cannot be loaded it means the extension was
installed for the wrong platform or an unsupported Blender version, and the
preferences explain how to fix it.

## Building from source

`build_extension.py` (Python 3.8+, standard library only) both downloads the
gemmi wheels and packs the per-platform extension archives:

```bash
python build_extension.py download --gemmi-version 0.7.5   # fetch wheels into wheels/
python build_extension.py build                            # build dist/*.zip per platform
python build_extension.py build --platforms macos-arm64    # single platform
```

- `download` fetches the CPython-3.13 gemmi wheel for each target platform from
  PyPI (unmodified) into `wheels/` and keeps the `wheels` array in
  `blender_manifest.toml` in sync. Omit `--gemmi-version` to use the latest
  release.
- `build` produces one installable `.zip` per platform in `dist/`, each with
  only the wheel that runs on that platform — the same output Blender's own
  `blender --command extension build --split-platforms` produces.

You can also run Blender's own build from this directory to validate the
manifest:

```bash
blender --command extension build --split-platforms
blender --command extension validate
```

The add-on imports CIF asymmetric-unit atoms and renders them through Geometry
Nodes. In the default **Ball & Stick** model, atom spheres use a quarter of the
element's van der Waals radius and bond tubes use 0.1× the mean van der Waals
radius of the bonded pair, both with smooth shading. On import the
scene is set to centimeters with **1 Å = 1 cm**: 1 Blender unit = 10 Å = 10 cm
(Unit Scale `0.1`). A 20 Å cell therefore reads 20 cm and spans 2 Blender units.

Each unordered element pair gets an independent connectivity checkbox and
inclusive minimum/maximum distance in Å, inside the collapsible **Connectivity**
sub-panel. After changing those settings, use **Rebuild Bonds**. Initial maximum
distances are inferred from the cctbx covalent radii plus 0.4 Å.

Atoms use the JMOL CPK colour scheme (the Science Notes table): colours are
stored as sRGB hex and converted to linear values so Blender displays the
intended hue. Elements missing from the table fall back to Jmol's pink.

The **CrystalViz** sidebar tab is split into collapsible sections: the main panel
shows the import button, space group, unit-cell parameters on a three-row,
two-column grid (a, b, c on the left; α, β, γ on the right) and atom count.
Cell parameters retain CIF standard uncertainties, for example `7.57360(10)`,
and exact integer angles display without decimal places;
**Build** holds Fill, Grow, Delete and a **Show Hydrogens** checkbox that hides
or shows hydrogen atoms together with their bonds; **Connectivity**,
**Unit Cell**, **Polyhedra**, **Atom Model & Style** and **About** are collapsed by default.
The same build actions are available from the **CrystalViz** menu in the 3D Viewport
header, and CIF import is also registered under **File > Import**.

The **Atom Model & Style** sub-panel switches the rendering model:

- **Ball & Stick** — quarter van der Waals spheres with 0.1× mean-vdW bonds.
- **ORTEP** — anisotropic displacement ellipsoids built from the CIF ADP
  tensors, shown at the 50 % probability level by default. The **Probability**
  slider below the model dropdown rescales the 3D Gaussian probability surface
  using the chi-square distribution with three degrees of freedom, without
  rebuilding the atom meshes. Cartesian ADP tensors are converted from the CIF
  crystallographic basis and rotated with each symmetry image. Atoms with only
  an isotropic Uiso value use a spherical ellipsoid; atoms without displacement
  parameters fall back to the Ball & Stick sphere. The probability slider
  affects non-hydrogen atoms; hydrogen atoms are displayed as spheres and use
  the adjustable **Hydrogen Size** control. ORTEP ellipsoids use a cutaway with
  colored inner faces and three principal-plane ellipses; their colors,
  roughness and ellipse thickness are adjustable. The removed octant tracks the
  active viewport direction and faces the render camera. Bonds are drawn thin.
- **Space-filling (Use VdW radii)** — full van der Waals spheres with bonds
  hidden.
- **Wire-sticks** — atoms and bonds use the same adjustable radius, set in Å
  with **Atom & Bond Radius**.

The expandable **Bond Style** panel lets bonds use one shared color or color
each half with the corresponding bonded atom's element color. Single-color
bonds have an adjustable color. Its **ChemDraw** checkbox adds parallel strokes
for inferred double/triple bonds, with slightly slimmer cylinders for multiple
bonds. Planar, approximately regular five- and six-membered aromatic rings
with conjugation and sufficiently equalized normalized bond lengths are shown
with an inner torus even when individual bond lengths suggest one Kekule double
bond; localized cycles instead show centered multiple-bond strokes in the plane
of their neighboring atoms. Single-bond radius is independent of the
**Multiple Bond Radius** control, and **Bond Spacing** adjusts the distance
between parallel strokes. Torus size, width, and color are adjustable.
Multiple-bond inference
compares observed lengths with the CIF covalent radii and limits assignments by
common valences; ring depiction also checks planarity, connectivity, and a
simple 4n+2 electron count, including a heuristic for pyrrole-like nitrogen
lone pairs. It remains a geometry-based estimate, not
a chemical-structure assignment. It does not use chemical databases or the
statistical fragment priors used by comprehensive assignment software. CIFs
often omit explicit bond orders, and unusual valence, disorder, or metal
coordination can still make an assignment ambiguous. The approach follows the
hard-rule, bond-length, and conjugation-rule stages described by
[Zhang et al. (2012)](https://doi.org/10.1186/1758-2946-4-26), and is also
inspired by the distinction between bond detection and bond typing in
[Bruno et al. (2011)](https://doi.org/10.1107/S0108768111024608); it does not
reproduce the database-derived statistics or the full algorithms. For models other than
Wire-sticks,
**Bond Radius** scales single bonds up to 2×; **Multiple Bond Radius** scales
double and triple bonds independently. In Wire-sticks, the shared atom-and-bond
radius control keeps the single-bond and atom radii equal.

The collapsible **Polyhedra** sub-panel (placed under Bond Style) builds a
convex coordination polyhedron around central atoms. Choose an element from the
dropdown — only elements that currently have at least one atom with four or more
bonded neighbours are offered — then click **Create Polyhedra** to build one
polyhedron per site of that element, or **Create from Selected** to enter viewport
selection mode and build only around the chosen atoms. Neighbours are taken from
the P1 site table plus periodic images, so coordination spheres are complete even
when only the asymmetric unit is displayed, and the ligand atoms are added to the
displayed structure so polyhedron edges connect to real atoms. Each polyhedron
is the convex hull of its connected ligands; faces are merged where coplanar so
square faces do not get spurious diagonal edges. Per-element appearance controls
(face color, defaulting to the central atom's CPK colour, face transparency,
edge color, edge transparency and edge width) are shown for each element that has
a style, and individual element types can be removed with **Delete {element}
polyhedra** or all at once with **Clear All Polyhedra**. Square-planar and other
fully coplanar coordinations are currently skipped.

Switching models rebuilds atom and bond objects from the stored site table, so a
structure imported before this feature still switches correctly (it simply has
no ellipsoid data until it is re-imported).

Use the **Fill** menu to choose what the structure displays: **Asymmetric
Unit-cell** (the CIF asymmetric unit), **Unit-Cell** (the symmetry-expanded
contents of one unit cell), or **Pack Unit-cells...**, which asks for the
number of unit cells along a, b and c. Unit-cell and packed fills use the
space-group operations to generate the cell contents, then unwrap each finite
molecule once so atoms bonded across a cell face stay together. This can place
some atoms just outside the unit-cell boundary without duplicating molecules.
Connectivity pair distance settings determine which atoms count as bonded;
periodic frameworks remain limited to the unit-cell contents, and isolated
sites are retained.
The **Unit Cell** sub-panel has a **Show Unit Cell** toggle and a
**Pack Unit-Cells (Faces & Edges)...** control for choosing repetitions along
a, b and c. This displays a repeated cell grid without changing or duplicating
atoms. Smooth-shaded icosphere markers indicate the cell-outline corners and
match the unit-cell edge thickness. The cell is drawn with adjustable face
color, face transparency, edge color and edge thickness.
Turning **Show Unit Cell** off hides the grid. Fill
replaces the displayed atoms, clears any growth selection, and rebuilds bonds.
Packing is limited to 200,000 atoms.

Use **Grow Structure** to add one shell around all atoms. Use **Grow Selected
Atoms** to enter viewport selection mode; click or box-select atoms, use
Shift-click or Shift-drag to add to the selection, and click a selected atom
again to remove just that atom. Click picking ignores hidden atoms and chooses
the frontmost atom within the click radius; box selection keeps its existing
replace/add behavior. Selected atoms are highlighted in orange.
Press Enter or right-click to grow immediately around the selection, or press
Esc to cancel. The worker expands the CIF to P1 with gemmi and searches
symmetry-equivalent sites and integer unit-cell translations around the chosen
seeds, while avoiding every atom already in the finite cluster. Selection
highlights are cleared after growth so new atoms keep their normal element
colors and sizes. Growth does not alter the source CIF or its unit
cell/space group, and creates a finite expanded cluster rather than an
infinitely periodic structure. Repeated growth reuses the periodic neighbour
table and visited-atom state while the cell, sites and connectivity settings
are unchanged. Each shell's atom and bond geometry is stored in its own meshes,
so existing Blender meshes are not mutated while their Geometry Nodes modifiers
are being evaluated.

Use **Delete Atoms** to remove atoms. It uses the same viewport selection as
**Grow Selected Atoms**: click or box-select atoms, click a selected atom to
remove it, Shift adds to the selection, then press Enter or right-click to
delete them, or press Esc to cancel. Atom meshes and bonds are rebuilt after
deletion.

The prototype now includes coordination-polyhedron tools: the collapsible **Polyhedra**
sub-panel (under Bond Style) builds a convex polyhedron around every atom of a
chosen element that has at least four bonded neighbours. Eligible elements are
those with at least one atom carrying four or more neighbours, detected from the
P1 site table plus periodic images, so coordination spheres are complete even
when only the asymmetric unit is displayed. **Create Polyhedra** builds them for
all sites of the selected element; **Create from Selected** enters viewport
selection mode and builds only around the chosen atoms. Each polyhedron's ligand
atoms are added to the displayed structure. Per-element appearance controls
(face color, which defaults to the central atom's CPK colour, face transparency,
edge color, edge transparency and edge width) live in the same panel, and
individual element types can be removed with **Delete {element} polyhedra** or
all at once with **Clear All Polyhedra**. Polyhedra are convex hulls of the
connected ligand atoms; square-planar and other coplanar coordinations are not
yet supported.
The source unit-cell parameters, space-group symbol and symmetry operators are
retained on the generated structure collection.

## cctbx parity

`gemmi_worker.py` reproduces the cctbx results exactly on the bundled example
CIFs: same P1 site tables, same atom order, same ASU instances, occupancies and
radii (worst position difference 7·10⁻¹⁵ Å). To match cctbx:

- sites within 0.5 Å of a special position are projected onto it before
  symmetry expansion (`min_distance_sym_equiv`), which reproduces cctbx atom
  counts for special positions;
- element symbols come from the CIF type symbol, so a water labelled `W` stays
  `W` as in cctbx;
- van der Waals and covalent radii are copied from `cctbx.eltbx`, so default
  connectivity windows are identical;
- CIFs with duplicate block names are read with `check_level=0`; the last block
  with a given name wins, as in cctbx.

Stored symmetry-operator strings may use equivalent translation representatives
(e.g. `-x+1/2,y+1/2,z` instead of `-x-1/2,y-1/2,z`); the groups are identical
and the strings are informational only.

Structures imported by an earlier version of this add-on store per-atom data
instead of the compact site table used now. Re-import the CIF to grow or
rebuild such a structure.
