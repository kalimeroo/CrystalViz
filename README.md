# CCTBX Geometry Nodes add-on prototype

This is a separate experimental Blender add-on; it does not replace or modify
the existing CCTBX Crystal Structures add-on.

Install the `cctbx_geometry_nodes_addon` folder as a Blender add-on, enable it,
and set **cctbx Python Executable** in its preferences to a Python interpreter
with `cctbx-base` installed. The add-on imports CIF asymmetric-unit atoms and
renders them through Geometry Nodes. In the default **Ball & Stick** model,
atom spheres use a quarter of the element's van der Waals radius and bond tubes
use 0.1× the mean van der Waals radius of the bonded pair, both with smooth
shading. On import the
scene is set to centimeters with **1 Å = 1 cm**: 1 Blender unit = 10 Å = 10 cm
(Unit Scale `0.1`). A 20 Å cell therefore reads 20 cm and spans 2 Blender units.

Each unordered element pair gets an independent connectivity checkbox and
inclusive minimum/maximum distance in Å, inside the collapsible **Connectivity**
sub-panel. After changing those settings, use **Rebuild Bonds**. Initial maximum
distances are inferred from the cctbx covalent radii plus 0.4 Å.

Atoms use the JMOL CPK colour scheme (the Science Notes table): colours are
stored as sRGB hex and converted to linear values so Blender displays the
intended hue. Elements missing from the table fall back to Jmol's pink.

The **CCTBX GN** sidebar tab is split into collapsible sections: the main panel
shows the import button, space group, unit-cell parameters on a three-row,
two-column grid (a, b, c on the left; α, β, γ on the right) and atom count;
**Build** holds Fill, Grow, Delete and a **Show Hydrogens** checkbox that hides
or shows hydrogen atoms together with their bonds; **Connectivity**,
**Unit Cell**, **Model & Style** and **About** are collapsed by default. The same
build actions are available from the **CCTBX** menu in the 3D Viewport header,
and CIF import is also registered under **File > Import**.

The **Model & Style** sub-panel switches the rendering model:

- **Ball & Stick** — quarter van der Waals spheres with 0.1× mean-vdW bonds.
- **Ellipsoids** — anisotropic displacement ellipsoids built from the CIF ADP
  tensors, shown at the 50 % probability level by default. The **Probability**
  slider below the model dropdown rescales them from 0 to 100 % without
  rebuilding the atom meshes. Atoms with only an isotropic Uiso value use a
  spherical ellipsoid; atoms without displacement parameters fall back to the
  Ball & Stick sphere. Bonds are drawn thin.
- **Space-filling (Use VdW radii)** — full van der Waals spheres with bonds
  hidden.
- **Wire-sticks** — very thin atoms and bonds.

Switching models rebuilds atom and bond objects from the stored site table, so a
structured imported before this feature still switches correctly (it simply has
no ellipsoid data until it is re-imported).

Use the **Fill** menu to choose what the structure displays: **Asymmetric
Unit-cell** (the CIF asymmetric unit), **Unit-Cell** (every atom in one unit
cell), or **Pack Unit-cells...**, which asks for the number of unit cells
along a, b and c. Unit-cell and packed fills keep every cell atom at its
position and complete bonded molecules that cross a cell boundary by adding
their missing atoms just outside it, so no molecule is left half-shown.
Periodic frameworks, which cannot be unwrapped into finite molecules, keep
their plain cell contents. The **Unit Cell** sub-panel has a **Show Unit Cell**
toggle; while it is on, the unit cell is drawn with adjustable face color, face
transparency, edge color and edge thickness. Turning it off hides the box. Fill
replaces the displayed atoms, clears any growth selection, and rebuilds bonds.
Packing is limited to 200,000 atoms.

Use **Grow Structure** to add one shell around all atoms. Use **Grow Selected
Atoms** to enter viewport selection mode; click or box-select atoms, and use
Shift-click or Shift-drag to add to the selection. Selected atoms are
highlighted in orange. Press Enter or right-click to grow immediately around
the selection, or press Esc to cancel. The worker expands the CIF to P1 with
cctbx and searches symmetry-equivalent sites and integer unit-cell translations
around the chosen seeds, while avoiding every atom already in the finite
cluster. Selection highlights are cleared after growth so new atoms keep their
normal element colors and sizes. Growth does not alter the source CIF or its
unit cell/space group, and creates a finite expanded cluster rather than an
infinitely periodic structure.

Use **Delete Atoms** to remove atoms. It uses the same viewport selection as
**Grow Selected Atoms**: click or box-select atoms, Shift adds to the
selection, then press Enter or right-click to delete them, or press Esc to
cancel. Atom meshes and bonds are rebuilt after deletion.

The prototype does not yet support coordination-polyhedron tools from the
original add-on.
The source unit-cell parameters, space-group symbol and symmetry operators are
retained on the generated structure collection.

Structures imported by an earlier version of this add-on store per-atom data
instead of the compact site table used now. Re-import the CIF to grow or
rebuild such a structure.
