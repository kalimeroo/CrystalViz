"""Test polyhedron geometry appears and is correctly positioned/scaled.

Checks:
  1. Polyhedron objects have real mesh/curve data (not empties)
  2. Positions use DISPLAY_SCALE (Å × 0.1), matching atom positions
  3. Shared datablocks: one mesh + one curve per site
  4. Edge parented to face (one editable unit)
  5. One face+edge pair per qualifying-site image

Run:
  blender --background --factory-startup --python test_polyhedra_geometry.py
"""

import sys

import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

CIF = _env.cif("Ag2Cr2O7.cif")
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms0 = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms0)

# Grow twice
bpy.ops.cgn.grow_further()
bpy.ops.cgn.grow_further()
atoms2 = CrystalViz._structure_atoms(root)
ag_images = [a for a in atoms2 if str(a.get("element", "X")).upper() == "AG"]

# Create polyhedra for Ag
scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()

# Find face and edge objects
ag_faces = [
    o
    for o in root.objects
    if o.get("cgn_role") == "polyhedron_face" and str(o.get("cgn_poly_element", "")).upper() == "AG"
]
ag_edges = [
    o
    for o in root.objects
    if o.get("cgn_role") == "polyhedron_edge" and str(o.get("cgn_poly_element", "")).upper() == "AG"
]
print(f"  Ag face objects: {len(ag_faces)}")
print(f"  Ag edge objects: {len(ag_edges)}")

built_sites = {int(o.get("cgn_p1_index", -1)) for o in ag_faces}
print(f"  Built sites: {sorted(built_sites)}")

# Fix 1: Geometry appears - objects have real data, not empties
all_have_mesh = all(
    o.type == "MESH" and o.data is not None and len(o.data.vertices) > 0 for o in ag_faces
)
all_have_curve = all(o.type == "CURVE" and o.data is not None for o in ag_edges)
check("all face objects are real meshes with vertices", all_have_mesh)
check("all edge objects are real curves", all_have_curve)

# Fix 2: Positions match atom positions with DISPLAY_SCALE
face_positions = {tuple(round(v, 4) for v in o.matrix_world.translation) for o in ag_faces}
qualifying_positions = set()
for a in ag_images:
    if int(a.get("p1_index", -1)) in built_sites:
        pos = tuple(round(v * 0.1, 4) for v in a["cart"])
        qualifying_positions.add(pos)

missing = qualifying_positions - face_positions
extra = face_positions - qualifying_positions
print(f"  qualifying positions: {len(qualifying_positions)}  face positions: {len(face_positions)}")
check(
    "polyhedron at every qualifying Ag position (correct scale)",
    len(missing) == 0 and len(extra) == 0,
    f"missing={len(missing)} extra={len(extra)}",
)

# Fix 3: Shared datablocks - one mesh + one curve per site
mesh_blocks = {o.data for o in ag_faces}
curve_blocks = {o.data for o in ag_edges}
check(
    "one mesh data block per site (shared datablock)",
    len(mesh_blocks) == len(built_sites),
    f"{len(mesh_blocks)} mesh blocks vs {len(built_sites)} sites",
)
check(
    "one curve data block per site (shared datablock)",
    len(curve_blocks) == len(built_sites),
    f"{len(curve_blocks)} curve blocks vs {len(built_sites)} sites",
)

# Fix 4: Edge parented to face
for f in ag_faces:
    children = [e for e in ag_edges if e.parent is f]
    check(
        f"face {f.name} has exactly 1 edge child",
        len(children) == 1,
        f"{len(children)} edge children",
    )

# Fix 5: One face+edge pair per qualifying-site image
qualifying_images = sum(1 for a in ag_images if int(a.get("p1_index", -1)) in built_sites)
check(
    "face count == qualifying-site images",
    len(ag_faces) == qualifying_images,
    f"{len(ag_faces)} faces vs {qualifying_images} images",
)
check(
    "edge count == face count",
    len(ag_edges) == len(ag_faces),
    f"{len(ag_edges)} edges vs {len(ag_faces)} faces",
)

# Object count is 2 per image (face + edge) but shares 2 datablocks per site
print(
    f"  total objects: {len(ag_faces) + len(ag_edges)}  "
    f"shared datablocks: {len(mesh_blocks) + len(curve_blocks)}"
)

CrystalViz.unregister()
check("unregister OK", True)

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
