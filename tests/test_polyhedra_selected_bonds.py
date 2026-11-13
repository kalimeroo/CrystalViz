"""Regression: Create from Selected draws centre-ligand bonds even when the
polyhedron vertex atoms were already present in the structure.

Scenario: the ligand atoms are already displayed (so no instance is added),
but the centre has no bonds.  Building its polyhedron must still redraw the
bonds to every vertex, like the full Create Polyhedra path.

Run:
  blender --background --factory-startup --python test_polyhedra_selected_bonds.py
"""

import sys

import bpy
from mathutils import Vector

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


def bond_segments(root):
    segments = []
    for obj in root.objects:
        if obj.get("cgn_role") != "bonds" or obj.type != "MESH":
            continue
        for edge in obj.data.edges:
            a = Vector(obj.data.vertices[edge.vertices[0]].co)
            b = Vector(obj.data.vertices[edge.vertices[1]].co)
            segments.append((a, b))
    return segments


def neighbours(center, segments, tol=0.02):
    found = []
    for a, b in segments:
        if (a - center).length < tol:
            found.append(b)
        elif (b - center).length < tol:
            found.append(a)
    return found


scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms0 = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms0)
ag_index = next(i for i, a in enumerate(atoms0) if str(a.get("element", "")).upper() == "AG")

# Build once so every ligand image of that centre is present in the structure.
built, _ = CrystalViz._build_polyhedra_objects(root, atoms0, [ag_index], scene, clear_existing=True)
atoms_full = CrystalViz._structure_atoms(root)
center = Vector(CrystalViz._display_point(atoms_full[ag_index]["cart"]))
expected_neighbours = len(neighbours(center, bond_segments(root)))
print(f"  initial build: built={built} atoms={len(atoms_full)} neighbours={expected_neighbours}")
check(
    "initial build added atoms and bonds",
    len(atoms_full) > len(atoms0) and expected_neighbours >= 4,
    f"{len(atoms0)} -> {len(atoms_full)} atoms, {expected_neighbours} neighbours",
)

# Remove the polyhedra but keep the (already complete) atom table, then drop
# every bond: this is the "vertices already displayed, no bonds" situation.
CrystalViz._clear_polyhedra(root)
for obj in list(root.objects):
    if obj.get("cgn_role") == "bonds":
        CrystalViz._remove_object(obj)
check("bonds were cleared for the test", len(bond_segments(root)) == 0)

# Create from Selected on the same atom: no new atom is needed, but the
# centre-ligand bonds must come back.
built, skipped = CrystalViz._build_polyhedra_objects(
    root, atoms_full, [ag_index], scene, clear_existing=True
)
atoms_after = CrystalViz._structure_atoms(root)
after_neighbours = len(neighbours(center, bond_segments(root)))
print(
    f"  selected build: built={built} skipped={skipped} "
    f"atoms={len(atoms_after)} neighbours={after_neighbours}"
)
check("polyhedron rebuilt", built == 1, f"built={built}")
check(
    "no extra atoms needed",
    len(atoms_after) == len(atoms_full),
    f"{len(atoms_full)} -> {len(atoms_after)}",
)
check(
    "centre-ligand bonds are drawn",
    after_neighbours == expected_neighbours,
    f"{after_neighbours} vs expected {expected_neighbours}",
)
check("at least four centre-ligand bonds", after_neighbours >= 4, f"{after_neighbours}")

CrystalViz.unregister()
check("unregister OK", True)
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
