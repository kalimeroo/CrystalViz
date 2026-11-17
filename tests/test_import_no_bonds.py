"""Importing a CIF displays atoms only — the same behaviour as Fill → ASU.

Import creates the structure's atoms, unit cell and default connectivity pairs,
but no bond objects (no chemdraw aromatic tori either).  Bonds come from the
same actions as after a fill: grow, Create Polyhedra, Delete Atoms, or a
rendering-model change.

Run headless:
  blender --background --factory-startup --python test_import_no_bonds.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import bpy
import CrystalViz

CrystalViz.register()

CIF = _env.CIF_DIR
PAIRS = (("Ti", "O", 0.1, 3.0), ("O", "O", 0.5, 3.0))

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def bond_objects(root):
    return [obj for obj in root.all_objects if obj.get("cgn_role") == "bonds"]


def torus_objects(root):
    return [obj for obj in bond_objects(root) if obj.get("cgn_aromatic_torus")]


def atom_objects(root):
    return [obj for obj in root.objects if obj.get("cgn_role") == "atoms"]


def atom_vertices(root):
    return sum(len(obj.data.vertices) for obj in atom_objects(root))


def atom_count(root):
    return int(root.get("cgn_atom_count", -1))


def structure_roots():
    return [
        collection
        for collection in bpy.data.collections
        if collection.get("cgn_role") == "structure"
    ]


def import_cif(name="Brookite.cif", chemdraw=False):
    scene = bpy.context.scene
    CrystalViz._cancel_all_debounce()
    scene.cgn_model = "BALL_STICK"
    scene.cgn_chemdraw = chemdraw
    scene.cgn_connectivity_pairs.clear()
    bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, name))
    scene = bpy.context.scene
    if name == "Brookite.cif":
        for element_a, element_b, low, high in PAIRS:
            pair = scene.cgn_connectivity_pairs.add()
            pair.element_a = element_a
            pair.element_b = element_b
            pair.enabled = True
            pair.min_distance = low
            pair.max_distance = high
    bpy.context.view_layer.update()
    return bpy.data.collections.get(scene.cgn_active_structure)


def segments(root):
    """Every bond segment as a display-space endpoint pair."""
    found = []
    for obj in bond_objects(root):
        if obj.get("cgn_aromatic_torus") or obj.type != "MESH":
            continue
        mesh = obj.data
        for edge in mesh.edges:
            found.append(
                (
                    tuple(mesh.vertices[edge.vertices[0]].co),
                    tuple(mesh.vertices[edge.vertices[1]].co),
                )
            )
    return found


def near(point, other, eps=1e-4):
    return all(abs(point[axis] - other[axis]) <= eps for axis in range(3))


# ---- A: importing creates atoms and connectivity windows but no bonds -----
print("\n=== A: import displays atoms only ===")
root = import_cif()
check(
    "A: import creates no bond objects", not bond_objects(root), f"bonds={len(bond_objects(root))}"
)
check(
    "A: atoms are displayed",
    atom_count(root) > 0 and atom_vertices(root) == atom_count(root),
    f"atoms={atom_count(root)} vertices={atom_vertices(root)}",
)
check(
    "A: import stores the default connectivity windows",
    any(pair.enabled for pair in bpy.context.scene.cgn_connectivity_pairs),
    f"pairs={len(bpy.context.scene.cgn_connectivity_pairs)}",
)
check(
    "A: only one structure exists after import",
    len(structure_roots()) == 1,
    f"roots={len(structure_roots())}",
)

# ---- B: re-importing replaces a bonded structure with a bond-less one -----
print("\n=== B: re-import drops the previous structure's bonds ===")
CrystalViz._bond_objects(
    root,
    CrystalViz._structure_atoms(root),
    bpy.context.scene.cgn_connectivity_pairs,
)
bpy.context.view_layer.update()
check(
    "B control: the first structure has bonds",
    len(bond_objects(root)) > 0,
    f"bonds={len(bond_objects(root))}",
)
root = import_cif()
check(
    "B: the re-imported structure has no bonds",
    not bond_objects(root),
    f"bonds={len(bond_objects(root))}",
)
check(
    "B: no bond object survives anywhere in the scene",
    not any(obj.get("cgn_role") == "bonds" for obj in bpy.data.objects),
)

# ---- C: growing right after an import builds the full bond set ------------
print("\n=== C: grow after import bonds the initial atoms ===")
root = import_cif()
initial_atoms = CrystalViz._structure_atoms(root)
initial_count = len(initial_atoms)
initial_points = [
    tuple(float(value) for value in CrystalViz._display_point(atom["cart"]))
    for atom in initial_atoms
]
check("C control: import left no bonds", not bond_objects(root))
bpy.ops.cgn.grow_further()
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
grown_atoms = CrystalViz._structure_atoms(root)
check(
    "C: the structure grew",
    len(grown_atoms) > initial_count,
    f"{initial_count} -> {len(grown_atoms)}",
)


def is_initial(point):
    return any(near(point, other) for other in initial_points)


grown_segments = segments(root)
initial_initial = [
    segment for segment in grown_segments if is_initial(segment[0]) and is_initial(segment[1])
]
initial_new = [
    segment for segment in grown_segments if is_initial(segment[0]) != is_initial(segment[1])
]
check(
    "C: bonds among the initial atoms are built",
    len(initial_initial) > 0,
    f"initial–initial={len(initial_initial)}",
)
check(
    "C: new atoms are bonded to the initial ones",
    len(initial_new) > 0,
    f"initial–new={len(initial_new)}",
)

# ---- D: Create Polyhedra after import brings bonds back -------------------
print("\n=== D: Create Polyhedra after import builds bonds ===")
root = import_cif()
atoms = CrystalViz._structure_atoms(root)
ti = [index for index, atom in enumerate(atoms) if str(atom.get("element", "X")).upper() == "TI"]
check("D control: import left no bonds", not bond_objects(root))
built, _skipped = CrystalViz._build_polyhedra_objects(root, atoms, ti, bpy.context.scene)
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
check("D: polyhedra were built", built > 0, f"built={built}")
check(
    "D: the polyhedron build restored the bonds",
    len(bond_objects(root)) > 0,
    f"bonds={len(bond_objects(root))}",
)

# ---- E: a rendering-model change after import brings bonds back -----------
print("\n=== E: model switch after import builds bonds ===")
root = import_cif()
check("E control: import left no bonds", not bond_objects(root))
bpy.context.scene.cgn_model = "WIRE_STICKS"
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
check(
    "E: the model switch rebuilt the bonds",
    len(bond_objects(root)) > 0,
    f"bonds={len(bond_objects(root))}",
)

# ---- F: chemdraw import creates no bonds and no aromatic tori -------------
print("\n=== F: chemdraw import stays bond-free ===")
root = import_cif("multiple_bonds.cif", chemdraw=True)
check(
    "F: chemdraw import creates no bond objects",
    not bond_objects(root),
    f"bonds={len(bond_objects(root))}",
)
check(
    "F: chemdraw import creates no aromatic tori",
    not torus_objects(root),
    f"tori={len(torus_objects(root))}",
)

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
