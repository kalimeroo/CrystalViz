"""Fills display atoms only: no bonds are created by Asymmetric Unit / Unit-Cell
/ Pack Unit-cells.

A fill replaces the displayed atom set, so the bonds derived from the previous
set no longer describe it.  The fill path therefore drops every bond object
(chemdraw aromatic tori included) and builds none.  Bonds reappear through the
existing actions: ``_bond_objects`` (used by Create Polyhedra, delete and a
model change) and a grow, which builds the full set when it starts from a
bond-less structure (right after a fill) and appends otherwise.

Run headless:
  blender --background --factory-startup --python test_fill_no_bonds.py
"""
import sys
import os
import json

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
    return [
        obj
        for obj in root.all_objects
        if obj.get("cgn_role") == "bonds"
    ]


def torus_objects(root):
    return [obj for obj in bond_objects(root) if obj.get("cgn_aromatic_torus")]


def atom_objects(root):
    return [obj for obj in root.objects if obj.get("cgn_role") == "atoms"]


def atom_vertices(root):
    return sum(len(obj.data.vertices) for obj in atom_objects(root))


def atom_count(root):
    return int(root.get("cgn_atom_count", -1))


def import_fresh(name="Brookite.cif", chemdraw=False):
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
    # Imports build no bonds either, so build the pre-fill baseline explicitly.
    root = bpy.data.collections.get(scene.cgn_active_structure)
    CrystalViz._bond_objects(
        root, CrystalViz._structure_atoms(root), scene.cgn_connectivity_pairs
    )
    bpy.context.view_layer.update()
    return bpy.data.collections.get(scene.cgn_active_structure)


def run_fill(mode):
    if mode == "ASU":
        return bpy.ops.cgn.fill_asu("INVOKE_DEFAULT")
    if mode == "CELL":
        return bpy.ops.cgn.fill_cell("INVOKE_DEFAULT")
    return bpy.ops.cgn.fill_pack(repeat_a=2, repeat_b=2, repeat_c=2)


# ---- 1: each fill mode drops the bonds and keeps the atoms ----------------
for mode in ("ASU", "CELL", "PACK"):
    print(f"\n=== fill {mode}: no bonds ===")
    root = import_fresh()
    before_atoms = atom_count(root)
    before_bonds = len(bond_objects(root))
    check(f"{mode} control: bonds exist before the fill", before_bonds > 0,
          f"bonds={before_bonds}")

    result = run_fill(mode)
    bpy.context.view_layer.update()
    root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)

    check(f"{mode}: fill finished", result == {"FINISHED"}, str(result))
    check(f"{mode}: every bond object is gone", not bond_objects(root),
          f"bonds={len(bond_objects(root))}")
    check(f"{mode}: atoms are still displayed",
          atom_count(root) > 0 and atom_vertices(root) > 0,
          f"atoms={atom_count(root)} vertices={atom_vertices(root)}")
    check(f"{mode}: atom objects match the stored atom count",
          atom_vertices(root) == atom_count(root),
          f"{atom_vertices(root)} vs {atom_count(root)}")
    if mode == "ASU":
        check("ASU: atom count is back to the asymmetric unit",
              atom_count(root) == before_atoms,
              f"{before_atoms} -> {atom_count(root)}")

    # Bonds can be rebuilt afterwards, through the call grow/polyhedra use.
    CrystalViz._bond_objects(
        root, CrystalViz._structure_atoms(root),
        bpy.context.scene.cgn_connectivity_pairs,
    )
    bpy.context.view_layer.update()
    check(f"{mode}: bonds can be rebuilt after the fill",
          len(bond_objects(root)) > 0,
          f"bonds={len(bond_objects(root))}")

# ---- 2: a second fill clears the bonds a rebuild brought back -------------
print("\n=== refill after a rebuild: still no bonds ===")
root = import_fresh()
run_fill("CELL")
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
CrystalViz._bond_objects(
    root, CrystalViz._structure_atoms(root),
    bpy.context.scene.cgn_connectivity_pairs,
)
bpy.context.view_layer.update()
check("refill control: bonds were rebuilt",
      len(bond_objects(root)) > 0, f"bonds={len(bond_objects(root))}")
run_fill("PACK")
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
check("refill: second fill clears the rebuilt bonds",
      not bond_objects(root), f"bonds={len(bond_objects(root))}")

# ---- 3: chemdraw aromatic tori are cleared as well ------------------------
print("\n=== chemdraw aromatic tori are cleared ===")
root = import_fresh("aromatic.cif", chemdraw=True)
check("aromatic control: import produced bond objects",
      len(bond_objects(root)) > 0, f"bonds={len(bond_objects(root))}")
check("aromatic control: an aromatic torus exists",
      len(torus_objects(root)) > 0, f"tori={len(torus_objects(root))}")
run_fill("CELL")
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
check("aromatic: fill clears tori too",
      not torus_objects(root) and not bond_objects(root),
      f"tori={len(torus_objects(root))} bonds={len(bond_objects(root))}")

# ---- 4: a model switch still brings bonds back ----------------------------
print("\n=== a model switch rebuilds bonds after a fill ===")
root = import_fresh()
run_fill("CELL")
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
check("model switch control: fill left no bonds", not bond_objects(root))
bpy.context.scene.cgn_model = "WIRE_STICKS"
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
check("model switch: bonds are rebuilt from the filled atoms",
      len(bond_objects(root)) > 0, f"bonds={len(bond_objects(root))}")

# ---- 5: growing after a fill bonds the initial atoms too ------------------
print("\n=== grow after a fill bonds the initial atoms ===")
root = import_fresh()
run_fill("CELL")
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
initial_atoms = CrystalViz._structure_atoms(root)
initial_count = len(initial_atoms)
initial_points = [
    tuple(float(value) for value in CrystalViz._display_point(atom["cart"]))
    for atom in initial_atoms
]


def endpoint_is_initial(point, eps=1e-4):
    return any(
        all(abs(point[axis] - other[axis]) <= eps for axis in range(3))
        for other in initial_points
    )


check("grow-after-fill control: fill left no bonds", not bond_objects(root),
      f"bonds={len(bond_objects(root))}")

root["cgn_selected_indices"] = json.dumps(list(range(min(3, initial_count))))
bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
grown_atoms = CrystalViz._structure_atoms(root)
check("grow-after-fill: the structure grew",
      len(grown_atoms) > initial_count,
      f"{initial_count} -> {len(grown_atoms)}")

segments = []
for obj in bond_objects(root):
    if obj.get("cgn_aromatic_torus") or obj.type != "MESH":
        continue
    mesh = obj.data
    for edge in mesh.edges:
        first = tuple(mesh.vertices[edge.vertices[0]].co)
        second = tuple(mesh.vertices[edge.vertices[1]].co)
        segments.append((first, second))

initial_initial = [
    segment
    for segment in segments
    if endpoint_is_initial(segment[0]) and endpoint_is_initial(segment[1])
]
initial_new = [
    segment
    for segment in segments
    if endpoint_is_initial(segment[0]) != endpoint_is_initial(segment[1])
]
check("grow-after-fill: bonds among the initial atoms are built",
      len(initial_initial) > 0,
      f"initial–initial={len(initial_initial)}")
check("grow-after-fill: new atoms are bonded to the initial ones",
      len(initial_new) > 0,
      f"initial–new={len(initial_new)}")

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
