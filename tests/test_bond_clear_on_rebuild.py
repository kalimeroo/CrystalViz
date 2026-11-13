"""A full bond rebuild must drop the old bonds when nothing can be bonded.

Regression: Fill → Grow → Delete with a box selection covering every atom but
one left the bond objects in place.  Deleting down to a single atom calls the
full ``_bond_objects`` rebuild, which returned early (fewer than two atoms) and
never removed the previous set.  The same early return also kept stale bonds
when every connectivity pair was disabled, when the widest window was zero, or
in the space-filling model, even though its comment promised they are dropped.

Run headless:
  blender --background --factory-startup --python test_bond_clear_on_rebuild.py
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


def atom_vertices(root):
    return sum(len(obj.data.vertices) for obj in root.objects if obj.get("cgn_role") == "atoms")


def setup_grown():
    """Import, fill the unit cell, then grow: the user's starting state."""
    scene = bpy.context.scene
    CrystalViz._cancel_all_debounce()
    scene.cgn_model = "BALL_STICK"
    scene.cgn_chemdraw = False
    scene.cgn_connectivity_pairs.clear()
    bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "Brookite.cif"))
    scene = bpy.context.scene
    for element_a, element_b, low, high in PAIRS:
        pair = scene.cgn_connectivity_pairs.add()
        pair.element_a = element_a
        pair.element_b = element_b
        pair.enabled = True
        pair.min_distance = low
        pair.max_distance = high
    bpy.ops.cgn.fill_cell("INVOKE_DEFAULT")
    bpy.context.view_layer.update()
    root = bpy.data.collections.get(scene.cgn_active_structure)
    bpy.ops.cgn.grow_further()
    bpy.context.view_layer.update()
    return root


def delete_all_but_first(root):
    """Replay exactly what ``CGN_OT_delete_atoms._confirm`` does."""
    scene = bpy.context.scene
    sites, instances, cell_matrix, initial_count = CrystalViz._structure_tables(root)
    atom_count = len(instances) // 4
    removed = list(range(1, atom_count))
    removed_offsets = {index * 4 for index in removed}
    removed_quads = {
        (
            int(instances[index * 4]),
            int(instances[index * 4 + 1]),
            int(instances[index * 4 + 2]),
            int(instances[index * 4 + 3]),
        )
        for index in removed
    }
    new_instances = [
        value
        for offset, value in enumerate(instances)
        if (offset - offset % 4) not in removed_offsets
    ]
    removed_below_initial = sum(1 for index in removed if index < initial_count)
    new_initial_count = max(0, initial_count - removed_below_initial)
    atoms = CrystalViz._materialize_atoms(sites, cell_matrix, new_instances, new_initial_count)
    CrystalViz._set_instances(root, new_instances)
    root["cgn_initial_count"] = new_initial_count
    CrystalViz._rebuild_atom_objects(root, atoms)
    CrystalViz._update_growth_selection(root, atoms, [])
    CrystalViz._remap_polyhedra(root, removed, atoms, scene, removed_quads)
    CrystalViz._bond_objects(root, atoms, scene.cgn_connectivity_pairs)
    bpy.context.view_layer.update()
    return atoms


# ---- A: the reported sequence (fill → grow → delete all but one) ----------
print("\n=== A: fill → grow → delete all but one atom ===")
root = setup_grown()
check(
    "A control: the grown structure has bonds and atoms",
    len(bond_objects(root)) > 0 and atom_vertices(root) > 1,
    f"bonds={len(bond_objects(root))} atoms={atom_vertices(root)}",
)
remaining = delete_all_but_first(root)
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
check(
    "A: exactly one atom remains",
    len(remaining) == 1
    and atom_vertices(root) == 1
    and CrystalViz._structure_atom_count(root) == 1,
    f"atoms={CrystalViz._structure_atom_count(root)} vertices={atom_vertices(root)}",
)
check("A: no bond object survives", not bond_objects(root), f"bonds={len(bond_objects(root))}")

# ---- B: the rebuild primitive drops bonds for 0/1 atoms ------------------
print("\n=== B: a rebuild with fewer than two atoms clears bonds ===")
root = setup_grown()
atoms = CrystalViz._structure_atoms(root)
check(
    "B control: bonds exist before the rebuild",
    len(bond_objects(root)) > 0,
    f"bonds={len(bond_objects(root))}",
)
CrystalViz._bond_objects(root, atoms[:1], bpy.context.scene.cgn_connectivity_pairs)
check(
    "B: one atom leaves no bond object", not bond_objects(root), f"bonds={len(bond_objects(root))}"
)
root = setup_grown()
CrystalViz._bond_objects(root, [], bpy.context.scene.cgn_connectivity_pairs)
check(
    "B: an empty structure leaves no bond object",
    not bond_objects(root),
    f"bonds={len(bond_objects(root))}",
)

# ---- C: no enabled pair clears the previous set --------------------------
print("\n=== C: a rebuild with every pair disabled clears bonds ===")
root = setup_grown()
check(
    "C control: bonds exist before the rebuild",
    len(bond_objects(root)) > 0,
    f"bonds={len(bond_objects(root))}",
)
for pair in bpy.context.scene.cgn_connectivity_pairs:
    pair.enabled = False
CrystalViz._bond_objects(
    root,
    CrystalViz._structure_atoms(root),
    bpy.context.scene.cgn_connectivity_pairs,
)
check(
    "C: disabling every pair drops the bonds",
    not bond_objects(root),
    f"bonds={len(bond_objects(root))}",
)

# ---- D: a zero-width window clears the previous set ----------------------
print("\n=== D: a rebuild with max distance 0 clears bonds ===")
root = setup_grown()
for pair in bpy.context.scene.cgn_connectivity_pairs:
    pair.enabled = True
    pair.max_distance = 0.0
CrystalViz._bond_objects(
    root,
    CrystalViz._structure_atoms(root),
    bpy.context.scene.cgn_connectivity_pairs,
)
check(
    "D: a zero window drops the bonds", not bond_objects(root), f"bonds={len(bond_objects(root))}"
)

# ---- E: space-filling clears bonds, ball & stick restores them -----------
print("\n=== E: the space-filling model drops bonds in the scene ===")
root = setup_grown()
check(
    "E control: bonds exist before the model switch",
    len(bond_objects(root)) > 0,
    f"bonds={len(bond_objects(root))}",
)
bpy.context.scene.cgn_model = "SPACE_FILLING"
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
check("E: space-filling shows no bonds", not bond_objects(root), f"bonds={len(bond_objects(root))}")
bpy.context.scene.cgn_model = "BALL_STICK"
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
check(
    "E: switching back restores the bonds",
    len(bond_objects(root)) > 0,
    f"bonds={len(bond_objects(root))}",
)

# ---- F: append-only growth still preserves bonds when nothing is enabled --
print("\n=== F: append-only grow keeps existing bonds ===")
root = setup_grown()
before = len(bond_objects(root))
for pair in bpy.context.scene.cgn_connectivity_pairs:
    pair.enabled = False
CrystalViz._bond_objects(
    root,
    CrystalViz._structure_atoms(root),
    bpy.context.scene.cgn_connectivity_pairs,
    start_index=CrystalViz._structure_atom_count(root),
    append_only=True,
)
check(
    "F: append-only leaves the existing bonds untouched",
    len(bond_objects(root)) == before,
    f"{before} -> {len(bond_objects(root))}",
)

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
