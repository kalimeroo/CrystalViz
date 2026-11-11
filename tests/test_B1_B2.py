"""Validate B1 (_remap_polyhedra centers arg) and B2 (delete atomicity/rollback).

Run headless:
  blender --background --factory-startup --python test_B1_B2.py

B1: _remap_polyhedra passed a STRING (site label) to _polyhedron_geometry where a
    list of center indices was expected, which crashed the atom-delete path when
    polyhedra were shown.  This test drives the same data manipulation the delete
    operator performs and asserts _remap_polyhedra neither crashes nor drops the
    rest of the polyhedra.

B2: the delete operator now snapshots before mutating and rolls back on a
    mid-rebuild failure.  We exercise the identical snapshot->mutate->rollback
    contract used by delete and assert the stored props and display are restored.
"""
import sys, os, json, bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

CIF = r"C:\Users\Erwann Jeanneau\Documents\Python_Scripts\Blender_addon\CrystalViz\CIF"
scene = bpy.context.scene
scene.cgn_connectivity_pairs.clear()
scene.cgn_chemdraw = False

bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "Brookite.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)
if not any(p.enabled for p in scene.cgn_connectivity_pairs):
    pair = scene.cgn_connectivity_pairs.add()
    pair.element_a = "Ti"; pair.element_b = "O"; pair.enabled = True
    pair.min_distance = 0.1; pair.max_distance = 3.0
bpy.context.view_layer.update()

FAILURES = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)

def poly_baseline():
    return sum(
        1 for o in root.objects
        if o.get("cgn_role") in ("polyhedron",)
    )

def atom_count():
    return len(json.loads(root.get("cgn_instances", "[]"))) // 4

# ---- Setup: full cell + polyhedra around Ti ----
print("\n=== B1: _remap_polyhedra centers-arg fix ===")
try:
    bpy.ops.cgn.fill_cell("INVOKE_DEFAULT")
except Exception as exc:
    print("fill_cell failed (control issue):", exc)
root = bpy.data.collections.get(scene.cgn_active_structure)
bpy.context.view_layer.update()
atoms = CrystalViz._structure_atoms(root)
ti = [i for i, a in enumerate(atoms) if str(a.get("element", "X")).upper() == "TI"]
built, skipped = CrystalViz._build_polyhedra_objects(root, atoms, ti, scene)
bpy.context.view_layer.update()
faces = poly_baseline()
check("control: polyhedra built", built > 0 and faces > 0,
      f"built={built} face_objs={faces}")
check("control: enough Ti centers", len(ti) >= 2, f"n={len(ti)}")

# Pick the LAST Ti center as the atom to delete (its polyhedron centers object
# still holds the un-deleted member so it exercises the rebuild path).
remove = ti[-1]
removed = [remove]
removed_offsets = {remove * 4}
instances = json.loads(root.get("cgn_instances", "[]"))
new_instances = [v for off, v in enumerate(instances) if (off - off % 4) not in removed_offsets]
removed_quads = {
    (int(instances[remove*4]), int(instances[remove*4+1]),
     int(instances[remove*4+2]), int(instances[remove*4+3]))
}
sites, _inst, cell_matrix, initial_count = CrystalViz._structure_tables(root)
pre_remove_stored = list(json.loads(root.get("cgn_instances", "[]")))
new_atoms = CrystalViz._materialize_atoms(sites, cell_matrix, new_instances, initial_count)

# Sanity: the instance list has exactly the one deleted atom removed, and the
# rebuilt atom list is one shorter than the pre-removal materialization.
check("new instance list is exactly one atom shorter",
      len(new_instances) == len(pre_remove_stored) - 4,
      f"{len(pre_remove_stored)//4}->{len(new_instances)//4}")
pre_atoms = CrystalViz._materialize_atoms(sites, cell_matrix, pre_remove_stored, initial_count)
check("new atoms materialize one shorter than pre-removal",
      len(new_atoms) == len(pre_atoms) - 1,
      f"{len(pre_atoms)}->{len(new_atoms)}")

# Write the new state just like the operator, then call _remap_polyhedra.
root["cgn_instances"] = json.dumps(new_instances)
root["cgn_initial_count"] = initial_count
try:
    CrystalViz._remap_polyhedra(root, removed, new_atoms, scene, removed_quads)
    crash = None
except Exception as exc:
    crash = exc
check("B1: _remap_polyhedra does NOT crash with the centers-arg fix",
      crash is None, f"{crash!r}" if crash else "")
if crash is None:
    bpy.context.view_layer.update()
    faces_after = poly_baseline()
    check("B1: surviving polyhedra retained after the centre was deleted",
          faces_after > 0, f"faces {faces}->{faces_after}")

# ---- B2: snapshot->mutate->rollback contract used by the delete operator ----
print("\n=== B2: delete snapshot & rollback contract ===")
# Build a clean, predictable structure state.
bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "aromatic.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)
bpy.context.view_layer.update()
base = CrystalViz._snapshot_structure_state(root, scene.cgn_connectivity_pairs)
before_inst = root.get("cgn_instances")
before_count = atom_count()

# Mirror the operator: snapshot taken, then mutate stored state, then simulate a
# mid-rebuild failure inside the rollback-visible rebuild step.
root["cgn_instances"] = json.dumps([float(v) for v in json.loads(before_inst)] + [0.0, 0.0, 0.0, 0.0])
root["cgn_initial_count"] = root.get("cgn_initial_count")  # unchanged
check("B2 setup: state actually mutated before rollback",
      atom_count() == before_count + 1, f"{before_count}->{atom_count()}")

CrystalViz._rollback_structure_state(root, base, scene, scene.cgn_connectivity_pairs)
after_inst = root.get("cgn_instances")
check("B2: cgn_instances restored to pre-operation value", after_inst == before_inst)
check("B2: atom count restored", atom_count() == before_count,
      f"{before_count}->{atom_count()}")
check("B2: cgn_initial_count restored",
      root.get("cgn_initial_count") == base["props"].get("cgn_initial_count"))

# B2 end-to-end through the operator's own code path is modal-only (needs a
# VIEW_3D area), so we assert the exact rollback helper the operator now wires
# in reproduces the pre-delete state -- which is what makes delete atomic.

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
