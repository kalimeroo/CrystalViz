"""Rollback contract with Geometry Nodes polyhedra.

``_snapshot_structure_state`` stores polyhedron centers from the scaffold's
``cgn_poly_map``; ``_rollback_structure_state`` must remove both scaffolds and
their prototypes, then rebuild them from the restored instances.

Run:
  blender --background --factory-startup --python test_gn_polyhedra_rollback.py
"""
import json
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


def scaffolds(root):
    return [obj for obj in root.objects if obj.get("cgn_role") == "polyhedron"]


def prototypes(root):
    return [
        obj for obj in root.all_objects
        if obj.get("cgn_role") == "polyhedron_prototype"
    ]


def centres(obj):
    return sorted(int(index) for index in json.loads(obj.get("cgn_poly_centers", "[]")))


scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms)
bpy.ops.cgn.grow_further()
bpy.ops.cgn.grow_further()
scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()

before = sorted((centres(obj) for obj in scaffolds(root)), key=tuple)
before_scaffolds = len(before)
before_prototypes = len(prototypes(root))
print(f"  before: {before_scaffolds} scaffolds, {before_prototypes} prototypes")

snapshot = CrystalViz._snapshot_structure_state(root, scene.cgn_connectivity_pairs)
check("snapshot captured the scaffold centers",
      sorted(snapshot["polyhedra"].values(), key=len) != [],
      str({key: len(value) for key, value in snapshot["polyhedra"].items()}))

# Simulate a failed grow: mutate the stored instances.
before_instances = root.get("cgn_instances")
root["cgn_instances"] = json.dumps(
    [float(value) for value in json.loads(before_instances)] + [0.0, 0.0, 0.0, 0.0]
)
check("structure state mutated before rollback",
      root.get("cgn_instances") != before_instances)

CrystalViz._rollback_structure_state(
    root, snapshot, scene, scene.cgn_connectivity_pairs
)

check("instances restored", root.get("cgn_instances") == before_instances)
after = sorted((centres(obj) for obj in scaffolds(root)), key=tuple)
check("rollback restores the same scaffolds", after == before,
      f"{before} -> {after}")
check("rollback restores one prototype per scaffold",
      len(prototypes(root)) == before_prototypes,
      f"{before_prototypes} -> {len(prototypes(root))}")
check("rollback leaves a single prototype collection",
      len([child for child in root.children
           if child.get("cgn_role") == "polyhedron_prototypes"]) == 1)

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
for obj in scaffolds(root):
    mesh = obj.evaluated_get(depsgraph).data
    check(f"{obj.name}: evaluated geometry after rollback",
          len(mesh.polygons) > 0
          and {polygon.material_index for polygon in mesh.polygons} == {0, 1},
          f"{len(mesh.polygons)} polys")

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
