"""Check polyhedron remap (after atom deletion) with GN prototypes."""
import json
import sys
import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

CIF = r"C:\Users\Erwann Jeanneau\Documents\Python_Scripts\Blender_addon\CrystalViz\CIF\Ag2Cr2O7.cif"
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def scaffolds(root, element="AG"):
    return [o for o in root.objects if o.get("cgn_role") == "polyhedron"
            and str(o.get("cgn_poly_element", "")).upper() == element]


def prototypes(root, element="AG"):
    return [o for o in root.all_objects if o.get("cgn_role") == "polyhedron_prototype"
            and str(o.get("cgn_poly_element", "")).upper() == element]


scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms)
bpy.ops.cgn.grow_further()
bpy.ops.cgn.grow_further()
scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()

before = len(scaffolds(root))
target = int(json.loads(scaffolds(root)[0].get("cgn_poly_centers", "[]"))[0])
print("  before", before, "scaffolds; deleting atom", target)
CrystalViz._remap_polyhedra(root, [target], atoms, scene)
after = scaffolds(root)
check("remap keeps at most the original number of sites", len(after) <= before,
      f"{before} -> {len(after)}")
check("remap rebuilds one prototype per scaffold",
      len(prototypes(root)) == len(after),
      f"{len(after)} scaffolds, {len(prototypes(root))} prototypes")
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
for obj in after:
    mesh = obj.evaluated_get(depsgraph).data
    check(f"{obj.name}: evaluated after remap", len(mesh.polygons) > 0,
          f"{len(mesh.polygons)} polys")

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
