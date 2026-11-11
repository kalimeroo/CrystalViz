"""Open the saved GN-polyhedra file, verify evaluation and OBJ export."""
import os
import sys
import bpy

BLEND = r"C:\Users\Erwann Jeanneau\AppData\Local\Temp\opencode\gn_polyhedra.blend"
OBJ = r"C:\Users\Erwann Jeanneau\AppData\Local\Temp\opencode\gn_polyhedra_export.obj"
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


bpy.ops.wm.open_mainfile(filepath=BLEND)
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

scene = bpy.context.scene
root = bpy.data.collections.get(scene.cgn_active_structure)
check("structure collection restored", root is not None)

polys = [o for o in root.objects if o.get("cgn_role") == "polyhedron"]
protos = [o for o in root.all_objects if o.get("cgn_role") == "polyhedron_prototype"]
check("scaffolds restored", len(polys) == 2, f"{len(polys)}")
check("prototypes restored", len(protos) == 2, f"{len(protos)}")
collection = next(
    (child for child in root.children
     if child.get("cgn_role") == "polyhedron_prototypes"),
    None,
)
check("prototype collection restored and hidden",
      collection is not None and collection.hide_viewport and collection.hide_render)

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
for obj in polys:
    mesh = obj.evaluated_get(depsgraph).data
    indices = {polygon.material_index for polygon in mesh.polygons}
    check(f"{obj.name}: evaluated after reload",
          len(mesh.polygons) > 0 and indices == {0, 1},
          f"{len(mesh.polygons)} polys, indices={sorted(indices)}")
    check(f"{obj.name}: two materials after reload", len(mesh.materials) == 2)

# Export the first scaffold and make sure the realised polyhedra reach the file.
obj = polys[0]
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
try:
    result = bpy.ops.wm.obj_export(
        filepath=OBJ, export_selected_objects=True, apply_modifiers=True
    )
    check("OBJ export operator finished", "FINISHED" in result, str(result))
except Exception as exc:
    check("OBJ export operator finished", False, repr(exc))

if os.path.exists(OBJ):
    with open(OBJ, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    vertices = [line for line in lines if line.startswith("v ")]
    faces = [line for line in lines if line.startswith("f ")]
    check("exported OBJ contains polyhedron vertices", len(vertices) > 0,
          f"{len(vertices)} verts")
    check("exported OBJ contains polyhedron faces", len(faces) > 0,
          f"{len(faces)} faces")
else:
    check("exported OBJ exists", False)

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
