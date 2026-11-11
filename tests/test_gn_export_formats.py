"""Check glTF and STL export of GN polyhedra from the saved file."""
import os
import sys
import bpy

BLEND = r"C:\Users\Erwann Jeanneau\AppData\Local\Temp\opencode\gn_polyhedra.blend"
GLB = r"C:\Users\Erwann Jeanneau\AppData\Local\Temp\opencode\gn_polyhedra.glb"
STL = r"C:\Users\Erwann Jeanneau\AppData\Local\Temp\opencode\gn_polyhedra.stl"

bpy.ops.wm.open_mainfile(filepath=BLEND)
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

scene = bpy.context.scene
root = bpy.data.collections.get(scene.cgn_active_structure)
obj = next(o for o in root.objects if o.get("cgn_role") == "polyhedron")
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

for path, operator in (
    (GLB, lambda: bpy.ops.export_scene.gltf(
        filepath=GLB, export_format="GLB", use_selection=True, export_apply=True)),
    (STL, lambda: bpy.ops.wm.stl_export(
        filepath=STL, export_selected_objects=True, apply_modifiers=True)),
):
    if os.path.exists(path):
        os.remove(path)
    try:
        result = operator()
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"{os.path.basename(path)}: {result} size={size}", flush=True)
    except Exception as exc:
        print(f"{os.path.basename(path)} FAILED: {exc!r}", flush=True)
