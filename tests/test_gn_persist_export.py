"""Check glTF and STL export of GN polyhedra from the persisted file.

Reads the file written by `test_gn_persist_build.py`.  The name is chosen so
`run_all.ps1` (alphabetical) runs build -> check -> export; as
`test_gn_export_formats.py` it ran *before* the build and silently exported a
stale scene left over from a previous run.
"""
import os
import sys
import tempfile

import bpy

BLEND = os.path.join(tempfile.gettempdir(), "crystalviz_gn_polyhedra.blend")
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


if not os.path.exists(BLEND):
    print(f"[FAIL] persisted scene exists  ({BLEND} missing; run test_gn_persist_build.py)")
    print("TOTAL FAILURES: 1")
    print("RESULT: FAIL")
    sys.exit(1)

bpy.ops.wm.open_mainfile(filepath=BLEND)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

scene = bpy.context.scene
root = bpy.data.collections.get(scene.cgn_active_structure)
check("structure collection restored", root is not None)
obj = next(
    (o for o in root.objects if o.get("cgn_role") == "polyhedron"), None
) if root else None
check("a polyhedron scaffold is present", obj is not None)
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

for name, path, operator in (
    (
        "GLB",
        os.path.join(tempfile.gettempdir(), "crystalviz_gn_polyhedra.glb"),
        lambda path: bpy.ops.export_scene.gltf(
            filepath=path, export_format="GLB", use_selection=True, export_apply=True
        ),
    ),
    (
        "STL",
        os.path.join(tempfile.gettempdir(), "crystalviz_gn_polyhedra.stl"),
        lambda path: bpy.ops.wm.stl_export(
            filepath=path, export_selected_objects=True, apply_modifiers=True
        ),
    ),
):
    if os.path.exists(path):
        os.remove(path)
    try:
        result = operator(path)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        check(f"{name} export wrote a non-empty file", size > 0, f"{size} bytes, {result}")
    except Exception as exc:
        check(f"{name} export wrote a non-empty file", False, repr(exc))

print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
