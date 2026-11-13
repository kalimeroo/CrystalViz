"""Headless Blender integration check for the S1/C1 worker change: import +
grow must produce sane, non-crashing results (parity vs reference already
verified in test_S1_parity.py at the worker level).

Run:
  blender --background --factory-startup --python test_S1_blender.py
"""

import json
import os
import sys

import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

CIF = _env.CIF_DIR
scene = bpy.context.scene
scene.cgn_connectivity_pairs.clear()
bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "Brookite.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)
if not any(p.enabled for p in scene.cgn_connectivity_pairs):
    p = scene.cgn_connectivity_pairs.add()
    p.element_a = "Ti"
    p.element_b = "O"
    p.enabled = True
    p.min_distance = 0.1
    p.max_distance = 3.0
bpy.context.view_layer.update()

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


cnt0 = len(json.loads(root.get("cgn_instances", "[]"))) // 4
root["cgn_selected_indices"] = json.dumps(list(range(5)))
res = bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
cnt1 = len(json.loads(root.get("cgn_instances", "[]"))) // 4
check("grow completes without crashing", True)
check("grow adds atoms", cnt1 > cnt0, f"{cnt0}->{cnt1}")
bpy.context.view_layer.update()
check("atom objects present", sum(1 for o in root.objects if o.get("cgn_role") == "atoms") > 0)

CrystalViz.unregister()
print("\nTOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
