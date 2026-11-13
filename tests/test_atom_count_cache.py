"""Verify the cached cgn_atom_count scalar used by the panel draw.

Run:
  blender --background --factory-startup --python test_atom_count_cache.py

Checks:
  * import / fill / grow keep cgn_atom_count == len(cgn_instances) // 4
  * _structure_atom_count() reads the scalar WITHOUT calling json.loads
    (proven by temporarily making json.loads raise)
  * legacy structures (scalar removed) still fall back to parsing once
  * snapshot/rollback restores the scalar
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


def real_count():
    return len(json.loads(root.get("cgn_instances", "[]"))) // 4


# ---- import ----
check("import writes cgn_atom_count", "cgn_atom_count" in root.keys())
check(
    "import scalar matches instances",
    int(root.get("cgn_atom_count", -1)) == real_count(),
    f"scalar={root.get('cgn_atom_count')} real={real_count()}",
)

# ---- draw helper must not parse JSON when scalar present ----
_orig_loads = CrystalViz.json.loads


def _boom(*a, **k):
    raise AssertionError("json.loads was called by _structure_atom_count")


CrystalViz.json.loads = _boom
try:
    n = CrystalViz._structure_atom_count(root)
    parse_free = True
except AssertionError:
    parse_free = False
    n = -1
finally:
    CrystalViz.json.loads = _orig_loads
check("_structure_atom_count reads scalar without json.loads", parse_free, f"n={n}")
check("_structure_atom_count value correct", n == real_count(), f"{n} vs {real_count()}")

# ---- fill updates the scalar ----
bpy.ops.cgn.fill_cell("INVOKE_DEFAULT")
root = bpy.data.collections.get(scene.cgn_active_structure)
bpy.context.view_layer.update()
check(
    "fill updates cgn_atom_count",
    int(root.get("cgn_atom_count", -1)) == real_count(),
    f"scalar={root.get('cgn_atom_count')} real={real_count()}",
)

# ---- grow updates the scalar ----
root["cgn_selected_indices"] = json.dumps(list(range(3)))
before = real_count()
bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
bpy.context.view_layer.update()
check(
    "grow updates cgn_atom_count",
    int(root.get("cgn_atom_count", -1)) == real_count(),
    f"scalar={root.get('cgn_atom_count')} real={real_count()} (before={before})",
)

# ---- legacy fallback: no scalar -> parse once, correct ----
saved = root.get("cgn_atom_count")
del root["cgn_atom_count"]
fallback = CrystalViz._structure_atom_count(root)
check("legacy fallback counts instances", fallback == real_count(), f"{fallback} vs {real_count()}")
root["cgn_atom_count"] = saved

# ---- snapshot/rollback restores the scalar ----
snap = CrystalViz._snapshot_structure_state(root, scene.cgn_connectivity_pairs)
check("snapshot captures cgn_atom_count", "cgn_atom_count" in snap["props"])
root["cgn_atom_count"] = 999999  # mutate as a failed op might
CrystalViz._rollback_structure_state(root, snap, scene, scene.cgn_connectivity_pairs)
check(
    "rollback restores cgn_atom_count",
    int(root.get("cgn_atom_count", -1)) == real_count(),
    f"scalar={root.get('cgn_atom_count')} real={real_count()}",
)

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
