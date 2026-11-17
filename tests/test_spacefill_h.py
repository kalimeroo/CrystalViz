"""Verify Space-fill H atoms use the full VdW radius (ignore Hydrogen Size).

Run:
  blender --background --factory-startup --python test_spacefill_h.py
"""

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
bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "multiple_bonds.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)
bpy.context.view_layer.update()

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def h_object():
    for o in root.objects:
        if o.get("cgn_role") == "atoms" and str(o.get("cgn_element", "")).upper() == "H":
            return o
    return None


def radius_of(obj):
    for mod in obj.modifiers:
        if mod.type == "NODES" and mod.node_group is not None:
            ident = CrystalViz._socket_identifier(mod.node_group, "Radius")
            inputs = getattr(getattr(mod, "properties", None), "inputs", None)
            input_value = getattr(inputs, ident, None)
            if input_value is not None and hasattr(input_value, "value"):
                return float(input_value.value)
            return float(mod[ident])
    return None


# ---- Space-fill: H uses full VdW ----
scene.cgn_model = "SPACE_FILLING"
CrystalViz._update_model(scene, bpy.context)
bpy.context.view_layer.update()
h = h_object()
check("H atom object exists in space-fill", h is not None)
vdw = float(h.get("cgn_vdw_radius"))
expected_full = vdw * CrystalViz._SPACE_FILLING_ATOM_VDW_FACTOR * CrystalViz._DISPLAY_SCALE
r = radius_of(h)
check(
    "space-fill H radius == full VdW radius (no Hydrogen Size)",
    abs(r - expected_full) < 1e-5,
    f"got={r:.6f} expected={expected_full:.6f} vdw={vdw}",
)

# Changing hydrogen scale must not alter space-fill H radius.
scene.cgn_hydrogen_scale = 0.5
CrystalViz._update_hydrogen_scale(scene, bpy.context)
r_after = radius_of(h_object())
check(
    "space-fill H radius unchanged after hydrogen scale change",
    abs(r_after - r) < 1e-12,
    f"{r:.6f}->{r_after:.6f}",
)
check(
    "space-fill H radius is NOT 0.5x (scale ignored)",
    abs(r_after - vdw * 0.5 * CrystalViz._SPACE_FILLING_ATOM_VDW_FACTOR * CrystalViz._DISPLAY_SCALE)
    > 1e-6,
)

# ---- ELLIPSOID: Hydrogen Size still applies to H ----
scene.cgn_hydrogen_scale = 0.5
scene.cgn_model = "ELLIPSOID"
CrystalViz._update_model(scene, bpy.context)
bpy.context.view_layer.update()
h2 = h_object()
# H in ellipsoid uses the atoms node group with radius = vdw * 0.25 * size_scale * scale
expected_half = vdw * CrystalViz._BALL_ATOM_VDW_FACTOR * 0.5 * CrystalViz._DISPLAY_SCALE
r2 = radius_of(h2)
check(
    "ellipsoid H radius still uses Hydrogen Size (0.5x)",
    abs(r2 - expected_half) < 1e-5,
    f"got={r2:.6f} expected={expected_half:.6f}",
)

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
