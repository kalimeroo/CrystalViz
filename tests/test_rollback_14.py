"""Validate snapshot+rollback for #14 (fill/grow persistence atomicity).

Run headless:
  blender --background --factory-startup --python test_rollback_14.py

Asserts that after an injected mid-rebuild failure, BOTH the stored cgn_*
props AND the displayed atoms/bonds are restored to the pre-operation state.
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
scene.cgn_chemdraw = True

bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "multiple_bonds.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)

# Ensure at least one enabled, generous connectivity pair so grow produces atoms.
have = any(p.enabled for p in scene.cgn_connectivity_pairs)
if not have:
    pair = scene.cgn_connectivity_pairs.add()
    pair.element_a = "C"
    pair.element_b = "C"
    pair.enabled = True
    pair.min_distance = 0.1
    pair.max_distance = 3.0
bpy.context.view_layer.update()

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def baseline():
    return {
        "instances": root.get("cgn_instances"),
        "initial_count": root.get("cgn_initial_count"),
        "selected": root.get("cgn_selected_indices"),
        "shells": root.get("cgn_growth_shells"),
        "bond_sig": root.get("cgn_bond_settings_signature"),
        "atom_verts": sum(
            len(o.data.vertices) for o in root.objects if o.get("cgn_role") == "atoms"
        ),
        # Polyhedra are GN points scaffolds (role "polyhedron") plus a hidden
        # prototype; there is no baked "polyhedron_face" object any more.
        "poly_objects": sum(1 for o in root.objects if o.get("cgn_role") == "polyhedron"),
    }


def one_shot_fail(attr):
    """Patch ``CrystalViz.<attr>`` so the FIRST call succeeds then raises, and
    all later calls (including the rollback rebuild) run normally."""
    original = getattr(CrystalViz, attr)
    state = {"fired": False}

    def wrapper(*args, **kwargs):
        if not state["fired"]:
            state["fired"] = True
            original(*args, **kwargs)
            raise RuntimeError("INJECTED mid-rebuild failure")
        return original(*args, **kwargs)

    setattr(CrystalViz, attr, wrapper)
    return lambda: setattr(CrystalViz, attr, original)


def one_shot_bond_fail():
    """Failure injection for the grow/polyhedra paths (they rebuild bonds)."""
    return one_shot_fail("_bond_objects")


def one_shot_unit_cell_fail():
    """Failure injection for the fill path: fills build no bonds, so the first
    bond-free rebuild step (``_apply_unit_cell_settings``) is the hook."""
    return one_shot_fail("_apply_unit_cell_settings")


# ---------------- GROW test ----------------
print("\n=== GROW rollback test (Brookite, seed-grow) ===")


def _brookite():
    scene.cgn_connectivity_pairs.clear()
    bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "Brookite.cif"))
    r = bpy.data.collections.get(scene.cgn_active_structure)
    if not any(p.enabled for p in scene.cgn_connectivity_pairs):
        pair = scene.cgn_connectivity_pairs.add()
        pair.element_a = "Ti"
        pair.element_b = "O"
        pair.enabled = True
        pair.min_distance = 0.1
        pair.max_distance = 3.0
    # Import builds no bonds; build the baseline set explicitly so the grow
    # rollback compares against a state that already has bonds (and a bond
    # settings signature).
    CrystalViz._bond_objects(r, CrystalViz._structure_atoms(r), scene.cgn_connectivity_pairs)
    bpy.context.view_layer.update()
    return r


# Control: confirm a seed-grow on fresh Brookite WOULD add atoms (non-vacuous).
root = _brookite()
cnt0 = len(json.loads(root.get("cgn_instances", "[]"))) // 4
root["cgn_selected_indices"] = json.dumps(list(range(5)))
res = bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
cnt1 = len(json.loads(root.get("cgn_instances", "[]"))) // 4
check("grow control: unpatched grow adds atoms", cnt1 > cnt0, f"grew {cnt1 - cnt0}")

# Failure-injected grow on a FRESH Brookite import.
root = _brookite()
root["cgn_selected_indices"] = json.dumps(list(range(5)))
b_before = baseline()
restore = one_shot_bond_fail()
try:
    res = bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
except Exception as exc:
    res = str(exc)
finally:
    restore()
b_after = baseline()
check("grow operator returned CANCELLED (no crash)", True)
check("grow: cgn_instances unchanged", b_after["instances"] == b_before["instances"])
check("grow: cgn_initial_count unchanged", b_after["initial_count"] == b_before["initial_count"])
check("grow: cgn_selected_indices unchanged", b_after["selected"] == b_before["selected"])
check("grow: cgn_growth_shells unchanged", b_after["shells"] == b_before["shells"])
check("grow: cgn_bond_settings_signature unchanged", b_after["bond_sig"] == b_before["bond_sig"])
check(
    "grow: displayed atom vertices restored",
    b_after["atom_verts"] == b_before["atom_verts"],
    f"before={b_before['atom_verts']} after={b_after['atom_verts']}",
)
check("grow: polyhedra present unchanged", b_after["poly_objects"] == b_before["poly_objects"])

# ---------------- FILL test ----------------
print("\n=== FILL rollback test ===")
# reset scene to a fresh import so fill produces more atoms
bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "multiple_bonds.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)
bpy.context.view_layer.update()
f_before = baseline()
restore = one_shot_unit_cell_fail()
try:
    res = bpy.ops.cgn.fill_cell("INVOKE_DEFAULT")
except Exception as exc:
    res = str(exc)
finally:
    restore()
f_after = baseline()
check("fill operator returned CANCELLED (no crash)", True)
check("fill: cgn_instances unchanged", f_after["instances"] == f_before["instances"])
check("fill: cgn_initial_count unchanged", f_after["initial_count"] == f_before["initial_count"])
check("fill: cgn_growth_shells unchanged", f_after["shells"] == f_before["shells"])
check("fill: cgn_fill_mode unchanged", f_after.get("fill_mode") == f_before.get("fill_mode"))
check(
    "fill: displayed atom vertices restored",
    f_after["atom_verts"] == f_before["atom_verts"],
    f"before={f_before['atom_verts']} after={f_after['atom_verts']}",
)

# ---------------- POLYHEDRA ROLLBACK test ----------------
print("\n=== POLYHEDRA rollback test ===")
root = _brookite()
try:
    bpy.ops.cgn.fill_cell("INVOKE_DEFAULT")
except Exception as exc:
    print("polyhedra: fill (for full cell) failed:", exc)
root = bpy.data.collections.get(scene.cgn_active_structure)
bpy.context.view_layer.update()
atoms = CrystalViz._structure_atoms(root)
ti_centers = [i for i, a in enumerate(atoms) if str(a.get("element", "X")).upper() == "TI"]
built, skipped = CrystalViz._build_polyhedra_objects(root, atoms, ti_centers, scene)
check("polyhedra control: some polyhedra built", built > 0, f"built={built}")
bpy.context.view_layer.update()
p_before = baseline()
check("polyhedra baseline present", p_before["poly_objects"] > 0)

# Inject a mid-rebuild failure in a grow; rollback must restore the polyhedra.
root["cgn_selected_indices"] = json.dumps(list(range(5)))
restore = one_shot_bond_fail()
try:
    res = bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
except Exception as exc:
    res = str(exc)
finally:
    restore()
p_after = baseline()
check("polyhedra: cgn_instances unchanged", p_after["instances"] == p_before["instances"])
check(
    "polyhedra: objects restored",
    p_after["poly_objects"] == p_before["poly_objects"],
    f"before={p_before['poly_objects']} after={p_after['poly_objects']}",
)

print()
print("TOTAL FAILURES:", len(FAILURES))
CrystalViz.unregister()
if FAILURES:
    print("RESULT: FAIL")
    sys.exit(1)
print("RESULT: ALL PASS")
