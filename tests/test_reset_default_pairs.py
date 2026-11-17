"""Regression: "Reset Default" must restore the import-time connectivity windows.

After the user edits a pair's minimum/maximum distance, clicking Reset Default
(``CGN_OT_rebuild_connectivity``) repopulates ``scene.cgn_connectivity_pairs``.
The reset recomputed the defaults from covalent radii alone, but import computes
them from the CIF's declared formal charges plus Shannon ionic radii for ionic
cation-anion pairs (e.g. Cu-S and Fe-S in Chalco.cif).  So reset produced values
that differed from the initial ones.  This pins that reset now reproduces the
exact import-time windows.

Run headless:
  blender --background --factory-startup --python test_reset_default_pairs.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import bpy
import CrystalViz

CrystalViz.register()

CIF = _env.CIF_DIR
FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def pair_windows():
    return {
        tuple(sorted((pair.element_a.upper(), pair.element_b.upper()))): tuple(
            (pair.min_distance, pair.max_distance)
        )
        for pair in bpy.context.scene.cgn_connectivity_pairs
        if pair.enabled
    }


def import_structure(name):
    scene = bpy.context.scene
    CrystalViz._cancel_all_debounce()
    scene.cgn_model = "BALL_STICK"
    scene.cgn_chemdraw = False
    scene.cgn_connectivity_pairs.clear()
    bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, name))
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    return bpy.data.collections.get(scene.cgn_active_structure)


def import_brookite():
    return import_structure("Brookite.cif")


def reset_default():
    bpy.ops.cgn.rebuild_connectivity("INVOKE_DEFAULT")


def close(a, b, eps=1e-3):
    return abs(a - b) <= eps


# ---- A: Reset Default restores the import-time windows ----------------------
print("\n=== A: reset restores the import-time windows ===")
root = import_structure("Chalco.cif")
import_windows = pair_windows()
check(
    "A control: Chalco has ionic Cu-S and Fe-S pairs",
    ("CU", "S") in import_windows and ("FE", "S") in import_windows,
    f"pairs={sorted(import_windows)}",
)
print("A: import windows:", {k: tuple(round(v, 2) for v in w) for k, w in import_windows.items()})

# The user edits the windows (e.g. grows the Cu-S range).
for pair in bpy.context.scene.cgn_connectivity_pairs:
    if tuple(sorted((pair.element_a.upper(), pair.element_b.upper()))) == ("CU", "S"):
        pair.min_distance = 0.5
        pair.max_distance = 4.0
    elif tuple(sorted((pair.element_a.upper(), pair.element_b.upper()))) == ("S", "S"):
        pair.min_distance = 0.1
        pair.max_distance = 1.0
bpy.context.view_layer.update()

reset_default()
reset_windows = pair_windows()
all_close = set(reset_windows) == set(import_windows) and all(
    close(reset_windows[key][0], import_windows[key][0])
    and close(reset_windows[key][1], import_windows[key][1])
    for key in import_windows
)
check(
    "A: reset reproduces every import-time window",
    all_close,
    f"got={ {k: tuple(round(v, 2) for v in w) for k, w in reset_windows.items()} } "
    f"expect={ {k: tuple(round(v, 2) for v in w) for k, w in import_windows.items()} }",
)

# ---- B: persisted chemistry drives reset's defaults -------------------------
print("\n=== B: persisted chemistry drives reset's default windows ===")
import json

charges = json.loads(root.get("cgn_atom_charges", "{}"))
ionic = json.loads(root.get("cgn_ionic_radii", "{}"))
metals = set(json.loads(root.get("cgn_metal_elements", "[]")))
check(
    "B: charges/ionic radii/metals were persisted on the root",
    bool(charges) and bool(ionic) and bool(metals),
    f"charges={sorted(charges)} metals={sorted(metals)}",
)
atoms = CrystalViz._structure_atoms(root)
radii = {str(a.get("element", "X")).upper(): float(a.get("covalent_radius", 0.75)) for a in atoms}
expected = {}
for element_a in sorted(set(radii)):
    for element_b in sorted(set(radii))[sorted(set(radii)).index(element_a) :]:
        expected[(element_a, element_b)] = CrystalViz._default_pair_window(
            element_a, element_b, radii, charges, metals, ionic
        )
CrystalViz._default_connectivity_pairs(bpy.context.scene, atoms, root=root)
via_reset = pair_windows()
for key, (low, high) in expected.items():
    got = via_reset.get(key)
    check(
        f"B: {key[0]}-{key[1]} uses the ionic default",
        got is not None and close(got[0], low) and close(got[1], high),
        f"got={got} expect={(low, high)}",
    )

# The whole point: the ionic default disagrees with the covalent-only default,
# so the old (chemistry-less) reset was wrong for this structure.
covalent = CrystalViz._default_pair_window("CU", "S", radii, {}, set(), {})
check(
    "B: ionic differs from covalent for Cu-S (the bug's root)",
    not (
        close(covalent[0], expected[("CU", "S")][0])
        and close(covalent[1], expected[("CU", "S")][1])
    ),
    f"covalent={tuple(round(v, 2) for v in covalent)} "
    f"ionic={tuple(round(v, 2) for v in expected[('CU', 'S')])}",
)

# ---- C: a root without persisted chemistry falls back to covalent -----------
print("\n=== C: a chemistry-less root stays covalent ===")
bare = import_structure("Chalco.cif")
for prop in ("cgn_atom_charges", "cgn_ionic_radii", "cgn_metal_elements"):
    if prop in bare:
        del bare[prop]
CrystalViz._default_connectivity_pairs(
    bpy.context.scene, CrystalViz._structure_atoms(bare), root=bare
)
covalent_windows = pair_windows()
exp_covalent = {}
radii_c = {
    str(a.get("element", "X")).upper(): float(a.get("covalent_radius", 0.75))
    for a in CrystalViz._structure_atoms(bare)
}
for element_a in sorted(set(radii_c)):
    for element_b in sorted(set(radii_c))[sorted(set(radii_c)).index(element_a) :]:
        exp_covalent[(element_a, element_b)] = CrystalViz._default_pair_window(
            element_a, element_b, radii_c, {}, set(), {}
        )
all_cov = set(covalent_windows) == set(exp_covalent) and all(
    close(covalent_windows[key][0], exp_covalent[key][0])
    and close(covalent_windows[key][1], exp_covalent[key][1])
    for key in exp_covalent
)
check("C: a chemistry-less root uses the covalent fallback", all_cov, f"got={covalent_windows}")

# ---- D: deleting atoms must not drop pair windows --------------------------
# User report: import Ag2Cr2O7, delete every atom except Ag, then Reset Default
# -> the Ag-Cr / Ag-O / Cr-O / O-O rows vanished because reset rebuilt the list
# from the *displayed* atoms only.  Reset must read the persistent P1 site
# table, which still holds every element after instances are deleted.
print("\n=== D: reset keeps every pair after atoms are deleted ===")
scene = bpy.context.scene
CrystalViz._cancel_all_debounce()
scene.cgn_connectivity_pairs.clear()
bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "Ag2Cr2O7.cif"))
root_d = bpy.data.collections.get(scene.cgn_active_structure)
import_windows_d = pair_windows()
check(
    "D control: Ag2Cr2O7 has several element pairs",
    len(import_windows_d) >= 3,
    f"pairs={sorted(import_windows_d)}",
)

sites_d, instances_d, _cell_d, _initial_d = CrystalViz._structure_tables(root_d)
ag_sites = {i for i, s in enumerate(sites_d) if str(s.get("element", "X")).upper() == "AG"}
ag_instances = []
for _off in range(0, len(instances_d) - len(instances_d) % 4, 4):
    if int(instances_d[_off]) in ag_sites:
        ag_instances.extend(instances_d[_off : _off + 4])
CrystalViz._set_instances(root_d, ag_instances)
remaining = CrystalViz._structure_atoms(root_d)
check(
    "D: only Ag atoms remain",
    {str(a.get("element", "X")).upper() for a in remaining} == {"AG"},
    f"elements={sorted({str(a.get('element', 'X')).upper() for a in remaining})}",
)

reset_default()
after_windows = pair_windows()
check(
    "D: reset keeps every pair after atoms were deleted",
    set(after_windows) == set(import_windows_d),
    f"got={sorted(after_windows)} expect={sorted(import_windows_d)}",
)
check(
    "D: reset restores the import-time values for those pairs",
    all(
        close(after_windows[key][0], import_windows_d[key][0])
        and close(after_windows[key][1], import_windows_d[key][1])
        for key in import_windows_d
    ),
    f"got={ {k: tuple(round(v, 2) for v in w) for k, w in after_windows.items()} }",
)

# ---- E: Update Bonds applies the current windows ---------------------------
# The new button rebuilds the displayed atoms' bonds from the current min/max,
# so edited distances can be applied without growing.
print("\n=== E: Update Bonds rebuilds from the current windows ===")


def bond_object_count(root):
    return sum(1 for o in root.objects if o.get("cgn_role") == "bonds")


root_e = import_brookite()
check("E: import leaves no bonds", bond_object_count(root_e) == 0, f"{bond_object_count(root_e)}")
bpy.ops.cgn.update_connectivity("INVOKE_DEFAULT")
check(
    "E: Update Bonds creates bonds from the default windows",
    bond_object_count(root_e) > 0,
    f"{bond_object_count(root_e)}",
)
for pair in bpy.context.scene.cgn_connectivity_pairs:
    pair.enabled = False
bpy.context.view_layer.update()
bpy.ops.cgn.update_connectivity("INVOKE_DEFAULT")
check(
    "E: disabling every pair then updating clears the bonds",
    bond_object_count(root_e) == 0,
    f"{bond_object_count(root_e)}",
)

# ---- F: metal-metal pairs are capped in compounds, not in elemental metals --
# COD Brookite declares no oxidation numbers, so nothing else would cap Ti-Ti:
# the plain covalent window (1.2 x sum, +0.4 slack) reaches 3.6 A and bonds
# every Ti to its edge-sharing neighbours, turning the TiO6 polyhedra into
# 11-vertex hulls.  Elemental copper must keep the wide window, because there
# the close-packed Cu-Cu contacts are the bonds.
print("\n=== F: metal-metal cap in compounds ===")


def radii_of(root):
    return {
        str(atom.get("element", "X")).upper(): float(atom.get("covalent_radius", 0.75))
        for atom in CrystalViz._structure_atoms(root)
    }


def plain_window(radii, element):
    reference = radii[element] * 2.0
    return min(CrystalViz._BOND_MAX_SCALE * reference, reference + CrystalViz._BOND_MAX_SLACK)


root_f = import_brookite()
windows_f = pair_windows()
radii_f = radii_of(root_f)
ti_ti = windows_f.get(("TI", "TI"))
check(
    "F: compound Ti-Ti is capped at 0.85 x the covalent sum",
    ti_ti is not None and close(ti_ti[1], CrystalViz._BOND_METAL_CAP_SCALE * radii_f["TI"] * 2.0),
    f"got={ti_ti} plain={plain_window(radii_f, 'TI'):.2f}",
)
check(
    "F: the cap is tighter than the plain covalent window",
    ti_ti is not None and ti_ti[1] < plain_window(radii_f, "TI"),
    f"got={ti_ti}",
)

root_cu = import_structure("Cu.cif")
windows_cu = pair_windows()
radii_cu = radii_of(root_cu)
cu_cu = windows_cu.get(("CU", "CU"))
check(
    "F: elemental copper keeps the wide Cu-Cu window",
    cu_cu is not None and close(cu_cu[1], plain_window(radii_cu, "CU")),
    f"got={cu_cu} expect={plain_window(radii_cu, 'CU'):.2f}",
)

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
