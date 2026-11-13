"""Regression: building polyhedra for one element must keep other elements'.

Reproduces: build Ag polyhedra, then build Cr polyhedra -> Ag must survive.

Run:
  blender --background --factory-startup --python test_polyhedra_multi_element.py
"""

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


def poly_objects(element):
    return [
        o
        for o in root.objects
        if o.get("cgn_role") == "polyhedron"
        and str(o.get("cgn_poly_element", "")).upper() == element
    ]


scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms0 = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms0)
bpy.ops.cgn.grow_further()

# Build Ag polyhedra.
scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()
ag_before = poly_objects("AG")
ag_names = {o.name for o in ag_before}
ag_centers = {o.name: o.get("cgn_poly_centers") for o in ag_before}
print(f"  Ag objects after Ag build: {len(ag_before)}")
check("Ag polyhedra built", len(ag_before) > 0, f"{len(ag_before)} objects")

# Build Cr polyhedra.
scene.cgn_polyhedron_settings.element = "CR"
bpy.ops.cgn.create_polyhedra()
ag_after = poly_objects("AG")
cr_after = poly_objects("CR")
print(f"  Ag objects after Cr build: {len(ag_after)}  Cr objects: {len(cr_after)}")
check("Cr polyhedra built", len(cr_after) > 0, f"{len(cr_after)} objects")
check(
    "Ag polyhedra survive a Cr build",
    {o.name for o in ag_after} == ag_names,
    f"Ag {len(ag_before)} -> {len(ag_after)}",
)
check(
    "Ag polyhedra geometry unchanged",
    all(o.get("cgn_poly_centers") == ag_centers.get(o.name) for o in ag_after),
)

# Building Ag again replaces only Ag, keeps Cr.
cr_names = {o.name for o in cr_after}
scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()
check("Cr polyhedra survive an Ag rebuild", {o.name for o in poly_objects("CR")} == cr_names)
check("still one Ag object per site", len(poly_objects("AG")) == len(ag_before))

CrystalViz.unregister()
check("unregister OK", True)
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
