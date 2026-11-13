"""Regression: Fill ASU → Grow must bond the initial atoms even when no shell is
added.

For small finite molecules (metal.cif, test.cif) growing after a Fill ASU adds no
new atoms — they do not pack again within the connectivity window — and the grow
operator used to return early at "no additional atoms matched" without ever
calling the bond builder.  The structure stayed bond-less even though
``_bond_objects`` can bond those ASU atoms.

The grow path now builds the full bond set when it starts from a bond-less
structure (right after a fill/import) even when nothing was added.  A later grow
that still finds nothing must NOT tear down those bonds (append-only guarantee).

Run headless:
  blender --background --factory-startup --python test_grow_fill_noadd_bonds.py
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


def bond_objects(root):
    return [obj for obj in root.all_objects if obj.get("cgn_role") == "bonds"]


def bond_edges(root):
    return sum(
        len(obj.data.edges)
        for obj in bond_objects(root)
        if obj.type == "MESH" and not obj.get("cgn_aromatic_torus")
    )


def atom_count(root):
    return int(root.get("cgn_atom_count", -1))


def import_and_fill_asu(name):
    scene = bpy.context.scene
    CrystalViz._cancel_all_debounce()
    scene.cgn_model = "BALL_STICK"
    scene.cgn_chemdraw = False
    scene.cgn_connectivity_pairs.clear()
    bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, name))
    bpy.ops.cgn.fill_asu("INVOKE_DEFAULT")
    bpy.context.view_layer.update()
    return bpy.data.collections.get(scene.cgn_active_structure)


def scenario(name, label):
    print(f"\n=== {label}: {name} ===")
    root = import_and_fill_asu(name)
    initial = atom_count(root)
    check(
        "control: fill left no bonds",
        not bond_objects(root),
        f"atoms={initial} bonds={len(bond_objects(root))}",
    )
    bpy.ops.cgn.grow_further()
    bpy.context.view_layer.update()
    root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
    check(
        "grow did not add atoms (finite molecule)",
        atom_count(root) == initial,
        f"{initial} -> {atom_count(root)}",
    )
    check(
        "grow bonded the initial atoms",
        len(bond_objects(root)) > 0 and bond_edges(root) > 0,
        f"bonds={len(bond_objects(root))} edges={bond_edges(root)}",
    )
    # A second grow that also adds nothing must NOT clear those bonds.
    before_edges = bond_edges(root)
    bpy.ops.cgn.grow_further()
    bpy.context.view_layer.update()
    root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
    check(
        "a second grow preserves the bonds",
        len(bond_objects(root)) > 0 and bond_edges(root) == before_edges,
        f"{before_edges} -> {bond_edges(root)}",
    )
    return root


# ---- A: metal.cif (organometallic, Mo/Ni hints + metals persisted) ----------
scenario("metal.cif", "A")

# ---- B: test.cif (organic molecule) ----------------------------------------
scenario("test.cif", "B")

# ---- C: a molecule whose grow DOES add atoms still works (sanity) ----------
print("\n=== C: a filling structure still gains atoms and bonds (sanity) ===")
root = import_and_fill_asu("Brookite.cif")
initial = atom_count(root)
check("C control: Brookite fill left no bonds", not bond_objects(root), f"atoms={initial}")
bpy.ops.cgn.grow_further()
bpy.context.view_layer.update()
root = bpy.data.collections.get(bpy.context.scene.cgn_active_structure)
check(
    "C: Brookite grow added atoms", atom_count(root) > initial, f"{initial} -> {atom_count(root)}"
)
check(
    "C: Brookite grow bonded the structure",
    len(bond_objects(root)) > 0,
    f"bonds={len(bond_objects(root))}",
)

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
