"""ChemDraw depiction of six-membered sulfur rings.

A conjugated six-ring sulfur (thiopyrylium, BIZTAF) draws a genuine S=C bond
inside its ring, exactly as pyridine draws N=C, so the ring has three
alternating doubles and no lone-pair share of its sextet.  A sulfoxide sulfur
(thiabenzene 1-oxide, ATBOXM) also carries a terminal S=O, drawn as a real
double bond that does not break the ring conjugation.

``S2.cif`` is the thiopyrylium cation: all four rings (the sulfur ring and its
three phenyls) are flat, so they are drawn as tori.  ``S1.cif`` is the
sulfoxide: its sulfur ring is puckered (a boat, out-of-plane deviation above
the ``_AROMATIC_PLANARITY_MAX`` gate), so it keeps the localized Kekule
depiction with the S=C and S=O doubles visible instead of a torus.

Run:
  blender --background --factory-startup --python test_chemdraw_sulfur.py
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def bond_objects(root):
    return [o for o in root.objects if o.get("cgn_role") == "bonds"]


def strokes(root, pair, order):
    """Number of drawn bond strokes for one element pair and bond order."""
    total = 0
    for obj in bond_objects(root):
        if obj.get("cgn_aromatic_torus"):
            continue
        if obj.get("cgn_pair") == pair and int(obj.get("cgn_bond_order", 1)) == order:
            total += len(obj.data.edges)
    return total


def tori(root):
    return [obj for obj in bond_objects(root) if obj.get("cgn_aromatic_torus")]


def rebuild(name):
    scene = bpy.context.scene
    scene.cgn_connectivity_pairs.clear()
    scene.cgn_chemdraw = True
    bpy.ops.cgn.import_cif(filepath=_env.cif(name))
    root = bpy.data.collections.get(scene.cgn_active_structure)
    CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
    CrystalViz._rebuild_bonds(scene)
    return root


# ------------------------------------------------ ATBOXM: thiabenzene 1-oxide
# S1 is four-coordinate (two ring carbons, a methyl and the terminal O).  The
# sulfur takes two extra orders: S=O and the ring S=C, leaving the ring with
# three alternating doubles (S=C, C1=C2, C3=C4).  The ring is a boat, so it is
# drawn Kekule rather than as a torus.
S1_CIF = _env.cif("S1.cif")


def check_sulfoxide():
    root = rebuild("S1.cif")

    check(
        "S1 draws the S=O as a double bond",
        strokes(root, "O|S", 2) == 2,
        f"{strokes(root, 'O|S', 2)} O|S double strokes",
    )
    check(
        "S1 leaves no single S-O",
        strokes(root, "O|S", 1) == 0,
        f"{strokes(root, 'O|S', 1)} O|S single strokes",
    )
    check(
        "S1 draws the ring S=C double bond",
        strokes(root, "C|S", 2) == 2,
        f"{strokes(root, 'C|S', 2)} C|S double strokes",
    )
    check(
        "S1 draws the two ring C=C doubles",
        strokes(root, "C|C", 2) == 4,
        f"{strokes(root, 'C|C', 2)} C|C double strokes",
    )
    check(
        "S1 keeps the ketone C=O",
        strokes(root, "C|O", 2) == 2,
        f"{strokes(root, 'C|O', 2)} C|O double strokes",
    )
    check(
        "S1 puckered ring is not given a torus",
        len(tori(root)) == 0,
        f"{len(tori(root))} tori",
    )


if os.path.exists(S1_CIF):
    check_sulfoxide()
else:
    print("[SKIP] CIF/S1.cif is missing; the sulfoxide checks did not run")


# ------------------------------------------------------ BIZTAF: thiopyrylium
# S2 is a thiopyrylium cation with three phenyl substituents.  The ring sulfur
# is two-coordinate and carries no oxygen: one ring bond is a genuine S=C
# double, so the ring is a 6 pi aromatic ring with no lone-pair share.  All
# four rings are flat, so they are drawn as tori (4 x 48 segments).
S2_CIF = _env.cif("S2.cif")


def check_thiopyrylium():
    root = rebuild("S2.cif")

    aromatic_objects = tori(root)
    check(
        "S2 draws one aromatic object",
        len(aromatic_objects) == 1,
        str(len(aromatic_objects)),
    )
    check(
        "S2 draws four tori (thiopyrylium + 3 phenyls, 192 segments)",
        len(aromatic_objects[0].data.edges) == 192 if aromatic_objects else False,
        str(len(aromatic_objects[0].data.edges)) if aromatic_objects else "no torus",
    )
    check(
        "S2 suppresses the S=C inside the torus",
        strokes(root, "C|S", 2) == 0,
        f"{strokes(root, 'C|S', 2)} C|S double strokes",
    )
    check(
        "S2 renders both ring S-C bonds as torus singles",
        strokes(root, "C|S", 1) == 2,
        f"{strokes(root, 'C|S', 1)} C|S single strokes",
    )
    check(
        "S2 suppresses every ring C=C inside the tori",
        strokes(root, "C|C", 2) == 0,
        f"{strokes(root, 'C|C', 2)} C|C double strokes",
    )


if os.path.exists(S2_CIF):
    check_thiopyrylium()
else:
    print("[SKIP] CIF/S2.cif is missing; the thiopyrylium checks did not run")

print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "ALL PASS" if not FAILURES else "FAIL")
