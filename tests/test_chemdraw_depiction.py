"""ChemDraw depiction of the reference organic structure.

Runs on ``CIF/multiple_bonds.cif`` (COD 4029335, C21 H17 N O4): a molecule with
four fused rings, two of them carrying an exocyclic C=O.  The expected Lewis
structure is the reference Kekulé one:

  * 7 C=C doubles (C1=C16, C2=C3, C4=C5, C6=C15, C8=C13, C9=C10, C11=C12)
  * 3 C=O doubles (C7=O2, C14=O1, C17=O3)
  * 1 aromatic ring: the benzene ring C8..C13.  The two rings bearing an
    exocyclic C=O are not aromatic, the ring whose only "double bond" leaves it
    is not either, and the pyrrole-type five-ring N1/C1/C6/C15/C16 is not: it
    shares its nitrogen with the six-ring N1/C1/C2/C3/C4/C5 (an annulated
    indolizine), so N1 is a bridgehead and its lone pair belongs to neither
    ring — both are depicted Kekulé.

Three of the seven C=C bonds are ring bonds of the aromatic benzene ring, so
chemdraw draws them as plain ring strokes: the other 4 C=C and the 3 C=O
survive as explicit multiple-bond objects.

Run:
  blender --background --factory-startup --python test_chemdraw_depiction.py
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

CIF = _env.cif("multiple_bonds.cif")
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


scene = bpy.context.scene
scene.cgn_connectivity_pairs.clear()
scene.cgn_chemdraw = True

bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
CrystalViz._rebuild_bonds(scene)

# The CIF's _geom_bond table gives 21 C-C, 17 C-H, 3 C-N, 2 C-O and 3 C=O
# contacts; the connectivity windows must reproduce exactly that graph.
check(
    "C-C bonds found",
    strokes(root, "C|C", 1) + strokes(root, "C|C", 2) // 2 == 21,
    f"{strokes(root, 'C|C', 1)} single strokes, {strokes(root, 'C|C', 2)} double strokes",
)
check("C-H bonds found", strokes(root, "C|H", 1) == 17, str(strokes(root, "C|H", 1)))
check("C-N bonds found", strokes(root, "C|N", 1) == 3, str(strokes(root, "C|N", 1)))

# 4 non-aromatic C=C doubles x 2 strokes, 3 C=O doubles x 2 strokes.
check("4 drawn C=C doubles", strokes(root, "C|C", 2) == 8, str(strokes(root, "C|C", 2)))
check("3 drawn C=O doubles", strokes(root, "C|O", 2) == 6, str(strokes(root, "C|O", 2)))

# The 14 remaining C-C single bonds plus the 3 ring doubles that chemdraw
# suppresses inside the aromatic benzene ring.
check(
    "aromatic ring doubles are drawn as ring strokes",
    strokes(root, "C|C", 1) == 17,
    str(strokes(root, "C|C", 1)),
)

check(
    "no triple bond is invented",
    not [obj for obj in bond_objects(root) if int(obj.get("cgn_bond_order", 1)) > 2],
)

tori = [obj for obj in bond_objects(root) if obj.get("cgn_aromatic_torus")]
check("one grouped aromatic object", len(tori) == 1, str(len(tori)))
check(
    "one aromatic ring (48 segments)",
    len(tori[0].data.edges) == 48 if tori else False,
    str(len(tori[0].data.edges)) if tori else "no torus",
)

print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "ALL PASS" if not FAILURES else "FAIL")
