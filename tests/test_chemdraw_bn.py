"""ChemDraw tori on a grown h-BN lattice.

Hexagonal boron nitride is the isoelectronic twin of graphite: the same
honeycomb layer, with B and N alternating instead of C and C.  Its sextet is
the three nitrogen lone pairs, so no B=N double bond is ever drawn; that also
means a six-ring has no complete arrangement of alternating single and double
bonds (three donors leave one and a half B=N), so the delocalisation torus is
the only honest depiction.

Every nitrogen of the layer belongs to three rings, which the indolizine rule
would read as a fused pi system and refuse to donate from.  A ring containing
boron is exempt, so each closed hexagon gets a torus from the first one that
closes -- the same picture graphite gives.

Run:
  blender --background --factory-startup --python test_chemdraw_bn.py
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

FAILURES = []
TORUS_EDGES = 48

# grow step -> (atoms, closed hexagons).  The hexagon count is the number of
# rings the grown graph closes, so it is also the number of expected tori.
GROWN = [(1, 12, 0), (2, 28, 4), (3, 48, 8), (4, 76, 20), (5, 108, 28), (6, 148, 48)]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def bond_objects(root):
    return [o for o in root.objects if o.get("cgn_role") == "bonds"]


def tori(root):
    return [o for o in bond_objects(root) if o.get("cgn_aromatic_torus")]


def torus_count(root):
    return sum(len(o.data.edges) for o in tori(root)) // TORUS_EDGES


def rebuild(name):
    scene = bpy.context.scene
    scene.cgn_connectivity_pairs.clear()
    scene.cgn_chemdraw = True
    bpy.ops.cgn.import_cif(filepath=_env.cif(name))
    root = bpy.data.collections.get(scene.cgn_active_structure)
    CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
    CrystalViz._rebuild_bonds(scene)
    return root


root = rebuild("BN.cif")
check("the asymmetric unit has four atoms", len(CrystalViz._structure_atoms(root)) == 4)
check("no torus before a ring closes", torus_count(root) == 0)

for step, atoms, hexagons in GROWN:
    bpy.ops.cgn.grow_further()
    bpy.context.view_layer.update()
    count = len(CrystalViz._structure_atoms(root))
    check(f"grow x{step} reaches {atoms} atoms", count == atoms, f"{count}")
    check(
        f"grow x{step} draws a torus on every closed hexagon",
        torus_count(root) == hexagons,
        f"{torus_count(root)} tori",
    )

check(
    "every torus is a full six-ring",
    sum(len(obj.data.edges) for obj in tori(root)) == GROWN[-1][2] * TORUS_EDGES,
)

print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "ALL PASS" if not FAILURES else "FAIL")
