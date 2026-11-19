"""ChemDraw tori on a grown graphite lattice.

A graphite patch is grown shell by shell, so its edge always ends in carbons
that are bonded to nothing else.  The Kekule structure doubles some of those
dangling carbons, and that used to read as an exocyclic C=CH2 and veto the ring.
Every closed hexagon must now get a delocalisation torus, from the first one
that closes; the asymmetric unit itself is one C-C bond and carries none.

The exemption is for truncated lattices only: a molecule with an exocyclic C=O
still keeps its veto and its single torus.

Run:
  blender --background --factory-startup --python test_chemdraw_graphite.py
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
GROWN = [(1, 6, 0), (2, 14, 2), (3, 24, 4), (4, 38, 10), (5, 54, 14)]


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


root = rebuild("Graphite.cif")
check("the asymmetric unit is one C-C bond", len(CrystalViz._structure_atoms(root)) == 2)
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

# The dangling-carbon exemption must not weaken the molecular rule: the
# exocyclic C=O of multiple_bonds.cif still vetoes its ring.
root = rebuild("multiple_bonds.cif")
check(
    "a molecular exocyclic C=O keeps its veto",
    torus_count(root) == 1,
    f"{torus_count(root)} tori",
)

print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "ALL PASS" if not FAILURES else "FAIL")
