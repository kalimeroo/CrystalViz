"""Regression: the grow append path classifies the union graph, not the shell.

Before this fix a grow appended bonds for the freshly grown atoms and inferred
multiple-bond orders and aromatic rings from *that partial graph alone*.  A bond
whose context was missing was valence-promoted (a lone half-molecule C-C became
a spurious C|C triple) and a ring closed by a mix of old and new bonds (pyrazine
N,N'-dioxide: half a molecule per asymmetric unit on an inversion centre) never
got its torus until a full rebuild ran.

``_bond_objects`` now reclassifies over the union of the bonds already on screen
plus the new bonds and redraws the whole set, so:

- no bond is drawn with an order the full-graph perception would not give it,
- a growth that closes a ring renders the aromatic torus immediately, and
- the ring bonds are demoted to single strokes once the torus is up.

This test forces the append path by building the asymmetric-unit bonds first:
with bond objects already present every grow must *append*, so the union redraw
is what paints the pyrazine torus.

Run:
  blender --background --factory-startup --python test_bond_grow_chemdraw_union.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import bpy
import CrystalViz

CrystalViz.register()

CIF = _env.cif("Pyrazine_dioxide.cif")
FAILURES = []
SUFFIX = re.compile(r"\.\d{3}$")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def bond_objects(root):
    return [o for o in root.all_objects if o.get("cgn_role") == "bonds"]


def tori(root):
    return [o for o in bond_objects(root) if o.get("cgn_aromatic_torus")]


def strokes(root, pair, order):
    """Number of drawn bond strokes for one element pair and bond order."""
    total = 0
    for obj in bond_objects(root):
        if obj.get("cgn_aromatic_torus"):
            continue
        if obj.get("cgn_pair") == pair and int(obj.get("cgn_bond_order", 1)) == order:
            total += len(obj.data.edges)
    return total


def report(root, label):
    torus = tori(root)
    torus_edges = sum(len(o.data.edges) for o in torus)
    keys = [
        (o.get("cgn_pair"), int(o.get("cgn_bond_order", 1)), o.get("cgn_bond_element"))
        for o in bond_objects(root)
        if not o.get("cgn_aromatic_torus")
    ]
    suffixes = [o.name for o in bond_objects(root) if SUFFIX.search(o.name)]
    duplicates = {key for key in keys if keys.count(key) > 1}
    segment_counts = {}
    for obj in bond_objects(root):
        if obj.get("cgn_aromatic_torus"):
            continue
        key = (obj.get("cgn_pair"), int(obj.get("cgn_bond_order", 1)))
        segment_counts[key] = segment_counts.get(key, 0) + len(obj.data.edges)
    print(f"  {label}: tori={len(torus)} edges={torus_edges} orders={sorted(segment_counts)}")
    check(f"{label}: at most one aromatic object", len(torus) <= 1, str(len(torus)))
    check(f"{label}: no duplicate bond groups", not duplicates, str(duplicates))
    check(f"{label}: no suffixed bond objects", not suffixes, str(suffixes))
    return len(torus) == 1


scene = bpy.context.scene
scene.cgn_connectivity_pairs.clear()
scene.cgn_chemdraw = True
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
# Building the asymmetric-unit bonds makes every grow take the append path.
CrystalViz._rebuild_bonds(scene)

check(
    "baseline: asymmetric unit draws no spurious C|C triple",
    strokes(root, "C|C", 3) == 0,
    f"{strokes(root, 'C|C', 3)} triples",
)

torus_up = False
for step in range(1, 5):
    bpy.ops.cgn.grow_further()
    bpy.context.view_layer.update()
    if report(root, f"grow x{step}"):
        torus_up = True
    # The core regression: a partial-graph append used to draw the lone
    # half-molecule C-C as a spurious triple; the union graph cannot.
    check(f"grow x{step}: draws no spurious C|C triple", strokes(root, "C|C", 3) == 0)
    check(f"grow x{step}: keeps N-O single", strokes(root, "N|O", 1) > 0)
    check(f"grow x{step}: draws no N=O", strokes(root, "N|O", 2) == 0)
    torus = tori(root)
    torus_edges = sum(len(o.data.edges) for o in torus)
    if torus:
        # Pyrazine N,N'-dioxide: one 6-membered ring with the two N-O single
        # strokes, so the torus carries 48 edges and the ring C-C bonds the
        # torus replaced must not stay doubled.
        check(
            f"grow x{step}: torus covers the full ring",
            torus_edges == 48,
            f"{torus_edges} edges",
        )
        check(f"grow x{step}: ring C-C is single", strokes(root, "C|C", 2) == 0)

check("a grow closes the ring and renders the aromatic torus", torus_up)

# The append-only guarantee still holds: a full rebuild with the same windows
# must not need any bond pair the appended grows lack.
drawn_pairs = {
    tuple(sorted(o.get("cgn_pair").split("|")))
    for o in bond_objects(root)
    if not o.get("cgn_aromatic_torus")
}
CrystalViz._rebuild_bonds(scene)
rebuilt_pairs = {
    tuple(sorted(o.get("cgn_pair").split("|")))
    for o in bond_objects(root)
    if not o.get("cgn_aromatic_torus")
}
missing = rebuilt_pairs - drawn_pairs
check("full rebuild needs no new bond pairs", not missing, str(missing))

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
