"""Regression: growing a disorder twin must not drop aromatic rings.

CIF/ABILIL01.cif (CSD, CCDC 2530362; a local fixture like S1.cif/S2.cif)
deposits *both* halves of an inversion-disordered molecule, so the P1
expansion holds 848 same-element twin pairs.  Growth used to pull the twin
hydrogen of an already-displayed ring atom into the scene: the ring carbon
then had four non-metal neighbours and its benzo ring was no longer
enumerated, so Fill -> Asymmetric Unit + Grow -> ChemDraw drew two phenyl
rings with Kekule double bonds instead of aromatic tori.
_canonical_base_indices now merges the twins (same element, position within
0.01 A) so grow never adds them and every ring keeps its torus.

Run:
  blender --background --factory-startup --python test_bond_grow_disorder_twin_aromatic.py
"""

import json
import sys

import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

CIF = _env.cif("ABILIL01.cif")
# The ASU holds 6 carbazole molecules: 12 benzo + the member C6 rings of the
# straddling images enumerate to 18 aromatic six-rings in all.  The 6 pyrrole
# (C4N) rings are also aromatic but keep a unique alternating Kekule
# arrangement (GR-6.3, like pyrrole/furan/azoles), so only the 18 C6 rings get
# a torus.  The two benzo rings whose molecule straddles the cell boundary used
# to lose their torus after grow (what the user saw: phenyl rings with double
# bonds instead of tori).
EXPECTED_TORI = 18
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def aromatic_ring_count(root):
    total = 0
    for obj in root.objects:
        if obj.get("cgn_role") != "bonds" or not obj.get("cgn_aromatic_torus"):
            continue
        try:
            total += len(json.loads(obj.get("cgn_aromatic_rings", "[]")))
        except (TypeError, ValueError):
            continue
    return total


bpy.ops.cgn.import_cif(filepath=CIF)
scene = bpy.context.scene
scene.cgn_chemdraw = True
root = bpy.data.collections.get(scene.cgn_active_structure)

# Reproduce the reported sequence: Fill -> Asymmetric Unit, then ChemDraw.
bpy.ops.cgn.fill_asu()
CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
CrystalViz._rebuild_bonds(scene)
asu_rings = aromatic_ring_count(root)
print(f"  after fill: {asu_rings} aromatic tori (expect {EXPECTED_TORI})")
check("fill draws every C6 ring as a torus", asu_rings == EXPECTED_TORI, f"{asu_rings}")

bpy.ops.cgn.grow_further(all_atoms=True)
CrystalViz._rebuild_bonds(scene)
grown_rings = aromatic_ring_count(root)
print(f"  after grow: {grown_rings} aromatic tori (expect {EXPECTED_TORI})")
check(
    "grow keeps every aromatic torus (no twin-H Kekule loss)",
    grown_rings == asu_rings == EXPECTED_TORI,
    f"{asu_rings} -> {grown_rings}",
)

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
