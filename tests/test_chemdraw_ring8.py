"""ChemDraw depiction of the eta8-cyclooctatetraenyl pi ligand.

ABIDUK (CSD): (eta8-COT)(eta5-Cp*) samarium.  Every COT carbon is bonded to
its two ring neighbours, one hydrogen and the metal, so no Kekule double bond
can be placed; the ring is the aromatic 10 pi COT2- dianion, and IUPAC
GR-1.7.2 requires the delocalisation curve -- the torus -- just as for the
cyclopentadienyl rings of a metallocene.  The eight-membered ring is
enumerated by ``find_rings`` (ring sizes 5-8), the haptic path recognises the
common Sm centre, and the same aromatic object that carries the Cp* torus
picks up the COT ring as a second closed polyline.

Run:
  blender --background --factory-startup --python test_chemdraw_ring8.py
"""

import json
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


def tori(root):
    return [obj for obj in bond_objects(root) if obj.get("cgn_aromatic_torus")]


scene = bpy.context.scene
scene.cgn_connectivity_pairs.clear()
scene.cgn_chemdraw = True
scene.cgn_delocalised_bonds = True

RING8_CIF = _env.cif("8ring.cif")

if not os.path.exists(RING8_CIF):
    print("[SKIP] CIF/8ring.cif is missing; the eta8-COT checks did not run")
else:
    bpy.ops.cgn.import_cif(filepath=RING8_CIF)
    root = bpy.data.collections.get(scene.cgn_active_structure)
    CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
    CrystalViz._rebuild_bonds(scene)

    aromatic_objects = tori(root)
    check(
        "one aromatic object carries both tori",
        len(aromatic_objects) == 1,
        str(len(aromatic_objects)),
    )
    check(
        "both the Cp* and the COT torus are drawn (2 rings x 48 segments)",
        len(aromatic_objects[0].data.edges) == 96 if aromatic_objects else False,
        str(len(aromatic_objects[0].data.edges)) if aromatic_objects else "no torus",
    )
    if aromatic_objects:
        entries = json.loads(aromatic_objects[0].get("cgn_aromatic_rings", "[]"))
        check("two aromatic rings recorded", len(entries) == 2, str(len(entries)))

print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "ALL PASS" if not FAILURES else "FAIL")
