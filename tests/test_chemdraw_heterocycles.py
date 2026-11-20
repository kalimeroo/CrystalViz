"""ChemDraw depiction of selenium, tellurium, arsenic, antimony and boron rings.

A benzo-chalcogenopyrylium (selenochromenylium, tellurochromenylium) keeps the
localized carbocation form its geometry shows: the valence repair leaves the C8
fusion carbon without a double bond, so the chalcogen takes no ring double and
the fused bicycle is drawn Kekule (four C=C strokes, no torus).  A plain
six-membered selenopyrylium/telluropyrylium still takes a genuine C=Se/C=Te
double bond, while a five-membered selenophene/tellurophene keeps its lone pair
and is drawn Kekule.

Arsenic and antimony mirror phosphorus: the group 15 valence ladder gives the
arsabenzene ring its As=C, so that ring is a torus, while a five-membered
arsole/stibole keeps its lone pair.  The diazastibolene is a genuine 8 pi ring,
so it stays non-aromatic and keeps its single C=C.

A three-connected boron has an empty p orbital: borazine is aromatic with no
B=N double bond at all (the sextet is the three nitrogen lone pairs), and a
six-ring with three donating nitrogens has no complete arrangement of
alternating single and double bonds (it would need one and a half B=N), so it
keeps its delocalisation torus -- just like the isoelectronic h-BN sheet.

Run:
  blender --background --factory-startup --python test_chemdraw_heterocycles.py
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


def torus_edges(root):
    return sum(len(obj.data.edges) for obj in tori(root))


def rebuild(name):
    scene = bpy.context.scene
    scene.cgn_connectivity_pairs.clear()
    scene.cgn_chemdraw = True
    bpy.ops.cgn.import_cif(filepath=_env.cif(name))
    root = bpy.data.collections.get(scene.cgn_active_structure)
    CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
    CrystalViz._rebuild_bonds(scene)
    return root


# ------------------------------------------- COD 7243860 / 7243859 (group 16)
# Both are a seleno-/tellurochromenylium (the chalcogen analogue of
# chromenylium, fused to a benzene ring) with a BF4 counter-ion.  The valence
# repair leaves the C8 fusion carbon without a double bond -- the ring-system
# carbocation -- so the chalcogen keeps its lone pair, takes no ring double and
# the whole bicycle is drawn Kekule: four C=C doubles and no torus.


def check_chromenylium(name, chalcogen):
    root = rebuild(name)
    pair = f"C|{chalcogen}"

    check(f"{name} has no aromatic torus", len(tori(root)) == 0, str(len(tori(root))))
    check(
        f"{name} keeps the chalcogen single (no C={chalcogen})",
        strokes(root, pair, 2) == 0,
        f"{strokes(root, pair, 2)} {pair} double strokes",
    )
    check(
        f"{name} draws both ring C-{chalcogen} bonds as singles",
        strokes(root, pair, 1) == 2,
        f"{strokes(root, pair, 1)} {pair} single strokes",
    )
    check(
        f"{name} draws the four localized C=C doubles",
        strokes(root, "C|C", 2) == 8,
        f"{strokes(root, 'C|C', 2)} C|C double strokes",
    )


# ----------------------------------------------- COD 1553554 / 1558536 (5-ring)
# Selenophene and tellurophene: the chalcogen keeps its lone pair (the ring has
# only one Kekule arrangement), so the five-ring is drawn Kekule with its two
# C=C strokes and the chalcogen stays single.  The two phenyl substituents are
# still tori.


def check_selenophene(name, chalcogen):
    root = rebuild(name)
    pair = f"C|{chalcogen}"

    check(
        f"{name} draws its two phenyls as tori (96 segments)",
        torus_edges(root) == 96,
        str(torus_edges(root)),
    )
    check(
        f"{name} keeps the chalcogen lone pair (no C={chalcogen})",
        strokes(root, pair, 2) == 0,
        f"{strokes(root, pair, 2)} {pair} double strokes",
    )
    check(
        f"{name} draws both ring C-{chalcogen} bonds as singles",
        strokes(root, pair, 1) == 2,
        f"{strokes(root, pair, 1)} {pair} single strokes",
    )
    check(
        f"{name} draws the two ring C=C doubles",
        strokes(root, "C|C", 2) == 4,
        f"{strokes(root, 'C|C', 2)} C|C double strokes",
    )


# ---------------------------------------------- COD 1573224 / 1555319 (group 15)
# Arsabenzene (fused to two thiophenes): the As=C makes the arsenic ring a
# 6 pi torus; the two thiophene rings keep their lone pairs and one localized
# double each.  Arsole is a 1,2,5-triphenylarsole: its five-ring is Kekule and
# the three phenyls are tori.


def check_arsabenzene():
    root = rebuild("Arsabenzene.cif")

    check(
        "Arsabenzene draws the arsenic ring as a torus (48 segments)",
        torus_edges(root) == 48,
        str(torus_edges(root)),
    )
    check(
        "Arsabenzene suppresses the As=C inside the torus",
        strokes(root, "AS|C", 2) == 0,
        f"{strokes(root, 'AS|C', 2)} AS|C double strokes",
    )
    check(
        "Arsabenzene renders both ring As-C bonds as singles",
        strokes(root, "AS|C", 1) == 2,
        f"{strokes(root, 'AS|C', 1)} AS|C single strokes",
    )
    check(
        "Arsabenzene keeps the two thiophene sulfurs single",
        strokes(root, "C|S", 1) == 4 and strokes(root, "C|S", 2) == 0,
        f"{strokes(root, 'C|S', 1)} single, {strokes(root, 'C|S', 2)} double",
    )
    check(
        "Arsabenzene draws one localized double per thiophene",
        strokes(root, "C|C", 2) == 4,
        f"{strokes(root, 'C|C', 2)} C|C double strokes",
    )


def check_arsole():
    root = rebuild("Arsole.cif")

    check(
        "Arsole draws its three phenyls as tori (144 segments)",
        torus_edges(root) == 144,
        str(torus_edges(root)),
    )
    check(
        "Arsole keeps the arsenic lone pair (no As=C)",
        strokes(root, "AS|C", 2) == 0,
        f"{strokes(root, 'AS|C', 2)} AS|C double strokes",
    )
    check(
        "Arsole draws all three As-C bonds as singles",
        strokes(root, "AS|C", 1) == 3,
        f"{strokes(root, 'AS|C', 1)} AS|C single strokes",
    )
    check(
        "Arsole draws the two ring C=C doubles",
        strokes(root, "C|C", 2) == 4,
        f"{strokes(root, 'C|C', 2)} C|C double strokes",
    )


# --------------------------------------------------------- COD 7111438 (Sb)
# 1,3-di-tert-butyl-2-chloro-1,3,2-diazastibolene: the ring carries two nitrogen
# lone pairs plus the antimony one, so it is an 8 pi ring and must stay
# non-aromatic (no torus) with its single C=C.


def check_diazastibolene():
    root = rebuild("Diazastibolene.cif")

    check("Diazastibolene has no aromatic torus", len(tori(root)) == 0, str(len(tori(root))))
    check(
        "Diazastibolene keeps its single ring C=C",
        strokes(root, "C|C", 2) == 2,
        f"{strokes(root, 'C|C', 2)} C|C double strokes",
    )
    check(
        "Diazastibolene draws both Sb-N bonds as singles",
        strokes(root, "N|SB", 1) == 2 and strokes(root, "N|SB", 2) == 0,
        f"{strokes(root, 'N|SB', 1)} single, {strokes(root, 'N|SB', 2)} double",
    )
    check(
        "Diazastibolene keeps the Sb-Cl bond single",
        strokes(root, "CL|SB", 1) == 1,
        f"{strokes(root, 'CL|SB', 1)} CL|SB single strokes",
    )


# --------------------------------------------------------- COD 2216154 (B)
# 2,4,6-trichloro-1,3,5-trimethylborazine: both B3N3 rings are aromatic in the
# chemistry API (six electrons from the three nitrogen lone pairs), and a
# six-ring with three donating nitrogens has no complete arrangement of
# alternating single and double bonds -- it would need one and a half B=N
# doubles -- so the ring keeps its delocalisation torus, exactly like the
# isoelectronic h-BN sheet.  No B=N double bond is drawn.


def check_borazine():
    root = rebuild("Borazine.cif")

    check(
        "Borazine draws two aromatic tori (96 segments)",
        len(tori(root)) == 1 and torus_edges(root) == 96,
        f"{len(tori(root))} object(s), {torus_edges(root)} segments",
    )
    check(
        "Borazine draws all twelve B-N bonds as singles",
        strokes(root, "B|N", 1) == 12 and strokes(root, "B|N", 2) == 0,
        f"{strokes(root, 'B|N', 1)} single, {strokes(root, 'B|N', 2)} double",
    )
    check(
        "Borazine invents no double bond at all",
        all(strokes(root, pair, 2) == 0 for pair in ("B|N", "B|CL", "C|N")),
        "checked B|N, B|CL, C|N",
    )


CASES = [
    ("Selenochromenylium.cif", lambda: check_chromenylium("Selenochromenylium.cif", "SE")),
    ("Tellurochromenylium.cif", lambda: check_chromenylium("Tellurochromenylium.cif", "TE")),
    ("Selenophene.cif", lambda: check_selenophene("Selenophene.cif", "SE")),
    ("Tellurophene.cif", lambda: check_selenophene("Tellurophene.cif", "TE")),
    ("Arsabenzene.cif", check_arsabenzene),
    ("Arsole.cif", check_arsole),
    ("Diazastibolene.cif", check_diazastibolene),
    ("Borazine.cif", check_borazine),
]

for cif_name, run in CASES:
    if os.path.exists(_env.cif(cif_name)):
        run()
    else:
        print(f"[SKIP] CIF/{cif_name} is missing; its checks did not run")

print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "ALL PASS" if not FAILURES else "FAIL")
