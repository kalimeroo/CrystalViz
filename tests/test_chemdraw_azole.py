"""ChemDraw depiction of five-membered rings: azoles and metallocene Cp.

IUPAC, Graphical Representation Standards for Chemical Structure Diagrams
(Brecher, Pure Appl. Chem. 2008, 80(2), 277-410), GR-6.3:

    Furan and pyrrole are examples of systems that can be represented with only
    one unique arrangement of single and double bonds even though they do have
    4n + 2 electrons.  It is acceptable to depict such systems using curves,
    but it is preferable to show the localized single and double bonds instead.

A ring whose sextet needs a heteroatom lone pair therefore gets no torus: the
triazole of ``aromatic.cif`` is drawn with its two C=N doubles.  A haptically
coordinated ring is the opposite case (GR-1.7.2 requires the curve), so the two
cyclopentadienyl rings of ``Ferrocene.cif`` keep their tori even though each Cp
carbon already carries four single bonds.

Run:
  blender --background --factory-startup --python test_chemdraw_azole.py
"""

import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

FAILURES = []

# Covalent radii (Angstrom) of the hand-built fragments.
RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "FE": 1.32}


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


def display(point):
    """Angstrom point -> Blender display point."""
    return tuple(float(value) * CrystalViz._DISPLAY_SCALE for value in point)


def ring_points(count, radius=1.2):
    """Regular planar polygon in Angstrom, first atom on +x."""
    return [
        (
            radius * math.cos(2.0 * math.pi * index / count),
            radius * math.sin(2.0 * math.pi * index / count),
            0.0,
        )
        for index in range(count)
    ]


def classify(elements, points, bonds):
    """Run the ChemDraw perception on a hand-built fragment."""
    segments = []
    for first, second in bonds:
        segments.append(
            (
                display(points[first]),
                display(points[second]),
                elements[first],
                elements[second],
            )
        )
    radii = {
        CrystalViz._bond_point_key(display(point)): RADII[elements[label]]
        for label, point in points.items()
    }
    orders = CrystalViz._chemdraw_bond_orders(segments, radii)
    aromatic_edges, aromatic = CrystalViz._classify_aromatic_rings(segments, radii, orders)
    return orders, aromatic_edges, aromatic


# ---------------------------------------------------------------- aromatic.cif
scene = bpy.context.scene
scene.cgn_connectivity_pairs.clear()
scene.cgn_chemdraw = True

# COD 4025500: a 1H-1,2,4-triazole ring carrying methyl and phenyl
# substituents, from Saleem & Tepe, J. Org. Chem. 75 (2010) 4330,
# doi:10.1021/jo100716m.  The phenyl ring keeps its torus; the triazole is
# drawn Kekule.  The checks are skipped when the fixture is absent.
AROMATIC_CIF = _env.cif("aromatic.cif")


def check_triazole_fixture():
    bpy.ops.cgn.import_cif(filepath=AROMATIC_CIF)
    root = bpy.data.collections.get(scene.cgn_active_structure)
    CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
    CrystalViz._rebuild_bonds(scene)

    # The triazole ring N1/N2/C2/N3/C3 has no torus: its two ring C=N bonds
    # (N2=C3 and N3=C2) are drawn as ordinary double strokes, 2 segments each.
    check(
        "triazole C=N doubles are drawn",
        strokes(root, "C|N", 2) == 4,
        f"{strokes(root, 'C|N', 2)} C|N double strokes",
    )
    check(
        "triazole ring bonds are single strokes",
        strokes(root, "C|N", 1) == 2,
        f"{strokes(root, 'C|N', 1)} C|N single strokes",
    )

    # The phenyl ring keeps its torus, so its three doubles stay suppressed.
    check(
        "phenyl doubles are suppressed",
        strokes(root, "C|C", 2) == 0,
        f"{strokes(root, 'C|C', 2)} C|C double strokes",
    )
    aromatic_objects = tori(root)
    check("one aromatic object", len(aromatic_objects) == 1, str(len(aromatic_objects)))
    check(
        "only the phenyl keeps a torus (48 segments)",
        len(aromatic_objects[0].data.edges) == 48 if aromatic_objects else False,
        str(len(aromatic_objects[0].data.edges)) if aromatic_objects else "no torus",
    )


if os.path.exists(AROMATIC_CIF):
    check_triazole_fixture()
else:
    print("[SKIP] CIF/aromatic.cif is missing; the triazole checks did not run")


# ------------------------------------------------------------ N-methylpyrrole
# IUPAC GR-6.3's pyrrole case: one unique arrangement of alternating bonds, so
# the ring is drawn Kekule even though the nitrogen donates its lone pair.
points = {}
ring = ring_points(5)
points["N0"] = ring[0]
for index in range(1, 5):
    points[f"C{index}"] = ring[index]
points["C5"] = (points["N0"][0] + 1.46, points["N0"][1], 0.0)
elements = {label: "N" if label == "N0" else "C" for label in points}
bonds = [
    ("N0", "C1"),
    ("C1", "C2"),
    ("C2", "C3"),
    ("C3", "C4"),
    ("C4", "N0"),
    ("N0", "C5"),
]
orders, aromatic_edges, aromatic = classify(elements, points, bonds)
check(
    "N-methylpyrrole is drawn with two doubles",
    len(orders) == 2,
    str(sorted(orders.values())),
)
check("N-methylpyrrole gets no torus", not aromatic, str(len(aromatic)))
check("N-methylpyrrole ring bonds are not suppressed", not aromatic_edges)

# ------------------------------------------------------------------- benzene
# Control: the same perception still gives the carbocycle a torus.
points = {f"C{index}": point for index, point in enumerate(ring_points(6, 1.4))}
elements = {label: "C" for label in points}
bonds = [(f"C{index}", f"C{(index + 1) % 6}") for index in range(6)]
orders, aromatic_edges, aromatic = classify(elements, points, bonds)
check("benzene keeps its torus", len(aromatic) == 1, str(len(aromatic)))
check("benzene ring bonds are suppressed", len(aromatic_edges) == 6, str(len(aromatic_edges)))

# ------------------------------------------------- ferrocene Cp + iron centre
# Every Cp carbon already has four single bonds (two ring carbons, one
# hydrogen, the iron), so no Kekule structure exists: GR-1.7.2 requires the
# delocalisation curve, which is the torus.
ring = ring_points(5)
points = {}
elements = {}
for index in range(5):
    points[f"C{index}"] = ring[index]
    points[f"H{index}"] = (
        ring[index][0] * 1.7 / 1.2,
        ring[index][1] * 1.7 / 1.2,
        0.0,
    )
    elements[f"C{index}"] = "C"
    elements[f"H{index}"] = "H"
points["FE0"] = (0.0, 0.0, 0.0)
elements["FE0"] = "FE"
bonds = [(f"C{index}", f"C{(index + 1) % 5}") for index in range(5)]
bonds += [(f"C{index}", f"H{index}") for index in range(5)]
bonds += [(f"C{index}", "FE0") for index in range(5)]
orders, aromatic_edges, aromatic = classify(elements, points, bonds)
check("Cp is drawn with no double bond", not orders, str(len(orders)))
check("Cp keeps a torus", len(aromatic) == 1, str(len(aromatic)))

# -------------------------------------------------------------- Ferrocene.cif
scene.cgn_connectivity_pairs.clear()
bpy.ops.cgn.import_cif(filepath=_env.cif("Ferrocene.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)
CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
CrystalViz._rebuild_bonds(scene)

aromatic_objects = tori(root)
check("ferrocene draws one aromatic object", len(aromatic_objects) == 1, str(len(aromatic_objects)))
check(
    "ferrocene draws five tori (2 Cp + 3 anthracene rings)",
    len(aromatic_objects[0].data.edges) == 240 if aromatic_objects else False,
    str(len(aromatic_objects[0].data.edges)) if aromatic_objects else "no torus",
)
check(
    "ferrocene draws the alkyne as a triple bond",
    strokes(root, "C|C", 3) == 3,
    str(strokes(root, "C|C", 3)),
)
check(
    "ferrocene draws no C=C double (arene bonds are tori)",
    strokes(root, "C|C", 2) == 0,
    str(strokes(root, "C|C", 2)),
)

print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "ALL PASS" if not FAILURES else "FAIL")
