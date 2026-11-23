"""ChemDraw depiction of delocalised oxo groups (1.5 bonds).

A carboxylate, carbonate, nitrate, phosphate, sulfate or sulfonate has more
terminal oxygens than its valence ladder can draw as genuine doubles.  Every
terminal X-O bond of such a group is drawn as a ChemDraw delocalised bond: one
solid stroke plus one dashed stroke at the multiple-bond spacing.  A nitro
``N(=O)2`` or a sulfone ``S(=O)2`` spends its whole oxo budget and keeps its
genuine doubles.

Run:
  blender --background --factory-startup --python test_chemdraw_delocalised.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def length(segment):
    return math.sqrt(sum((segment[1][axis] - segment[0][axis]) ** 2 for axis in range(3)))


def cross(first, second):
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def unit(segment):
    delta = tuple(segment[1][axis] - segment[0][axis] for axis in range(3))
    size = math.sqrt(sum(value * value for value in delta))
    return tuple(value / size for value in delta)


def check_rendering():
    """The carboxylate's two C-O edges become solid + dashed order-2 strokes."""
    carbon = (0.0, 0.0, 0.0)
    oxygen_a = (1.25, 0.0, 0.0)
    oxygen_b = (-0.3, 1.2, 0.0)
    bonds = [(carbon, oxygen_a), (carbon, oxygen_b)]
    segments = [(first, second, "C", "O") for first, second in bonds]
    bond_orders = {
        CrystalViz._bond_edge_key(first, second): (2 if index == 0 else 1)
        for index, (first, second) in enumerate(bonds)
    }
    delocalised = frozenset(bond_orders)

    rendered = CrystalViz._chemdraw_segments(
        segments, bond_orders, {}, bond_spacing=0.32, delocalised_edges=delocalised
    )
    check(
        "carboxylate renders only order-2 strokes",
        all(int(segment[4]) == 2 for segment in rendered),
        str(sorted({int(segment[4]) for segment in rendered})),
    )
    check(
        "carboxylate draws 8 strokes (1 solid + 3 dashes per C-O)",
        len(rendered) == 8,
        str(len(rendered)),
    )

    # Group the strokes by the bond they are parallel to.
    groups = {0: [], 1: []}
    for segment in rendered:
        direction = unit(segment)
        for index, (first, second) in enumerate(bonds):
            if max(abs(value) for value in cross(direction, unit((first, second)))) < 1e-6:
                groups[index].append(segment)
                break
    for index, (first, second) in enumerate(bonds):
        perpendicular = CrystalViz._bond_perpendicular(unit((first, second)), None, {})
        full = [segment for segment in groups[index] if length(segment) > 0.5]
        dashes = [segment for segment in groups[index] if length(segment) <= 0.5]
        check(f"C-O {index + 1} has one solid stroke", len(full) == 1, str(len(full)))
        check(f"C-O {index + 1} has three dashes", len(dashes) == 3, str(len(dashes)))
        check(
            f"C-O {index + 1} dashes are short",
            all(length(dash) <= CrystalViz._CHEMDRAW_DASH_LENGTH + 1e-6 for dash in dashes),
        )
        solid_offset = sum(full[0][0][axis] * perpendicular[axis] for axis in range(3))
        dash_offset = sum(dashes[0][0][axis] * perpendicular[axis] for axis in range(3))
        check(
            f"C-O {index + 1} draws the solid and dashed strokes on opposite sides",
            solid_offset * dash_offset < 0.0,
            f"solid={solid_offset:.3f} dash={dash_offset:.3f}",
        )


def check_control_double_bond():
    """A genuine ketone C=O is still two full-length strokes."""
    carbon = (5.0, 0.0, 0.0)
    oxygen = (6.22, 0.0, 0.0)
    segments = [(carbon, oxygen, "C", "O")]
    bond_orders = {CrystalViz._bond_edge_key(carbon, oxygen): 2}
    rendered = CrystalViz._chemdraw_segments(
        segments, bond_orders, {}, bond_spacing=0.32, delocalised_edges=frozenset()
    )
    check("ketone renders two strokes", len(rendered) == 2, str(len(rendered)))
    check(
        "ketone strokes are full-length",
        all(length(segment) > 1.2 for segment in rendered),
        str([round(length(segment), 3) for segment in rendered]),
    )


def check_detection():
    """The chemistry adapter marks the resonance groups and nothing else."""
    segments = []
    radii = {}

    def add(first, second, element_a, element_b):
        segments.append((first, second, element_a, element_b))
        radii[CrystalViz._bond_point_key(first)] = {
            "C": 0.76,
            "N": 0.71,
            "O": 0.66,
            "P": 1.07,
            "S": 1.05,
        }[element_a]
        radii[CrystalViz._bond_point_key(second)] = {
            "C": 0.76,
            "N": 0.71,
            "O": 0.66,
            "P": 1.07,
            "S": 1.05,
        }[element_b]

    # Carboxylate: C0 with O1, O2 and the chain carbon C1.
    add((0.0, 0.0, 0.0), (1.25, 0.0, 0.0), "C", "O")  # C0-O1
    add((0.0, 0.0, 0.0), (-0.3, 1.2, 0.0), "C", "O")  # C0-O2
    add((0.0, 0.0, 0.0), (0.0, -1.5, 0.0), "C", "C")  # C0-C1
    # Nitro: N0 with O3, O4 and the chain carbon C2.
    add((10.0, 0.0, 0.0), (11.22, 0.0, 0.0), "N", "O")
    add((10.0, 0.0, 0.0), (10.0, 1.24, 0.0), "N", "O")
    add((10.0, 0.0, 0.0), (10.0, -1.47, 0.0), "N", "C")
    # Phosphate: P0 with four terminal oxygens.
    for point in ((21.52, 0.0, 0.0), (18.48, 0.0, 0.0), (20.0, 1.52, 0.0), (20.0, 0.0, 1.52)):
        add((20.0, 0.0, 0.0), point, "P", "O")
    # Sulfone: S0 with two terminal oxygens and two chain carbons.
    add((30.0, 0.0, 0.0), (31.43, 0.0, 0.0), "S", "O")
    add((30.0, 0.0, 0.0), (30.0, 1.43, 0.0), "S", "O")
    add((30.0, 0.0, 0.0), (30.0, -1.78, 0.0), "S", "C")
    add((30.0, 0.0, 0.0), (28.3, 0.9, 0.0), "S", "C")
    # Ketone: C4 with one terminal oxygen and two chain carbons.
    add((40.0, 0.0, 0.0), (41.22, 0.0, 0.0), "C", "O")
    add((40.0, 0.0, 0.0), (40.0, -1.51, 0.0), "C", "C")
    add((40.0, 0.0, 0.0), (40.0, 1.51, 0.0), "C", "C")

    marked = CrystalViz._chemdraw_delocalised_edges(segments, radii)
    keys = {CrystalViz._bond_edge_key(first, second) for first, second, _a, _b in segments}
    carboxylate = {
        CrystalViz._bond_edge_key((0.0, 0.0, 0.0), (1.25, 0.0, 0.0)),
        CrystalViz._bond_edge_key((0.0, 0.0, 0.0), (-0.3, 1.2, 0.0)),
    }
    phosphate = {
        CrystalViz._bond_edge_key((20.0, 0.0, 0.0), point)
        for point in ((21.52, 0.0, 0.0), (18.48, 0.0, 0.0), (20.0, 1.52, 0.0), (20.0, 0.0, 1.52))
    }
    check("carboxylate is delocalised", carboxylate <= marked, str(len(carboxylate & marked)))
    check("phosphate is delocalised", phosphate <= marked, str(len(phosphate & marked)))
    check(
        "nitro, sulfone and ketone stay localized",
        not (marked & (keys - carboxylate - phosphate)),
        str(sorted(marked - carboxylate - phosphate)),
    )


def main():
    check_rendering()
    check_control_double_bond()
    check_detection()


main()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "ALL PASS" if not FAILURES else "FAIL")
sys.exit(0 if not FAILURES else 1)
