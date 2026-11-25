"""ChemDraw depiction of delocalised oxo groups (1.5 bonds).

A carboxylate, carbonate, nitrate, phosphate, sulfate or sulfonate has more
terminal oxygens than its valence ladder can draw as genuine doubles.  Every
terminal X-O bond of such a group is drawn as a ChemDraw delocalised bond: one
solid stroke plus one dashed stroke at the multiple-bond spacing.  The dashed
stroke carries twice as many dash segments as the first implementation, and it
sits *inside* the O-X-O angle (toward the other delocalised oxygen of the same
centre) with the solid stroke *outside*, like a hand-drawn nitrite or
carboxylate.  A nitro ``N(=O)2`` or a sulfone ``S(=O)2`` spends its whole oxo
budget and keeps its genuine doubles.

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
        "carboxylate draws 12 strokes (1 solid + 5 dashes per C-O)",
        len(rendered) == 12,
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
        check(f"C-O {index + 1} has five dashes", len(dashes) == 5, str(len(dashes)))
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


def check_inside_outside():
    """The dashed stroke of each O-X-O bond sits inside the angle, the solid
    stroke outside — the nitrite/carboxylate hand-drawing convention."""
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
    directions = CrystalViz._multiple_bond_plane_directions(segments, bond_orders)
    rendered = CrystalViz._chemdraw_segments(
        segments,
        bond_orders,
        {},
        directions,
        bond_spacing=0.32,
        delocalised_edges=delocalised,
    )
    # The inside of the O-X-O angle at bond i is the side toward the other
    # oxygen: O_b -> O_a (and O_a -> O_b for the second bond).
    inside = {
        0: tuple(oxygen_b[axis] - oxygen_a[axis] for axis in range(3)),
        1: tuple(oxygen_a[axis] - oxygen_b[axis] for axis in range(3)),
    }
    for index, (first, second) in enumerate(bonds):
        direction = unit((first, second))
        strokes = [
            segment
            for segment in rendered
            if max(abs(value) for value in cross(unit(segment), direction)) < 1e-6
        ]
        full = [segment for segment in strokes if length(segment) > 0.5]
        dashes = [segment for segment in strokes if length(segment) <= 0.5]
        check(f"O-C-O {index + 1}: one solid stroke", len(full) == 1, str(len(full)))
        check(f"O-C-O {index + 1}: five dashes", len(dashes) == 5, str(len(dashes)))

        def centre(segment):
            return tuple(0.5 * (segment[0][axis] + segment[1][axis]) for axis in range(3))

        bond_mid = centre((first, second))
        dash_dir = unit((bond_mid, centre(dashes[0])))
        solid_dir = unit((bond_mid, centre(full[0])))
        dash_dot = sum(dash_dir[axis] * inside[index][axis] for axis in range(3))
        solid_dot = sum(solid_dir[axis] * inside[index][axis] for axis in range(3))
        check(
            f"O-C-O {index + 1}: dashes inside the angle, solid outside",
            dash_dot > 0.0 and solid_dot < 0.0,
            f"dash dot={dash_dot:.3f} solid dot={solid_dot:.3f}",
        )


def check_shared_centre_planes():
    """Two double bonds of one centre (O=S=O, O=C=O) are drawn coplanar.

    Each double bond's offset is fitted independently over its own neighbour
    atoms, so without the shared-centre resolution a centre that carries two
    multiple bonds can get each offset tilted in its own plane and the strokes
    no longer read as one flat group.  The resolution forces every
    multiple-bond offset of one centre inside the shared plane of its bond
    axes; a near-linear cumulene spans nearly no plane, so its offsets end up
    sharing one stroke direction and the two double bonds stay parallel.
    """

    def offset_strokes(rendered, edge_key):
        """The two full-length order-2 offset strokes of one double bond."""
        axis = unit(edge_key)
        return [
            segment
            for segment in rendered
            if int(segment[4]) == 2
            and max(abs(value) for value in cross(unit(segment), axis)) < 1e-6
        ]

    def plane_distance(points, origin, direction_a, direction_b):
        normal = cross(direction_a, direction_b)
        size = math.sqrt(sum(value * value for value in normal))
        normal = tuple(value / size for value in normal)
        return max(
            abs(sum((point[axis] - origin[axis]) * normal[axis] for axis in range(3)))
            for point in points
        )

    # Bent sulfone: S with two genuine S=O doubles and two single-bond
    # carbons, deliberately not sitting in one plane so each per-edge fit
    # would tilt its own way without the shared-centre resolution.
    sulfur = (0.0, 0.0, 0.0)
    oxygen_a = (1.42, 0.08, 0.30)
    oxygen_b = (-0.66, 1.26, 0.12)
    carbon_a = (0.55, -0.85, 0.05)
    carbon_b = (-0.95, -0.95, -0.15)
    bends = [(sulfur, oxygen_a), (sulfur, oxygen_b), (sulfur, carbon_a), (sulfur, carbon_b)]
    segments = [
        (first, second, "S", "O" if index < 2 else "C")
        for index, (first, second) in enumerate(bends)
    ]
    orders = {
        CrystalViz._bond_edge_key(first, second): (2 if index < 2 else 1)
        for index, (first, second) in enumerate(bends)
    }
    directions = CrystalViz._multiple_bond_plane_directions(segments, orders)
    rendered = CrystalViz._chemdraw_segments(segments, orders, {}, directions, bond_spacing=0.32)
    axis_a = unit((sulfur, oxygen_a))
    axis_b = unit((sulfur, oxygen_b))
    perp_a = directions[CrystalViz._bond_edge_key(sulfur, oxygen_a)]
    perp_b = directions[CrystalViz._bond_edge_key(sulfur, oxygen_b)]
    check(
        "O=S=O offsets are perpendicular to their own bonds",
        all(
            abs(sum(perp[index] * bond_axis[index] for index in range(3))) < 1e-6
            for perp, bond_axis in ((perp_a, axis_a), (perp_b, axis_b))
        ),
        f"perp_a={[round(v, 4) for v in perp_a]} perp_b={[round(v, 4) for v in perp_b]}",
    )
    shared_normal = cross(axis_a, axis_b)
    shared_size = math.sqrt(sum(value * value for value in shared_normal))
    shared_normal = tuple(value / shared_size for value in shared_normal)
    check(
        "O=S=O offsets lie in the plane of the two S=O bonds",
        all(
            abs(sum(perp[axis] * shared_normal[axis] for axis in range(3))) < 1e-6
            for perp in (perp_a, perp_b)
        ),
        f"dot_a={sum(perp_a[a] * shared_normal[a] for a in range(3)):.2e} "
        f"dot_b={sum(perp_b[a] * shared_normal[a] for a in range(3)):.2e}",
    )
    stroke_a = offset_strokes(rendered, CrystalViz._bond_edge_key(sulfur, oxygen_a))
    stroke_b = offset_strokes(rendered, CrystalViz._bond_edge_key(sulfur, oxygen_b))
    endpoints = []
    for stroke in stroke_a + stroke_b:
        endpoints.extend((stroke[0], stroke[1]))
    check(
        "O=S=O: both double bonds are two full strokes",
        len(stroke_a) == 2 and len(stroke_b) == 2,
    )
    check(
        "O=S=O stroke pairs are coplanar",
        plane_distance(endpoints, sulfur, axis_a, perp_a) < 1e-6,
        f"max distance={plane_distance(endpoints, sulfur, axis_a, perp_a):.2e}",
    )

    # Linear cumulene O=C=O: the two offsets resolve to one shared stroke
    # direction, keeping both double bonds parallel.
    carbon = (0.0, 0.0, 0.0)
    chain_a = (1.2, 0.1, 0.3)
    chain_b = (-0.95 * chain_a[0], -0.95 * chain_a[1], -0.95 * chain_a[2])
    chain_segments = [(carbon, chain_a, "C", "O"), (carbon, chain_b, "C", "O")]
    chain_edges = [(carbon, chain_a), (carbon, chain_b)]
    chain_orders = {CrystalViz._bond_edge_key(first, second): 2 for first, second in chain_edges}
    chain_directions = CrystalViz._multiple_bond_plane_directions(chain_segments, chain_orders)
    chain_rendered = CrystalViz._chemdraw_segments(
        chain_segments, chain_orders, {}, chain_directions, bond_spacing=0.32
    )
    chain_perp_a = chain_directions[CrystalViz._bond_edge_key(carbon, chain_a)]
    chain_perp_b = chain_directions[CrystalViz._bond_edge_key(carbon, chain_b)]
    check(
        "O=C=O offsets are the same shared stroke direction",
        sum(chain_perp_a[axis] * chain_perp_b[axis] for axis in range(3)) > 1.0 - 1e-6,
        f"dot={sum(chain_perp_a[a] * chain_perp_b[a] for a in range(3)):.6f}",
    )
    chain_axis_a = unit((carbon, chain_a))
    chain_endpoints = []
    for stroke in offset_strokes(
        chain_rendered, CrystalViz._bond_edge_key(carbon, chain_a)
    ) + offset_strokes(chain_rendered, CrystalViz._bond_edge_key(carbon, chain_b)):
        chain_endpoints.extend((stroke[0], stroke[1]))
    check(
        "O=C=O stroke pairs are coplanar",
        plane_distance(chain_endpoints, carbon, chain_axis_a, chain_perp_a) < 1e-6,
        f"max distance={plane_distance(chain_endpoints, carbon, chain_axis_a, chain_perp_a):.2e}",
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


def check_delocalised_shared_plane():
    """The two 1.5 strokes of a resonance O-X-O group share one stroke plane.

    A hypophosphite ``PO2H`` (or a carboxylate) centre is drawn as two
    delocalised 1.5 bonds: the valence ladder seeds one genuine ``X=O`` double
    (that edge is in ``bond_orders``) and the whole group is delocalised, so
    the *second* ``X-O`` edge is absent from ``bond_orders``.  The direction
    pass must fit and resolve that edge as a multiple bond too, or its strokes
    fall back to the arbitrary reference-axis perpendicular inside
    ``_chemdraw_segments`` and tilt out of the O-X-O plane.
    """
    phosphorus = (0.0, 0.0, 0.0)
    oxygen_a = (1.42, 0.08, 0.30)
    oxygen_b = (-0.66, 1.26, 0.12)
    hydrogen_a = (0.55, -0.85, 0.05)
    hydrogen_b = (-0.95, -0.95, -0.15)
    bonds = [
        (phosphorus, oxygen_a),
        (phosphorus, oxygen_b),
        (phosphorus, hydrogen_a),
        (phosphorus, hydrogen_b),
    ]
    segments = [
        (first, second, "P", "O" if index < 2 else "H")
        for index, (first, second) in enumerate(bonds)
    ]
    # Hypophosphite bookkeeping: P is covalently 4-connected (2 H + 2 O), so
    # its (3, 5) ladder leaves one spare order — exactly one seeded P=O.  Both
    # P-O edges are terminal oxide, so the group is delocalised as a whole.
    edges = [CrystalViz._bond_edge_key(first, second) for first, second in bonds]
    bond_orders = {edges[0]: 2}
    delocalised = frozenset((edges[0], edges[1]))
    directions = CrystalViz._multiple_bond_plane_directions(segments, bond_orders, delocalised)
    rendered = CrystalViz._chemdraw_segments(
        segments, bond_orders, {}, directions, bond_spacing=0.32, delocalised_edges=delocalised
    )

    axis_a = unit((phosphorus, oxygen_a))
    axis_b = unit((phosphorus, oxygen_b))
    shared_normal = cross(axis_a, axis_b)
    shared_size = math.sqrt(sum(value * value for value in shared_normal))
    shared_normal = tuple(value / shared_size for value in shared_normal)

    def stroke_points(axis, rendered):
        points_out = []
        for segment in rendered:
            if int(segment[4]) != 2:
                continue
            if max(abs(value) for value in cross(unit(segment), axis)) > 1e-6:
                continue
            points_out.extend((segment[0], segment[1]))
        return points_out

    stroke_a = stroke_points(axis_a, rendered)
    stroke_b = stroke_points(axis_b, rendered)
    check(
        "PO2H delocalised bonds have strokes (not bare singles)",
        len(stroke_a) >= 2 and len(stroke_b) >= 2,
        f"a={len(stroke_a)} b={len(stroke_b)}",
    )
    worst = max(
        abs(sum((point[axis] - phosphorus[axis]) * shared_normal[axis] for axis in range(3)))
        for point in stroke_a + stroke_b
    )
    check(
        "PO2H delocalised stroke pairs are coplanar",
        worst < 1e-6,
        f"max out-of-plane={worst:.2e}",
    )
    perp_a = directions.get(edges[0])
    perp_b = directions.get(edges[1])
    check(
        "PO2H both P-O edges carry a fitted stroke direction",
        perp_a is not None and perp_b is not None,
        f"a={perp_a is not None} b={perp_b is not None}",
    )
    check(
        "PO2H offsets lie in the plane of the two P-O bonds",
        perp_b is not None
        and abs(sum(perp_b[axis] * shared_normal[axis] for axis in range(3))) < 1e-6,
        f"dot={sum(perp_b[a] * shared_normal[a] for a in range(3)):.2e}" if perp_b else "no perp",
    )


def main():
    check_rendering()
    check_inside_outside()
    check_shared_centre_planes()
    check_delocalised_shared_plane()
    check_control_double_bond()
    check_detection()


main()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "ALL PASS" if not FAILURES else "FAIL")
sys.exit(0 if not FAILURES else 1)
