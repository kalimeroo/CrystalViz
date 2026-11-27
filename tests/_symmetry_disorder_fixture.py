"""Synthetic CIF reproducing a mirror-generated *tert*-butyl disorder.

``CIF/Ca.cif`` is the structure that drove the load-time resolver, but it is a
local-only fixture (``.gitignore``), so the tests build an equivalent structure
themselves:

* space group ``P 1 2/m 1`` with the mirror at ``y = 1/2``;
* a benzene ring lying *in* the mirror plane, so none of its atoms is
  disordered, with a *tert*-butyl carbon on one ring carbon (also in the
  plane);
* the three methyl carbons of the t-Bu are general positions written at 0.5
  occupancy, so the space group generates a second, mirror-related orientation
  of the whole group -- the "symmetry-generated disorder" of ``Ca.cif``;
* one of the three methyls (with its hydrogens) is deposited at its *mirrored*
  position, exactly like ``Ca.cif``: the CIF's own combination is then not a
  tetrahedron (two methyls 1.4 A apart) and only the two whole-group
  combinations are chemically valid, of which the one keeping the most CIF
  images wins;
* a half-occupied potassium/water pair on general positions, which is genuine
  substitutional disorder and must never be resolved.

The module imports no ``bpy`` and no ``gemmi`` so both the worker suite and the
Blender import suite can share it.
"""

import math

A = 24.0
B = 14.0
C = 20.0

_TETRA_COS = -1.0 / 3.0
_TETRA_SIN = math.sqrt(1.0 - _TETRA_COS * _TETRA_COS)

RING_RADIUS = 1.39
RING_C_H = 1.08
T_BU_LENGTH = 1.52
METHYL_C_H = 1.09

RING_CENTER = (6.0, B / 2.0, 10.0)


def _add(first, second):
    return tuple(a + b for a, b in zip(first, second, strict=True))


def _scale(vector, factor):
    return tuple(value * factor for value in vector)


def _unit(vector):
    length = math.sqrt(sum(value * value for value in vector))
    return tuple(value / length for value in vector)


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _mirror(point):
    """Reflect a Cartesian point through the ``y = B/2`` plane."""
    return (point[0], B - point[1], point[2])


def _fractional(point):
    return [point[0] / A, point[1] / B, point[2] / C]


def _ring_atoms():
    """Benzene ring in the mirror plane, plus its five C-H hydrogens."""
    atoms = []
    for index in range(6):
        angle = math.radians(60.0 * index)
        direction = (math.cos(angle), 0.0, math.sin(angle))
        carbon = _add(RING_CENTER, _scale(direction, RING_RADIUS))
        atoms.append((f"C{index + 1}", "C", carbon, 1.0))
        if index:
            hydrogen = _add(RING_CENTER, _scale(direction, RING_RADIUS + RING_C_H))
            atoms.append((f"H{index + 1}", "H", hydrogen, 1.0))
    return atoms


def _t_butyl_atoms():
    """t-Bu on ring carbon C1; one methyl is deposited at its mirror image."""
    ring_c1 = _add(RING_CENTER, (RING_RADIUS, 0.0, 0.0))
    apex = _add(ring_c1, (T_BU_LENGTH, 0.0, 0.0))
    atoms = [("C7", "C", apex, 1.0)]
    # Rotating the three methyls by 60 degrees about the C1-C7 axis puts two
    # on one side of the mirror and one on the other, which is what makes the
    # group 2-fold disordered.
    for methyl_index, psi in enumerate((60.0, 180.0, 300.0)):
        angle = math.radians(psi)
        direction = (
            -_TETRA_COS,
            _TETRA_SIN * math.cos(angle),
            _TETRA_SIN * math.sin(angle),
        )
        carbon = _add(apex, _scale(direction, T_BU_LENGTH))
        label = f"C{8 + methyl_index}"
        deposited = _mirror(carbon) if methyl_index == 0 else carbon
        atoms.append((label, "C", deposited, 0.5))
        # Hydrogens are built around the *parent* methyl and mirrored together
        # with it, so each image stays a complete methyl.
        parent_to_apex = _unit(_add(apex, _scale(carbon, -1.0)))
        helper = (0.0, 0.0, 1.0)
        if abs(parent_to_apex[2]) > 0.9:
            helper = (1.0, 0.0, 0.0)
        first = _unit(_cross(parent_to_apex, helper))
        second = _unit(_cross(parent_to_apex, first))
        for hydrogen_index in range(3):
            spin = math.radians(120.0 * hydrogen_index)
            radial = _add(_scale(first, math.cos(spin)), _scale(second, math.sin(spin)))
            direction_h = _add(_scale(parent_to_apex, _TETRA_COS), _scale(radial, _TETRA_SIN))
            hydrogen = _add(carbon, _scale(direction_h, METHYL_C_H))
            if methyl_index == 0:
                hydrogen = _mirror(hydrogen)
            atoms.append((f"H{8 + methyl_index}{'ABC'[hydrogen_index]}", "H", hydrogen, 0.5))
    return atoms


def _solvent_atoms():
    """Half-occupied K/water pair: substitutional, not symmetry-generated."""
    return [
        ("K1", "K", (0.15 * A, 0.2 * B, 0.15 * C), 0.5),
        ("O1", "O", (0.15 * A, 0.2 * B, 0.15 * C + 2.7), 0.5),
    ]


def cif_text():
    """Return the synthetic P 1 2/m 1 CIF as a string."""
    lines = [
        "data_mirror_tbutyl_disorder",
        f"_cell_length_a {A:.4f}",
        f"_cell_length_b {B:.4f}",
        f"_cell_length_c {C:.4f}",
        "_cell_angle_alpha 90.0",
        "_cell_angle_beta 90.0",
        "_cell_angle_gamma 90.0",
        "_symmetry_space_group_name_H-M 'P 1 2/m 1'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    for label, element, point, occupancy in _ring_atoms() + _t_butyl_atoms() + _solvent_atoms():
        x, y, z = _fractional(point)
        lines.append(f"{label} {element} {x:.6f} {y:.6f} {z:.6f} {occupancy:.2f}")
    return "\n".join(lines) + "\n"


def write_cif(path):
    """Write the fixture to *path* and return the path."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(cif_text())
    return path
