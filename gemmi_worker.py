"""
Gemmi Worker: Crystallographic data processing for CrystalViz.

This module handles CIF parsing, symmetry expansion to P1 cells, and the
calculation of coordination shells. It is designed to run within Blender's
Python environment without requiring external subprocesses.

Symmetry implementation follows cctbx standards: sites within 0.5 Å of a
special position are projected onto it before expansion to ensure parity
in atom counts and coordinates. Radius tables are derived from cctbx to
maintain consistency in atom sizing and default connectivity windows.
"""

import math
import re
from typing import Any

import gemmi

# --------------------------------------------------------------------------
# cctbx-compatible radius tables (cctbx.eltbx.van_der_waals_radii and
# cctbx.eltbx.covalent_radii). Kept verbatim so the default bond windows
# (r_a + r_b + 0.4 A) match the cctbx build exactly.
# --------------------------------------------------------------------------

_VDW_RADII = {
    "Ac": 2.12,
    "Ag": 1.72,
    "Al": 1.5,
    "Am": 1.66,
    "Ar": 1.88,
    "As": 0.83,
    "At": 1.12,
    "Au": 1.66,
    "B": 1.75,
    "Ba": 2.41,
    "Be": 0.63,
    "Bi": 1.73,
    "Bk": 1.64,
    "Br": 1.85,
    "C": 1.775,
    "Ca": 1.95,
    "Cd": 1.58,
    "Ce": 1.86,
    "Cf": 1.63,
    "Cl": 1.75,
    "Cm": 1.65,
    "Co": 1.13,
    "Cr": 1.13,
    "Cs": 3.01,
    "Cu": 1.4,
    "D": 1.2,
    "Dy": 1.63,
    "Er": 1.59,
    "Es": 1.62,
    "Eu": 1.96,
    "F": 1.47,
    "Fe": 1.26,
    "Fm": 1.61,
    "Fr": 3.24,
    "Ga": 1.87,
    "Gd": 1.69,
    "Ge": 1.48,
    "H": 1.2,
    "He": 1.4,
    "Hf": 1.4,
    "Hg": 1.55,
    "Ho": 1.61,
    "I": 1.98,
    "In": 1.93,
    "Ir": 1.22,
    "K": 2.75,
    "Kr": 2.02,
    "La": 1.83,
    "Li": 1.82,
    "Lr": 1.58,
    "Lu": 1.53,
    "Md": 1.6,
    "Mg": 1.73,
    "Mn": 1.19,
    "Mo": 1.75,
    "N": 1.5,
    "Na": 2.27,
    "Nb": 1.33,
    "Nd": 1.79,
    "Ne": 1.54,
    "Ni": 1.63,
    "No": 1.59,
    "Np": 1.71,
    "O": 1.45,
    "Os": 1.58,
    "P": 1.9,
    "Pa": 1.6,
    "Pb": 2.02,
    "Pd": 1.63,
    "Pm": 1.76,
    "Po": 1.21,
    "Pr": 1.62,
    "Pt": 1.72,
    "Pu": 1.67,
    "Ra": 2.57,
    "Rb": 2.65,
    "Re": 1.3,
    "Rh": 1.22,
    "Rn": 2.3,
    "Ru": 1.2,
    "S": 1.8,
    "Sb": 1.12,
    "Sc": 1.32,
    "Se": 1.9,
    "Si": 2.1,
    "Sm": 1.74,
    "Sn": 2.17,
    "Sr": 2.02,
    "Ta": 1.22,
    "Tb": 1.66,
    "Tc": 2.0,
    "Te": 1.26,
    "Th": 1.84,
    "Ti": 1.95,
    "Tl": 1.96,
    "Tm": 1.57,
    "U": 1.75,
    "V": 1.06,
    "W": 1.26,
    "Xe": 2.16,
    "Y": 1.61,
    "Yb": 1.54,
    "Zn": 1.39,
    "Zr": 1.42,
}

_COVALENT_RADII = {
    "Ac": 2.15,
    "Ag": 1.45,
    "Al": 1.21,
    "Am": 1.8,
    "Ar": 1.06,
    "As": 1.19,
    "At": 1.5,
    "Au": 1.36,
    "B": 0.84,
    "Ba": 2.15,
    "Be": 0.96,
    "Bi": 1.48,
    "Bk": 0.84,
    "Br": 1.2,
    "C": 0.76,
    "Ca": 1.76,
    "Cd": 1.44,
    "Ce": 2.04,
    "Cf": 0.76,
    "Cl": 1.02,
    "Cm": 1.69,
    "Co": 1.26,
    "Cr": 1.39,
    "Cs": 2.44,
    "Cu": 1.32,
    "D": 0.31,
    "Dy": 1.92,
    "Er": 1.89,
    "Eu": 1.98,
    "F": 0.57,
    "Fe": 1.32,
    "Fm": 0.57,
    "Fr": 2.6,
    "Ga": 1.22,
    "Gd": 1.96,
    "Ge": 1.2,
    "H": 0.31,
    "He": 0.28,
    "Hf": 1.75,
    "Hg": 1.32,
    "Ho": 1.92,
    "I": 1.39,
    "In": 1.42,
    "Ir": 1.41,
    "K": 2.03,
    "Kr": 1.16,
    "La": 2.07,
    "Li": 1.28,
    "Lu": 1.87,
    "Mg": 1.41,
    "Mn": 1.39,
    "Mo": 1.54,
    "N": 0.71,
    "Na": 1.66,
    "Nb": 1.64,
    "Nd": 2.01,
    "Ne": 0.58,
    "Ni": 1.24,
    "No": 0.71,
    "Np": 1.9,
    "O": 0.66,
    "Os": 1.44,
    "P": 1.07,
    "Pa": 2.0,
    "Pb": 1.46,
    "Pd": 1.39,
    "Pm": 1.99,
    "Po": 1.4,
    "Pr": 2.03,
    "Pt": 1.36,
    "Pu": 1.87,
    "Ra": 2.21,
    "Rb": 2.2,
    "Re": 1.51,
    "Rh": 1.42,
    "Rn": 1.5,
    "Ru": 1.46,
    "S": 1.05,
    "Sb": 1.39,
    "Sc": 1.7,
    "Se": 1.2,
    "Si": 1.11,
    "Sm": 1.98,
    "Sn": 1.39,
    "Sr": 1.95,
    "Ta": 1.7,
    "Tb": 1.94,
    "Tc": 1.47,
    "Te": 1.38,
    "Th": 2.06,
    "Ti": 1.6,
    "Tl": 1.45,
    "Tm": 1.9,
    "U": 1.96,
    "V": 1.53,
    "W": 1.62,
    "Xe": 1.4,
    "Y": 1.9,
    "Yb": 1.87,
    "Zn": 1.22,
    "Zr": 1.75,
}

# --------------------------------------------------------------------------
# Shannon ionic radii (CN 6 unless that coordination is not tabulated).
# Only used when a CIF declares formal charges AND the two elements of a pair
# carry opposite-sign non-zero charges: covalent radii describe a bond between
# neutral atoms, and they systematically under-estimate a cation-anion bond
# because the anion radius doubles when it carries its charge (O 0.66 -> 1.40).
# Ag-O in Ag2Cr2O7 is the reference case: 1.45 + 0.66 = 2.11 A with covalent
# radii (misses half the coordination sphere) versus 1.15 + 1.40 = 2.55 A.
# Elements without an entry (H, most rare ions) fall back to covalent radii.
# --------------------------------------------------------------------------

_IONIC_RADII = {
    ("O", -2): 1.40,
    ("S", -2): 1.84,
    ("Se", -2): 1.98,
    ("Te", -2): 2.21,
    ("N", -3): 1.46,
    ("F", -1): 1.33,
    ("Cl", -1): 1.81,
    ("Br", -1): 1.96,
    ("I", -1): 2.20,
    ("Li", 1): 0.76,
    ("Na", 1): 1.02,
    ("K", 1): 1.38,
    ("Rb", 1): 1.52,
    ("Cs", 1): 1.67,
    ("Ag", 1): 1.15,
    ("Cu", 1): 0.77,
    ("Tl", 1): 1.50,
    ("Au", 1): 1.37,
    ("Be", 2): 0.45,
    ("Mg", 2): 0.72,
    ("Ca", 2): 1.00,
    ("Sr", 2): 1.18,
    ("Ba", 2): 1.35,
    ("Cr", 2): 0.73,
    ("Mn", 2): 0.83,
    ("Fe", 2): 0.78,
    ("Co", 2): 0.745,
    ("Ni", 2): 0.69,
    ("Cu", 2): 0.73,
    ("Zn", 2): 0.74,
    ("Cd", 2): 0.95,
    ("Hg", 2): 1.02,
    ("Pd", 2): 0.64,
    ("Pt", 2): 0.80,
    ("Sn", 2): 1.18,
    ("Pb", 2): 1.19,
    ("B", 3): 0.27,
    ("Al", 3): 0.535,
    ("Ga", 3): 0.62,
    ("In", 3): 0.80,
    ("Tl", 3): 0.885,
    ("Sc", 3): 0.745,
    ("Y", 3): 0.90,
    ("La", 3): 1.032,
    ("Ce", 3): 1.01,
    ("Pr", 3): 0.99,
    ("Nd", 3): 0.983,
    ("Pm", 3): 0.97,
    ("Sm", 3): 0.958,
    ("Eu", 3): 0.947,
    ("Gd", 3): 0.938,
    ("Tb", 3): 0.923,
    ("Dy", 3): 0.912,
    ("Ho", 3): 0.901,
    ("Er", 3): 0.890,
    ("Tm", 3): 0.880,
    ("Yb", 3): 0.868,
    ("Lu", 3): 0.861,
    ("Ti", 3): 0.670,
    ("V", 3): 0.640,
    ("Cr", 3): 0.615,
    ("Mn", 3): 0.645,
    ("Fe", 3): 0.645,
    ("Co", 3): 0.545,
    ("Ru", 3): 0.68,
    ("Rh", 3): 0.665,
    ("Ir", 3): 0.68,
    ("As", 3): 0.58,
    ("Sb", 3): 0.76,
    ("Bi", 3): 1.03,
    ("C", 4): 0.15,
    ("Si", 4): 0.40,
    ("Ge", 4): 0.53,
    ("Sn", 4): 0.69,
    ("Pb", 4): 0.775,
    ("Ti", 4): 0.605,
    ("Zr", 4): 0.72,
    ("Hf", 4): 0.71,
    ("V", 4): 0.58,
    ("Mn", 4): 0.53,
    ("Mo", 4): 0.65,
    ("Tc", 4): 0.645,
    ("Re", 4): 0.63,
    ("Os", 4): 0.63,
    ("Se", 4): 0.50,
    ("Te", 4): 0.97,
    ("Th", 4): 0.94,
    ("U", 4): 0.89,
    ("Pu", 4): 0.86,
    ("P", 5): 0.38,
    ("V", 5): 0.54,
    ("Nb", 5): 0.64,
    ("Ta", 5): 0.64,
    ("As", 5): 0.46,
    ("Sb", 5): 0.60,
    ("Cr", 6): 0.44,
    ("Mo", 6): 0.59,
    ("W", 6): 0.60,
    ("U", 6): 0.73,
}

_DEFAULT_VDW_RADIUS = 1.70
_DEFAULT_COVALENT_RADIUS = 0.75
_CART_TOLERANCE = 1e-3  # Angstrom, position tolerance when matching atoms
# Two same-element atoms closer than this are the same atom deposited twice.
# Redeposited "full molecule + symmetry twins" CIFs (e.g. SHELX B-suffix
# halves) can repeat a site at positions that differ only in the last deposited
# fractional decimal -- up to a few 1e-4 A in a ~25 A cell -- which is an order
# of magnitude beyond the 1e-5 A rounding the old equality check used, yet far
# below any real interatomic or disorder-alternate separation (typically
# >= 0.2 A even for adjacent hydrogen positions), so merging such atoms never
# removes a genuine one.
_DUPLICATE_POSITION_TOLERANCE = 0.01  # Angstrom
_LABEL_ELEMENT = re.compile(r"([A-Za-z]{1,2})")


# --------------------------------------------------------------------------
# Elements and radii
# --------------------------------------------------------------------------


def _is_element(symbol):
    try:
        return gemmi.Element(symbol).atomic_number > 0
    except Exception:
        return False


def _element_from_text(text):
    """Parse a CIF label or type symbol into a title-case element symbol."""
    if not text:
        return None
    match = _LABEL_ELEMENT.match(str(text).strip())
    if match is None:
        return None
    letters = match.group(1)
    for candidate in (letters[:2].title(), letters[:1].upper()):
        if _is_element(candidate):
            return candidate
    return None


def _site_element(site):
    """Element symbol of a gemmi small-structure site, CIF-first like cctbx."""
    for text in (site.type_symbol, site.label):
        symbol = _element_from_text(text)
        if symbol is not None:
            return symbol
    name = getattr(site.element, "name", None)
    if name and name != "X":
        return name
    return "X"


def _radius_tables(element):
    symbol = str(element or "").title()
    vdw = _VDW_RADII.get(symbol, _DEFAULT_VDW_RADIUS)
    covalent = _COVALENT_RADII.get(symbol, _DEFAULT_COVALENT_RADIUS)
    return float(vdw), float(covalent)


# --------------------------------------------------------------------------
# Cell helpers
# --------------------------------------------------------------------------


def _cell_columns(unit_cell):
    """Orthogonalized basis vectors: cart = sum(frac[i] * columns[i])."""
    columns = []
    for axis in range(3):
        basis = [0.0, 0.0, 0.0]
        basis[axis] = 1.0
        position = unit_cell.orthogonalize(gemmi.Fractional(*basis))
        columns.extend((float(position.x), float(position.y), float(position.z)))
    return columns


def _inverse_3x3(columns):
    """Transpose of the inverse orthogonalization matrix (M^-T), row-major."""
    a, b, c, d, e, f, g, h, i = [float(value) for value in columns]
    # columns = [a d g; b e h; c f i] (basis vectors as columns)
    det = a * (e * i - f * h) - d * (b * i - c * h) + g * (b * f - c * e)
    if abs(det) < 1e-12:
        raise ValueError("The unit cell is degenerate.")
    return [
        (e * i - f * h) / det,
        (c * h - b * i) / det,
        (b * f - c * e) / det,
        (f * g - d * i) / det,
        (a * i - c * g) / det,
        (c * d - a * f) / det,
        (d * h - e * g) / det,
        (b * g - a * h) / det,
        (a * e - b * d) / det,
    ]


class _UnitCell:
    """Minimal ``uctbx.unit_cell`` look-alike backed by ``gemmi.UnitCell``."""

    def __init__(self, parameters):
        self.cell = gemmi.UnitCell(*[float(value) for value in parameters])
        self._columns = _cell_columns(self.cell)
        self._fractionalization = _inverse_3x3(self._columns)

    def orthogonalize(self, frac):
        position = self.cell.orthogonalize(
            gemmi.Fractional(float(frac[0]), float(frac[1]), float(frac[2]))
        )
        return [float(position.x), float(position.y), float(position.z)]

    def fractionalize(self, cart):
        frac = self.cell.fractionalize(
            gemmi.Position(float(cart[0]), float(cart[1]), float(cart[2]))
        )
        return [float(frac.x), float(frac.y), float(frac.z)]

    def fractionalization_matrix(self):
        return list(self._fractionalization)

    def parameters(self):
        return [float(value) for value in self.cell.parameters]


def _cart_distance(unit_cell, frac_a, frac_b):
    """Cartesian distance (Å) after component-wise rounding of the deltas.

    This is a *deduplication* metric for wrapped symmetry-equivalent images,
    not a general minimum-image distance: in a skewed (triclinic) cell the
    component-wise rounded image is not always the closest one.  Never use it
    for bond lengths -- resolve the image with gemmi's
    ``UnitCell.find_nearest_pbc_images`` and measure it with
    ``_image_distance`` instead.
    """
    delta = [float(frac_a[axis]) - float(frac_b[axis]) for axis in range(3)]
    delta = [value - round(value) for value in delta]
    cart = unit_cell.orthogonalize(delta)
    return math.sqrt(sum(value * value for value in cart))


def _cart_from_frac(frac, cell_matrix):
    """Cartesian from fractional using the column-major cell matrix.

    ``cell_matrix[row + 3 * col]`` is column ``col``'s ``row`` component:
    cart[row] = sum_col cell_matrix[row + 3*col] * frac[col].
    """
    return [
        frac[0] * cell_matrix[axis]
        + frac[1] * cell_matrix[3 + axis]
        + frac[2] * cell_matrix[6 + axis]
        for axis in range(3)
    ]


def _image_distance(frac_a, frac_b, cell_matrix):
    """Cartesian distance (Å) between two fractional points, no periodicity.

    Use this once a specific periodic image has been chosen (for example an
    image returned by ``UnitCell.find_nearest_pbc_images``).
    """
    delta = [float(frac_a[axis]) - float(frac_b[axis]) for axis in range(3)]
    cart = _cart_from_frac(delta, cell_matrix)
    return math.sqrt(sum(value * value for value in cart))


# --------------------------------------------------------------------------
# Symmetry
# --------------------------------------------------------------------------


def _space_group_ops(structure):
    space_group = structure.spacegroup
    if space_group is None:
        raise ValueError("The CIF does not define a space group that gemmi recognises.")
    return list(space_group.operations())


def _op_key(operation):
    rotation = []
    for row in operation.rot:
        if isinstance(row, (list, tuple)):
            rotation.extend(int(value) for value in row)
        else:
            rotation.append(int(row))
    return (tuple(rotation), tuple(int(value) for value in operation.tran))


def _expand_site(ops, unit_cell, frac):
    """Distinct symmetry images of a site, wrapped into [0, 1).

    Returns (fractional position, symmetry operation) pairs so anisotropic
    displacement tensors can be rotated with their image.
    """
    images = []
    for operation in ops:
        if any(
            _cart_distance(unit_cell, _symmetry_image(operation, frac), other[0]) <= _CART_TOLERANCE
            for other in images
        ):
            continue
        image = _wrapped_fractional(operation.apply_to_xyz(frac))
        images.append((image, operation))
    return images


def _symmetry_image(operation, frac):
    """Wrapped fractional position of ``frac`` under one symmetry operation."""
    return _wrapped_fractional(operation.apply_to_xyz(frac))


def _symmetry_mates(ops, unit_cell, frac):
    """Distinct symmetry-equivalent positions of a fractional point, P1 style.

    Thin convenience API over :func:`_expand_site` that returns just the
    distinct wrapped positions (no operations).  This is the foundation the
    deferred periodic-lattice / VESTA-style instance rendering consumes to
    enumerate all symmetry mates of an asymmetric-unit atom without needing a
    full site table or Blender.  Distinctness uses the same periodic
    ``_cart_distance`` tolerance as ``_expand_site``.
    """
    return [image for image, _operation in _expand_site(ops, unit_cell, frac)]


def _wrapped_fractional(frac):
    """Wrap fractional coordinates into [0, 1)."""
    wrapped = []
    for value in frac:
        value = float(value) % 1.0
        wrapped.append(0.0 if value >= 1.0 else value)
    return wrapped


# --------------------------------------------------------------------------
# CIF reading
# --------------------------------------------------------------------------


def _read_structure(cif_path, block_name=None):
    """Return (structure, block name, CIF block) for one crystal structure."""
    with open(cif_path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    # check_level=0 tolerates duplicate block names (some AMCSD files have two
    # data_global blocks); like cctbx, the last block with a given name wins.
    try:
        document = gemmi.cif.read_string(text, check_level=0)
    except Exception as exc:
        raise ValueError(f"This file could not be parsed as a CIF: {exc}") from exc
    structures = {}
    for block in document:
        # Some blocks are not small-structure blocks; skip them.
        try:
            structure = gemmi.make_small_structure_from_block(block)
        except Exception:  # nosec B112
            continue
        if not structure.sites or not structure.cell.a > 0.0:
            continue
        structures[block.name] = (structure, block)
    if not structures:
        raise ValueError("No crystal structures were found in this CIF.")
    if block_name:
        if block_name not in structures:
            raise ValueError(
                f"CIF block '{block_name}' was not found. "
                f"Available blocks: {', '.join(structures.keys())}"
            )
        key = block_name
    elif len(structures) == 1:
        key = next(iter(structures))
    else:
        raise ValueError(
            "This CIF contains multiple structures. Enter a CIF block name: "
            + ", ".join(structures.keys())
        )
    structure, block = structures[key]
    return structure, key, block


def _atom_type_charges(block, structure):
    """Formal oxidation numbers declared by the CIF, keyed by element.

    Two sources, in order of trust:

    1. the ``_atom_type`` loop (``_atom_type_oxidation_number``), used by ICSD
       and COD files.  The charge is read from ``_atom_type_symbol`` (``Ti+0``)
       rather than ``_atom_site_type_symbol`` because the two routinely
       disagree -- brookite declares ``Ti+0`` per site but ``4.000`` here.
    2. the charge gemmi parses from ``_atom_site_type_symbol`` (``Ag1+``),
       which is how CSD/AMCSD files usually carry it (Ag2Cr2O7).

    CSD organometallics usually declare neither, so an empty dict means "every
    atom is neutral" and connectivity stays purely distance based.
    """
    charges = {}
    try:
        symbols = list(block.find_values("_atom_type_symbol"))
        numbers = list(block.find_values("_atom_type_oxidation_number"))
    except Exception:
        symbols, numbers = [], []
    for symbol, number in zip(symbols, numbers, strict=False):
        text = str(number).strip().strip("'\"")
        if text in ("", "?", "."):
            continue
        try:
            charge = float(text)
        except ValueError:
            continue
        element = _element_from_text(symbol)
        if element is not None:
            charges[element.upper()] = charge
    for site in structure.sites:
        element = _site_element(site).upper()
        if element in charges:
            continue
        try:
            charge = float(site.charge)
        except (TypeError, ValueError):
            continue
        if abs(charge) > 1e-6:
            charges[element] = charge
    return charges


def _ionic_radii(charges):
    """Shannon radius of every element whose declared charge is tabulated."""
    radii = {}
    for element, charge in charges.items():
        radius = _IONIC_RADII.get((element.title(), int(round(charge))))
        if radius is not None:
            radii[element.upper()] = float(radius)
    return radii


def _metal_elements(structure):
    """Uppercase symbols of the metallic elements present in the structure."""
    metals = set()
    for site in structure.sites:
        symbol = _site_element(site)
        if not _is_element(symbol):
            continue
        # Unknown symbols are skipped rather than failing the whole CIF.
        try:
            if gemmi.Element(symbol).is_metal:
                metals.add(symbol.upper())
        except Exception:  # nosec B112
            continue
    return sorted(metals)


def _matrix3_from_columns(columns):
    """3x3 matrix whose columns are the three orthogonalized cell vectors."""
    return [
        [float(columns[0]), float(columns[3]), float(columns[6])],
        [float(columns[1]), float(columns[4]), float(columns[7])],
        [float(columns[2]), float(columns[5]), float(columns[8])],
    ]


def _matrix3_multiply(left, right):
    return [
        [sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _matrix3_transpose(matrix):
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _symmetry_rotation(operation):
    """Fractional rotation matrix of a gemmi symmetry operation."""
    flattened = []
    for row in operation.rot:
        if isinstance(row, (list, tuple)):
            flattened.extend(row)
        else:
            flattened.append(row)
    return [[float(flattened[row * 3 + column]) / 24.0 for column in range(3)] for row in range(3)]


def _site_u_cart(site, unit_cell, operation=None):
    """Cartesian ADP tensor (U11, U22, U33, U12, U13, U23) in A^2.

    CIF Uij values are expressed along the direct cell axes, so they are first
    converted to the reciprocal-basis tensor and then to Cartesian space.
    Symmetry images rotate the tensor with the operation that generated them.

    Careful: gemmi's ``aniso.elements_voigt()`` returns the *mechanics* Voigt
    order (U11, U22, U33, U23, U13, U12), not the standard CIF order —
    ``values[3]`` is U23 and ``values[5]`` is U12, which matches the
    off-diagonal layout of ``u_cif`` below (verified against cctbx's
    ``adptbx.u_cif_as_u_cart``).
    """
    aniso = getattr(site, "aniso", None)
    values = [0.0] * 6
    if aniso is not None:
        try:
            values = [float(value) for value in aniso.elements_voigt()]
        except Exception:
            values = [0.0] * 6
    if values[0] > 0.0 and values[1] > 0.0 and values[2] > 0.0:
        u_cif = [
            [values[0], values[5], values[4]],
            [values[5], values[1], values[3]],
            [values[4], values[3], values[2]],
        ]
        reciprocal = unit_cell.cell.reciprocal().parameters
        astar, bstar, cstar = (
            float(reciprocal[0]),
            float(reciprocal[1]),
            float(reciprocal[2]),
        )
        cell = _matrix3_from_columns(unit_cell._columns)
        scaling = _matrix3_multiply(
            cell,
            [
                [astar, 0.0, 0.0],
                [0.0, bstar, 0.0],
                [0.0, 0.0, cstar],
            ],
        )
        tensor = _matrix3_multiply(
            _matrix3_multiply(scaling, u_cif),
            _matrix3_transpose(scaling),
        )
        if operation is not None:
            fractional = _matrix3_from_columns(unit_cell._fractionalization)
            rotation = _matrix3_multiply(
                _matrix3_multiply(cell, _symmetry_rotation(operation)),
                fractional,
            )
            tensor = _matrix3_multiply(
                _matrix3_multiply(rotation, tensor),
                _matrix3_transpose(rotation),
            )
        return [
            tensor[0][0],
            tensor[1][1],
            tensor[2][2],
            tensor[0][1],
            tensor[0][2],
            tensor[1][2],
        ]
    u_iso = float(getattr(site, "u_iso", 0.0) or 0.0)
    if u_iso > 0.0:
        return [u_iso, u_iso, u_iso, 0.0, 0.0, 0.0]
    return None


def _dummy_labels(block):
    """Labels of the sites the CIF flags ``_atom_site_calc_flag`` = ``dum``.

    gemmi does not expose ``calc_flag`` on its sites, and a dummy site has no
    ``_atom_site_type_symbol``: the label fallback in :func:`_site_element`
    would then turn a ring centroid like ``Cg1`` into carbon, which bonds to
    every atom of the ring it sits on and destroys the ring (cyanoferrocene,
    COD 2005233).
    """
    labels = [str(value) for value in block.find_values("_atom_site_label")]
    flags = [str(value) for value in block.find_values("_atom_site_calc_flag")]
    return frozenset(
        label for label, flag in zip(labels, flags, strict=False) if flag.strip().lower() == "dum"
    )


# Occupancy difference below which two disorder components count as equally
# occupied: the unprimed label keeps the site.
_DISORDER_OCCUPANCY_EPSILON = 1e-3

# Suffixes that mark an alternative disorder component in a small-molecule CIF.
_DISORDER_PRIME_CHARACTERS = "'\""

# Two labels of one base name this close (A) are the *same atomic position*
# deposited once per component -- the signature of a prime-labelled disorder
# deposition.  Distinct atoms of a molecule are never within half an Angstrom
# of each other: the shortest real covalent bonds run around 1.2 A.  When no
# base carries such a coincident pair, trailing primes are an author's naming
# convention (a second molecule of the asymmetric unit, say), not disorder.
_DISORDER_OVERLAP_A = 0.5


def _disorder_overlaps(levels, unit_cell):
    """Whether a base name carries two components at one atomic position.

    A prime-labelled disorder deposition lists the same atom twice -- once per
    alternate component -- so some base names are represented by a plain label
    and a primed label that land on (almost) the same point (AKUQAB's thiophene
    keeps ``C1``/``C1'`` 0.08 A apart).  An author's numbering convention never
    does that: two distinct atoms of a molecule sit at least a covalent bond
    (around 1.2 A) apart.  A single coincident pair marks the whole block
    as a disorder deposition; without any, the primes are just names.
    """
    for by_level in levels.values():
        if len(by_level) < 2:
            continue
        for level_a, sites_a in by_level.items():
            for level_b, sites_b in by_level.items():
                if level_b <= level_a:
                    continue
                for site_a in sites_a:
                    for site_b in sites_b:
                        distance = _cart_distance(unit_cell, site_a.fract, site_b.fract)
                        if distance <= _DISORDER_OVERLAP_A:
                            return True
    return False


def _disorder_minor_labels(structure, block, unit_cell):
    """Labels of the alternate disorder components that load must omit.

    CSD CIFs label the second and third components of a disordered site with a
    trailing prime or double prime (``S1'``, ``S1''``) while the first component
    keeps the plain label.  The omission is per base label: a primed label is
    only dropped when an unprimed sibling of the same base exists, so an atom
    that legitimately carries a prime with no counterpart (``C12'`` and no
    ``C12``) is kept.

    Occupancy wins when the CIF declares it: a base whose components differ in
    occupancy keeps the more occupied one and drops the other, so a primed
    component can survive while the plain one goes.  A base whose components
    are equally occupied abstains, which keeps fixed hydrogen occupancies from
    skewing the choice; without an occupancy column the primed (and
    double-primed) components are dropped and the unprimed one kept.

    The suffix is a labelling convention, not a CIF guarantee: an author can
    also number a second independent molecule of the asymmetric unit with
    primes.  :func:`_disorder_overlaps` tells the two apart from the deposited
    geometry -- a genuine disorder deposition lists the same atomic position
    twice (once per component), while a naming convention never puts two
    distinct atoms within half an Angstrom.  When no base name overlaps itself
    that way, the block's primes are pure names and no level is removed unless
    occupancy says otherwise; :func:`load_structure` reports any real loss in
    ``omitted_disorder`` so the import message makes it visible.  The formal
    ``_atom_site_disorder_group`` tag is not consulted.
    """
    levels = {}
    for site in structure.sites:
        label = str(site.label)
        base = label.rstrip(_DISORDER_PRIME_CHARACTERS)
        if not base:
            continue
        levels.setdefault(base, {}).setdefault(len(label) - len(base), []).append(site)
    occupancy_declared = any(
        str(value).strip() for value in block.find_values("_atom_site_occupancy")
    )

    def mean_occupancy(level_sites):
        values = []
        for site in level_sites:
            try:
                value = float(site.occ)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return sum(values) / len(values) if values else None

    def clear_winner(by_level):
        """Prime level this base's declared occupancies prefer, if any."""
        if not occupancy_declared or len(by_level) < 2:
            return None
        means = {level: mean_occupancy(sites) for level, sites in by_level.items()}
        means = {level: mean for level, mean in means.items() if mean is not None}
        if len(means) < 2:
            return None
        ordered = sorted(means.values(), reverse=True)
        if ordered[0] - ordered[1] <= _DISORDER_OCCUPANCY_EPSILON:
            return None
        return max(means, key=means.get)

    # One vote per base that clearly prefers a level: bases whose levels are
    # tied abstain, so a structure-wide component (all primed atoms, say) wins
    # even when a few fixed-occupancy atoms vote the other way.
    votes = {}
    for by_level in levels.values():
        winner = clear_winner(by_level)
        if winner is not None:
            votes[winner] = votes.get(winner, 0) + 1
    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    preferred = ranked[0][0] if ranked else 0

    naming_convention = not _disorder_overlaps(levels, unit_cell)

    dropped = set()
    for by_level in levels.values():
        if len(by_level) < 2:
            continue
        winner = clear_winner(by_level)
        if winner is None:
            if naming_convention and not preferred:
                # Every base is a distinct atom's worth of geometry: the
                # primes are only names, so occupancy evidence alone (above)
                # may remove a level.
                continue
            winner = preferred if preferred in by_level else 0
        if winner not in by_level:
            winner = min(by_level)
        for level, level_sites in by_level.items():
            if level != winner:
                dropped.update(str(site.label) for site in level_sites)
    return frozenset(dropped)


def _disorder_question_labels(structure):
    """Labels of the ``?``-suffixed alternate disorder components to omit.

    Some CSD deposits number the second component of a disordered site on its
    own lines and flag it with a trailing question mark (PEPQAG's ring flips
    sit ``H2?`` 0.06 A from ``H1``, ``C32?`` 0.36 A from ``S1``).  There is no
    shared base name for the prime pairing to see, so the ``?`` mark itself is
    the tell, and unlike the primes it is unconditional: every site carrying a
    trailing ``?`` is an alternate component and is ignored, with no geometry
    gate and no occupancy vote.
    """
    return frozenset(str(site.label) for site in structure.sites if str(site.label).endswith("?"))


def _skip_site(site, skip_labels):
    """True for a dummy site or a site with an explicit zero occupancy."""
    if site.label in skip_labels:
        return True
    try:
        return float(site.occ) <= 0.0
    except (TypeError, ValueError):
        return False


def _p1_sites(structure, unit_cell, skip_labels=frozenset()):
    """Cell contents in P1 as a small table of sites."""
    ops = _space_group_ops(structure)
    sites = []
    for site in structure.sites:
        if _skip_site(site, skip_labels):
            continue
        element = _site_element(site)
        vdw_radius, covalent_radius = _radius_tables(element)
        frac = [float(v) for v in site.fract]
        for image, operation in _expand_site(ops, unit_cell, frac):
            site_data = {
                "label": site.label,
                "source_label": site.label,
                "element": element.upper(),
                "frac": image,
                "occupancy": float(site.occ),
                "vdw_radius": vdw_radius,
                "covalent_radius": covalent_radius,
            }
            u_cart = _site_u_cart(site, unit_cell, operation)
            if u_cart is not None:
                site_data["u_cart"] = u_cart
            sites.append(site_data)
    return sites


def load_structure(cif_path, block_name=None, keep_disorder=False):
    """Return the P1 site table and the instances that make up the ASU."""
    structure, key, block = _read_structure(cif_path, block_name)
    unit_cell = _UnitCell(structure.cell.parameters)
    ops = _space_group_ops(structure)
    dummy_labels = _dummy_labels(block)
    if keep_disorder:
        # The caller asked for every disorder component (the import operator's
        # "Keep All Disorder Components" toggle): neither the prime- nor the
        # ?-labelled alternates are omitted, only the space-group resolution
        # below keeps both combinations as separate components.
        disorder_labels = frozenset()
    else:
        disorder_labels = (
            _disorder_minor_labels(structure, block, unit_cell)
            | _disorder_question_labels(structure)
        )
    skip_labels = dummy_labels | disorder_labels
    sites = _p1_sites(structure, unit_cell, skip_labels)
    asu_fracs = {
        site.label: [float(value) for value in site.fract]
        for site in structure.sites
        if not _skip_site(site, skip_labels)
    }
    # Resolve the disorder the space group itself generates (a half-occupied
    # substituent whose alternate position is not a second CIF label).  This
    # runs after the prime-label omission, so the two rules compose: primed
    # alternate components are already gone, and only anchored groups whose
    # anchor sits on the symmetry element are touched here.
    resolution = _resolve_symmetry_disorder(
        sites, unit_cell, ops, asu_fracs, keep_disorder=keep_disorder
    )
    sites = resolution["sites"]
    base_atoms = _base_atoms(unit_cell, sites)
    lookup = _base_atom_lookup(unit_cell, base_atoms)
    asu_atoms = []
    for site in structure.sites:
        if _skip_site(site, skip_labels):
            continue
        frac = resolution["asu_fracs"].get(site.label)
        if frac is None:
            frac = [float(value) for value in site.fract]
        asu_atoms.append(
            {
                "label": site.label,
                "element": _site_element(site).upper(),
                "frac": frac,
                "cart": unit_cell.orthogonalize(frac),
            }
        )
    instances = _instances_from_atoms(unit_cell, base_atoms, lookup, asu_atoms)
    if not instances:
        raise ValueError("The CIF atoms could not be placed in the unit cell.")
    space_group = structure.spacegroup
    charges = _atom_type_charges(block, structure)
    return {
        "label": key,
        "space_group": f"{space_group.xhm()} (No. {space_group.number})",
        "space_group_ops": [op.triplet() for op in ops],
        "unit_cell": unit_cell.parameters(),
        "unit_cell_display": [
            str(block.find_value(tag) or "")
            for tag in (
                "_cell_length_a",
                "_cell_length_b",
                "_cell_length_c",
                "_cell_angle_alpha",
                "_cell_angle_beta",
                "_cell_angle_gamma",
            )
        ],
        "cell_matrix": list(unit_cell._columns),
        "sites": sites,
        "instances": instances,
        # Number of alternate disorder components omitted at load, so the
        # import message can report them (see _disorder_minor_labels and
        # _disorder_question_labels).
        "omitted_disorder": len(disorder_labels),
        # Number of symmetry-disordered groups resolved to one component, and
        # the number of alternate P1 images dropped to do so (0 when the caller
        # asked to keep every component).
        "resolved_disorder": resolution["resolved_groups"],
        "dropped_disorder": resolution["dropped_sites"],
        # Connectivity hints: declared oxidation numbers, Shannon radii for the
        # charged elements, and which elements gemmi considers metallic.  All
        # three are used by the add-on to derive the default distance windows.
        "atom_type_charges": charges,
        "ionic_radii": _ionic_radii(charges),
        "metal_elements": _metal_elements(structure),
    }


# --------------------------------------------------------------------------
# Growth (ported from the cctbx worker)
# --------------------------------------------------------------------------


def _enabled_windows(pair_settings):
    """Return {sorted element pair: (min_distance, max_distance)}."""
    windows = {}
    for setting in pair_settings:
        element_a = str(setting.get("element_a", "")).upper()
        element_b = str(setting.get("element_b", "")).upper()
        minimum = float(setting.get("min_distance", 0.0))
        maximum = float(setting.get("max_distance", 0.0))
        if setting.get("enabled", False) and 0.0 <= minimum <= maximum:
            windows[tuple(sorted((element_a, element_b)))] = (minimum, maximum)
    return windows


def _fractional_tolerance(unit_cell, cart_tolerance):
    """Per-axis fractional tolerance equivalent to cart_tolerance (Angstrom).

    ``fractionalization_matrix()`` stores ``M^-T`` (the transpose of the true
    inverse orthogonalization matrix) row-major: element ``(row, col)`` is
    ``M^-1[col][row]``.  A Cartesian displacement ``d`` becomes ``M^-1 d`` in
    fractional coordinates, so the largest change on axis ``i`` is
    ``||row_i(M^-1)|| * |d|``.  Row ``i`` of ``M^-1`` is column ``i`` of the
    stored matrix, i.e. the strided elements ``(0, i), (1, i), (2, i)``.
    Reading the contiguous rows instead mixes the axes and is only correct
    when the cell is orthogonal.
    """
    matrix = list(unit_cell.fractionalization_matrix())
    return [
        cart_tolerance * math.sqrt(sum(matrix[row * 3 + axis] ** 2 for row in range(3)))
        for axis in range(3)
    ]


def _canonical_base_indices(base_atoms):
    """Index of the representative atom for every (element, site) group.

    Disordered CIFs can list the same element twice at one site (the P1
    expansion then contains exact duplicates). Growth treats those as a single
    atom, like the position based duplicate check of the original
    implementation did.

    Two sites of the same element count as one atom whenever their Cartesian
    positions coincide to within ``_DUPLICATE_POSITION_TOLERANCE`` (0.01 A).
    The bucket scan looks at the 27 neighbouring cells of a ``tolerance``-sized
    grid, so any two points within the tolerance land in the same or an
    adjacent cell and share a representative; the strict distance check then
    decides.  This merges both the exact copies of a disorder site and the
    pseudo-symmetry twins of a redeposited full-molecule CIF (the same atom
    listed twice with 5-decimal fractional coordinates can differ by up to a
    few 1e-4 A), so ``grow_connectivity_shell`` never adds an atom sitting on
    top of one already on screen -- which would otherwise give a ring carbon
    an extra (duplicate) hydrogen and break its aromatic-ring eligibility.
    """
    tolerance = _DUPLICATE_POSITION_TOLERANCE
    tolerance_squared = tolerance * tolerance
    grid = {}
    canonical = []
    for index, atom in enumerate(base_atoms):
        element = atom["element"]
        cart = atom["cart"]
        cell = tuple(int(math.floor(value / tolerance)) for value in cart)
        rep_index = index
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    bucket = (element, cell[0] + da, cell[1] + db, cell[2] + dc)
                    for candidate_cart, candidate_index in grid.get(bucket, ()):
                        if (
                            sum((cart[axis] - candidate_cart[axis]) ** 2 for axis in range(3))
                            <= tolerance_squared
                        ):
                            rep_index = candidate_index
                            break
                    if rep_index != index:
                        break
                if rep_index != index:
                    break
            if rep_index != index:
                break
        if rep_index == index:
            grid.setdefault((element, *cell), []).append((cart, index))
        canonical.append(rep_index)
    return canonical


def _base_atom_lookup(unit_cell, base_atoms):
    """Bucket the P1 cell contents for periodic fractional lookups.

    Buckets wrap around the cell, so the 27 neighbouring buckets of a query
    always contain every base atom within the matching tolerance, no matter
    where in the cell the atom sits.
    """
    tolerance = _fractional_tolerance(unit_cell, _CART_TOLERANCE)
    buckets = max(1, int(math.ceil(1.0 / max(tolerance))))
    width = 1.0 / buckets
    grid = {}
    for index, atom in enumerate(base_atoms):
        wrapped = _wrapped_fractional(atom["frac"])
        cell = tuple(int(math.floor(value / width)) % buckets for value in wrapped)
        grid.setdefault(cell, []).append((index, wrapped))
    return {
        "tolerance": tolerance,
        "buckets": buckets,
        "width": width,
        "grid": grid,
    }


def _match_base_atom(unit_cell, base_atoms, lookup, frac, cart, element=None, label=None):
    """Return (base atom index, integer cell translation) or None.

    Disordered CIFs can list different elements at the same site, so a
    position alone is ambiguous. Candidates of the same label win, then
    candidates of the same element, and only then anything close enough.
    """
    tolerance = lookup["tolerance"]
    buckets = lookup["buckets"]
    width = lookup["width"]
    wrapped = _wrapped_fractional(frac)
    cell = tuple(int(math.floor(value / width)) % buckets for value in wrapped)
    best = None
    best_other_element = None
    best_label = None
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbour = (
                    (cell[0] + da) % buckets,
                    (cell[1] + db) % buckets,
                    (cell[2] + dc) % buckets,
                )
                for candidate, base_wrapped in lookup["grid"].get(neighbour, ()):
                    delta = [wrapped[axis] - base_wrapped[axis] for axis in range(3)]
                    if any(
                        abs(value - round(value)) > tolerance[axis]
                        for axis, value in enumerate(delta)
                    ):
                        continue
                    base_frac = base_atoms[candidate]["frac"]
                    offset = tuple(
                        int(round(float(frac[axis]) - base_frac[axis])) for axis in range(3)
                    )
                    if cart is None:
                        return candidate, offset
                    expected = unit_cell.orthogonalize(
                        [base_frac[axis] + offset[axis] for axis in range(3)]
                    )
                    distance_squared = sum((expected[axis] - cart[axis]) ** 2 for axis in range(3))
                    if distance_squared > _CART_TOLERANCE**2:
                        continue
                    found = (distance_squared, candidate, offset)
                    if label is not None and base_atoms[candidate]["label"] == label:
                        if best_label is None or distance_squared < best_label[0]:
                            best_label = found
                        continue
                    if element is not None and base_atoms[candidate]["element"] != element:
                        if best_other_element is None or distance_squared < best_other_element[0]:
                            best_other_element = found
                        continue
                    if best is None or distance_squared < best[0]:
                        best = found
    if best_label is not None:
        best = best_label
    if best is None:
        best = best_other_element
    if best is None:
        return None
    return best[1], best[2]


def _base_atoms(unit_cell, sites):
    """Worker-side view of the site table: element, fractional and cartesian."""
    atoms = []
    for site in sites:
        frac = [float(value) for value in site["frac"]]
        atoms.append(
            {
                "label": site.get("label"),
                "element": str(site.get("element", "X")).upper(),
                "frac": frac,
                "cart": unit_cell.orthogonalize(frac),
            }
        )
    return atoms


def _instances_from_atoms(unit_cell, base_atoms, lookup, atoms):
    """Match atom dictionaries to (site, translation) instances, keeping order."""
    instances = []
    for atom in atoms:
        frac = atom.get("frac")
        cart = atom.get("cart")
        if frac is None and cart is not None:
            frac = unit_cell.fractionalize(cart)
        if frac is None:
            continue
        match = _match_base_atom(
            unit_cell,
            base_atoms,
            lookup,
            frac,
            cart,
            str(atom.get("element", "X")).upper(),
            atom.get("label"),
        )
        if match is None:
            continue
        instances.extend(
            (
                match[0],
                match[1][0],
                match[1][1],
                match[1][2],
            )
        )
    return instances


def _instance_keys(instances, canonical):
    """Turn a flat [site, tx, ty, tz, ...] list into exact growth keys."""
    keys = set()
    count = len(instances) - len(instances) % 4
    for offset in range(0, count, 4):
        site = int(instances[offset])
        if not 0 <= site < len(canonical):
            continue
        keys.add(
            (
                canonical[site],
                (
                    int(instances[offset + 1]),
                    int(instances[offset + 2]),
                    int(instances[offset + 3]),
                ),
            )
        )
    return keys


def _periodic_neighbour_table_gemmi(
    unit_cell, base_atoms, base_elements, windows_by_element, canonical=None
):
    """For each cell atom, list its bonded (base atom, cell offset) pairs.

    Uses gemmi's UnitCell.find_nearest_pbc_images() (C++ search) to resolve the
    exact periodic images.  To avoid an O(N^2) all-pairs scan, a gemmi
    NeighborSearch is built once (as a pure spatial index) and used only to
    prune the candidate ``j`` list to atoms actually within the largest enabled
    window; the authoritative image/translation still comes from
    find_nearest_pbc_images().  Because NeighborSearch only shrinks the set of
    ``(i, j)`` pairs checked (never changes which atoms pair up or their
    (tx, ty, tz)), the returned (candidate_index, tx, ty, tz) tuples are
    byte-identical to the all-pairs scan on the example CIFs.

    Both seeds and candidates are restricted to canonical atoms.  A
    non-canonical atom is a duplicate copy of another site (same element,
    same position to ``_DUPLICATE_POSITION_TOLERANCE``), so listing it as a
    candidate would let growth add an atom sitting on top of its
    representative whenever the enabled window admits the sub-tolerance
    separation -- i.e. with ``min_distance`` set to 0, where the
    ``dist <= 1e-12`` self filter no longer catches it.
    """
    cell = unit_cell.cell  # gemmi.UnitCell
    cell_matrix = unit_cell._columns  # column-major, matches _cart_from_frac

    # Spatial index over the base atoms for candidate pruning.  radius covers
    # the largest enabled window; a tiny epsilon keeps boundary hits safe (any
    # extra candidate is filtered back out by the authoritative distance check).
    global_max = 0.0
    for pair_windows in windows_by_element.values():
        for bounds in pair_windows.values():
            global_max = max(global_max, bounds[1])
    index = None
    if global_max > 0.0 and base_atoms:
        index = _neighbor_search_index(unit_cell, base_atoms, global_max + 1e-4)

    table = []
    for base_index, base_atom in enumerate(base_atoms):
        if canonical is not None and canonical[base_index] != base_index:
            table.append(())
            continue
        pair_windows = windows_by_element.get(base_elements[base_index])
        if not pair_windows:
            table.append(())
            continue
        frac_i = gemmi.Fractional(*base_atom["frac"])
        if index is None:
            candidates = list(range(len(base_atoms)))
        else:
            search, sites = index
            candidates = sorted(
                {
                    mark.atom_idx
                    for mark in search.find_site_neighbors(
                        sites[base_index], 0.0, global_max + 1e-4
                    )
                }
            )
        neighbours = []
        seen = set()
        for j in candidates:
            if canonical is not None and canonical[j] != j:
                # Duplicate copy of another site; the representative lists its
                # neighbours already.
                continue
            element_j = base_elements[j]
            bounds = pair_windows.get(element_j)
            if bounds is None:
                continue
            frac_j = gemmi.Fractional(*base_atoms[j]["frac"])
            for im in cell.find_nearest_pbc_images(frac_i, bounds[1], frac_j, 0):
                f_img = cell.fract_image(im, frac_j)
                tx = int(round(f_img.x - frac_j.x))
                ty = int(round(f_img.y - frac_j.y))
                tz = int(round(f_img.z - frac_j.z))
                dist = _image_distance(base_atom["frac"], [f_img.x, f_img.y, f_img.z], cell_matrix)
                if dist <= 1e-12:
                    continue
                if dist < bounds[0]:
                    continue
                key = (j, tx, ty, tz)
                if key not in seen:
                    seen.add(key)
                    neighbours.append((j, tx, ty, tz))
        table.append(tuple(neighbours))
    return table


def _neighbor_search_index(unit_cell, base_atoms, radius):
    """Build a gemmi.NeighborSearch spatial index over the base atoms.

    Returns ``(search, sites)`` where ``sites`` parallels ``base_atoms`` so a
    query atom can be addressed by base-atom index.
    """
    sites = gemmi.SmallStructure.SiteList()
    for i, atom in enumerate(base_atoms):
        site = gemmi.SmallStructure.Site()
        site.fract = gemmi.Fractional(*atom["frac"])
        site.type_symbol = atom["element"]
        site.label = atom.get("label") or str(i)
        sites.append(site)
    structure = gemmi.SmallStructure()
    structure.cell = unit_cell.cell
    structure.sites = sites
    search = gemmi.NeighborSearch(structure, radius)
    for n, site in enumerate(structure.sites):
        search.add_site(site, n)
    return search, sites


# --------------------------------------------------------------------------
# Symmetry-generated disorder
# --------------------------------------------------------------------------

# A partially occupied site is disorder rather than an under-occupied atom
# when it is close enough to fully occupied (e.g. 0.5 + 0.5).
_DISORDER_MIN_OCCUPANCY = 0.999
# Bond window and maximum coordination number used only to decide whether an
# anchored group resolves to a chemically valid conformer.  The window is the
# same 1.25 * (r_a + r_b) used by the resolver prototype; it is deliberately
# tighter than the display windows so a genuine bond is never mistaken for a
# clash, while the impossible 1.31 A methyl-methyl contact of the Ca case is.
_DISORDER_BOND_SCALE = 1.25
_DISORDER_MAX_VALENCE = {
    "H": 1,
    "D": 1,
    "B": 3,
    "C": 4,
    "N": 4,
    "O": 2,
    "F": 1,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}
_DISORDER_DEFAULT_VALENCE = 8
# Bounds that keep the per-anchor search cheap on large CIFs.
_DISORDER_MAX_CLUSTER_SITES = 64
_DISORDER_SEARCH_BUDGET = 20000
_DISORDER_MAX_SOLUTIONS = 64


def _disorder_is_identity(operation):
    """True for the identity operation, lattice translations included.

    gemmi stores the rotation and translation in integer form scaled by
    ``Op.DEN``; a pure lattice translation (``x+1,y,z``) is the identity in a
    periodic cell and must not count as the symmetry element that generates a
    disorder.
    """
    denominator = int(operation.DEN)
    rotation = tuple(int(value) for row in operation.rot for value in row)
    if rotation != (denominator, 0, 0, 0, denominator, 0, 0, 0, denominator):
        return False
    return all(int(value) % denominator == 0 for value in operation.tran)


def _disorder_max_valence(element):
    """Largest coordination number a fully occupied anchor can support."""
    return _DISORDER_MAX_VALENCE.get(str(element).upper(), _DISORDER_DEFAULT_VALENCE)


def _disorder_fixing_operations(ops, frac, unit_cell):
    """Non-identity space-group operations that leave *frac* fixed.

    A symmetry-generated disorder needs the anchor itself to sit on the
    symmetry element: the mirror that swaps the two methyl orientations of
    ``CIF/Ca.cif`` fixes the quaternary carbon, while a general-position atom
    merely bonded to a partially occupied neighbour (a sulfonate oxygen next to
    the disordered potassium/water region) is not fixed by any operation.
    """
    fixing = []
    for operation in ops:
        if _disorder_is_identity(operation):
            continue
        image = _symmetry_image(operation, frac)
        if _cart_distance(unit_cell, image, frac) <= _CART_TOLERANCE:
            fixing.append(operation)
    return fixing


def _disorder_images_closed_under(operation, fracs, unit_cell):
    """True when *operation* permutes the image positions among themselves."""
    for frac in fracs:
        image = _symmetry_image(operation, frac)
        if not any(_cart_distance(unit_cell, image, other) <= _CART_TOLERANCE for other in fracs):
            return False
    return True


def _disorder_anchor_solutions(
    sites,
    unit_cell,
    asu_fracs,
    anchor,
    anchor_edges,
    partial_edges,
    is_partial,
    cluster,
    depth,
):
    """Enumerate the chemically valid image combinations of one anchored cluster.

    The search walks the cluster's labels parents-first (``depth`` was assigned
    outward from the anchor) and only places an image that stays bonded to the
    anchor or to an already chosen image (``connected``) while keeping every
    chosen atom within its maximum coordination number (``within_valence`` and
    the ``valid_solution`` check -- the same valence cap that rejects the
    overlapping images of a mirror-generated *tert*-butyl).  A completed
    combination scores one point per CIF identity position it keeps, so a call
    with more unchanged ASU atoms outranks one that relocates them.

    Returns ``{"order", "candidates", "solutions"}`` with the solutions sorted
    best-first (score descending, then the chosen site-index tuple).
    """
    labels = sorted({sites[index]["label"] for index in cluster})
    candidates = {
        label: sorted(index for index in cluster if sites[index]["label"] == label)
        for label in labels
    }
    label_depth = {label: min(depth[index] for index in candidates[label]) for label in labels}
    order = sorted(labels, key=lambda label: (label_depth[label], label))

    chosen = {}
    chosen_sites = set()
    solutions = []
    budget = [_DISORDER_SEARCH_BUDGET]

    def connected(index):
        return any(
            other == anchor or other in chosen_sites for other in partial_edges.get(index, ())
        )

    def within_valence(index):
        count = 0
        for other in partial_edges.get(index, ()):
            if other == anchor or other in chosen_sites or not is_partial(other):
                count += 1
        return count <= _disorder_max_valence(sites[index]["element"])

    def valid_solution():
        for index in chosen_sites:
            count = sum(
                1
                for other in partial_edges.get(index, ())
                if other in chosen_sites or not is_partial(other)
            )
            if count > _disorder_max_valence(sites[index]["element"]):
                return False
        anchor_count = sum(
            1 for other in anchor_edges if other in chosen_sites or not is_partial(other)
        )
        return anchor_count <= _disorder_max_valence(sites[anchor]["element"])

    def walk(position):
        if budget[0] <= 0 or len(solutions) >= _DISORDER_MAX_SOLUTIONS:
            return
        budget[0] -= 1
        if position == len(order):
            if not valid_solution():
                return
            score = 0
            for label, index in chosen.items():
                wanted = asu_fracs.get(label)
                if wanted is None:
                    continue
                identity = _wrapped_fractional(wanted)
                if _cart_distance(unit_cell, sites[index]["frac"], identity) <= _CART_TOLERANCE:
                    score += 1
            solutions.append((score, tuple(chosen[label] for label in order)))
            return
        label = order[position]
        for index in candidates[label]:
            if not connected(index) or not within_valence(index):
                continue
            chosen[label] = index
            chosen_sites.add(index)
            walk(position + 1)
            chosen_sites.discard(index)
            del chosen[label]

    walk(0)
    solutions.sort(key=lambda solution: (-solution[0], solution[1]))
    return {
        "order": order,
        "candidates": candidates,
        "solutions": solutions,
    }


def _resolve_symmetry_disorder(sites, unit_cell, ops, asu_fracs, keep_disorder=False):
    """Resolve symmetry-generated disorder of anchored groups at load time.

    A partially occupied substituent whose alternate position is produced by
    the space group (not written as a second CIF label) shows up twice after
    the P1 expansion.  ``CIF/Ca.cif`` is the driving case: the mirror plane
    through ``C20`` turns one *tert*-butyl group into six half-occupied methyl
    images, of which only one chemically valid combination exists per anchor.

    Only *anchored* groups are resolved.  An anchor is a fully occupied atom
    whose remaining coordination capacity (max valence minus its fully occupied
    neighbours) equals the number of distinct partially occupied labels bonded
    to it, where every one of those labels must have at least two images around
    the anchor and be permuted by a symmetry operation that fixes the anchor.
    That keeps genuine substitutional disorder (the potassium/water region of
    ``Ca.cif``, whose atoms sit on general positions and are not alternatives of
    one another) untouched.

    For each anchor the candidate images are enumerated parents-first; a
    candidate must stay bonded to the anchor or an already chosen image, and no
    chosen atom may exceed its maximum coordination number.  The clash check is
    the valence cap itself: the invalid ``C21``-``C22`` contact (1.31 A) falls
    inside the bond window and would give both carbons five neighbours.  Among
    the valid combinations the one keeping the most CIF images (identity
    positions) wins, with the site-index tuple as a deterministic tie-break.

    With ``keep_disorder`` every image is kept; each valid combination becomes
    a component and the returned sites carry a ``component`` key so bonding can
    refuse cross-component pairs.  Otherwise the losing images are dropped and
    the kept ones are promoted to full occupancy.

    Returns a dict with the new site table, the relocated asymmetric-unit
    fractional positions (``{label: frac}``), the number of resolved anchors
    (distinct disordered groups of the ASU) and the number of dropped P1 images.
    """
    partial_labels = {}
    for index, site in enumerate(sites):
        if float(site.get("occupancy", 1.0)) < _DISORDER_MIN_OCCUPANCY:
            partial_labels.setdefault(site["label"], []).append(index)
    if not partial_labels:
        return {
            "sites": sites,
            "asu_fracs": {},
            "resolved_groups": 0,
            "dropped_sites": 0,
        }

    def is_partial(index):
        return sites[index]["label"] in partial_labels

    max_radius = max(float(site["covalent_radius"]) for site in sites)
    index_radius = _DISORDER_BOND_SCALE * 2.0 * max_radius
    base_atoms = _base_atoms(unit_cell, sites)
    search, gsites = _neighbor_search_index(unit_cell, base_atoms, index_radius)
    positions = [gemmi.Position(*unit_cell.orthogonalize(site["frac"])) for site in sites]

    def edges_of(index):
        """Neighbouring site indices inside the resolver's bond window.

        The gemmi spatial index prunes the candidate set; the exact minimum
        image distance is then measured by the search itself.  Each distinct
        periodic image is kept, so an atom bonded twice to the same site (a
        bidentate contact) still counts twice for its coordination number.
        """
        radius = float(sites[index]["covalent_radius"])
        found = []
        seen = set()
        for mark in search.find_site_neighbors(gsites[index], 0.0, index_radius):
            other = int(mark.atom_idx)
            if other == index:
                continue
            key = (other, int(mark.image_idx))
            if key in seen:
                continue
            seen.add(key)
            distance = search.dist(positions[index], mark.pos)
            if distance <= 1e-6:
                continue
            if distance <= _DISORDER_BOND_SCALE * (radius + float(sites[other]["covalent_radius"])):
                found.append(other)
        return found

    partial_edges = {}
    for indices in partial_labels.values():
        for index in indices:
            partial_edges[index] = edges_of(index)

    candidate_anchors = set()
    for neighbours in partial_edges.values():
        for other in neighbours:
            if not is_partial(other):
                candidate_anchors.add(other)

    resolutions = []
    for anchor in sorted(candidate_anchors):
        anchor_edges = edges_of(anchor)
        full_bonds = sum(1 for other in anchor_edges if not is_partial(other))
        groups = {}
        for other in anchor_edges:
            if is_partial(other):
                groups.setdefault(sites[other]["label"], set()).add(other)
        if not groups:
            continue
        if full_bonds + len(groups) != _disorder_max_valence(sites[anchor]["element"]):
            continue
        multi = {label: sorted(indices) for label, indices in groups.items() if len(indices) >= 2}
        if not multi:
            continue
        fixing = _disorder_fixing_operations(ops, sites[anchor]["frac"], unit_cell)
        if not fixing:
            continue
        explained = True
        for indices in multi.values():
            fracs = [sites[index]["frac"] for index in indices]
            if not any(
                _disorder_images_closed_under(operation, fracs, unit_cell) for operation in fixing
            ):
                explained = False
                break
        if not explained:
            continue

        cluster = set()
        depth = {}
        for indices in groups.values():
            for index in indices:
                cluster.add(index)
                depth[index] = 1
        frontier = list(cluster)
        while frontier:
            index = frontier.pop()
            for other in partial_edges.get(index, ()):
                if is_partial(other) and other not in cluster:
                    cluster.add(other)
                    depth[other] = depth[index] + 1
                    frontier.append(other)
        if len(cluster) > _DISORDER_MAX_CLUSTER_SITES:
            continue

        found = _disorder_anchor_solutions(
            sites,
            unit_cell,
            asu_fracs,
            anchor,
            anchor_edges,
            partial_edges,
            is_partial,
            cluster,
            depth,
        )
        order = found["order"]
        candidates = found["candidates"]
        solutions = found["solutions"]
        if not solutions:
            continue
        resolutions.append(
            {
                "anchor": anchor,
                "cluster": cluster,
                "order": order,
                "candidates": candidates,
                "best": solutions[0][1],
                "solutions": solutions,
            }
        )

    if not resolutions:
        return {
            "sites": sites,
            "asu_fracs": {},
            "resolved_groups": 0,
            "dropped_sites": 0,
        }

    resolved_groups = len({sites[resolution["anchor"]]["label"] for resolution in resolutions})
    kept = set()
    candidates_all = set()
    for resolution in resolutions:
        kept.update(resolution["best"])
        candidates_all.update(resolution["cluster"])
    dropped = candidates_all - kept

    if keep_disorder:
        # Every image stays, so the ASU keeps its CIF coordinates verbatim; the
        # component ids are what stops the alternatives from being bonded to
        # each other.
        new_sites = [dict(site) for site in sites]
        component = 1
        for resolution in resolutions:
            assigned = {}
            for _score, combination in resolution["solutions"]:
                for index in combination:
                    if index not in assigned:
                        assigned[index] = component
                component += 1
            for index in sorted(resolution["cluster"]):
                if index in assigned:
                    new_sites[index]["component"] = assigned[index]
                else:
                    new_sites[index]["component"] = component
                    component += 1
        return {
            "sites": new_sites,
            "asu_fracs": {},
            "resolved_groups": resolved_groups,
            "dropped_sites": 0,
        }

    # The ASU atom must sit on a kept image (the loader matches it back to the
    # site table), so move it to the chosen conformer unless the CIF already
    # gave that image.
    asu_relocated = {}
    for resolution in resolutions:
        order = resolution["order"]
        best_map = dict(zip(order, resolution["best"], strict=True))
        best_sites = set(resolution["best"])
        for label in order:
            wanted = asu_fracs.get(label)
            if wanted is None:
                continue
            identity = _wrapped_fractional(wanted)
            identity_index = None
            for index in resolution["candidates"][label]:
                if _cart_distance(unit_cell, sites[index]["frac"], identity) <= _CART_TOLERANCE:
                    identity_index = index
                    break
            if identity_index is None or identity_index in best_sites:
                continue
            chosen = sites[best_map[label]]["frac"]
            # Keep the atom in the same cell neighbourhood as the CIF position:
            # the ASU fractional coordinates are not wrapped (C23 of Ca.cif has
            # z = 1.0334), so the chosen image needs the lattice translation
            # that leaves it nearest the original position.
            asu_relocated[label] = [
                float(chosen[axis]) + int(round(wanted[axis] - chosen[axis])) for axis in range(3)
            ]

    new_sites = []
    for index, site in enumerate(sites):
        if index in dropped:
            continue
        if index in kept:
            site = dict(site)
            site["occupancy"] = 1.0
        new_sites.append(site)
    return {
        "sites": new_sites,
        "asu_fracs": asu_relocated,
        "resolved_groups": resolved_groups,
        "dropped_sites": len(dropped),
    }


def _grow_shell_keys(
    unit_cell, base_atoms, canonical, visited, seed_keys, windows, neighbour_table=None
):
    """Return the instance keys of the next coordination shell.

    A new atom must lie inside one enabled distance window of a seed atom, so
    growth is expressed as a graph walk instead of a bounding box over the
    whole structure:

    1. build the periodic neighbour table of one unit cell (depends only on
       the cell contents and the windows),
    2. walk one step from every seed key, skipping keys already present.

    Cost is O(cell atoms * local density + seed atoms * coordination number).
    It does not depend on the extent of the displayed structure, so repeated
    growth no longer re-enumerates the volume around it.
    """
    if not windows:
        raise ValueError("Enable at least one valid element-pair distance window.")
    max_distance = max(bound[1] for bound in windows.values())
    if max_distance <= 0.0:
        raise ValueError("Enabled connectivity maximum distances must be positive.")

    if neighbour_table is None:
        windows_by_element = {}
        for (element_a, element_b), bounds in windows.items():
            windows_by_element.setdefault(element_a, {})[element_b] = bounds
            windows_by_element.setdefault(element_b, {})[element_a] = bounds

        base_elements = [atom["element"] for atom in base_atoms]
        neighbour_table = _periodic_neighbour_table_gemmi(
            unit_cell,
            base_atoms,
            base_elements,
            windows_by_element,
            canonical,
        )
    elif len(neighbour_table) != len(base_atoms):
        raise ValueError("The cached periodic neighbour table is invalid.")

    new_keys = []
    for base_index, translation in sorted(seed_keys):
        for candidate, ta, tb, tc in neighbour_table[base_index]:
            key = (
                candidate,
                (
                    translation[0] + ta,
                    translation[1] + tb,
                    translation[2] + tc,
                ),
            )
            if key in visited:
                continue
            visited.add(key)
            new_keys.append(key)
    return new_keys, neighbour_table


_GROWTH_CACHE: dict[str, Any] | None = None


def _growth_cache_key(unit_cell_parameters, sites, windows):
    return (
        tuple(float(value) for value in unit_cell_parameters),
        tuple(
            (
                str(site.get("element", "X")).upper(),
                tuple(float(value) for value in site["frac"]),
            )
            for site in sites
        ),
        tuple(sorted(windows.items())),
    )


def grow_connectivity_shell(
    unit_cell_parameters, sites, current_instances, seed_instances, pair_settings
):
    """Return the instance keys of the next shell around the seed atoms."""
    global _GROWTH_CACHE
    if not current_instances:
        raise ValueError("There are no current atoms to grow from.")
    if not seed_instances:
        raise ValueError("Select at least one atom to grow from.")

    windows = _enabled_windows(pair_settings)
    cache_key = _growth_cache_key(unit_cell_parameters, sites, windows)
    current_signature = tuple(int(value) for value in current_instances)
    cache = _GROWTH_CACHE
    cache_matches = cache is not None and cache["key"] == cache_key
    if cache_matches and cache is not None:
        unit_cell = cache["unit_cell"]
        base_atoms = cache["base_atoms"]
        canonical = cache["canonical"]
        neighbour_table = cache["neighbour_table"]
    else:
        unit_cell = _UnitCell(unit_cell_parameters)
        base_atoms = _base_atoms(unit_cell, sites)
        if not base_atoms:
            raise ValueError("The site table is empty.")
        canonical = _canonical_base_indices(base_atoms)
        neighbour_table = None

    state_matches = cache is not None and cache_matches and cache["instances"] == current_signature
    visited = (
        cache["visited"]
        if cache is not None and state_matches
        else _instance_keys(current_instances, canonical)
    )
    seed_keys = _instance_keys(seed_instances, canonical)
    if not seed_keys:
        raise ValueError("The selected atoms are not part of the structure.")

    new_keys, neighbour_table = _grow_shell_keys(
        unit_cell,
        base_atoms,
        canonical,
        visited,
        seed_keys,
        windows,
        neighbour_table,
    )

    new_instances = []
    for site_index, translation in sorted(new_keys):
        new_instances.extend(
            (
                site_index,
                translation[0],
                translation[1],
                translation[2],
            )
        )
    _GROWTH_CACHE = {
        "key": cache_key,
        "unit_cell": unit_cell,
        "base_atoms": base_atoms,
        "canonical": canonical,
        "neighbour_table": neighbour_table,
        "visited": visited,
        "instances": current_signature + tuple(new_instances),
    }
    return {"new_instances": new_instances, "grew_by": len(new_keys)}
