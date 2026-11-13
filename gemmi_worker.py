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


def _p1_sites(structure, unit_cell):
    """Cell contents in P1 as a small table of sites."""
    ops = _space_group_ops(structure)
    sites = []
    for site in structure.sites:
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


def load_structure(cif_path, block_name=None):
    """Return the P1 site table and the instances that make up the ASU."""
    structure, key, block = _read_structure(cif_path, block_name)
    unit_cell = _UnitCell(structure.cell.parameters)
    ops = _space_group_ops(structure)
    sites = _p1_sites(structure, unit_cell)
    base_atoms = _base_atoms(unit_cell, sites)
    lookup = _base_atom_lookup(unit_cell, base_atoms)
    asu_atoms = []
    for site in structure.sites:
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
    """
    groups = {}
    canonical = []
    for index, atom in enumerate(base_atoms):
        key = (
            atom["element"],
            *(round(float(value), 5) for value in atom["cart"]),
        )
        canonical.append(groups.setdefault(key, index))
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
    non-canonical atom is a duplicate copy of another site (same element, same
    position to 1e-5 A), so listing it as a candidate would let growth add an
    atom sitting on top of its representative whenever the enabled window
    admits the ~1e-6 A separation -- i.e. with ``min_distance`` set to 0,
    where the ``dist <= 1e-12`` self filter no longer catches it.
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
    sites = []
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


_GROWTH_CACHE = None


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
    cache_matches = _GROWTH_CACHE is not None and _GROWTH_CACHE["key"] == cache_key
    if cache_matches:
        unit_cell = _GROWTH_CACHE["unit_cell"]
        base_atoms = _GROWTH_CACHE["base_atoms"]
        canonical = _GROWTH_CACHE["canonical"]
        neighbour_table = _GROWTH_CACHE["neighbour_table"]
    else:
        unit_cell = _UnitCell(unit_cell_parameters)
        base_atoms = _base_atoms(unit_cell, sites)
        if not base_atoms:
            raise ValueError("The site table is empty.")
        canonical = _canonical_base_indices(base_atoms)
        neighbour_table = None

    state_matches = cache_matches and _GROWTH_CACHE["instances"] == current_signature
    visited = (
        _GROWTH_CACHE["visited"] if state_matches else _instance_keys(current_instances, canonical)
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
