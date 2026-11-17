"""Unit tests for the pure-Python chemistry perception (``chemistry.py``).

No Blender, no gemmi: run them with either CPython or Blender's Python:

  python tests/test_chemistry.py
  blender --background --factory-startup --python tests/test_chemistry.py
"""

import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chemistry  # noqa: E402  (the add-on root is the package directory)

CIF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CIF")

# Covalent radii (Angstrom) used to normalise the fixture's bond lengths.
RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "FE": 1.32}


def molecule(atoms, bonds):
    """Build the (elements, edges, lengths) triple from labelled bonds."""
    elements = dict(atoms)
    edges = []
    lengths = {}
    for first, second, distance in bonds:
        edges.append((first, second))
        lengths[chemistry.edge_key(first, second)] = distance / (
            RADII[elements[first]] + RADII[elements[second]]
        )
    return elements, edges, lengths


def cycle(prefix, count, element="C", distance=1.39):
    """Atoms and ring bonds of a simple cycle."""
    nodes = [f"{prefix}{index}" for index in range(count)]
    atoms = {node: element for node in nodes}
    bonds = [(nodes[index], nodes[(index + 1) % count], distance) for index in range(count)]
    return atoms, bonds


def ring_bonds(nodes, distance=1.39):
    """Closed ring bonds through *nodes*, in order."""
    return [
        (nodes[index], nodes[(index + 1) % len(nodes)], distance) for index in range(len(nodes))
    ]


def honeycomb(columns, rows, distance=1.42):
    """A hydrogen-less graphene patch: ``(elements, edges, lengths)``."""
    first = (1.5, math.sqrt(3) / 2)
    second = (1.5, -math.sqrt(3) / 2)
    a_sites = {}
    b_sites = {}
    for column in range(columns):
        for row in range(rows):
            x = column * first[0] + row * second[0]
            y = column * first[1] + row * second[1]
            a_sites[(column, row)] = (round(x, 4), round(y, 4), 0.0)
            b_sites[(column, row)] = (round(x + 1.0, 4), round(y, 4), 0.0)
    atoms = {point: "C" for point in list(a_sites.values()) + list(b_sites.values())}
    bonds = []
    for (column, row), point in a_sites.items():
        for other in (
            b_sites.get((column, row)),
            b_sites.get((column - 1, row)),
            b_sites.get((column, row - 1)),
        ):
            if other is not None:
                bonds.append((point, other, distance))
    return molecule(atoms, bonds)


def multiple_bonds(orders):
    return {key: order for key, order in orders.items() if order > 1}


def valences(elements, edges, orders):
    """Total bond order of every atom."""
    total = {node: 0 for node in elements}
    for first, second in edges:
        order = orders.get(chemistry.edge_key(first, second), 1)
        total[first] += order
        total[second] += order
    return total


class AromaticRingsTest(unittest.TestCase):
    def test_benzene_is_three_kekule_doubles(self):
        atoms, bonds = cycle("C", 6)
        for index in range(6):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        doubles = multiple_bonds(orders)
        self.assertEqual(len(doubles), 3, doubles)
        self.assertTrue(all(order == 2 for order in doubles.values()))

        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual(len(rings), 1)
        ring_nodes, ring_edges = rings[0]
        self.assertEqual(len(ring_nodes), 6)
        self.assertEqual(len(ring_edges), 6)

    def test_benzene_without_hydrogens_keeps_one_extra_order_per_ring_atom(self):
        # A CIF without hydrogens: every ring carbon has two neighbours but
        # really carries an implicit hydrogen, so it must not become an allene
        # chain (or a set of triples).
        atoms, bonds = cycle("C", 6)
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(len(multiple_bonds(orders)), 3)
        self.assertEqual(len(chemistry.aromatic_rings(elements, edges, orders)), 1)

    def test_pyridine(self):
        nodes = ["N0", "C0", "C1", "C2", "C3", "C4"]
        atoms = {node: ("N" if node == "N0" else "C") for node in nodes}
        bonds = ring_bonds(nodes, 1.39)
        for index in range(5):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(len(multiple_bonds(orders)), 3)
        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual(len(rings), 1)
        self.assertIn("N0", rings[0][0])

    def test_pyrrole_nitrogen_contributes_its_lone_pair(self):
        nodes = ["N0", "C0", "C1", "C2", "C3"]
        atoms = {node: ("N" if node == "N0" else "C") for node in nodes}
        atoms["H0"] = "H"
        bonds = ring_bonds(nodes, 1.39)
        bonds.append(("N0", "H0", 1.01))
        for index in range(4):
            atoms[f"H{index + 1}"] = "H"
            bonds.append((f"C{index}", f"H{index + 1}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        doubles = multiple_bonds(orders)
        # N-H is saturated: the two ring doubles must stay on the carbons.
        self.assertEqual(len(doubles), 2, doubles)
        self.assertFalse(any("N0" in key for key in doubles), doubles)
        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual(len(rings), 1)

    def test_naphthalene_has_two_aromatic_rings(self):
        atoms = {f"C{index}": "C" for index in range(10)}
        bonds = [
            ("C0", "C1", 1.40),
            ("C1", "C2", 1.40),
            ("C2", "C3", 1.40),
            ("C3", "C4", 1.40),
            ("C4", "C5", 1.40),
            ("C5", "C0", 1.40),
            ("C4", "C6", 1.40),
            ("C6", "C7", 1.40),
            ("C7", "C8", 1.40),
            ("C8", "C9", 1.40),
            ("C9", "C5", 1.40),
        ]
        for index in range(10):
            if index in (4, 5):
                continue
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(len(multiple_bonds(orders)), 5)
        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual(len(rings), 2)
        self.assertEqual({len(nodes) for nodes, _edges in rings}, {6})

    def test_bridgehead_nitrogen_donates_no_lone_pair(self):
        # Indolizine: a five-ring and a six-ring fused on their N-C bond.  The
        # lone pair of the bridgehead nitrogen belongs to the fused ten-pi
        # system, so neither ring may claim it as a sextet of its own and both
        # are depicted with their Kekule doubles.
        atoms = {"N0": "N", "C1": "C"}
        bonds = [
            ("N0", "C1", 1.39),
            ("C1", "C2", 1.39),
            ("C2", "C3", 1.39),
            ("C3", "C4", 1.39),
            ("C4", "C5", 1.39),
            ("C5", "N0", 1.37),
            ("N0", "C8", 1.38),
            ("C8", "C7", 1.39),
            ("C7", "C6", 1.39),
            ("C6", "C1", 1.39),
        ]
        for index in range(2, 9):
            atoms[f"C{index}"] = "C"
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        doubles = multiple_bonds(orders)
        self.assertEqual(len(doubles), 4, doubles)
        self.assertTrue(all(order == 2 for order in doubles.values()), doubles)
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])

    def test_indole_keeps_both_rings_aromatic(self):
        # The same five-ring, but its nitrogen belongs to that ring alone: a
        # fused ring system must not stop a heteroatom from donating.
        atoms = {"N0": "N", "H0": "H", "C1": "C", "C2": "C"}
        bonds = [
            ("N0", "H0", 1.01),
            ("N0", "C1", 1.37),
            ("N0", "C4", 1.38),
            ("C1", "C2", 1.42),
            ("C2", "C3", 1.39),
            ("C3", "C4", 1.38),
            ("C1", "C8", 1.40),
            ("C2", "C5", 1.40),
            ("C5", "C6", 1.39),
            ("C6", "C7", 1.39),
            ("C7", "C8", 1.39),
        ]
        for index in range(3, 9):
            atoms[f"C{index}"] = "C"
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(len(multiple_bonds(orders)), 4)
        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual({len(nodes) for nodes, _edges in rings}, {5, 6})

    def test_hydrogen_less_polycyclic_sheet(self):
        # A graphene patch has no hydrogens at all: every carbon has two ring
        # neighbours and one fusion bond.  Fusion double bonds must not veto the
        # neighbouring ring, and the bounded searches must stay bounded.
        elements, edges, lengths = honeycomb(6, 6)
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(len(multiple_bonds(orders)), len(elements) // 2)

        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertGreaterEqual(len(rings), 20)
        self.assertEqual({len(nodes) for nodes, _edges in rings}, {6})

    def test_exocyclic_double_bond_vetoes_the_ring(self):
        # Cyclohexa-2,4-dien-1-one: the ring carries a C=O, so it is not
        # described as an aromatic torus even with two ring double bonds.
        atoms, bonds = cycle("C", 6)
        atoms["O0"] = "O"
        bonds.append(("C0", "O0", 1.22))
        for index in range(6):
            if index == 0:
                continue
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders[chemistry.edge_key("C0", "O0")], 2)
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])

    def test_cyclohexane_has_no_multiple_bonds(self):
        atoms, bonds = cycle("C", 6, distance=1.54)
        for index in range(6):
            for hydrogen in ("A", "B"):
                atoms[f"H{index}{hydrogen}"] = "H"
                bonds.append((f"C{index}", f"H{index}{hydrogen}", 1.09))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(multiple_bonds(orders), {})
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])

    def test_saturated_ring_is_not_aromatic_even_with_a_nitrogen(self):
        # Piperidine: one nitrogen, no double bond anywhere in the ring.
        atoms, bonds = cycle("C", 5, distance=1.54)
        atoms["N0"] = "N"
        atoms["H0"] = "H"
        bonds.append(("N0", "C0", 1.47))
        bonds.append(("N0", "H0", 1.01))
        for index in range(5):
            for hydrogen in ("A", "B"):
                atoms[f"H{index}{hydrogen}"] = "H"
                bonds.append((f"C{index}", f"H{index}{hydrogen}", 1.09))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(multiple_bonds(orders), {})
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])


class MultipleBondTest(unittest.TestCase):
    def test_acetylene_is_a_triple_bond(self):
        elements, edges, lengths = molecule(
            {"C0": "C", "C1": "C", "H0": "H", "H1": "H"},
            [("C0", "C1", 1.20), ("C0", "H0", 1.06), ("C1", "H1", 1.06)],
        )
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders, {chemistry.edge_key("C0", "C1"): 3})

    def test_acetonitrile_triple_is_on_the_nitrile_carbon(self):
        elements, edges, lengths = molecule(
            {"C0": "C", "C1": "C", "N0": "N", "H0": "H", "H1": "H", "H2": "H"},
            [
                ("C0", "C1", 1.46),
                ("C1", "N0", 1.16),
                ("C0", "H0", 1.09),
                ("C0", "H1", 1.09),
                ("C0", "H2", 1.09),
            ],
        )
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders, {chemistry.edge_key("C1", "N0"): 3})

    def test_allene_gets_two_double_bonds(self):
        elements, edges, lengths = molecule(
            {"C0": "C", "C1": "C", "C2": "C", "H0": "H", "H1": "H", "H2": "H", "H3": "H"},
            [
                ("C0", "C1", 1.31),
                ("C1", "C2", 1.31),
                ("C0", "H0", 1.08),
                ("C0", "H1", 1.08),
                ("C2", "H2", 1.08),
                ("C2", "H3", 1.08),
            ],
        )
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(
            set(multiple_bonds(orders)),
            {chemistry.edge_key("C0", "C1"), chemistry.edge_key("C1", "C2")},
        )

    def test_carbonyl_oxygen_is_double_bonded(self):
        elements, edges, lengths = molecule(
            {"C0": "C", "O0": "O", "H0": "H", "H1": "H"},
            [("C0", "O0", 1.21), ("C0", "H0", 1.11), ("C0", "H1", 1.11)],
        )
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders, {chemistry.edge_key("C0", "O0"): 2})

    def test_long_contacts_are_never_promoted(self):
        # 1.54 A is a single bond: the candidate gate rejects it even though
        # both carbons still "want" an extra order.
        elements, edges, lengths = molecule(
            {"C0": "C", "C1": "C"},
            [("C0", "C1", 1.54)],
        )
        self.assertEqual(chemistry.bond_orders(elements, edges, lengths), {})

    def test_unknown_elements_keep_single_bonds(self):
        elements, edges, lengths = molecule(
            {"FE0": "FE", "O0": "O", "O1": "O"},
            [("FE0", "O0", 1.90), ("FE0", "O1", 1.90)],
        )
        # FE is not in the valence table: it must not push O into a double bond.
        self.assertEqual(chemistry.bond_orders(elements, edges, lengths), {})

    def test_edges_may_be_given_in_either_orientation(self):
        forward = molecule(
            {"C0": "C", "C1": "C", "H0": "H", "H1": "H"},
            [("C0", "C1", 1.20), ("C0", "H0", 1.06), ("C1", "H1", 1.06)],
        )
        backward = molecule(
            {"C0": "C", "C1": "C", "H0": "H", "H1": "H"},
            [("C1", "C0", 1.20), ("H0", "C0", 1.06), ("H1", "C1", 1.06)],
        )
        self.assertEqual(
            chemistry.bond_orders(*forward),
            chemistry.bond_orders(*backward),
        )


class FindRingsTest(unittest.TestCase):
    def test_four_membered_ring_is_not_reported(self):
        atoms, bonds = cycle("C", 4, distance=1.55)
        elements, edges, _lengths = molecule(atoms, bonds)
        self.assertEqual(chemistry.find_rings(elements, edges), [])

    def test_each_ring_is_reported_once(self):
        atoms, bonds = cycle("C", 6)
        elements, edges, _lengths = molecule(atoms, bonds)
        rings = chemistry.find_rings(elements, edges)
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0], frozenset(f"C{index}" for index in range(6)))

    def test_hydrogens_are_never_ring_atoms(self):
        atoms, bonds = cycle("C", 6)
        atoms["H0"] = "H"
        bonds.append(("C0", "H0", 1.08))
        elements, edges, _lengths = molecule(atoms, bonds)
        rings = chemistry.find_rings(elements, edges)
        self.assertEqual(len(rings), 1)
        self.assertNotIn("H0", rings[0])

    def test_metal_oxide_ring_is_not_an_aromatic_ring(self):
        # A closed Fe-O ring is a ring for crystallography, not for chemistry.
        atoms = {
            "FE0": "FE",
            "FE1": "FE",
            "FE2": "FE",
            "O0": "O",
            "O1": "O",
            "O2": "O",
        }
        bonds = [
            ("FE0", "O0", 1.90),
            ("O0", "FE1", 1.90),
            ("FE1", "O1", 1.90),
            ("O1", "FE2", 1.90),
            ("FE2", "O2", 1.90),
            ("O2", "FE0", 1.90),
        ]
        elements, edges, _lengths = molecule(atoms, bonds)
        self.assertEqual(chemistry.find_rings(elements, edges), [])


def read_geom_bonds(path):
    """Read the ``_geom_bond`` loop of a CIF as ``(elements, edges, lengths)``."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.strip() == "_geom_bond_atom_site_label_1"
    )
    elements = {}
    edges = []
    lengths = {}
    for line in lines[start + 3 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith(("loop_", "_", "#")):
            break
        first, second, distance = stripped.split()
        distance = float(re.sub(r"\(.*\)", "", distance))
        element_a = re.match(r"([A-Za-z]+)", first).group(1).upper()
        element_b = re.match(r"([A-Za-z]+)", second).group(1).upper()
        elements[first] = element_a
        elements[second] = element_b
        edges.append((first, second))
        lengths[chemistry.edge_key(first, second)] = distance / (
            RADII[element_a] + RADII[element_b]
        )
    return elements, edges, lengths


class MultipleBondsCifTest(unittest.TestCase):
    """The COD structure the ChemDraw depiction is validated against."""

    EXPECTED_DOUBLES = {
        ("C1", "C16"),
        ("C2", "C3"),
        ("C4", "C5"),
        ("C6", "C15"),
        ("C8", "C13"),
        ("C9", "C10"),
        ("C11", "C12"),
        ("C7", "O2"),
        ("C14", "O1"),
        ("C17", "O3"),
    }
    # Only the benzene ring is aromatic.  The pyrrole-type five-ring
    # (N1/C1/C6/C15/C16) shares its nitrogen with the six-ring
    # (N1/C1/C2/C3/C4/C5) — an annulated indolizine — so N1 is a bridgehead and
    # donates its lone pair to neither ring: both are drawn Kekulé.
    EXPECTED_RINGS = {
        frozenset({"C8", "C9", "C10", "C11", "C12", "C13"}),
    }

    @classmethod
    def setUpClass(cls):
        cls.elements, cls.edges, cls.lengths = read_geom_bonds(
            os.path.join(CIF_DIR, "multiple_bonds.cif")
        )
        cls.orders = chemistry.bond_orders(cls.elements, cls.edges, cls.lengths)

    def test_kekule_structure(self):
        doubles = {frozenset(key) for key, order in self.orders.items() if order == 2}
        self.assertEqual(doubles, {frozenset(key) for key in self.EXPECTED_DOUBLES})
        self.assertFalse([order for order in self.orders.values() if order > 2])

    def test_aromatic_rings(self):
        rings = chemistry.aromatic_rings(self.elements, self.edges, self.orders)
        self.assertEqual({nodes for nodes, _edges in rings}, self.EXPECTED_RINGS)

    def test_every_atom_reaches_its_target_valence(self):
        totals = valences(self.elements, self.edges, self.orders)
        for node, element in self.elements.items():
            ladder = chemistry.VALENCE_LADDERS[element]
            self.assertIn(totals[node], ladder, f"{node} ({element}) -> {totals[node]}")

    def test_result_does_not_depend_on_bond_order(self):
        shuffled = chemistry.bond_orders(
            self.elements,
            list(reversed(self.edges)),
            self.lengths,
        )
        self.assertEqual(shuffled, self.orders)


if __name__ == "__main__":
    # Blender keeps its own CLI flags in ``sys.argv``, so ``unittest.main()``
    # would try to parse ``--background``.  Drive the runner explicitly and
    # print the RESULT line ``tests/run_all.ps1`` looks for.
    SUITE = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    RESULT = unittest.TextTestRunner(verbosity=2).run(SUITE)
    print("RESULT: ALL PASS" if RESULT.wasSuccessful() else "RESULT: FAIL")
    sys.exit(0 if RESULT.wasSuccessful() else 1)
