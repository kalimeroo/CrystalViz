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
RADII = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 0.99,
    "SE": 1.20,
    "AS": 1.19,
    "TE": 1.38,
    "SB": 1.39,
    "FE": 1.32,
}


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


def honeycomb(columns, rows, distance=1.42, sublattices=("C", "C")):
    """A hydrogen-less honeycomb patch: ``(elements, edges, lengths)``.

    *sublattices* names the element of the A and B sublattices, so
    ``("B", "N")`` builds the isoelectronic h-BN sheet.
    """
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
    atoms = {point: sublattices[0] for point in a_sites.values()}
    atoms.update({point: sublattices[1] for point in b_sites.values()})
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


def azine_ring(nitrogens, oxide=False):
    """A pyridine-like six-ring with nitrogens at the given ring positions."""
    nodes = [f"X{index}" for index in range(6)]
    atoms = {node: ("N" if index in nitrogens else "C") for index, node in enumerate(nodes)}
    bonds = ring_bonds(nodes)
    for index, node in enumerate(nodes):
        if index in nitrogens:
            if oxide:
                atoms[f"O{index}"] = "O"
                bonds.append((node, f"O{index}", 1.31))
        else:
            atoms[f"H{index}"] = "H"
            bonds.append((node, f"H{index}", 1.08))
    return molecule(atoms, bonds)


def sulfur_ring(oxide=False, methyl=False):
    """A thiopyrylium-like six-ring (sulfur + five carbons).

    The S-C distances come from the two COD entries that drove the rule:
    BIZTAF (thiopyrylium, no oxygen) and ATBOXM (thiabenzene 1-oxide, S=O plus
    a methyl on the sulfur).
    """
    nodes = ["S", "C1", "C2", "C3", "C4", "C5"]
    atoms = {node: ("S" if node == "S" else "C") for node in nodes}
    bonds = [
        ("S", "C1", 1.692 if oxide else 1.672),
        ("C1", "C2", 1.39),
        ("C2", "C3", 1.39),
        ("C3", "C4", 1.39),
        ("C4", "C5", 1.39),
        ("C5", "S", 1.689 if oxide else 1.702),
    ]
    for node in nodes[1:]:
        atoms[node + "H"] = "H"
        bonds.append((node, node + "H", 1.08))
    if oxide:
        atoms["O1"] = "O"
        bonds.append(("S", "O1", 1.466))
    if methyl:
        atoms["C6"] = "C"
        bonds.append(("S", "C6", 1.771))
        for suffix in ("a", "b", "c"):
            atoms["H6" + suffix] = "H"
            bonds.append(("C6", "H6" + suffix, 1.08))
    return molecule(atoms, bonds)


def hetero_ring(element, size, first, second, carbon_carbon=1.39):
    """A conjugated ring of *size* with one heteroatom and one H per carbon.

    *first* and *second* are the two heteroatom-ring distances, taken from the
    COD entry named in each test; *carbon_carbon* is the ring C-C distance.
    """
    carbons = [f"C{index}" for index in range(1, size)]
    atoms = {element: element}
    bonds = [(element, carbons[0], first)]
    for index in range(len(carbons) - 1):
        bonds.append((carbons[index], carbons[index + 1], carbon_carbon))
    bonds.append((carbons[-1], element, second))
    for node in carbons:
        atoms[node] = "C"
        atoms[node + "H"] = "H"
        bonds.append((node, node + "H", 1.08))
    return atoms, bonds


def borazine_ring():
    """B3N3H6: every boron and nitrogen is three-connected (COD 2216154)."""
    nodes = ["B1", "N1", "B2", "N2", "B3", "N3"]
    atoms = {node: node[0] for node in nodes}
    bonds = ring_bonds(nodes, 1.43)
    for node in nodes:
        atoms[node + "H"] = "H"
        bonds.append((node, node + "H", 1.19 if node.startswith("B") else 1.01))
    return molecule(atoms, bonds)


def boroxine_ring():
    """B3O3 with a methyl on every boron (COD 1548569 has aryl groups)."""
    nodes = ["B1", "O1", "B2", "O2", "B3", "O3"]
    atoms = {node: node[0] for node in nodes}
    bonds = ring_bonds(nodes, 1.37)
    for node in ("B1", "B2", "B3"):
        atoms[node + "C"] = "C"
        bonds.append((node, node + "C", 1.55))
        for suffix in ("a", "b", "c"):
            atoms[node + "C" + suffix] = "H"
            bonds.append((node + "C", node + "C" + suffix, 1.08))
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

    def test_pyrrole_sextet_is_donated_by_the_nitrogen(self):
        # IUPAC GR-6.3: pyrrole has one unique arrangement of alternating single
        # and double bonds, so the depiction layer draws it Kekule.  That is
        # decided from the lone-pair share of the sextet.
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
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0].pi_electrons, 6)
        self.assertEqual(rings[0].lone_pair_electrons, 2)
        self.assertTrue(rings[0].kekule)

    def test_carbocyclic_sextet_uses_no_lone_pair(self):
        # Benzene and pyridine keep the delocalisation curve: their sextet is
        # built only from ring double bonds.
        atoms, bonds = cycle("C", 6)
        for index in range(6):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual([ring.lone_pair_electrons for ring in rings], [0])

        nodes = ["N0", "C0", "C1", "C2", "C3", "C4"]
        atoms = {node: ("N" if node == "N0" else "C") for node in nodes}
        bonds = ring_bonds(nodes, 1.39)
        for index in range(5):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual([ring.lone_pair_electrons for ring in rings], [0])

    def test_metallocene_cyclopentadienyl_is_a_six_pi_ligand(self):
        # Ferrocene: every Cp carbon is bonded to two ring carbons, one hydrogen
        # and the iron, so no Kekule double bond can be placed.  The ring is the
        # aromatic 6 pi Cp- ligand, so it is still perceived as aromatic, and
        # the metal is reported (IUPAC GR-1.7.2 draws it with a curve).
        atoms, bonds = cycle("C", 5)
        atoms["FE0"] = "FE"
        for index in range(5):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
            bonds.append((f"C{index}", "FE0", 2.05))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(multiple_bonds(orders), {})
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0].pi_electrons, 6)
        self.assertEqual(rings[0].lone_pair_electrons, 0)
        self.assertEqual(rings[0].haptic_metal, "FE")

    def test_coordinated_ring_with_an_exocyclic_double_bond_stays_kekule(self):
        # An eta6-coordinated p-benzoquinone: the two exocyclic C=O bonds veto
        # the haptic shortcut just as they veto a Kekule ring, so the metal does
        # not turn the quinone into an aromatic ligand.
        elements = {f"C{index}": "C" for index in range(6)}
        elements.update({"O0": "O", "O5": "O", "FE0": "FE"})
        edges = [
            ("C0", "C1"),
            ("C1", "C2"),
            ("C2", "C3"),
            ("C3", "C4"),
            ("C4", "C5"),
            ("C5", "C0"),
            ("C0", "O0"),
            ("C5", "O5"),
        ]
        edges.extend((f"C{index}", "FE0") for index in range(6))
        orders = {
            chemistry.edge_key("C0", "O0"): 2,
            chemistry.edge_key("C5", "O5"): 2,
            chemistry.edge_key("C1", "C2"): 2,
            chemistry.edge_key("C3", "C4"): 2,
        }
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])

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
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual({len(ring.nodes) for ring in rings}, {5, 6})
        self.assertTrue(all(ring.kekule for ring in rings))

    def test_hydrogen_less_polycyclic_sheet(self):
        # A graphene patch has no hydrogens at all: every carbon has two ring
        # neighbours and one fusion bond.  Fusion double bonds must not veto the
        # neighbouring ring, and the bounded searches must stay bounded.
        elements, edges, lengths = honeycomb(6, 6)
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(len(multiple_bonds(orders)), len(elements) // 2)

        # The patch ends in dangling carbons, and the Kekule structure doubles
        # them.  That is a truncated lattice, not an exocyclic C=CH2, so no ring
        # may be vetoed for it.
        adjacency = {node: set() for node in elements}
        for first, second in edges:
            adjacency[first].add(second)
            adjacency[second].add(first)
        dangling = [node for node in elements if len(adjacency[node]) == 1]
        self.assertTrue(dangling)
        for node in dangling:
            self.assertEqual(
                max(orders.get(chemistry.edge_key(node, other), 1) for other in adjacency[node]),
                2,
            )

        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual(len(rings), len(chemistry.find_rings(elements, edges)))
        self.assertEqual({len(nodes) for nodes, _edges in rings}, {6})

    def test_methylene_with_hydrogens_still_vetoes_the_ring(self):
        # The dangling-carbon exemption only reaches a carbon with no other
        # neighbour: a real exocyclic =CH2 carries its hydrogens and still
        # breaks the conjugation of the ring it hangs from.
        atoms, bonds = cycle("C", 6)
        atoms["C6"] = "C"
        bonds.append(("C0", "C6", 1.34))
        for suffix in ("a", "b"):
            atoms[f"H6{suffix}"] = "H"
            bonds.append(("C6", f"H6{suffix}", 1.08))
        for index in range(1, 6):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders[chemistry.edge_key("C0", "C6")], 2)
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])

    def test_hydrogen_less_quinodimethane_is_the_known_exception(self):
        # Accepted trade-off of the dangling-carbon exemption: a CIF with no
        # hydrogens cannot tell =CH2 from a lattice edge, so the heavy-atom
        # graph of p-quinodimethane is drawn as an aromatic ring.  Structures
        # that carry their hydrogens keep the exocyclic veto above.
        atoms, bonds = cycle("C", 6)
        for label, ring_atom in (("C6", "C0"), ("C7", "C3")):
            atoms[label] = "C"
            bonds.append((ring_atom, label, 1.34))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(len(chemistry.aromatic_rings(elements, edges, orders)), 1)

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

    def test_alkyne_on_a_saturated_cp_carbon_stays_a_triple(self):
        # Ferrocene.cif: the Cp carbon that carries the alkyne is four
        # coordinate (two ring carbons, the iron, the alkyne carbon), so it has
        # no valence deficit left to spend.  The alkyne must take both extra
        # orders on its own bond instead of borrowing one from the arene, which
        # would invent an allene (the fixture drew C12=C13 as a double bond).
        atoms, bonds = cycle("C", 5, distance=1.42)
        atoms["FE0"] = "FE"
        for index in range(1, 5):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        for index in range(5):
            bonds.append((f"C{index}", "FE0", 2.05))
        # Anthracene, substituted at its 9-position (M1) by the alkyne.
        rings = [
            ["L0", "L1", "L2", "L3", "L4", "L5"],
            ["L3", "M0", "M1", "M2", "M3", "L4"],
            ["M2", "R0", "R1", "R2", "R3", "M3"],
        ]
        atoms.update({node: "C" for ring in rings for node in ring})
        seen = set()
        for ring in rings:
            for first, second, distance in ring_bonds(ring, 1.40):
                key = chemistry.edge_key(first, second)
                if key in seen:
                    continue
                seen.add(key)
                bonds.append((first, second, distance))
        fusion = {"L3", "L4", "M2", "M3"}
        hydrogen_sites = [
            node for node in list(atoms) if node[0] in "LMR" and node not in fusion and node != "M1"
        ]
        for index, node in enumerate(hydrogen_sites):
            atoms[f"HX{index}"] = "H"
            bonds.append((node, f"HX{index}", 1.08))
        atoms.update({"X0": "C", "X1": "C", "X2": "C"})
        bonds.extend(
            [("C0", "X0", 1.44), ("X0", "X1", 1.19), ("X1", "X2", 1.44), ("X2", "M1", 1.43)]
        )
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders.get(chemistry.edge_key("X0", "X1")), 3)
        self.assertNotIn(chemistry.edge_key("X1", "X2"), orders)
        self.assertNotIn(chemistry.edge_key("X2", "M1"), orders)
        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual(len(rings), 4)  # the Cp ligand and the three arene rings

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


class NOxideValenceTest(unittest.TestCase):
    """Dative N-O bonds (N-oxides) and the genuine N=O of a nitro group.

    A three-connected nitrogen carrying one terminal oxygen is four-valent:
    its fourth bond is a real multiple bond into the ring, while the N-O bond
    itself stays single.  Without that the ring has an odd number of
    unsaturated carbons and no Kekule structure exists at all.
    """

    def test_pyridine_n_oxide_ring_is_aromatic(self):
        elements, edges, lengths = azine_ring({5}, oxide=True)
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertNotIn(chemistry.edge_key("X5", "O5"), multiple_bonds(orders))
        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual([len(nodes) for nodes, _ in rings], [6])
        totals = valences(elements, edges, orders)
        for node in ("X0", "X1", "X2", "X3", "X4", "X5"):
            self.assertEqual(totals[node], 4, node)

    def test_pyrazine_dioxide_ring_is_aromatic(self):
        elements, edges, lengths = azine_ring({2, 5}, oxide=True)
        orders = chemistry.bond_orders(elements, edges, lengths)

        for key in (chemistry.edge_key("X2", "O2"), chemistry.edge_key("X5", "O5")):
            self.assertNotIn(key, multiple_bonds(orders))
        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual([len(nodes) for nodes, _ in rings], [6])
        totals = valences(elements, edges, orders)
        for node in ("X0", "X1", "X2", "X3", "X4", "X5"):
            self.assertEqual(totals[node], 4, node)

    def test_nitro_group_draws_two_n_oxygen_double_bonds(self):
        # Nitrobenzene: two terminal oxygens, so neither N-O bond is dative and
        # the group is drawn N(=O)2, the delocalised O-N-O as two doubles.
        nodes = [f"X{index}" for index in range(6)]
        atoms = {node: "C" for node in nodes}
        atoms.update({"N0": "N", "O1": "O", "O2": "O"})
        bonds = ring_bonds(nodes)
        bonds += [("X0", "N0", 1.47), ("N0", "O1", 1.22), ("N0", "O2", 1.24)]
        for index in range(1, 6):
            atoms[f"H{index}"] = "H"
            bonds.append((f"X{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        nitro = {key: order for key, order in multiple_bonds(orders).items() if "N0" in key}
        self.assertEqual(
            nitro, {chemistry.edge_key("N0", "O1"): 2, chemistry.edge_key("N0", "O2"): 2}
        )
        self.assertEqual(len(chemistry.aromatic_rings(elements, edges, orders)), 1)

    def test_pyridine_without_oxygen_is_unchanged(self):
        elements, edges, lengths = azine_ring({5})
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(len(multiple_bonds(orders)), 3)
        self.assertEqual(len(chemistry.aromatic_rings(elements, edges, orders)), 1)

    def test_nitroso_double_bond_is_kept(self):
        # A two-connected N=O is a genuine double bond, not a dative one.
        elements, edges, lengths = molecule(
            {"N0": "N", "O0": "O", "C0": "C"},
            [("C0", "N0", 1.45), ("N0", "O0", 1.21)],
        )
        self.assertEqual(
            chemistry.bond_orders(elements, edges, lengths),
            {chemistry.edge_key("N0", "O0"): 2},
        )

    def test_hydroxylamine_stays_single_bonded(self):
        # The N-oxide promotion must not invent a multiple bond where the
        # extra valence has nowhere to go.
        elements, edges, lengths = molecule(
            {"N0": "N", "O0": "O", "H1": "H", "H2": "H", "H3": "H"},
            [
                ("N0", "O0", 1.45),
                ("N0", "H1", 1.02),
                ("N0", "H2", 1.02),
                ("O0", "H3", 0.97),
            ],
        )
        self.assertEqual(chemistry.bond_orders(elements, edges, lengths), {})


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


class SulfurRingTest(unittest.TestCase):
    """Six-membered sulfur rings: thiopyrylium and thiabenzene 1-oxide.

    A conjugated six-ring sulfur draws a genuine S=C bond, exactly as pyridine
    draws N=C, so the ring has three alternating doubles and no lone-pair share
    of its sextet.  A sulfoxide sulfur also carries a terminal S=O, which does
    not break that conjugation.  The rule stays limited to six-membered rings
    and to sulfur bonds already shortened towards a double bond: thiophene and
    a saturated thiacyclohexane keep their previous behaviour.
    """

    def test_thiopyrylium_sulfur_takes_one_ring_double_bond(self):
        elements, edges, lengths = sulfur_ring()
        orders = chemistry.bond_orders(elements, edges, lengths)

        s_doubles = [key for key in multiple_bonds(orders) if "S" in key]
        self.assertEqual(len(s_doubles), 1)
        self.assertEqual(valences(elements, edges, orders)["S"], 3)
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual([len(ring.nodes) for ring in rings], [6])
        self.assertEqual(rings[0].pi_electrons, 6)
        self.assertEqual(rings[0].lone_pair_electrons, 0)

    def test_thiabenzene_oxide_keeps_its_s_oxygen_double(self):
        elements, edges, lengths = sulfur_ring(oxide=True, methyl=True)
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(multiple_bonds(orders).get(chemistry.edge_key("S", "O1")), 2)
        self.assertEqual(len([key for key in multiple_bonds(orders) if "S" in key]), 2)
        self.assertEqual(valences(elements, edges, orders)["S"], 6)
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual([len(ring.nodes) for ring in rings], [6])
        self.assertEqual(rings[0].pi_electrons, 6)
        self.assertEqual(rings[0].lone_pair_electrons, 0)

    def test_sulfone_draws_two_s_oxygen_double_bonds(self):
        # A methyl phenyl sulfone: two terminal oxygens, two carbons.  The
        # sulfur is hexavalent, so both S-O bonds are genuine doubles (the S
        # analogue of nitro N(=O)2); the S-C bonds stay single and the phenyl
        # ring keeps its aromaticity.
        nodes = [f"X{index}" for index in range(6)]
        atoms = {node: "C" for node in nodes}
        atoms.update({"S0": "S", "O1": "O", "O2": "O", "M0": "C"})
        bonds = ring_bonds(nodes)
        bonds += [
            ("X0", "S0", 1.77),
            ("S0", "M0", 1.79),
            ("S0", "O1", 1.44),
            ("S0", "O2", 1.45),
        ]
        for index in range(1, 6):
            atoms[f"H{index}"] = "H"
            bonds.append((f"X{index}", f"H{index}", 1.08))
        for index in range(3):
            atoms[f"H{index + 1}m"] = "H"
            bonds.append(("M0", f"H{index + 1}m", 1.09))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        sulfone = {key: order for key, order in multiple_bonds(orders).items() if "S0" in key}
        self.assertEqual(
            sulfone, {chemistry.edge_key("S0", "O1"): 2, chemistry.edge_key("S0", "O2"): 2}
        )
        self.assertEqual(valences(elements, edges, orders)["S0"], 6)
        self.assertEqual(len(chemistry.aromatic_rings(elements, edges, orders)), 1)

    def test_cyclic_sulfone_keeps_both_s_oxygen_doubles(self):
        # A cyclic sulfone (sulfur on the ring, both oxygens exocyclic): the
        # ring-walking exclusion keeps the sulfur out of the pi system, and
        # the ring cap must not cost it one of the two S=O doubles.
        nodes = ["S0", "C1", "C2", "C3", "C4", "C5"]
        atoms = {node: "C" for node in nodes}
        atoms.update({"S0": "S", "O1": "O", "O2": "O"})
        bonds = [
            ("S0", "C1", 1.77),
            ("C1", "C2", 1.52),
            ("C2", "C3", 1.52),
            ("C3", "C4", 1.52),
            ("C4", "C5", 1.52),
            ("C5", "S0", 1.77),
            ("S0", "O1", 1.44),
            ("S0", "O2", 1.45),
        ]
        for index in range(1, 6):
            for slot in (1, 2):
                atoms[f"H{index}{slot}"] = "H"
                bonds.append((f"C{index}", f"H{index}{slot}", 1.09))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        sulfone = {key: order for key, order in multiple_bonds(orders).items() if "S0" in key}
        self.assertEqual(
            sulfone, {chemistry.edge_key("S0", "O1"): 2, chemistry.edge_key("S0", "O2"): 2}
        )
        self.assertEqual(valences(elements, edges, orders)["S0"], 6)

    def test_metal_coordinated_sulfone_keeps_both_s_oxygen_doubles(self):
        # The Ca.CIF case: a sulfone whose S=O oxygens each also coordinate a
        # cation (the metal salt picks up every S=O oxygen, so each has graph
        # degree two, one covalent and one ionic).  The cation bonds must not
        # hide the sulfone: both S-O bonds still carry order two, the S-C
        # bonds stay single, and the metal contacts keep their own singles.
        atoms = {
            "S0": "S",
            "O1": "O",
            "O2": "O",
            "C1": "C",
            "C2": "C",
            "FE1": "FE",
            "FE2": "FE",
        }
        bonds = [
            ("S0", "O1", 1.44),
            ("S0", "O2", 1.45),
            ("S0", "C1", 1.77),
            ("S0", "C2", 1.78),
            ("O1", "FE1", 2.0),
            ("O2", "FE2", 2.0),
        ]
        for index in (1, 2):
            for slot in (1, 2, 3):
                atoms[f"H{index}{slot}"] = "H"
                bonds.append((f"C{index}", f"H{index}{slot}", 1.09))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        doubles = multiple_bonds(orders)
        self.assertEqual(
            {key: value for key, value in doubles.items() if "S0" in key},
            {chemistry.edge_key("S0", "O1"): 2, chemistry.edge_key("S0", "O2"): 2},
        )
        self.assertEqual(valences(elements, edges, orders)["S0"], 6)
        self.assertNotIn(chemistry.edge_key("O1", "FE1"), doubles)
        self.assertNotIn(chemistry.edge_key("O2", "FE2"), doubles)

    def test_thiophene_keeps_its_lone_pair(self):
        nodes = ["S", "C1", "C2", "C3", "C4"]
        atoms = {"S": "S"}
        bonds = [
            ("S", "C1", 1.714),
            ("C1", "C2", 1.37),
            ("C2", "C3", 1.42),
            ("C3", "C4", 1.37),
            ("C4", "S", 1.714),
        ]
        for node in nodes[1:]:
            atoms[node] = "C"
            atoms[node + "H"] = "H"
            bonds.append((node, node + "H", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        doubles = multiple_bonds(orders)
        self.assertEqual(len(doubles), 2)
        self.assertFalse(any("S" in key for key in doubles))
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(ring.pi_electrons, ring.lone_pair_electrons) for ring in rings], [(6, 2)]
        )

    def test_saturated_thiacyclohexane_has_no_multiple_bonds(self):
        nodes = ["S", "C1", "C2", "C3", "C4", "C5"]
        atoms = {"S": "S"}
        bonds = []
        for index in range(6):
            first = nodes[index]
            second = nodes[(index + 1) % 6]
            distance = 1.82 if "S" in (first, second) else 1.54
            bonds.append((first, second, distance))
        for node in nodes[1:]:
            atoms[node] = "C"
            for suffix in ("a", "b"):
                atoms[node + suffix] = "H"
                bonds.append((node, node + suffix, 1.08))
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(multiple_bonds(orders), {})
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])

    def test_sulfone_sulfur_is_not_a_ring_atom(self):
        # A sulfone sulfur (two terminal oxygens) is not part of the ring pi
        # system: it must not make the ring walkable.
        nodes = ["S", "C1", "C2", "C3", "C4", "C5"]
        atoms = {"S": "S", "O1": "O", "O2": "O"}
        bonds = [("S", "O1", 1.44), ("S", "O2", 1.44)]
        for index in range(6):
            first = nodes[index]
            second = nodes[(index + 1) % 6]
            distance = 1.78 if "S" in (first, second) else 1.39
            bonds.append((first, second, distance))
        for node in nodes[1:]:
            atoms[node] = "C"
            atoms[node + "H"] = "H"
            bonds.append((node, node + "H", 1.08))
        elements, edges, _lengths = molecule(atoms, bonds)

        self.assertEqual(chemistry.find_rings(elements, edges), [])


class ChalcogenRingTest(unittest.TestCase):
    """Selenium and tellurium follow the sulfur rule (COD 7243860 / 7243859).

    A conjugated six-membered ring takes a genuine C=Se/C=Te double bond, so the
    ring is a 6 pi ring with no lone-pair share (selenopyrylium,
    telluropyrylium).  The five-membered selenophene/tellurophene cannot hold an
    all-double-bond Kekule structure, so the chalcogen keeps its lone pair, and
    a saturated selenane stays single-bonded.  A selenoxide selenium is
    four-coordinate, keeps its terminal Se=O double and still counts as a ring
    atom, exactly like a sulfoxide sulfur.
    """

    def test_selenopyrylium_takes_one_ring_double_bond(self):
        atoms, bonds = hetero_ring("SE", 6, 1.806, 1.853)
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        se_doubles = [key for key in multiple_bonds(orders) if "SE" in key]
        self.assertEqual(len(se_doubles), 1)
        self.assertEqual(valences(elements, edges, orders)["SE"], 3)
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(len(ring.nodes), ring.pi_electrons, ring.lone_pair_electrons) for ring in rings],
            [(6, 6, 0)],
        )

    def test_telluropyrylium_takes_one_ring_double_bond(self):
        atoms, bonds = hetero_ring("TE", 6, 2.003, 2.041)
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        te_doubles = [key for key in multiple_bonds(orders) if "TE" in key]
        self.assertEqual(len(te_doubles), 1)
        self.assertEqual(valences(elements, edges, orders)["TE"], 3)
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(len(ring.nodes), ring.pi_electrons, ring.lone_pair_electrons) for ring in rings],
            [(6, 6, 0)],
        )

    def test_selenophene_keeps_its_lone_pair(self):
        atoms, bonds = hetero_ring("SE", 5, 1.872, 1.884)
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        doubles = multiple_bonds(orders)
        self.assertEqual(len(doubles), 2)
        self.assertFalse(any("SE" in key for key in doubles))
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(ring.pi_electrons, ring.lone_pair_electrons) for ring in rings], [(6, 2)]
        )

    def test_tellurophene_keeps_its_lone_pair(self):
        atoms, bonds = hetero_ring("TE", 5, 2.071, 2.078)
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        doubles = multiple_bonds(orders)
        self.assertEqual(len(doubles), 2)
        self.assertFalse(any("TE" in key for key in doubles))
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(ring.pi_electrons, ring.lone_pair_electrons) for ring in rings], [(6, 2)]
        )

    def test_selenoxide_keeps_its_se_oxygen_double(self):
        atoms, bonds = hetero_ring("SE", 6, 1.806, 1.853)
        atoms["O1"] = "O"
        bonds.append(("SE", "O1", 1.646))
        atoms["C6"] = "C"
        bonds.append(("SE", "C6", 1.98))
        for suffix in ("a", "b", "c"):
            atoms["C6" + suffix] = "H"
            bonds.append(("C6", "C6" + suffix, 1.08))
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(multiple_bonds(orders).get(chemistry.edge_key("SE", "O1")), 2)
        self.assertEqual(len([key for key in multiple_bonds(orders) if "SE" in key]), 2)
        self.assertEqual(valences(elements, edges, orders)["SE"], 6)
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(len(ring.nodes), ring.pi_electrons, ring.lone_pair_electrons) for ring in rings],
            [(6, 6, 0)],
        )

    def test_saturated_selenane_has_no_multiple_bonds(self):
        atoms, bonds = hetero_ring("SE", 6, 1.96, 1.96, carbon_carbon=1.54)
        for node in [f"C{index}" for index in range(1, 6)]:
            atoms[node + "H2"] = "H"
            bonds.append((node, node + "H2", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(multiple_bonds(orders), {})
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])

    def test_selenone_draws_two_se_oxygen_double_bonds(self):
        # A selenone Se(=O)2 follows the sulfone: two terminal oxygens, so the
        # selenium is hexavalent and both Se-O bonds draw genuine doubles.
        nodes = [f"X{index}" for index in range(6)]
        atoms = {node: "C" for node in nodes}
        atoms.update({"S0": "SE", "O1": "O", "O2": "O", "M0": "C"})
        bonds = ring_bonds(nodes)
        bonds += [
            ("X0", "S0", 1.96),
            ("S0", "M0", 1.98),
            ("S0", "O1", 1.62),
            ("S0", "O2", 1.64),
        ]
        for index in range(1, 6):
            atoms[f"H{index}"] = "H"
            bonds.append((f"X{index}", f"H{index}", 1.08))
        for index in range(3):
            atoms[f"H{index + 1}m"] = "H"
            bonds.append(("M0", f"H{index + 1}m", 1.09))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        selenone = {key: order for key, order in multiple_bonds(orders).items() if "S0" in key}
        self.assertEqual(
            selenone, {chemistry.edge_key("S0", "O1"): 2, chemistry.edge_key("S0", "O2"): 2}
        )
        self.assertEqual(valences(elements, edges, orders)["S0"], 6)


class OxoPatternTest(unittest.TestCase):
    """Tier-1 oxo seeding: sulfonate/triflate/phosphate oxo patterns.

    The genuine X=O doubles of an oxo acid are decided by the oxo pattern
    (the centre plus its terminal oxidic oxygens) *before* valence
    bookkeeping runs, so a cation salt cannot hide them: a metal-coordinated
    sulfonate still draws exactly two S=O doubles, a phosphate one P=O, and
    the reserved edges survive every later stage untouched.  Boron and the
    halogens are never oxo centres, and a stretched terminal S-O contact is
    still gated out of promotion.
    """

    def _coordinated_sulfonate(self):
        # test1.cif (ABEFUL) shape: an S(O)3 carbon-bearing centre whose every
        # oxygen also contacts a cation (each O has graph degree two, one
        # covalent and one ionic).
        atoms = {"S0": "S", "O1": "O", "O2": "O", "O3": "O", "C1": "C"}
        bonds = [
            ("S0", "O1", 1.43),
            ("S0", "O2", 1.435),
            ("S0", "O3", 1.45),
            ("S0", "C1", 1.77),
        ]
        for index in (1, 2, 3):
            atoms[f"FE{index}"] = "FE"
            bonds.append((f"O{index}", f"FE{index}", 2.0))
        return molecule(atoms, bonds)

    def test_metal_coordinated_sulfonate_draws_two_s_oxygen_doubles(self):
        # Exactly two S=O doubles no matter that every oxygen also bonds a
        # cation: the third oxygen stays single (charged).
        elements, edges, lengths = self._coordinated_sulfonate()
        orders = chemistry.bond_orders(elements, edges, lengths)

        s_doubles = {key: order for key, order in multiple_bonds(orders).items() if "S0" in key}
        self.assertEqual(
            s_doubles, {chemistry.edge_key("S0", "O1"): 2, chemistry.edge_key("S0", "O2"): 2}
        )
        self.assertEqual(valences(elements, edges, orders)["S0"], 6)

    def test_sulfonate_double_set_is_metal_invariant(self):
        # Removing every cation contact must not move a single S=O double.
        plain = molecule(
            {"S0": "S", "O1": "O", "O2": "O", "O3": "O", "C1": "C"},
            [("S0", "O1", 1.43), ("S0", "O2", 1.435), ("S0", "O3", 1.45), ("S0", "C1", 1.77)],
        )
        self.assertEqual(
            chemistry.bond_orders(*self._coordinated_sulfonate()), chemistry.bond_orders(*plain)
        )

    def test_triflate_draws_two_s_oxygen_doubles(self):
        # CF3-SO3-: the oxo pattern is the sulfonate's regardless of the
        # trifluoromethyl substituent on the sulfur.
        atoms = {"S0": "S", "O1": "O", "O2": "O", "O3": "O", "C1": "C"}
        bonds = [
            ("S0", "O1", 1.43),
            ("S0", "O2", 1.435),
            ("S0", "O3", 1.45),
            ("S0", "C1", 1.82),
        ]
        for index in (1, 2, 3):
            atoms[f"F{index}"] = "F"
            bonds.append(("C1", f"F{index}", 1.33))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        s_doubles = {key: order for key, order in multiple_bonds(orders).items() if "S0" in key}
        self.assertEqual(
            s_doubles, {chemistry.edge_key("S0", "O1"): 2, chemistry.edge_key("S0", "O2"): 2}
        )
        self.assertEqual(valences(elements, edges, orders)["S0"], 6)

    def test_phosphate_draws_one_p_oxygen_double(self):
        # A metal phosphate salt: four terminal oxygens, so exactly one P=O
        # (the phosphate analogue of the nitro N=O2 two-for-three) even though
        # every oxygen also coordinates a cation.
        atoms = {"P0": "P", "O1": "O", "O2": "O", "O3": "O", "O4": "O"}
        bonds = [("P0", f"O{index}", 1.52) for index in range(1, 5)]
        for index in (1, 2, 3, 4):
            atoms[f"FE{index}"] = "FE"
            bonds.append((f"O{index}", f"FE{index}", 2.0))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        p_doubles = {key: order for key, order in multiple_bonds(orders).items() if "P0" in key}
        self.assertEqual(p_doubles, {chemistry.edge_key("P0", "O1"): 2})
        self.assertEqual(valences(elements, edges, orders)["P0"], 5)

    def test_perchlorate_has_no_double_bonds(self):
        # Chlorine is neither an oxo-pattern centre nor a multiple-bond
        # element, so a perchlorate stays all-single.
        atoms = {"CL0": "CL", "O1": "O", "O2": "O", "O3": "O", "O4": "O"}
        bonds = [("CL0", f"O{index}", 1.44) for index in range(1, 5)]
        elements, edges, lengths = molecule(atoms, bonds)

        self.assertEqual(chemistry.bond_orders(elements, edges, lengths), {})

    def test_long_terminal_s_oxygen_contact_is_not_promoted(self):
        # A stretched terminal S-O contact (still single) must lose out to the
        # short genuine S=O bonds: the oxo count picks the shortest two and the
        # long bond neither passes the length gate nor finds a deficient
        # partner.
        atoms = {"S0": "S", "O1": "O", "O2": "O", "O3": "O", "C1": "C"}
        bonds = [
            ("S0", "O1", 1.44),
            ("S0", "O2", 1.45),
            ("S0", "O3", 1.72),
            ("S0", "C1", 1.77),
        ]
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        s_doubles = {key: order for key, order in multiple_bonds(orders).items() if "S0" in key}
        self.assertEqual(
            s_doubles, {chemistry.edge_key("S0", "O1"): 2, chemistry.edge_key("S0", "O2"): 2}
        )
        self.assertEqual(valences(elements, edges, orders)["S0"], 6)


class PnictogenRingTest(unittest.TestCase):
    """Arsenic and antimony mirror phosphorus (COD 1573224 / 1555319 / 7111438).

    The group 15 ladder (3, 5) already gives a two-connected ring atom its extra
    order, so arsabenzene draws an As=C ring double (three alternating doubles,
    no lone-pair share).  A five-membered arsole/stibole has an odd number of
    unsatisfied orders, so the heteroatom keeps its lone pair instead.
    """

    def test_arsabenzene_takes_one_ring_double_bond(self):
        atoms, bonds = hetero_ring("AS", 6, 1.859, 1.860)
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        as_doubles = [key for key in multiple_bonds(orders) if "AS" in key]
        self.assertEqual(len(as_doubles), 1)
        self.assertEqual(valences(elements, edges, orders)["AS"], 3)
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(len(ring.nodes), ring.pi_electrons, ring.lone_pair_electrons) for ring in rings],
            [(6, 6, 0)],
        )

    def test_arsole_keeps_its_lone_pair(self):
        atoms, bonds = hetero_ring("AS", 5, 1.943, 1.953)
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        doubles = multiple_bonds(orders)
        self.assertEqual(len(doubles), 2)
        self.assertFalse(any("AS" in key for key in doubles))
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(ring.pi_electrons, ring.lone_pair_electrons) for ring in rings], [(6, 2)]
        )

    def test_stibole_keeps_its_lone_pair(self):
        atoms, bonds = hetero_ring("SB", 5, 2.10, 2.11)
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        doubles = multiple_bonds(orders)
        self.assertEqual(len(doubles), 2)
        self.assertFalse(any("SB" in key for key in doubles))
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(ring.pi_electrons, ring.lone_pair_electrons) for ring in rings], [(6, 2)]
        )


class BoronRingTest(unittest.TestCase):
    """A boron with no double bond contributes 0 pi electrons, not a veto.

    Borazine (B3N3H6) and boroxine (B3O3R3) have an empty p orbital on every
    three-connected boron: the six pi electrons come from the nitrogen or oxygen
    lone pairs, and the ring is aromatic even though no B=N/B=O double bond is
    drawn.  A carbon in the same position is left with an unsatisfied valence
    and still rejects the ring (a saturated boracyclohexane stays non-aromatic).

    Unlike an indolizine, a boron ring is an aromatic network: its heteroatom
    donates its lone pair even when it is shared with the neighbouring rings, so
    an h-BN sheet is aromatic exactly like the graphene sheet it mirrors.
    """

    def test_borazine_is_aromatic_from_the_nitrogen_lone_pairs(self):
        elements, edges, lengths = borazine_ring()
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(multiple_bonds(orders), {})
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(len(ring.nodes), ring.pi_electrons, ring.lone_pair_electrons) for ring in rings],
            [(6, 6, 6)],
        )
        self.assertFalse(rings[0].kekule)

    def test_boroxine_is_aromatic_from_the_oxygen_lone_pairs(self):
        elements, edges, lengths = boroxine_ring()
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(multiple_bonds(orders), {})
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(
            [(len(ring.nodes), ring.pi_electrons, ring.lone_pair_electrons) for ring in rings],
            [(6, 6, 6)],
        )
        self.assertFalse(rings[0].kekule)

    def test_shared_nitrogen_of_a_boron_sheet_still_donates(self):
        # h-BN: every nitrogen belongs to three rings, so the indolizine rule
        # (a lone pair goes only to a ring the heteroatom belongs to alone)
        # would withhold every pair and leave the whole sheet non-aromatic.
        # A ring containing boron is a network, so the shared nitrogen keeps
        # its share of the sextet.  Without boron the same sharing still
        # withholds it -- see test_bridgehead_nitrogen_donates_no_lone_pair.
        elements, edges, lengths = honeycomb(3, 3, 1.45, ("B", "N"))
        orders = chemistry.bond_orders(elements, edges, lengths)

        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertTrue(rings)
        self.assertEqual(
            {(len(ring.nodes), ring.pi_electrons) for ring in rings},
            {(6, 6)},
        )
        self.assertTrue(all(not ring.kekule for ring in rings))

    def test_saturated_boracyclohexane_stays_non_aromatic(self):
        # Boron alone does not make a ring conjugated: the five carbons still
        # have no double bond, so the ring must stay rejected.
        atoms, bonds = hetero_ring("B", 6, 1.60, 1.60, carbon_carbon=1.54)
        for node in [f"C{index}" for index in range(1, 6)]:
            atoms[node + "H2"] = "H"
            bonds.append((node, node + "H2", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])


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
