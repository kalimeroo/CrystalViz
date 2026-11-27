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
    "NA": 1.66,
    "TI": 1.60,
    "SM": 1.98,
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

    def test_eta8_cyclooctatetraenyl_is_a_ten_pi_ligand(self):
        # ABIDUK (CSD): (eta8-COT)(eta5-Cp*) samarium.  Every COT carbon is
        # bonded to its two ring neighbours, one hydrogen and the samarium, so
        # no Kekule double bond can be placed and the Kekule count rejects the
        # ring for an empty p orbital.  The ligand is the aromatic 10 pi
        # COT2- dianion, so it is perceived as aromatic with the metal reported.
        atoms, bonds = cycle("C", 8)
        atoms["SM0"] = "SM"
        for index in range(8):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
            bonds.append((f"C{index}", "SM0", 2.50))
        elements, edges, lengths = molecule(atoms, bonds)

        rings = chemistry.find_rings(elements, edges)
        self.assertEqual([len(ring) for ring in rings], [8])
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(multiple_bonds(orders), {})
        aromatic = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(len(aromatic), 1)
        self.assertEqual(aromatic[0].pi_electrons, 10)
        self.assertEqual(aromatic[0].lone_pair_electrons, 0)
        self.assertEqual(aromatic[0].haptic_metal, "SM")

    def test_plain_eight_ring_carbon_stays_non_aromatic(self):
        # Without the metal, an eight-membered all-carbon ring is not the COT
        # dianion: the ring either carries a Kekule set of four alternating
        # doubles (8 pi, two over the Huckel sextet) or none, and either way is
        # not aromatic -- the 8-ring enumeration must not make it one.
        atoms, bonds = cycle("C", 8)
        for index in range(8):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        self.assertEqual([len(ring) for ring in chemistry.find_rings(elements, edges)], [8])
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])

    def test_tropylium_c7h7_is_a_six_pi_aromatic_ring(self):
        # A bare tropylium cation: three Kekule doubles (six pi electrons) and
        # one empty p orbital at the positively charged centre.  The ring is
        # aromatic like any other six-pi Huckel ring and, having no lone-pair
        # donor, keeps its delocalisation curve in the depiction layer -- the
        # seven-ring analogue of a phenyl ring.
        atoms = {f"C{index}": "C" for index in range(7)}
        bonds = ring_bonds([f"C{index}" for index in range(7)], 1.396)
        for index in range(7):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        self.assertEqual([len(ring) for ring in chemistry.find_rings(elements, edges)], [7])
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(len(multiple_bonds(orders)), 3, multiple_bonds(orders))
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True, lengths=lengths)
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0].pi_electrons, 6)
        self.assertEqual(rings[0].lone_pair_electrons, 0)
        self.assertIsNone(rings[0].haptic_metal)
        self.assertFalse(rings[0].kekule)

    def test_tropylium_centre_needs_the_normalised_lengths(self):
        # The empty-p centre is recognised geometrically because the graph
        # carries no charge: without normalised lengths the allowance cannot
        # fire and the ring is rejected, so callers that skip the geometry
        # never get a fake sextet.
        atoms = {f"C{index}": "C" for index in range(7)}
        bonds = ring_bonds([f"C{index}" for index in range(7)], 1.396)
        for index in range(7):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])

    def test_cycloheptatriene_without_hydrogens_stays_non_aromatic(self):
        # A 1,3,5-cycloheptatriene CIF that omits the hydrogens turns the CH2
        # into a walkable degree-two carbon, so the ring IS enumerated; the
        # length gate must still refuse it, because the CH2's long 1.50 A C-C
        # singles are not the delocalised 1.39-1.42 A bonds of a tropylium
        # centre.
        atoms = {f"C{index}": "C" for index in range(7)}
        bonds = []
        for index in range(7):
            first, second = f"C{index}", f"C{(index + 1) % 7}"
            bonds.append((first, second, 1.34 if index in (1, 3, 5) else 1.50))
        elements, edges, lengths = molecule(atoms, bonds)

        self.assertEqual([len(ring) for ring in chemistry.find_rings(elements, edges)], [7])
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders, lengths=lengths), [])

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

    def test_fused_pyridine_thiophene_pyridine_recovers_both_pyridines(self):
        # AHUCIT: a thiophene fused to two N-alkylpyridinium rings.  A
        # shortest-edge-first valence repair spends the fusion bonds on the
        # thiophene, leaving each pyridine one electron over the sextet (7 pi,
        # not 4n+2).  The three-connected nitrogen already fills the third rung
        # of its (3, 4) ladder, so the repair gives it no double; the Clar
        # re-placement may promote it (it belongs to a single ring) and gives
        # each terminal pyridine its own Kekule sextet, which is the
        # arrangement with the most aromatic rings.
        atoms = {
            "S1": "S",
            "C5": "C",
            "C6": "C",
            "C7": "C",
            "C8": "C",
            "C9": "C",
            "C10": "C",
            "C11": "C",
            "C12": "C",
            "C13": "C",
            "C14": "C",
            "N1": "N",
            "N2": "N",
            "C15": "C",
            "C16": "C",
            "H5": "H",
            "H8": "H",
            "H9": "H",
            "H10": "H",
            "H13": "H",
            "H14": "H",
            "H15A": "H",
            "H15B": "H",
            "H15C": "H",
            "H16A": "H",
            "H16B": "H",
            "H16C": "H",
        }
        bonds = [
            # thiophene (fused to pyridine 2 on C6-C12, to pyridine 1 on C7-C11)
            ("S1", "C6", 1.70),
            ("C6", "C12", 1.39),
            ("C12", "C11", 1.39),
            ("C11", "C7", 1.39),
            ("C7", "S1", 1.70),
            # pyridine 2
            ("C6", "C5", 1.39),
            ("C5", "N1", 1.34),
            ("N1", "C14", 1.34),
            ("C14", "C13", 1.39),
            ("C13", "C12", 1.39),
            # pyridine 1
            ("C7", "C8", 1.39),
            ("C8", "N2", 1.34),
            ("N2", "C9", 1.34),
            ("C9", "C10", 1.39),
            ("C10", "C11", 1.39),
            # N-methyl groups (the pyridinium substituent)
            ("N1", "C15", 1.47),
            ("C15", "H15A", 1.09),
            ("C15", "H15B", 1.09),
            ("C15", "H15C", 1.09),
            ("N2", "C16", 1.47),
            ("C16", "H16A", 1.09),
            ("C16", "H16B", 1.09),
            ("C16", "H16C", 1.09),
            # hydrogens on the non-fusion carbons
            ("C5", "H5", 1.08),
            ("C8", "H8", 1.08),
            ("C9", "H9", 1.08),
            ("C10", "H10", 1.08),
            ("C13", "H13", 1.08),
            ("C14", "H14", 1.08),
        ]
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        rings = chemistry.aromatic_rings(elements, edges, orders)
        sizes = sorted(len(nodes) for nodes, _edges in rings)
        self.assertEqual(sizes, [5, 6, 6], rings)
        pyridines = [nodes for nodes, _edges in rings if len(nodes) == 6]
        self.assertTrue(all("N1" in nodes or "N2" in nodes for nodes in pyridines))

    def test_saturated_fused_ring_stays_non_aromatic(self):
        # Decalin: every carbon already carries four single bonds, so the Clar
        # search has no valence left to place a double bond.
        atoms = {f"C{index}": "C" for index in range(10)}
        bonds = [
            ("C0", "C1", 1.54),
            ("C1", "C2", 1.54),
            ("C2", "C3", 1.54),
            ("C3", "C4", 1.54),
            ("C4", "C5", 1.54),
            ("C5", "C0", 1.54),
            ("C4", "C6", 1.54),
            ("C6", "C7", 1.54),
            ("C7", "C8", 1.54),
            ("C8", "C9", 1.54),
            ("C9", "C5", 1.54),
        ]
        for index in range(10):
            hydrogens = 1 if index in (4, 5) else 2
            for suffix in range(hydrogens):
                atoms[f"H{index}_{suffix}"] = "H"
                bonds.append((f"C{index}", f"H{index}_{suffix}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(multiple_bonds(orders), {})
        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])

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

    def test_indole_without_the_nh_hydrogen_keeps_both_rings(self):
        # The same indole, but the CIF omits the N-H hydrogen (GEXRAC): the
        # repair sees a two-connected nitrogen and gives it a double to reach
        # the third rung.  That double steals the lone pair the five-ring needs,
        # so the Clar re-placement must be free to flatten it -- a lone-pair
        # heteroatom's double is not mandatory the way a carbon's is.
        atoms = {"N0": "N", "C1": "C", "C2": "C"}
        bonds = [
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
        self.assertNotIn(chemistry.edge_key("N0", "C1"), orders)
        self.assertNotIn(chemistry.edge_key("N0", "C4"), orders)
        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual({len(nodes) for nodes, _edges in rings}, {5, 6})

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

    def test_metal_carbonyl_draws_c_oxygen_triple(self):
        # An M-C#O metal carbonyl: the carbon bonds only the dative metal and
        # a short oxygen, so the C-O bond carries both spare orders while the
        # charge sits on the metal -- the plain C=O seed is skipped for such a
        # centre, and the (4,) ladder alone cannot ask for a triple.
        elements, edges, lengths = molecule(
            {"C0": "C", "O0": "O", "FE0": "FE"},
            [("C0", "O0", 1.13), ("C0", "FE0", 1.85)],
        )
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders, {chemistry.edge_key("C0", "O0"): 3})

    def test_metal_acyl_keeps_c_oxygen_double(self):
        # A metal acyl M-C(=O)R: the metal neighbour takes it out of the
        # tier-1 C=O seed, but its carbon still has one spare order and the
        # short C-O is its only usable candidate, so C=O is drawn anyway.
        elements, edges, lengths = molecule(
            {"C0": "C", "O0": "O", "FE0": "FE", "C1": "C", "H1a": "H", "H1b": "H", "H1c": "H"},
            [
                ("C0", "C1", 1.50),
                ("C1", "H1a", 1.09),
                ("C1", "H1b", 1.09),
                ("C1", "H1c", 1.09),
                ("C0", "O0", 1.21),
                ("C0", "FE0", 1.85),
            ],
        )
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders, {chemistry.edge_key("C0", "O0"): 2})


class NOxideValenceTest(unittest.TestCase):
    """Dative N-O bonds (N-oxides) and the genuine N=O of a nitro group.

    A three-connected nitrogen carrying one terminal oxygen is four-valent:
    its fourth bond is a real multiple bond into the ring, while the N-O bond
    itself stays single.  Without that the ring has an odd number of
    unsaturated carbons and no Kekule structure exists at all.  The oxygen is
    only an oxide when the N-O bond is short (pyridine N-oxide ~0.975 of the
    covalent-radii sum): an N-OH ring has a long N-O and must stay non-aromatic
    even when the graph drops the hydroxyl hydrogen.
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

    def test_ring_n_hydroxyl_without_hydrogen_stays_non_aromatic(self):
        # An N-OH ring is keyed on the N-O bond length, not on the graph
        # alone: a real CIF often drops the hydroxyl hydrogen, and without it
        # the oxygen would look terminal and the ring would be promoted to a
        # pyridine N-oxide.  The long N-O (~1.44 A, 1.05 of the radii sum)
        # keeps the nitrogen three-valent, so the ring stays non-aromatic and
        # the N-O bond single.
        elements, edges, lengths = azine_ring({5}, oxide=True)
        lengths[chemistry.edge_key("X5", "O5")] = 1.44 / (RADII["N"] + RADII["O"])
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])
        self.assertNotIn(chemistry.edge_key("X5", "O5"), multiple_bonds(orders))
        totals = valences(elements, edges, orders)
        self.assertEqual(totals["X5"], 3)

    def test_ring_n_hydroxyl_with_explicit_hydrogen_stays_non_aromatic(self):
        # Same N-OH ring with the hydroxyl hydrogen present in the graph: the
        # oxygen is visibly non-terminal, so the nitrogen is three-valent and
        # the ring non-aromatic.
        elements, edges, lengths = azine_ring({5}, oxide=True)
        elements["HO"] = "H"
        edges.append(("O5", "HO"))
        lengths[chemistry.edge_key("O5", "HO")] = 0.97 / (RADII["H"] + RADII["O"])
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(chemistry.aromatic_rings(elements, edges, orders), [])
        self.assertNotIn(chemistry.edge_key("X5", "O5"), multiple_bonds(orders))
        self.assertEqual(valences(elements, edges, orders)["X5"], 3)


class DiazoValenceTest(unittest.TestCase):
    """The terminal N-N chain of a diazo or azide group.

    The diazo group R2C=N+=N- is a cumulene: the terminal N- is two-valent
    and the middle N+ four-valent, so both the C=N and the N=N are drawn
    double and the diazo carbon reaches its four bonds.  EABXIO (succinimidyl
    diazoacetate) is the CSD entry that exposed the gap: the CSD types C6=N2
    (1.313 A) and N2=N3 (1.109 A) as Double.
    """

    def test_diazo_carbon_reaches_four_bonds(self):
        # EABXIO's diazoacetate end: the diazo carbon bonds the ester carbonyl
        # carbon, its hydrogen and the middle nitrogen.
        elements, edges, lengths = molecule(
            {
                "C0": "C",
                "C1": "C",
                "C2": "C",
                "O1": "O",
                "H0": "H",
                "N1": "N",
                "N2": "N",
                "Ha": "H",
                "Hb": "H",
                "Hc": "H",
            },
            [
                ("C0", "C1", 1.46),
                ("C0", "H0", 1.08),
                ("C0", "N1", 1.31),
                ("N1", "N2", 1.11),
                ("C1", "O1", 1.20),
                ("C1", "C2", 1.51),
                ("C2", "Ha", 1.08),
                ("C2", "Hb", 1.08),
                ("C2", "Hc", 1.08),
            ],
        )
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(orders.get(chemistry.edge_key("C0", "N1")), 2)
        self.assertEqual(orders.get(chemistry.edge_key("N1", "N2")), 2)
        totals = valences(elements, edges, orders)
        self.assertEqual(totals["C0"], 4)
        self.assertEqual((totals["N1"], totals["N2"]), (4, 2))

    def test_azide_draws_both_nitrogen_double_bonds(self):
        elements, edges, lengths = molecule(
            {
                "C0": "C",
                "N1": "N",
                "N2": "N",
                "N3": "N",
                "Ha": "H",
                "Hb": "H",
                "Hc": "H",
            },
            [
                ("C0", "N1", 1.47),
                ("N1", "N2", 1.24),
                ("N2", "N3", 1.13),
                ("C0", "Ha", 1.08),
                ("C0", "Hb", 1.08),
                ("C0", "Hc", 1.08),
            ],
        )
        orders = chemistry.bond_orders(elements, edges, lengths)

        self.assertEqual(orders.get(chemistry.edge_key("N1", "N2")), 2)
        self.assertEqual(orders.get(chemistry.edge_key("N2", "N3")), 2)
        totals = valences(elements, edges, orders)
        self.assertEqual((totals["N1"], totals["N2"], totals["N3"]), (3, 4, 2))

    def test_diazonium_keeps_its_single_carbon_nitrogen_bond(self):
        # R-N+=N has a long C-N bond: the middle nitrogen must not be promoted
        # to four-valent, or the aromatic ring carbon would take a double bond.
        atoms, bonds = cycle("C", 6)
        for index in range(1, 6):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        atoms.update({"N1": "N", "N2": "N"})
        bonds += [("C0", "N1", 1.42), ("N1", "N2", 1.10)]
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertNotIn(chemistry.edge_key("C0", "N1"), multiple_bonds(orders))
        self.assertEqual(orders.get(chemistry.edge_key("N1", "N2")), 2)
        self.assertEqual(valences(elements, edges, orders)["C0"], 4)


class MetalRingNitrogenTest(unittest.TestCase):
    """A metal bond is dative: it does not consume the ring N's valence.

    The CSD types the N-C ring bonds of a metal-bound six-ring nitrogen
    Aromatic in 92.7% of the hits, and the five-ring nitrogen is split between
    the pyrrole (lone pair), pyrrolide (single) and dipyrrin (N=C) forms.  The
    nitrogen is therefore promoted only when a carbon of its ring is otherwise
    left with three bonds.
    """

    def test_metal_bound_pyridine_keeps_its_ring_double(self):
        nodes = ["N0", "C0", "C1", "C2", "C3", "C4"]
        atoms = {node: ("N" if node == "N0" else "C") for node in nodes}
        atoms["FE0"] = "FE"
        bonds = ring_bonds(nodes, 1.39)
        bonds.append(("N0", "FE0", 2.0))
        for index in range(5):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        doubles = multiple_bonds(orders)
        self.assertEqual(len(doubles), 3, doubles)
        self.assertTrue(any("N0" in key for key in doubles), doubles)
        rings = chemistry.aromatic_rings(elements, edges, orders)
        self.assertEqual(len(rings), 1)
        self.assertIn("N0", rings[0][0])

    def test_metal_bound_pyrrole_keeps_its_lone_pair(self):
        # Four carbons pair up among themselves, so the nitrogen is not needed:
        # it stays three-valent and donates its lone pair.
        nodes = ["N0", "C0", "C1", "C2", "C3"]
        atoms = {node: ("N" if node == "N0" else "C") for node in nodes}
        atoms["FE0"] = "FE"
        bonds = ring_bonds(nodes, 1.39)
        bonds.append(("N0", "FE0", 2.0))
        for index in range(4):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        doubles = multiple_bonds(orders)
        self.assertEqual(len(doubles), 2, doubles)
        self.assertFalse(any("N0" in key for key in doubles), doubles)
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0].lone_pair_electrons, 2)

    def _dipyrrin_like(self):
        """EDOBEG's core (CSD EDOBEG): a metal-bound dipyrrin.

        Ring B (N2, C34-C37) carries an exocyclic C19=C37 double to the meso
        carbon, so its four carbons cannot pair up and N2 must take the double.
        Ring A (N1, C15-C18) pairs its own carbons and keeps its lone pair.
        """
        atoms = {
            "N1": "N",
            "N2": "N",
            "C15": "C",
            "C16": "C",
            "C17": "C",
            "C18": "C",
            "C19": "C",
            "C34": "C",
            "C35": "C",
            "C36": "C",
            "C37": "C",
            "FE1": "FE",
        }
        bonds = [
            ("N2", "C34", 1.383),
            ("C34", "C35", 1.439),
            ("C35", "C36", 1.337),
            ("C36", "C37", 1.458),
            ("C37", "N2", 1.374),
            ("C19", "C37", 1.391),
            ("C19", "C18", 1.395),
            ("N1", "C18", 1.364),
            ("C17", "C18", 1.434),
            ("C17", "C16", 1.338),
            ("C16", "C15", 1.449),
            ("C15", "N1", 1.392),
            ("N1", "FE1", 2.0),
            ("N2", "FE1", 2.0),
        ]
        for node in ("C15", "C16", "C17", "C35", "C36"):
            atoms[node + "H"] = "H"
            bonds.append((node, node + "H", 1.08))
        # Two phenyls: one on the meso carbon (C19-C38) and one on C34 through
        # the short biaryl link C34-C20 of EDOBEG.
        for base, anchor, distance in (
            ("C38", "C19", 1.49),
            ("C20", "C34", 1.392),
        ):
            phenyl = [base, base + "a", base + "b", base + "c", base + "d", base + "e"]
            atoms.update({node: "C" for node in phenyl})
            bonds.extend(ring_bonds(phenyl, 1.39))
            bonds.append((anchor, base, distance))
            for node in phenyl[1:]:
                atoms[node + "H"] = "H"
                bonds.append((node, node + "H", 1.08))
        return molecule(atoms, bonds)

    def test_dipyrrin_ring_takes_the_nitrogen_double(self):
        elements, edges, lengths = self._dipyrrin_like()
        orders = chemistry.bond_orders(elements, edges, lengths)
        doubles = multiple_bonds(orders)

        self.assertEqual(orders.get(chemistry.edge_key("N2", "C34")), 2)
        self.assertEqual(orders.get(chemistry.edge_key("C35", "C36")), 2)
        self.assertEqual(orders.get(chemistry.edge_key("C19", "C37")), 2)
        self.assertEqual(orders.get(chemistry.edge_key("C15", "C16")), 2)
        self.assertEqual(orders.get(chemistry.edge_key("C17", "C18")), 2)
        self.assertNotIn(chemistry.edge_key("C34", "C20"), doubles)
        for node, element in elements.items():
            if element == "C":
                self.assertEqual(valences(elements, edges, orders)[node], 4, node)
        # The broken five-ring is not aromatic; the intact five-ring and the
        # two phenyls are.
        ring_sizes = sorted(
            len(ring.nodes)
            for ring in chemistry.aromatic_rings(elements, edges, orders, detail=True)
        )
        self.assertEqual(ring_sizes, [5, 6, 6], ring_sizes)

    def test_biaryl_link_stays_single(self):
        atoms = {}
        bonds = []
        for prefix in ("A", "B"):
            nodes = [f"{prefix}{index}" for index in range(6)]
            atoms.update({node: "C" for node in nodes})
            bonds.extend(ring_bonds(nodes, 1.39))
            for index in range(1, 6):
                atoms[f"{prefix}H{index}"] = "H"
                bonds.append((nodes[index], f"{prefix}H{index}", 1.08))
        bonds.append(("A0", "B0", 1.392))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertNotIn(chemistry.edge_key("A0", "B0"), multiple_bonds(orders))
        self.assertEqual(len(multiple_bonds(orders)), 6)
        self.assertEqual(len(chemistry.aromatic_rings(elements, edges, orders)), 2)
        for node, element in elements.items():
            if element == "C":
                self.assertEqual(valences(elements, edges, orders)[node], 4, node)

    def test_inter_ring_double_bond_is_kept_without_another_partner(self):
        # Bifluorenylidene-like: each link carbon is bonded to two ring carbons
        # that are already paired, so the inter-ring bond is the only way left
        # to reach four bonds.
        atoms = {}
        bonds = []
        for prefix in ("A", "B"):
            nodes = [f"{prefix}{index}" for index in range(5)]
            atoms.update({node: "C" for node in nodes})
            bonds.extend(ring_bonds(nodes, 1.39))
            for index in range(1, 5):
                atoms[f"{prefix}H{index}"] = "H"
                bonds.append((nodes[index], f"{prefix}H{index}", 1.08))
        bonds.append(("A0", "B0", 1.35))
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders.get(chemistry.edge_key("A0", "B0")), 2)
        for node, element in elements.items():
            if element == "C":
                self.assertEqual(valences(elements, edges, orders)[node], 4, node)

    def test_nitrogen_bridging_two_alkali_cations_stays_a_ring_atom(self):
        # CEMGAD's C1 family: a phenanthroline nitrogen coordinates two sodium
        # cations, giving it four neighbours.  Sodium is not a HAPTIC_METALS
        # member (it never forms an eta5/6/7 pi ligand), but its contact is a
        # dative one: the ring must still be enumerated and the nitrogen still
        # promoted, exactly as when it carries a transition-metal bond.
        nodes = ["N0", "C0", "C1", "C2", "C3", "C4"]
        atoms = {node: ("N" if node == "N0" else "C") for node in nodes}
        atoms["NA0"] = "NA"
        atoms["NA1"] = "NA"
        bonds = ring_bonds(nodes, 1.39)
        bonds.append(("N0", "NA0", 2.0))
        bonds.append(("N0", "NA1", 2.0))
        for index in range(5):
            atoms[f"H{index}"] = "H"
            bonds.append((f"C{index}", f"H{index}", 1.08))
        elements, edges, lengths = molecule(atoms, bonds)

        rings = chemistry.find_rings(elements, edges)
        self.assertIn(frozenset(nodes), rings)
        orders = chemistry.bond_orders(elements, edges, lengths)
        doubles = multiple_bonds(orders)
        self.assertEqual(len(doubles), 3, doubles)
        self.assertTrue(any("N0" in key for key in doubles), doubles)
        details = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0].lone_pair_electrons, 0)

    def test_alkali_bridged_phenanthroline_keeps_all_three_rings(self):
        # CEMGAD: a phenanthroline whose two pyridine nitrogens each bridge
        # alkali cations.  Before the NONMETALS eligibility rule the N2-type
        # rings were never enumerated, the fusion edges read as exocyclic
        # doubles, and none of the three rings came out aromatic.
        ring_a = ["N1", "C1", "C2", "C3", "C4", "C5"]
        central = ["C1", "C2", "C6", "C12", "C8", "C7"]
        ring_c = ["N2", "C7", "C8", "C9", "C10", "C11"]
        atoms = {
            "N1": "N",
            "N2": "N",
            "NA0": "NA",
            "NA1": "NA",
        }
        atoms.update(
            {
                node: "C"
                for node in (
                    "C1",
                    "C2",
                    "C3",
                    "C4",
                    "C5",
                    "C6",
                    "C7",
                    "C8",
                    "C9",
                    "C10",
                    "C11",
                    "C12",
                )
            }
        )
        bonds = []
        for nodes, distance in (
            (ring_a, 1.35),
            (central, 1.39),
            (ring_c, 1.35),
        ):
            bonds.extend(ring_bonds(nodes, distance))
        # N1 chelates NA0 and bridges NA1; N2 chelates NA1 (CEMGAD bridges a
        # second cation as well, both nitrogens of the ligand are then
        # 4-coordinate).
        bonds.append(("N1", "NA0", 2.0))
        bonds.append(("N1", "NA1", 2.0))
        bonds.append(("N2", "NA1", 2.0))
        for node in ("C3", "C4", "C6", "C10", "C12"):
            atoms[node + "H"] = "H"
            bonds.append((node, node + "H", 1.08))
        bonds.append(("C9", "C9H", 1.08))
        atoms["C9H"] = "H"
        bonds.append(("C11", "C11H", 1.08))
        atoms["C11H"] = "H"
        elements, edges, lengths = molecule(atoms, bonds)

        rings = chemistry.find_rings(elements, edges)
        for nodes in (ring_a, central, ring_c):
            self.assertIn(frozenset(nodes), rings)
        orders = chemistry.bond_orders(elements, edges, lengths)
        details = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        by_nodes = {frozenset(detail.nodes): detail for detail in details}
        for nodes in (ring_a, central, ring_c):
            self.assertIn(frozenset(nodes), by_nodes, nodes)
        self.assertEqual(len(details), 3)
        # the alkalis stay off the pi system: neither ring leans on a lone pair
        # to make its sextet (each pyridine N takes a ring double instead).
        for detail in details:
            self.assertEqual(detail.lone_pair_electrons, 0, detail.nodes)


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


def _pepqag_asu():
    """CSD PEPQAG, one molecule of the asymmetric unit (58 atoms).

    Two six-membered selenium rings joined through biaryl links: a
    selenopyrylium-type torus ring (the SE carries a genuine ring C=Se)
    and an ylidene ring whose SE cannot take a ring double, so the ring
    pairs its two C=C and spends the ylidene double exocyclically (the
    genuine 0.9308 C=C).  Four five-membered thiophenes complete the
    framework -- three aromatic with a lone-pair sulfur, one a
    dihydro-thiophene whose carbons cannot all pair.  The deposited
    coordinates are also the graph node keys, and the coordinate-labelled
    insertion order below is the exact order the import feeds to
    :func:`~chemistry.bond_orders` -- the relaxation-round Kekule race
    (PEPQAG) is sensitive to that order, which is the point of the two
    tests that use this fixture.
    """
    nodes = [
        ((-6.5165, 16.8309, 0.9854), "H"),
        ((-5.642, 16.6101, 1.2844), "C"),
        ((-5.2546, 18.2288, 2.4738), "H"),
        ((-4.9542, 17.396, 2.1279), "C"),
        ((-4.8629, 15.2261, 0.7979), "S"),
        ((-3.6699, 16.7653, 2.4297), "C"),
        ((-3.4881, 15.5939, 1.7724), "C"),
        ((-3.0295, 17.1391, 3.0236), "H"),
        ((-2.7381, 12.9367, 1.2676), "SE"),
        ((-2.3563, 14.6851, 1.8591), "C"),
        ((-1.9891, 10.2422, -0.2687), "S"),
        ((-1.4156, 8.1352, -1.0143), "H"),
        ((-1.1512, 15.0271, 2.3896), "C"),
        ((-1.1403, 8.8294, -0.4272), "C"),
        ((-1.0689, 15.9158, 2.7149), "H"),
        ((-1.0062, 12.2011, 1.4027), "C"),
        ((-0.8749, 10.8597, 0.8663), "C"),
        ((-0.0251, 8.7323, 0.339), "C"),
        ((0.0078, 14.1939, 2.5119), "C"),
        ((0.0215, 12.8662, 1.9731), "C"),
        ((0.4621, 9.8274, 1.3079), "C"),
        ((0.4888, 7.9366, 0.2743), "H"),
        ((0.8475, 12.3997, 2.0231), "H"),
        ((1.0997, 15.6377, 3.4206), "H"),
        ((1.1367, 14.7274, 3.1489), "C"),
        ((1.1956, 9.9085, 1.9046), "H"),
        ((2.3133, 14.0175, 3.4142), "C"),
        ((2.3693, 13.1286, 3.0857), "H"),
        ((3.3897, 15.3902, 4.4693), "H"),
        ((3.405, 14.5077, 4.1189), "C"),
        ((4.4866, 12.8055, 3.9787), "H"),
        ((4.5214, 13.6963, 4.3089), "C"),
        ((5.6836, 14.1072, 4.9499), "C"),
        ((5.6876, 14.9825, 5.321), "H"),
        ((6.0675, 9.523, 3.5694), "H"),
        ((6.1313, 11.619, 4.3095), "H"),
        ((6.609, 7.0777, 3.0746), "H"),
        ((6.8628, 13.3402, 5.1), "C"),
        ((6.8904, 9.1032, 3.7927), "C"),
        ((6.923, 11.9806, 4.6904), "C"),
        ((7.1871, 7.6842, 3.5239), "C"),
        ((7.8352, 14.8771, 6.0184), "H"),
        ((7.92, 9.7422, 4.392), "C"),
        ((7.9721, 13.9836, 5.7274), "C"),
        ((7.9916, 11.1383, 4.7835), "C"),
        ((8.3898, 7.3814, 4.0076), "C"),
        ((8.7445, 6.5006, 3.9815), "H"),
        ((9.203, 13.4547, 5.9522), "C"),
        ((9.2218, 8.6611, 4.6595), "S"),
        ((9.6771, 11.7199, 5.395), "SE"),
        ((9.735, 16.2512, 6.2637), "H"),
        ((10.2608, 14.2037, 6.6115), "C"),
        ((10.3465, 15.638, 6.6702), "C"),
        ((11.5337, 13.5367, 7.4764), "S"),
        ((11.5344, 16.0114, 7.4516), "C"),
        ((11.8024, 16.9072, 7.6211), "H"),
        ((12.189, 14.9444, 7.8926), "C"),
        ((12.9813, 15.007, 8.4122), "H"),
    ]
    bonds = [
        ((-6.5165, 16.8309, 0.9854), (-5.642, 16.6101, 1.2844), 0.8880517651885156),
        ((-5.642, 16.6101, 1.2844), (-4.9542, 17.396, 2.1279), 0.8831610198316495),
        ((-5.642, 16.6101, 1.2844), (-4.8629, 15.2261, 0.7979), 0.9177073177612413),
        ((-5.2546, 18.2288, 2.4738), (-4.9542, 17.396, 2.1279), 0.8883623765678151),
        ((-4.9542, 17.396, 2.1279), (-3.6699, 16.7653, 2.4297), 0.9620361701272656),
        ((-4.8629, 15.2261, 0.7979), (-3.4881, 15.5939, 1.7724), 0.9529596920812352),
        ((-3.6699, 16.7653, 2.4297), (-3.4881, 15.5939, 1.7724), 0.8917117132233537),
        ((-3.6699, 16.7653, 2.4297), (-3.0295, 17.1391, 3.0236), 0.8879486290429958),
        ((-3.4881, 15.5939, 1.7724), (-2.3563, 14.6851, 1.8591), 0.9565986978492933),
        ((-2.7381, 12.9367, 1.2676), (-2.3563, 14.6851, 1.8591), 0.961670877813702),
        ((-2.7381, 12.9367, 1.2676), (-1.0062, 12.2011, 1.4027), 0.9624673497765474),
        ((-2.3563, 14.6851, 1.8591), (-1.1512, 15.0271, 2.3896), 0.894990217742069),
        ((-1.9891, 10.2422, -0.2687), (-1.1403, 8.8294, -0.4272), 0.9147896161747426),
        ((-1.9891, 10.2422, -0.2687), (-0.8749, 10.8597, 0.8663), 0.9426489739671767),
        ((-1.4156, 8.1352, -1.0143), (-1.1403, 8.8294, -0.4272), 0.8877845025955486),
        ((-1.1512, 15.0271, 2.3896), (-1.0689, 15.9158, 2.7149), 0.887803905176081),
        ((-1.1512, 15.0271, 2.3896), (0.0078, 14.1939, 2.5119), 0.9425469178821484),
        ((-1.1403, 8.8294, -0.4272), (-0.0251, 8.7323, 0.339), 0.8924372399293029),
        ((-1.0062, 12.2011, 1.4027), (-0.8749, 10.8597, 0.8663), 0.9543879977229617),
        ((-1.0062, 12.2011, 1.4027), (0.0215, 12.8662, 1.9731), 0.8885126786757467),
        ((-0.8749, 10.8597, 0.8663), (0.4621, 9.8274, 1.3079), 1.1486236236198735),
        ((-0.0251, 8.7323, 0.339), (0.4621, 9.8274, 1.3079), 1.0139269392021437),
        ((-0.0251, 8.7323, 0.339), (0.4888, 7.9366, 0.2743), 0.8873787027174563),
        ((0.0078, 14.1939, 2.5119), (0.0215, 12.8662, 1.9731), 0.9427620326979954),
        ((0.0078, 14.1939, 2.5119), (1.1367, 14.7274, 3.1489), 0.9221470280296151),
        ((0.0215, 12.8662, 1.9731), (0.8475, 12.3997, 2.0231), 0.8877593093281202),
        ((0.4621, 9.8274, 1.3079), (1.1956, 9.9085, 1.9046), 0.88696820282327),
        ((1.0997, 15.6377, 3.4206), (1.1367, 14.7274, 3.1489), 0.8884880690895626),
        ((1.1367, 14.7274, 3.1489), (2.3133, 14.0175, 3.4142), 0.9207296088794429),
        ((2.3133, 14.0175, 3.4142), (2.3693, 13.1286, 3.0857), 0.8872177012745464),
        ((2.3133, 14.0175, 3.4142), (3.405, 14.5077, 4.1189), 0.9136257745504),
        ((3.3897, 15.3902, 4.4693), (3.405, 14.5077, 4.1189), 0.8875417056632889),
        ((3.405, 14.5077, 4.1189), (4.5214, 13.6963, 4.3089), 0.916587858769469),
        ((4.4866, 12.8055, 3.9787), (4.5214, 13.6963, 4.3089), 0.8884756376757467),
        ((4.5214, 13.6963, 4.3089), (5.6836, 14.1072, 4.9499), 0.9140485933326231),
        ((5.6836, 14.1072, 4.9499), (5.6876, 14.9825, 5.321), 0.8884665741742005),
        ((5.6836, 14.1072, 4.9499), (6.8628, 13.3402, 5.1), 0.930759843158247),
        ((6.0675, 9.523, 3.5694), (6.8904, 9.1032, 3.7927), 0.8882446519887559),
        ((6.1313, 11.619, 4.3095), (6.923, 11.9806, 4.6904), 0.8879307166385356),
        ((6.609, 7.0777, 3.0746), (7.1871, 7.6842, 3.5239), 0.8884877598566029),
        ((6.8628, 13.3402, 5.1), (6.923, 11.9806, 4.6904), 0.935021480560345),
        ((6.8628, 13.3402, 5.1), (7.9721, 13.9836, 5.7274), 0.9392789885407603),
        ((6.8904, 9.1032, 3.7927), (7.1871, 7.6842, 3.5239), 0.9700174238229671),
        ((6.8904, 9.1032, 3.7927), (7.92, 9.7422, 4.392), 0.8894195633077419),
        ((6.923, 11.9806, 4.6904), (7.9916, 11.1383, 4.7835), 0.8972486293096603),
        ((7.1871, 7.6842, 3.5239), (8.3898, 7.3814, 4.0076), 0.8758164285496157),
        ((7.8352, 14.8771, 6.0184), (7.9721, 13.9836, 5.7274), 0.8874625924007865),
        ((7.92, 9.7422, 4.392), (7.9916, 11.1383, 4.7835), 0.955043865131485),
        ((7.92, 9.7422, 4.392), (9.2218, 8.6611, 4.6595), 0.9465028097924306),
        ((7.9721, 13.9836, 5.7274), (9.203, 13.4547, 5.9522), 0.8936660553927326),
        ((7.9916, 11.1383, 4.7835), (9.6771, 11.7199, 5.395), 0.9617383561816373),
        ((8.3898, 7.3814, 4.0076), (8.7445, 6.5006, 3.9815), 0.8877108490611966),
        ((8.3898, 7.3814, 4.0076), (9.2218, 8.6611, 4.6595), 0.9170018102557147),
        ((9.203, 13.4547, 5.9522), (9.6771, 11.7199, 5.395), 0.960591246477738),
        ((9.203, 13.4547, 5.9522), (10.2608, 14.2037, 6.6115), 0.9567113857788991),
        ((9.735, 16.2512, 6.2637), (10.3465, 15.638, 6.6702), 0.8940773944440188),
        ((10.2608, 14.2037, 6.6115), (10.3465, 15.638, 6.6702), 0.9460733337236258),
        ((10.2608, 14.2037, 6.6115), (11.5337, 13.5367, 7.4764), 0.9267125225882406),
        ((10.3465, 15.638, 6.6702), (11.5344, 16.0114, 7.4516), 0.967172616823985),
        ((11.5337, 13.5367, 7.4764), (12.189, 14.9444, 7.8926), 0.8881610470215288),
        ((11.5344, 16.0114, 7.4516), (11.8024, 16.9072, 7.6211), 0.8880265403414106),
        ((11.5344, 16.0114, 7.4516), (12.189, 14.9444, 7.8926), 0.8732018681964766),
        ((12.189, 14.9444, 7.8926), (12.9813, 15.007, 8.4122), 0.887381058016527),
    ]
    elements = {node: symbol for node, symbol in nodes}
    edges = [(first, second) for first, second, _ in bonds]
    lengths = {chemistry.edge_key(first, second): length for first, second, length in bonds}
    return elements, edges, lengths


def _pepqag_cell():
    """CSD PEPQAG, the whole cell-window graph (154 atoms, 158 bonds).

    Two full molecules at their cell translations plus the six- and
    seven-atom slivers the cell window truncates at the boundary -- the
    exact graph the import feeds to :func:`~chemistry.bond_orders` in the
    CELL window (the two Se6 rings of the copy of the ASU molecule sit at
    SE(-1.382, -1.029, 1.268) and SE(11.033, -2.246, 5.395)).  The Kekule
    race that :func:`_settled_chalc_ring_reserved` settles only shows up
    on this whole-graph node order: the asymmetric-unit copy below keeps
    its ylidene ring by luck of the insertion order, while the cell copy
    re-solves differently and would gain a spurious C=Se on the ylidene
    SE without the widened (chalc-only-unsatisfied counts as settled)
    freeze.  The coordinates are the graph node keys, verbatim from the
    import, and the insertion order below is the exact order the import
    feeds to :func:`~chemistry.bond_orders`.
    """
    elements = {
        (-6.6942, 8.9928, 5.3693): "H",
        (-5.902, 9.0555, 5.8888): "C",
        (-5.5153, 7.0927, 6.1603): "H",
        (-5.2474, 7.9884, 6.3298): "C",
        (-5.2467, 10.4632, 6.305): "S",
        (-5.1606, 2.8654, 0.9854): "H",
        (-4.2861, 2.6446, 1.2844): "C",
        (-4.0594, 8.3619, 7.1112): "C",
        (-3.9737, 9.7961, 7.1699): "C",
        (-3.8988, 4.2633, 2.4738): "H",
        (-3.5983, 3.4305, 2.1279): "C",
        (-3.5071, 1.2606, 0.7979): "S",
        (-3.4479, 7.7487, 7.5178): "H",
        (-3.3901, 12.2799, 8.3864): "SE",
        (-2.9347, 15.3388, 9.1219): "S",
        (-2.9159, 10.5452, 7.8292): "C",
        (-2.4574, 17.4992, 9.8): "H",
        (-2.3141, 2.7997, 2.4297): "C",
        (-2.1323, 1.6283, 1.7724): "C",
        (-2.1027, 16.6185, 9.7738): "C",
        (-1.7045, 12.8616, 8.9979): "C",
        (-1.6851, 10.0163, 8.054): "C",
        (-1.6736, 3.1735, 3.0236): "H",
        (-1.6329, 14.2576, 9.3894): "C",
        (-1.5481, 9.1228, 7.7631): "H",
        (-1.3823, -1.0289, 1.2676): "SE",
        (-1.0005, 0.7196, 1.8591): "C",
        (-0.9, 16.3157, 10.2575): "C",
        (-0.6751, 6.6794, 7.8003): "N",
        (-0.636, 12.0193, 9.0911): "C",
        (-0.6333, -3.7234, -0.2687): "S",
        (-0.6033, 14.8967, 9.9888): "C",
        (-0.5757, 10.6597, 8.6815): "C",
        (-0.3219, 16.9222, 10.7068): "H",
        (-0.0597, -5.8304, -1.0143): "H",
        (0.1558, 12.3809, 9.472): "H",
        (0.2046, 1.0615, 2.3896): "C",
        (0.2155, -5.1362, -0.4272): "C",
        (0.2196, 14.4769, 10.2121): "H",
        (0.2425, 5.9956, 7.7493): "C",
        (0.2869, 1.9502, 2.7149): "H",
        (0.3496, -1.7644, 1.4027): "C",
        (0.4809, -3.1059, 0.8663): "C",
        (0.5995, 9.0174, 8.4604): "H",
        (0.6035, 9.8926, 8.8316): "C",
        (1.0643, 4.6989, 3.4225): "F",
        (1.1587, 4.2681, 8.087): "H",
        (1.3307, -5.2332, 0.339): "C",
        (1.3637, 0.2284, 2.5119): "C",
        (1.3774, -1.0994, 1.9731): "C",
        (1.3911, 5.1454, 7.719): "C",
        (1.6957, 5.0423, 6.7929): "H",
        (1.7656, 10.3036, 9.4725): "C",
        (1.8005, 11.1944, 9.8027): "H",
        (1.8179, -4.1382, 1.3079): "C",
        (1.8447, -6.0289, 0.2743): "H",
        (1.894, 6.5959, 2.5195): "F",
        (2.0712, 6.2375, 4.7084): "F",
        (2.1074, 5.5455, 8.2551): "H",
        (2.2033, -1.5659, 2.0231): "H",
        (2.4555, 1.6721, 3.4206): "H",
        (2.4925, 0.7618, 3.1489): "C",
        (2.4967, 5.4115, 3.4214): "P",
        (2.5514, -4.057, 1.9046): "H",
        (2.6504, 1.4377, 12.9766): "H",
        (2.8821, 9.4922, 9.6626): "C",
        (2.8908, 4.632, 2.1015): "F",
        (2.8974, 8.6097, 9.3121): "H",
        (3.0915, 4.2567, 4.3195): "F",
        (3.4936, 8.6531, 0.2494): "H",
        (3.5958, 1.3983, 12.723): "C",
        (3.6183, 10.7725, 2.4352): "N",
        (3.6691, 0.052, 3.4142): "C",
        (3.7058, 3.9102, 10.363): "F",
        (3.7251, -0.8369, 3.0857): "H",
        (3.8056, 9.833, 1.8316): "C",
        (3.818, 7.8491, 1.5945): "H",
        (3.8249, 2.1853, 12.1869): "H",
        (3.8373, 0.2014, 11.9499): "C",
        (3.9178, 10.8712, 10.6958): "H",
        (3.9371, 6.1241, 3.4185): "F",
        (3.9738, 9.9823, 10.3672): "C",
        (4.0246, -0.7381, 11.3463): "N",
        (4.0471, 8.636, 1.0584): "C",
        (4.1493, 1.3812, 13.532): "H",
        (4.5514, 5.7776, 9.4619): "F",
        (4.7455, 1.4247, 4.4693): "H",
        (4.7521, 5.4023, 11.6799): "F",
        (4.7608, 0.5421, 4.1189): "C",
        (4.9925, 8.5966, 0.8048): "H",
        (5.0915, 14.0914, 11.8769): "H",
        (5.1462, 4.6229, 10.3601): "P",
        (5.1504, 9.2725, 10.6325): "C",
        (5.1874, 8.3622, 10.3609): "H",
        (5.4396, 11.6002, 11.7583): "H",
        (5.5355, 4.4888, 5.5264): "H",
        (5.5717, 3.7968, 9.073): "F",
        (5.7489, 3.4385, 11.2619): "F",
        (5.7982, 16.0633, 13.5072): "H",
        (5.825, 14.1725, 12.4736): "C",
        (5.8424, -1.1601, 3.9787): "H",
        (5.8773, -0.2693, 4.3089): "C",
        (5.9472, 4.9921, 6.9886): "H",
        (6.2518, 4.8889, 6.0625): "C",
        (6.2655, 11.1337, 11.8084): "C",
        (6.2792, 9.806, 11.2695): "C",
        (6.3122, 15.2676, 13.4424): "C",
        (6.4842, 5.7662, 5.6945): "H",
        (6.5786, 5.3354, 10.359): "F",
        (7.0394, 0.1417, 4.9499): "C",
        (7.0434, 1.0169, 5.321): "H",
        (7.162, 13.1402, 12.9151): "C",
        (7.2933, 11.7988, 12.3788): "C",
        (7.356, 8.0841, 11.0665): "H",
        (7.4004, 4.0387, 6.0321): "C",
        (7.4233, -4.4425, 3.5694): "H",
        (7.4274, 15.1705, 14.2087): "C",
        (7.4383, 8.9728, 11.3919): "C",
        (7.4871, -2.3465, 4.3095): "H",
        (7.7026, 15.8647, 14.7958): "H",
        (7.9648, -6.8878, 3.0746): "H",
        (8.2186, -0.6254, 5.1): "C",
        (8.2462, -4.8623, 3.7927): "C",
        (8.2762, 13.7577, 14.0502): "S",
        (8.2789, -1.985, 4.6904): "C",
        (8.318, 3.355, 5.9811): "N",
        (8.5429, -6.2814, 3.5239): "C",
        (8.6434, 9.3147, 11.9223): "C",
        (9.0252, 11.0632, 12.5138): "SE",
        (9.191, 0.9116, 6.0184): "H",
        (9.2758, -4.2233, 4.392): "C",
        (9.3165, 6.8608, 10.7578): "H",
        (9.328, 0.0181, 5.7274): "C",
        (9.3474, -2.8273, 4.7835): "C",
        (9.7456, -6.5842, 4.0076): "C",
        (9.7752, 8.406, 12.009): "C",
        (9.957, 7.2346, 11.3518): "C",
        (10.1003, -7.4649, 3.9815): "H",
        (10.5588, -0.5108, 5.9522): "C",
        (10.5776, -5.3044, 4.6595): "S",
        (11.033, -2.2456, 5.395): "SE",
        (11.0908, 2.2856, 6.2637): "H",
        (11.15, 8.7738, 12.9835): "S",
        (11.2412, 6.6039, 11.6536): "C",
        (11.5417, 5.7711, 11.3077): "H",
        (11.6166, 0.2382, 6.6115): "C",
        (11.7023, 1.6725, 6.6702): "C",
        (11.929, 7.3898, 12.497): "C",
        (12.8035, 7.169, 12.7961): "H",
        (12.8896, -0.4289, 7.4764): "S",
        (12.8903, 2.0459, 7.4516): "C",
        (13.1582, 2.9416, 7.6211): "H",
        (13.5449, 0.9788, 7.8926): "C",
        (14.3371, 1.0415, 8.4122): "H",
    }
    edges = [
        ((-6.6942, 8.9928, 5.3693), (-5.902, 9.0555, 5.8888)),
        ((-5.902, 9.0555, 5.8888), (-5.2474, 7.9884, 6.3298)),
        ((-5.902, 9.0555, 5.8888), (-5.2467, 10.4632, 6.305)),
        ((-5.5153, 7.0927, 6.1603), (-5.2474, 7.9884, 6.3298)),
        ((-5.2474, 7.9884, 6.3298), (-4.0594, 8.3619, 7.1112)),
        ((-5.2467, 10.4632, 6.305), (-3.9737, 9.7961, 7.1699)),
        ((-5.1606, 2.8654, 0.9854), (-4.2861, 2.6446, 1.2844)),
        ((-4.2861, 2.6446, 1.2844), (-3.5983, 3.4305, 2.1279)),
        ((-4.2861, 2.6446, 1.2844), (-3.5071, 1.2606, 0.7979)),
        ((-4.0594, 8.3619, 7.1112), (-3.9737, 9.7961, 7.1699)),
        ((-4.0594, 8.3619, 7.1112), (-3.4479, 7.7487, 7.5178)),
        ((-3.9737, 9.7961, 7.1699), (-2.9159, 10.5452, 7.8292)),
        ((-3.8988, 4.2633, 2.4738), (-3.5983, 3.4305, 2.1279)),
        ((-3.5983, 3.4305, 2.1279), (-2.3141, 2.7997, 2.4297)),
        ((-3.5071, 1.2606, 0.7979), (-2.1323, 1.6283, 1.7724)),
        ((-3.3901, 12.2799, 8.3864), (-2.9159, 10.5452, 7.8292)),
        ((-3.3901, 12.2799, 8.3864), (-1.7045, 12.8616, 8.9979)),
        ((-2.9347, 15.3388, 9.1219), (-2.1027, 16.6185, 9.7738)),
        ((-2.9347, 15.3388, 9.1219), (-1.6329, 14.2576, 9.3894)),
        ((-2.9159, 10.5452, 7.8292), (-1.6851, 10.0163, 8.054)),
        ((-2.4574, 17.4992, 9.8), (-2.1027, 16.6185, 9.7738)),
        ((-2.3141, 2.7997, 2.4297), (-2.1323, 1.6283, 1.7724)),
        ((-2.3141, 2.7997, 2.4297), (-1.6736, 3.1735, 3.0236)),
        ((-2.1323, 1.6283, 1.7724), (-1.0005, 0.7196, 1.8591)),
        ((-2.1027, 16.6185, 9.7738), (-0.9, 16.3157, 10.2575)),
        ((-1.7045, 12.8616, 8.9979), (-1.6329, 14.2576, 9.3894)),
        ((-1.7045, 12.8616, 8.9979), (-0.636, 12.0193, 9.0911)),
        ((-1.6851, 10.0163, 8.054), (-1.5481, 9.1228, 7.7631)),
        ((-1.6851, 10.0163, 8.054), (-0.5757, 10.6597, 8.6815)),
        ((-1.6329, 14.2576, 9.3894), (-0.6033, 14.8967, 9.9888)),
        ((-1.3823, -1.0289, 1.2676), (-1.0005, 0.7196, 1.8591)),
        ((-1.3823, -1.0289, 1.2676), (0.3496, -1.7644, 1.4027)),
        ((-1.0005, 0.7196, 1.8591), (0.2046, 1.0615, 2.3896)),
        ((-0.9, 16.3157, 10.2575), (-0.6033, 14.8967, 9.9888)),
        ((-0.9, 16.3157, 10.2575), (-0.3219, 16.9222, 10.7068)),
        ((-0.6751, 6.6794, 7.8003), (0.2425, 5.9956, 7.7493)),
        ((-0.636, 12.0193, 9.0911), (-0.5757, 10.6597, 8.6815)),
        ((-0.636, 12.0193, 9.0911), (0.1558, 12.3809, 9.472)),
        ((-0.6333, -3.7234, -0.2687), (0.2155, -5.1362, -0.4272)),
        ((-0.6333, -3.7234, -0.2687), (0.4809, -3.1059, 0.8663)),
        ((-0.6033, 14.8967, 9.9888), (0.2196, 14.4769, 10.2121)),
        ((-0.5757, 10.6597, 8.6815), (0.6035, 9.8926, 8.8316)),
        ((-0.0597, -5.8304, -1.0143), (0.2155, -5.1362, -0.4272)),
        ((0.2046, 1.0615, 2.3896), (0.2869, 1.9502, 2.7149)),
        ((0.2046, 1.0615, 2.3896), (1.3637, 0.2284, 2.5119)),
        ((0.2155, -5.1362, -0.4272), (1.3307, -5.2332, 0.339)),
        ((0.2425, 5.9956, 7.7493), (1.3911, 5.1454, 7.719)),
        ((0.3496, -1.7644, 1.4027), (0.4809, -3.1059, 0.8663)),
        ((0.3496, -1.7644, 1.4027), (1.3774, -1.0994, 1.9731)),
        ((0.4809, -3.1059, 0.8663), (1.8179, -4.1382, 1.3079)),
        ((0.5995, 9.0174, 8.4604), (0.6035, 9.8926, 8.8316)),
        ((0.6035, 9.8926, 8.8316), (1.7656, 10.3036, 9.4725)),
        ((1.0643, 4.6989, 3.4225), (2.4967, 5.4115, 3.4214)),
        ((1.1587, 4.2681, 8.087), (1.3911, 5.1454, 7.719)),
        ((1.3307, -5.2332, 0.339), (1.8179, -4.1382, 1.3079)),
        ((1.3307, -5.2332, 0.339), (1.8447, -6.0289, 0.2743)),
        ((1.3637, 0.2284, 2.5119), (1.3774, -1.0994, 1.9731)),
        ((1.3637, 0.2284, 2.5119), (2.4925, 0.7618, 3.1489)),
        ((1.3774, -1.0994, 1.9731), (2.2033, -1.5659, 2.0231)),
        ((1.3911, 5.1454, 7.719), (1.6957, 5.0423, 6.7929)),
        ((1.3911, 5.1454, 7.719), (2.1074, 5.5455, 8.2551)),
        ((1.7656, 10.3036, 9.4725), (1.8005, 11.1944, 9.8027)),
        ((1.7656, 10.3036, 9.4725), (2.8821, 9.4922, 9.6626)),
        ((1.8179, -4.1382, 1.3079), (2.5514, -4.057, 1.9046)),
        ((1.894, 6.5959, 2.5195), (2.4967, 5.4115, 3.4214)),
        ((2.0712, 6.2375, 4.7084), (2.4967, 5.4115, 3.4214)),
        ((2.4555, 1.6721, 3.4206), (2.4925, 0.7618, 3.1489)),
        ((2.4925, 0.7618, 3.1489), (3.6691, 0.052, 3.4142)),
        ((2.4967, 5.4115, 3.4214), (2.8908, 4.632, 2.1015)),
        ((2.4967, 5.4115, 3.4214), (3.0915, 4.2567, 4.3195)),
        ((2.4967, 5.4115, 3.4214), (3.9371, 6.1241, 3.4185)),
        ((2.6504, 1.4377, 12.9766), (3.5958, 1.3983, 12.723)),
        ((2.8821, 9.4922, 9.6626), (2.8974, 8.6097, 9.3121)),
        ((2.8821, 9.4922, 9.6626), (3.9738, 9.9823, 10.3672)),
        ((3.4936, 8.6531, 0.2494), (4.0471, 8.636, 1.0584)),
        ((3.5958, 1.3983, 12.723), (3.8249, 2.1853, 12.1869)),
        ((3.5958, 1.3983, 12.723), (3.8373, 0.2014, 11.9499)),
        ((3.5958, 1.3983, 12.723), (4.1493, 1.3812, 13.532)),
        ((3.6183, 10.7725, 2.4352), (3.8056, 9.833, 1.8316)),
        ((3.6691, 0.052, 3.4142), (3.7251, -0.8369, 3.0857)),
        ((3.6691, 0.052, 3.4142), (4.7608, 0.5421, 4.1189)),
        ((3.7058, 3.9102, 10.363), (5.1462, 4.6229, 10.3601)),
        ((3.8056, 9.833, 1.8316), (4.0471, 8.636, 1.0584)),
        ((3.818, 7.8491, 1.5945), (4.0471, 8.636, 1.0584)),
        ((3.8373, 0.2014, 11.9499), (4.0246, -0.7381, 11.3463)),
        ((3.9178, 10.8712, 10.6958), (3.9738, 9.9823, 10.3672)),
        ((3.9738, 9.9823, 10.3672), (5.1504, 9.2725, 10.6325)),
        ((4.0471, 8.636, 1.0584), (4.9925, 8.5966, 0.8048)),
        ((4.5514, 5.7776, 9.4619), (5.1462, 4.6229, 10.3601)),
        ((4.7455, 1.4247, 4.4693), (4.7608, 0.5421, 4.1189)),
        ((4.7521, 5.4023, 11.6799), (5.1462, 4.6229, 10.3601)),
        ((4.7608, 0.5421, 4.1189), (5.8773, -0.2693, 4.3089)),
        ((5.0915, 14.0914, 11.8769), (5.825, 14.1725, 12.4736)),
        ((5.1462, 4.6229, 10.3601), (5.5717, 3.7968, 9.073)),
        ((5.1462, 4.6229, 10.3601), (5.7489, 3.4385, 11.2619)),
        ((5.1462, 4.6229, 10.3601), (6.5786, 5.3354, 10.359)),
        ((5.1504, 9.2725, 10.6325), (5.1874, 8.3622, 10.3609)),
        ((5.1504, 9.2725, 10.6325), (6.2792, 9.806, 11.2695)),
        ((5.4396, 11.6002, 11.7583), (6.2655, 11.1337, 11.8084)),
        ((5.5355, 4.4888, 5.5264), (6.2518, 4.8889, 6.0625)),
        ((5.7982, 16.0633, 13.5072), (6.3122, 15.2676, 13.4424)),
        ((5.825, 14.1725, 12.4736), (6.3122, 15.2676, 13.4424)),
        ((5.825, 14.1725, 12.4736), (7.162, 13.1402, 12.9151)),
        ((5.8424, -1.1601, 3.9787), (5.8773, -0.2693, 4.3089)),
        ((5.8773, -0.2693, 4.3089), (7.0394, 0.1417, 4.9499)),
        ((5.9472, 4.9921, 6.9886), (6.2518, 4.8889, 6.0625)),
        ((6.2518, 4.8889, 6.0625), (6.4842, 5.7662, 5.6945)),
        ((6.2518, 4.8889, 6.0625), (7.4004, 4.0387, 6.0321)),
        ((6.2655, 11.1337, 11.8084), (6.2792, 9.806, 11.2695)),
        ((6.2655, 11.1337, 11.8084), (7.2933, 11.7988, 12.3788)),
        ((6.2792, 9.806, 11.2695), (7.4383, 8.9728, 11.3919)),
        ((6.3122, 15.2676, 13.4424), (7.4274, 15.1705, 14.2087)),
        ((7.0394, 0.1417, 4.9499), (7.0434, 1.0169, 5.321)),
        ((7.0394, 0.1417, 4.9499), (8.2186, -0.6254, 5.1)),
        ((7.162, 13.1402, 12.9151), (7.2933, 11.7988, 12.3788)),
        ((7.162, 13.1402, 12.9151), (8.2762, 13.7577, 14.0502)),
        ((7.2933, 11.7988, 12.3788), (9.0252, 11.0632, 12.5138)),
        ((7.356, 8.0841, 11.0665), (7.4383, 8.9728, 11.3919)),
        ((7.4004, 4.0387, 6.0321), (8.318, 3.355, 5.9811)),
        ((7.4233, -4.4425, 3.5694), (8.2462, -4.8623, 3.7927)),
        ((7.4274, 15.1705, 14.2087), (7.7026, 15.8647, 14.7958)),
        ((7.4274, 15.1705, 14.2087), (8.2762, 13.7577, 14.0502)),
        ((7.4383, 8.9728, 11.3919), (8.6434, 9.3147, 11.9223)),
        ((7.4871, -2.3465, 4.3095), (8.2789, -1.985, 4.6904)),
        ((7.9648, -6.8878, 3.0746), (8.5429, -6.2814, 3.5239)),
        ((8.2186, -0.6254, 5.1), (8.2789, -1.985, 4.6904)),
        ((8.2186, -0.6254, 5.1), (9.328, 0.0181, 5.7274)),
        ((8.2462, -4.8623, 3.7927), (8.5429, -6.2814, 3.5239)),
        ((8.2462, -4.8623, 3.7927), (9.2758, -4.2233, 4.392)),
        ((8.2789, -1.985, 4.6904), (9.3474, -2.8273, 4.7835)),
        ((8.5429, -6.2814, 3.5239), (9.7456, -6.5842, 4.0076)),
        ((8.6434, 9.3147, 11.9223), (9.0252, 11.0632, 12.5138)),
        ((8.6434, 9.3147, 11.9223), (9.7752, 8.406, 12.009)),
        ((9.191, 0.9116, 6.0184), (9.328, 0.0181, 5.7274)),
        ((9.2758, -4.2233, 4.392), (9.3474, -2.8273, 4.7835)),
        ((9.2758, -4.2233, 4.392), (10.5776, -5.3044, 4.6595)),
        ((9.3165, 6.8608, 10.7578), (9.957, 7.2346, 11.3518)),
        ((9.328, 0.0181, 5.7274), (10.5588, -0.5108, 5.9522)),
        ((9.3474, -2.8273, 4.7835), (11.033, -2.2456, 5.395)),
        ((9.7456, -6.5842, 4.0076), (10.1003, -7.4649, 3.9815)),
        ((9.7456, -6.5842, 4.0076), (10.5776, -5.3044, 4.6595)),
        ((9.7752, 8.406, 12.009), (9.957, 7.2346, 11.3518)),
        ((9.7752, 8.406, 12.009), (11.15, 8.7738, 12.9835)),
        ((9.957, 7.2346, 11.3518), (11.2412, 6.6039, 11.6536)),
        ((10.5588, -0.5108, 5.9522), (11.033, -2.2456, 5.395)),
        ((10.5588, -0.5108, 5.9522), (11.6166, 0.2382, 6.6115)),
        ((11.0908, 2.2856, 6.2637), (11.7023, 1.6725, 6.6702)),
        ((11.15, 8.7738, 12.9835), (11.929, 7.3898, 12.497)),
        ((11.2412, 6.6039, 11.6536), (11.5417, 5.7711, 11.3077)),
        ((11.2412, 6.6039, 11.6536), (11.929, 7.3898, 12.497)),
        ((11.6166, 0.2382, 6.6115), (11.7023, 1.6725, 6.6702)),
        ((11.6166, 0.2382, 6.6115), (12.8896, -0.4289, 7.4764)),
        ((11.7023, 1.6725, 6.6702), (12.8903, 2.0459, 7.4516)),
        ((11.929, 7.3898, 12.497), (12.8035, 7.169, 12.7961)),
        ((12.8896, -0.4289, 7.4764), (13.5449, 0.9788, 7.8926)),
        ((12.8903, 2.0459, 7.4516), (13.1582, 2.9416, 7.6211)),
        ((12.8903, 2.0459, 7.4516), (13.5449, 0.9788, 7.8926)),
        ((13.5449, 0.9788, 7.8926), (14.3371, 1.0415, 8.4122)),
    ]
    lengths = {
        ((-6.6942, 8.9928, 5.3693), (-5.902, 9.0555, 5.8888)): 0.8873810580165276,
        ((-5.902, 9.0555, 5.8888), (-5.2474, 7.9884, 6.3298)): 0.8732018681964765,
        ((-5.902, 9.0555, 5.8888), (-5.2467, 10.4632, 6.305)): 0.8881610470215262,
        ((-5.5153, 7.0927, 6.1603), (-5.2474, 7.9884, 6.3298)): 0.8880265403414135,
        ((-5.2474, 7.9884, 6.3298), (-4.0594, 8.3619, 7.1112)): 0.9671726168239861,
        ((-5.2467, 10.4632, 6.305), (-3.9737, 9.7961, 7.1699)): 0.9267125225882413,
        ((-5.1606, 2.8654, 0.9854), (-4.2861, 2.6446, 1.2844)): 0.8880517651885149,
        ((-4.2861, 2.6446, 1.2844), (-3.5983, 3.4305, 2.1279)): 0.8831610198316483,
        ((-4.2861, 2.6446, 1.2844), (-3.5071, 1.2606, 0.7979)): 0.9177073177612417,
        ((-4.0594, 8.3619, 7.1112), (-3.9737, 9.7961, 7.1699)): 0.9460733337236247,
        ((-4.0594, 8.3619, 7.1112), (-3.4479, 7.7487, 7.5178)): 0.8940773944440157,
        ((-3.9737, 9.7961, 7.1699), (-2.9159, 10.5452, 7.8292)): 0.9567113857788996,
        ((-3.8988, 4.2633, 2.4738), (-3.5983, 3.4305, 2.1279)): 0.8883623765678162,
        ((-3.5983, 3.4305, 2.1279), (-2.3141, 2.7997, 2.4297)): 0.9620361701272652,
        ((-3.5071, 1.2606, 0.7979), (-2.1323, 1.6283, 1.7724)): 0.9529596920812353,
        ((-3.3901, 12.2799, 8.3864), (-2.9159, 10.5452, 7.8292)): 0.9605912464777366,
        ((-3.3901, 12.2799, 8.3864), (-1.7045, 12.8616, 8.9979)): 0.9617383561816377,
        ((-2.9347, 15.3388, 9.1219), (-2.1027, 16.6185, 9.7738)): 0.9170018102557154,
        ((-2.9347, 15.3388, 9.1219), (-1.6329, 14.2576, 9.3894)): 0.9465028097924315,
        ((-2.9159, 10.5452, 7.8292), (-1.6851, 10.0163, 8.054)): 0.8936660553927348,
        ((-2.4574, 17.4992, 9.8), (-2.1027, 16.6185, 9.7738)): 0.8877108490611983,
        ((-2.3141, 2.7997, 2.4297), (-2.1323, 1.6283, 1.7724)): 0.8917117132233523,
        ((-2.3141, 2.7997, 2.4297), (-1.6736, 3.1735, 3.0236)): 0.8879486290429964,
        ((-2.1323, 1.6283, 1.7724), (-1.0005, 0.7196, 1.8591)): 0.9565986978492944,
        ((-2.1027, 16.6185, 9.7738), (-0.9, 16.3157, 10.2575)): 0.8758164285496143,
        ((-1.7045, 12.8616, 8.9979), (-1.6329, 14.2576, 9.3894)): 0.955043865131485,
        ((-1.7045, 12.8616, 8.9979), (-0.636, 12.0193, 9.0911)): 0.8972486293096597,
        ((-1.6851, 10.0163, 8.054), (-1.5481, 9.1228, 7.7631)): 0.8874625924007861,
        ((-1.6851, 10.0163, 8.054), (-0.5757, 10.6597, 8.6815)): 0.9392789885407598,
        ((-1.6329, 14.2576, 9.3894), (-0.6033, 14.8967, 9.9888)): 0.8894195633077442,
        ((-1.3823, -1.0289, 1.2676), (-1.0005, 0.7196, 1.8591)): 0.961670877813702,
        ((-1.3823, -1.0289, 1.2676), (0.3496, -1.7644, 1.4027)): 0.9624673497765475,
        ((-1.0005, 0.7196, 1.8591), (0.2046, 1.0615, 2.3896)): 0.8949902177420691,
        ((-0.9, 16.3157, 10.2575), (-0.6033, 14.8967, 9.9888)): 0.9700174238229677,
        ((-0.9, 16.3157, 10.2575), (-0.3219, 16.9222, 10.7068)): 0.8884877598566017,
        ((-0.6751, 6.6794, 7.8003), (0.2425, 5.9956, 7.7493)): 0.7792104882861074,
        ((-0.636, 12.0193, 9.0911), (-0.5757, 10.6597, 8.6815)): 0.9350214805603437,
        ((-0.636, 12.0193, 9.0911), (0.1558, 12.3809, 9.472)): 0.8879307166385374,
        ((-0.6333, -3.7234, -0.2687), (0.2155, -5.1362, -0.4272)): 0.9147896161747422,
        ((-0.6333, -3.7234, -0.2687), (0.4809, -3.1059, 0.8663)): 0.9426489739671766,
        ((-0.6033, 14.8967, 9.9888), (0.2196, 14.4769, 10.2121)): 0.8882446519887566,
        ((-0.5757, 10.6597, 8.6815), (0.6035, 9.8926, 8.8316)): 0.9307598431582476,
        ((-0.0597, -5.8304, -1.0143), (0.2155, -5.1362, -0.4272)): 0.8877845025955486,
        ((0.2046, 1.0615, 2.3896), (0.2869, 1.9502, 2.7149)): 0.8878039051760798,
        ((0.2046, 1.0615, 2.3896), (1.3637, 0.2284, 2.5119)): 0.9425469178821487,
        ((0.2155, -5.1362, -0.4272), (1.3307, -5.2332, 0.339)): 0.8924372399293029,
        ((0.2425, 5.9956, 7.7493), (1.3911, 5.1454, 7.719)): 0.9403981352553122,
        ((0.3496, -1.7644, 1.4027), (0.4809, -3.1059, 0.8663)): 0.954387997722963,
        ((0.3496, -1.7644, 1.4027), (1.3774, -1.0994, 1.9731)): 0.8885126786757467,
        ((0.4809, -3.1059, 0.8663), (1.8179, -4.1382, 1.3079)): 1.148623623619873,
        ((0.5995, 9.0174, 8.4604), (0.6035, 9.8926, 8.8316)): 0.8884665741741996,
        ((0.6035, 9.8926, 8.8316), (1.7656, 10.3036, 9.4725)): 0.914048593332623,
        ((1.0643, 4.6989, 3.4225), (2.4967, 5.4115, 3.4214)): 0.9755158641411987,
        ((1.1587, 4.2681, 8.087), (1.3911, 5.1454, 7.719)): 0.9152262801340295,
        ((1.3307, -5.2332, 0.339), (1.8179, -4.1382, 1.3079)): 1.0139269392021424,
        ((1.3307, -5.2332, 0.339), (1.8447, -6.0289, 0.2743)): 0.8873787027174577,
        ((1.3637, 0.2284, 2.5119), (1.3774, -1.0994, 1.9731)): 0.9427620326979951,
        ((1.3637, 0.2284, 2.5119), (2.4925, 0.7618, 3.1489)): 0.9221470280296149,
        ((1.3774, -1.0994, 1.9731), (2.2033, -1.5659, 2.0231)): 0.8877593093281207,
        ((1.3911, 5.1454, 7.719), (1.6957, 5.0423, 6.7929)): 0.9162297542250543,
        ((1.3911, 5.1454, 7.719), (2.1074, 5.5455, 8.2551)): 0.9159663647164878,
        ((1.7656, 10.3036, 9.4725), (1.8005, 11.1944, 9.8027)): 0.888475637675748,
        ((1.7656, 10.3036, 9.4725), (2.8821, 9.4922, 9.6626)): 0.9165878587694682,
        ((1.8179, -4.1382, 1.3079), (2.5514, -4.057, 1.9046)): 0.8869682028232702,
        ((1.894, 6.5959, 2.5195), (2.4967, 5.4115, 3.4214)): 0.9793127093883738,
        ((2.0712, 6.2375, 4.7084), (2.4967, 5.4115, 3.4214)): 0.9679526626012623,
        ((2.4555, 1.6721, 3.4206), (2.4925, 0.7618, 3.1489)): 0.8884880690895618,
        ((2.4925, 0.7618, 3.1489), (3.6691, 0.052, 3.4142)): 0.9207296088794433,
        ((2.4967, 5.4115, 3.4214), (2.8908, 4.632, 2.1015)): 0.9650407757751348,
        ((2.4967, 5.4115, 3.4214), (3.0915, 4.2567, 4.3195)): 0.9629167378816267,
        ((2.4967, 5.4115, 3.4214), (3.9371, 6.1241, 3.4185)): 0.9799029816447191,
        ((2.6504, 1.4377, 12.9766), (3.5958, 1.3983, 12.723)): 0.9154870135244416,
        ((2.8821, 9.4922, 9.6626), (2.8974, 8.6097, 9.3121)): 0.8875417056632898,
        ((2.8821, 9.4922, 9.6626), (3.9738, 9.9823, 10.3672)): 0.9136257745503995,
        ((3.4936, 8.6531, 0.2494), (4.0471, 8.636, 1.0584)): 0.9162233545575502,
        ((3.5958, 1.3983, 12.723), (3.8249, 2.1853, 12.1869)): 0.9153113712310261,
        ((3.5958, 1.3983, 12.723), (3.8373, 0.2014, 11.9499)): 0.9508361682661038,
        ((3.5958, 1.3983, 12.723), (4.1493, 1.3812, 13.532)): 0.916223354557549,
        ((3.6183, 10.7725, 2.4352), (3.8056, 9.833, 1.8316)): 0.7702522644465132,
        ((3.6691, 0.052, 3.4142), (3.7251, -0.8369, 3.0857)): 0.8872177012745467,
        ((3.6691, 0.052, 3.4142), (4.7608, 0.5421, 4.1189)): 0.9136257745503997,
        ((3.7058, 3.9102, 10.363), (5.1462, 4.6229, 10.3601)): 0.9799029816447191,
        ((3.8056, 9.833, 1.8316), (4.0471, 8.636, 1.0584)): 0.9508361682661046,
        ((3.818, 7.8491, 1.5945), (4.0471, 8.636, 1.0584)): 0.9153113712310238,
        ((3.8373, 0.2014, 11.9499), (4.0246, -0.7381, 11.3463)): 0.7702522644465133,
        ((3.9178, 10.8712, 10.6958), (3.9738, 9.9823, 10.3672)): 0.8872177012745465,
        ((3.9738, 9.9823, 10.3672), (5.1504, 9.2725, 10.6325)): 0.9207296088794424,
        ((4.0471, 8.636, 1.0584), (4.9925, 8.5966, 0.8048)): 0.9154870135244424,
        ((4.5514, 5.7776, 9.4619), (5.1462, 4.6229, 10.3601)): 0.962916737881627,
        ((4.7455, 1.4247, 4.4693), (4.7608, 0.5421, 4.1189)): 0.88754170566329,
        ((4.7521, 5.4023, 11.6799), (5.1462, 4.6229, 10.3601)): 0.965040775775135,
        ((4.7608, 0.5421, 4.1189), (5.8773, -0.2693, 4.3089)): 0.9165878587694684,
        ((5.0915, 14.0914, 11.8769), (5.825, 14.1725, 12.4736)): 0.8869682028232707,
        ((5.1462, 4.6229, 10.3601), (5.5717, 3.7968, 9.073)): 0.9679526626012622,
        ((5.1462, 4.6229, 10.3601), (5.7489, 3.4385, 11.2619)): 0.9793127093883739,
        ((5.1462, 4.6229, 10.3601), (6.5786, 5.3354, 10.359)): 0.9755158641411983,
        ((5.1504, 9.2725, 10.6325), (5.1874, 8.3622, 10.3609)): 0.8884880690895621,
        ((5.1504, 9.2725, 10.6325), (6.2792, 9.806, 11.2695)): 0.9221470280296157,
        ((5.4396, 11.6002, 11.7583), (6.2655, 11.1337, 11.8084)): 0.8877593093281212,
        ((5.5355, 4.4888, 5.5264), (6.2518, 4.8889, 6.0625)): 0.9159663647164875,
        ((5.7982, 16.0633, 13.5072), (6.3122, 15.2676, 13.4424)): 0.8873787027174604,
        ((5.825, 14.1725, 12.4736), (6.3122, 15.2676, 13.4424)): 1.0139269392021397,
        ((5.825, 14.1725, 12.4736), (7.162, 13.1402, 12.9151)): 1.1486236236198746,
        ((5.8424, -1.1601, 3.9787), (5.8773, -0.2693, 4.3089)): 0.8884756376757484,
        ((5.8773, -0.2693, 4.3089), (7.0394, 0.1417, 4.9499)): 0.9140485933326231,
        ((5.9472, 4.9921, 6.9886), (6.2518, 4.8889, 6.0625)): 0.9162297542250536,
        ((6.2518, 4.8889, 6.0625), (6.4842, 5.7662, 5.6945)): 0.9152262801340291,
        ((6.2518, 4.8889, 6.0625), (7.4004, 4.0387, 6.0321)): 0.9403981352553119,
        ((6.2655, 11.1337, 11.8084), (6.2792, 9.806, 11.2695)): 0.942762032697994,
        ((6.2655, 11.1337, 11.8084), (7.2933, 11.7988, 12.3788)): 0.8885126786757465,
        ((6.2792, 9.806, 11.2695), (7.4383, 8.9728, 11.3919)): 0.9425469178821483,
        ((6.3122, 15.2676, 13.4424), (7.4274, 15.1705, 14.2087)): 0.8924372399293029,
        ((7.0394, 0.1417, 4.9499), (7.0434, 1.0169, 5.321)): 0.8884665741741988,
        ((7.0394, 0.1417, 4.9499), (8.2186, -0.6254, 5.1)): 0.9307598431582468,
        ((7.162, 13.1402, 12.9151), (7.2933, 11.7988, 12.3788)): 0.954387997722963,
        ((7.162, 13.1402, 12.9151), (8.2762, 13.7577, 14.0502)): 0.9426489739671763,
        ((7.2933, 11.7988, 12.3788), (9.0252, 11.0632, 12.5138)): 0.9624673497765481,
        ((7.356, 8.0841, 11.0665), (7.4383, 8.9728, 11.3919)): 0.8878039051760788,
        ((7.4004, 4.0387, 6.0321), (8.318, 3.355, 5.9811)): 0.7792104882861093,
        ((7.4233, -4.4425, 3.5694), (8.2462, -4.8623, 3.7927)): 0.8882446519887556,
        ((7.4274, 15.1705, 14.2087), (7.7026, 15.8647, 14.7958)): 0.8877845025955473,
        ((7.4274, 15.1705, 14.2087), (8.2762, 13.7577, 14.0502)): 0.9147896161747424,
        ((7.4383, 8.9728, 11.3919), (8.6434, 9.3147, 11.9223)): 0.8949902177420717,
        ((7.4871, -2.3465, 4.3095), (8.2789, -1.985, 4.6904)): 0.8879307166385358,
        ((7.9648, -6.8878, 3.0746), (8.5429, -6.2814, 3.5239)): 0.8884877598566014,
        ((8.2186, -0.6254, 5.1), (8.2789, -1.985, 4.6904)): 0.9350214805603456,
        ((8.2186, -0.6254, 5.1), (9.328, 0.0181, 5.7274)): 0.9392789885407593,
        ((8.2462, -4.8623, 3.7927), (8.5429, -6.2814, 3.5239)): 0.9700174238229673,
        ((8.2462, -4.8623, 3.7927), (9.2758, -4.2233, 4.392)): 0.8894195633077435,
        ((8.2789, -1.985, 4.6904), (9.3474, -2.8273, 4.7835)): 0.8972486293096594,
        ((8.5429, -6.2814, 3.5239), (9.7456, -6.5842, 4.0076)): 0.8758164285496155,
        ((8.6434, 9.3147, 11.9223), (9.0252, 11.0632, 12.5138)): 0.9616708778137004,
        ((8.6434, 9.3147, 11.9223), (9.7752, 8.406, 12.009)): 0.9565986978492939,
        ((9.191, 0.9116, 6.0184), (9.328, 0.0181, 5.7274)): 0.8874625924007865,
        ((9.2758, -4.2233, 4.392), (9.3474, -2.8273, 4.7835)): 0.9550438651314863,
        ((9.2758, -4.2233, 4.392), (10.5776, -5.3044, 4.6595)): 0.9465028097924296,
        ((9.3165, 6.8608, 10.7578), (9.957, 7.2346, 11.3518)): 0.8879486290429978,
        ((9.328, 0.0181, 5.7274), (10.5588, -0.5108, 5.9522)): 0.893666055392734,
        ((9.3474, -2.8273, 4.7835), (11.033, -2.2456, 5.395)): 0.961738356181637,
        ((9.7456, -6.5842, 4.0076), (10.1003, -7.4649, 3.9815)): 0.8877108490611966,
        ((9.7456, -6.5842, 4.0076), (10.5776, -5.3044, 4.6595)): 0.9170018102557153,
        ((9.7752, 8.406, 12.009), (9.957, 7.2346, 11.3518)): 0.8917117132233529,
        ((9.7752, 8.406, 12.009), (11.15, 8.7738, 12.9835)): 0.9529596920812349,
        ((9.957, 7.2346, 11.3518), (11.2412, 6.6039, 11.6536)): 0.9620361701272651,
        ((10.5588, -0.5108, 5.9522), (11.033, -2.2456, 5.395)): 0.9605912464777382,
        ((10.5588, -0.5108, 5.9522), (11.6166, 0.2382, 6.6115)): 0.9567113857788991,
        ((11.0908, 2.2856, 6.2637), (11.7023, 1.6725, 6.6702)): 0.8940773944440167,
        ((11.15, 8.7738, 12.9835), (11.929, 7.3898, 12.497)): 0.9177073177612417,
        ((11.2412, 6.6039, 11.6536), (11.5417, 5.7711, 11.3077)): 0.8883623765678174,
        ((11.2412, 6.6039, 11.6536), (11.929, 7.3898, 12.497)): 0.8831610198316487,
        ((11.6166, 0.2382, 6.6115), (11.7023, 1.6725, 6.6702)): 0.9460733337236251,
        ((11.6166, 0.2382, 6.6115), (12.8896, -0.4289, 7.4764)): 0.9267125225882408,
        ((11.7023, 1.6725, 6.6702), (12.8903, 2.0459, 7.4516)): 0.9671726168239857,
        ((11.929, 7.3898, 12.497), (12.8035, 7.169, 12.7961)): 0.888051765188514,
        ((12.8896, -0.4289, 7.4764), (13.5449, 0.9788, 7.8926)): 0.8881610470215273,
        ((12.8903, 2.0459, 7.4516), (13.1582, 2.9416, 7.6211)): 0.8880265403414137,
        ((12.8903, 2.0459, 7.4516), (13.5449, 0.9788, 7.8926)): 0.8732018681964763,
        ((13.5449, 0.9788, 7.8926), (14.3371, 1.0415, 8.4122)): 0.8873810580165273,
    }
    return elements, edges, lengths


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

    def _assert_pepqag_scheme(self, elements, edges, lengths):
        """The PEPQAG scheme: a torus Se6 ring and an ylidene Se6 ring."""
        orders = chemistry.bond_orders(elements, edges, lengths)
        orders = {key: orders.get(key, 1) for key in lengths}
        torus_se = (-2.7381, 12.9367, 1.2676)
        ylidene_se = (9.6771, 11.7199, 5.395)

        # Torus ring: the SE takes its genuine ring double (0.9617), the ring
        # pairs its own C=C (0.8885, 0.9425), and the biaryl link towards the
        # dihydro-thiophene (0.9544) stays single -- the relaxation round must
        # not steal a ring double onto it.
        self.assertEqual(orders.get(chemistry.edge_key(torus_se, (-2.3563, 14.6851, 1.8591))), 2)
        self.assertEqual(orders.get(chemistry.edge_key(torus_se, (-1.0062, 12.2011, 1.4027))), 1)
        self.assertEqual(
            orders.get(chemistry.edge_key((-1.0062, 12.2011, 1.4027), (0.0215, 12.8662, 1.9731))), 2
        )
        self.assertEqual(
            orders.get(chemistry.edge_key((-1.1512, 15.0271, 2.3896), (0.0078, 14.1939, 2.5119))), 2
        )
        self.assertEqual(
            orders.get(chemistry.edge_key((-1.0062, 12.2011, 1.4027), (-0.8749, 10.8597, 0.8663))),
            1,
        )

        # Ylidene ring: the SE stays three-bonded (both ring C-Se bonds are
        # single), the ring pairs its two C=C (0.8972, 0.8937) and spends the
        # genuine ylidene double exocyclically (0.9308); the biaryl links
        # (0.9550, 0.9567) stay single.
        self.assertEqual(orders.get(chemistry.edge_key(ylidene_se, (9.203, 13.4547, 5.9522))), 1)
        self.assertEqual(orders.get(chemistry.edge_key(ylidene_se, (7.9916, 11.1383, 4.7835))), 1)
        self.assertEqual(
            orders.get(chemistry.edge_key((6.923, 11.9806, 4.6904), (7.9916, 11.1383, 4.7835))), 2
        )
        self.assertEqual(
            orders.get(chemistry.edge_key((7.9721, 13.9836, 5.7274), (9.203, 13.4547, 5.9522))), 2
        )
        self.assertEqual(
            orders.get(chemistry.edge_key((5.6836, 14.1072, 4.9499), (6.8628, 13.3402, 5.1))), 2
        )
        self.assertEqual(
            orders.get(chemistry.edge_key((7.92, 9.7422, 4.392), (7.9916, 11.1383, 4.7835))), 1
        )
        self.assertEqual(
            orders.get(chemistry.edge_key((9.203, 13.4547, 5.9522), (10.2608, 14.2037, 6.6115))), 1
        )

        # Perception: the torus ring is the Se six-ring with a sextet, drawn
        # Kekule (genuine C=Se, no lone-pair share of the sextet); the ylidene
        # ring is not aromatic at all (its exocyclic C=C voids the ring), and
        # the dihydro-thiophene joins it outside the aromatic set, so exactly
        # four aromatic rings remain (torus + three thiophenes).
        rings = chemistry.aromatic_rings(elements, edges, orders, detail=True)
        torus = [ring for ring in rings if torus_se in ring.nodes]
        self.assertEqual(len(torus), 1)
        self.assertEqual((torus[0].pi_electrons, torus[0].lone_pair_electrons), (6, 0))
        self.assertTrue(torus[0].kekule)
        self.assertEqual([ring for ring in rings if ylidene_se in ring.nodes], [])
        self.assertEqual(len(rings), 4)

    def test_pepqag_torus_and_ylidene_rings_survive_the_relaxation_round(self):
        # PEPQAG (CSD): the asymmetric unit fills one molecule whose two Se6
        # rings race in the biaryl-link relaxation round -- the winning
        # augmenting path is decided by the absolute node order, so the same
        # structure can draw differently in the ASU window and in the cell
        # window.  The first solve (every biaryl link held single) is the
        # chemically honest one: it leaves the torus ring its genuine C=Se and
        # the ylidene ring its exocyclic double.  The round-0 Kekule freeze
        # keeps the relaxation rounds from re-placing it.
        self._assert_pepqag_scheme(*_pepqag_asu())

    def test_pepqag_se_ring_scheme_is_independent_of_node_order(self):
        # The race is nothing but the graph's insertion order (the cell window
        # re-orders each molecule's nodes).  Re-insert every node, edge and
        # length in reverse and the scheme must come out identical.
        elements, edges, lengths = _pepqag_asu()
        self._assert_pepqag_scheme(
            {node: symbol for node, symbol in reversed(list(elements.items()))},
            list(reversed(edges)),
            {key: lengths[key] for key in reversed(list(lengths))},
        )

    def test_pepqag_cell_window_keeps_the_ylidene_ring(self):
        # The whole-graph CELL window re-solves both molecules at once, and its
        # node order is the one place the ASU copy of the same molecule cannot
        # predict the Kekule race: without the widened settled-chalcogen freeze
        # the re-solve satisfies the ylidene SE's unspent chalc order by
        # inventing a ring C=Se (0.9606 short enough to be a double) and the
        # molecule draws two torus rings instead of one torus and one ylidene.
        # The widened freeze treats the ring as settled while only the chalc
        # itself holds the deficit, so both ring C-Se stay single, the two
        # in-ring C=C and the genuine 0.9308 exocyclic ylidene double survive,
        # and the torus copy keeps its C=Se -- the same scheme the ASU window
        # draws (edge-level assertions only: the ring classification itself is
        # hash-order sensitive, the bond orders are not).
        elements, edges, lengths = _pepqag_cell()
        orders = chemistry.bond_orders(elements, edges, lengths)
        orders = {key: orders.get(key, 1) for key in lengths}

        ylidene_se = (11.033, -2.2456, 5.395)
        torus_se = (-1.3823, -1.0289, 1.2676)

        # Ylidene ring: the SE keeps both ring bonds single (0.9606, 0.9617) --
        # no invented C=Se -- and the ring pairs its two C=C (0.8937, 0.8972).
        self.assertEqual(orders.get(chemistry.edge_key(ylidene_se, (10.5588, -0.5108, 5.9522))), 1)
        self.assertEqual(orders.get(chemistry.edge_key(ylidene_se, (9.3474, -2.8273, 4.7835))), 1)
        self.assertEqual(
            orders.get(chemistry.edge_key((9.328, 0.0181, 5.7274), (10.5588, -0.5108, 5.9522))), 2
        )
        self.assertEqual(
            orders.get(chemistry.edge_key((8.2789, -1.985, 4.6904), (9.3474, -2.8273, 4.7835))), 2
        )
        # The genuine 0.9308 exocyclic ylidene double survives the re-solve and
        # the biaryl links (0.9550, 0.9567) stay single.
        self.assertEqual(
            orders.get(chemistry.edge_key((7.0394, 0.1417, 4.9499), (8.2186, -0.6254, 5.1))), 2
        )
        self.assertEqual(
            orders.get(chemistry.edge_key((9.2758, -4.2233, 4.392), (9.3474, -2.8273, 4.7835))), 1
        )
        self.assertEqual(
            orders.get(chemistry.edge_key((10.5588, -0.5108, 5.9522), (11.6166, 0.2382, 6.6115))), 1
        )
        # Torus copy: the SE keeps its genuine ring C=Se (0.9617) and its other
        # ring bond single, and the ring pairs its own C=C (0.8885, 0.9425),
        # exactly like the ASU torus ring.
        self.assertEqual(orders.get(chemistry.edge_key(torus_se, (-1.0005, 0.7196, 1.8591))), 2)
        self.assertEqual(orders.get(chemistry.edge_key(torus_se, (0.3496, -1.7644, 1.4027))), 1)
        self.assertEqual(
            orders.get(chemistry.edge_key((0.3496, -1.7644, 1.4027), (1.3774, -1.0994, 1.9731))),
            2,
        )
        self.assertEqual(
            orders.get(chemistry.edge_key((0.2046, 1.0615, 2.3896), (1.3637, 0.2284, 2.5119))),
            2,
        )

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


class PhosphoniumValenceTest(unittest.TestCase):
    """The rung-5 cap on a four-connected group-15 centre.

    A 4-connected P/As/Sb centre either is P(=X)R3 (drawn double through the
    5th rung), a genuine ylide P=C, or a saturated cation/complex.  The (3, 5)
    ladder cannot tell them apart, so the geometry does: a metal neighbour
    means a phosphine/arsine complex (Ar3P.M) whose every P-C the CSD types
    Single even when short (ADAKIB's 0.965 gold-bound ylide), and an
    all-carbon substituent set whose bonds are all too long for a P=C double
    is a true phosphonium (Ph4P+) that stays four-valent.  Only a short P-C
    (<= _P_YLIDE_MAX_LENGTH_SCALE) keeps the ylide's 5th rung.
    """

    @staticmethod
    def _methyl(centre, tag, elements, bonds):
        """A 4-connected methyl stub on *centre* that wants no extra order."""
        carbon = f"C{tag}"
        elements[carbon] = "C"
        # Normalised 0.98: safely inside the candidate gate, safely above the
        # 0.97 ylide threshold, so a P-C stub never looks like a double bond.
        bonds.append((centre, carbon, 0.98 * (RADII[elements[centre]] + RADII["C"])))
        for suffix in ("a", "b", "c"):
            hydrogen = f"{carbon}{suffix}"
            elements[hydrogen] = "H"
            bonds.append((carbon, hydrogen, 1.09))
        return carbon

    def test_true_phosphonium_draws_all_p_c_single(self):
        # Ph4P+: no metal, no H adjacent, and every P-C too long to be a
        # ylide, so the rung-5 order is withheld and all four bonds are single.
        atoms = {"P1": "P"}
        bonds = []
        for index in range(4):
            self._methyl("P1", str(index), atoms, bonds)
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders, {})

    def test_ylide_keeps_its_p_c_double(self):
        # A genuine phosphonium ylide R3P=CH2: its short P-C (0.93 of the
        # radii sum) clears the ylide gate and keeps the 5th rung, so the
        # P=C double is drawn.
        atoms = {"P1": "P", "CY": "C", "HYa": "H", "HYb": "H"}
        bonds = [("P1", "CY", 1.70), ("CY", "HYa", 1.08), ("CY", "HYb", 1.08)]
        for index in range(3):
            self._methyl("P1", str(index), atoms, bonds)
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders, {chemistry.edge_key("P1", "CY"): 2})

    def test_metal_bound_ylide_draws_all_single(self):
        # ADAKIB: a 0.965 gold-bound ylide still draws P-C single -- the
        # dative metal bond consumes the rung-5 order, so even a short P-C is
        # kept single by the CSD.
        atoms = {"P1": "P", "FE0": "FE", "CY": "C", "HYa": "H", "HYb": "H"}
        bonds = [
            ("P1", "FE0", 2.30),
            ("P1", "CY", 1.70),
            ("CY", "HYa", 1.08),
            ("CY", "HYb", 1.08),
        ]
        for index in range(2):
            self._methyl("P1", str(index), atoms, bonds)
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertNotIn(chemistry.edge_key("P1", "CY"), orders)


class MetalPnictogenChalcogenTest(unittest.TestCase):
    """P=X doubles whose chalcogen/N partner also coordinates a metal.

    The CSD types a phosphine sulfide, selenide or phosphinimine P=S/P=Se/P=N
    Double even when the chalcogen or nitrogen also bonds a cation: the S/N
    then has graph degree two or three, the valence ladder is full, and the
    double must be granted from the geometry instead (LEVQUZ's cerium
    thiophosphoryl complex, AVEYAE/BEFNUW's metal-bound phosphinimines).  The
    double sits on the phosphorus side, so the promotion is keyed on the
    partner's *sole* ladder neighbour being a group-15 pnictogen:

    * a two-coordinate chalcogen (one pnictogen + one metal) is promoted;
    * a three-coordinate phosphinimine nitrogen (pnictogen + carbon + metal)
      is promoted, but only while the pnictogen itself is a plain R3P= centre
      -- a P that already coordinates a metal (UJITAI's Zr-bound phosphine)
      keeps its P-N single, exactly as the CSD types it;
    * a bridge with two ladder neighbours (a thiolate R-S-P-M) is never
      promoted; its P-S bond stays single.
    """

    def _methyl(self, centre, tag, elements, bonds):
        return PhosphoniumValenceTest._methyl(centre, tag, elements, bonds)

    def test_metal_coordinated_phosphine_sulfide_keeps_its_double(self):
        # LEVQUZ: every P=S sulfur also coordinates the cerium cation.  The S
        # has graph degree two (P + metal) so its (2, 4, 6) ladder is full --
        # the P=S double only survives because the sulfur's sole ladder
        # neighbour is the phosphorus itself.
        atoms = {"P1": "P", "S1": "S", "FE1": "FE"}
        bonds = [("P1", "S1", 1.99), ("S1", "FE1", 2.50)]
        for index in range(3):
            self._methyl("P1", str(index), atoms, bonds)
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders, {chemistry.edge_key("P1", "S1"): 2})

    def test_three_connected_phosphinimine_nitrogen_keeps_its_double(self):
        # BEFNUW: an iminophosphorane R3P=N-R'-M whose degree-three nitrogen
        # (P + C + metal) bridges the metal still draws P=N: the nitrogen is
        # promoted because its pnictogen partner is a plain R3P= centre with
        # no metal of its own.
        atoms = {"P1": "P", "N1": "N", "FE1": "FE"}
        bonds = [("P1", "N1", 1.61), ("N1", "FE1", 2.10)]
        for index in range(3):
            self._methyl("P1", str(index), atoms, bonds)
        self._methyl("N1", "3", atoms, bonds)
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders, {chemistry.edge_key("P1", "N1"): 2})

    def test_metal_phosphine_keeps_its_p_n_bond_single(self):
        # UJITAI: the phosphorus is itself coordinated to a metal (Zr here
        # replaced by FE), so the rung-5 cap zeroes its deficit and its P=N
        # bond stays single even though the nitrogen also bridges a metal.
        atoms = {"P1": "P", "N1": "N", "FE1": "FE", "FE2": "FE"}
        bonds = [
            ("P1", "FE1", 2.50),
            ("P1", "N1", 1.61),
            ("N1", "FE2", 2.10),
        ]
        for index in range(2):
            self._methyl("P1", str(index), atoms, bonds)
        self._methyl("N1", "2", atoms, bonds)
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertNotIn(chemistry.edge_key("P1", "N1"), orders)
        self.assertNotIn(chemistry.edge_key("N1", "C2"), orders)

    def test_thiolate_phosphine_sulfur_stays_single(self):
        # A P-S-C thiolate/thiophosphate ester: the sulfur's ladder holds the
        # phosphorus *and* the carbon, so it is not a phosphine sulfide.  R1
        # only promotes a partner whose sole ladder neighbour is a pnictogen,
        # and a two-connected S gets no deficit from the (2, 4, 6) ladder
        # either -- the P-S bond stays single (the ESOFEY/FOSBAS CSD class).
        atoms = {"P1": "P", "S1": "S"}
        bonds = [("P1", "S1", 2.03)]
        for index in range(3):
            self._methyl("P1", str(index), atoms, bonds)
        self._methyl("S1", "3", atoms, bonds)
        elements, edges, lengths = molecule(atoms, bonds)

        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders, {})


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


class DelocalisedOxoEdgesTests(unittest.TestCase):
    """Terminal oxo groups drawn as a ChemDraw 1.5 (solid+dashed) bond."""

    def marked(self, atoms, bonds):
        elements, edges, lengths = molecule(atoms, bonds)
        return chemistry.delocalised_oxo_edges(elements, edges, lengths)

    def test_carboxylate_marks_both_carbon_oxygen_bonds(self):
        atoms = {"C0": "C", "C1": "C", "O1": "O", "O2": "O"}
        bonds = [("C0", "C1", 1.50), ("C1", "O1", 1.25), ("C1", "O2", 1.26)]
        self.assertEqual(
            self.marked(atoms, bonds),
            frozenset({chemistry.edge_key("C1", "O1"), chemistry.edge_key("C1", "O2")}),
        )

    def test_carboxylate_with_both_oxygens_on_metals_is_delocalised(self):
        atoms = {"C0": "C", "C1": "C", "O1": "O", "O2": "O", "NA1": "NA", "NA2": "NA"}
        bonds = [
            ("C0", "C1", 1.50),
            ("C1", "O1", 1.25),
            ("C1", "O2", 1.26),
            ("O1", "NA1", 2.35),
            ("O2", "NA2", 2.35),
        ]
        self.assertEqual(len(self.marked(atoms, bonds)), 2)

    def test_carbonate_marks_all_three_carbon_oxygen_bonds(self):
        atoms = {"C0": "C", "O1": "O", "O2": "O", "O3": "O"}
        bonds = [("C0", "O1", 1.28), ("C0", "O2", 1.29), ("C0", "O3", 1.28)]
        self.assertEqual(len(self.marked(atoms, bonds)), 3)

    def test_nitrate_marks_all_three_nitrogen_oxygen_bonds(self):
        atoms = {"N0": "N", "O1": "O", "O2": "O", "O3": "O"}
        bonds = [("N0", "O1", 1.24), ("N0", "O2", 1.25), ("N0", "O3", 1.24)]
        self.assertEqual(len(self.marked(atoms, bonds)), 3)

    def test_phosphate_marks_all_four_phosphorus_oxygen_bonds(self):
        atoms = {"P0": "P", "O1": "O", "O2": "O", "O3": "O", "O4": "O"}
        bonds = [
            ("P0", "O1", 1.52),
            ("P0", "O2", 1.53),
            ("P0", "O3", 1.51),
            ("P0", "O4", 1.52),
        ]
        self.assertEqual(len(self.marked(atoms, bonds)), 4)

    def test_phosphonate_marks_three_of_four_bonds(self):
        atoms = {"C0": "C", "P0": "P", "O1": "O", "O2": "O", "O3": "O"}
        bonds = [
            ("C0", "P0", 1.82),
            ("P0", "O1", 1.51),
            ("P0", "O2", 1.52),
            ("P0", "O3", 1.53),
        ]
        self.assertEqual(len(self.marked(atoms, bonds)), 3)

    def test_sulfate_marks_all_four_sulfur_oxygen_bonds(self):
        atoms = {"S0": "S", "O1": "O", "O2": "O", "O3": "O", "O4": "O"}
        bonds = [
            ("S0", "O1", 1.47),
            ("S0", "O2", 1.48),
            ("S0", "O3", 1.46),
            ("S0", "O4", 1.47),
        ]
        self.assertEqual(len(self.marked(atoms, bonds)), 4)

    def test_sulfonate_marks_three_of_four_bonds(self):
        atoms = {"C0": "C", "S0": "S", "O1": "O", "O2": "O", "O3": "O"}
        bonds = [
            ("C0", "S0", 1.78),
            ("S0", "O1", 1.44),
            ("S0", "O2", 1.45),
            ("S0", "O3", 1.46),
        ]
        self.assertEqual(len(self.marked(atoms, bonds)), 3)

    def test_sulfonate_delocalises_only_the_metal_coordinated_oxygen(self):
        # A monodentate metal sulfonate salt: the coordinating oxygen is the
        # anionic member of the resonance set and is drawn delocalised, while
        # the two metal-free oxygens keep their genuine S=O double-bonds.
        atoms = {"C0": "C", "S0": "S", "O1": "O", "O2": "O", "O3": "O", "NA1": "NA"}
        bonds = [
            ("C0", "S0", 1.78),
            ("S0", "O1", 1.44),
            ("S0", "O2", 1.45),
            ("S0", "O3", 1.46),
            ("O1", "NA1", 2.35),
        ]
        self.assertEqual(self.marked(atoms, bonds), frozenset({chemistry.edge_key("S0", "O1")}))

    def test_bidentate_metal_triflate_keeps_the_free_oxygen_genuine(self):
        # test2.cif (ABOLIP): the Ti(OTf)2 triflate coordinates two sulfonate
        # oxygens (Ti-O ~2.27 A, graph degree two) and leaves the shortest
        # third S-O free.  Only the two coordinating oxygens are delocalised,
        # so the drawing keeps one genuine S=O double and the sulfur reads
        # hexavalent: 2 + 1.5 + 1.5 + 1 = 6 instead of 3 x 1.5 + 1 = 5.5.
        atoms = {
            "S0": "S",
            "O1": "O",
            "O2": "O",
            "O3": "O",
            "C1": "C",
            "F1": "F",
            "F2": "F",
            "F3": "F",
            "TI1": "TI",
        }
        bonds = [
            ("S0", "O3", 1.4256),
            ("S0", "O1", 1.4670),
            ("S0", "O2", 1.4679),
            ("S0", "C1", 1.8384),
            ("C1", "F1", 1.33),
            ("C1", "F2", 1.33),
            ("C1", "F3", 1.33),
            ("O1", "TI1", 2.27),
            ("O2", "TI1", 2.27),
        ]
        elements, edges, lengths = molecule(atoms, bonds)
        marked = chemistry.delocalised_oxo_edges(elements, edges, lengths)
        self.assertEqual(
            marked, frozenset({chemistry.edge_key("S0", "O1"), chemistry.edge_key("S0", "O2")})
        )
        # The drawing keeps one genuine double: S=O3 is an order-2 bond_orders
        # edge and is NOT overridden by the overlay, so the toggle-ON depiction
        # reads 2 + 1.5 + 1.5 + 1 = 6 (hexavalent) instead of 3 x 1.5 + 1.
        orders = chemistry.bond_orders(elements, edges, lengths)
        self.assertEqual(orders.get(chemistry.edge_key("S0", "O3")), 2)
        self.assertNotIn(chemistry.edge_key("S0", "O3"), marked)
        self.assertEqual(valences(elements, edges, orders)["S0"], 6)

    def test_selenate_and_tellurate_are_delocalised(self):
        for element, distance in (("SE", 1.64), ("TE", 1.80)):
            with self.subTest(element):
                atoms = {"X0": element, "O1": "O", "O2": "O", "O3": "O", "O4": "O"}
                bonds = [("X0", f"O{index}", distance) for index in range(1, 5)]
                self.assertEqual(len(self.marked(atoms, bonds)), 4)

    def test_nitro_keeps_its_genuine_doubles(self):
        atoms = {"C0": "C", "N0": "N", "O1": "O", "O2": "O"}
        bonds = [("C0", "N0", 1.47), ("N0", "O1", 1.22), ("N0", "O2", 1.24)]
        self.assertEqual(self.marked(atoms, bonds), frozenset())

    def test_sulfone_keeps_its_genuine_doubles(self):
        atoms = {"C0": "C", "C1": "C", "S0": "S", "O1": "O", "O2": "O"}
        bonds = [
            ("C0", "S0", 1.78),
            ("C1", "S0", 1.78),
            ("S0", "O1", 1.43),
            ("S0", "O2", 1.44),
        ]
        self.assertEqual(self.marked(atoms, bonds), frozenset())

    def test_ketone_aldehyde_ester_and_amide_are_not_delocalised(self):
        cases = {
            "ketone": (
                {"C0": "C", "C1": "C", "C2": "C", "O1": "O"},
                [("C0", "C1", 1.51), ("C1", "C2", 1.51), ("C1", "O1", 1.22)],
            ),
            "aldehyde": (
                {"C0": "C", "C1": "C", "H1": "H", "O1": "O"},
                [("C0", "C1", 1.51), ("C1", "H1", 1.09), ("C1", "O1", 1.21)],
            ),
            "ester": (
                {"C0": "C", "C1": "C", "O1": "O", "O2": "O", "C2": "C"},
                [
                    ("C0", "C1", 1.51),
                    ("C1", "O1", 1.21),
                    ("C1", "O2", 1.34),
                    ("O2", "C2", 1.44),
                ],
            ),
            "amide": (
                {"C0": "C", "C1": "C", "N1": "N", "O1": "O", "H1": "H"},
                [
                    ("C0", "C1", 1.51),
                    ("C1", "O1", 1.23),
                    ("C1", "N1", 1.33),
                    ("N1", "H1", 1.01),
                ],
            ),
        }
        for name, (atoms, bonds) in cases.items():
            with self.subTest(name):
                self.assertEqual(self.marked(atoms, bonds), frozenset(), name)

    def test_carboxylic_acid_with_its_hydrogen_is_not_delocalised(self):
        atoms = {"C0": "C", "C1": "C", "O1": "O", "O2": "O", "H1": "H"}
        bonds = [
            ("C0", "C1", 1.51),
            ("C1", "O1", 1.21),
            ("C1", "O2", 1.31),
            ("O2", "H1", 0.98),
        ]
        self.assertEqual(self.marked(atoms, bonds), frozenset())

    def test_nitric_acid_with_its_hydrogen_is_not_delocalised(self):
        atoms = {"N0": "N", "O1": "O", "O2": "O", "O3": "O", "H1": "H"}
        bonds = [
            ("N0", "O1", 1.21),
            ("N0", "O2", 1.22),
            ("N0", "O3", 1.40),
            ("O3", "H1", 0.98),
        ]
        self.assertEqual(self.marked(atoms, bonds), frozenset())

    def test_phenoxide_is_not_delocalised(self):
        nodes = [f"X{index}" for index in range(6)]
        atoms = {node: "C" for node in nodes}
        atoms["O1"] = "O"
        bonds = ring_bonds(nodes) + [("X0", "O1", 1.36)]
        self.assertEqual(self.marked(atoms, bonds), frozenset())

    def test_n_oxide_is_not_delocalised(self):
        atoms = {"N0": "N", "C0": "C", "C1": "C", "O1": "O"}
        bonds = [("N0", "C0", 1.40), ("N0", "C1", 1.40), ("N0", "O1", 1.31)]
        self.assertEqual(self.marked(atoms, bonds), frozenset())

    def test_metal_carbonyl_carbon_is_skipped(self):
        atoms = {"FE0": "FE", "C0": "C", "O1": "O", "O2": "O", "O3": "O"}
        bonds = [
            ("FE0", "C0", 1.80),
            ("C0", "O1", 1.20),
            ("C0", "O2", 1.21),
            ("C0", "O3", 1.22),
        ]
        self.assertEqual(self.marked(atoms, bonds), frozenset())

    def test_result_does_not_depend_on_edge_order(self):
        atoms = {"C0": "C", "C1": "C", "O1": "O", "O2": "O"}
        bonds = [("C0", "C1", 1.50), ("C1", "O1", 1.25), ("C1", "O2", 1.26)]
        elements, edges, lengths = molecule(atoms, bonds)
        forward = chemistry.delocalised_oxo_edges(elements, edges, lengths)
        backward = chemistry.delocalised_oxo_edges(elements, list(reversed(edges)), lengths)
        self.assertEqual(forward, backward)


if __name__ == "__main__":
    # Blender keeps its own CLI flags in ``sys.argv``, so ``unittest.main()``
    # would try to parse ``--background``.  Drive the runner explicitly and
    # print the RESULT line ``tests/run_all.ps1`` looks for.
    SUITE = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    RESULT = unittest.TextTestRunner(verbosity=2).run(SUITE)
    print("RESULT: ALL PASS" if RESULT.wasSuccessful() else "RESULT: FAIL")
    sys.exit(0 if RESULT.wasSuccessful() else 1)
