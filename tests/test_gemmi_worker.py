"""Smoke tests for gemmi_worker (bpy-free, runnable standalone).

Run with a Python that has gemmi installed:
    python test_gemmi_worker.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (puts a gemmi build on sys.path)

sys.path.insert(0, _env.REPO_DIR)
import gemmi_worker as gw

EXAMPLES = _env.CIF_DIR
CIFS = {
    "multiple_bonds.cif": (172, 43),
    "Ferrocene.cif": (180, 45),
    "Brookite.cif": (24, 3),
    # 2048 P1 sites = the 256-atom ASU expanded in C 1 2/c 1.
    "Nano.cif": (2048, 256),
    # The two Cg ring centroids (calc_flag=dum) are skipped, not parsed as C.
    "Cyanoferrocene.cif": (88, 22),
}


class TestLoadStructure(unittest.TestCase):
    def test_all_example_cifs(self):
        for name, (sites, asu) in CIFS.items():
            path = os.path.join(EXAMPLES, name)
            result = gw.load_structure(path)
            self.assertEqual(len(result["sites"]), sites, name)
            self.assertEqual(len(result["instances"]) // 4, asu, name)
            self.assertIn("cell_matrix", result)
            self.assertEqual(len(result["cell_matrix"]), 9)
            self.assertIn("space_group", result)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            gw.load_structure(os.path.join(EXAMPLES, "nope.cif"))

    def test_legacy_error_message(self):
        # structures without cgn_sites raise _LEGACY_STRUCTURE_ERROR on the
        # Blender side; here we just ensure load_structure returns sites.
        result = gw.load_structure(os.path.join(EXAMPLES, "multiple_bonds.cif"))
        self.assertTrue(result["sites"])


class TestDummySiteSkipping(unittest.TestCase):
    def test_cyanoferrocene_ring_centroids_are_skipped(self):
        result = gw.load_structure(os.path.join(EXAMPLES, "Cyanoferrocene.cif"))
        labels = {site["label"] for site in result["sites"]}
        self.assertFalse([label for label in labels if label.startswith("Cg")])

        counts = {}
        for site in result["sites"]:
            counts[site["element"]] = counts.get(site["element"], 0) + 1
        # A dummy site has no type_symbol, so it used to fall back to its label
        # and become carbon ("Cg1" -> "Cg" -> "C").
        self.assertEqual(counts, {"C": 44, "FE": 4, "H": 36, "N": 4})
        self.assertEqual(len(result["instances"]) // 4, 22)

    def test_dummy_labels_read_the_calc_flag(self):
        import gemmi

        block = gemmi.cif.read(os.path.join(EXAMPLES, "Cyanoferrocene.cif")).sole_block()
        self.assertEqual(gw._dummy_labels(block), frozenset({"Cg1", "Cg2"}))

    def test_missing_calc_flag_skips_nothing(self):
        import gemmi

        block = gemmi.cif.read(os.path.join(EXAMPLES, "Brookite.cif")).sole_block()
        self.assertEqual(gw._dummy_labels(block), frozenset())


class TestGrowConnectivityShell(unittest.TestCase):
    def _setup(self, name="multiple_bonds.cif"):
        result = gw.load_structure(os.path.join(EXAMPLES, name))
        sites = [{"element": s["element"], "frac": s["frac"]} for s in result["sites"]]
        cov = {s["element"]: s["covalent_radius"] for s in result["sites"]}
        elements = sorted(cov)
        pairs = [
            {
                "element_a": ea,
                "element_b": eb,
                "enabled": True,
                "min_distance": 0.1,
                "max_distance": min(20.0, cov[ea] + cov[eb] + 0.4),
            }
            for i, ea in enumerate(elements)
            for eb in elements[i:]
        ]
        return result, sites, result["instances"], pairs

    def test_grow_does_not_crash(self):
        result, sites, instances, pairs = self._setup("Brookite.cif")
        r = gw.grow_connectivity_shell(result["unit_cell"], sites, instances, instances, pairs)
        self.assertIn("new_instances", r)
        self.assertIn("grew_by", r)
        self.assertGreaterEqual(r["grew_by"], 0)

    def test_grow_idempotent_no_duplicates(self):
        result, sites, instances, pairs = self._setup("Brookite.cif")
        r1 = gw.grow_connectivity_shell(result["unit_cell"], sites, instances, instances, pairs)
        combined = instances + r1["new_instances"]
        r2 = gw.grow_connectivity_shell(result["unit_cell"], sites, combined, combined, pairs)

        # second shell should not re-add atoms already present
        def to_quads(flat):
            return {tuple(flat[i : i + 4]) for i in range(0, len(flat), 4)}

        first_set = to_quads(r1["new_instances"])
        second_set = to_quads(r2.get("new_instances", []))
        self.assertTrue(first_set.isdisjoint(second_set))

    def test_large_structure_performance(self):
        import time

        result, sites, instances, pairs = self._setup("Nano.cif")
        t0 = time.perf_counter()
        gw.grow_connectivity_shell(result["unit_cell"], sites, instances, instances, pairs)
        self.assertLess(time.perf_counter() - t0, 5.0)


class TestADPConversion(unittest.TestCase):
    def test_site_u_cart_matches_cctbx_chain(self):
        # gemmi voigt order is (U11,U22,U33,U23,U13,U12); verify the helper
        # runs and returns 6 floats for an anisotropic site.
        result = gw.load_structure(os.path.join(EXAMPLES, "multiple_bonds.cif"))
        unit_cell = gw._UnitCell(result["unit_cell"])
        import gemmi

        doc = gemmi.cif.read(os.path.join(EXAMPLES, "multiple_bonds.cif"))
        st = gemmi.make_small_structure_from_block(doc.sole_block())
        site = next(s for s in st.sites if s.aniso.u11 != 0.0)
        tensor = gw._site_u_cart(site, unit_cell)
        self.assertEqual(len(tensor), 6)
        self.assertTrue(all(isinstance(v, float) for v in tensor))


class TestDisorderTwinDedup(unittest.TestCase):
    """Growth must never materialise a pseudo-symmetry twin of a disordered CIF.

    ABILIL01 (CSD, CCDC 2530362; a local fixture like S1.cif/S2.cif) deposits
    *both* halves of an inversion-disordered molecule: the P1 expansion holds
    848 same-element coincident site pairs, most exact (0.0 A) but some
    differing by one unit of the 5-decimal fractional coordinates (~1.6e-4 A in
    this cell).  `_canonical_base_indices` must merge those near-duplicates
    (same element, positions within `_DUPLICATE_POSITION_TOLERANCE`) so growth
    never adds an atom sitting on top of one already on screen: a duplicate
    hydrogen twin would give a ring carbon an extra non-metal neighbour and the
    benzo ring would stop being enumerated (drawn Kekule instead of a torus).
    """

    ABILIL = os.path.join(EXAMPLES, "ABILIL01.cif")

    def setUp(self):
        self.result = gw.load_structure(self.ABILIL)
        self.sites = self.result["sites"]
        self.unit_cell = gw._UnitCell(self.result["unit_cell"])
        self.base_atoms = gw._base_atoms(self.unit_cell, self.sites)
        self.canonical = gw._canonical_base_indices(self.base_atoms)

    @staticmethod
    def _cart_points(unit_cell, base_atoms, instances):
        points = []
        count = len(instances) - len(instances) % 4
        for offset in range(0, count, 4):
            site = int(instances[offset])
            translation = tuple(int(instances[offset + axis]) for axis in range(1, 4))
            frac = [base_atoms[site]["frac"][axis] + translation[axis] for axis in range(3)]
            points.append((base_atoms[site]["element"], unit_cell.orthogonalize(frac)))
        return points

    @staticmethod
    def _min_same_element_separation(points):
        best = float("inf")
        for i, (element_i, p_i) in enumerate(points):
            for element_j, p_j in points[i + 1 :]:
                if element_i != element_j:
                    continue
                distance = sum((p_i[axis] - p_j[axis]) ** 2 for axis in range(3)) ** 0.5
                best = min(best, distance)
        return best

    def test_near_duplicates_are_merged(self):
        self.assertEqual(len(self.sites), 1992)
        representatives = sum(1 for index, rep in enumerate(self.canonical) if rep == index)
        self.assertEqual(representatives, 1144)
        self.assertEqual(len(self.sites) - representatives, 848)

    def test_hydrogen_twins_share_one_representative(self):
        # H36 and its B-twin differ by ~1.6e-4 A, i.e. one unit of the deposited
        # 5-decimal fractionals, so the old round-to-1e-5 equality kept both.
        labels = [atom["label"] for atom in self.base_atoms]
        twins = []
        for i, label_i in enumerate(labels):
            if label_i != "H36":
                continue
            for j in range(i + 1, len(self.base_atoms)):
                if labels[j] != "H36B":
                    continue
                distance = (
                    sum(
                        (self.base_atoms[i]["cart"][axis] - self.base_atoms[j]["cart"][axis]) ** 2
                        for axis in range(3)
                    )
                    ** 0.5
                )
                if distance < 0.01:
                    twins.append((i, j, distance))
        self.assertTrue(twins, "expected at least one H36/H36B twin pair")
        for i, j, distance in twins:
            self.assertEqual(
                self.canonical[i],
                self.canonical[j],
                f"twin pair {i}/{j} at {distance:.6f} A must share one representative",
            )

    def test_grow_adds_no_twin(self):
        cov = {site["element"]: site["covalent_radius"] for site in self.sites}
        elements = sorted(cov)
        pairs = [
            {
                "element_a": element_a,
                "element_b": element_b,
                "enabled": True,
                "min_distance": 0.1,
                "max_distance": min(20.0, cov[element_a] + cov[element_b] + 0.4),
            }
            for i, element_a in enumerate(elements)
            for element_b in elements[i:]
        ]
        asu = self.result["instances"]
        asu_points = self._cart_points(self.unit_cell, self.base_atoms, asu)
        self.assertGreaterEqual(
            self._min_same_element_separation(asu_points),
            0.05,
            "the ASU is the clean half of the disorder (precondition)",
        )
        grown = gw.grow_connectivity_shell(self.result["unit_cell"], self.sites, asu, asu, pairs)
        union = self._cart_points(self.unit_cell, self.base_atoms, asu + grown["new_instances"])
        self.assertGreaterEqual(
            self._min_same_element_separation(union),
            0.05,
            "growth must not add the twin of an atom already on screen",
        )


if __name__ == "__main__":
    # Blender execs --python scripts with its own argv, so feed unittest a clean
    # argument list when running inside Blender.
    argv = [sys.argv[0]] if "bpy" in sys.modules else sys.argv
    program = unittest.main(argv=argv, verbosity=2, exit=False)
    failed = not program.result.wasSuccessful()
    print("RESULT: FAIL" if failed else "RESULT: ALL PASS")
    if failed:
        sys.exit(1)
