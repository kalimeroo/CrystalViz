"""Smoke tests for gemmi_worker (bpy-free, runnable standalone).

Run with a Python that has gemmi installed:
    python test_gemmi_worker.py
"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (puts a gemmi build on sys.path)
import _symmetry_disorder_fixture as _fixture

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


def _disorder_cif(rows, with_occupancy):
    """Minimal P1 CIF from ``(label, element, x, y, z, occupancy)`` rows."""
    lines = [
        "data_disorder",
        "_cell_length_a 10.0",
        "_cell_length_b 10.0",
        "_cell_length_c 10.0",
        "_cell_angle_alpha 90.0",
        "_cell_angle_beta 90.0",
        "_cell_angle_gamma 90.0",
        "_symmetry_space_group_name_H-M 'P 1'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    if with_occupancy:
        lines.append("_atom_site_occupancy")
    for label, element, x, y, z, occupancy in rows:
        values = [label, element, f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"]
        if with_occupancy:
            values.append(f"{occupancy:.3f}")
        lines.append(" ".join(values))
    return "\n".join(lines) + "\n"


class TestDisorderPrimeLabels(unittest.TestCase):
    """Alternate disorder components are omitted at load.

    CSD CIFs mark a disordered site's second/third component with a trailing
    prime/double prime (S1', S1'').  The AKUQAB 5-ring flips between two
    orientations; keeping both gives the shared exocyclic carbon an extra
    neighbour and the ring was never enumerated.  Occupancy decides when it is
    declared (the more occupied component survives); without it the primed
    components go.
    """

    def _load(self, rows, with_occupancy=True):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "disorder.cif")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(_disorder_cif(rows, with_occupancy))
            return gw.load_structure(path)

    @staticmethod
    def _labels(result):
        return [site["label"] for site in result["sites"]]

    def test_primes_are_dropped_without_occupancy(self):
        rows = [
            ("S1", "S", 0.20, 0.30, 0.40, 1.0),
            ("C1", "C", 0.25, 0.30, 0.40, 1.0),
            ("S1'", "S", 0.20, 0.30, 0.42, 1.0),
            ("C1'", "C", 0.25, 0.30, 0.40, 1.0),
        ]
        result = self._load(rows, with_occupancy=False)
        self.assertEqual(self._labels(result), ["S1", "C1"])
        self.assertEqual(result["omitted_disorder"], 2)

    def test_instances_reference_only_kept_sites(self):
        rows = [
            ("S1", "S", 0.20, 0.30, 0.40, 1.0),
            ("C1", "C", 0.25, 0.30, 0.40, 1.0),
            ("S1'", "S", 0.20, 0.30, 0.42, 1.0),
            ("C1'", "C", 0.25, 0.30, 0.42, 1.0),
        ]
        result = self._load(rows, with_occupancy=False)
        self.assertEqual(len(result["sites"]), 2)
        used = {
            int(result["instances"][offset]) for offset in range(0, len(result["instances"]), 4)
        }
        self.assertTrue(used <= set(range(len(result["sites"]))), str(used))

    def test_occupancy_lets_a_primed_component_win(self):
        rows = [
            ("S1", "S", 0.20, 0.30, 0.40, 0.40),
            ("C1", "C", 0.25, 0.30, 0.40, 0.40),
            ("S1'", "S", 0.20, 0.30, 0.42, 0.60),
            ("C1'", "C", 0.25, 0.30, 0.42, 0.60),
        ]
        result = self._load(rows)
        self.assertEqual(self._labels(result), ["S1'", "C1'"])
        self.assertEqual(result["omitted_disorder"], 2)

    def test_equal_occupancy_keeps_the_unprimed_component(self):
        rows = [
            ("S1", "S", 0.20, 0.30, 0.40, 0.50),
            ("C1", "C", 0.25, 0.30, 0.40, 0.50),
            ("S1'", "S", 0.20, 0.30, 0.42, 0.50),
            ("C1'", "C", 0.25, 0.30, 0.42, 0.50),
        ]
        result = self._load(rows)
        self.assertEqual(self._labels(result), ["S1", "C1"])

    def test_fixed_hydrogen_occupancy_does_not_outvote_the_heavy_atoms(self):
        # Hydrogens of a partially occupied component are often left at 1.0:
        # that base abstains, so the heavy-atom component still decides.
        rows = [
            ("S1", "S", 0.20, 0.30, 0.40, 0.60),
            ("H1", "H", 0.22, 0.30, 0.40, 1.00),
            ("S1'", "S", 0.20, 0.30, 0.42, 0.40),
            ("H1'", "H", 0.22, 0.30, 0.42, 1.00),
        ]
        result = self._load(rows)
        self.assertEqual(self._labels(result), ["S1", "H1"])

    def test_lone_primed_label_is_kept(self):
        rows = [
            ("S1", "S", 0.20, 0.30, 0.40, 1.0),
            ("C1'", "C", 0.60, 0.30, 0.40, 1.0),
        ]
        result = self._load(rows)
        self.assertEqual(self._labels(result), ["S1", "C1'"])
        self.assertEqual(result["omitted_disorder"], 0)

    def test_double_prime_third_component_is_dropped(self):
        rows = [
            ("S1", "S", 0.20, 0.30, 0.40, 0.50),
            ("S1'", "S", 0.20, 0.30, 0.42, 0.30),
            ("S1''", "S", 0.20, 0.30, 0.44, 0.20),
        ]
        result = self._load(rows)
        self.assertEqual(self._labels(result), ["S1"])
        self.assertEqual(result["omitted_disorder"], 2)

    def test_letter_suffixes_are_not_disorder(self):
        # H8A/H8B are the two hydrogens of one CH2, not alternate components.
        rows = [
            ("C8", "C", 0.20, 0.30, 0.40, 1.0),
            ("H8A", "H", 0.22, 0.30, 0.40, 1.0),
            ("H8B", "H", 0.18, 0.32, 0.40, 1.0),
        ]
        result = self._load(rows)
        self.assertEqual(self._labels(result), ["C8", "H8A", "H8B"])
        self.assertEqual(result["omitted_disorder"], 0)


class TestSymmetryDisorder(unittest.TestCase):
    """A substituent whose alternate position the space group generates.

    ``Ca.cif`` (local-only) is the driving case: a *tert*-butyl group and a
    methyl sit on a mirror plane, their atoms are written once at half
    occupancy, and the P1 expansion therefore holds two orientations of both
    groups.  The loader keeps one chemically valid orientation per anchor and
    promotes it to full occupancy; half-occupied solvent on general positions is
    genuine substitutional disorder and stays untouched.
    """

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = _fixture.write_cif(os.path.join(cls._directory.name, "mirror_tbu.cif"))

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def _load(self, **kwargs):
        return gw.load_structure(self.path, **kwargs)

    @staticmethod
    def _distances(unit_cell, sites, indexes):
        points = [unit_cell.orthogonalize(sites[index]["frac"]) for index in indexes]
        return [
            math.dist(points[first], points[second])
            for first in range(len(points))
            for second in range(first + 1, len(points))
        ]

    @staticmethod
    def _around(unit_cell, sites, centre, predicate, radius):
        origin = unit_cell.orthogonalize(sites[centre]["frac"])
        out = []
        for index, site in enumerate(sites):
            if index == centre or not predicate(site):
                continue
            point = unit_cell.orthogonalize(site["frac"])
            if math.dist(origin, point) <= radius:
                out.append(index)
        return out

    def test_mirror_generated_tbutyl_resolves_to_one_tetrahedron(self):
        result = self._load()
        unit_cell = gw._UnitCell(result["unit_cell"])
        sites = result["sites"]
        self.assertEqual(result["resolved_disorder"], 1)
        self.assertEqual(result["dropped_disorder"], 24)
        self.assertEqual(len(sites), 56)
        apices = [index for index, site in enumerate(sites) if site["label"] == "C7"]
        self.assertEqual(len(apices), 2, "one t-Bu per asymmetric-unit copy")
        for apex in apices:
            methyls = self._around(
                unit_cell, sites, apex, lambda site: site["label"] in ("C8", "C9", "C10"), 1.9
            )
            self.assertEqual(len(methyls), 3, "the two orientations must not be kept together")
            for methyl in methyls:
                self.assertEqual(sites[methyl]["occupancy"], 1.0)
                base = sites[methyl]["label"]
                hydrogens = self._around(
                    unit_cell,
                    sites,
                    methyl,
                    lambda site, base=base: site["label"].startswith("H" + base[1:]),
                    1.4,
                )
                self.assertEqual(len(hydrogens), 3, sites[methyl]["label"])
            for distance in self._distances(unit_cell, sites, methyls):
                self.assertGreaterEqual(distance, 2.3, "methyl-methyl clash survived")
            for methyl in methyls:
                self.assertAlmostEqual(
                    math.dist(
                        unit_cell.orthogonalize(sites[apex]["frac"]),
                        unit_cell.orthogonalize(sites[methyl]["frac"]),
                    ),
                    1.52,
                    places=2,
                )

    def test_asymmetric_unit_moves_to_the_chosen_image(self):
        # The CIF deposits one methyl at its mirrored position, so the CIF's own
        # combination is not a tetrahedron; the ASU atom has to follow the
        # resolver's choice or it would no longer sit on any kept site.
        result = self._load()
        unit_cell = gw._UnitCell(result["unit_cell"])
        sites = result["sites"]
        instances = result["instances"]
        asu = []
        for offset in range(0, len(instances), 4):
            site = sites[instances[offset]]
            frac = [site["frac"][axis] + instances[offset + 1 + axis] for axis in range(3)]
            asu.append((site["label"], unit_cell.orthogonalize(frac)))
        apices = [point for label, point in asu if label == "C7"]
        self.assertEqual(len(apices), 1)
        methyls = [point for label, point in asu if label in ("C8", "C9", "C10")]
        self.assertEqual(len(methyls), 3)
        self.assertEqual(len({tuple(round(v, 4) for v in point) for point in methyls}), 3)
        for first in range(3):
            for second in range(first + 1, 3):
                self.assertGreaterEqual(math.dist(methyls[first], methyls[second]), 2.3)

    def test_substitutional_solvent_is_left_alone(self):
        result = self._load()
        sites = result["sites"]
        for label in ("K1", "O1"):
            images = [site for site in sites if site["label"] == label]
            self.assertEqual(len(images), 4, label)
            for site in images:
                self.assertEqual(site["occupancy"], 0.5, label)

    def test_keep_disorder_keeps_both_components(self):
        result = self._load(keep_disorder=True)
        unit_cell = gw._UnitCell(result["unit_cell"])
        sites = result["sites"]
        self.assertEqual(result["resolved_disorder"], 1)
        self.assertEqual(result["dropped_disorder"], 0)
        self.assertEqual(len(sites), 80)
        apices = [index for index, site in enumerate(sites) if site["label"] == "C7"]
        self.assertEqual(len(apices), 2)
        for apex in apices:
            methyls = self._around(
                unit_cell, sites, apex, lambda site: site["label"] in ("C8", "C9", "C10"), 1.9
            )
            self.assertEqual(len(methyls), 6)
            components = {}
            for index in methyls:
                components.setdefault(sites[index]["component"], []).append(index)
            self.assertEqual(len(components), 2, "each anchor holds two components")
            for members in components.values():
                self.assertGreaterEqual(len(members), 3)
                for distance in self._distances(unit_cell, sites, members):
                    self.assertGreaterEqual(
                        distance, 2.3, "a component must be a valid tetrahedron"
                    )
        for label in ("K1", "O1"):
            for site in sites:
                if site["label"] == label:
                    self.assertEqual(site["occupancy"], 0.5)
                    self.assertEqual(site.get("component", 0), 0)

    def test_general_position_anchor_is_not_resolved(self):
        # A sulfonate oxygen merely bonded to a disordered potassium/water
        # cluster sits on a general position: its two K images are not symmetry
        # mates around it, so the group must not be resolved (this is the false
        # positive that used to swallow the Ca.cif solvent region).
        rows = [
            ("S1", "S", 0.10, 0.40, 0.10, 1.0),
            ("O1", "O", 0.10, 0.45, 0.17, 1.0),
            ("O2", "O", 0.10, 0.36, 0.16, 1.0),
            ("K1", "K", 0.10, 0.38, 0.25, 0.5),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "general_position.cif")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(_p2m_cif(rows))
            result = gw.load_structure(path)
        self.assertEqual(result["resolved_disorder"], 0)
        self.assertEqual(result["dropped_disorder"], 0)
        self.assertEqual(len(result["sites"]), 16)
        self.assertEqual(len([site for site in result["sites"] if site["label"] == "K1"]), 4)


def _p2m_cif(rows, cell=(20.0, 14.0, 20.0)):
    """P 1 2/m 1 CIF from ``(label, element, x, y, z, occupancy)`` rows."""
    lines = [
        "data_symmetry_disorder",
        f"_cell_length_a {cell[0]}",
        f"_cell_length_b {cell[1]}",
        f"_cell_length_c {cell[2]}",
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
    for label, element, x, y, z, occupancy in rows:
        lines.append(f"{label} {element} {x:.6f} {y:.6f} {z:.6f} {occupancy:.2f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    # Blender execs --python scripts with its own argv, so feed unittest a clean
    # argument list when running inside Blender.
    argv = [sys.argv[0]] if "bpy" in sys.modules else sys.argv
    program = unittest.main(argv=argv, verbosity=2, exit=False)
    failed = not program.result.wasSuccessful()
    print("RESULT: FAIL" if failed else "RESULT: ALL PASS")
    if failed:
        sys.exit(1)
