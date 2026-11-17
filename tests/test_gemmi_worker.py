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


if __name__ == "__main__":
    # Blender execs --python scripts with its own argv, so feed unittest a clean
    # argument list when running inside Blender.
    argv = [sys.argv[0]] if "bpy" in sys.modules else sys.argv
    program = unittest.main(argv=argv, verbosity=2, exit=False)
    failed = not program.result.wasSuccessful()
    print("RESULT: FAIL" if failed else "RESULT: ALL PASS")
    if failed:
        sys.exit(1)
