"""Growth must not walk into duplicate copies of a site.

Disordered CIFs can list the same element twice at one site; the P1 table then
holds two atoms a few thousandths of an Angstrom apart.  _canonical_base_indices
groups them so growth seeds and visited keys use the representative only.
Before the fix, the periodic neighbour table skipped non-canonical *seeds* but
still listed non-canonical *candidates*, so with min_distance = 0 growth added
the duplicate copy as a new atom sitting on top of the representative.

The duplicates here are 2e-6 A apart, i.e. above the 1e-12 self filter and
below the 1e-5 canonical grouping resolution, so only the canonical check can
remove them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (puts a gemmi build on sys.path)
sys.path.insert(0, _env.REPO_DIR)
import gemmi_worker as gw

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


PARAMETERS = [5.0, 5.0, 5.0, 90.0, 90.0, 90.0]
# Site 0/1 are C duplicates 2e-6 A apart; 2 is a genuine C-O neighbour of 0;
# 3/4 are O duplicates far from the seed.
SITES = [
    {"label": "C0", "element": "C", "frac": [0.3, 0.3, 0.3]},
    {"label": "C1", "element": "C", "frac": [0.3 + 4e-7, 0.3, 0.3]},
    {"label": "O2", "element": "O", "frac": [0.3, 0.3, 0.5]},
    {"label": "O3", "element": "O", "frac": [0.0, 0.0, 0.0]},
    {"label": "O4", "element": "O", "frac": [4e-7, 0.0, 0.0]},
]


def instances(*sites):
    flat = []
    for site in sites:
        flat.extend((site, 0, 0, 0))
    return flat


def grow(seed_site, minimum, pairs, current=None, maximum=2.0):
    current = instances(seed_site) if current is None else current
    return gw.grow_connectivity_shell(
        PARAMETERS,
        SITES,
        current,
        instances(seed_site),
        [
            {
                "element_a": a,
                "element_b": b,
                "enabled": True,
                "min_distance": minimum,
                "max_distance": maximum,
            }
            for a, b in pairs
        ],
    )


print("=== setup: the duplicate is grouped but not coincident ===")
unit_cell = gw._UnitCell(PARAMETERS)
atoms = gw._base_atoms(unit_cell, SITES)
canonical = gw._canonical_base_indices(atoms)
check("duplicate sites share a canonical index", canonical == [0, 0, 2, 3, 3],
      str(canonical))
separation = gw._image_distance(atoms[0]["frac"], atoms[1]["frac"],
                                unit_cell._columns)
check("duplicate separation is above the 1e-12 self filter",
      1e-12 < separation < 1e-5, f"{separation:.3e} A")

print("\n=== neighbour table: duplicates are not candidates ===")
windows = {
    "C": {"C": (0.0, 2.0), "O": (0.0, 2.0)},
    "O": {"C": (0.0, 2.0), "O": (0.0, 2.0)},
}
elements = [atom["element"] for atom in atoms]
table = gw._periodic_neighbour_table_gemmi(
    unit_cell, atoms, elements, windows, canonical
)
candidates = [entry[0] for entry in table[0]]
check("canonical seed bonds the real neighbour", 2 in candidates, str(table[0]))
check("canonical seed skips the duplicate copy", 1 not in candidates, str(table[0]))
check("non-canonical rows stay empty", table[1] == (), str(table[1]))

# Control: without the canonical map the duplicate really is a candidate, so
# the guard (not the distance filter) is what removes it above.
raw = gw._periodic_neighbour_table_gemmi(unit_cell, atoms, elements, windows)
raw_candidates = [entry[0] for entry in raw[0]]
check("control: without canonical the duplicate is a candidate",
      1 in raw_candidates, str(raw[0]))

print("\n=== growth with min_distance = 0 ===")
result = grow(0, 0.0, [("C", "C"), ("C", "O")])
check("only the real neighbour is grown", result["new_instances"] == [2, 0, 0, 0],
      str(result["new_instances"]))
check("the duplicate copy is not grown", 1 not in result["new_instances"],
      str(result["new_instances"]))
check("grew_by counts it once", result["grew_by"] == 1, str(result["grew_by"]))

print("\n=== duplicate away from the seed ===")
result = grow(3, 0.0, [("O", "O")], current=instances(3, 4))
check("nothing is grown from a lone duplicate", result["new_instances"] == [],
      str(result["new_instances"]))

print("\n=== default min_distance = 0.1 is unchanged ===")
result = grow(0, 0.1, [("C", "C"), ("C", "O")])
check("real neighbour still grown", result["new_instances"] == [2, 0, 0, 0],
      str(result["new_instances"]))

print("\n=== a longer window still walks out to the next shell ===")
result = grow(0, 0.0, [("C", "O")], maximum=3.0)
check("both oxygens are reached",
      sorted(tuple(result["new_instances"][offset:offset + 4])
             for offset in range(0, len(result["new_instances"]), 4))
      == [(2, 0, 0, 0), (3, 0, 0, 0)],
      str(result["new_instances"]))

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
