"""Validate S1/C1 neighbour-table optimisation preserves cctbx parity.

The patched ``_periodic_neighbour_table_gemmi`` prunes candidate neighbours
with a gemmi.NeighborSearch spatial index (via ``atom_idx``) and then resolves
the exact periodic images with the same ``find_nearest_pbc_images`` as before.
This test rebuilds the ORIGINAL all-pairs table as a reference and asserts the
new table is byte-identical, and that end-to-end growth output is unchanged.

Run:  python test_S1_parity.py
"""
import sys, os, math
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (puts a gemmi build on sys.path)
sys.path.insert(0, _env.REPO_DIR)  # so `import gemmi_worker` resolves
import gemmi
import gemmi_worker as gw

CIF = _env.CIF_DIR

FAILURES = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def reference_table(unit_cell, base_atoms, base_elements, windows_by_element, canonical):
    """The ORIGINAL all-pairs algorithm (pre-optimisation)."""
    cell = unit_cell.cell
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
        neighbours = []
        seen = set()
        for j, atom_j in enumerate(base_atoms):
            element_j = base_elements[j]
            bounds = pair_windows.get(element_j)
            if bounds is None:
                continue
            frac_j = gemmi.Fractional(*atom_j["frac"])
            for im in cell.find_nearest_pbc_images(frac_i, bounds[1], frac_j, 0):
                f_img = cell.fract_image(im, frac_j)
                tx = int(round(f_img.x - frac_j.x))
                ty = int(round(f_img.y - frac_j.y))
                tz = int(round(f_img.z - frac_j.z))
                c_img = cell.orthogonalize(f_img)
                c_i = cell.orthogonalize(frac_i)
                dist = math.sqrt((c_img.x - c_i.x)**2 + (c_img.y - c_i.y)**2 + (c_img.z - c_i.z)**2)
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


def windows_from_pairs(pair_settings):
    windows = {}
    for s in pair_settings:
        ea = str(s.get("element_a", "")).upper()
        eb = str(s.get("element_b", "")).upper()
        mn = float(s.get("min_distance", 0.0))
        mx = float(s.get("max_distance", 0.0))
        if s.get("enabled", False) and 0.0 <= mn <= mx:
            windows[tuple(sorted((ea, eb)))] = (mn, mx)
    return windows


# Replicate the window table shape _grow_shell_keys builds, from a pair list.
def build_context(cif_path, pair_settings):
    data = gw.load_structure(cif_path)
    unit_cell = gw._UnitCell(data["unit_cell"])
    base_atoms = gw._base_atoms(unit_cell, data["sites"])
    base_elements = [a["element"] for a in base_atoms]
    canonical = gw._canonical_base_indices(base_atoms)
    windows = windows_from_pairs(pair_settings)
    wbe = {}
    for (ea, eb), bounds in windows.items():
        wbe.setdefault(ea, {})[eb] = bounds
        wbe.setdefault(eb, {})[ea] = bounds
    return unit_cell, base_atoms, base_elements, canonical, wbe, data, windows


# ---- Table-level parity across example CIFs ----
print("=== Table-level cctbx parity ===")
CIFS = [
    ("Brookite.cif", [{"element_a": "Ti", "element_b": "O", "enabled": True,
                       "min_distance": 1.4, "max_distance": 2.4}]),
    ("aromatic.cif", [{"element_a": "C", "element_b": "C", "enabled": True,
                       "min_distance": 1.1, "max_distance": 1.9},
                      {"element_a": "C", "element_b": "H", "enabled": True,
                       "min_distance": 0.8, "max_distance": 1.4},
                      {"element_a": "C", "element_b": "N", "enabled": True,
                       "min_distance": 1.1, "max_distance": 1.6},
                      {"element_a": "H", "element_b": "O", "enabled": True,
                       "min_distance": 0.8, "max_distance": 1.2}]),
    ("test.cif", [{"element_a": "Ti", "element_b": "O", "enabled": True,
                   "min_distance": 1.4, "max_distance": 2.4},
                  {"element_a": "O", "element_b": "O", "enabled": True,
                   "min_distance": 2.0, "max_distance": 3.2}]),
]
for name, pairs in CIFS:
    uc, ba, be, canonical, wbe, data, windows = build_context(os.path.join(CIF, name), pairs)
    new = gw._periodic_neighbour_table_gemmi(uc, ba, be, wbe, canonical)
    ref = reference_table(uc, ba, be, wbe, canonical)
    check(f"{name}: neighbour table byte-identical to reference", new == ref,
          f"base_atoms={len(ba)}")

# ---- End-to-end growth parity (Brookite) ----
print("\n=== End-to-end growth parity ===")
# Reference: emulate one shell using the reference table via _grow_shell_keys,
# but _grow_shell_keys builds the table itself (the patched one).  Instead we
# compare the NEW table fed into the same key-walk against the reference walk.
uc, ba, be, canonical, wbe, data, windows = build_context(os.path.join(CIF, "Brookite.cif"), CIFS[0][1])
all_instances = data["instances"]
seed = [0, 0, 0, 0]  # base atom 0, no translation
new_tbl = gw._periodic_neighbour_table_gemmi(uc, ba, be, wbe, canonical)
ref_tbl = reference_table(uc, ba, be, wbe, canonical)
visited = gw._instance_keys(all_instances, canonical)
seed_keys = gw._instance_keys(seed, canonical)
n_ref, _ = gw._grow_shell_keys(uc, ba, canonical, set(visited), seed_keys, windows, ref_tbl)
n_new, _ = gw._grow_shell_keys(uc, ba, canonical, set(visited), seed_keys, windows, new_tbl)
# note: pass copies of visited since _grow_shell_keys mutates it
check("Brookite: one growth shell identical (new vs reference keys)",
      sorted(n_ref) == sorted(n_new), f"ref={len(n_ref)} new={len(n_new)}")

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
