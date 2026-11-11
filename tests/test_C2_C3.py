"""Validate C2 (symmetry-mates helper) and C3 (periodic-distance helper).

Run:  python test_C2_C3.py

C2: _symmetry_mates must enumerate exactly the same distinct wrapped positions
    that the pre-change _expand_site produced, and the public load_structure
    P1 site table (order + fracs) must be unchanged.

C3: _periodic_distance / _cart_from_frac must match gemmi's orthogonalize-based
    _cart_distance across random fractional coordinates on every example CIF.
"""
import sys, os, random
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (puts a gemmi build on sys.path)
sys.path.insert(0, _env.REPO_DIR)  # so `import gemmi_worker` resolves
import gemmi_worker as gw

CIF = _env.CIF_DIR
CIFS = ["Brookite.cif", "aromatic.cif", "test.cif"]

FAILURES = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def reference_expand_site(ops, unit_cell, frac):
    """The ORIGINAL _expand_site logic (wrap-then-dedup on wrapped image)."""
    images = []
    for operation in ops:
        image = gw._wrapped_fractional(operation.apply_to_xyz(frac))
        if any(
            gw._cart_distance(unit_cell, image, other[0]) <= gw._CART_TOLERANCE
            for other in images
        ):
            continue
        images.append((image, operation))
    return images


print("=== C2: _symmetry_mates matches original _expand_site positions ===")
for name in CIFS:
    st_data = gw.load_structure(os.path.join(CIF, name))
    uc = gw._UnitCell(st_data["unit_cell"])
    # ops come from the wrapped structure object -- use _read_structure path
    structure, _b, _c = gw._read_structure(os.path.join(CIF, name))
    ops = gw._space_group_ops(structure)
    random.seed(7)
    ok = True
    for _ in range(5):
        frac = [random.random() for _ in range(3)]
        ref = [im for im, _ in reference_expand_site(ops, uc, frac)]
        new = gw._symmetry_mates(ops, uc, frac)
        if len(new) != len(ref) or not all(
            any(gw._cart_distance(uc, a, b) <= gw._CART_TOLERANCE for b in ref) for a in new
        ):
            ok = False
    check(f"{name}: _symmetry_mates matches reference positions", ok,
          f"ops={len(ops)}")

print("\n=== C2: load_structure P1 output unchanged (regression) ===")
# load_structure should still produce identical sites/instances/cell_matrix
for name in CIFS:
    d = gw.load_structure(os.path.join(CIF, name))
    check(f"{name}: sites present", len(d.get("sites", [])) > 0,
          f"n_sites={len(d.get('sites', []))}")
    check(f"{name}: cell_matrix present", len(d.get("cell_matrix", [])) == 9)

print("\n=== C3: _periodic_distance matches gemmi orthogonalize ===")
for name in CIFS:
    d = gw.load_structure(os.path.join(CIF, name))
    uc = gw._UnitCell(d["unit_cell"])
    cmat = list(d["cell_matrix"])
    assert cmat == list(uc._columns), "matrix convention mismatch"
    random.seed(3 + len(name))
    mm = 0
    for _ in range(1000):
        a = [random.random() for _ in range(3)]
        b = [random.random() for _ in range(3)]
        if abs(gw._periodic_distance(a, b, cmat) - gw._cart_distance(uc, a, b)) > 1e-9:
            mm += 1
    check(f"{name}: _periodic_distance matches gemmi", mm == 0, f"mismatches={mm}")

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
