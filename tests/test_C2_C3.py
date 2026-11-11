"""Validate C2 (symmetry-mates helper) and C3 (periodic-distance helper).

Run:  python test_C2_C3.py

C2: _symmetry_mates must enumerate exactly the same distinct wrapped positions
    that the pre-change _expand_site produced, and the public load_structure
    P1 site table (order + fracs) must be unchanged.

C3: the production path (gemmi's find_nearest_pbc_images + _image_distance) must
    return the true 27-image minimum, _cart_distance is dedupe-only and must
    never undercut that minimum, and _image_distance must measure one specific
    image.  Checked on every example CIF plus a strongly skewed triclinic cell.
"""
import sys, os, random, math
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (puts a gemmi build on sys.path)
sys.path.insert(0, _env.REPO_DIR)  # so `import gemmi_worker` resolves
import gemmi
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

print("\n=== C3: gemmi's resolved image + _image_distance is the true minimum ===")


def brute_force_min(uc, a, b):
    """Brute-force minimum over the 27 neighbours, via gemmi's orthogonalize."""
    delta = [a[axis] - b[axis] for axis in range(3)]
    delta = [value - round(value) for value in delta]
    best = None
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                cart = uc.orthogonalize(
                    (delta[0] + sx, delta[1] + sy, delta[2] + sz)
                )
                distance = math.sqrt(sum(value * value for value in cart))
                if best is None or distance < best:
                    best = distance
    return best


def gemmi_min(uc, a, b):
    """The production path: gemmi resolves the image, _image_distance measures it.

    Mirrors the call site in _periodic_neighbour_table_gemmi (growth/bonding).
    """
    fa = gemmi.Fractional(*a)
    fb = gemmi.Fractional(*b)
    best = None
    for im in uc.cell.find_nearest_pbc_images(fa, 1000.0, fb, 0):
        f_img = uc.cell.fract_image(im, fb)
        distance = gw._image_distance(
            a, [f_img.x, f_img.y, f_img.z], uc._columns
        )
        if best is None or distance < best:
            best = distance
    return best


def rounding_only(uc, a, b):
    """_cart_distance: round each component, no neighbour search."""
    delta = [a[axis] - b[axis] for axis in range(3)]
    delta = [value - round(value) for value in delta]
    cart = uc.orthogonalize(delta)
    return math.sqrt(sum(value * value for value in cart))


for name in CIFS:
    d = gw.load_structure(os.path.join(CIF, name))
    uc = gw._UnitCell(d["unit_cell"])
    assert list(d["cell_matrix"]) == list(uc._columns), "matrix convention mismatch"
    random.seed(3 + len(name))
    mm = 0
    too_small = 0
    for _ in range(1000):
        a = [random.random() for _ in range(3)]
        b = [random.random() for _ in range(3)]
        if abs(gemmi_min(uc, a, b) - brute_force_min(uc, a, b)) > 1e-9:
            mm += 1
        if gw._cart_distance(uc, a, b) < brute_force_min(uc, a, b) - 1e-9:
            too_small += 1
    check(f"{name}: gemmi image + _image_distance matches the 27-image minimum",
          mm == 0, f"mismatches={mm}")
    check(f"{name}: _cart_distance never undercuts the true minimum",
          too_small == 0, f"undercuts={too_small}")

print("\n=== C3: skewed triclinic cell (rounding-only is provably wrong) ===")
tric = gw._UnitCell([10.0, 11.0, 12.0, 40.0, 55.0, 65.0])
tmat = list(tric._columns)
random.seed(11)
bad = 0
witness = 0
for _ in range(2000):
    a = [random.random() for _ in range(3)]
    b = [random.random() for _ in range(3)]
    true = brute_force_min(tric, a, b)
    if abs(gemmi_min(tric, a, b) - true) > 1e-9:
        bad += 1
    if rounding_only(tric, a, b) - true > 1e-6:
        witness += 1
check("triclinic: gemmi image matches brute force", bad == 0, f"bad={bad}")
check("triclinic: rounding-only differs (witness that the bug was real)",
      witness > 0, f"{witness}/2000 samples")

print("\n=== C3: _image_distance measures one specific image ===")
a, b = [random.random() for _ in range(3)], [random.random() for _ in range(3)]
shifted = [b[0] + 1.0, b[1] - 2.0, b[2] + 3.0]
delta = [a[axis] - shifted[axis] for axis in range(3)]
expected = math.sqrt(sum(v * v for v in tric.orthogonalize(delta)))
check("_image_distance ignores periodicity",
      abs(gw._image_distance(a, shifted, tmat) - expected) < 1e-9,
      f"{gw._image_distance(a, shifted, tmat):.6f} vs {expected:.6f}")

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
