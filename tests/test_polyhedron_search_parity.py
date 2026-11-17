"""The cached polyhedron search must match the per-center brute force.

The coordination search used to build a ``gemmi.UnitCell`` and rescan every P1
site for each centre.  It is now one object per structure revision: one cell,
one gemmi ``NeighborSearch`` index, and ligand lists memoised per (element,
position).  This suite compares that fast path against an independent
brute-force reference on every example CIF (so the spatial index can never drop
a ligand), repeats it on a small cell where several images of an atom fall
inside the window, checks the pool-scan fallback, and checks the cache/memo
contract the panel draw relies on.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + the add-on parent on sys.path)
import bpy
import CrystalViz
import gemmi

CrystalViz.register()

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def reference_cell(cell_matrix):
    """Independent gemmi cell construction from the column-major matrix."""
    vectors = [
        cell_matrix[0:3],
        cell_matrix[3:6],
        cell_matrix[6:9],
    ]

    def norm(v):
        return math.sqrt(sum(value * value for value in v))

    def angle(u, v):
        cosine = sum(a * b for a, b in zip(u, v, strict=True)) / (norm(u) * norm(v))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    a, b, c = vectors
    return gemmi.UnitCell(norm(a), norm(b), norm(c), angle(b, c), angle(a, c), angle(a, b))


def reference_ligands(center_cart, element, pool, cell_matrix, windows):
    """The pre-change algorithm: new cell per centre, every pool atom scanned."""
    cell = reference_cell(cell_matrix)
    center_frac = cell.fractionalize(gemmi.Position(*center_cart))
    ligands = []
    seen = set()
    for atom in pool:
        bounds = CrystalViz._pair_key(element, atom["element"])
        if bounds not in windows:
            continue
        minimum, maximum = windows[bounds]
        if maximum <= 0.0:
            continue
        pj = atom["frac"]
        pj_gemmi = gemmi.Fractional(pj[0], pj[1], pj[2])
        for im in cell.find_nearest_pbc_images(center_frac, maximum, pj_gemmi, 0):
            f_img = cell.fract_image(im, pj_gemmi)
            tx = int(round(f_img.x - pj[0]))
            ty = int(round(f_img.y - pj[1]))
            tz = int(round(f_img.z - pj[2]))
            c_img = cell.orthogonalize(f_img)
            delta = (
                center_cart[0] - c_img.x,
                center_cart[1] - c_img.y,
                center_cart[2] - c_img.z,
            )
            d2 = sum(value * value for value in delta)
            if d2 <= 1e-12:
                continue
            distance = math.sqrt(d2)
            if distance < minimum or distance > maximum:
                continue
            pos = CrystalViz._display_point(c_img)
            key = tuple(round(v, 6) for v in pos)
            if key in seen:
                continue
            seen.add(key)
            ligands.append((pos, atom["index"], (tx, ty, tz)))
    return tuple(ligands)


scene = bpy.context.scene
CIFS = ["Brookite.cif", "multiple_bonds.cif", "Ag2Cr2O7.cif"]

for cif in CIFS:
    print(f"\n=== {cif}: search matches the brute-force reference ===")
    bpy.ops.cgn.import_cif(filepath=_env.cif(cif))
    root = bpy.data.collections.get(scene.cgn_active_structure)
    atoms = CrystalViz._structure_atoms(root)
    CrystalViz._default_connectivity_pairs(scene, atoms)
    pool, cell_matrix = CrystalViz._p1_pool(root)
    windows = CrystalViz._enabled_pair_windows(scene.cgn_connectivity_pairs)
    search = CrystalViz._polyhedron_search(root, scene)
    check(f"{cif}: search built", search is not None and search.cell is not None)

    mismatches = 0
    reference_eligible = set()
    for site in pool:
        expected = reference_ligands(site["cart"], site["element"], pool, cell_matrix, windows)
        actual = search.ligands(site["cart"], site["element"])
        if tuple(actual) != expected:
            mismatches += 1
            if mismatches == 1:
                print(
                    f"    first mismatch at site {site['index']} "
                    f"({len(actual)} vs {len(expected)} ligands)"
                )
        if len(expected) >= CrystalViz._POLY_MIN_LIGANDS:
            reference_eligible.add(site["element"])
    check(
        f"{cif}: every centre's ligand list is identical",
        mismatches == 0,
        f"mismatches={mismatches}/{len(pool)}",
    )
    check(
        f"{cif}: eligible elements match the reference",
        search.eligible_elements() == sorted(reference_eligible),
        f"{search.eligible_elements()} vs {sorted(reference_eligible)}",
    )

    print(f"--- {cif}: cache contract ---")
    memo_size = len(search._ligands)
    same = CrystalViz._polyhedron_search(root, scene) is search
    check(f"{cif}: one search object per revision", same)
    CrystalViz._cached_polyhedron_elements(root, scene)
    check(
        f"{cif}: a redraw reuses the memo",
        len(search._ligands) == memo_size,
        f"{memo_size} -> {len(search._ligands)}",
    )
    check(f"{cif}: eligible elements are cached", search._eligible is not None)

    # A capped query must not poison the memo with a truncated list.
    site = max(pool, key=lambda item: len(search.ligands(item["cart"], item["element"])))
    full = search.ligands(site["cart"], site["element"])
    search._ligands.pop(
        (site["element"],) + tuple(round(float(v), 6) for v in site["cart"]),
        None,
    )
    capped = search.ligands(site["cart"], site["element"], limit=CrystalViz._POLY_MIN_LIGANDS)
    check(
        f"{cif}: capped query stops at the minimum",
        len(capped) == CrystalViz._POLY_MIN_LIGANDS,
        f"n={len(capped)}",
    )
    key = (site["element"],) + tuple(round(float(v), 6) for v in site["cart"])
    check(f"{cif}: capped query is not memoised", key not in search._ligands)
    check(
        f"{cif}: the full query still returns everything",
        search.ligands(site["cart"], site["element"]) == full,
    )

    # Changing a window invalidates the memo but keeps the same pool.
    cheap = scene.cgn_connectivity_pairs[0]
    cheap.max_distance = max(0.0, float(cheap.max_distance) - 0.2)
    rekeyed = CrystalViz._polyhedron_search(root, scene)
    check(f"{cif}: same object after a window change", rekeyed is search)
    check(
        f"{cif}: window change clears the memo",
        len(search._ligands) == 0,
        str(len(search._ligands)),
    )
    CrystalViz._default_connectivity_pairs(scene, atoms)
    CrystalViz._polyhedron_search(root, scene)

# A small cell puts several periodic images of every site inside the window, so
# the gemmi index returns each atom more than once.  It must not drop any, and
# the pool-scan fallback used when no index can be built must agree with it.
print("\n=== synthetic small cell: image enumeration ====")
SMALL = 3.6
cell_matrix = [SMALL, 0.0, 0.0, 0.0, SMALL, 0.0, 0.0, 0.0, SMALL]
raw = [
    ("C", (0.0, 0.0, 0.0)),
    ("C", (0.5, 0.5, 0.5)),
    ("NA", (0.25, 0.25, 0.25)),
    ("C", (0.05, 0.45, 0.85)),
]
pool = []
for index, (element, frac) in enumerate(raw):
    cart = [
        frac[0] * cell_matrix[axis]
        + frac[1] * cell_matrix[3 + axis]
        + frac[2] * cell_matrix[6 + axis]
        for axis in range(3)
    ]
    pool.append({"element": element, "frac": list(frac), "cart": cart, "index": index})
windows = {CrystalViz._pair_key("C", "C"): (0.0, 3.5), CrystalViz._pair_key("C", "NA"): (0.0, 3.5)}

indexed = CrystalViz._PolyhedronSearch(pool, cell_matrix)
indexed.update_windows(windows)
check("small cell: index built", indexed._neighbor_index() is not None)
small_mismatch = 0
for atom in pool:
    expected = reference_ligands(atom["cart"], atom["element"], pool, cell_matrix, windows)
    actual = indexed.ligands(atom["cart"], atom["element"])
    if tuple(actual) != expected:
        small_mismatch += 1
check(
    "small cell: indexed ligands match the reference",
    small_mismatch == 0,
    f"mismatches={small_mismatch}/{len(pool)}",
)
check(
    "small cell: multiple images are enumerated",
    any(len(indexed.ligands(a["cart"], a["element"])) > len(raw) for a in pool),
    str([len(indexed.ligands(a["cart"], a["element"])) for a in pool]),
)

fallback = CrystalViz._PolyhedronSearch(pool, cell_matrix)
fallback.update_windows(windows)
fallback._index = ()  # force the pool scan
check("small cell: the fallback scan is used", fallback._neighbor_index() is None)
fallback_mismatch = 0
for atom in pool:
    expected = reference_ligands(atom["cart"], atom["element"], pool, cell_matrix, windows)
    if fallback.ligands(atom["cart"], atom["element"]) != tuple(expected):
        fallback_mismatch += 1
check(
    "small cell: fallback agrees with the reference",
    fallback_mismatch == 0,
    f"mismatches={fallback_mismatch}/{len(pool)}",
)

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
