"""The bond builder rejects far candidates per element, not per global window.

Before this change the spatial hash rejected a candidate only when its distance
exceeded the *largest* enabled window in the whole pair table.  A single
permissive pair therefore made every centre test out to that large distance.
The builder now computes ``reach[element]`` (that element's widest enabled
window) and rejects against it, so a permissive pair inflates only centres of
the elements it actually involves.  This suite pins the change down: output
must be byte-identical to an exhaustive O(N^2) reference that tests every pair
against its own window, and a permissive pair must not alter the bonds of an
unrelated short-range element.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + the add-on parent on sys.path)
import bpy
import CrystalViz

CrystalViz.register()

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


scene = bpy.context.scene


def atom(element, position, vdw=1.7, covalent=0.76):
    return {
        "label": element,
        "element": element,
        "cart": list(position),
        "vdw_radius": vdw,
        "covalent_radius": covalent,
    }


def make_root(name):
    root = bpy.data.collections.new(name)
    scene.collection.children.link(root)
    return root


def set_pairs(pairs):
    scene.cgn_connectivity_pairs.clear()
    for element_a, element_b, low, high in pairs:
        item = scene.cgn_connectivity_pairs.add()
        item.element_a = element_a
        item.element_b = element_b
        item.enabled = True
        item.min_distance = low
        item.max_distance = high
    return scene.cgn_connectivity_pairs


def bond_segments(root):
    """Every drawn bond as a pair of display-space endpoints (0.1 scale).

    Each segment's two endpoints are sorted so the comparison is ordering-free
    (a bond is an undirected edge).
    """
    segments = []
    for obj in root.objects:
        if obj.get("cgn_role") != "bonds":
            continue
        mesh = obj.data
        for edge in mesh.edges:
            seg = (
                tuple(round(c, 6) for c in mesh.vertices[edge.vertices[0]].co),
                tuple(round(c, 6) for c in mesh.vertices[edge.vertices[1]].co),
            )
            segments.append(tuple(sorted(seg)))
    return sorted(segments)


def brute_bond_segments(atoms, pairs):
    """Exhaustive O(N^2) reference: each atom pair against its own window.

    Mirrors the canonical-base rule: a duplicate atom (same element, same point)
    is represented by its first occurrence, which is the only one that bonds.
    """
    canonical = CrystalViz._canonical_atom_indices(atoms)
    window = {}
    for element_a, element_b, low, high in pairs:
        window.setdefault((element_a, element_b), (low, high))
    segments = set()
    for i in range(len(atoms)):
        if canonical[i] != i:
            continue
        ai = atoms[i]
        for j in range(i + 1, len(atoms)):
            if canonical[j] != j:
                continue
            aj = atoms[j]
            key = tuple(sorted((ai["element"], aj["element"])))
            if key not in window:
                continue
            low, high = window[key]
            d = sum((a - b) ** 2 for a, b in zip(ai["cart"], aj["cart"])) ** 0.5
            if low <= d <= high:
                seg = (
                    tuple(round(c * 0.1, 6) for c in ai["cart"]),
                    tuple(round(c * 0.1, 6) for c in aj["cart"]),
                )
                segments.add(tuple(sorted(seg)))
    return sorted(segments)


# ---- 1. parity vs brute force with disparate reaches --------------------
print("=== parity vs O(N^2) reference (disparate reaches) ===")
# H bonds only to O at <=1.2 A; C bonds C at <=4.0 A; the permissive C-C window
# must not change which O-H contacts are found.
atoms = [
    atom("O", (0.0, 0.0, 0.0)),
    atom("H", (0.9, 0.0, 0.0)),
    atom("H", (1.5, 0.0, 0.0)),
    atom("H", (1.9, 0.0, 0.0)),
    atom("C", (4.5, 4.5, 4.5)),
    atom("C", (4.5, 5.0, 4.5)),
    atom("C", (4.2, 4.0, 4.8)),
]
pairs = [
    ("H", "O", 0.0, 1.2),
    ("C", "C", 0.0, 4.0),
]
root = make_root("CGN_Test_PPR_Parity1")
pair_settings = set_pairs(pairs)
CrystalViz._bond_objects(root, atoms, pair_settings)
actual = bond_segments(root)
expected = brute_bond_segments(atoms, pairs)
check("parity: identical bonds", actual == expected,
      f"actual={len(actual)} expected={len(expected)}")
check("the short-range O-H window is respected",
      all(sum((a - b) ** 2 for a, b in zip(*seg)) ** 0.5 <= 0.12 + 1e-6
          for seg in actual), str(len(actual)))

# ---- 2. permissive pair for one element leaves others untouched ---------
print("\n=== a permissive pair does not relabel short-range centres ===")
# Fe-Fe window is huge (8 A); if the global max were driving rejection it would
# still find nothing new, but we assert the built set is *exactly* the pairs in
# the table -- including that the O-H contacts are found with the tight window.
atoms2 = [
    atom("O", (0.0, 0.0, 0.0)),
    atom("H", (1.0, 0.0, 0.0)),
    atom("H", (2.5, 0.0, 0.0)),
    atom("FE", (10.0, 10.0, 10.0)),
    atom("FE", (12.0, 10.0, 10.0)),
]
pairs2 = [("H", "O", 0.0, 1.2), ("FE", "FE", 0.0, 8.0)]
root = make_root("CGN_Test_PPR_Parity2")
pair_settings2 = set_pairs(pairs2)
CrystalViz._bond_objects(root, atoms2, pair_settings2)
actual2 = bond_segments(root)
expected2 = brute_bond_segments(atoms2, pairs2)
check("parity: identical bonds (one short, one very wide)",
      actual2 == expected2, f"actual={len(actual2)} expected={len(expected2)}")

# ---- 3. element with no enabled pair draws no bonds but stays bucketed --
print("\n=== element with no enabled pair ===")
# N has no pair entry; it must not bond to anything, and the search must not
# error.  C-O (1.2 A) must still bond.
atoms3 = [
    atom("O", (0.0, 0.0, 0.0)),
    atom("N", (2.0, 2.0, 2.0)),
    atom("C", (1.2, 0.0, 0.0)),
    atom("N", (1.0, 1.0, 2.2)),
]
pairs3 = [("C", "O", 0.0, 2.0)]
root = make_root("CGN_Test_PPR_NoPair")
pair_settings3 = set_pairs(pairs3)
CrystalViz._bond_objects(root, atoms3, pair_settings3)
actual3 = bond_segments(root)
expected3 = brute_bond_segments(atoms3, pairs3)
check("parity: identical bonds (element with no pair entry)",
      actual3 == expected3, f"actual={len(actual3)} expected={len(expected3)}")
check("only the C-O contact is drawn", len(actual3) == 1, f"n={len(actual3)}")

# ---- 4. asymmetry: A-B enabled, B has no other pairs --------------------
print("\n=== asymmetric element reach ===")
# K has no pair except K-S (set wide); S is present.  Both centres of the K-S
# pair must find each other from either side.
atoms4 = [
    atom("K", (0.0, 0.0, 0.0)),
    atom("S", (0.5, 1.2, 2.0)),
    atom("K", (0.3, 1.0, 1.7)),
    atom("H", (9.0, 9.0, 9.0)),
]
pairs4 = [("K", "S", 0.0, 2.5)]
root = make_root("CGN_Test_PPR_Asym")
pair_settings4 = set_pairs(pairs4)
CrystalViz._bond_objects(root, atoms4, pair_settings4)
actual4 = bond_segments(root)
expected4 = brute_bond_segments(atoms4, pairs4)
check("parity: identical bonds (asymmetric reach)",
      actual4 == expected4, f"actual={len(actual4)} expected={len(expected4)}")

# ---- 5. appended (grow) path keeps per-element reach --------------------
print("\n=== append path with per-element reach ===")
pairs5 = [("H", "O", 0.0, 1.2), ("FE", "FE", 0.0, 8.0)]
pair_settings5 = set_pairs(pairs5)
atoms5a = [atom("O", (0.0, 0.0, 0.0)), atom("H", (0.8, 0.0, 0.0))]
root = make_root("CGN_Test_PPR_Append")
CrystalViz._bond_objects(root, atoms5a, pair_settings5)
n_first = len(bond_segments(root))
atoms5b = atoms5a + [atom("H", (1.0, 0.0, 0.0))]
CrystalViz._bond_objects(
    root, atoms5b, pair_settings5,
    start_index=len(atoms5a), append_only=True,
)
actual5 = bond_segments(root)
expected5 = brute_bond_segments(atoms5b, pairs5)
check("parity: identical bonds after append", actual5 == expected5,
      f"actual={len(actual5)} expected={len(expected5)}")
check("append added the second H-O bond",
      len(actual5) == 2 and n_first == 1, f"first={n_first} then={len(actual5)}")

# ---- 6. min_distance > 0 respected with per-element reach ---------------
print("\n=== min_distance lower bound still respected ===")
atoms6 = [
    atom("O", (0.0, 0.0, 0.0)),
    atom("H", (0.3, 0.0, 0.0)),
    atom("H", (1.2, 0.0, 0.0)),
]
pairs6 = [("H", "O", 0.6, 1.5), ("CL", "CL", 0.0, 4.0)]
root = make_root("CGN_Test_PPR_MinDist")
pair_settings6 = set_pairs(pairs6)
CrystalViz._bond_objects(root, atoms6, pair_settings6)
actual6 = bond_segments(root)
expected6 = brute_bond_segments(atoms6, pairs6)
check("parity: identical bonds (min_distance bound)",
      actual6 == expected6, f"actual={len(actual6)} expected={len(expected6)}")
check("the too-close H is excluded",
      len(actual6) == 1, f"n={len(actual6)}")

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
