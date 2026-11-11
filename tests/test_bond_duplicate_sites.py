"""The bond builder must bond a disordered site once, not once per copy.

A CIF can list the same element twice at one site, so the P1 atom list holds
two atoms at the same point.  The bond builder used to treat them as two atoms:
with ``min_distance = 0`` it created a zero-length bond between the copies plus
a second (overlapping) bond from each copy to every neighbour.  Only the
representative now receives bonds, mirroring the worker's canonical rule that
stops growth adding the copies.
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
    """Every drawn bond as a pair of display-space endpoints."""
    segments = []
    for obj in root.objects:
        if obj.get("cgn_role") != "bonds":
            continue
        mesh = obj.data
        for edge in mesh.edges:
            segments.append(
                (
                    tuple(mesh.vertices[edge.vertices[0]].co),
                    tuple(mesh.vertices[edge.vertices[1]].co),
                )
            )
    return segments


def segment_length(segment):
    first, second = segment
    return sum((a - b) ** 2 for a, b in zip(first, second)) ** 0.5


C = atom("C", (0.0, 0.0, 0.0))
O = atom("O", (1.2, 0.0, 0.0), vdw=1.52, covalent=0.66)

# ---- 1. exact duplicate -------------------------------------------------
print("=== exact duplicate site (min_distance = 0) ===")
root = make_root("CGN_Test_Dup_Exact")
pairs = set_pairs([("C", "C", 0.0, 2.0), ("C", "O", 0.0, 2.0)])
CrystalViz._bond_objects(root, [C, atom("C", (0.0, 0.0, 0.0)), O], pairs)
segments = bond_segments(root)
check("one bond for the disordered site", len(segments) == 1, f"n={len(segments)}")
check("the drawn bond is the real C-O contact",
      segments and abs(segment_length(segments[0]) - 0.12) < 1e-6,
      str(segments[:1]))
check("no zero-length bond", all(segment_length(s) > 1e-3 for s in segments))

# ---- 2. near duplicate (above the 1e-12 self filter) --------------------
print("\n=== near duplicate, 2e-6 A apart (min_distance = 0) ===")
root = make_root("CGN_Test_Dup_Near")
pairs = set_pairs([("C", "C", 0.0, 2.0), ("C", "O", 0.0, 2.0)])
CrystalViz._bond_objects(root, [C, atom("C", (2e-6, 0.0, 0.0)), O], pairs)
segments = bond_segments(root)
check("one bond for the near duplicate", len(segments) == 1, f"n={len(segments)}")
check("no zero-length bond", all(segment_length(s) > 1e-3 for s in segments))

# ---- 3. different elements at one point ---------------------------------
print("\n=== different elements at one point (true disorder) ===")
root = make_root("CGN_Test_Dup_Elements")
pairs = set_pairs([("FE", "MN", 0.0, 2.0)])
CrystalViz._bond_objects(
    root,
    [atom("FE", (0.0, 0.0, 0.0)), atom("MN", (0.0, 0.0, 0.0))],
    pairs,
)
check("coincident different elements stay unbonded",
      bond_segments(root) == [], str(bond_segments(root)))

# ---- 4. ordinary atoms are still bonded ---------------------------------
print("\n=== a normal two-bond chain is untouched ===")
root = make_root("CGN_Test_Dup_Chain")
pairs = set_pairs([("C", "O", 0.0, 2.0)])
chain = [C, O, atom("C", (2.2, 0.0, 0.0))]
CrystalViz._bond_objects(root, chain, pairs)
segments = bond_segments(root)
check("both C-O contacts are drawn", len(segments) == 2, f"n={len(segments)}")
check("lengths are 1.2 A and 1.0 A",
      sorted(round(segment_length(s) * 10.0, 6) for s in segments) == [1.0, 1.2],
      str(sorted(round(segment_length(s) * 10.0, 6) for s in segments)))

# ---- 5. grow/append path ------------------------------------------------
print("\n=== append (grow) path ===")
root = make_root("CGN_Test_Dup_Append")
pairs = set_pairs([("C", "C", 0.0, 2.0), ("C", "O", 0.0, 2.0)])
CrystalViz._bond_objects(root, [C, O], pairs)
check("initial build: one bond", len(bond_segments(root)) == 1,
      str(len(bond_segments(root))))
CrystalViz._bond_objects(
    root,
    [C, O, atom("C", (0.0, 0.0, 0.0))],
    pairs,
    start_index=2,
    append_only=True,
)
check("appending a duplicate adds no bond", len(bond_segments(root)) == 1,
      f"n={len(bond_segments(root))}")
CrystalViz._bond_objects(
    root,
    [C, O, atom("C", (0.0, 0.0, 0.0)), atom("C", (-1.4, 0.0, 0.0))],
    pairs,
    start_index=2,
    append_only=True,
)
check("the same append with a real atom adds its bond",
      len(bond_segments(root)) == 2, f"n={len(bond_segments(root))}")

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
