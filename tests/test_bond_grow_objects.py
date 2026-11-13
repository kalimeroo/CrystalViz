"""Regression: growing must not add bond objects to the outliner.

Grow is append-only: new bond segments for the freshly grown atoms are folded
into the existing per-pair/order(/element) bond objects.  The object count must
therefore stay at "one object per distinct bond group" instead of growing by
one set of CGN_Bonds_* objects (with .001, .002 suffixes) per grow.

Import displays atoms only (no bonds), so this test builds the baseline bond
set explicitly with ``_bond_objects`` — the same call the grow path uses.

Run:
  blender --background --factory-startup --python test_bond_grow_objects.py
"""

import re
import sys

import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

CIF = _env.cif("Ag2Cr2O7.cif")
FAILURES = []
SUFFIX = re.compile(r"\.\d{3}$")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def bond_objects(root):
    return [
        obj
        for obj in root.objects
        if obj.get("cgn_role") == "bonds" and not obj.get("cgn_aromatic_torus")
    ]


def group_key(obj):
    return (
        obj.get("cgn_pair"),
        int(obj.get("cgn_bond_order", 1)),
        obj.get("cgn_bond_element"),
    )


def segment_count(root):
    return sum(len(obj.data.edges) for obj in bond_objects(root))


def segment_sets(root):
    result = {}
    for obj in bond_objects(root):
        key = group_key(obj)
        mesh = obj.data
        keys = set()
        for edge in mesh.edges:
            first = tuple(round(v, 4) for v in mesh.vertices[edge.vertices[0]].co)
            second = tuple(round(v, 4) for v in mesh.vertices[edge.vertices[1]].co)
            keys.add(tuple(sorted((first, second))))
        result.setdefault(key, set()).update(keys)
    return result


def report(root, label):
    objs = bond_objects(root)
    keys = [group_key(obj) for obj in objs]
    duplicate_groups = {key for key in keys if keys.count(key) > 1}
    suffixes = [obj.name for obj in objs if SUFFIX.search(obj.name)]
    print(
        f"  {label}: {len(objs)} bond objects, {len(set(keys))} groups, "
        f"{segment_count(root)} segments"
    )
    if duplicate_groups:
        print(f"    duplicate groups: {sorted(duplicate_groups)}")
    if suffixes:
        print(f"    suffixed names: {suffixes}")
    return objs, keys, duplicate_groups, suffixes


def exercise(two_color):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.cgn_bond_style = "TWO_COLOR" if two_color else "SINGLE_COLOR"
    bpy.ops.cgn.import_cif(filepath=CIF)
    root = bpy.data.collections.get(scene.cgn_active_structure)
    CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
    # A fill/import shows atoms only; build the baseline bond set so the grow
    # assertions below start from existing per-group objects.
    CrystalViz._bond_objects(root, CrystalViz._structure_atoms(root), scene.cgn_connectivity_pairs)
    bpy.context.view_layer.update()

    label = "two-colour" if two_color else "single-colour"
    print(f"=== {label} ===")
    objs, keys, dupes, suffixes = report(root, "after import")
    check(f"{label}: no duplicate groups after import", not dupes, str(dupes))
    check(f"{label}: no suffixed bond objects after import", not suffixes, str(suffixes))
    baseline_objects = len(objs)
    baseline_segments = segment_count(root)

    previous_count = baseline_objects
    previous_segments = baseline_segments
    for step in range(1, 4):
        bpy.ops.cgn.grow_further()
        objs, keys, dupes, suffixes = report(root, f"after grow {step}")
        check(f"{label}: grow {step} keeps one object per bond group", not dupes, str(dupes))
        check(f"{label}: grow {step} creates no suffixed bond objects", not suffixes, str(suffixes))
        check(
            f"{label}: grow {step} does not multiply bond objects",
            len(objs) <= baseline_objects + 4,
            f"{baseline_objects} -> {len(objs)}",
        )
        check(
            f"{label}: grow {step} adds bond segments",
            segment_count(root) > previous_segments,
            f"{previous_segments} -> {segment_count(root)}",
        )
        previous_count = len(objs)
        previous_segments = segment_count(root)
    # Option B: append-only growth must be a SUPERSET of a full rebuild — it
    # never loses a bond, but it may keep bonds from earlier connectivity
    # settings that the current windows no longer produce.
    grown = segment_sets(root)
    CrystalViz._bond_objects(root, CrystalViz._structure_atoms(root), scene.cgn_connectivity_pairs)
    rebuilt = segment_sets(root)
    missing = {group: rebuilt.get(group, set()) - grown.get(group, set()) for group in rebuilt}
    missing = {group: keys for group, keys in missing.items() if keys}
    check(
        f"{label}: append-only bonds contain every rebuilt bond",
        not missing,
        str({k: len(v) for k, v in missing.items()}),
    )
    return previous_count


single_objects = exercise(two_color=False)
two_objects = exercise(two_color=True)
print(f"  final single-colour objects: {single_objects}, two-colour objects: {two_objects}")

# Consolidation: a .blend grown by an older build contains duplicate bond
# objects for the same group.  The next grow must fold them back into one.
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
CrystalViz._bond_objects(root, CrystalViz._structure_atoms(root), scene.cgn_connectivity_pairs)
bpy.context.view_layer.update()
objs = bond_objects(root)
victim = objs[0]
duplicate = victim.copy()
duplicate.data = victim.data.copy()
duplicate.name = victim.name + ".001"
root.objects.link(duplicate)
print(f"=== consolidation ===\n  injected {duplicate.name}")
before = report(root, "after injection")
check("consolidation: duplicate existed before grow", bool(before[2]), str(before[2]))
bpy.ops.cgn.grow_further()
after = report(root, "after grow")
check("consolidation: duplicates folded on grow", not after[2], str(after[2]))
check("consolidation: suffixed duplicates removed", not after[3], str(after[3]))

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
