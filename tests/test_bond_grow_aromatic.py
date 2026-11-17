"""Regression: chemdraw growth must not spawn one CGN_Bonds_Aromatic per grow.

Run:
  blender --background --factory-startup --python test_bond_grow_aromatic.py
"""

import sys

import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

CIF = _env.cif("multiple_bonds.cif")
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def bond_objects(root):
    return [o for o in root.objects if o.get("cgn_role") == "bonds"]


def torus_objects(root):
    return [o for o in bond_objects(root) if o.get("cgn_aromatic_torus")]


def groups(root):
    keys = [
        (o.get("cgn_pair"), int(o.get("cgn_bond_order", 1)), o.get("cgn_bond_element"))
        for o in bond_objects(root)
        if not o.get("cgn_aromatic_torus")
    ]
    return keys


bpy.ops.cgn.import_cif(filepath=CIF)
scene = bpy.context.scene
scene.cgn_chemdraw = True
root = bpy.data.collections.get(scene.cgn_active_structure)
CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
CrystalViz._rebuild_bonds(scene)

print(
    f"  after import: {len(bond_objects(root))} bond objects, "
    f"{len(torus_objects(root))} aromatic objects"
)
baseline_groups = len(set(groups(root)))
baseline_torus = len(torus_objects(root))

for step in range(1, 4):
    bpy.ops.cgn.grow_further()
    keys = groups(root)
    torus = torus_objects(root)
    print(
        f"  after grow {step}: {len(bond_objects(root))} bond objects, "
        f"{len(torus)} aromatic objects, {len(keys)} non-aromatic objects"
    )
    check(f"grow {step}: at most one aromatic object", len(torus) <= 1, f"{len(torus)}")
    check(
        f"grow {step}: no duplicate non-aromatic groups",
        len(set(keys)) == len(keys),
        f"{len(keys)} objects / {len(set(keys))} groups",
    )
    check(
        f"grow {step}: no .00x bond objects",
        not any(
            name[-4:-3] == "." and name[-3:].isdigit()
            for name in (o.name for o in bond_objects(root))
        ),
        "",
    )

# Force new aromatic rings into the scene: duplicate every instance one cell
# over (a synthetic grow) and run the append-only bond path on the new atoms.
sites, instances, cell_matrix, initial_count = CrystalViz._structure_tables(root)
instances = list(instances)
copy = [
    (int(instances[o]), int(instances[o + 1]) + 1, int(instances[o + 2]), int(instances[o + 3]))
    for o in range(0, len(instances) - len(instances) % 4, 4)
]
for quad in copy:
    instances.extend(quad)
old_atoms = CrystalViz._structure_atoms(root)
old_count = len(old_atoms)
CrystalViz._set_instances(root, instances)
atoms = CrystalViz._structure_atoms(root)
torus_before = len(torus_objects(root))
entries_before = len(__import__("json").loads(torus_objects(root)[0]["cgn_aromatic_rings"]))
CrystalViz._bond_objects(
    root,
    atoms,
    scene.cgn_connectivity_pairs,
    start_index=old_count,
    append_only=True,
)
torus_after = torus_objects(root)
print(
    f"  synthetic grow: atoms {old_count} -> {len(atoms)}, "
    f"aromatic objects {torus_before} -> {len(torus_after)}, "
    f"rings {entries_before} -> "
    f"{len(__import__('json').loads(torus_after[0]['cgn_aromatic_rings']))}"
)
check(
    "synthetic grow keeps exactly one aromatic object", len(torus_after) == 1, f"{len(torus_after)}"
)
check(
    "synthetic grow adds the new rings to it",
    len(__import__("json").loads(torus_after[0]["cgn_aromatic_rings"])) > entries_before,
)

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
