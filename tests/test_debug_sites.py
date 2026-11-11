"""Debug: count centers per site and verify instance creation."""
import sys, bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

CIF = _env.cif("Ag2Cr2O7.cif")

scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms0 = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms0)

bpy.ops.cgn.grow_further()
bpy.ops.cgn.grow_further()
atoms2 = CrystalViz._structure_atoms(root)

# Replicate the grouping logic from _build_polyhedra_objects
ag_indices = [i for i, a in enumerate(atoms2) if str(a.get("element", "X")).upper() == "AG"]
print(f"Total Ag atoms: {len(ag_indices)}")

site_groups = {}
site_rep = {}
for center in ag_indices:
    atom = atoms2[center]
    p1_index = atom.get("p1_index", center)
    site_groups.setdefault(p1_index, []).append(center)
    trans = tuple(atom.get("growth_translation", (0, 0, 0)))
    if p1_index not in site_rep or trans == (0, 0, 0):
        site_rep[p1_index] = center

print(f"Site groups: {len(site_groups)}")
for p1, centers in sorted(site_groups.items()):
    rep = site_rep[p1]
    rep_trans = tuple(atoms2[rep].get("growth_translation", (0,0,0)))
    print(f"  P1={p1}: {len(centers)} centers, rep translation={rep_trans}")
    for c in centers:
        trans = tuple(atoms2[c].get("growth_translation", (0,0,0)))
        print(f"    center {c}: translation={trans}")

# Check which sites qualify (4+ ligands)
search = CrystalViz._polyhedron_search(root, scene)
for p1, centers in sorted(site_groups.items()):
    rep = site_rep[p1]
    rep_atom = atoms2[rep]
    element = str(rep_atom.get("element", "X")).upper()
    ligands = search.ligands(rep_atom["cart"], element)
    print(f"  P1={p1}: {len(ligands)} ligands (need 4+) -> {'QUALIFIES' if len(ligands) >= 4 else 'SKIP'}")

CrystalViz.unregister()
