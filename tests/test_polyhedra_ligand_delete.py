"""Deleting a ligand atom (polyhedron vertex) must remove its polyhedron.

Deleting the central atom of a coordination polyhedron drops that image's
polyhedron.  Removing a ligand atom that sits on a polyhedron vertex now does
the same: the centre *images* whose coordination shell used the deleted ligand
lose their polyhedron, while the other images (different periodic copies, which
use different ligand instances) keep theirs.

Run headless:
  blender --background --factory-startup --python test_polyhedra_ligand_delete.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import bpy
import CrystalViz

CrystalViz.register()

CIF = _env.CIF_DIR

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def _reset_pairs(scene):
    scene.cgn_connectivity_pairs.clear()
    scene.cgn_chemdraw = False
    bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "Brookite.cif"))
    if not any(p.enabled for p in scene.cgn_connectivity_pairs):
        pair = scene.cgn_connectivity_pairs.add()
        pair.element_a = "Ti"
        pair.element_b = "O"
        pair.enabled = True
        pair.min_distance = 0.1
        pair.max_distance = 3.0
    bpy.context.view_layer.update()


def _ti_atoms(atoms):
    return [i for i, a in enumerate(atoms) if str(a.get("element", "X")).upper() == "TI"]


def setup_fill_polyhedra():
    """Import Brookite, fill one cell, build Ti polyhedra (one image per site)."""
    scene = bpy.context.scene
    _reset_pairs(scene)
    try:
        bpy.ops.cgn.fill_cell("INVOKE_DEFAULT")
    except Exception as exc:  # pragma: no cover - control issue only
        print("fill_cell failed (control issue):", exc)
    root = bpy.data.collections.get(scene.cgn_active_structure)
    bpy.context.view_layer.update()
    atoms = CrystalViz._structure_atoms(root)
    ti = _ti_atoms(atoms)
    built, _skipped = CrystalViz._build_polyhedra_objects(root, atoms, ti, scene)
    bpy.context.view_layer.update()
    return scene, root, atoms, ti, built


def setup_grow_polyhedra(shells=2):
    """Import Brookite, grow two shells, build Ti polyhedra (multi-image sites)."""
    scene = bpy.context.scene
    _reset_pairs(scene)
    for _ in range(shells):
        bpy.ops.cgn.grow_further()
        bpy.context.view_layer.update()
    root = bpy.data.collections.get(scene.cgn_active_structure)
    bpy.context.view_layer.update()
    atoms = CrystalViz._structure_atoms(root)
    ti = _ti_atoms(atoms)
    built, _skipped = CrystalViz._build_polyhedra_objects(root, atoms, ti, scene)
    bpy.context.view_layer.update()
    return scene, root, atoms, ti, built


def poly_p1(root):
    return {
        int(obj.get("cgn_p1_index"))
        for obj in root.objects
        if obj.get("cgn_role") == "polyhedron"
    }


def scaffold_positions(root):
    positions = []
    for obj in root.objects:
        if obj.get("cgn_role") == "polyhedron":
            positions.extend(tuple(vertex.co) for vertex in obj.data.vertices)
    return positions


def near(point, other, eps=1e-5):
    return all(abs(point[axis] - other[axis]) <= eps for axis in range(3))


def quad_of_atom(root, atom_index):
    instances = json.loads(root.get("cgn_instances", "[]"))
    base = atom_index * 4
    return (
        int(instances[base]),
        int(instances[base + 1]),
        int(instances[base + 2]),
        int(instances[base + 3]),
    )


def centre_shells(root, scene, atoms, centres):
    """Absolute ligand quads around each centre image (ligands() is already
    positioned around the centre's actual cart)."""
    search = CrystalViz._polyhedron_search(root, scene)
    shells = {}
    for centre in centres:
        atom = atoms[centre]
        shells[centre] = {
            (site, tx, ty, tz)
            for _pos, site, (tx, ty, tz) in search.ligands(
                atom["cart"], str(atom.get("element", "X")).upper()
            )
        }
    return shells


def simulate_delete(root, scene, atom_index, removed_quad):
    """Mirror the delete operator's data path and call _remap_polyhedra."""
    instances = json.loads(root.get("cgn_instances", "[]"))
    new_instances = [
        value
        for offset, value in enumerate(instances)
        if (offset - offset % 4) != atom_index * 4
    ]
    sites, _inst, cell_matrix, initial_count = CrystalViz._structure_tables(root)
    new_atoms = CrystalViz._materialize_atoms(
        sites, cell_matrix, new_instances, initial_count
    )
    root["cgn_instances"] = json.dumps(new_instances)
    root["cgn_initial_count"] = initial_count
    CrystalViz._remap_polyhedra(root, [atom_index], new_atoms, scene, {removed_quad})
    bpy.context.view_layer.update()
    return new_atoms


# ---- Scenario A: delete a LIGAND vertex (single image per site) -------------
print("\n=== A: delete a ligand atom sitting on a polyhedron vertex ===")
scene, root, atoms, ti, built = setup_fill_polyhedra()
before = poly_p1(root)
check("A control: Ti polyhedra were built", built > 0 and len(before) == len(ti),
      f"built={built} sites={len(before)} centres={len(ti)}")

shells = centre_shells(root, scene, atoms, ti)
ligand_use = {}
for centre, quads in shells.items():
    for quad in quads:
        ligand_use.setdefault(quad, []).append(centre)

ligand_atom = None
for index, atom in enumerate(atoms):
    if str(atom.get("element", "X")).upper() == "O":
        if quad_of_atom(root, index) in ligand_use:
            ligand_atom = index
            break
check("A control: found a displayed ligand atom", ligand_atom is not None)

if ligand_atom is not None:
    removed_quad = quad_of_atom(root, ligand_atom)
    affected = {int(atoms[c].get("p1_index")) for c in ligand_use[removed_quad]}
    check("A control: deleted ligand is a vertex of at least one polyhedron",
          bool(affected), f"affected p1 sites={sorted(affected)}")
    simulate_delete(root, scene, ligand_atom, removed_quad)
    after = poly_p1(root)
    check("A: polyhedra that used the deleted ligand are removed",
          not (affected & after), f"affected={sorted(affected)} remaining={sorted(after)}")
    check("A: unaffected polyhedra are retained",
          after == before - affected, f"before={sorted(before)} after={sorted(after)}")

# ---- Scenario B: deleting a CENTRE still drops that polyhedron (parity) -----
print("\n=== B: deleting a central atom (parity) ===")
scene, root, atoms, ti, built = setup_fill_polyhedra()
before = poly_p1(root)
centre_atom = ti[0]
centre_site = int(atoms[centre_atom].get("p1_index"))
simulate_delete(root, scene, centre_atom, quad_of_atom(root, centre_atom))
after = poly_p1(root)
check("B: the deleted centre's polyhedron is gone",
      centre_site not in after, f"deleted site={centre_site} remaining={sorted(after)}")
check("B: other polyhedra are retained",
      after == before - {centre_site}, f"before={sorted(before)} after={sorted(after)}")

# ---- Scenario C: delete a ligand vertex AFTER a grow structure --------------
# Regression: the old check only inspected the home-cell representative, so a
# ligand of a translated image survived.  Every affected image (per periodic
# copy) must lose its polyhedron; the rest must survive.
print("\n=== C: delete a ligand vertex after a grow structure (per image) ===")
scene, root, atoms, ti, built = setup_grow_polyhedra(shells=2)
before_positions = scaffold_positions(root)
before_sites = poly_p1(root)
check("C control: multi-image polyhedra were built",
      built > 0 and len(before_positions) > len(before_sites),
      f"images={len(before_positions)} sites={len(before_sites)}")

shells = centre_shells(root, scene, atoms, ti)
ligand_use = {}
for centre, quads in shells.items():
    for quad in quads:
        ligand_use.setdefault(quad, []).append(centre)

translated = {
    tuple(int(v) for v in atoms[c].get("growth_translation", (0, 0, 0)))
    for c in ti
}
ligand_atom = None
for index, atom in enumerate(atoms):
    if str(atom.get("element", "X")).upper() != "O":
        continue
    users = ligand_use.get(quad_of_atom(root, index), [])
    if any(
        tuple(int(v) for v in atoms[c].get("growth_translation", (0, 0, 0)))
        != (0, 0, 0)
        for c in users
    ):
        ligand_atom = index
        break
check("C control: found a ligand of a translated (grown) image",
      ligand_atom is not None, f"translations={sorted(translated)}")

if ligand_atom is not None:
    removed_quad = quad_of_atom(root, ligand_atom)
    affected = ligand_use[removed_quad]
    affected_positions = [CrystalViz._display_point(atoms[c]["cart"]) for c in affected]
    simulate_delete(root, scene, ligand_atom, removed_quad)
    after_positions = scaffold_positions(root)
    check("C: affected images' polyhedra are removed",
          all(
              not any(near(pos, kept) for kept in after_positions)
              for pos in affected_positions
          ),
          f"affected images={len(affected)}")
    check("C: unaffected images' polyhedra are retained",
          len(after_positions) == len(before_positions) - len(affected),
          f"{len(before_positions)}->{len(after_positions)} affected={len(affected)}")

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
