"""Hide/show the centre→ligand bonds ("spokes") of displayed polyhedra.

The Polyhedra panel's "Hide Bonds Inside Polyhedra" option suppresses the bonds
that radiate from every polyhedron centre to its ligands, so the faces read as
the coordination instead of being crossed by their own bonds.  Only the
displayed polyhedra contribute to the mask: a "Create from Selected" subset
hides its own spokes only, clearing polyhedra restores every bond, and a grow
leaves the bonds of newly added (still unpacked) centres visible.

Run headless:
  blender --background --factory-startup --python test_polyhedra_bond_visibility.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import bpy
import CrystalViz

CrystalViz.register()

CIF = _env.CIF_DIR
PAIRS = (("Ti", "O", 0.1, 3.0), ("O", "O", 0.5, 3.0))

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def import_brookite():
    scene = bpy.context.scene
    scene.cgn_hide_polyhedron_bonds = False
    CrystalViz._cancel_all_debounce()
    scene.cgn_connectivity_pairs.clear()
    scene.cgn_chemdraw = False
    bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "Brookite.cif"))
    for element_a, element_b, low, high in PAIRS:
        pair = scene.cgn_connectivity_pairs.add()
        pair.element_a = element_a
        pair.element_b = element_b
        pair.enabled = True
        pair.min_distance = low
        pair.max_distance = high
    bpy.context.view_layer.update()
    bpy.ops.cgn.fill_cell("INVOKE_DEFAULT")
    bpy.context.view_layer.update()
    return scene


def ti_atoms(atoms):
    return [
        index for index, atom in enumerate(atoms) if str(atom.get("element", "X")).upper() == "TI"
    ]


def set_hidden(scene, hidden):
    scene.cgn_hide_polyhedron_bonds = hidden
    CrystalViz._flush_debounce()
    bpy.context.view_layer.update()


def bond_segments(root, pair=None):
    """Every bond segment as display-space endpoint pairs."""
    segments = []
    for obj in root.objects:
        if obj.get("cgn_role") != "bonds" or obj.get("cgn_aromatic_torus"):
            continue
        if pair is not None and obj.get("cgn_pair") != pair:
            continue
        mesh = obj.data
        for edge in mesh.edges:
            first = mesh.vertices[edge.vertices[0]].co
            second = mesh.vertices[edge.vertices[1]].co
            segments.append(((first.x, first.y, first.z), (second.x, second.y, second.z)))
    return segments


def segment_key(segment):
    return tuple(sorted(tuple(round(value, 5) for value in point) for point in segment))


def centre_points(atoms, indices):
    return [
        tuple(float(value) for value in CrystalViz._display_point(atoms[index]["cart"]))
        for index in indices
    ]


def displayed_centre_points(root, atoms):
    indices = []
    for obj in root.objects:
        if obj.get("cgn_role") != "polyhedron":
            continue
        indices.extend(int(index) for index in json.loads(obj.get("cgn_poly_centers", "[]")))
    return centre_points(atoms, indices)


def near(point, other, eps=1e-5):
    return all(abs(point[axis] - other[axis]) <= eps for axis in range(3))


def on_any_point(segment, points):
    return any(near(point, segment[0]) or near(point, segment[1]) for point in points)


def build_polyhedra(root, scene, atoms, centres=None):
    if centres is None:
        centres = ti_atoms(atoms)
    built, _skipped = CrystalViz._build_polyhedra_objects(root, atoms, centres, scene)
    bpy.context.view_layer.update()
    return built


def quad_of_atom(root, atom_index):
    instances = json.loads(root.get("cgn_instances", "[]"))
    base = atom_index * 4
    return tuple(int(instances[base + axis]) for axis in range(4))


# ---- A: the toggle hides exactly the spokes and restores them --------------
print("\n=== A: toggle hides centre→ligand bonds only ===")
scene = import_brookite()
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms = CrystalViz._structure_atoms(root)
built = build_polyhedra(root, scene, atoms)
atoms = CrystalViz._structure_atoms(root)
ti = ti_atoms(atoms)
centres = centre_points(atoms, ti)

baseline = bond_segments(root)
baseline_o_ti = bond_segments(root, "O|TI")
baseline_o_o = bond_segments(root, "O|O")
check(
    "A control: polyhedra and both bond pairs exist",
    built > 0 and len(baseline_o_ti) > 0 and len(baseline_o_o) > 0,
    f"built={built} O|TI={len(baseline_o_ti)} O|O={len(baseline_o_o)}",
)
check(
    "A control: every Ti–O bond is a centre spoke",
    all(on_any_point(segment, centres) for segment in baseline_o_ti),
    f"spokes={len(baseline_o_ti)} centres={len(centres)}",
)

set_hidden(scene, True)
hidden_o_ti = bond_segments(root, "O|TI")
hidden_o_o = bond_segments(root, "O|O")
check(
    "A: no bond touches a polyhedron centre anymore",
    not any(on_any_point(segment, centres) for segment in bond_segments(root)),
)
check("A: the centre–ligand pair is gone entirely", not hidden_o_ti, f"O|TI={len(hidden_o_ti)}")
check(
    "A: non-centre bonds are untouched",
    sorted(map(segment_key, baseline_o_o)) == sorted(map(segment_key, hidden_o_o)),
    f"O|O={len(hidden_o_o)}",
)

set_hidden(scene, False)
restored = bond_segments(root)
check(
    "A: unticking restores the exact pre-toggle bond set",
    sorted(map(segment_key, baseline)) == sorted(map(segment_key, restored)),
    f"{len(baseline)} -> {len(restored)}",
)

# ---- B: only displayed polyhedra mask their own spokes ---------------------
print("\n=== B: a subset of polyhedra masks only its own spokes ===")
CrystalViz._clear_polyhedra(root)
CrystalViz._rebuild_bonds(scene)
bpy.context.view_layer.update()
build_polyhedra(root, scene, atoms, [ti[0]])
subset_centres = centre_points(atoms, [ti[0]])
set_hidden(scene, True)
subset_o_ti = bond_segments(root, "O|TI")
check(
    "B: the built centre's spokes are hidden",
    not any(on_any_point(segment, subset_centres) for segment in subset_o_ti),
    f"O|TI={len(subset_o_ti)}",
)
check(
    "B: every other Ti–O bond stays visible",
    len(subset_o_ti) > 0
    and len(subset_o_ti)
    == sum(1 for segment in baseline_o_ti if not on_any_point(segment, subset_centres)),
    f"visible={len(subset_o_ti)}",
)

# ---- C: clearing polyhedra while hidden restores the bonds -----------------
print("\n=== C: clearing polyhedra restores the spokes ===")
bpy.ops.cgn.clear_polyhedra()
bpy.context.view_layer.update()
cleared = bond_segments(root)
check(
    "C: clearing while hidden brings every bond back",
    sorted(map(segment_key, cleared)) == sorted(map(segment_key, baseline)),
    f"{len(baseline)} -> {len(cleared)}",
)

# ---- D: enabling the option before building polyhedra ----------------------
print("\n=== D: created-after-toggle polyhedra are masked immediately ===")
scene = import_brookite()
root = bpy.data.collections.get(scene.cgn_active_structure)
scene.cgn_hide_polyhedron_bonds = True
CrystalViz._cancel_all_debounce()
atoms = CrystalViz._structure_atoms(root)
build_polyhedra(root, scene, atoms)
atoms = CrystalViz._structure_atoms(root)
created_centres = centre_points(atoms, ti_atoms(atoms))
check(
    "D: no spoke is drawn right after creation",
    not any(on_any_point(segment, created_centres) for segment in bond_segments(root)),
    f"segments={len(bond_segments(root))}",
)
check("D: the ligand–ligand bonds are still drawn", len(bond_segments(root, "O|O")) > 0)

# ---- E: grow keeps hiding only the existing centres ------------------------
print("\n=== E: grown centres (without polyhedra) keep their bonds ===")
bpy.ops.cgn.grow_further()
bpy.context.view_layer.update()
atoms = CrystalViz._structure_atoms(root)
grown_centres = displayed_centre_points(root, atoms)
grown_o_ti = bond_segments(root, "O|TI")
check(
    "E: no bond touches an existing polyhedron centre",
    not any(on_any_point(segment, grown_centres) for segment in bond_segments(root)),
    f"centres={len(grown_centres)}",
)
check(
    "E: grown Ti atoms keep their bonds (no polyhedron yet)",
    any(not on_any_point(segment, grown_centres) for segment in grown_o_ti),
    f"O|TI={len(grown_o_ti)}",
)

# ---- F: deleting a ligand re-derives the mask from the rebuilt polyhedra ---
print("\n=== F: ligand deletion while hidden keeps the mask in sync ===")
scene = import_brookite()
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms = CrystalViz._structure_atoms(root)
build_polyhedra(root, scene, atoms)
atoms = CrystalViz._structure_atoms(root)
set_hidden(scene, True)

search = CrystalViz._polyhedron_search(root, scene)
quad_of = {index: quad_of_atom(root, index) for index in range(len(atoms))}
ligand_use = {}
for centre in ti_atoms(atoms):
    atom = atoms[centre]
    for _pos, site_index, (tx, ty, tz) in search.ligands(
        atom["cart"], str(atom.get("element", "X")).upper()
    ):
        ligand_use.setdefault((site_index, tx, ty, tz), []).append(centre)

ligand_atom = None
for index, atom in enumerate(atoms):
    if str(atom.get("element", "X")).upper() != "O":
        continue
    users = ligand_use.get(quad_of[index], [])
    if len(users) == 1:
        ligand_atom = index
        break
affected_centre = ligand_use[quad_of[ligand_atom]][0] if ligand_atom is not None else None
check(
    "F control: found a ligand of exactly one centre",
    ligand_atom is not None,
    f"ligand={ligand_atom} centre={affected_centre}",
)

if ligand_atom is not None:
    removed_quad = quad_of[ligand_atom]
    instances = json.loads(root.get("cgn_instances", "[]"))
    new_instances = [
        value for offset, value in enumerate(instances) if (offset - offset % 4) != ligand_atom * 4
    ]
    sites, _inst, cell_matrix, initial_count = CrystalViz._structure_tables(root)
    new_atoms = CrystalViz._materialize_atoms(sites, cell_matrix, new_instances, initial_count)
    CrystalViz._set_instances(root, new_instances)
    root["cgn_initial_count"] = initial_count
    CrystalViz._rebuild_atom_objects(root, new_atoms)
    CrystalViz._remap_polyhedra(root, [ligand_atom], new_atoms, scene, {removed_quad})
    CrystalViz._bond_objects(root, new_atoms, scene.cgn_connectivity_pairs)
    bpy.context.view_layer.update()

    current = displayed_centre_points(root, new_atoms)
    check(
        "F: the mask follows the rebuilt polyhedra (no stale index)",
        not any(on_any_point(segment, current) for segment in bond_segments(root)),
        f"centres={len(current)}",
    )
    dropped = all(
        affected_centre
        not in [int(index) for index in json.loads(obj.get("cgn_poly_centers", "[]"))]
        for obj in root.objects
        if obj.get("cgn_role") == "polyhedron"
    )
    # The affected centre's polyhedron was dropped, so its spokes are visible.
    centre_position = centre_points(new_atoms, [affected_centre])[0]
    centre_spokes = [
        segment
        for segment in bond_segments(root, "O|TI")
        if near(segment[0], centre_position) or near(segment[1], centre_position)
    ]
    check(
        "F: the dropped polyhedron's spokes come back",
        dropped and bool(centre_spokes),
        f"dropped={dropped} spokes={len(centre_spokes)}",
    )

# ---- G: the option default -------------------------------------------------
print("\n=== G: default off ===")
probe_scene = bpy.data.scenes.new("CGN_Visibility_Probe")
check(
    "G: the option defaults to off on a fresh scene",
    probe_scene.cgn_hide_polyhedron_bonds is False,
    f"default={probe_scene.cgn_hide_polyhedron_bonds}",
)
bpy.data.scenes.remove(probe_scene)

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
