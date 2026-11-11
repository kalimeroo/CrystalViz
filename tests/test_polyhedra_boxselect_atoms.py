"""Regression: box-selecting several images of one site completes every
polyhedron with its ligand atoms (not just the representative image's shell).

Run:
  blender --background --factory-startup --python test_polyhedra_boxselect_atoms.py
"""
import sys, bpy
from mathutils import Vector

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

CIF = _env.cif("Ag2Cr2O7.cif")
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))

sites, instances, cell_matrix, initial_count = CrystalViz._structure_tables(root)
sites = list(sites)
instances = list(instances)

ag_sites = [
    index for index, site in enumerate(sites)
    if str(site.get("element", "")).upper() == "AG"
]
check("CIF has an Ag site", bool(ag_sites))
ag_site = ag_sites[0]

# Simulate a box selection across several images of the SAME site by adding
# two translated instances of that Ag site (no ligand atoms around them).
existing_quads = {
    (int(instances[o]), int(instances[o + 1]), int(instances[o + 2]), int(instances[o + 3]))
    for o in range(0, len(instances) - len(instances) % 4, 4)
}
for translation in [(1, 0, 0), (0, 1, 0)]:
    quad = (ag_site, *translation)
    if quad not in existing_quads:
        instances.extend(quad)
        existing_quads.add(quad)
CrystalViz._set_instances(root, instances)

atoms = CrystalViz._structure_atoms(root)
centers = [
    index for index, atom in enumerate(atoms)
    if int(atom.get("p1_index", -1)) == ag_site
]
print(f"  selected Ag images: {centers}")

before_positions = {
    tuple(round(v, 4) for v in CrystalViz._display_point(atom["cart"]))
    for atom in atoms
}

built, skipped = CrystalViz._build_polyhedra_objects(
    root, atoms, centers, scene, clear_existing=True
)
print(f"  built={built} skipped={skipped}")
check("polyhedron built for the site", built == 1, f"built={built}")

after = CrystalViz._structure_atoms(root)
after_positions = {
    tuple(round(v, 4) for v in CrystalViz._display_point(atom["cart"]))
    for atom in after
}
added = after_positions - before_positions
print(f"  atoms {len(atoms)}->{len(after)}  added {len(added)}")

obj = next(
    o for o in root.objects
    if o.get("cgn_role") == "polyhedron" and int(o.get("cgn_p1_index", -1)) == ag_site
)
mesh = obj.data
coords = [Vector(v.co) for v in mesh.vertices]
face_vertices = {
    tuple(round(component, 4) for component in coords[index])
    for polygon in mesh.polygons if polygon.material_index == 0
    for index in polygon.vertices
}
print(f"  polyhedron face vertices: {len(face_vertices)}")

missing = [
    position for position in face_vertices
    if min(
        (Vector(position) - Vector(other)).length for other in after_positions
    ) > 0.02
]
check("every polyhedron edge vertex has a real atom after a box selection",
      not missing, f"{len(missing)} vertex position(s) without atom")

# Each selected image must be covered by the site object's baked geometry.
for center_index in centers:
    center = Vector(CrystalViz._display_point(atoms[center_index]["cart"]))
    near = [c for c in coords if (c - center).length < 0.6]
    check(f"image {center_index} has baked polyhedron vertices", bool(near),
          f"{len(near)} vertices near centre")

CrystalViz.unregister()
check("unregister OK", True)
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
