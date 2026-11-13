"""Verify polyhedron collection instancing: one empty per atom, shared prototype per site.

After grow, check that:
  - One empty per qualifying-site image (not 2 objects per atom)
  - Polyhedra at correct positions (DISPLAY_SCALE applied)
  - One prototype collection per site (face+edge, parented)
  - Shared mesh/curve data blocks (one per site)

Run:
  blender --background --factory-startup --python test_polyhedra_instances.py
"""

import sys

import bpy

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
atoms0 = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms0)

# Grow twice
bpy.ops.cgn.grow_further()
bpy.ops.cgn.grow_further()
atoms2 = CrystalViz._structure_atoms(root)
ag_images = [a for a in atoms2 if str(a.get("element", "X")).upper() == "AG"]

# Create polyhedra for Ag
scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()

# Instance empties in root (one per qualifying-site image)
ag_instances = [
    o
    for o in root.objects
    if o.get("cgn_role") == "polyhedron_instance"
    and str(o.get("cgn_poly_element", "")).upper() == "AG"
]
built_sites = {int(inst.get("cgn_p1_index", -1)) for inst in ag_instances}
print(f"  Built polyhedra for Ag sites: {sorted(built_sites)}")
print(f"  Instance empties: {len(ag_instances)}")

# For each built site, check ALL its images have a polyhedron at their position
all_ok = True
for p1_index in sorted(built_sites):
    site_images = [a for a in ag_images if int(a.get("p1_index", -1)) == p1_index]
    site_positions = {
        tuple(round(v * 0.1, 4) for v in a["cart"])  # DISPLAY_SCALE applied
        for a in site_images
    }
    site_instance_positions = {
        tuple(round(v, 4) for v in o.matrix_world.translation)
        for o in ag_instances
        if int(o.get("cgn_p1_index", -1)) == p1_index
    }
    missing = site_positions - site_instance_positions
    extra = site_instance_positions - site_positions
    print(
        f"  Site P1={p1_index}: {len(site_images)} images, "
        f"{len(site_instance_positions)} instances, "
        f"{len(missing)} missing, {len(extra)} extra"
    )
    if missing or extra:
        all_ok = False

check("every qualifying-site image has a polyhedron at its position (correct scale)", all_ok)

# One instance per qualifying-site image
qualifying_images = sum(1 for a in ag_images if int(a.get("p1_index", -1)) in built_sites)
check(
    "instance empties == qualifying-site images",
    len(ag_instances) == qualifying_images,
    f"{len(ag_instances)} instances vs {qualifying_images} qualifying images",
)

# Every instance is an empty with COLLECTION instancing
all_empties = all(
    inst.type == "EMPTY" and inst.instance_type == "COLLECTION" for inst in ag_instances
)
check("every object is an empty with COLLECTION instancing", all_empties)

# Prototype collections: one per site, each with face+edge (parented)
proto_coll_names = {
    inst.get("cgn_poly_proto_collection")
    for inst in ag_instances
    if inst.get("cgn_poly_proto_collection")
}
check(
    "one prototype collection per site",
    len(proto_coll_names) == len(built_sites),
    f"{len(proto_coll_names)} proto collections vs {len(built_sites)} sites",
)

for coll_name in proto_coll_names:
    coll = bpy.data.collections.get(coll_name)
    check(f"prototype collection {coll_name} exists", coll is not None)
    if coll:
        objs = list(coll.objects)
        faces = [o for o in objs if o.get("cgn_role") == "polyhedron_face"]
        edges = [o for o in objs if o.get("cgn_role") == "polyhedron_edge"]
        check(
            f"proto {coll_name}: 1 face + 1 edge",
            len(faces) == 1 and len(edges) == 1,
            f"{len(faces)} faces, {len(edges)} edges",
        )
        if faces and edges:
            check(f"proto {coll_name}: edge parented to face", edges[0].parent is faces[0])

# Shared mesh data: one block per site
mesh_blocks = set()
for coll_name in proto_coll_names:
    coll = bpy.data.collections.get(coll_name)
    if coll:
        for o in coll.objects:
            if o.get("cgn_role") == "polyhedron_face" and o.data:
                mesh_blocks.add(o.data)
check(
    "one mesh data block per site (shared data)",
    len(mesh_blocks) == len(built_sites),
    f"{len(mesh_blocks)} mesh blocks vs {len(built_sites)} sites",
)

# Object count far below 2 per atom
check(
    "fewer objects than 2 per Ag image (9 images)",
    len(ag_instances) < 18,
    f"{len(ag_instances)} instances",
)

CrystalViz.unregister()
check("unregister OK", True)

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
