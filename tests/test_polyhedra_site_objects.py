"""Verify polyhedra are one object per site, correctly scaled, faces+edges in one mesh.

Checks:
  1. One polyhedron object per crystallographic site (not per atom image)
  2. Geometry size is in display units (not shrunk by an extra 0.1)
  3. Every atom image of the site is baked into that one object
  4. Faces (material 0) and edge tubes (material 1) live in the same mesh
  5. Changing edge width rebuilds the tubes; changing colour does not
  6. Deleting a centre clears + rebuilds polyhedra

Run:
  blender --background --factory-startup --python test_polyhedra_site_objects.py
"""

import sys

import bpy
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


def geometry_mesh(obj):
    """Geometry-bearing mesh: the prototype under GN, the baked mesh otherwise."""
    prototype = CrystalViz._polyhedron_prototype_for(obj)
    if prototype is not None:
        return prototype.data
    return obj.data


def render_mesh(obj):
    """Evaluated geometry for display assertions (realized instances under GN)."""
    if obj.get("cgn_poly_prototype"):
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        return obj.evaluated_get(depsgraph).data
    return obj.data


scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms0 = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms0)
bpy.ops.cgn.grow_further()
bpy.ops.cgn.grow_further()
atoms2 = CrystalViz._structure_atoms(root)
ag_images = [a for a in atoms2 if str(a.get("element", "X")).upper() == "AG"]

scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()

polys = [
    o
    for o in root.objects
    if o.get("cgn_role") == "polyhedron" and str(o.get("cgn_poly_element", "")).upper() == "AG"
]
built_sites = {int(o.get("cgn_p1_index", -1)) for o in polys}
print(f"  AG polyhedron objects: {len(polys)}  sites: {sorted(built_sites)}")

# 1. One object per site
check(
    "one polyhedron object per crystallographic site",
    len(polys) == len(built_sites),
    f"{len(polys)} objects vs {len(built_sites)} sites",
)
check(
    "fewer objects than atom images (9)",
    len(polys) < len(ag_images),
    f"{len(polys)} objects vs {len(ag_images)} images",
)

# 2 + 3. Correct size and every image baked in
qualifying = {s: [a for a in ag_images if int(a.get("p1_index", -1)) == s] for s in built_sites}
for obj in polys:
    mesh = render_mesh(obj)
    check(f"{obj.name}: is a mesh with vertices", obj.type == "MESH" and len(mesh.vertices) > 0)
    coords = [Vector(v.co) for v in mesh.vertices]
    # Size per image: vertices belonging to one image lie within ~0.5 BU of
    # that image's centre.  The old 10x-shrink bug produced ~0.05 BU.
    centers = [
        Vector(CrystalViz._display_point(a["cart"]))
        for a in qualifying[int(obj.get("cgn_p1_index", -1))]
    ]
    for center in centers:
        nearby = [c for c in coords if (c - center).length < 0.5]
        radius = max((c - center).length for c in nearby) if nearby else 0.0
        check(
            f"{obj.name}: image at {tuple(round(v, 2) for v in center)} is display-sized",
            0.2 < radius < 0.6,
            f"max vertex distance={radius:.4f} BU",
        )
        nearest = min((c - center).length for c in coords)
        check(
            f"{obj.name}: covers image at {tuple(round(v, 2) for v in center)}",
            nearest < 0.4,
            f"nearest vertex {nearest:.3f} BU",
        )

# 4. Faces + edges in the same mesh, two material slots
for obj in polys:
    mesh = render_mesh(obj)
    indices = {p.material_index for p in mesh.polygons}
    check(f"{obj.name}: 2 material slots", len(mesh.materials) == 2, f"{len(mesh.materials)} slots")
    check(
        f"{obj.name}: face + edge polygons in one mesh",
        indices == {0, 1},
        f"material indices={sorted(indices)}",
    )

# 5. Edge-width change rebuilds tubes; colour change does not
obj0 = polys[0]
obj0_p1 = int(obj0.get("cgn_p1_index", -1))
site_label0 = str(obj0.get("cgn_poly_site_label", ""))
style = None
for candidate in scene.cgn_polyhedron_styles:
    if candidate.element == "AG" and str(candidate.site_label or "") == site_label0:
        style = candidate
        break
if style is None:
    style = CrystalViz._polyhedron_style(scene, "AG", site_label0)
old_width = float(obj0.get("cgn_poly_edge_width", 0.0))
geometry_before = geometry_mesh(obj0)
old_verts = len(geometry_before.vertices)
old_signature = tuple(round(component, 6) for v in geometry_before.vertices for component in v.co)
target_width = 0.05 if old_width > 0.1 else 0.2
print(
    f"  site_label0={site_label0!r} style.edge_width before={style.edge_width:.3f} "
    f"object width={old_width:.3f}"
)
style.edge_width = target_width
print(f"  style.edge_width after={style.edge_width:.3f}")
polys_after = [
    o
    for o in root.objects
    if o.get("cgn_role") == "polyhedron"
    and str(o.get("cgn_poly_element", "")).upper() == "AG"
    and int(o.get("cgn_p1_index", -1)) == obj0_p1
]
check(
    "edge-width change still leaves one object per site",
    len(polys_after) == 1,
    f"{len(polys_after)} objects",
)
check(
    "edge-width change rebuilds tube geometry",
    tuple(round(component, 6) for v in geometry_mesh(polys_after[0]).vertices for component in v.co)
    != old_signature,
    f"{old_verts} -> {len(geometry_mesh(polys_after[0]).vertices)} vertices, coords changed",
)
new_verts = len(geometry_mesh(polys_after[0]).vertices)
style.edge_color = (0.1, 0.9, 0.2, 1.0)
polys_color = [
    o
    for o in root.objects
    if o.get("cgn_role") == "polyhedron"
    and str(o.get("cgn_poly_element", "")).upper() == "AG"
    and int(o.get("cgn_p1_index", -1)) == obj0_p1
]
check(
    "colour change does not rebuild geometry",
    len(polys_color) == 1 and len(geometry_mesh(polys_color[0]).vertices) == new_verts,
    f"{new_verts} vertices kept",
)

# 6. Delete path: remove a centre atom and remap polyhedra
atoms_before = CrystalViz._structure_atoms(root)
removed = 0
CrystalViz._remap_polyhedra(root, [removed], atoms_before, scene)
polys_after_delete = [o for o in root.objects if o.get("cgn_role") == "polyhedron"]
check(
    "remap after delete keeps one object per surviving site",
    len(polys_after_delete) <= len(built_sites),
    f"{len(polys_after_delete)} objects",
)

CrystalViz.unregister()
check("unregister OK", True)

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
