"""Regression suite for the Geometry Nodes polyhedra implementation.

Polyhedra are now one scaffold object per crystallographic site (a points mesh
with one vertex per image) plus one canonical prototype instanced by the
``CGN_GN_Polyhedra`` node group.  This suite checks the full lifecycle:
creation, evaluated geometry/materials, prototype storage, edge-width and
colour edits, growth merge, element coexistence, clear and apply-to-edit.

Run:
  blender --background --factory-startup --python test_gn_polyhedra.py
"""

import json
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


def evaluated(obj):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return obj.evaluated_get(depsgraph).data


def scaffolds(root, element=None):
    return [
        obj
        for obj in root.objects
        if obj.get("cgn_role") == "polyhedron"
        and (element is None or str(obj.get("cgn_poly_element", "")).upper() == element)
    ]


def prototypes(root, element=None):
    return [
        obj
        for obj in root.all_objects
        if obj.get("cgn_role") == "polyhedron_prototype"
        and (element is None or str(obj.get("cgn_poly_element", "")).upper() == element)
    ]


def vertex_signature(mesh):
    return tuple(round(component, 6) for vertex in mesh.vertices for component in vertex.co)


# --- setup -----------------------------------------------------------------
scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms0 = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms0)
bpy.ops.cgn.grow_further()
bpy.ops.cgn.grow_further()
atoms = CrystalViz._structure_atoms(root)

scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()

ag_scaffolds = scaffolds(root, "AG")
ag_prototypes = prototypes(root, "AG")
built_sites = {int(obj.get("cgn_p1_index", -1)) for obj in ag_scaffolds}
ag_images = [atom for atom in atoms if str(atom.get("element", "X")).upper() == "AG"]
print(
    f"  AG scaffolds={len(ag_scaffolds)} prototypes={len(ag_prototypes)} "
    f"images={len(ag_images)} sites={sorted(built_sites)}"
)

# --- 1. one scaffold + one prototype per site ------------------------------
check(
    "one scaffold object per crystallographic site",
    len(ag_scaffolds) == len(built_sites),
    f"{len(ag_scaffolds)} objects vs {len(built_sites)} sites",
)
check(
    "one prototype per scaffold",
    len(ag_prototypes) == len(ag_scaffolds),
    f"{len(ag_prototypes)} prototypes",
)

# --- 2. scaffold points == images of the site ------------------------------
for obj in ag_scaffolds:
    p1 = int(obj.get("cgn_p1_index", -1))
    images = [atom for atom in ag_images if int(atom.get("p1_index", -1)) == p1]
    centers = [int(index) for index in json.loads(obj.get("cgn_poly_centers", "[]"))]
    check(
        f"{obj.name}: one point per image",
        len(obj.data.vertices) == len(images) == len(centers),
        f"{len(obj.data.vertices)} points / {len(images)} images",
    )
    check(
        f"{obj.name}: points are at the image centres",
        all(
            (
                obj.data.vertices[offset].co
                - CrystalViz.Vector(CrystalViz._display_point(atom["cart"]))
            ).length
            < 1e-6
            for offset, atom in enumerate(
                [atom for atom in ag_images if int(atom.get("p1_index", -1)) == p1]
            )
        ),
    )
    check(
        f"{obj.name}: scaffold is points-only",
        len(obj.data.vertices) > 0 and len(obj.data.polygons) == 0,
    )

# --- 3. evaluated geometry: display-sized, faces + edge tubes --------------
for obj in ag_scaffolds:
    mesh = evaluated(obj)
    indices = {polygon.material_index for polygon in mesh.polygons}
    slots = [material.name if material else None for material in mesh.materials]
    print(
        f"  {obj.name}: evaluated verts={len(mesh.vertices)} polys={len(mesh.polygons)} "
        f"indices={sorted(indices)}"
    )
    check(
        f"{obj.name}: evaluated faces and edge tubes",
        len(mesh.polygons) > 0 and indices == {0, 1},
        f"indices={sorted(indices)}",
    )
    check(
        f"{obj.name}: evaluated mesh keeps two material slots", len(mesh.materials) == 2, f"{slots}"
    )
    p1 = int(obj.get("cgn_p1_index", -1))
    centers = [
        CrystalViz.Vector(CrystalViz._display_point(atom["cart"]))
        for atom in ag_images
        if int(atom.get("p1_index", -1)) == p1
    ]
    for center in centers:
        nearby = [
            (CrystalViz.Vector(vertex.co) - center).length
            for vertex in mesh.vertices
            if (CrystalViz.Vector(vertex.co) - center).length < 0.5
        ]
        radius = max(nearby) if nearby else 0.0
        check(
            f"{obj.name}: image {tuple(round(v, 2) for v in center)} is display-sized",
            0.2 < radius < 0.6,
            f"max distance={radius:.4f} BU",
        )

# --- 4. prototype storage --------------------------------------------------
collection = next(
    (child for child in root.children if child.get("cgn_role") == "polyhedron_prototypes"),
    None,
)
check("prototype child collection exists", collection is not None)
if collection is not None:
    check(
        "prototype collection is hidden from viewport/render",
        collection.hide_viewport and collection.hide_render,
    )
for prototype in ag_prototypes:
    check(
        f"{prototype.name}: lives in the prototype collection",
        prototype.name in collection.objects if collection else False,
    )
    check(
        f"{prototype.name}: stores canonical geometry",
        bool(json.loads(prototype.get("cgn_poly_local_vertices", "[]")))
        and bool(json.loads(prototype.get("cgn_poly_local_edges", "[]"))),
    )
    check(
        f"{prototype.name}: two materials (face, edge)",
        len(prototype.data.materials) == 2
        and "Face_Material" in prototype.data.materials[0].name
        and "Edge_Material" in prototype.data.materials[1].name,
    )
    check(
        f"{prototype.name}: material indices 0/1",
        {polygon.material_index for polygon in prototype.data.polygons} == {0, 1},
    )

# --- 5. edge-width change rebuilds only the prototype ----------------------
obj0 = ag_scaffolds[0]
prototype0 = CrystalViz._polyhedron_prototype_for(obj0)
style = CrystalViz._polyhedron_style(scene, "AG", str(obj0.get("cgn_poly_site_label", "")))
old_width = float(obj0.get("cgn_poly_edge_width", 0.0))
target_width = 0.05 if old_width > 0.1 else 0.2
scaffold_points = len(obj0.data.vertices)
old_signature = vertex_signature(prototype0.data)
old_eval_signature = vertex_signature(evaluated(obj0))
old_proto_name = prototype0.name
style.edge_width = target_width

same_prototype = CrystalViz._polyhedron_prototype_for(obj0)
check(
    "edge-width change keeps the same prototype object",
    same_prototype is not None and same_prototype.name == old_proto_name,
)
check(
    "edge-width change rebuilds the prototype tubes",
    vertex_signature(prototype0.data) != old_signature,
    f"{len(old_signature) // 3} -> {len(prototype0.data.vertices)} verts",
)
check(
    "edge-width change leaves the scaffold points untouched",
    len(obj0.data.vertices) == scaffold_points,
    f"{scaffold_points} points",
)
check(
    "edge-width change updates the evaluated geometry",
    vertex_signature(evaluated(obj0)) != old_eval_signature,
)
check(
    "edge-width change stores the new width on the scaffold",
    abs(float(obj0.get("cgn_poly_edge_width", 0.0)) - target_width) < 1e-6,
    str(obj0.get("cgn_poly_edge_width")),
)

# --- 6. colour change does not rebuild geometry ---------------------------
new_signature = vertex_signature(prototype0.data)
new_eval = len(evaluated(obj0).vertices)
style.edge_color = (0.1, 0.9, 0.2, 1.0)
check("colour change keeps prototype geometry", vertex_signature(prototype0.data) == new_signature)
check("colour change keeps evaluated geometry", len(evaluated(obj0).vertices) == new_eval)
check(
    "colour change reaches the prototype material",
    tuple(round(value, 3) for value in prototype0.data.materials[1].diffuse_color[:3])
    == (0.1, 0.9, 0.2),
    str(tuple(round(value, 3) for value in prototype0.data.materials[1].diffuse_color[:3])),
)

# --- 7. growth merge appends points only -----------------------------------
p1 = int(obj0.get("cgn_p1_index", -1))
previous = {int(index) for index in json.loads(obj0.get("cgn_poly_centers", "[]"))}
bpy.ops.cgn.grow_further()
atoms3 = CrystalViz._structure_atoms(root)
site_images = [index for index, atom in enumerate(atoms3) if int(atom.get("p1_index", -1)) == p1]
new_images = [index for index in site_images if index not in previous]
check("grow produced a new image of the site", bool(new_images))
if new_images:
    built, skipped = CrystalViz._build_polyhedra_objects(
        root, atoms3, [new_images[0]], scene, clear_existing=False
    )
    check(
        "merge reports one built site",
        built == 1 and skipped == 0,
        f"built={built} skipped={skipped}",
    )
    merged_obj = [obj for obj in scaffolds(root, "AG") if int(obj.get("cgn_p1_index", -1)) == p1]
    check("merge keeps one scaffold per site", len(merged_obj) == 1)
    merged = {int(index) for index in json.loads(merged_obj[0].get("cgn_poly_centers", "[]"))}
    check("merge retains old images", previous <= merged)
    check("merge appends the new image", new_images[0] in merged)
    check(
        "merge appends one point",
        len(merged_obj[0].data.vertices) == len(previous) + 1,
        f"{len(merged_obj[0].data.vertices)} points",
    )
    check(
        "merge does not duplicate the prototype", len(prototypes(root, "AG")) == len(ag_prototypes)
    )

# --- 8. multi-element coexistence -----------------------------------------
scene.cgn_polyhedron_settings.element = "CR"
bpy.ops.cgn.create_polyhedra()
check(
    "CR polyhedra coexist with AG",
    bool(scaffolds(root, "CR")) and bool(scaffolds(root, "AG")),
    f"CR={len(scaffolds(root, 'CR'))} AG={len(scaffolds(root, 'AG'))}",
)
cr_prototypes = prototypes(root, "CR")
check(
    "CR prototypes carry CR materials",
    all(
        all(material and "CR" in material.name for material in prototype.data.materials)
        for prototype in cr_prototypes
    ),
)
CrystalViz._clear_polyhedra(root, "CR")
check("clearing CR keeps AG", bool(scaffolds(root, "AG")) and not prototypes(root, "CR"))
check("clearing CR removed CR scaffolds", not scaffolds(root, "CR"))

# --- 9. apply the modifier to get an editable mesh -------------------------
editable = obj0.copy()
editable.data = obj0.data.copy()
root.objects.link(editable)
bpy.context.view_layer.objects.active = editable
editable.select_set(True)
result = bpy.ops.object.modifier_apply(modifier=editable.modifiers[0].name)
check("modifier apply succeeds", "FINISHED" in result, str(result))
check(
    "applied polyhedron is an ordinary editable mesh",
    len(editable.data.polygons) > 0,
    f"{len(editable.data.polygons)} faces",
)
check("applied mesh keeps both materials", len(editable.data.materials) == 2)
CrystalViz._remove_object(editable)

# --- 10. clear everything --------------------------------------------------
CrystalViz._clear_polyhedra(root)
check("clear removes every scaffold", not scaffolds(root))
check("clear removes every prototype", not prototypes(root))
check(
    "clear removes the empty prototype collection",
    not any(child.get("cgn_role") == "polyhedron_prototypes" for child in root.children),
)

CrystalViz.unregister()
check("unregister OK", True)
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
