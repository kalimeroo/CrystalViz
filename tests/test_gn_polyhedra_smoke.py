"""Smoke test for the Geometry Nodes polyhedra implementation."""
import sys
import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

CIF = r"C:\Users\Erwann Jeanneau\Documents\Python_Scripts\Blender_addon\CrystalViz\CIF\Ag2Cr2O7.cif"


def p(*a):
    print(*a, flush=True)


scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms0 = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms0)
bpy.ops.cgn.grow_further()
bpy.ops.cgn.grow_further()
atoms = CrystalViz._structure_atoms(root)
p("atoms", len(atoms))

scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()
p("created")

polys = [o for o in root.objects if o.get("cgn_role") == "polyhedron"]
protos = [o for o in root.all_objects if o.get("cgn_role") == "polyhedron_prototype"]
p("scaffolds", [(o.name, len(o.data.vertices), o.get("cgn_poly_prototype")) for o in polys])
p("prototypes", [(o.name, len(o.data.vertices), len(o.data.polygons),
                  [m.name if m else None for m in o.data.materials]) for o in protos])
p("prototype collections", [(c.name, c.hide_viewport, len(c.all_objects))
                            for c in root.children])

bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
for obj in polys:
    ev = obj.evaluated_get(dg)
    indices = sorted({poly.material_index for poly in ev.data.polygons})
    p("eval", obj.name, "verts", len(ev.data.vertices), "polys", len(ev.data.polygons),
      "slots", [m.name if m else None for m in ev.data.materials], "idx", indices)

p("done")
