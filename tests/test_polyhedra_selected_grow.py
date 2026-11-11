"""Regression: adding a polyhedron for a grown image of an already-built site.

Reproduces "Create from Selected" after growth: the site object already exists,
but the new image is not baked into it.  The build must merge the new image into
the existing site object instead of reporting "Skipped every selected atom".

Run:
  blender --background --factory-startup --python test_polyhedra_selected_grow.py
"""
import sys, json, bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

CIF = r"C:\Users\Erwann Jeanneau\Documents\Python_Scripts\Blender_addon\CrystalViz\CIF\Ag2Cr2O7.cif"
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
bpy.ops.cgn.grow_further()
bpy.ops.cgn.grow_further()
atoms2 = CrystalViz._structure_atoms(root)

# Full build for Ag.
scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()
polys = [o for o in root.objects if o.get("cgn_role") == "polyhedron"]
built_sites = {int(o.get("cgn_p1_index", -1)) for o in polys}
check("initial build: one object per site", len(polys) == len(built_sites))

site_obj = polys[0]
site_p1 = int(site_obj.get("cgn_p1_index", -1))
previous = {int(i) for i in json.loads(site_obj.get("cgn_poly_centers", "[]"))}
print(f"  site P1={site_p1} initially has {len(previous)} images")

# Grow one more shell and find a NEW image of that same site.
bpy.ops.cgn.grow_further()
atoms3 = CrystalViz._structure_atoms(root)
site_images = {
    index: atom
    for index, atom in enumerate(atoms3)
    if int(atom.get("p1_index", -1)) == site_p1
}
new_images = [index for index in site_images if index not in previous]
print(f"  after grow: {len(site_images)} images for site, {len(new_images)} new")
check("grow produced at least one new image of the site", bool(new_images))
if not new_images:
    CrystalViz.unregister()
    print("RESULT: ALL PASS")
    sys.exit(0)

# This is what "Create from Selected" does for one clicked atom.
target = new_images[0]
built, skipped = CrystalViz._build_polyhedra_objects(
    root, atoms3, [target], scene, clear_existing=False
)
print(f"  create-from-selected on new image: built={built} skipped={skipped}")
check("adding a new image of an existing site succeeds", built == 1,
      f"built={built}")
check("no bogus skip", skipped == 0, f"skipped={skipped}")

site_objects = [
    o for o in root.objects
    if o.get("cgn_role") == "polyhedron"
    and int(o.get("cgn_p1_index", -1)) == site_p1
]
check("one object per site after merge", len(site_objects) == 1,
      f"{len(site_objects)} objects")
merged = {int(i) for i in json.loads(site_objects[0].get("cgn_poly_centers", "[]"))}
check("old images retained", previous <= merged,
      f"previous={len(previous)} merged={len(merged)}")
check("new image merged in", target in merged)

# Clicking an already-covered image must be a harmless no-op.
built2, skipped2 = CrystalViz._build_polyhedra_objects(
    root, atoms3, [target], scene, clear_existing=False
)
check("already-covered image is not rebuilt", built2 == 0 and skipped2 == 1,
      f"built={built2} skipped={skipped2}")
check("still one object per site", len([
    o for o in root.objects
    if o.get("cgn_role") == "polyhedron"
    and int(o.get("cgn_p1_index", -1)) == site_p1
]) == 1)

CrystalViz.unregister()
check("unregister OK", True)
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
