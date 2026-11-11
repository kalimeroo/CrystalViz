"""End-to-end test: one polyhedron per crystallographic site after grow.

Scenario from the bug report:
  fill/pack Ag2Cr2O7.cif -> add polyhedra: 2 Ag sites -> 2 polyhedra (OK)
  grow BEFORE adding polyhedra: many Ag images -> must still be ONE polyhedron
  per crystallographic site (not one object per periodic image).

Run:
  blender --background --factory-startup --python test_polyhedra_one_per_site.py
"""
import sys, json, bpy

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
# Import
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
check("structure imported", root is not None)

# Default connectivity windows so Ag qualifies as a center element
atoms0 = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms0)

ag_sites_before = {
    int(a.get("p1_index", -1))
    for a in atoms0
    if str(a.get("element", "X")).upper() == "AG"
}
print(f"  before grow: Ag atoms={len([a for a in atoms0 if str(a.get('element','X')).upper()=='AG'])}  unique Ag P1 sites={len(ag_sites_before)}")

# Grow one shell from all atoms
bpy.ops.cgn.grow_further()
atoms1 = CrystalViz._structure_atoms(root)
ag_images_after = [a for a in atoms1 if str(a.get("element", "X")).upper() == "AG"]
ag_sites_after = {int(a.get("p1_index", -1)) for a in ag_images_after}
print(f"  after grow: Ag images={len(ag_images_after)}  unique Ag P1 sites={len(ag_sites_after)}")
check("grow produced more Ag images than unique sites",
      len(ag_images_after) > len(ag_sites_after),
      f"{len(ag_images_after)} images vs {len(ag_sites_after)} sites")

# Create polyhedra for Ag (exercises the dedup fix in the operator)
scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()

ag_faces = [
    o for o in root.objects
    if o.get("cgn_role") == "polyhedron_face"
    and str(o.get("cgn_poly_element", "")).upper() == "AG"
]
print(f"  polyhedron_face objects for Ag: {len(ag_faces)}")
check("no more polyhedra than unique Ag sites (one per site, not per image)",
      len(ag_faces) <= len(ag_sites_after),
      f"{len(ag_faces)} faces vs {len(ag_sites_after)} sites")
check("strictly fewer polyhedra than Ag images (dedup actually worked)",
      len(ag_faces) < len(ag_images_after),
      f"{len(ag_faces)} faces vs {len(ag_images_after)} images")

# Also verify each face object's poly_map references exactly one center (one site)
for face in ag_faces:
    pmap = json.loads(face.get("cgn_poly_map", "{}") or "{}")
    check(f"poly_map for {face.name} has one center", len(pmap) == 1,
          f"{len(pmap)} centers")

CrystalViz.unregister()
check("unregister OK", True)

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
