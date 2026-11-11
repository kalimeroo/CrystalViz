"""Build a GN-polyhedra scene and save it for the persistence tests.

Run order matters: `test_gn_persist_build.py` -> `test_gn_persist_check.py` ->
`test_gn_persist_export.py` (alphabetical, as `run_all.ps1` does it).  All three
share the file below through the OS temp directory.
"""
import os
import sys
import tempfile

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

CIF = _env.cif("Ag2Cr2O7.cif")
BLEND = os.path.join(tempfile.gettempdir(), "crystalviz_gn_polyhedra.blend")

scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
atoms = CrystalViz._structure_atoms(root)
CrystalViz._default_connectivity_pairs(scene, atoms)
bpy.ops.cgn.grow_further()
bpy.ops.cgn.grow_further()
scene.cgn_polyhedron_settings.element = "AG"
bpy.ops.cgn.create_polyhedra()

polys = [o for o in root.objects if o.get("cgn_role") == "polyhedron"]
protos = [o for o in root.all_objects if o.get("cgn_role") == "polyhedron_prototype"]
print("BUILD scaffolds", len(polys), "prototypes", len(protos), flush=True)
print("BUILD blend", BLEND, flush=True)
bpy.ops.wm.save_as_mainfile(filepath=BLEND)
print("BUILD done", flush=True)
