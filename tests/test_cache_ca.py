"""Measure the structure-cache win on the large Ca_autored.cif.

Run:
  blender --background --factory-startup --python test_cache_ca.py
"""

import sys
import time

import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

CIF = _env.cif("Ca_autored.cif")
scene = bpy.context.scene
scene.cgn_connectivity_pairs.clear()

t0 = time.perf_counter()
bpy.ops.cgn.import_cif(filepath=CIF)
import_s = time.perf_counter() - t0
root = bpy.data.collections.get(scene.cgn_active_structure)
n = CrystalViz._structure_atom_count(root)

CrystalViz._invalidate_structure_cache()
t0 = time.perf_counter()
for _ in range(5):
    CrystalViz._invalidate_structure_cache()
    CrystalViz._structure_atoms(root)
uncached = (time.perf_counter() - t0) / 5
CrystalViz._structure_atoms(root)
t0 = time.perf_counter()
for _ in range(50):
    CrystalViz._structure_atoms(root)
cached = (time.perf_counter() - t0) / 50

print(f"import={import_s * 1000:.0f} ms  atoms={n}")
print(
    f"_structure_atoms: uncached={uncached * 1000:.1f} ms  cached={cached * 1000:.2f} ms  "
    f"speedup={uncached / max(cached, 1e-9):.0f}x"
)

ok = n > 0 and cached < uncached / 5
CrystalViz.unregister()
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
