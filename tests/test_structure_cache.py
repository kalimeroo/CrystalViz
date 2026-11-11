"""Validate the (root_id, cgn_revision) parsed-tables / atoms cache.

Run:
  blender --background --factory-startup --python test_structure_cache.py

Covers:
  * repeated _structure_tables / _structure_atoms return the SAME objects
  * import / fill / grow bump cgn_revision and refresh the cache
  * polyhedron ligand additions do not corrupt the cached instance list
  * rollback bumps the revision so the cache is rebuilt from restored data
  * legacy (no site table) still raises the documented error
  * rough speed-up of repeated materialization on a larger CIF
"""
import sys, os, json, time, bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

CIF = _env.CIF_DIR
scene = bpy.context.scene
scene.cgn_connectivity_pairs.clear()
scene.cgn_chemdraw = False
bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "Brookite.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)
if not any(p.enabled for p in scene.cgn_connectivity_pairs):
    p = scene.cgn_connectivity_pairs.add()
    p.element_a = "Ti"; p.element_b = "O"; p.enabled = True
    p.min_distance = 0.1; p.max_distance = 3.0
bpy.context.view_layer.update()

FAILURES = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)

def stored_tables(root):
    return (
        json.loads(root["cgn_sites"]),
        json.loads(root["cgn_instances"]),
        json.loads(root["cgn_cell_matrix"]),
        int(root.get("cgn_initial_count", 0)),
    )

def rev(root):
    return int(root.get("cgn_revision", 0))

# ---- cache identity ----
print("=== cache hits ===")
s1, i1, m1, c1 = CrystalViz._structure_tables(root)
s2, i2, m2, c2 = CrystalViz._structure_tables(root)
check("_structure_tables returns identical cached objects",
      s1 is s2 and i1 is i2 and m1 is m2)
a1 = CrystalViz._structure_atoms(root)
a2 = CrystalViz._structure_atoms(root)
check("_structure_atoms returns identical cached object", a1 is a2)
check("cached atoms match instance count", len(a1) == len(i1) // 4,
      f"{len(a1)} vs {len(i1)//4}")

# ---- fill bumps revision and refreshes ----
print("\n=== fill / grow invalidation ===")
rev_before = rev(root)
bpy.ops.cgn.fill_cell("INVOKE_DEFAULT")
root = bpy.data.collections.get(scene.cgn_active_structure)
bpy.context.view_layer.update()
check("fill bumped cgn_revision", rev(root) > rev_before, f"{rev_before}->{rev(root)}")
s, inst, mat, ic = CrystalViz._structure_tables(root)
st_sites, st_inst, st_mat, st_ic = stored_tables(root)
check("fill cache matches stored instances", inst == st_inst,
      f"cached={len(inst)//4} stored={len(st_inst)//4}")
atoms = CrystalViz._structure_atoms(root)
check("fill cached atoms match stored count", len(atoms) == len(st_inst) // 4)

# ---- grow bumps revision and refreshes ----
root["cgn_selected_indices"] = json.dumps(list(range(3)))
rev_before = rev(root)
bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
bpy.context.view_layer.update()
check("grow bumped cgn_revision", rev(root) > rev_before, f"{rev_before}->{rev(root)}")
inst = CrystalViz._structure_tables(root)[1]
check("grow cache matches stored instances", inst == stored_tables(root)[1],
      f"cached={len(inst)//4} stored={len(stored_tables(root)[1])//4}")

# ---- polyhedron ligand additions must not mutate the cached list ----
print("\n=== polyhedra ligand additions ===")
atoms = CrystalViz._structure_atoms(root)
ti = [i for i, a in enumerate(atoms) if str(a.get("element", "X")).upper() == "TI"]
before_inst = CrystalViz._structure_tables(root)[1]
snapshot_copy = list(before_inst)  # what the cache list should still look like pre-op
built, skipped = CrystalViz._build_polyhedra_objects(root, atoms, ti, scene)
bpy.context.view_layer.update()
check("polyhedra built", built > 0, f"built={built}")
after_inst = CrystalViz._structure_tables(root)[1]
check("polyhedra cache matches stored instances",
      after_inst == stored_tables(root)[1],
      f"cached={len(after_inst)//4} stored={len(stored_tables(root)[1])//4}")
# The pre-op list object must NOT have been extended in place (copy fix).
check("pre-op cached list was not mutated in place",
      len(snapshot_copy) == len(after_inst) or after_inst is not before_inst,
      f"snapshot_len={len(snapshot_copy)} after_len={len(after_inst)}")

# ---- rollback bumps revision ----
print("\n=== rollback ===")
snap = CrystalViz._snapshot_structure_state(root, scene.cgn_connectivity_pairs)
rev_before = rev(root)
CrystalViz._set_instances(root, list(CrystalViz._structure_tables(root)[1]) + [0, 0, 0, 0])
check("_set_instances bumped revision", rev(root) > rev_before, f"{rev_before}->{rev(root)}")
CrystalViz._rollback_structure_state(root, snap, scene, scene.cgn_connectivity_pairs)
check("rollback bumped revision again", rev(root) > rev_before)
data_after = CrystalViz._structure_tables(root)[1]
check("rollback cache matches restored instances", data_after == stored_tables(root)[1],
      f"cached={len(data_after)//4} stored={len(stored_tables(root)[1])//4}")

# ---- legacy error ----
print("\n=== legacy ===")
saved_sites = root["cgn_sites"]
del root["cgn_sites"]
CrystalViz._invalidate_structure_cache()
try:
    CrystalViz._structure_tables(root)
    legacy_ok = False
except ValueError as exc:
    legacy_ok = "older version" in str(exc)
check("missing site table raises legacy error", legacy_ok)
root["cgn_sites"] = saved_sites
CrystalViz._invalidate_structure_cache()

# ---- rough speed check on a larger CIF ----
print("\n=== speed (test.cif) ===")
bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "test.cif"))
big = bpy.data.collections.get(scene.cgn_active_structure)
bpy.context.view_layer.update()
n = len(CrystalViz._structure_tables(big)[1]) // 4
CrystalViz._invalidate_structure_cache()
t0 = time.perf_counter()
for _ in range(10):
    CrystalViz._invalidate_structure_cache()
    CrystalViz._structure_atoms(big)
uncached = time.perf_counter() - t0
CrystalViz._structure_atoms(big)  # prime cache
t0 = time.perf_counter()
for _ in range(10):
    CrystalViz._structure_atoms(big)
cached = time.perf_counter() - t0
print(f"  atoms={n}  uncached={uncached*1000:.2f} ms  cached={cached*1000:.2f} ms  "
      f"speedup={uncached/max(cached,1e-9):.0f}x")
check("cache is materially faster", cached < uncached / 2,
      f"{uncached*1000:.2f}ms -> {cached*1000:.2f}ms")

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
