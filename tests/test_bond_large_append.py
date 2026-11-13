"""Stress/regression: bond segment appends on a large mesh.

Blender 5.2 crashes on Mesh.edges.add() over an already-populated mesh, so the
add-on appends through bmesh.  This test exercises that on a 100k-edge bond
mesh: geometry is added, material slots survive, and duplicate folding works.

Run:
  blender --background --factory-startup --python test_bond_large_append.py
"""

import sys
import time

import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

FAILURES = []
N_EXISTING = 100000


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_bond_object():
    vertices = []
    edges = []
    for i in range(N_EXISTING):
        base = len(vertices)
        x = i * 2.0
        vertices.extend(((x, 0.0, 0.0), (x + 1.0, 0.0, 0.0)))
        edges.append((base, base + 1))
    mesh = bpy.data.meshes.new("CGN_Bonds_Bench_Edges")
    mesh.from_pydata(vertices, edges, [])
    mesh.update()
    material = bpy.data.materials.new("CGN_Bond_Mat_Bench")
    mesh.materials.append(material)
    obj = bpy.data.objects.new("CGN_Bonds_Bench", mesh)
    obj["cgn_role"] = "bonds"
    bpy.context.scene.collection.objects.link(obj)
    return obj


obj = make_bond_object()
mesh = obj.data
check("initial mesh built", len(mesh.edges) == N_EXISTING, f"{len(mesh.edges)} edges")

start = time.perf_counter()
added = CrystalViz._append_bond_segments(
    obj,
    [((1e6 + i * 2.0, 5.0, 0.0), (1e6 + i * 2.0 + 1.0, 5.0, 0.0)) for i in range(1000)],
)
elapsed = time.perf_counter() - start
check("append reports 1000 new segments", added == 1000, f"added={added}")
check("mesh grew to 101000 edges", len(mesh.edges) == N_EXISTING + 1000, f"{len(mesh.edges)}")
check(
    "mesh has two vertices per segment",
    len(mesh.vertices) == (N_EXISTING + 1000) * 2,
    f"{len(mesh.vertices)}",
)
check("material slot survived the append", len(mesh.materials) == 1, f"{len(mesh.materials)}")
print(f"  append of 1000 segments into {N_EXISTING} existing edges: {elapsed * 1000:.1f} ms")

# Dedupe path: re-appending the same 50 segments must add nothing.
repeated = [((1e6 + i * 2.0, 5.0, 0.0), (1e6 + i * 2.0 + 1.0, 5.0, 0.0)) for i in range(50)]
before = len(mesh.edges)
added = CrystalViz._append_bond_segments(obj, repeated, dedupe=True)
check("dedupe skips segments already in the mesh", added == 0, f"added={added}")
check("dedupe left the mesh unchanged", len(mesh.edges) == before, f"{before} -> {len(mesh.edges)}")

# A second fast append with brand-new segments still works.
start = time.perf_counter()
added = CrystalViz._append_bond_segments(
    obj,
    [((9e6 + i * 2.0, 7.0, 0.0), (9e6 + i * 2.0 + 1.0, 7.0, 0.0)) for i in range(500)],
)
elapsed = time.perf_counter() - start
check("second append added 500 segments", added == 500, f"added={added}")
print(f"  second append (500 segments): {elapsed * 1000:.1f} ms")

CrystalViz.unregister()
print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
