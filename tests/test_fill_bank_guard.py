"""Fill-limit guards: the bank build and the boundary extension are bounded.

1. _unwrapped_cell_instances must refuse a small cell + wide bond window
   BEFORE allocating the image bank (the entries are
   len(sites) * prod(2*cells[i]+1), not len(sites)).
2. _extend_cell_boundary_instances must raise mid-loop, not after building
   up to 8x the images.
3. Normal fills are unaffected.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import bpy
import CrystalViz

CrystalViz.register()

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


class Pair:
    def __init__(self, a, b, lo, hi):
        self.element_a, self.element_b = a, b
        self.enabled, self.min_distance, self.max_distance = True, lo, hi


# ---- 1. bank estimate ------------------------------------------------
sites = [{"frac": (0.1 * i, 0.2 * i, 0.3 * i), "element": "C"} for i in range(30)]
identity_cell = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]  # 1 A cube
pairs = [Pair("C", "C", 0.1, CrystalViz._MAX_BOND_WINDOW)]
t0 = time.perf_counter()
try:
    CrystalViz._unwrapped_cell_instances(sites, identity_cell, pairs)
    check("bank guard raises", False, "no exception")
except ValueError as exc:
    dt = time.perf_counter() - t0
    check("bank guard raises", True, str(exc)[:70])
    check("bank guard is fast (<0.1 s, no allocation)", dt < 0.1, f"{dt * 1000:.1f} ms")

# same sites, sane window: must still succeed
pairs = [Pair("C", "C", 0.1, 3.0)]
res = CrystalViz._unwrapped_cell_instances(sites, identity_cell, pairs)
check("normal window still returns instances", bool(res), f"{len(res) // 4} atoms")

# ---- 2. boundary extension guard ------------------------------------
corner_sites = [{"frac": (0.0, 0.0, 0.0), "element": "C"} for _ in range(500)]
# 500 distinct corner sites -> up to 4000 images after extension
flat = [v for i in range(500) for v in (i, 0, 0, 0)]
t0 = time.perf_counter()
try:
    CrystalViz._extend_cell_boundary_instances(corner_sites, flat, max_instances=1000)
    check("extension guard raises", False, "no exception")
except ValueError as exc:
    dt = time.perf_counter() - t0
    check("extension guard raises", True, str(exc)[:60])
    check("extension guard is fast (<0.1 s)", dt < 0.1, f"{dt * 1000:.1f} ms")

full = CrystalViz._extend_cell_boundary_instances(corner_sites, [0, 0, 0, 0, 0, 1, 0, 0])
# 8 images for the single base-cell corner atom (its periodic copy at
# translation 1 is no longer extended, so it cannot spawn a second cell).
check("extension without a limit still works", len(full) // 4 == 8, f"{len(full) // 4}")

# ---- 3. real fills ---------------------------------------------------
scene = bpy.context.scene
for cif, _repeats, expect_cell in (
    ("diamond.cif", (1, 1, 1), 18),
    ("test.cif", (1, 1, 1), None),
    ("aromatic.cif", (2, 2, 2), None),
):
    path = os.path.join(_env.CIF_DIR, cif)
    bpy.ops.cgn.import_cif(filepath=path)
    root = bpy.data.collections.get(scene.cgn_active_structure)
    CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))
    if cif == "aromatic.cif":
        result = bpy.ops.cgn.fill_pack(repeat_a=2, repeat_b=2, repeat_c=2)
    else:
        result = bpy.ops.cgn.fill_cell()
    atoms = CrystalViz._structure_atoms(root)
    ok = "FINISHED" in result
    if expect_cell is not None:
        ok = ok and len(atoms) == expect_cell
    check(f"fill {cif} OK", ok, f"result={set(result)} atoms={len(atoms)}")

# ---- 4. the UI cap itself --------------------------------------------
scene.cgn_connectivity_pairs.clear()
item = scene.cgn_connectivity_pairs.add()
item.min_distance = 0.1
item.max_distance = 15.0
check(
    "max bond window is capped at _MAX_BOND_WINDOW",
    abs(item.max_distance - CrystalViz._MAX_BOND_WINDOW) < 1e-6,
    f"stored {item.max_distance:.2f}, cap {CrystalViz._MAX_BOND_WINDOW}",
)
item.min_distance = 15.0
check(
    "min bond window honours the same cap",
    abs(item.min_distance - CrystalViz._MAX_BOND_WINDOW) < 1e-6,
    f"stored {item.min_distance:.2f}",
)

print("TOTAL FAILURES:", len(FAILS))
print("RESULT: FAIL" if FAILS else "RESULT: ALL PASS")
