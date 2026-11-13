"""Verify the post-extension fill-limit guard added to _fill_structure.

Strategy: pick a structure whose pre-extension count is below a chosen limit L
but whose post-extension count is above it.  With that L, the in-loop check
cannot fire (it only bounds the pre-extension list), so a CANCELLED result
proves the NEW post-extension check works.
"""

import sys

import bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CIF = _env.cif("diamond.cif")
CrystalViz.register()
scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)
CrystalViz._default_connectivity_pairs(scene, CrystalViz._structure_atoms(root))

sites, instances, cell_matrix, _ = CrystalViz._structure_tables(root)

# Reproduce _fill_structure's CELL-mode pre-extension list for pack = 1.
cell_instances = CrystalViz._unwrapped_cell_instances(
    sites, cell_matrix, scene.cgn_connectivity_pairs
)
if cell_instances is None:
    cell_instances = [v for i in range(len(sites)) for v in (i, 0, 0, 0)]
pre = len(cell_instances) // 4
post_list = CrystalViz._extend_cell_boundary_instances(sites, list(cell_instances))
post = len(post_list) // 4
print(f"diamond CELL pack=1: pre-extension={pre} post-extension={post} (x{post / pre:.2f})")

original_limit = CrystalViz._MAX_FILL_ATOMS
try:
    # Limit between pre and post -> only the post-extension check can fire.
    L = (pre + post) // 2
    if not (pre <= L < post):
        L = pre  # fall back; still below post
    CrystalViz._MAX_FILL_ATOMS = L
    print(f"setting _MAX_FILL_ATOMS={L} (pre={pre}, post={post})")
    try:
        result = bpy.ops.cgn.fill_pack(repeat_a=1, repeat_b=1, repeat_c=1)
        cancelled = result == {"CANCELLED"}
    except RuntimeError as exc:
        # operator.report({"ERROR"}) + return CANCELLED surfaces as RuntimeError
        # from bpy.ops in scripts; that is the expected cancellation path.
        print(f"RuntimeError (expected): {exc}")
        result = {"CANCELLED"}
        cancelled = True
    atoms_after = len(CrystalViz._structure_atoms(root))
    print(f"fill result={result} cancelled={cancelled} atoms_after={atoms_after}")
    print(f"CHECK 1 (guard fires)  : {'PASS' if cancelled else 'FAIL'}")
    # Structure must be untouched: still the ASU import, not a filled cell.
    print(f"CHECK 2 (no partial fill): {'PASS' if atoms_after != post else 'FAIL'}")
finally:
    CrystalViz._MAX_FILL_ATOMS = original_limit

# Normal fill with the real limit must still work and match the pre-change count.
result = bpy.ops.cgn.fill_pack(repeat_a=1, repeat_b=1, repeat_c=1)
atoms_after = len(CrystalViz._structure_atoms(root))
print(f"normal fill result={result} atoms={atoms_after} expected={post}")
regression_ok = result == {"FINISHED"} and atoms_after == post
print(f"CHECK 3 (no regression)  : {'PASS' if regression_ok else 'FAIL'}")
