"""Regression: the growth-selection highlight must be reused, not rebuilt.

Code review flagged ``_update_growth_selection`` for JSON-dumping the whole
selection and tearing down/recreating the highlight mesh + Geometry-Nodes
object on every click/box-selection.  The fix:

  * ``_update_growth_selection`` persists ``cgn_selected_indices`` and updates
    the highlight; it reuses the existing ``growth_selection`` object's mesh in
    place instead of removing and recreating the object/material/modifier.
  * ``_refresh_growth_selection`` updates only the highlight (no persistence),
    so mid-modal apply events do not re-serialize the selection.

This pins: one highlight object across updates, correct vertex counts, empty
selection removes it, and ``_refresh_growth_selection`` never writes the
persisted selection.

Run:
  blender --background --factory-startup --python test_selection_highlight.py
"""
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


def highlight_objects(root):
    return [obj for obj in root.objects if obj.get("cgn_role") == "growth_selection"]


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.cgn.import_cif(filepath=CIF)
    root = bpy.data.collections.get(scene.cgn_active_structure)
    atoms = CrystalViz._structure_atoms(root)
    total = len(atoms)

    # --- First update: creates one highlight object and persists the selection.
    CrystalViz._update_growth_selection(root, atoms, [0, 2, 1, 3])
    objs = highlight_objects(root)
    check("one highlight object created", len(objs) == 1, str(len(objs)))
    check("selection persisted by update",
          root.get("cgn_selected_indices") == "[0, 1, 2, 3]",
          root.get("cgn_selected_indices"))
    if objs:
        check("highlight vertex count matches selection",
              len(objs[0].data.vertices) == 4, len(objs[0].data.vertices))
    first = objs[0] if objs else None

    # --- A later update reuses the same object (no .001, no orphan).
    CrystalViz._update_growth_selection(root, atoms, list(range(10)))
    objs = highlight_objects(root)
    check("still one highlight object after update", len(objs) == 1, str(len(objs)))
    if first is not None and objs:
        check("same object reused", objs[0] is first, f"{first.name} vs {objs[0].name}")
    check("no suffixed highlight objects",
          all(".001" not in obj.name for obj in objs), [o.name for o in objs])
    check("selection updated in place",
          root.get("cgn_selected_indices") == str(list(range(10))),
          root.get("cgn_selected_indices"))
    if objs:
        check("highlight vertex count follows update",
              len(objs[0].data.vertices) == 10, len(objs[0].data.vertices))

    # --- _refresh_growth_selection updates the mesh but does NOT persist.
    stored_before = root.get("cgn_selected_indices")
    CrystalViz._refresh_growth_selection(root, atoms, [5, 6, 7])
    objs = highlight_objects(root)
    check("refresh still one highlight object", len(objs) == 1, str(len(objs)))
    if objs:
        check("refresh updates vertex count",
              len(objs[0].data.vertices) == 3, len(objs[0].data.vertices))
    check("refresh does not persist selection",
          root.get("cgn_selected_indices") == stored_before,
          root.get("cgn_selected_indices"))

    # --- A refresh that actually changed many vertices (large selection) still
    # results in exactly one object (the reused one) — no accumulation.
    CrystalViz._refresh_growth_selection(root, atoms, list(range(total)))
    objs = highlight_objects(root)
    check("large refresh still one object", len(objs) == 1, str(len(objs)))
    if objs:
        check("large refresh reaches all atoms",
              len(objs[0].data.vertices) == total, f"{len(objs[0].data.vertices)}/{total}")

    # --- Empty selection (via _update_growth_selection) removes the object and
    # persists an empty list.
    CrystalViz._update_growth_selection(root, atoms, [])
    objs = highlight_objects(root)
    check("empty selection removes highlight object", len(objs) == 0, str(len(objs)))
    check("empty selection persists empty", root.get("cgn_selected_indices") == "[]",
          root.get("cgn_selected_indices"))

    # --- Recreate after removal is still a single object.
    CrystalViz._update_growth_selection(root, atoms, [0])
    objs = highlight_objects(root)
    check("recreate after empty is single object", len(objs) == 1, str(len(objs)))
    check("clobbered indices sorted + persisted",
          root.get("cgn_selected_indices") == "[0]", root.get("cgn_selected_indices"))

    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILURE(S)'}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
