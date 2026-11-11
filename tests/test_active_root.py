"""_active_root must only return a structure the scene actually contains.

``scene.cgn_active_structure`` stores a name, and a name keeps resolving after
the collection is unlinked from the scene.  Before the fix, operators acted on
that invisible data; now the scene's collection tree is checked.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import bpy
import CrystalViz

CrystalViz.register()

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def atom_count(root):
    return len(json.loads(root.get("cgn_instances", "[]"))) // 4


def run_operator(operator, **kwargs):
    """Call an operator, mapping its ERROR report (RuntimeError) to CANCELLED.

    Returns ``(result, message)`` so the tests can assert on the reported text.
    """
    try:
        return operator(**kwargs), ""
    except RuntimeError as exc:
        return {"CANCELLED"}, str(exc)


scene = bpy.context.scene
bpy.ops.cgn.import_cif(filepath=_env.cif("diamond.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)
check("setup: structure imported and linked", root is not None)
check("linked structure is found", CrystalViz._active_root(scene) is root)

# ---- 1. unlinked collection must not be returned -------------------------
scene.collection.children.unlink(root)
check("collection survives in bpy.data (unlink, not delete)", root.name in bpy.data.collections)
check("unlinked collection is still role=structure",
      root.get("cgn_role") == "structure")
check("unlinked structure is rejected", CrystalViz._active_root(scene) is None)

before = atom_count(root)
result, message = run_operator(bpy.ops.cgn.fill_cell)
check("operator refuses the unlinked structure", "CANCELLED" in result, str(result))
check("message names the unlinked structure",
      "is not linked to this scene" in message, message.strip())
check("no atoms were written to the hidden data", atom_count(root) == before,
      f"{before} -> {atom_count(root)}")

# ---- 2. relinking restores it -------------------------------------------
scene.collection.children.link(root)
check("relinked structure is found again", CrystalViz._active_root(scene) is root)
result, message = run_operator(bpy.ops.cgn.fill_cell)
check("operator works again", "FINISHED" in result, str(result))

# ---- 3. nested linking is still accepted --------------------------------
scene.collection.children.unlink(root)
container = bpy.data.collections.new("CGN_Test_Container")
scene.collection.children.link(container)
container.children.link(root)
check("collection nested one level down is found",
      CrystalViz._active_root(scene) is root)
deeper = bpy.data.collections.new("CGN_Test_Deep")
container.children.link(deeper)
deeper.children.link(root)
container.children.unlink(root)  # keep only the two-levels-down copy
check("collection nested two levels down is found",
      CrystalViz._active_root(scene) is root)

# ---- 4. a different scene must not see it -------------------------------
other = bpy.data.scenes.new("CGN_Test_Other")
other.cgn_active_structure = root.name
check("other scene does not inherit the structure",
      CrystalViz._active_root(other) is None)
check("original scene still resolves it", CrystalViz._active_root(scene) is root)

# ---- 5. deleting the collection is still handled ------------------------
bpy.data.collections.remove(root, do_unlink=True)
check("deleted structure is rejected", CrystalViz._active_root(scene) is None)
result, message = run_operator(bpy.ops.cgn.fill_cell)
check("operator refuses after deletion", "CANCELLED" in result, str(result))
check("deleted structure falls back to the import message",
      "Import a CIF before" in message, message.strip())

print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
