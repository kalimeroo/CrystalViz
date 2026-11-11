"""Material helpers must not assume a node tree / Principled BSDF exists.

Blender 5.2 deprecates ``Material.use_nodes`` and always builds a node tree,
so the original AttributeError (``material.node_tree`` is None) cannot be
reproduced here any more.  The add-on still dereferenced it unguarded, which
breaks for legacy/appended material data and for any material whose tree has
no Principled BSDF node.  This suite covers:

  * ``_principled_shader`` with a node-less material (duck-typed fixture,
    because Blender refuses to create one),
  * the three call sites with a tree that has no Principled BSDF node,
  * the call sites when the material does not exist at all,
  * a real import, which must still colour its atoms.
"""
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


def base_color(material):
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is None:
        return None
    return tuple(round(c, 4) for c in shader.inputs["Base Color"].default_value)


class HeadlessMaterial:
    """Stand-in for a material datablock whose ``node_tree`` is None."""

    def __init__(self):
        self.node_tree = None
        self.use_nodes = False
        self.enabled = []

    def __setattr__(self, name, value):
        if name == "use_nodes" and value and self.node_tree is None:
            self.enabled.append(value)
        object.__setattr__(self, name, value)


# ---- 1. the helper itself ------------------------------------------------
headless = HeadlessMaterial()
result = CrystalViz._principled_shader(headless)
check("helper returns None for a node-less material", result is None)
check("helper re-enables nodes on a node-less material", headless.use_nodes is True)
check("helper tolerates None", CrystalViz._principled_shader(None) is None)

# ---- 2. a tree without a Principled BSDF node ----------------------------
def de_principled(name):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in [n for n in nodes if n.type == "BSDF_PRINCIPLED"]:
        nodes.remove(node)
    return material


de_principled("CGN_Mat_C")
check("fixture: no Principled node left",
      bpy.data.materials["CGN_Mat_C"].node_tree.nodes.get("Principled BSDF") is None)
try:
    material = CrystalViz._make_material("CGN_Mat_C", (0.9, 0.1, 0.05, 1.0))
    ok = material is bpy.data.materials["CGN_Mat_C"]
except Exception as exc:  # noqa: BLE001 - any crash is a failure here
    ok = False
    print("   ", repr(exc))
check("_make_material survives a tree without Principled BSDF", ok)
check("_make_material still sets diffuse_color",
      tuple(round(c, 4) for c in bpy.data.materials["CGN_Mat_C"].diffuse_color)
      == (0.9, 0.1, 0.05, 1.0))

style = bpy.context.scene.cgn_atom_styles.add()
style.element = "C"
style.color = (0.1, 0.8, 0.2, 1.0)
try:
    CrystalViz._update_atom_style(style, bpy.context)
    ok = True
except Exception as exc:  # noqa: BLE001
    ok = False
    print("   ", repr(exc))
check("_update_atom_style survives a tree without Principled BSDF", ok)

de_principled("CGN_ORTEP_Inner_Faces")
de_principled("CGN_ORTEP_Ellipse_Lines")
try:
    CrystalViz._update_ortep_materials(bpy.context.scene, bpy.context)
    ok = True
except Exception as exc:  # noqa: BLE001
    ok = False
    print("   ", repr(exc))
check("_update_ortep_materials survives trees without Principled BSDF", ok)

# ---- 3. missing material datablocks are ignored --------------------------
for name in ("CGN_Mat_Zz", "CGN_ORTEP_Inner_Faces", "CGN_ORTEP_Ellipse_Lines"):
    existing = bpy.data.materials.get(name)
    if existing is not None:
        bpy.data.materials.remove(existing)
style.element = "Zz"
try:
    CrystalViz._update_atom_style(style, bpy.context)
    CrystalViz._update_ortep_materials(bpy.context.scene, bpy.context)
    ok = True
except Exception as exc:  # noqa: BLE001
    ok = False
    print("   ", repr(exc))
check("missing materials are skipped, not fatal", ok)

# ---- 4. a real import still colours its atoms ----------------------------
bpy.context.scene.cgn_atom_styles.clear()
for material in list(bpy.data.materials):
    if material.name.startswith("CGN_Mat_"):
        bpy.data.materials.remove(material)

bpy.ops.cgn.import_cif(filepath=_env.cif("diamond.cif"))
carbon = bpy.data.materials.get("CGN_Mat_C")
check("import created the carbon material", carbon is not None)
check("imported material has a node tree",
      carbon is not None and carbon.node_tree is not None)
if carbon is not None and carbon.node_tree is not None:
    expected = tuple(round(c, 4) for c in CrystalViz._ELEMENT_COLORS["C"])
    check("imported material carries the element colour",
          base_color(carbon) == expected, f"{base_color(carbon)} vs {expected}")

print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
