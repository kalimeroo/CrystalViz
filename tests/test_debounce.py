"""Validate debounced property updates + in-place refreshes.

Run:
  blender --background --factory-startup --python test_debounce.py

Covers:
  * repeated slider ticks coalesce into ONE pending callback, none run early
  * _flush_debounce applies the pending callback exactly once
  * hydrogen scale updates the H modifier radius in place (same object/mesh)
  * ORTEP ellipse thickness rewrites the template mesh in place and matches a
    freshly rebuilt mesh
  * bond style / spacing rebuild is deferred until the flush
  * render handler flushes pending callbacks; unregister cancels the timer
"""
import sys, os, bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

CIF = _env.cif("aromatic.cif")

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


scene = bpy.context.scene
scene.cgn_connectivity_pairs.clear()
bpy.ops.cgn.import_cif(filepath=CIF)
root = bpy.data.collections.get(scene.cgn_active_structure)

# Give the structure enough connectivity to build bonds.
for a, b, dmax in (("C", "C", 1.8), ("C", "H", 1.3), ("C", "O", 1.6)):
    item = scene.cgn_connectivity_pairs.add()
    item.element_a = a
    item.element_b = b
    item.enabled = True
    item.min_distance = 0.1
    item.max_distance = dmax
CrystalViz._rebuild_bonds(scene)
bpy.context.view_layer.update()

# ------------------------------------------------------------------ debounce
print("\n=== debounce infrastructure ===")
calls = {"n": 0}
original_rebuild = CrystalViz._rebuild_bonds
CrystalViz._rebuild_bonds = lambda sc: calls.__setitem__("n", calls["n"] + 1)
for _ in range(5):
    CrystalViz._update_bond_style(scene, None)
check(
    "5 slider ticks coalesce into one pending callback",
    list(CrystalViz._DEBOUNCE_TIMERS) == ["bond_style"],
    str(list(CrystalViz._DEBOUNCE_TIMERS)),
)
check("no callback ran before the delay", calls["n"] == 0)
check(
    "one debounce timer is registered",
    bpy.app.timers.is_registered(CrystalViz._debounce_timer),
)
CrystalViz._flush_debounce()
check("flush applies the coalesced callback once", calls["n"] == 1, f"calls={calls['n']}")
check("pending queue cleared after flush", CrystalViz._DEBOUNCE_TIMERS == {})
CrystalViz._rebuild_bonds = original_rebuild

# A failing callback must not abort the flush of the others.
ran = {"ok": False}


def bad():
    raise RuntimeError("intentional")


def good():
    ran["ok"] = True


CrystalViz._DEBOUNCE_TIMERS["bad"] = bad
CrystalViz._DEBOUNCE_TIMERS["good"] = good
CrystalViz._flush_debounce()
check("flush survives a failing callback and runs the rest", ran["ok"])

# Non-ELLIPSOID models must not schedule a hydrogen rebuild.
scene.cgn_model = "BALL_STICK"
CrystalViz._DEBOUNCE_TIMERS.clear()
CrystalViz._update_hydrogen_scale(scene, None)
check("BALL_STICK ignores hydrogen scale", CrystalViz._DEBOUNCE_TIMERS == {})

# ------------------------------------------------------------------ ellipsoid
print("\n=== ORTEP / ellipsoid mode ===")
scene.cgn_model = "ELLIPSOID"
bpy.context.view_layer.update()

atom_objects = [o for o in root.objects if o.get("cgn_role") == "atoms"]
check("ellipsoid mode built atom objects", len(atom_objects) > 0, f"n={len(atom_objects)}")
pointers_before = {o.as_pointer() for o in atom_objects}

h_objects = [
    o
    for o in atom_objects
    if str(o.get("cgn_element", "")).upper() == "H"
]
check("H atom object exists in ellipsoid mode", len(h_objects) == 1, f"n={len(h_objects)}")


def radius_input(obj):
    for modifier in obj.modifiers:
        if (
            modifier.type == "NODES"
            and modifier.node_group is not None
            and modifier.node_group.name == CrystalViz._NODE_GROUP_ATOMS
        ):
            ident = CrystalViz._socket_identifier(modifier.node_group, "Radius")
            return modifier.properties.inputs[ident]["value"]
    return None


if h_objects:
    h_obj = h_objects[0]
    h_pointer = h_obj.as_pointer()
    h_mesh_pointer = h_obj.data.as_pointer()
    r_before = radius_input(h_obj)
    scene.cgn_hydrogen_scale = 2.0
    check("hydrogen change is deferred", "hydrogen_scale" in CrystalViz._DEBOUNCE_TIMERS)
    check("H radius untouched before flush", radius_input(h_obj) == r_before)
    CrystalViz._flush_debounce()
    expected = (
        h_obj["cgn_vdw_radius"]
        * CrystalViz._BALL_ATOM_VDW_FACTOR
        * 2.0
        * CrystalViz._DISPLAY_SCALE
    )
    r_after = radius_input(h_obj)
    check("H radius updated after flush", abs(r_after - expected) < 1e-5,
          f"{r_before} -> {r_after}, expected {expected}")
    check("H object reused (no rebuild)", h_obj.as_pointer() == h_pointer)
    check("H mesh reused (no rebuild)", h_obj.data.as_pointer() == h_mesh_pointer)
    check(
        "no atom object was recreated",
        {o.as_pointer() for o in root.objects if o.get("cgn_role") == "atoms"}
        == pointers_before,
    )

# ------------------------------------------------------------------ template
print("\n=== ORTEP template in-place update ===")
template = bpy.data.objects.get(CrystalViz._ORTEP_TEMPLATE_OBJECT)
check("ORTEP template exists", template is not None)
if template is not None:
    mesh = template.data
    mesh_pointer = mesh.as_pointer()
    coords_before = [tuple(v.co) for v in mesh.vertices]
    surface_before = [d.value for d in mesh.attributes["cgn_ortep_surface"].data]
    polygon_count = len(mesh.polygons)
    scene.cgn_ortep_ellipse_thickness = 0.05
    check("thickness change is deferred",
          "ortep_ellipse_thickness" in CrystalViz._DEBOUNCE_TIMERS)
    check("template mesh untouched before flush",
          template.data.as_pointer() == mesh_pointer
          and [tuple(v.co) for v in template.data.vertices] == coords_before)
    CrystalViz._flush_debounce()
    check("template mesh reused (no rebuild)",
          template.data.as_pointer() == mesh_pointer)
    coords_after = [tuple(v.co) for v in template.data.vertices]
    check(
        "template vertices moved",
        any(
            abs(a - b) > 1e-6
            for va, vb in zip(coords_before, coords_after)
            for a, b in zip(va, vb)
        ),
    )
    check("topology unchanged", len(template.data.polygons) == polygon_count)
    check(
        "face attributes unchanged",
        [d.value for d in template.data.attributes["cgn_ortep_surface"].data]
        == surface_before,
    )
    check(
        "stored thickness updated",
        abs(template.data["cgn_ortep_ellipse_thickness"] - 0.05) < 1e-6,
        str(template.data["cgn_ortep_ellipse_thickness"]),
    )
    reference = CrystalViz._build_ortep_template_mesh(0.05)
    max_diff = max(
        abs(template.data.vertices[i].co[axis] - reference.vertices[i].co[axis])
        for i in range(len(reference.vertices))
        for axis in range(3)
    )
    check("in-place geometry matches a fresh build", max_diff < 1e-5,
          f"max_diff={max_diff:.2e}")
    bpy.data.meshes.remove(reference)

stale = CrystalViz._build_ortep_template_mesh(0.02)
stale["cgn_ortep_template_version"] = 2
check("stale template refuses in-place update",
      CrystalViz._update_ortep_template_mesh(stale, 0.03) is False)
bpy.data.meshes.remove(stale)

# ------------------------------------------------------------------ bonds
print("\n=== bond rebuild deferral ===")
CrystalViz._rebuild_bonds(scene)
bpy.context.view_layer.update()
bonds_before = {o.as_pointer() for o in root.objects if o.get("cgn_role") == "bonds"}
check("bonds exist", len(bonds_before) > 0, f"n={len(bonds_before)}")

calls["n"] = 0
original_rebuild = CrystalViz._rebuild_bonds
CrystalViz._rebuild_bonds = lambda sc: calls.__setitem__("n", calls["n"] + 1)
for _ in range(4):
    CrystalViz._update_bond_style(scene, None)
check("4 bond style changes coalesce", list(CrystalViz._DEBOUNCE_TIMERS) == ["bond_style"],
      str(list(CrystalViz._DEBOUNCE_TIMERS)))
CrystalViz._flush_debounce()
check("bond rebuild ran once on flush", calls["n"] == 1, f"calls={calls['n']}")
CrystalViz._rebuild_bonds = original_rebuild

scene.cgn_bond_spacing = 0.4
check("bond spacing change is deferred", "bond_style" in CrystalViz._DEBOUNCE_TIMERS)
check(
    "bond objects untouched before flush",
    {o.as_pointer() for o in root.objects if o.get("cgn_role") == "bonds"}
    == bonds_before,
)
CrystalViz._flush_debounce()
bonds_after = {o.as_pointer() for o in root.objects if o.get("cgn_role") == "bonds"}
check("bonds rebuilt after flush", bonds_after and bonds_after != bonds_before,
      f"{len(bonds_before)} -> {len(bonds_after)}")

# ------------------------------------------------------------------ render/unregister
print("\n=== render flush / unregister ===")
CrystalViz._update_bond_style(scene, None)
check("pending callback before render", "bond_style" in CrystalViz._DEBOUNCE_TIMERS)
CrystalViz._orient_ortep_for_render(scene)
check("render handler flushes pending callbacks", CrystalViz._DEBOUNCE_TIMERS == {})

CrystalViz._update_bond_style(scene, None)
CrystalViz.unregister()
check("unregister cancels pending callbacks", CrystalViz._DEBOUNCE_TIMERS == {})
check(
    "unregister unregisters the debounce timer",
    not bpy.app.timers.is_registered(CrystalViz._debounce_timer),
)

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
