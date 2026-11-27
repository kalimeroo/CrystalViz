"""View projection operator: align the 3D viewport along a/b/c or [h k l],
or towards an (h k l) plane.

Run:
  blender --background --factory-startup --python test_view_projection.py

The CGN_OT_view_along operator ("cgn.view_along", the View panel) points the
active 3D viewport down a crystallographic direction and snaps it to
orthographic projection.  The a/b/c buttons use the
corresponding column of the stored orthogonalised cell matrix; the [h k l]
case combines the axes the same way, so [1 0 0] must reproduce "along a".
The (h k l) plane case looks along the plane normal, the reciprocal-lattice
vector h a* + k b* + l c*, so the plane is seen face on; the look sense
follows the indices — positive indices view from the origin along the
normal, negated indices the opposite way (a vertical mirror).
Each projection also lays the cell out the conventional way: along a, b runs
horizontal and c vertical; along b, a horizontal and c vertical; along c, a
horizontal and b vertical (the plane projection keeps the same horizontal
choice — the first axis whose projection is usable — but its vertical side
follows the index sense).  Success means the resulting view forward (-Z
local) is parallel to that direction, the in-plane roll matches the
convention, the view is re-centred on the structure, and a zero [0 0 0]
direction is rejected.
"""

import json
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz  # noqa: E402

CrystalViz.register()

CIF = _env.cif("multiple_bonds.cif")
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def view3d():
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            rv3d = area.spaces.active.region_3d
            if rv3d is not None:
                return rv3d
    raise RuntimeError("no 3D viewport on the active screen")


def forward(rv3d):
    """World-space direction the viewport looks down (its local -Z)."""
    return rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))


def right(rv3d):
    """World-space direction the viewport points right (its local +X)."""
    return rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))


def up(rv3d):
    """World-space direction the viewport points up (its local +Y)."""
    return rv3d.view_rotation @ Vector((0.0, 1.0, 0.0))


def projected(axis, onto):
    """*axis* with the component along *onto* removed (2-D projection)."""
    axis = Vector(axis)
    norm2 = Vector(onto).length_squared
    if norm2 == 0.0:
        return axis
    return axis - Vector(onto) * (axis.dot(onto) / norm2)


def laid_out(rv3d, horizontal, vertical):
    """True when the view lays the cell out the conventional way.

    The *horizontal* axis reads exactly horizontal (positive end right) and
    the *vertical* axis' positive end lands up.  On the monoclinic
    multiple_bonds.cif the along-b pair is not perpendicular, so there the
    secondary axis reads tilted; the sign is what pins the drawing as not
    mirrored.
    """
    fwd = forward(rv3d)
    horizontal_dir = projected(horizontal, fwd)
    vertical_dir = projected(vertical, fwd)
    return (
        parallel(right(rv3d), horizontal_dir)
        and right(rv3d).dot(horizontal_dir) > 0.0
        and up(rv3d).dot(vertical_dir) > 0.0
    )


def parallel(got, want, tol=1e-5):
    got = Vector(got)
    want = Vector(want)
    length = got.length
    if length == 0.0 or want.length == 0.0:
        return False
    return abs(got.normalized().dot(want.normalized())) > 1.0 - tol


def plane_normal(cell, h_idx, k_idx, l_idx):
    """Unit normal to the (h k l) plane = the reciprocal vector h a* + k b* + l c*."""
    a = Vector((cell[0], cell[1], cell[2]))
    b = Vector((cell[3], cell[4], cell[5]))
    c = Vector((cell[6], cell[7], cell[8]))
    volume = a.dot(b.cross(c))
    return (h_idx * b.cross(c) + k_idx * c.cross(a) + l_idx * a.cross(b)) / volume


def facing(rv3d, direction):
    """True when the view looks exactly in *direction*'s sense (not mirrored)."""
    return forward(rv3d).normalized().dot(Vector(direction).normalized()) > 1.0 - 1e-5


def looking_away(rv3d, direction):
    """True when the view looks opposite to *direction* (negated indices)."""
    return forward(rv3d).normalized().dot(Vector(direction).normalized()) < -(1.0 - 1e-5)


def horizontal_axis(rv3d, axis):
    """True when *axis* reads exactly horizontal (positive end right)."""
    fwd = forward(rv3d)
    horiz = projected(axis, fwd)
    return parallel(right(rv3d), horiz) and right(rv3d).dot(horiz) > 0.0


bpy.ops.cgn.import_cif(filepath=CIF)
scene = bpy.context.scene
root = bpy.data.collections.get(scene.cgn_active_structure)
cell = json.loads(root["cgn_cell_matrix"])
rv3d = view3d()
a_axis = Vector((cell[0], cell[1], cell[2]))
b_axis = Vector((cell[3], cell[4], cell[5]))
c_axis = Vector((cell[6], cell[7], cell[8]))

print("\n=== A: projection is orthographic ===")
bpy.ops.cgn.view_along(direction="A")
check("press along a keeps orthographic", rv3d.view_perspective == "ORTHO", rv3d.view_perspective)
check("forward parallel to a", parallel(forward(rv3d), a_axis), str(forward(rv3d)))
check(
    "along a: b horizontal, c vertical",
    laid_out(rv3d, b_axis, c_axis),
    f"right={tuple(right(rv3d))} up={tuple(up(rv3d))}",
)
center = CrystalViz._structure_view_center(root)
check(
    "view centred on the structure",
    center is not None and all(abs(center[i] - rv3d.view_location[i]) < 1e-5 for i in range(3)),
    str(tuple(rv3d.view_location)),
)

print("\n=== B: along b and along c ===")
bpy.ops.cgn.view_along(direction="B")
check("forward parallel to b", parallel(forward(rv3d), b_axis), str(forward(rv3d)))
check(
    "along b: a horizontal, c vertical",
    laid_out(rv3d, a_axis, c_axis),
    f"right={tuple(right(rv3d))} up={tuple(up(rv3d))}",
)
bpy.ops.cgn.view_along(direction="C")
check("forward parallel to c", parallel(forward(rv3d), c_axis), str(forward(rv3d)))
check(
    "along c: a horizontal, b vertical",
    laid_out(rv3d, a_axis, b_axis),
    f"right={tuple(right(rv3d))} up={tuple(up(rv3d))}",
)

print("\n=== C: [h k l] matches the axis combination ===")
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 1.0, 0.0, 0.0
bpy.ops.cgn.view_along(direction="HKL")
check("[1 0 0] parallel to a", parallel(forward(rv3d), a_axis), str(forward(rv3d)))
check(
    "[1 0 0] lays out like along a",
    laid_out(rv3d, b_axis, c_axis),
    f"right={tuple(right(rv3d))} up={tuple(up(rv3d))}",
)
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 0.0, 1.0, 0.0
bpy.ops.cgn.view_along(direction="HKL")
check("[0 1 0] parallel to b", parallel(forward(rv3d), b_axis), str(forward(rv3d)))
check(
    "[0 1 0] lays out like along b",
    laid_out(rv3d, a_axis, c_axis),
    f"right={tuple(right(rv3d))} up={tuple(up(rv3d))}",
)
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 0.0, 0.0, 1.0
bpy.ops.cgn.view_along(direction="HKL")
check("[0 0 1] parallel to c", parallel(forward(rv3d), c_axis), str(forward(rv3d)))
check(
    "[0 0 1] lays out like along c",
    laid_out(rv3d, a_axis, b_axis),
    f"right={tuple(right(rv3d))} up={tuple(up(rv3d))}",
)
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 1.0, 1.0, 1.0
bpy.ops.cgn.view_along(direction="HKL")
sum_axis = a_axis + b_axis + c_axis
check("[1 1 1] parallel to a+b+c", parallel(forward(rv3d), sum_axis), str(forward(rv3d)))
check(
    "[1 1 1] keeps a right and b up",
    laid_out(rv3d, a_axis, b_axis),
    f"right={tuple(right(rv3d))} up={tuple(up(rv3d))}",
)

print("\n=== D: a zero [0 0 0] direction is rejected ===")
before = Vector(forward(rv3d))
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 0.0, 0.0, 0.0
try:
    bpy.ops.cgn.view_along(direction="HKL")
    check("zero [0 0 0] is rejected", False, "no error raised")
except RuntimeError as exc:
    check("zero [0 0 0] is rejected", "zero" in str(exc).lower(), str(exc))
check("view unchanged after rejection", parallel(before, forward(rv3d)))
# Restore a sane stored value for later panels/tests.
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 0.0, 0.0, 1.0

print("\n=== E: towards an (h k l) plane looks from the origin along the normal ===")
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 1.0, 0.0, 0.0
bpy.ops.cgn.view_along(direction="PLANE")
a_star = plane_normal(cell, 1.0, 0.0, 0.0)
check(
    "(1 0 0) looks from origin towards +a*",
    facing(rv3d, a_star),
    str(forward(rv3d)),
)
check(
    "(1 0 0) keeps a exactly horizontal",
    horizontal_axis(rv3d, a_axis),
    f"right={tuple(right(rv3d))}",
)
up_100 = Vector(up(rv3d))
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = -1.0, 0.0, 0.0
bpy.ops.cgn.view_along(direction="PLANE")
check(
    "(-1 0 0) looks from +a* back to origin",
    looking_away(rv3d, a_star),
    str(forward(rv3d)),
)
check(
    "(-1 0 0) is the vertical mirror of (1 0 0)",
    up(rv3d).normalized().dot(up_100.normalized()) < -(1.0 - 1e-5),
    f"up={tuple(up(rv3d))}",
)
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 0.0, 1.0, 0.0
bpy.ops.cgn.view_along(direction="PLANE")
b_star = plane_normal(cell, 0.0, 1.0, 0.0)
check(
    "(0 1 0) looks from origin towards +b*",
    facing(rv3d, b_star),
    str(forward(rv3d)),
)
check(
    "(0 1 0) keeps a exactly horizontal",
    horizontal_axis(rv3d, a_axis),
    f"right={tuple(right(rv3d))}",
)
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 0.0, -1.0, 0.0
bpy.ops.cgn.view_along(direction="PLANE")
check(
    "(0 -1 0) looks from +b* back to origin",
    looking_away(rv3d, b_star),
    str(forward(rv3d)),
)
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 0.0, 0.0, 1.0
bpy.ops.cgn.view_along(direction="PLANE")
c_star = plane_normal(cell, 0.0, 0.0, 1.0)
check(
    "(0 0 1) looks from origin towards +c*",
    facing(rv3d, c_star),
    str(forward(rv3d)),
)
check(
    "(0 0 1) keeps a exactly horizontal",
    horizontal_axis(rv3d, a_axis),
    f"right={tuple(right(rv3d))}",
)

print("\n=== F: a zero [0 0 0] plane normal is rejected ===")
before = Vector(forward(rv3d))
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 0.0, 0.0, 0.0
try:
    bpy.ops.cgn.view_along(direction="PLANE")
    check("zero (0 0 0) plane is rejected", False, "no error raised")
except RuntimeError as exc:
    check("zero (0 0 0) plane is rejected", "zero" in str(exc).lower(), str(exc))
check("view unchanged after rejection", parallel(before, forward(rv3d)))
# Restore a sane stored value for later panels/tests.
scene.cgn_view_h, scene.cgn_view_k, scene.cgn_view_l = 0.0, 0.0, 1.0

print("\nRESULT: " + ("ALL PASS" if not FAILURES else "FAIL"))
if FAILURES:
    print("FAILURES:", FAILURES)
    raise SystemExit(1)
