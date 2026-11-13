"""Validate NumPy vectorization + ELLIPSOID-only ORTEP view timer.

Run:
  blender --background --factory-startup --python test_ortep_vectorization.py

Covers:
  * _euler_xyz_rotation_matrices matches mathutils Euler("XYZ").to_matrix()
  * _matrix_to_euler_xyz round-trips (including gimbal-lock angles)
  * batched _add_ellipsoid_attributes matches the old per-atom implementation
    (axes, orientation up to the ellipsoid's axis-sign symmetry, fallbacks)
  * _orient_ortep_openings matches the old per-vertex implementation for both
    view-direction and camera-position branches, including object transforms
  * the ORTEP view timer runs only while the model is ELLIPSOID
  * vectorized paths are materially faster on large inputs
"""

import math
import sys
import time

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

CIF = _env.CIF_DIR

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- old reference
def old_base_components(atom, size_scale=1.0):
    values = atom.get("u_cart")
    if values and len(values) == 6:
        u11, u22, u33, u12, u13, u23 = (float(v) for v in values)
        if all(math.isfinite(v) for v in values) and u11 > 0.0 and u22 > 0.0 and u33 > 0.0:
            tensor = np.array(((u11, u12, u13), (u12, u22, u23), (u13, u23, u33)), dtype=float)
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(tensor)
                eigenvalues = np.clip(eigenvalues, 0.0, None)
                axes = np.sqrt(eigenvalues * CrystalViz._ELLIPSOID_CHI2_50)
                axes = np.maximum(axes * CrystalViz._DISPLAY_SCALE * size_scale, 1e-4)
                if np.linalg.det(eigenvectors) < 0.0:
                    eigenvectors[:, 2] *= -1.0
                rotation = Matrix(eigenvectors.tolist()).to_euler("XYZ")
                return (
                    tuple(float(v) for v in axes),
                    (float(rotation.x), float(rotation.y), float(rotation.z)),
                )
            except Exception:  # nosec B110
                pass
    radius = CrystalViz._atom_radius(
        atom.get("vdw_radius", CrystalViz._DEFAULT_VDW_RADIUS), "BALL_STICK"
    )
    radius *= size_scale
    return (radius, radius, radius), (0.0, 0.0, 0.0)


def old_add_ellipsoid_attributes(mesh, atoms):
    scene = bpy.context.scene
    hydrogen_scale = scene.cgn_hydrogen_scale
    scale_attribute = mesh.attributes.new("cgn_scale", "FLOAT_VECTOR", "POINT")
    rotation_attribute = mesh.attributes.new("cgn_rotation", "FLOAT_VECTOR", "POINT")
    for index, atom in enumerate(atoms):
        size_scale = hydrogen_scale if str(atom.get("element", "")).upper() == "H" else 1.0
        axes, rotation = old_base_components(atom, size_scale)
        scale_attribute.data[index].vector = axes
        rotation_attribute.data[index].vector = rotation


def old_orient(obj, view_direction=None, camera_position=None):
    """Old per-vertex orientation, returned as arrays (does not touch the mesh)."""
    scale_attribute = obj.data.attributes["cgn_scale"]
    rotation_attribute = obj.data.attributes["cgn_rotation"]
    world_to_object = obj.matrix_world.to_3x3().inverted_safe()
    result = []
    for index, vertex in enumerate(obj.data.vertices):
        if camera_position is not None:
            point = obj.matrix_world @ vertex.co
            direction = world_to_object @ (camera_position - point)
        else:
            direction = world_to_object @ view_direction
        rotation = Euler(rotation_attribute.data[index].vector, "XYZ").to_matrix()
        local_direction = rotation.transposed() @ direction
        current = scale_attribute.data[index].vector
        magnitudes = tuple(abs(value) for value in current)
        result.append(
            tuple(
                magnitude if component >= 0.0 else -magnitude
                for magnitude, component in zip(magnitudes, local_direction, strict=True)
            )
        )
    return np.array(result, dtype=np.float64)


def read_vector_attribute(attribute, count):
    data = np.empty(count * 3, dtype=np.float32)
    attribute.data.foreach_get("vector", data)
    return data.reshape(-1, 3).astype(np.float64)


def write_scale(obj, values):
    attribute = obj.data.attributes["cgn_scale"]
    attribute.data.foreach_set("vector", np.asarray(values, dtype=np.float32).ravel())
    obj.data.update()


def make_point_mesh(name, count):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(0.0, 0.0, 0.0)] * count, [], [])
    mesh.update()
    return mesh


def rotation_matrix_from_euler(values):
    return CrystalViz._euler_xyz_rotation_matrices(np.asarray(values, dtype=np.float64))


def axis_set_distance(first_axes, first_rotation, second_axes, second_rotation):
    """Distance between two ellipsoids' semi-axis vector sets (sign agnostic)."""
    first = rotation_matrix_from_euler([first_rotation])[0]
    second = rotation_matrix_from_euler([second_rotation])[0]
    worst = 0.0
    for axis in range(3):
        a = first[:, axis] * first_axes[axis]
        b = second[:, axis] * second_axes[axis]
        worst = max(worst, min(np.linalg.norm(a - b), np.linalg.norm(a + b)))
    return worst


# ---------------------------------------------------------------- euler helpers
print("\n=== Euler helpers ===")
angles = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.3, -0.7, 1.1],
        [1.2, math.pi / 2.0, -0.4],
        [0.4, -math.pi / 2.0, 0.9],
        [2.0, math.pi / 2.0, -1.0],
    ]
)
matrices = CrystalViz._euler_xyz_rotation_matrices(angles)
error = max(
    float(np.max(np.abs(matrix - np.array(Euler(tuple(angle), "XYZ").to_matrix()))))
    for angle, matrix in zip(angles, matrices, strict=True)
)
check("Euler matrices match mathutils", error < 1e-6, f"max_err={error:.2e}")

rng = np.random.default_rng(7)
random_angles = rng.uniform(-math.pi, math.pi, size=(200, 3))
random_matrices = CrystalViz._euler_xyz_rotation_matrices(random_angles)
random_reference = np.array(
    [np.array(Euler(tuple(angle), "XYZ").to_matrix()) for angle in random_angles]
)
error = float(np.max(np.abs(random_matrices - random_reference)))
check("random Euler matrices match mathutils", error < 1e-6, f"max_err={error:.2e}")

roundtrip = CrystalViz._euler_xyz_rotation_matrices(
    CrystalViz._matrix_to_euler_xyz(random_matrices)
)
error = float(np.max(np.abs(roundtrip - random_matrices)))
check("matrix -> Euler -> matrix roundtrip", error < 1e-9, f"max_err={error:.2e}")
gimbal_roundtrip = CrystalViz._euler_xyz_rotation_matrices(
    CrystalViz._matrix_to_euler_xyz(matrices)
)
error = float(np.max(np.abs(gimbal_roundtrip - matrices)))
check("gimbal-lock roundtrip", error < 1e-9, f"max_err={error:.2e}")

# ---------------------------------------------------------------- ADP parity
print("\n=== ellipsoid attribute parity ===")
scene = bpy.context.scene
scene.cgn_connectivity_pairs.clear()
scene.cgn_hydrogen_scale = 1.6
bpy.ops.cgn.import_cif(filepath=f"{CIF}\\test.cif")
root = bpy.data.collections.get(scene.cgn_active_structure)
real_atoms = CrystalViz._structure_atoms(root)
with_adp = [atom for atom in real_atoms if atom.get("u_cart")]
check("test.cif provides ADP atoms", len(with_adp) > 0, f"{len(with_adp)}/{len(real_atoms)}")

synthetic = [
    {"element": "C", "u_cart": [0.011, 0.021, 0.015, 0.001, -0.002, 0.003], "vdw_radius": 1.7},
    {"element": "C", "u_cart": [0.008, 0.008, 0.008, 0.0, 0.0, 0.0], "vdw_radius": 1.7},
    {"element": "H", "u_cart": [0.02, 0.01, 0.014, 0.002, 0.0, -0.001], "vdw_radius": 1.2},
    {"element": "O", "u_cart": None, "vdw_radius": 1.52},
    {"element": "N", "u_cart": [0.0, 0.01, 0.01, 0.0, 0.0, 0.0], "vdw_radius": 1.55},
    {"element": "S", "u_cart": [float("nan"), 0.01, 0.01, 0.0, 0.0, 0.0]},
    {"element": "C", "u_cart": [0.01, 0.015, 0.02, 0.0, 0.0, 0.0]},
]
parity_atoms = with_adp[:12] + synthetic if with_adp else synthetic

mesh_old = make_point_mesh("OldEllipsoidPoints", len(parity_atoms))
mesh_new = make_point_mesh("NewEllipsoidPoints", len(parity_atoms))
old_add_ellipsoid_attributes(mesh_old, parity_atoms)
CrystalViz._add_ellipsoid_attributes(mesh_new, parity_atoms)

old_scales = read_vector_attribute(mesh_old.attributes["cgn_scale"], len(parity_atoms))
new_scales = read_vector_attribute(mesh_new.attributes["cgn_scale"], len(parity_atoms))
old_rotations = read_vector_attribute(mesh_old.attributes["cgn_rotation"], len(parity_atoms))
new_rotations = read_vector_attribute(mesh_new.attributes["cgn_rotation"], len(parity_atoms))
scale_error = float(np.max(np.abs(old_scales - new_scales)))
check("semi-axes match old implementation", scale_error < 1e-6, f"max_err={scale_error:.2e}")
orientation_error = max(
    axis_set_distance(old_scales[i], old_rotations[i], new_scales[i], new_rotations[i])
    for i in range(len(parity_atoms))
)
check(
    "ellipsoid orientations match (up to axis sign)",
    orientation_error < 1e-5,
    f"max_err={orientation_error:.2e}",
)
for mesh in (mesh_old, mesh_new):
    bpy.data.meshes.remove(mesh)

# ---------------------------------------------------------------- orient parity
print("\n=== ORTEP opening orientation parity ===")
scene.cgn_model = "ELLIPSOID"
bpy.context.view_layer.update()

test_mesh = make_point_mesh("TestOrientPoints", 8)
scale_attribute = test_mesh.attributes.new("cgn_scale", "FLOAT_VECTOR", "POINT")
rotation_attribute = test_mesh.attributes.new("cgn_rotation", "FLOAT_VECTOR", "POINT")
base_scales = rng.uniform(0.01, 0.08, size=(8, 3))
base_rotations = rng.uniform(-math.pi, math.pi, size=(8, 3))
test_mesh.vertices.foreach_set(
    "co",
    np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.2, -0.4),
            (-0.3, 1.1, 0.5),
            (0.6, -0.7, 1.2),
            (2.0, 1.5, -1.0),
            (-1.1, 0.4, 2.2),
            (0.9, -1.3, 0.1),
            (0.2, 0.8, -2.1),
        ]
    ).ravel(),
)
test_mesh.update()
scale_attribute.data.foreach_set("vector", base_scales.astype(np.float32).ravel())
rotation_attribute.data.foreach_set("vector", base_rotations.astype(np.float32).ravel())

test_obj = bpy.data.objects.new("CGN_TestOrient", test_mesh)
test_obj["cgn_role"] = "atoms"
test_obj["cgn_element"] = "C"
root.objects.link(test_obj)
test_obj.matrix_world = (
    Matrix.Translation((1.5, -2.0, 0.75))
    @ Matrix.Rotation(0.6, 4, "Z")
    @ Matrix.Rotation(-0.4, 4, "X")
)
bpy.context.view_layer.update()

stored_scales = read_vector_attribute(test_mesh.attributes["cgn_scale"], 8)
stored_rotations = read_vector_attribute(test_mesh.attributes["cgn_rotation"], 8)

direction = Vector((0.3, -0.8, 0.5))
expected = old_orient(test_obj, view_direction=direction)
write_scale(test_obj, stored_scales)
CrystalViz._orient_ortep_openings(scene, view_direction=direction)
actual = read_vector_attribute(test_mesh.attributes["cgn_scale"], 8)
error = float(np.max(np.abs(expected - actual)))
check("view-direction orientation matches old loop", error < 1e-5, f"max_err={error:.2e}")

again = actual.copy()
CrystalViz._orient_ortep_openings(scene, view_direction=direction)
check(
    "repeated orientation is a no-op",
    np.array_equal(again, read_vector_attribute(test_mesh.attributes["cgn_scale"], 8)),
)

camera_position = Vector((-3.2, 4.1, 2.6))
expected = old_orient(test_obj, camera_position=camera_position)
write_scale(test_obj, stored_scales)
CrystalViz._orient_ortep_openings(scene, camera_position=camera_position)
actual = read_vector_attribute(test_mesh.attributes["cgn_scale"], 8)
error = float(np.max(np.abs(expected - actual)))
check("camera-position orientation matches old loop", error < 1e-5, f"max_err={error:.2e}")

# A few real structure objects must also match at identity transforms.
real_objects = [
    obj
    for obj in root.objects
    if obj.get("cgn_role") == "atoms"
    and str(obj.get("cgn_element", "")).upper() != "H"
    and obj.data.attributes.get("cgn_scale") is not None
    and obj is not test_obj
]
real_error = 0.0
for obj in real_objects:
    count = len(obj.data.vertices)
    before = read_vector_attribute(obj.data.attributes["cgn_scale"], count)
    expected = old_orient(obj, view_direction=direction)
    write_scale(obj, before)
    CrystalViz._orient_ortep_openings(scene, view_direction=direction)
    after = read_vector_attribute(obj.data.attributes["cgn_scale"], count)
    real_error = max(real_error, float(np.max(np.abs(expected - after))))
check(
    "real structure ellipsoids match old loop",
    real_error < 1e-5,
    f"objects={len(real_objects)} max_err={real_error:.2e}",
)

# ---------------------------------------------------------------- timer gating
print("\n=== ORTEP view timer gating ===")
check(
    "timer runs in ELLIPSOID",
    bpy.app.timers.is_registered(CrystalViz._update_ortep_view_orientation),
)
result = CrystalViz._update_ortep_view_orientation()
check(
    "callback returns the 0.25 s interval in ELLIPSOID",
    result == CrystalViz._ORTEP_VIEW_TIMER_INTERVAL,
    f"result={result}",
)

scene.cgn_model = "BALL_STICK"
bpy.context.view_layer.update()
check(
    "switching away from ELLIPSOID stops the timer",
    not bpy.app.timers.is_registered(CrystalViz._update_ortep_view_orientation),
)
check(
    "callback returns None outside ELLIPSOID",
    CrystalViz._update_ortep_view_orientation() is None,
)

scene.cgn_model = "ELLIPSOID"
bpy.context.view_layer.update()
check(
    "switching back to ELLIPSOID restarts the timer",
    bpy.app.timers.is_registered(CrystalViz._update_ortep_view_orientation),
)

# ---------------------------------------------------------------- speed
print("\n=== speed ===")
count = 6000
speed_atoms = []
for index in range(count):
    base = 0.008 + (index % 17) * 0.0004
    speed_atoms.append(
        {
            "element": "C",
            "vdw_radius": 1.7,
            "u_cart": [
                base,
                base * 1.3,
                base * 0.8,
                base * 0.05,
                -base * 0.03,
                base * 0.02,
            ],
        }
    )
old_mesh = make_point_mesh("OldSpeedPoints", count)
t0 = time.perf_counter()
old_add_ellipsoid_attributes(old_mesh, speed_atoms)
old_add_time = time.perf_counter() - t0
new_mesh = make_point_mesh("NewSpeedPoints", count)
t0 = time.perf_counter()
CrystalViz._add_ellipsoid_attributes(new_mesh, speed_atoms)
new_add_time = time.perf_counter() - t0
print(
    f"  attributes n={count}: old={old_add_time * 1000:.1f} ms  "
    f"new={new_add_time * 1000:.1f} ms  speedup={old_add_time / max(new_add_time, 1e-9):.1f}x"
)
check(
    "batched attribute generation is faster",
    new_add_time < old_add_time / 2,
    f"{old_add_time * 1000:.1f} -> {new_add_time * 1000:.1f} ms",
)
for mesh in (old_mesh, new_mesh):
    bpy.data.meshes.remove(mesh)

vertex_count = 40000
speed_mesh = make_point_mesh("SpeedOrientPoints", vertex_count)
speed_mesh.vertices.foreach_set(
    "co", rng.uniform(-3.0, 3.0, size=vertex_count * 3).astype(np.float32)
)
speed_mesh.update()
speed_scale = speed_mesh.attributes.new("cgn_scale", "FLOAT_VECTOR", "POINT")
speed_rotation = speed_mesh.attributes.new("cgn_rotation", "FLOAT_VECTOR", "POINT")
speed_scales = rng.uniform(0.01, 0.1, size=(vertex_count, 3))
speed_rotations = rng.uniform(-math.pi, math.pi, size=(vertex_count, 3))
speed_scale.data.foreach_set("vector", speed_scales.astype(np.float32).ravel())
speed_rotation.data.foreach_set("vector", speed_rotations.astype(np.float32).ravel())
speed_obj = bpy.data.objects.new("CGN_SpeedOrient", speed_mesh)
speed_obj["cgn_role"] = "atoms"
speed_obj["cgn_element"] = "C"
root.objects.link(speed_obj)
bpy.context.view_layer.update()

t0 = time.perf_counter()
speed_expected = old_orient(speed_obj, view_direction=direction)
old_orient_time = time.perf_counter() - t0
write_scale(speed_obj, speed_scales)
t0 = time.perf_counter()
CrystalViz._orient_ortep_openings(scene, view_direction=direction)
new_orient_time = time.perf_counter() - t0
speed_actual = read_vector_attribute(speed_mesh.attributes["cgn_scale"], vertex_count)
error = float(np.max(np.abs(speed_expected - speed_actual)))
print(
    f"  orient n={vertex_count}: old={old_orient_time * 1000:.1f} ms  "
    f"new={new_orient_time * 1000:.1f} ms  "
    f"speedup={old_orient_time / max(new_orient_time, 1e-9):.1f}x"
)
check("vectorized orientation matches at scale", error < 1e-5, f"max_err={error:.2e}")
check(
    "vectorized orientation is faster",
    new_orient_time < old_orient_time / 3,
    f"{old_orient_time * 1000:.1f} -> {new_orient_time * 1000:.1f} ms",
)

CrystalViz.unregister()
check(
    "unregister stops the timer",
    not bpy.app.timers.is_registered(CrystalViz._update_ortep_view_orientation),
)

print()
print("TOTAL FAILURES:", len(FAILURES))
print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
sys.exit(1 if FAILURES else 0)
