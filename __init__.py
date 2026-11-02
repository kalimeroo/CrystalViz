bl_info = {
    "name": "CCTBX Geometry Nodes (gemmi)",
    "author": "OpenAI",
    "version": (0, 5, 6),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > CCTBX GN; File > Import",
    "description": (
        "Import CIF atoms and render editable connectivity with Geometry "
        "Nodes (gemmi backend, no external Python)"
    ),
    "category": "Import-Export",
}

import bpy
import json
import math
import os
import sys
import traceback
import uuid
import gpu
import numpy as np
from bpy.types import AddonPreferences, Menu, Operator, Panel, PropertyGroup
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector


def _srgb_to_linear(value):
    value /= 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _cpk_color(hex_value):
    """Convert a Jmol CPK hex colour to linear RGBA for Blender."""
    rgb = tuple(int(hex_value[index:index + 2], 16) for index in (0, 2, 4))
    return tuple(_srgb_to_linear(channel) for channel in rgb) + (1.0,)


# Jmol CPK colours (sRGB hex) from the Science Notes JMOL CPK table.
_JMOL_CPK_HEX = {
    "H": "FFFFFF", "D": "FFFFC0", "T": "FFFFA0", "HE": "D9FFFF",
    "LI": "CC80FF", "BE": "C2FF00", "B": "FFB5B5", "C": "909090",
    "N": "3050F8", "O": "FF0D0D", "F": "90E050", "NE": "B3E3F5",
    "NA": "AB5CF2", "MG": "8AFF00", "AL": "BFA6A6", "SI": "F0C8A0",
    "P": "FF8000", "S": "FFFF30", "CL": "1FF01F", "AR": "80D1E3",
    "K": "8F40D4", "CA": "3DFF00", "SC": "E6E6E6", "TI": "BFC2C7",
    "V": "A6A6AB", "CR": "8A99C7", "MN": "9C7AC7", "FE": "E06633",
    "CO": "F090A0", "NI": "50D050", "CU": "C88033", "ZN": "7D80B0",
    "GA": "C28F8F", "GE": "668F8F", "AS": "BD80E3", "SE": "FFA100",
    "BR": "A62929", "KR": "5CB8D1", "RB": "702EB0", "SR": "00FF00",
    "Y": "94FFFF", "ZR": "94E0E0", "NB": "73C2C9", "MO": "54B5B5",
    "TC": "3B9E9E", "RU": "248F8F", "RH": "0A7D8C", "PD": "006985",
    "AG": "C0C0C0", "CD": "FFD98F", "IN": "A67573", "SN": "668080",
    "SB": "9E63B5", "TE": "D47A00", "I": "940094", "XE": "429EB0",
    "CS": "57178F", "BA": "00C900", "LA": "70D4FF", "CE": "FFFFC7",
    "PR": "D9FFC7", "ND": "C7FFC7", "PM": "A3FFC7", "SM": "8FFFC7",
    "EU": "61FFC7", "GD": "45FFC7", "TB": "30FFC7", "DY": "1FFFC7",
    "HO": "00FF9C", "ER": "00E675", "TM": "00D452", "YB": "00BF38",
    "LU": "00AB24", "HF": "4DC2FF", "TA": "4DA6FF", "W": "2194D6",
    "RE": "267DAB", "OS": "266696", "IR": "175487", "PT": "D0D0E0",
    "AU": "FFD123", "HG": "B8B8D0", "TL": "A6544D", "PB": "575961",
    "BI": "9E4FB5", "PO": "AB5C00", "AT": "754F45", "RN": "428296",
    "FR": "420066", "RA": "007D00", "AC": "70ABFA", "TH": "00BAFF",
    "PA": "00A1FF", "U": "008FFF", "NP": "0080FF", "PU": "006BFF",
    "AM": "545CF2", "CM": "785CE3", "BK": "8A4FE3", "CF": "A136D4",
    "ES": "B31FD4", "FM": "B31FBA", "MD": "B30DA6", "NO": "BD0D87",
    "LR": "C70066", "RF": "CC0059", "DB": "D1004F", "SG": "D90045",
    "BH": "E00038", "HS": "E6002E", "MT": "EB0026",
}
_ELEMENT_COLORS = {
    element: _cpk_color(value) for element, value in _JMOL_CPK_HEX.items()
}
_DEFAULT_COLOR = _cpk_color("FF1493")
_DISPLAY_SCALE = 0.1
_BOND_VDW_FACTOR = 0.1
_BALL_ATOM_VDW_FACTOR = 0.25
_SPACE_FILLING_ATOM_VDW_FACTOR = 1.0
_WIRE_ATOM_VDW_FACTOR = 1.0 / 16.0
_ELLIPSOID_BOND_VDW_FACTOR = 0.05
_WIRE_BOND_VDW_FACTOR = 0.03
# Chi-square with 3 degrees of freedom at 50 % probability (ORTEP convention).
_ELLIPSOID_SCALE_50 = 1.5381722
_DEFAULT_VDW_RADIUS = 1.5
_DEFAULT_BOND_RADIUS = (
    _BOND_VDW_FACTOR * _DEFAULT_VDW_RADIUS * _DISPLAY_SCALE
)
_BOND_TOLERANCE = 0.4
_MAX_FILL_REPEATS = 20
_MAX_FILL_ATOMS = 200000
_UNIT_CELL_EDGE_COLOR = (0.0, 0.0, 0.0, 1.0)
_UNIT_CELL_FACE_COLOR = (0.45, 0.65, 0.95, 1.0)
_UNIT_CELL_FACE_ALPHA = 0.08
_UNIT_CELL_EDGE_THICKNESS = 0.05
_NODE_GROUP_ATOMS = "CGN_GN_AtomSpheres"
_NODE_GROUP_BONDS = "CGN_GN_BondTubes"
_NODE_GROUP_ELLIPSOIDS = "CGN_GN_AtomEllipsoids"
_MODEL_ITEMS = (
    ('BALL_STICK', "Ball & Stick", "Spheres at quarter van der Waals radius"),
    ('ELLIPSOID', "Ellipsoids", "Anisotropic displacement ellipsoids"),
    (
        'SPACE_FILLING',
        "Space-filling (Use VdW radii)",
        "Spheres at full van der Waals radius, without bonds",
    ),
    ('WIRE_STICKS', "Wire-sticks", "Thin atoms and thin bonds"),
)
_DOCS_URL = "https://github.com/kalimeroo/BlenderCCTBX"


def _gemmi_status():
    """Return (version, error) for the gemmi module in Blender's Python."""
    try:
        import gemmi
    except Exception as exc:  # ImportError, or a broken wheel
        return None, str(exc)
    return getattr(gemmi, "__version__", "unknown"), None


def _gemmi_install_hint():
    """Command that installs gemmi where Blender's Python can import it."""
    target = os.path.join(
        bpy.utils.user_resource('EXTENSIONS'),
        ".local",
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )
    return f'"{sys.executable}" -m pip install --target "{target}" gemmi'


def _gemmi_worker():
    """Import the in-process gemmi worker, with an install hint if missing."""
    try:
        from . import gemmi_worker
    except ImportError as exc:
        raise RuntimeError(
            "gemmi is not installed in Blender's Python.\n"
            "Install it with:\n" + _gemmi_install_hint()
        ) from exc
    return gemmi_worker


class CGN_AddonPreferences(AddonPreferences):
    bl_idname = __package__

    def draw(self, context):
        layout = self.layout
        version, error = _gemmi_status()
        if error:
            layout.label(
                text="gemmi is not available in Blender's Python.",
                icon='ERROR',
            )
            layout.label(text="Install it with:")
            layout.label(text=_gemmi_install_hint())
        else:
            layout.label(
                text=f"gemmi {version} (in-process, no external interpreter)",
                icon='CHECKMARK',
            )


class CGN_ConnectivityPair(PropertyGroup):
    element_a: StringProperty(options={'HIDDEN'})
    element_b: StringProperty(options={'HIDDEN'})
    enabled: BoolProperty(name="Connect", default=True)
    min_distance: FloatProperty(
        name="Min (Å)", default=0.1, min=0.0, max=20.0, precision=3,
    )
    max_distance: FloatProperty(
        name="Max (Å)", default=3.0, min=0.0, max=20.0, precision=3,
    )


def _unit_cell_settings_updated(self, context):
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    try:
        _apply_unit_cell_settings(scene)
    except Exception:
        traceback.print_exc()


class CGN_UnitCellSettings(PropertyGroup):
    show_unit_cell: BoolProperty(
        name="Show Unit Cell",
        description="Display the unit-cell box around the structure",
        default=True,
        update=_unit_cell_settings_updated,
    )
    face_color: FloatVectorProperty(
        name="Face Color",
        description="Unit-cell face color",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=_UNIT_CELL_FACE_COLOR,
        update=_unit_cell_settings_updated,
    )
    face_transparency: FloatProperty(
        name="Transparency",
        description="Unit-cell face transparency (1 is fully transparent)",
        default=1.0 - _UNIT_CELL_FACE_ALPHA,
        min=0.0,
        max=1.0,
        precision=2,
        update=_unit_cell_settings_updated,
    )
    edge_color: FloatVectorProperty(
        name="Edge Color",
        description="Unit-cell edge color",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=_UNIT_CELL_EDGE_COLOR,
        update=_unit_cell_settings_updated,
    )
    edge_thickness: FloatProperty(
        name="Edge Thickness",
        description="Unit-cell edge thickness",
        default=_UNIT_CELL_EDGE_THICKNESS,
        min=0.0,
        max=1.0,
        precision=3,
        update=_unit_cell_settings_updated,
    )


def _active_root(scene):
    name = scene.cgn_active_structure
    root = bpy.data.collections.get(name) if name else None
    if root is not None and root.get("cgn_role") == "structure":
        return root
    return None


def _socket_identifier(node_group, name, in_out='INPUT'):
    for item in node_group.interface.items_tree:
        if (
            item.item_type == 'SOCKET'
            and item.in_out == in_out
            and item.name == name
        ):
            return item.identifier
    raise RuntimeError(f"Geometry Nodes socket '{name}' was not created.")


def _ensure_shade_smooth(node_group):
    """Add a Set Shade Smooth node to an atom node group if it is missing."""
    for node in node_group.nodes:
        if node.bl_idname == "GeometryNodeSetShadeSmooth":
            return
    instance = next(
        (
            node for node in node_group.nodes
            if node.bl_idname == "GeometryNodeInstanceOnPoints"
        ),
        None,
    )
    if instance is None:
        return
    link = next(
        (
            link for link in node_group.links
            if link.to_node == instance and link.to_socket.name == "Instance"
        ),
        None,
    )
    if link is None:
        return
    try:
        smooth = node_group.nodes.new("GeometryNodeSetShadeSmooth")
    except RuntimeError:
        return
    smooth.location = (0, -150)
    source_socket = link.from_socket
    node_group.links.remove(link)
    node_group.links.new(source_socket, smooth.inputs["Geometry"])
    node_group.links.new(smooth.outputs["Geometry"], instance.inputs["Instance"])


def _make_node_group(name, is_bond_group):
    node_group = bpy.data.node_groups.get(name)
    if node_group is not None:
        if not is_bond_group:
            _ensure_shade_smooth(node_group)
        return node_group

    node_group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    geometry_in = node_group.interface.new_socket(
        name="Geometry", in_out='INPUT', socket_type="NodeSocketGeometry"
    )
    radius_in = node_group.interface.new_socket(
        name="Radius", in_out='INPUT', socket_type="NodeSocketFloat"
    )
    radius_in.default_value = (
        _DEFAULT_BOND_RADIUS if is_bond_group else 0.5 * _DISPLAY_SCALE
    )
    material_in = node_group.interface.new_socket(
        name="Material", in_out='INPUT', socket_type="NodeSocketMaterial"
    )
    geometry_out = node_group.interface.new_socket(
        name="Geometry", in_out='OUTPUT', socket_type="NodeSocketGeometry"
    )

    nodes = node_group.nodes
    links = node_group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    group_in.location = (-600, 0)
    group_out.location = (500, 0)
    if is_bond_group:
        mesh_to_curve = nodes.new("GeometryNodeMeshToCurve")
        profile = nodes.new("GeometryNodeCurvePrimitiveCircle")
        curve_to_mesh = nodes.new("GeometryNodeCurveToMesh")
        set_material = nodes.new("GeometryNodeSetMaterial")
        mesh_to_curve.location = (-350, 100)
        profile.location = (-350, -150)
        curve_to_mesh.location = (-100, 100)
        set_material.location = (150, 100)
        profile.inputs["Resolution"].default_value = 8
        links.new(group_in.outputs[geometry_in.identifier], mesh_to_curve.inputs["Mesh"])
        links.new(mesh_to_curve.outputs["Curve"], curve_to_mesh.inputs["Curve"])
        links.new(profile.outputs["Curve"], curve_to_mesh.inputs["Profile Curve"])
        links.new(group_in.outputs[radius_in.identifier], profile.inputs["Radius"])
        links.new(curve_to_mesh.outputs["Mesh"], set_material.inputs["Geometry"])
        links.new(group_in.outputs[material_in.identifier], set_material.inputs["Material"])
        links.new(set_material.outputs["Geometry"], group_out.inputs[geometry_out.identifier])
    else:
        ico = nodes.new("GeometryNodeMeshIcoSphere")
        set_material = nodes.new("GeometryNodeSetMaterial")
        smooth = nodes.new("GeometryNodeSetShadeSmooth")
        instance = nodes.new("GeometryNodeInstanceOnPoints")
        ico.location = (-350, -150)
        set_material.location = (-100, -150)
        smooth.location = (0, -150)
        instance.location = (150, 100)
        ico.inputs["Subdivisions"].default_value = 2
        links.new(group_in.outputs[geometry_in.identifier], instance.inputs["Points"])
        links.new(group_in.outputs[radius_in.identifier], ico.inputs["Radius"])
        links.new(ico.outputs["Mesh"], set_material.inputs["Geometry"])
        links.new(group_in.outputs[material_in.identifier], set_material.inputs["Material"])
        links.new(set_material.outputs["Geometry"], smooth.inputs["Geometry"])
        links.new(smooth.outputs["Geometry"], instance.inputs["Instance"])
        links.new(instance.outputs["Instances"], group_out.inputs[geometry_out.identifier])
    return node_group


def _current_model(scene=None):
    """Return the active rendering model name."""
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    model = getattr(scene, "cgn_model", 'BALL_STICK') if scene else 'BALL_STICK'
    if model not in {item[0] for item in _MODEL_ITEMS}:
        model = 'BALL_STICK'
    return model


def _atom_radius(vdw_radius, model=None):
    """Display-space atom radius for the active model."""
    if model is None:
        model = _current_model()
    if model == 'SPACE_FILLING':
        factor = _SPACE_FILLING_ATOM_VDW_FACTOR
    elif model == 'WIRE_STICKS':
        factor = _WIRE_ATOM_VDW_FACTOR
    else:
        factor = _BALL_ATOM_VDW_FACTOR
    return float(vdw_radius) * factor * _DISPLAY_SCALE


def _bond_radius(vdw_a, vdw_b, model=None):
    """Display-space bond radius for the active model."""
    if model is None:
        model = _current_model()
    mean_vdw = 0.5 * (float(vdw_a) + float(vdw_b))
    if model == 'WIRE_STICKS':
        factor = _WIRE_BOND_VDW_FACTOR
    elif model == 'ELLIPSOID':
        factor = _ELLIPSOID_BOND_VDW_FACTOR
    else:
        factor = _BOND_VDW_FACTOR
    return mean_vdw * factor * _DISPLAY_SCALE


def _ellipsoid_probability_factor(probability):
    """Scale factor relative to the 50 % probability surface."""
    p = max(0.001, min(99.999, float(probability))) / 100.0
    factor = math.sqrt(
        (-2.0 * math.log(1.0 - p)) / (-2.0 * math.log(0.5))
    )
    return max(0.001, min(6.0, factor))


def _ellipsoid_base_components(atom):
    """Return (semi-axes in BU, Euler XYZ rotation) at 50 % probability.

    The axes come from the Cartesian ADP tensor when present. Atoms without
    displacement parameters remain spherical using the ball-and-stick radius.
    """
    try:
        import numpy
    except ImportError:
        numpy = None

    values = atom.get("u_cart")
    if numpy is not None and values and len(values) == 6:
        u11, u22, u33, u12, u13, u23 = (float(value) for value in values)
        if (
            all(math.isfinite(value) for value in values)
            and u11 > 0.0 and u22 > 0.0 and u33 > 0.0
        ):
            tensor = numpy.array(
                (
                    (u11, u12, u13),
                    (u12, u22, u23),
                    (u13, u23, u33),
                ),
                dtype=float,
            )
            try:
                eigenvalues, eigenvectors = numpy.linalg.eigh(tensor)
                eigenvalues = numpy.clip(eigenvalues, 0.0, None)
                axes = numpy.sqrt(eigenvalues * _ELLIPSOID_SCALE_50)
                axes = numpy.maximum(axes * _DISPLAY_SCALE, 1e-4)
                if numpy.linalg.det(eigenvectors) < 0.0:
                    eigenvectors[:, 2] *= -1.0
                rotation = Matrix(eigenvectors.tolist()).to_euler('XYZ')
                return (
                    tuple(float(value) for value in axes),
                    (float(rotation.x), float(rotation.y), float(rotation.z)),
                )
            except Exception:
                pass

    radius = _atom_radius(
        atom.get("vdw_radius", _DEFAULT_VDW_RADIUS), 'BALL_STICK'
    )
    return (radius, radius, radius), (0.0, 0.0, 0.0)


def _make_ellipsoid_node_group():
    """Build (or fetch) the per-point ellipsoid instancing node group."""
    node_group = bpy.data.node_groups.get(_NODE_GROUP_ELLIPSOIDS)
    if node_group is not None:
        return node_group

    node_group = bpy.data.node_groups.new(
        _NODE_GROUP_ELLIPSOIDS, "GeometryNodeTree"
    )
    geometry_in = node_group.interface.new_socket(
        name="Geometry", in_out='INPUT', socket_type="NodeSocketGeometry"
    )
    material_in = node_group.interface.new_socket(
        name="Material", in_out='INPUT', socket_type="NodeSocketMaterial"
    )
    probability_in = node_group.interface.new_socket(
        name="Probability Scale", in_out='INPUT', socket_type="NodeSocketFloat"
    )
    probability_in.default_value = 1.0
    geometry_out = node_group.interface.new_socket(
        name="Geometry", in_out='OUTPUT', socket_type="NodeSocketGeometry"
    )

    nodes = node_group.nodes
    links = node_group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    set_material = nodes.new("GeometryNodeSetMaterial")
    smooth = nodes.new("GeometryNodeSetShadeSmooth")
    scale_attribute = nodes.new("GeometryNodeInputNamedAttribute")
    rotation_attribute = nodes.new("GeometryNodeInputNamedAttribute")
    vector_scale = nodes.new("ShaderNodeVectorMath")
    instance = nodes.new("GeometryNodeInstanceOnPoints")

    group_in.location = (-900, 0)
    group_out.location = (500, 0)
    ico.location = (-650, -180)
    set_material.location = (-420, -180)
    smooth.location = (-220, -180)
    scale_attribute.location = (-650, 80)
    vector_scale.location = (-420, 80)
    rotation_attribute.location = (-650, 260)
    instance.location = (0, 0)

    ico.inputs["Subdivisions"].default_value = 3
    ico.inputs["Radius"].default_value = 1.0
    scale_attribute.data_type = 'FLOAT_VECTOR'
    scale_attribute.inputs["Name"].default_value = "cgn_scale"
    rotation_attribute.data_type = 'FLOAT_VECTOR'
    rotation_attribute.inputs["Name"].default_value = "cgn_rotation"
    vector_scale.operation = 'SCALE'

    links.new(group_in.outputs[geometry_in.identifier], instance.inputs["Points"])
    links.new(ico.outputs["Mesh"], set_material.inputs["Geometry"])
    links.new(group_in.outputs[material_in.identifier], set_material.inputs["Material"])
    links.new(set_material.outputs["Geometry"], smooth.inputs["Geometry"])
    links.new(smooth.outputs["Geometry"], instance.inputs["Instance"])
    links.new(scale_attribute.outputs["Attribute"], vector_scale.inputs[0])
    links.new(group_in.outputs[probability_in.identifier], vector_scale.inputs["Scale"])
    links.new(vector_scale.outputs["Vector"], instance.inputs["Scale"])
    links.new(rotation_attribute.outputs["Attribute"], instance.inputs["Rotation"])
    links.new(instance.outputs["Instances"], group_out.inputs[geometry_out.identifier])
    return node_group


def _add_ellipsoid_attributes(mesh, atoms):
    """Store per-point semi-axes and rotations on an atom point mesh."""
    scale_attribute = mesh.attributes.new("cgn_scale", 'FLOAT_VECTOR', 'POINT')
    rotation_attribute = mesh.attributes.new(
        "cgn_rotation", 'FLOAT_VECTOR', 'POINT'
    )
    for index, atom in enumerate(atoms):
        axes, rotation = _ellipsoid_base_components(atom)
        scale_attribute.data[index].vector = axes
        rotation_attribute.data[index].vector = rotation


def _set_modifier_input(modifier, node_group, socket_name, value):
    identifier = _socket_identifier(node_group, socket_name)
    try:
        modifier[identifier] = value
    except TypeError:
        input_value = getattr(modifier.properties.inputs, identifier, None)
        if input_value is None:
            raise RuntimeError(
                f"Geometry Nodes modifier input '{socket_name}' is unavailable."
            )
        try:
            input_value["value"] = value
        except (TypeError, KeyError):
            input_value.value = value


def _make_material(name, color):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.use_nodes = True
    material.diffuse_color = color
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = 0.35
    return material


def _add_geometry_nodes_modifier(
    obj, node_group, radius=None, material=None, probability=None
):
    modifier = obj.modifiers.new("Geometry Nodes", 'NODES')
    modifier.node_group = node_group
    if radius is not None:
        _set_modifier_input(modifier, node_group, "Radius", radius)
    if material is not None:
        _set_modifier_input(modifier, node_group, "Material", material)
    if probability is not None:
        _set_modifier_input(
            modifier, node_group, "Probability Scale", probability
        )


def _clear_collection(collection):
    for child in list(collection.children):
        _clear_collection(child)
        bpy.data.collections.remove(child, do_unlink=True)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


_LEGACY_STRUCTURE_ERROR = (
    "This structure was imported by an older version of the add-on and has no "
    "site table. Re-import the CIF to grow or rebuild it."
)


def _materialize_atoms(sites, cell_matrix, instances, initial_count=0):
    """Expand the site table and instance quadruples into atom dictionaries.

    An instance is (site index, tx, ty, tz): the atom sits at the site's
    fractional position translated by that integer cell offset. Atom order
    follows the instance order, so selection indices stay valid. Only the
    atoms actually needed are built; nothing full-size is stored.
    """
    atoms = []
    count = len(instances) - len(instances) % 4
    for index, offset in enumerate(range(0, count, 4)):
        site_index = int(instances[offset])
        if not 0 <= site_index < len(sites):
            continue
        site = sites[site_index]
        translation = (
            int(instances[offset + 1]),
            int(instances[offset + 2]),
            int(instances[offset + 3]),
        )
        frac = [
            float(site["frac"][axis]) + translation[axis] for axis in range(3)
        ]
        cart = [
            frac[0] * cell_matrix[axis]
            + frac[1] * cell_matrix[3 + axis]
            + frac[2] * cell_matrix[6 + axis]
            for axis in range(3)
        ]
        atom = dict(site)
        atom.update({
            "frac": frac,
            "cart": cart,
            "p1_index": site_index,
            "growth_translation": list(translation),
        })
        if index >= initial_count:
            atom["label"] = (
                f"{site.get('source_label', site.get('label', 'X'))} "
                f"[P1:{site_index}; {translation[0]},"
                f"{translation[1]},{translation[2]}]"
            )
        atoms.append(atom)
    return atoms


def _structure_tables(root):
    """Return (sites, instances, cell_matrix, initial_count) for a structure."""
    sites_json = root.get("cgn_sites", "")
    instances_json = root.get("cgn_instances", "")
    matrix_json = root.get("cgn_cell_matrix", "")
    if not sites_json or not instances_json or not matrix_json:
        raise ValueError(_LEGACY_STRUCTURE_ERROR)
    sites = json.loads(sites_json)
    instances = json.loads(instances_json)
    cell_matrix = json.loads(matrix_json)
    if len(cell_matrix) != 9:
        raise ValueError("The stored unit-cell matrix is invalid.")
    initial_count = int(root.get("cgn_initial_count", 0))
    return sites, instances, cell_matrix, initial_count


def _structure_atoms(root):
    sites, instances, cell_matrix, initial_count = _structure_tables(root)
    return _materialize_atoms(sites, cell_matrix, instances, initial_count)


def _structure_atom_count(root):
    instances_json = root.get("cgn_instances", "")
    if not instances_json:
        return 0
    return len(json.loads(instances_json)) // 4


def _display_point(cart):
    """Convert an Ångström-space position to Blender display units."""
    return tuple(float(value) * _DISPLAY_SCALE for value in cart)


def _show_hydrogens():
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return True
    return bool(getattr(scene, "cgn_show_hydrogens", True))


def _apply_hydrogen_visibility(root, visible):
    """Show or hide every hydrogen atom object and its bond objects."""
    if root is None:
        return
    for obj in root.objects:
        role = obj.get("cgn_role")
        if role == "atoms":
            is_hydrogen = str(obj.get("cgn_element", "")).upper() == "H"
        elif role == "bonds":
            is_hydrogen = "H" in str(obj.get("cgn_pair", "")).upper().split("|")
        else:
            continue
        if is_hydrogen:
            obj.hide_viewport = not visible
            obj.hide_render = not visible


def _update_show_hydrogens(scene, context):
    _apply_hydrogen_visibility(
        _active_root(scene), scene.cgn_show_hydrogens
    )


def _update_model(scene, context):
    """Rebuild atoms and bonds when the rendering model changes."""
    root = _active_root(scene)
    if root is None:
        return
    try:
        sites, instances, cell_matrix, initial_count = _structure_tables(root)
        atoms = _materialize_atoms(
            sites, cell_matrix, instances, initial_count
        )
        _rebuild_atom_objects(root, atoms)
        _bond_objects(root, atoms, scene.cgn_connectivity_pairs)
        _apply_hydrogen_visibility(root, scene.cgn_show_hydrogens)
        context.view_layer.update()
    except Exception:
        traceback.print_exc()


def _update_ellipsoid_probability(scene, context):
    """Rescale existing ellipsoids without rebuilding geometry."""
    root = _active_root(scene)
    if root is None or _current_model(scene) != 'ELLIPSOID':
        return
    factor = _ellipsoid_probability_factor(
        scene.cgn_ellipsoid_probability
    )
    for obj in root.objects:
        if obj.get("cgn_role") != "atoms":
            continue
        for modifier in obj.modifiers:
            if (
                modifier.type == 'NODES'
                and modifier.node_group is not None
                and modifier.node_group.name == _NODE_GROUP_ELLIPSOIDS
            ):
                try:
                    _set_modifier_input(
                        modifier, modifier.node_group,
                        "Probability Scale", factor,
                    )
                except Exception:
                    traceback.print_exc()
    try:
        context.view_layer.update()
    except Exception:
        pass


def _make_atom_objects(root, atoms):
    by_element = {}
    for atom in atoms:
        element = str(atom.get("element", "X")).upper()
        by_element.setdefault(element, []).append(atom)

    model = _current_model()
    ellipsoids = model == 'ELLIPSOID'
    if ellipsoids:
        node_group = _make_ellipsoid_node_group()
        scene = getattr(bpy.context, "scene", None)
        probability = _ellipsoid_probability_factor(
            getattr(scene, "cgn_ellipsoid_probability", 50.0)
        )
    else:
        node_group = _make_node_group(_NODE_GROUP_ATOMS, is_bond_group=False)
        probability = None
    for element, element_atoms in sorted(by_element.items()):
        mesh = bpy.data.meshes.new(f"CGN_Atoms_{element}_Points")
        mesh.from_pydata(
            [_display_point(atom["cart"]) for atom in element_atoms], [], []
        )
        if ellipsoids:
            _add_ellipsoid_attributes(mesh, element_atoms)
        mesh.update()
        obj = bpy.data.objects.new(f"CGN_Atoms_{element}", mesh)
        obj["cgn_role"] = "atoms"
        obj["cgn_element"] = element
        root.objects.link(obj)
        if element == "H" and not _show_hydrogens():
            obj.hide_viewport = True
            obj.hide_render = True
        material = _make_material(
            f"CGN_Mat_{element}", _ELEMENT_COLORS.get(element, _DEFAULT_COLOR)
        )
        if ellipsoids:
            _add_geometry_nodes_modifier(
                obj, node_group, material=material, probability=probability
            )
        else:
            radius = _atom_radius(
                float(
                    element_atoms[0].get("vdw_radius", _DEFAULT_VDW_RADIUS)
                ),
                model,
            )
            _add_geometry_nodes_modifier(obj, node_group, radius, material)


def _rebuild_atom_objects(root, atoms):
    for obj in list(root.objects):
        if obj.get("cgn_role") == "atoms":
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    _make_atom_objects(root, atoms)


def _update_growth_selection(root, atoms, indices):
    root["cgn_selected_indices"] = json.dumps(sorted(set(indices)))
    for obj in list(root.objects):
        if obj.get("cgn_role") == "growth_selection":
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    selected_atoms = [
        atoms[index] for index in sorted(set(indices))
        if 0 <= index < len(atoms)
    ]
    if not selected_atoms:
        return

    mesh = bpy.data.meshes.new("CGN_GrowthSelection_Points")
    mesh.from_pydata(
        [_display_point(atom["cart"]) for atom in selected_atoms], [], []
    )
    mesh.update()
    obj = bpy.data.objects.new("CGN_GrowthSelection", mesh)
    obj["cgn_role"] = "growth_selection"
    root.objects.link(obj)
    material = _make_material("CGN_GrowthSelection_Material", (1.0, 0.48, 0.02, 1.0))
    _add_geometry_nodes_modifier(
        obj, _make_node_group(_NODE_GROUP_ATOMS, is_bond_group=False),
        0.9 * _DISPLAY_SCALE, material,
    )


def _unit_cell_values(settings=None):
    """Return (face_rgba, face_alpha, edge_rgba, edge_thickness)."""
    if settings is None:
        return (
            tuple(_UNIT_CELL_FACE_COLOR),
            _UNIT_CELL_FACE_ALPHA,
            tuple(_UNIT_CELL_EDGE_COLOR),
            _UNIT_CELL_EDGE_THICKNESS,
        )
    face = tuple(float(channel) for channel in settings.face_color)
    alpha = max(0.0, min(1.0, 1.0 - float(settings.face_transparency)))
    edge = tuple(float(channel) for channel in settings.edge_color)
    thickness = max(0.0, min(1.0, float(settings.edge_thickness)))
    return face, alpha, edge, thickness


def _make_unit_cell_face_material(settings=None):
    material = bpy.data.materials.get("CGN_UnitCell_Face_Material")
    if material is None:
        material = bpy.data.materials.new("CGN_UnitCell_Face_Material")
    material.use_nodes = True
    color, alpha, _edge, _thickness = _unit_cell_values(settings)
    material.diffuse_color = (color[0], color[1], color[2], alpha)
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = (
            color[0], color[1], color[2], 1.0,
        )
        shader.inputs["Roughness"].default_value = 0.5
        if "Alpha" in shader.inputs:
            shader.inputs["Alpha"].default_value = alpha
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = 'BLENDED'
    elif hasattr(material, "blend_method"):
        material.blend_method = 'BLEND'
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = True
    material.use_backface_culling = False
    return material


def _make_unit_cell_edge_material(settings=None):
    material = bpy.data.materials.get("CGN_UnitCell_Edge_Material")
    if material is None:
        material = bpy.data.materials.new("CGN_UnitCell_Edge_Material")
    material.use_nodes = True
    _face, _alpha, color, _thickness = _unit_cell_values(settings)
    material.diffuse_color = color
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = 0.5
    return material


def _cell_corner(cell_matrix, fx, fy, fz):
    return tuple(
        fx * cell_matrix[axis]
        + fy * cell_matrix[3 + axis]
        + fz * cell_matrix[6 + axis]
        for axis in range(3)
    )


def _clear_unit_cell(root):
    for obj in list(root.objects):
        if obj.get("cgn_role") not in {"unit_cell_edge", "unit_cell_face"}:
            continue
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is None or data.users != 0:
            continue
        if isinstance(data, bpy.types.Curve):
            bpy.data.curves.remove(data)
        elif isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)


def _stored_cell_matrix(root):
    raw = root.get("cgn_cell_matrix", "")
    if not raw:
        return None
    try:
        matrix = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 9:
        return None
    return matrix


def _show_unit_cell(root, cell_matrix, settings=None):
    """Draw the original unit cell: transparent faces and solid black edges."""
    _clear_unit_cell(root)
    _face, _alpha, _edge, thickness = _unit_cell_values(settings)
    corners = [
        _display_point(_cell_corner(cell_matrix, i, j, k))
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    edges = [
        (index, index ^ (1 << bit))
        for index in range(8)
        for bit in range(3)
        if index < (index ^ (1 << bit))
    ]

    curve = bpy.data.curves.new("CGN_UnitCell_Edges", 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 1
    curve.bevel_resolution = 2
    curve.bevel_depth = thickness * _DISPLAY_SCALE
    for first, second in edges:
        spline = curve.splines.new('POLY')
        spline.points.add(1)
        spline.points[0].co = (*corners[first], 1.0)
        spline.points[1].co = (*corners[second], 1.0)
    curve.materials.append(_make_unit_cell_edge_material(settings))
    edge_obj = bpy.data.objects.new("CGN_UnitCell_Edges", curve)
    edge_obj["cgn_role"] = "unit_cell_edge"
    root.objects.link(edge_obj)

    faces = [
        (0, 1, 3, 2), (4, 6, 7, 5),
        (0, 4, 5, 1), (2, 3, 7, 6),
        (0, 2, 6, 4), (1, 5, 7, 3),
    ]
    mesh = bpy.data.meshes.new("CGN_UnitCell_Faces")
    mesh.from_pydata(corners, [], faces)
    mesh.update()
    mesh.materials.append(_make_unit_cell_face_material(settings))
    face_obj = bpy.data.objects.new("CGN_UnitCell_Faces", mesh)
    face_obj["cgn_role"] = "unit_cell_face"
    root.objects.link(face_obj)


def _update_unit_cell_objects(edge_obj, face_obj, settings):
    _face, _alpha, _edge, thickness = _unit_cell_values(settings)
    if isinstance(edge_obj.data, bpy.types.Curve):
        edge_obj.data.bevel_depth = thickness * _DISPLAY_SCALE
    for obj, material in (
        (edge_obj, _make_unit_cell_edge_material(settings)),
        (face_obj, _make_unit_cell_face_material(settings)),
    ):
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)


def _apply_unit_cell_settings(scene):
    """Show, hide, or restyle the active structure's unit-cell box."""
    root = _active_root(scene)
    if root is None:
        return
    settings = getattr(scene, "cgn_unit_cell_settings", None)
    if settings is not None and not settings.show_unit_cell:
        _clear_unit_cell(root)
        return
    cell_matrix = _stored_cell_matrix(root)
    if not cell_matrix:
        _clear_unit_cell(root)
        return
    edge_obj = next(
        (
            obj for obj in root.objects
            if obj.get("cgn_role") == "unit_cell_edge"
        ),
        None,
    )
    face_obj = next(
        (
            obj for obj in root.objects
            if obj.get("cgn_role") == "unit_cell_face"
        ),
        None,
    )
    if edge_obj is None or face_obj is None:
        _show_unit_cell(root, cell_matrix, settings)
        return
    _update_unit_cell_objects(edge_obj, face_obj, settings)


def _pair_key(element_a, element_b):
    return tuple(sorted((str(element_a).upper(), str(element_b).upper())))


def _bond_objects(root, atoms, pair_settings):
    for obj in list(root.objects):
        if obj.get("cgn_role") == "bonds":
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)

    model = _current_model()
    if model == 'SPACE_FILLING':
        return 0

    enabled = {}
    for setting in pair_settings:
        key = _pair_key(setting.element_a, setting.element_b)
        if setting.enabled and setting.max_distance >= setting.min_distance:
            enabled[key] = (setting.min_distance, setting.max_distance)
    if not enabled or len(atoms) < 2:
        return 0

    max_distance = max(pair[1] for pair in enabled.values())
    if max_distance <= 0:
        return 0
    cell_width = max(max_distance, 1e-6)
    buckets = {}
    by_pair = {}
    vdw_by_element = {}
    for atom in atoms:
        element = str(atom.get("element", "X")).upper()
        vdw_by_element.setdefault(
            element, float(atom.get("vdw_radius", _DEFAULT_VDW_RADIUS))
        )
    for index, atom in enumerate(atoms):
        x, y, z = atom["cart"]
        cell = (
            math.floor(x / cell_width),
            math.floor(y / cell_width),
            math.floor(z / cell_width),
        )
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for other_index in buckets.get(neighbor_cell, ()):
                        other = atoms[other_index]
                        key = _pair_key(atom.get("element"), other.get("element"))
                        bounds = enabled.get(key)
                        if bounds is None:
                            continue
                        delta = (
                            x - other["cart"][0],
                            y - other["cart"][1],
                            z - other["cart"][2],
                        )
                        distance_sq = sum(component * component for component in delta)
                        if distance_sq <= 1e-12:
                            continue
                        distance = math.sqrt(distance_sq)
                        if bounds[0] <= distance <= bounds[1]:
                            by_pair.setdefault(key, []).append(
                                (other["cart"], atom["cart"])
                            )
        buckets.setdefault(cell, []).append(index)

    node_group = _make_node_group(_NODE_GROUP_BONDS, is_bond_group=True)
    bond_count = 0
    for key, segments in sorted(by_pair.items()):
        vertices = []
        edges = []
        for first, second in segments:
            start = len(vertices)
            vertices.extend((_display_point(first), _display_point(second)))
            edges.append((start, start + 1))
        if not edges:
            continue
        pair_name = "_".join(key)
        mesh = bpy.data.meshes.new(f"CGN_Bonds_{pair_name}_Edges")
        mesh.from_pydata(vertices, edges, [])
        mesh.update()
        obj = bpy.data.objects.new(f"CGN_Bonds_{pair_name}", mesh)
        obj["cgn_role"] = "bonds"
        obj["cgn_pair"] = "|".join(key)
        root.objects.link(obj)
        if "H" in key and not _show_hydrogens():
            obj.hide_viewport = True
            obj.hide_render = True
        material = _make_material(
            f"CGN_Bond_Mat_{pair_name}", (0.16, 0.16, 0.18, 1.0)
        )
        vdw_a = vdw_by_element.get(key[0], _DEFAULT_VDW_RADIUS)
        vdw_b = vdw_by_element.get(key[1], _DEFAULT_VDW_RADIUS)
        bond_radius = _bond_radius(vdw_a, vdw_b, model)
        _add_geometry_nodes_modifier(
            obj, node_group, bond_radius, material
        )
        bond_count += len(edges)
    return bond_count


def _set_scene_units(scene):
    """Display the scene at 1 Å = 1 cm (1 Blender unit = 10 Å)."""
    units = scene.unit_settings
    units.system = 'METRIC'
    units.scale_length = 0.1
    units.length_unit = 'CENTIMETERS'


def _import_result(scene, result, cif_path):
    sites = result.get("sites", [])
    instances = result.get("instances", [])
    cell_matrix = result.get("cell_matrix", [])
    if len(cell_matrix) != 9:
        raise ValueError("The CIF reader returned an invalid cell matrix.")
    initial_count = len(instances) // 4
    atoms = _materialize_atoms(sites, cell_matrix, instances, initial_count)
    if not atoms:
        raise ValueError("The selected CIF contains no atoms.")

    old_root = _active_root(scene)
    root = bpy.data.collections.new("CCTBX_GN_Structure")
    root["cgn_role"] = "structure"
    root["cgn_instance_id"] = uuid.uuid4().hex
    root["cgn_space_group"] = result.get("space_group", "unknown")
    root["cgn_space_group_ops"] = json.dumps(
        result.get("space_group_ops", [])
    )
    root["cgn_unit_cell"] = json.dumps(result.get("unit_cell", []))
    root["cgn_cif_block"] = result.get("label", "")
    root["cgn_cif_path"] = cif_path
    root["cgn_sites"] = json.dumps(sites)
    root["cgn_instances"] = json.dumps(instances)
    root["cgn_asu_instances"] = json.dumps(instances)
    root["cgn_cell_matrix"] = json.dumps(cell_matrix)
    root["cgn_initial_count"] = initial_count
    root["cgn_selected_indices"] = "[]"
    root["cgn_growth_shells"] = 0
    root["cgn_fill_mode"] = "ASU"
    root["cgn_fill_repeats"] = json.dumps([2, 2, 2])
    scene.collection.children.link(root)
    old_pairs = [
        (
            pair.element_a, pair.element_b, pair.enabled,
            pair.min_distance, pair.max_distance,
        )
        for pair in scene.cgn_connectivity_pairs
    ]
    try:
        _make_atom_objects(root, atoms)
        scene.cgn_connectivity_pairs.clear()
        elements = sorted({str(a.get("element", "X")).upper() for a in atoms})
        radii = {}
        for atom in atoms:
            element = str(atom.get("element", "X")).upper()
            radii[element] = float(atom.get("covalent_radius", 0.75))
        for index, element_a in enumerate(elements):
            for element_b in elements[index:]:
                item = scene.cgn_connectivity_pairs.add()
                item.element_a = element_a
                item.element_b = element_b
                item.enabled = True
                item.min_distance = 0.1
                item.max_distance = min(
                    20.0, radii[element_a] + radii[element_b] + _BOND_TOLERANCE
                )
        _bond_objects(root, atoms, scene.cgn_connectivity_pairs)
        if old_root is not None:
            _clear_collection(old_root)
            bpy.data.collections.remove(old_root, do_unlink=True)
        scene.cgn_active_structure = root.name
        _set_scene_units(scene)
        _apply_unit_cell_settings(scene)
    except Exception:
        if bpy.data.collections.get(root.name) == root:
            _clear_collection(root)
            bpy.data.collections.remove(root, do_unlink=True)
        scene.cgn_connectivity_pairs.clear()
        for element_a, element_b, enabled, minimum, maximum in old_pairs:
            item = scene.cgn_connectivity_pairs.add()
            item.element_a = element_a
            item.element_b = element_b
            item.enabled = enabled
            item.min_distance = minimum
            item.max_distance = maximum
        raise
    return root


def _unwrapped_cell_instances(
    sites, cell_matrix, pair_settings, max_instances=_MAX_FILL_ATOMS
):
    """Return the cell contents plus the atoms that complete whole molecules.

    The P1 site table stores every atom inside the cell, so a molecule that
    crosses a cell boundary is split into fragments. This walks the periodic
    bond graph and assigns each atom an integer cell translation so bonded
    atoms stay together. Every cell atom is then kept at its wrapped position,
    and each molecule fragment that is visible in the cell is completed by
    adding the missing atoms just outside it. Periodic frameworks cannot be
    unwrapped into finite molecules and keep their plain cell contents.
    """
    windows = {}
    for setting in pair_settings:
        if setting.enabled and setting.max_distance >= setting.min_distance:
            windows[_pair_key(setting.element_a, setting.element_b)] = (
                setting.min_distance, setting.max_distance,
            )
    if not windows:
        return None
    if len(sites) > max_instances:
        raise ValueError(
            f"Molecule completion exceeds the {max_instances}-atom fill limit."
        )
    max_distance = max(bound[1] for bound in windows.values())
    if max_distance <= 0.0:
        return None

    try:
        matrix = Matrix((
            (cell_matrix[0], cell_matrix[3], cell_matrix[6]),
            (cell_matrix[1], cell_matrix[4], cell_matrix[7]),
            (cell_matrix[2], cell_matrix[5], cell_matrix[8]),
        ))
        inverse = matrix.inverted()
    except (ValueError, ZeroDivisionError):
        return None

    # A Cartesian move of max_distance cannot change a fractional coordinate
    # by more than max_distance times the length of that row of the inverse.
    reach = [
        max_distance * inverse.row[axis].length for axis in range(3)
    ]
    cells = [max(1, int(math.ceil(value))) for value in reach]

    fracs = [Vector(site["frac"]) for site in sites]
    carts = [matrix @ frac for frac in fracs]
    elements = [str(site.get("element", "X")).upper() for site in sites]
    width = max(max_distance, 1e-6)

    bank = {}
    for index, frac in enumerate(fracs):
        for tx in range(-cells[0], cells[0] + 1):
            for ty in range(-cells[1], cells[1] + 1):
                for tz in range(-cells[2], cells[2] + 1):
                    image_frac = Vector((
                        frac.x + tx, frac.y + ty, frac.z + tz,
                    ))
                    image_cart = matrix @ image_frac
                    key = (
                        math.floor(image_cart.x / width),
                        math.floor(image_cart.y / width),
                        math.floor(image_cart.z / width),
                    )
                    bank.setdefault(key, []).append(
                        (index, (tx, ty, tz), image_cart)
                    )

    adjacency = [[] for _ in sites]
    seen = set()
    for index, cart in enumerate(carts):
        key = (
            math.floor(cart.x / width),
            math.floor(cart.y / width),
            math.floor(cart.z / width),
        )
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbour = (key[0] + dx, key[1] + dy, key[2] + dz)
                    for other_index, translation, image_cart in bank.get(
                        neighbour, ()
                    ):
                        bounds = windows.get(
                            _pair_key(elements[index], elements[other_index])
                        )
                        if bounds is None:
                            continue
                        distance = (image_cart - cart).length
                        if distance <= 1e-8:
                            continue
                        if not (bounds[0] <= distance <= bounds[1]):
                            continue
                        if (
                            other_index == index
                            and translation == (0, 0, 0)
                        ):
                            continue
                        edge = (index, other_index, translation)
                        if edge in seen:
                            continue
                        reverse = (
                            other_index, index,
                            (-translation[0], -translation[1],
                             -translation[2]),
                        )
                        seen.add(edge)
                        seen.add(reverse)
                        adjacency[index].append((other_index, translation))
                        adjacency[other_index].append((index, reverse[2]))

    translations = [None] * len(sites)
    components = []
    periodic = []
    for start in range(len(sites)):
        if translations[start] is not None:
            continue
        translations[start] = (0, 0, 0)
        component = []
        is_periodic = False
        queue = [start]
        while queue:
            current = queue.pop()
            component.append(current)
            current_t = translations[current]
            for other, translation in adjacency[current]:
                expected = (
                    current_t[0] + translation[0],
                    current_t[1] + translation[1],
                    current_t[2] + translation[2],
                )
                if translations[other] is None:
                    translations[other] = expected
                    queue.append(other)
                elif translations[other] != expected:
                    # A cycle whose net translation is not zero means the
                    # component runs through the cell in every direction (a
                    # framework). Such a structure cannot be unwrapped into a
                    # finite molecule, so it keeps its plain cell contents.
                    is_periodic = True
        components.append(component)
        if is_periodic:
            periodic.append(len(components) - 1)

    instances = {(index, 0, 0, 0) for index in range(len(sites))}
    for component_index, component in enumerate(components):
        if component_index in periodic:
            continue
        anchors = {
            (-translations[index][0], -translations[index][1],
             -translations[index][2])
            for index in component
        }
        for anchor in anchors:
            for index in component:
                translation = translations[index]
                instances.add((
                    index,
                    translation[0] + anchor[0],
                    translation[1] + anchor[1],
                    translation[2] + anchor[2],
                ))
                if len(instances) > max_instances:
                    raise ValueError(
                        f"Molecule completion exceeds the "
                        f"{max_instances}-atom fill limit."
                    )

    return [
        value
        for instance in sorted(instances)
        for value in instance
    ]


def _fill_structure(operator, context, mode, repeats=(1, 1, 1)):
    """Replace displayed atoms with the ASU, completed cell, or packed cells."""
    root = _active_root(context.scene)
    if root is None:
        operator.report({'ERROR'}, "Import a CIF before filling the structure.")
        return {'CANCELLED'}
    try:
        sites, _instances, cell_matrix, _initial_count = _structure_tables(root)
    except (ValueError, TypeError) as exc:
        operator.report({'ERROR'}, str(exc))
        return {'CANCELLED'}
    if not sites:
        operator.report({'ERROR'}, "The stored site table is empty.")
        return {'CANCELLED'}

    if mode == 'ASU':
        baseline = root.get("cgn_asu_instances", "")
        if not baseline:
            operator.report(
                {'ERROR'},
                "Re-import this CIF once to enable the Fill menu.",
            )
            return {'CANCELLED'}
        try:
            new_instances = json.loads(baseline)
        except (ValueError, TypeError) as exc:
            operator.report(
                {'ERROR'}, f"Stored asymmetric-unit table is invalid: {exc}"
            )
            return {'CANCELLED'}
        if not new_instances:
            operator.report(
                {'ERROR'},
                "Re-import this CIF once to enable the Fill menu.",
            )
            return {'CANCELLED'}
        repeats = (1, 1, 1)
        description = "asymmetric unit"
    else:
        if mode == 'PACK':
            repeats = tuple(
                max(1, min(_MAX_FILL_REPEATS, int(value)))
                for value in repeats
            )
            description = (
                f"{repeats[0]}×{repeats[1]}×{repeats[2]} unit cells"
            )
        else:
            repeats = (1, 1, 1)
            description = "one unit cell"

        try:
            cell_instances = _unwrapped_cell_instances(
                sites,
                cell_matrix,
                context.scene.cgn_connectivity_pairs,
            )
        except ValueError as exc:
            operator.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        if cell_instances is None:
            cell_instances = [
                value
                for site_index in range(len(sites))
                for value in (site_index, 0, 0, 0)
            ]
        cell_atom_count = len(cell_instances) // 4

        seen = set()
        new_instances = []
        for tx in range(repeats[0]):
            for ty in range(repeats[1]):
                for tz in range(repeats[2]):
                    for offset in range(0, len(cell_instances), 4):
                        key = (
                            cell_instances[offset],
                            cell_instances[offset + 1] + tx,
                            cell_instances[offset + 2] + ty,
                            cell_instances[offset + 3] + tz,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        if len(seen) > _MAX_FILL_ATOMS:
                            operator.report(
                                {'ERROR'},
                                f"This fill would create more than "
                                f"{_MAX_FILL_ATOMS} atoms; reduce the repeats.",
                            )
                            return {'CANCELLED'}
                        new_instances.extend(key)

    atoms = _materialize_atoms(
        sites, cell_matrix, new_instances, len(new_instances) // 4
    )
    if not atoms:
        operator.report({'ERROR'}, "The fill produced no atoms.")
        return {'CANCELLED'}
    try:
        _rebuild_atom_objects(root, atoms)
        _update_growth_selection(root, atoms, [])
        _bond_objects(root, atoms, context.scene.cgn_connectivity_pairs)
        _apply_unit_cell_settings(context.scene)
    except Exception as exc:
        traceback.print_exc()
        operator.report({'ERROR'}, str(exc))
        return {'CANCELLED'}

    root["cgn_instances"] = json.dumps(new_instances)
    root["cgn_initial_count"] = len(atoms)
    root["cgn_fill_mode"] = mode
    if mode != 'ASU':
        root["cgn_cell_atoms"] = cell_atom_count
    if mode == 'PACK':
        root["cgn_fill_repeats"] = json.dumps(list(repeats))
    root["cgn_growth_shells"] = 0
    operator.report(
        {'INFO'}, f"Filled {len(atoms)} atoms ({description})."
    )
    return {'FINISHED'}


class CGN_OT_import_cif(Operator):
    bl_idname = "cgn.import_cif"
    bl_label = "Import CIF"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype="FILE_PATH")
    block_name: StringProperty(
        name="CIF Block",
        description="Optional exact data-block name for multi-structure CIF files",
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "The selected CIF file does not exist.")
            return {'CANCELLED'}
        try:
            worker = _gemmi_worker()
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        try:
            result = worker.load_structure(
                bpy.path.abspath(self.filepath), self.block_name or None
            )
            root = _import_result(context.scene, result, self.filepath)
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            f"Imported {len(result.get('instances', [])) // 4} ASU atoms "
            f"({root.get('cgn_space_group')}).",
        )
        return {'FINISHED'}


class CGN_OT_rebuild_connectivity(Operator):
    bl_idname = "cgn.rebuild_connectivity"
    bl_label = "Rebuild Bonds"
    bl_description = "Apply enabled element-pair distance windows to the imported atoms"

    def execute(self, context):
        root = _active_root(context.scene)
        if root is None:
            self.report({'ERROR'}, "Import a CIF before rebuilding connectivity.")
            return {'CANCELLED'}
        try:
            atoms = _structure_atoms(root)
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        count = _bond_objects(root, atoms, context.scene.cgn_connectivity_pairs)
        self.report({'INFO'}, f"Built {count} bonds using the current distance windows.")
        return {'FINISHED'}


class CGN_OT_fill_asu(Operator):
    bl_idname = "cgn.fill_asu"
    bl_label = "Fill Asymmetric Unit-cell"
    bl_description = "Show only the atoms of the CIF asymmetric unit"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return _fill_structure(self, context, 'ASU')


class CGN_OT_fill_cell(Operator):
    bl_idname = "cgn.fill_cell"
    bl_label = "Fill Unit-Cell"
    bl_description = (
        "Show one unit cell and complete finite molecules across its boundaries"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return _fill_structure(self, context, 'CELL')


class CGN_OT_fill_pack(Operator):
    bl_idname = "cgn.fill_pack"
    bl_label = "Pack Unit-cells"
    bl_description = (
        "Show repeated unit cells with finite molecules completed at the edges"
    )
    bl_options = {'REGISTER', 'UNDO'}

    repeat_a: IntProperty(
        name="a",
        description="Number of unit cells along a",
        default=2,
        min=1,
        max=_MAX_FILL_REPEATS,
    )
    repeat_b: IntProperty(
        name="b",
        description="Number of unit cells along b",
        default=2,
        min=1,
        max=_MAX_FILL_REPEATS,
    )
    repeat_c: IntProperty(
        name="c",
        description="Number of unit cells along c",
        default=2,
        min=1,
        max=_MAX_FILL_REPEATS,
    )

    def invoke(self, context, event):
        root = _active_root(context.scene)
        if root is not None:
            try:
                repeats = [
                    int(value) for value in json.loads(
                        root.get("cgn_fill_repeats", "[]")
                    )
                ]
            except (ValueError, TypeError):
                repeats = []
            if len(repeats) == 3:
                self.repeat_a = max(
                    1, min(_MAX_FILL_REPEATS, repeats[0])
                )
                self.repeat_b = max(
                    1, min(_MAX_FILL_REPEATS, repeats[1])
                )
                self.repeat_c = max(
                    1, min(_MAX_FILL_REPEATS, repeats[2])
                )
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Number of unit cells along each axis:")
        row = layout.row(align=True)
        row.prop(self, "repeat_a")
        row.prop(self, "repeat_b")
        row.prop(self, "repeat_c")
        root = _active_root(context.scene)
        if root is not None:
            try:
                sites = json.loads(root.get("cgn_sites", "[]"))
            except (ValueError, TypeError):
                sites = []
            if sites:
                cell_atoms = int(root.get("cgn_cell_atoms", len(sites)))
                count = (
                    cell_atoms * self.repeat_a * self.repeat_b
                    * self.repeat_c
                )
                layout.label(text=f"Atoms (approx.): {count}")

    def execute(self, context):
        return _fill_structure(
            self, context, 'PACK',
            (self.repeat_a, self.repeat_b, self.repeat_c),
        )


class CGN_MT_fill(Menu):
    bl_idname = "CGN_MT_fill"
    bl_label = "Fill"

    def draw(self, context):
        layout = self.layout
        layout.operator("cgn.fill_asu", text="Asymmetric Unit-cell")
        layout.operator("cgn.fill_cell", text="Unit-Cell")
        layout.operator("cgn.fill_pack", text="Pack Unit-cells...")


def _visible_atom_mask(atoms, root):
    """Return a mask of atoms whose element object is visible in the viewport."""
    count = len(atoms)
    if count == 0:
        return np.empty(0, dtype=bool)
    if root is None or root.hide_viewport:
        return np.zeros(count, dtype=bool)

    hidden_elements = set()
    for obj in root.objects:
        if obj.get("cgn_role") != "atoms":
            continue
        try:
            hidden = obj.hide_viewport or obj.hide_get()
        except RuntimeError:
            hidden = obj.hide_viewport
        if hidden:
            hidden_elements.add(str(obj.get("cgn_element", "")).upper())
    if not hidden_elements:
        return np.ones(count, dtype=bool)
    return np.fromiter(
        (
            str(atom.get("element", "X")).upper() not in hidden_elements
            for atom in atoms
        ),
        dtype=bool,
        count=count,
    )


def _project_points(coords, matrix, width, height):
    """Project homogeneous points to region pixels with NumPy.

    Returns atom indices, pixel positions and NDC depth values for points in
    front of the camera and inside the region. Smaller depth values are
    nearer to the camera.
    """
    if len(coords) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    clip = coords @ np.asarray(matrix, dtype=np.float64).T
    w = clip[:, 3]
    front = np.nonzero(w > 0.0)[0]
    if front.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    ndc = clip[front, :3] / w[front, None]
    x = width * 0.5 * (ndc[:, 0] + 1.0)
    y = height * 0.5 * (ndc[:, 1] + 1.0)
    inside = (
        (x >= 0.0) & (x < width)
        & (y >= 0.0) & (y < height)
    )
    selected = front[inside]
    points = np.column_stack((x[inside], y[inside]))
    return selected, points, ndc[inside, 2]


def _click_candidate(indices, points, depths, visible, origin, max_distance):
    """Return the frontmost visible atom within *max_distance* pixels."""
    if len(indices) == 0:
        return None
    distance = np.hypot(
        points[:, 0] - float(origin[0]),
        points[:, 1] - float(origin[1]),
    )
    candidates = np.nonzero(visible & (distance <= max_distance))[0]
    if candidates.size == 0:
        return None
    # lexsort uses the last key first: depth, then pixel distance.
    best = candidates[
        np.lexsort((distance[candidates], depths[candidates]))[0]
    ]
    return int(indices[best])


def _select_click(selected, candidate, additive):
    """Apply one click to *selected*: toggle the atom or replace/add."""
    if candidate is not None and candidate in selected:
        # Clicking an already selected atom toggles it off without
        # disturbing the rest of the selection.
        selected.discard(candidate)
    else:
        if not additive:
            selected.clear()
        if candidate is not None:
            selected.add(candidate)


class _CGN_OT_select_atoms(Operator):
    """Shared click/box atom selection modal for the grow and delete tools."""

    max_pixel_distance = 20.0
    box_drag_threshold = 6.0
    select_error = "Import a CIF before selecting atoms."
    status_text = ""
    keep_stored_selection = True

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run atom selection from the 3D Viewport.")
            return {'CANCELLED'}
        root = _active_root(context.scene)
        if root is None:
            self.report({'ERROR'}, self.select_error)
            return {'CANCELLED'}
        self._region = next(
            (region for region in context.area.regions if region.type == 'WINDOW'),
            None,
        )
        if self._region is None:
            self.report({'ERROR'}, "The 3D Viewport window region is unavailable.")
            return {'CANCELLED'}

        self._root_name = root.name
        self._root_id = root.get("cgn_instance_id")
        try:
            self._atoms = _structure_atoms(root)
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        try:
            self._initial_selection = set(json.loads(
                root.get("cgn_selected_indices", "[]")
            ))
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Stored atom selection is invalid: {exc}")
            return {'CANCELLED'}
        self._selected = (
            set(self._initial_selection) if self.keep_stored_selection else set()
        )
        self._drag_start = None
        self._drag_end = None
        self._drag_additive = False
        self._dragging = False
        self._projection_coords = np.asarray(
            [(*_display_point(atom["cart"]), 1.0) for atom in self._atoms],
            dtype=np.float64,
        ).reshape((-1, 4))
        self._atom_visibility = _visible_atom_mask(self._atoms, root)
        self._projection_cache_key = None
        self._projection_cache = None
        _update_growth_selection(root, self._atoms, self._selected)
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_box, (), 'WINDOW', 'POST_PIXEL'
        )
        context.workspace.status_text_set(self.status_text)
        context.window.cursor_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _finish(self, context, cancel=False):
        if cancel:
            self._selected = set(self._initial_selection)
            root = bpy.data.collections.get(self._root_name)
            if (
                root is not None
                and root.get("cgn_instance_id") == self._root_id
            ):
                _update_growth_selection(
                    root, self._atoms, self._selected
                )
        if getattr(self, "_draw_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(
                self._draw_handle, 'WINDOW'
            )
            self._draw_handle = None
        context.workspace.status_text_set(None)
        context.window.cursor_set('DEFAULT')
        if context.area is not None:
            context.area.tag_redraw()

    def _region_mouse(self, event, clamp=False):
        point = Vector((
            event.mouse_x - self._region.x,
            event.mouse_y - self._region.y,
        ))
        if clamp:
            point.x = min(max(point.x, 0), self._region.width - 1)
            point.y = min(max(point.y, 0), self._region.height - 1)
            return point
        if (
            0 <= point.x < self._region.width
            and 0 <= point.y < self._region.height
        ):
            return point
        return None

    def _project_atoms(self, context):
        """Return cached projected atoms as NumPy arrays."""
        region_3d = context.area.spaces.active.region_3d
        matrix = region_3d.perspective_matrix
        key = (
            self._region.width,
            self._region.height,
            matrix.copy(),
        )
        if self._projection_cache_key != key:
            indices, points, depths = _project_points(
                self._projection_coords,
                matrix,
                self._region.width,
                self._region.height,
            )
            self._projection_cache_key = key
            self._projection_cache = (
                indices,
                points,
                depths,
                self._atom_visibility[indices],
            )
        return self._projection_cache

    def _draw_box(self):
        if not self._dragging or self._drag_start is None or self._drag_end is None:
            return
        x_min, x_max = sorted((self._drag_start.x, self._drag_end.x))
        y_min, y_max = sorted((self._drag_start.y, self._drag_end.y))
        points = (
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_min, y_min),
        )
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": points})
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        try:
            shader.bind()
            shader.uniform_float("color", (1.0, 0.55, 0.05, 0.9))
            batch.draw(shader)
        finally:
            gpu.state.line_width_set(1.0)
            gpu.state.blend_set('NONE')

    def _apply_selection(self, context):
        if self._drag_start is None or self._drag_end is None:
            return
        root = bpy.data.collections.get(self._root_name)
        if (
            root is None
            or root.get("cgn_instance_id") != self._root_id
        ):
            self.report({'ERROR'}, "The active structure changed during selection.")
            return

        if self._dragging:
            if not self._drag_additive:
                self._selected.clear()
            x_min, x_max = sorted((self._drag_start.x, self._drag_end.x))
            y_min, y_max = sorted((self._drag_start.y, self._drag_end.y))
            indices, points, _depths, _visible = self._project_atoms(context)
            boxed = (
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
            )
            self._selected.update(int(index) for index in indices[boxed])
        else:
            indices, points, depths, visible = self._project_atoms(context)
            nearest = _click_candidate(
                indices, points, depths, visible,
                self._drag_start, self.max_pixel_distance,
            )
            _select_click(self._selected, nearest, self._drag_additive)

        _update_growth_selection(root, self._atoms, self._selected)
        context.area.tag_redraw()

    def modal(self, context, event):
        if event.type == 'ESC':
            self._finish(context, cancel=True)
            return {'CANCELLED'}
        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            return self._confirm(context)
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            return self._confirm(context)

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            point = self._region_mouse(event)
            if point is not None:
                self._drag_start = point
                self._drag_end = point.copy()
                self._drag_additive = event.shift
                self._dragging = False
            return {'RUNNING_MODAL'}
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'} and self._drag_start is not None:
            point = self._region_mouse(event, clamp=True)
            self._drag_end = point
            if (self._drag_end - self._drag_start).length >= self.box_drag_threshold:
                self._dragging = True
            if context.area is not None:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self._drag_start is not None:
                point = self._region_mouse(event, clamp=True)
                if point is not None:
                    self._drag_end = point
                self._apply_selection(context)
                self._drag_start = None
                self._drag_end = None
                self._dragging = False
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}


def _grow_from_selection():
    """Run the grow operator after the selection modal has fully finished."""
    try:
        bpy.ops.cgn.grow_further('INVOKE_DEFAULT', all_atoms=False)
    except RuntimeError as exc:
        print(f"CCTBX Geometry Nodes: grow failed: {exc}")
    return None


class CGN_OT_select_growth_atoms(_CGN_OT_select_atoms):
    bl_idname = "cgn.select_growth_atoms"
    bl_label = "Grow Selected Atoms"
    bl_description = (
        "Select atoms in the viewport, then confirm to grow around them; "
        "click a selected atom to remove it, Shift adds to the selection"
    )

    select_error = "Import a CIF before selecting growth atoms."
    status_text = (
        "Click atoms to add/remove, drag a box to select; Shift adds. "
        "Enter/right-click grows around them, Esc cancels."
    )

    def _confirm(self, context):
        if not self._selected:
            self._finish(context)
            self.report({'ERROR'}, "Select one or more atoms before growing.")
            return {'CANCELLED'}
        self._finish(context)
        # The grow operator is modal on the cctbx build, and starting a modal
        # operator from inside this modal does not receive events. Defer it
        # until the selection operator has returned.
        bpy.app.timers.register(_grow_from_selection, first_interval=0.01)
        return {'FINISHED'}


class CGN_OT_delete_atoms(_CGN_OT_select_atoms):
    bl_idname = "cgn.delete_atoms"
    bl_label = "Delete Atoms"
    bl_description = (
        "Click atoms or drag a box around them, then confirm to delete; "
        "click a selected atom to remove it, Shift adds to the selection"
    )
    bl_options = {'REGISTER', 'UNDO'}

    select_error = "Import a CIF before deleting atoms."
    status_text = (
        "Click atoms to add/remove, drag a box to select; Shift adds. "
        "Enter/right-click deletes them, Esc cancels."
    )
    keep_stored_selection = False

    def _confirm(self, context):
        if not self._selected:
            self._finish(context)
            self.report({'ERROR'}, "Select one or more atoms before deleting.")
            return {'CANCELLED'}
        root = bpy.data.collections.get(self._root_name)
        if root is None or root.get("cgn_instance_id") != self._root_id:
            self._finish(context)
            self.report({'ERROR'}, "The active structure changed during selection.")
            return {'CANCELLED'}
        try:
            sites, instances, cell_matrix, initial_count = _structure_tables(root)
        except (ValueError, TypeError) as exc:
            self._finish(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        atom_count = len(instances) // 4
        removed = sorted({
            index for index in self._selected
            if isinstance(index, int) and 0 <= index < atom_count
        })
        if not removed:
            self._finish(context)
            self.report({'ERROR'}, "Select one or more atoms before deleting.")
            return {'CANCELLED'}
        if len(removed) >= atom_count:
            self._finish(context)
            self.report(
                {'ERROR'},
                "Deleting every atom is not supported; re-import the CIF instead.",
            )
            return {'CANCELLED'}

        removed_offsets = {index * 4 for index in removed}
        new_instances = [
            value for offset, value in enumerate(instances)
            if (offset - offset % 4) not in removed_offsets
        ]
        removed_below_initial = sum(
            1 for index in removed if index < initial_count
        )
        new_initial_count = max(0, initial_count - removed_below_initial)
        atoms = _materialize_atoms(
            sites, cell_matrix, new_instances, new_initial_count
        )

        self._finish(context)
        root["cgn_instances"] = json.dumps(new_instances)
        root["cgn_initial_count"] = new_initial_count
        try:
            _rebuild_atom_objects(root, atoms)
            _update_growth_selection(root, atoms, [])
            _bond_objects(
                root, atoms, context.scene.cgn_connectivity_pairs
            )
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        try:
            bpy.ops.ed.undo_push(message="Delete Atoms")
        except RuntimeError:
            pass
        self.report(
            {'INFO'},
            f"Deleted {len(removed)} atoms; structure now contains "
            f"{len(atoms)} atoms.",
        )
        return {'FINISHED'}


class CGN_OT_grow_further(Operator):
    bl_idname = "cgn.grow_further"
    bl_label = "Grow Structure"
    bl_description = (
        "Add one shell around all atoms or the selected growth atoms"
    )

    all_atoms: BoolProperty(
        name="Grow from all atoms",
        description="Use every atom as a growth seed",
        default=True,
        options={'HIDDEN'},
    )

    def execute(self, context):
        root = _active_root(context.scene)
        if root is None:
            self.report({'ERROR'}, "Import a CIF before growing the structure.")
            return {'CANCELLED'}
        enabled_pairs = [
            item for item in context.scene.cgn_connectivity_pairs
            if item.enabled and item.max_distance >= item.min_distance
        ]
        if not enabled_pairs:
            self.report({'ERROR'}, "Enable at least one valid connectivity pair.")
            return {'CANCELLED'}
        try:
            sites, instances, cell_matrix, initial_count = _structure_tables(root)
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        try:
            selected_indices = json.loads(
                root.get("cgn_selected_indices", "[]")
            )
        except (ValueError, TypeError) as exc:
            self.report({'ERROR'}, f"Stored growth selection is invalid: {exc}")
            return {'CANCELLED'}
        atom_count = len(instances) // 4
        if not atom_count:
            self.report({'ERROR'}, "The active structure contains no atoms.")
            return {'CANCELLED'}
        if self.all_atoms:
            selected_indices = list(range(atom_count))
        else:
            selected_indices = sorted({
                index for index in selected_indices
                if isinstance(index, int) and 0 <= index < atom_count
            })
        if not selected_indices:
            self.report(
                {'ERROR'},
                "Select one or more atoms before growing the structure.",
            )
            return {'CANCELLED'}
        seed_instances = []
        for index in selected_indices:
            offset = index * 4
            seed_instances.extend(instances[offset:offset + 4])

        try:
            worker = _gemmi_worker()
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        try:
            result = worker.grow_connectivity_shell(
                json.loads(root.get("cgn_unit_cell", "[]")),
                [
                    {"element": site["element"], "frac": site["frac"]}
                    for site in sites
                ],
                instances,
                seed_instances,
                [
                    {
                        "element_a": item.element_a,
                        "element_b": item.element_b,
                        "enabled": item.enabled,
                        "min_distance": item.min_distance,
                        "max_distance": item.max_distance,
                    }
                    for item in context.scene.cgn_connectivity_pairs
                ],
            )
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        grew_by = int(result.get("grew_by", 0))
        if grew_by == 0:
            self.report(
                {'INFO'},
                "No additional atoms match the current connectivity settings.",
            )
            return {'FINISHED'}
        new_instances = result.get("new_instances", [])
        if len(new_instances) < grew_by * 4:
            self.report(
                {'ERROR'}, "The gemmi worker returned an invalid growth result."
            )
            return {'CANCELLED'}
        try:
            instances = instances + new_instances
            atoms = _materialize_atoms(
                sites, cell_matrix, instances, initial_count
            )
            _rebuild_atom_objects(root, atoms)
            _update_growth_selection(root, atoms, [])
            _bond_objects(
                root, atoms, context.scene.cgn_connectivity_pairs
            )
            root["cgn_instances"] = json.dumps(instances)
            root["cgn_growth_shells"] = int(
                root.get("cgn_growth_shells", 0)
            ) + 1
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            f"Added {grew_by} atoms; structure now contains "
            f"{len(atoms)} atoms.",
        )
        return {'FINISHED'}


class CGN_PT_panel(Panel):
    bl_label = "CCTBX Geometry Nodes"
    bl_idname = "CGN_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "CCTBX GN"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        layout.operator("cgn.import_cif", icon='IMPORT')
        root = _active_root(context.scene)
        if root is None:
            layout.label(text="Import a CIF to begin.")
            return

        layout.label(
            text=f"Space group: {root.get('cgn_space_group', 'unknown')}",
            icon='INFO',
        )
        try:
            parameters = json.loads(root.get("cgn_unit_cell", "[]"))
        except (ValueError, TypeError):
            parameters = []
        if len(parameters) >= 6:
            a, b, c, alpha, beta, gamma = (
                float(value) for value in parameters[:6]
            )
            grid = layout.grid_flow(
                row_major=True, columns=2, even_columns=True, align=True
            )
            grid.label(
                text=f"a = {a:.4f} Å", icon='DRIVER_DISTANCE'
            )
            grid.label(
                text=f"α = {alpha:.3f}°",
                icon='DRIVER_ROTATIONAL_DIFFERENCE',
            )
            grid.label(
                text=f"b = {b:.4f} Å", icon='DRIVER_DISTANCE'
            )
            grid.label(
                text=f"β = {beta:.3f}°",
                icon='DRIVER_ROTATIONAL_DIFFERENCE',
            )
            grid.label(
                text=f"c = {c:.4f} Å", icon='DRIVER_DISTANCE'
            )
            grid.label(
                text=f"γ = {gamma:.3f}°",
                icon='DRIVER_ROTATIONAL_DIFFERENCE',
            )
        if root.get("cgn_instances", ""):
            layout.label(
                text=f"Atoms: {_structure_atom_count(root)}",
                icon='MESH_UVSPHERE',
            )
        else:
            layout.label(
                text="Re-import this CIF (old structure format).",
                icon='ERROR',
            )


class CGN_PT_build(Panel):
    bl_label = "Build"
    bl_idname = "CGN_PT_build"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "CCTBX GN"
    bl_parent_id = "CGN_PT_panel"
    bl_order = 1

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def draw(self, context):
        layout = self.layout
        layout.menu("CGN_MT_fill", text="Fill", icon='MESH_GRID')
        layout.operator(
            "cgn.grow_further", text="Grow Structure", icon='ADD'
        ).all_atoms = True
        layout.operator(
            "cgn.select_growth_atoms",
            text="Grow Selected Atoms",
            icon='RESTRICT_SELECT_OFF',
        )
        layout.separator()
        layout.operator("cgn.delete_atoms", text="Delete Atoms", icon='X')
        layout.separator()
        layout.prop(
            context.scene,
            "cgn_show_hydrogens",
            text="Show Hydrogens",
            icon='HIDE_OFF' if context.scene.cgn_show_hydrogens else 'HIDE_ON',
        )


class CGN_PT_connectivity(Panel):
    bl_label = "Connectivity"
    bl_idname = "CGN_PT_connectivity"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "CCTBX GN"
    bl_parent_id = "CGN_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 2

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def draw(self, context):
        layout = self.layout
        if _active_root(context.scene) is None:
            return
        layout.label(
            text="Distances in Å. Toggle pairs and set inclusive minimum/maximum."
        )
        for item in context.scene.cgn_connectivity_pairs:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.label(text=f"{item.element_a} — {item.element_b}")
            row.prop(item, "min_distance", text="Min")
            row.prop(item, "max_distance", text="Max")
        layout.operator("cgn.rebuild_connectivity", icon='FILE_REFRESH')


class CGN_PT_unit_cell(Panel):
    bl_label = "Unit Cell"
    bl_idname = "CGN_PT_unit_cell"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "CCTBX GN"
    bl_parent_id = "CGN_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 3

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.cgn_unit_cell_settings
        layout.prop(
            settings, "show_unit_cell", text="Show Unit Cell", icon='MESH_CUBE'
        )
        if not settings.show_unit_cell:
            return
        column = layout.column(align=True)
        column.prop(settings, "face_color", text="Face Color")
        column.prop(settings, "face_transparency", text="Face Transparency")
        column.prop(settings, "edge_color", text="Edge Color")
        column.prop(settings, "edge_thickness", text="Edge Thickness")


class CGN_PT_model_style(Panel):
    bl_label = "Model & Style"
    bl_idname = "CGN_PT_model_style"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "CCTBX GN"
    bl_parent_id = "CGN_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 4

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "cgn_model", text="Model")
        if scene.cgn_model == 'ELLIPSOID':
            column = layout.column(align=True)
            column.prop(
                scene,
                "cgn_ellipsoid_probability",
                text="Probability (%)",
                slider=True,
            )
            column.label(
                text=(
                    f"{scene.cgn_ellipsoid_probability:.0f} % probability "
                    "level"
                ),
                icon='DRIVER_ROTATIONAL_DIFFERENCE',
            )
        elif scene.cgn_model == 'SPACE_FILLING':
            layout.label(
                text="Full van der Waals radii; bonds are hidden.",
                icon='INFO',
            )


class CGN_PT_about(Panel):
    bl_label = "About"
    bl_idname = "CGN_PT_about"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "CCTBX GN"
    bl_parent_id = "CGN_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 5

    def draw(self, context):
        layout = self.layout
        version = ".".join(str(part) for part in bl_info["version"])
        layout.label(text=f"Version {version}", icon='INFO')
        gemmi_version, error = _gemmi_status()
        if error:
            layout.label(text="gemmi not available", icon='ERROR')
        else:
            layout.label(text=f"gemmi {gemmi_version}", icon='PYTHON')
        layout.operator(
            "wm.url_open", text="Documentation", icon='URL'
        ).url = _DOCS_URL


class CGN_MT_viewport(Menu):
    bl_idname = "CGN_MT_viewport"
    bl_label = "CCTBX"

    def draw(self, context):
        layout = self.layout
        layout.operator("cgn.import_cif", text="Import CIF...", icon='IMPORT')
        if _active_root(context.scene) is None:
            return
        layout.separator()
        layout.menu("CGN_MT_fill", text="Fill", icon='MESH_GRID')
        layout.operator(
            "cgn.grow_further", text="Grow Structure", icon='ADD'
        ).all_atoms = True
        layout.operator(
            "cgn.select_growth_atoms",
            text="Grow Selected Atoms",
            icon='RESTRICT_SELECT_OFF',
        )
        layout.operator("cgn.delete_atoms", text="Delete Atoms", icon='X')
        layout.separator()
        layout.operator(
            "cgn.rebuild_connectivity", text="Rebuild Bonds", icon='FILE_REFRESH'
        )


def _draw_viewport_menu(self, context):
    self.layout.menu("CGN_MT_viewport")


def _draw_file_import(self, context):
    self.layout.operator("cgn.import_cif", text="CIF (.cif)", icon='IMPORT')


_CLASSES = (
    CGN_AddonPreferences,
    CGN_ConnectivityPair,
    CGN_UnitCellSettings,
    CGN_OT_import_cif,
    CGN_OT_rebuild_connectivity,
    CGN_OT_fill_asu,
    CGN_OT_fill_cell,
    CGN_OT_fill_pack,
    CGN_OT_select_growth_atoms,
    CGN_OT_delete_atoms,
    CGN_OT_grow_further,
    CGN_MT_fill,
    CGN_MT_viewport,
    CGN_PT_panel,
    CGN_PT_build,
    CGN_PT_connectivity,
    CGN_PT_unit_cell,
    CGN_PT_model_style,
    CGN_PT_about,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.cgn_connectivity_pairs = CollectionProperty(
        type=CGN_ConnectivityPair
    )
    bpy.types.Scene.cgn_active_structure = StringProperty(options={'HIDDEN'})
    bpy.types.Scene.cgn_show_hydrogens = BoolProperty(
        name="Show Hydrogens",
        description="Show hydrogen atoms and their bonds",
        default=True,
        update=_update_show_hydrogens,
    )
    bpy.types.Scene.cgn_unit_cell_settings = PointerProperty(
        type=CGN_UnitCellSettings
    )
    bpy.types.Scene.cgn_model = EnumProperty(
        name="Model",
        description="Rendering model for atoms and bonds",
        items=_MODEL_ITEMS,
        default='BALL_STICK',
        update=_update_model,
    )
    bpy.types.Scene.cgn_ellipsoid_probability = FloatProperty(
        name="Probability",
        description="Ellipsoid probability level in percent",
        default=50.0,
        min=0.0,
        max=100.0,
        update=_update_ellipsoid_probability,
    )
    bpy.types.VIEW3D_MT_editor_menus.append(_draw_viewport_menu)
    bpy.types.TOPBAR_MT_file_import.append(_draw_file_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(_draw_file_import)
    bpy.types.VIEW3D_MT_editor_menus.remove(_draw_viewport_menu)
    del bpy.types.Scene.cgn_ellipsoid_probability
    del bpy.types.Scene.cgn_model
    del bpy.types.Scene.cgn_unit_cell_settings
    del bpy.types.Scene.cgn_show_hydrogens
    del bpy.types.Scene.cgn_active_structure
    del bpy.types.Scene.cgn_connectivity_pairs
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
