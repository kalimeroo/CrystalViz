bl_info = {
    "name": "CrystalViz",
    "author": "Erwann Jeanneau",
    "version": (0, 9, 6),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > CrystalViz; File > Import",
    "description": (
        "Import CIF atoms and render editable connectivity with Geometry "
        "Nodes (gemmi backend, no external Python)"
    ),
    "category": "Import-Export",
}

import json
import math
import os
import re
import traceback
import uuid

import bmesh
import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
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
from bpy.types import AddonPreferences, Menu, Operator, Panel, PropertyGroup
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector


def _srgb_to_linear(value):
    value /= 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _cpk_color(hex_value):
    """Convert a Jmol CPK hex colour to linear RGBA for Blender."""
    rgb = tuple(int(hex_value[index : index + 2], 16) for index in (0, 2, 4))
    return tuple(_srgb_to_linear(channel) for channel in rgb) + (1.0,)


# Jmol CPK colours (sRGB hex) from the Science Notes JMOL CPK table.
_JMOL_CPK_HEX = {
    "H": "FFFFFF",
    "D": "FFFFC0",
    "T": "FFFFA0",
    "HE": "D9FFFF",
    "LI": "CC80FF",
    "BE": "C2FF00",
    "B": "FFB5B5",
    "C": "909090",
    "N": "3050F8",
    "O": "FF0D0D",
    "F": "90E050",
    "NE": "B3E3F5",
    "NA": "AB5CF2",
    "MG": "8AFF00",
    "AL": "BFA6A6",
    "SI": "F0C8A0",
    "P": "FF8000",
    "S": "FFFF30",
    "CL": "1FF01F",
    "AR": "80D1E3",
    "K": "8F40D4",
    "CA": "3DFF00",
    "SC": "E6E6E6",
    "TI": "BFC2C7",
    "V": "A6A6AB",
    "CR": "8A99C7",
    "MN": "9C7AC7",
    "FE": "E06633",
    "CO": "F090A0",
    "NI": "50D050",
    "CU": "C88033",
    "ZN": "7D80B0",
    "GA": "C28F8F",
    "GE": "668F8F",
    "AS": "BD80E3",
    "SE": "FFA100",
    "BR": "A62929",
    "KR": "5CB8D1",
    "RB": "702EB0",
    "SR": "00FF00",
    "Y": "94FFFF",
    "ZR": "94E0E0",
    "NB": "73C2C9",
    "MO": "54B5B5",
    "TC": "3B9E9E",
    "RU": "248F8F",
    "RH": "0A7D8C",
    "PD": "006985",
    "AG": "C0C0C0",
    "CD": "FFD98F",
    "IN": "A67573",
    "SN": "668080",
    "SB": "9E63B5",
    "TE": "D47A00",
    "I": "940094",
    "XE": "429EB0",
    "CS": "57178F",
    "BA": "00C900",
    "LA": "70D4FF",
    "CE": "FFFFC7",
    "PR": "D9FFC7",
    "ND": "C7FFC7",
    "PM": "A3FFC7",
    "SM": "8FFFC7",
    "EU": "61FFC7",
    "GD": "45FFC7",
    "TB": "30FFC7",
    "DY": "1FFFC7",
    "HO": "00FF9C",
    "ER": "00E675",
    "TM": "00D452",
    "YB": "00BF38",
    "LU": "00AB24",
    "HF": "4DC2FF",
    "TA": "4DA6FF",
    "W": "2194D6",
    "RE": "267DAB",
    "OS": "266696",
    "IR": "175487",
    "PT": "D0D0E0",
    "AU": "FFD123",
    "HG": "B8B8D0",
    "TL": "A6544D",
    "PB": "575961",
    "BI": "9E4FB5",
    "PO": "AB5C00",
    "AT": "754F45",
    "RN": "428296",
    "FR": "420066",
    "RA": "007D00",
    "AC": "70ABFA",
    "TH": "00BAFF",
    "PA": "00A1FF",
    "U": "008FFF",
    "NP": "0080FF",
    "PU": "006BFF",
    "AM": "545CF2",
    "CM": "785CE3",
    "BK": "8A4FE3",
    "CF": "A136D4",
    "ES": "B31FD4",
    "FM": "B31FBA",
    "MD": "B30DA6",
    "NO": "BD0D87",
    "LR": "C70066",
    "RF": "CC0059",
    "DB": "D1004F",
    "SG": "D90045",
    "BH": "E00038",
    "HS": "E6002E",
    "MT": "EB0026",
}
_ELEMENT_COLORS = {
    element: _cpk_color(value) for element, value in _JMOL_CPK_HEX.items()
}
_DEFAULT_COLOR = _cpk_color("FF1493")
_DISPLAY_SCALE = 0.1
_BOND_VDW_FACTOR = 0.1
_BALL_ATOM_VDW_FACTOR = 0.25
_SPACE_FILLING_ATOM_VDW_FACTOR = 1.0
_ELLIPSOID_BOND_VDW_FACTOR = 0.05
_WIRE_RADIUS_DEFAULT = 0.1
_BOND_RADIUS_SCALE_DEFAULT = 1.0
_MULTIPLE_BOND_RADIUS_SCALE_DEFAULT = 1.0
_CHEMDRAW_BOND_SPACING_DEFAULT = 0.32
_BOND_COLOR_DEFAULT = (0.16, 0.16, 0.18, 1.0)
_AROMATIC_TORUS_SIZE_DEFAULT = 0.55
_AROMATIC_TORUS_WIDTH_DEFAULT = 0.5
_AROMATIC_BOND_LENGTH_SPREAD_MAX = 0.06
_CHEMDRAW_MULTIPLE_BOND_RADIUS_SCALE = {2: 0.9, 3: 0.85}
_CHEMDRAW_MAX_VALENCE = {
    "B": 3,
    "C": 4,
    "N": 4,
    "O": 2,
    "F": 1,
    "SI": 4,
    "P": 5,
    "S": 6,
    "CL": 1,
    "BR": 1,
    "I": 1,
}
# Chi-square quantile with 3 degrees of freedom at 50 % probability.
_ELLIPSOID_CHI2_50 = 2.3659738843753377
_DEFAULT_VDW_RADIUS = 1.5
_DEFAULT_BOND_RADIUS = _BOND_VDW_FACTOR * _DEFAULT_VDW_RADIUS * _DISPLAY_SCALE
# Default connectivity window, as a fraction of (r_a + r_b) with the covalent
# radii of the two elements.
# The lower bound rejects collapsed contacts such as the 1.18 A K-K pair in
# Ca_autored.cif.
_BOND_MIN_SCALE = 0.6
_BOND_MAX_SCALE = 1.2
_BOND_MAX_SLACK = 0.4
# Cap applied to same-sign metal-metal pairs (for example Ti(IV)-Ti(IV) in an
# oxide, or Cu(I)-Cu(I) in Cu2O): only contacts shorter than this fraction of
# the covalent radius sum survive, which keeps genuine cluster bonds
# (Rh2(OAc)4, Re2Cl8(2-)) while dropping cation-cation contacts.
_BOND_METAL_CAP_SCALE = 0.85
# Hard ceiling for a connectivity window (A).  Anything wider than this is not
# a chemical bond; it only multiplies the periodic image bank, so the UI caps
# both bounds here and the computed defaults are clamped to the same value.
_MAX_BOND_WINDOW = 10.0
_MAX_FILL_REPEATS = 20
_MAX_FILL_ATOMS = 200000
_UNIT_CELL_EDGE_COLOR = (0.0, 0.0, 0.0, 1.0)
_UNIT_CELL_FACE_COLOR = (0.45, 0.65, 0.95, 1.0)
_UNIT_CELL_FACE_ALPHA = 0.08
_UNIT_CELL_EDGE_THICKNESS = 0.05
_POLY_FACE_COLOR = (0.85, 0.55, 0.15, 1.0)
_POLY_FACE_ALPHA = 0.25
_POLY_EDGE_COLOR = (0.0, 0.0, 0.0, 1.0)
_POLY_EDGE_THICKNESS = 0.04
_POLY_MIN_LIGANDS = 4
_NODE_GROUP_ATOMS = "CGN_GN_AtomSpheres"
_NODE_GROUP_BONDS = "CGN_GN_BondTubes"
_NODE_GROUP_ELLIPSOIDS = "CGN_GN_AtomEllipsoids_ORTEP"
_LEGACY_NODE_GROUP_ELLIPSOIDS = "CGN_GN_AtomEllipsoids"
_NODE_GROUP_POLYHEDRA = "CGN_GN_Polyhedra"
_POLY_PROTOTYPE_COLLECTION = "CGN_Polyhedron_Prototypes"
_ORTEP_TEMPLATE_OBJECT = "CGN_ORTEP_Template"
_ORTEP_INNER_COLOR = (0.12, 0.12, 0.16, 1.0)
_ORTEP_ELLIPSE_COLOR = (0.015, 0.015, 0.02, 1.0)
_ORTEP_ELLIPSE_THICKNESS = 0.018
_ORTEP_VIEW_SIGNATURES = {}
_BOND_SPATIAL_CACHE = None
_STRUCTURE_CACHE = None
_MODEL_ITEMS = (
    ("BALL_STICK", "Ball & Stick", "Spheres at quarter van der Waals radius"),
    ("ELLIPSOID", "ORTEP", "ORTEP anisotropic displacement ellipsoids"),
    (
        "SPACE_FILLING",
        "Space-filling",
        "Spheres at full van der Waals radius, without bonds",
    ),
    ("WIRE_STICKS", "Wire-sticks", "Thin atoms and thin bonds"),
)
_CIF_CELL_VALUE_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\(\d+\))?(?:[eE][+-]?\d+)?$"
)
_DOCS_URL = "https://github.com/kalimeroo/CrystalViz"


def _gemmi_status():
    """Return (version, error) for the gemmi module in Blender's Python."""
    try:
        import gemmi
    except Exception as exc:  # ImportError, or a broken wheel
        return None, str(exc)
    return getattr(gemmi, "__version__", "unknown"), None


def _gemmi_install_hint():
    """Diagnostic text when the bundled gemmi wheel cannot be imported.

    gemmi is now bundled with the extension, so end users never install it
    manually. A missing/broken gemmi usually means the extension was installed
    for the wrong platform or into an unsupported Blender version.
    """
    return (
        "gemmi should already be bundled with this extension. If it is "
        "missing, reinstall the extension archive that matches your platform "
        "(Windows, Linux or macOS) and make sure it was downloaded/installed "
        "from the release for your operating system."
    )


def _gemmi_worker():
    """Import the in-process gemmi worker, with an install hint if missing."""
    try:
        from . import gemmi_worker
    except ImportError as exc:
        raise RuntimeError(
            "The bundled gemmi library could not be loaded.\n" + _gemmi_install_hint()
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
                icon="ERROR",
            )
            for section in _gemmi_install_hint().split("\n"):
                layout.label(text=section)
        else:
            layout.label(
                text=f"gemmi {version} (bundled, in-process, no external interpreter)",
                icon="CHECKMARK",
            )


class CGN_ConnectivityPair(PropertyGroup):
    element_a: StringProperty(options={"HIDDEN"})
    element_b: StringProperty(options={"HIDDEN"})
    enabled: BoolProperty(name="Connect", default=True)
    min_distance: FloatProperty(
        name="Min (Å)",
        default=0.1,
        min=0.0,
        max=_MAX_BOND_WINDOW,
        precision=3,
    )
    max_distance: FloatProperty(
        name="Max (Å)",
        default=3.0,
        min=0.0,
        max=_MAX_BOND_WINDOW,
        precision=3,
    )


_DEBOUNCE_DELAY = 0.15
_DEBOUNCE_TIMERS = {}


def _debounce(name, callback):
    """Coalesce repeated property changes into one scheduled run.

    Property update callbacks fire for every tick of a slider (and every step
    of a colour picker).  The first change schedules ``callback`` roughly
    0.15 s out; further changes before it runs replace the stored callback and
    reuse the same timer, so a drag costs at most one heavy rebuild per window
    instead of one per tick.  The callback is kept in ``_DEBOUNCE_TIMERS`` so
    :func:`_flush_debounce` can apply a pending change immediately (render,
    export, headless tests).
    """
    _DEBOUNCE_TIMERS[name] = callback
    if not bpy.app.timers.is_registered(_debounce_timer):
        bpy.app.timers.register(_debounce_timer, first_interval=_DEBOUNCE_DELAY)


def _debounce_timer():
    _flush_debounce()
    return None


def _flush_debounce():
    """Apply every pending debounced callback now (used by render/tests)."""
    pending = list(_DEBOUNCE_TIMERS.values())
    _DEBOUNCE_TIMERS.clear()
    for callback in pending:
        try:
            callback()
        except Exception:
            traceback.print_exc()


def _cancel_all_debounce():
    _DEBOUNCE_TIMERS.clear()
    if bpy.app.timers.is_registered(_debounce_timer):
        bpy.app.timers.unregister(_debounce_timer)


def _principled_shader(material):
    """Return *material*'s Principled BSDF node, enabling nodes if necessary.

    Materials are looked up by name, so a same-named datablock can already
    exist with ``use_nodes`` disabled (the user toggled it, or it came from an
    appended/imported file).  ``material.node_tree`` is None in that state, so
    every caller must go through here instead of dereferencing it.
    """
    if material is None:
        return None
    if material.node_tree is None:
        material.use_nodes = True
    if material.node_tree is None:  # pragma: no cover - Blender always builds one
        return None
    return material.node_tree.nodes.get("Principled BSDF")


def _update_atom_style(self, context):
    material = bpy.data.materials.get(f"CGN_Mat_{self.element}")
    if material is None:
        return
    material.diffuse_color = self.color
    shader = _principled_shader(material)
    if shader is not None:
        shader.inputs["Base Color"].default_value = self.color
        shader.inputs["Roughness"].default_value = self.roughness
    root = _active_root(context.scene)
    if root is not None and context.scene.cgn_model == "BALL_STICK":
        for obj in root.objects:
            if obj.get("cgn_role") != "atoms" or obj.get("cgn_element") != self.element:
                continue
            radius = float(obj.get("cgn_vdw_radius", _DEFAULT_VDW_RADIUS))
            for modifier in obj.modifiers:
                if modifier.type == "NODES" and modifier.node_group is not None:
                    _set_modifier_input(
                        modifier,
                        modifier.node_group,
                        "Radius",
                        radius * self.size * _DISPLAY_SCALE,
                    )
            obj.update_tag(refresh={"OBJECT", "DATA"})


class CGN_AtomStyle(PropertyGroup):
    element: StringProperty(options={"HIDDEN"})
    color: FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        update=_update_atom_style,
    )
    roughness: FloatProperty(
        name="Roughness",
        default=0.25,
        min=0.0,
        max=1.0,
        precision=2,
        update=_update_atom_style,
    )
    size: FloatProperty(
        name="Size",
        description="Atom radius as a fraction of its van der Waals radius",
        subtype="FACTOR",
        default=0.25,
        min=0.0,
        max=1.0,
        update=_update_atom_style,
    )


def _update_ortep_materials(scene, context):
    for name, color, roughness in (
        (
            "CGN_ORTEP_Inner_Faces",
            scene.cgn_ortep_inner_color,
            scene.cgn_ortep_inner_roughness,
        ),
        (
            "CGN_ORTEP_Ellipse_Lines",
            scene.cgn_ortep_ellipse_color,
            scene.cgn_ortep_ellipse_roughness,
        ),
    ):
        material = bpy.data.materials.get(name)
        if material is None:
            continue
        material.diffuse_color = color
        shader = _principled_shader(material)
        if shader is not None:
            shader.inputs["Base Color"].default_value = color
            shader.inputs["Roughness"].default_value = roughness


def _update_ortep_ellipse_thickness(scene, context):
    _debounce(
        "ortep_ellipse_thickness",
        lambda: _apply_ortep_ellipse_thickness(scene),
    )


def _apply_ortep_ellipse_thickness(scene):
    # If the template object was deleted, recreate it (and rebuild the
    # ellipsoid node group's pointer to it) instead of silently giving up,
    # otherwise the ORTEP ellipsoids are permanently broken.
    if bpy.data.objects.get(_ORTEP_TEMPLATE_OBJECT) is None:
        _make_ellipsoid_node_group()
    template = bpy.data.objects.get(_ORTEP_TEMPLATE_OBJECT)
    if template is None:
        return
    thickness = scene.cgn_ortep_ellipse_thickness
    if _update_ortep_template_mesh(template.data, thickness):
        template.update_tag(refresh={"OBJECT", "DATA"})
        return
    old_mesh = template.data
    template.data = _build_ortep_template_mesh(thickness)
    if old_mesh is not None and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    template.update_tag(refresh={"OBJECT", "DATA"})


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
        subtype="COLOR",
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
        subtype="COLOR",
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


def _polyhedron_element_items(_self, context):
    """Enum items: structure elements able to carry four or more bonds.

    Elements qualify when at least one of their atoms currently has four or
    more connected neighbours under the enabled connectivity windows, which is
    the weakest arrangement that still forms a closed polyhedron.
    """
    scene = getattr(context, "scene", None)
    if scene is None:
        return (("", "No eligible element", "No structure is active"),)
    root = _active_root(scene)
    elements = _cached_polyhedron_elements(root, scene)
    if not elements:
        return (
            (
                "",
                "No eligible element",
                "Import a structure and enable connectivity windows; listed "
                "elements need at least one atom with four bonded neighbours",
            ),
        )
    return tuple(
        (
            element,
            element,
            f"Build polyhedra around every {element} site with "
            f"{_POLY_MIN_LIGANDS} or more bonded neighbours",
        )
        for element in elements
    )


def _polyhedron_element_changed(self, context):
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    element = str(getattr(self, "element", ""))
    if element:
        _polyhedron_style(scene, element)


class CGN_PolyhedronSettings(PropertyGroup):
    element: EnumProperty(
        name="Element",
        description=(
            "Coordination center element; only elements that currently have "
            "atoms with four or more bonded neighbours are offered"
        ),
        items=_polyhedron_element_items,
        update=_polyhedron_element_changed,
    )


def _update_polyhedron_style(self, context):
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    root = _active_root(scene)
    if root is None:
        return
    element = str(getattr(self, "element", ""))
    if not element:
        return
    site_label = str(getattr(self, "site_label", ""))
    face_material = _make_polyhedron_face_material(element, self, site_label)
    edge_material = _make_polyhedron_edge_material(element, self, site_label)
    _face, _edge, thickness = _polyhedron_style_values(self)

    prototypes = {}
    prototype_users = {}
    legacy = []
    for obj in list(root.all_objects):
        if obj.get("cgn_role") != "polyhedron":
            continue
        if str(obj.get("cgn_poly_element", "")) != element:
            continue
        if str(obj.get("cgn_poly_site_label", "")) != site_label:
            continue
        prototype = _polyhedron_prototype_for(obj)
        if prototype is None:
            # Scaffold from the baked-mesh implementation (or a prototype the
            # user deleted): fall back to a full rebuild below.
            legacy.append(obj)
            continue
        prototypes[prototype.name] = prototype
        prototype_users.setdefault(prototype.name, []).append(obj)

    for name, prototype in prototypes.items():
        mesh = prototype.data if isinstance(prototype.data, bpy.types.Mesh) else None
        if mesh is None:
            continue
        # Colour/transparency changes only swap the prototype's material slots;
        # every image is an instance of it, so nothing else needs rebuilding.
        if len(mesh.materials) >= 2:
            mesh.materials[0] = face_material
            mesh.materials[1] = edge_material
        else:
            mesh.materials.clear()
            mesh.materials.append(face_material)
            mesh.materials.append(edge_material)
        try:
            stored_width = float(prototype.get("cgn_poly_edge_width", thickness))
        except (TypeError, ValueError):
            stored_width = thickness
        # Custom properties are stored as 32-bit floats, so compare with a
        # tolerance above float32 rounding instead of 1e-9: otherwise every
        # colour tweak after a width change would rebuild the tubes again.
        if abs(stored_width - thickness) > 1e-6 and _rebuild_polyhedron_prototype(
            prototype, thickness
        ):
            # Keep the scaffold's stored width in sync so the next style change
            # does not rebuild the prototype again for the same value.
            for obj in prototype_users.get(name, ()):
                obj["cgn_poly_edge_width"] = float(thickness)

    if not legacy:
        return
    centers = []
    for obj in legacy:
        try:
            centers.extend(
                int(index) for index in json.loads(obj.get("cgn_poly_centers", "[]"))
            )
        except (TypeError, ValueError):
            continue
    for obj in legacy:
        _remove_object(obj)
    if not centers:
        return
    try:
        atoms = _structure_atoms(root)
    except (ValueError, TypeError):
        return
    _build_polyhedra_objects(
        root,
        atoms,
        centers,
        scene,
        clear_existing=False,
        extend_structure=False,
    )


class CGN_PolyhedronStyle(PropertyGroup):
    element: StringProperty(options={"HIDDEN"})
    site_label: StringProperty(options={"HIDDEN"})
    expanded: BoolProperty(
        name="Expanded",
        description="Show this polyhedron's style controls",
        default=False,
    )
    face_color: FloatVectorProperty(
        name="Face Color",
        description="Polyhedron face color (defaults to the central atom colour)",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
        update=_update_polyhedron_style,
    )
    face_transparency: FloatProperty(
        name="Transparency",
        description="Polyhedron face transparency (1 is fully transparent)",
        default=1.0 - _POLY_FACE_ALPHA,
        min=0.0,
        max=1.0,
        precision=2,
        update=_update_polyhedron_style,
    )
    edge_color: FloatVectorProperty(
        name="Edge Color",
        description="Polyhedron edge color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=_POLY_EDGE_COLOR,
        update=_update_polyhedron_style,
    )
    edge_transparency: FloatProperty(
        name="Edge Transparency",
        description="Polyhedron edge transparency (1 is fully transparent)",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=2,
        update=_update_polyhedron_style,
    )
    edge_width: FloatProperty(
        name="Edge Width",
        description="Polyhedron edge width",
        default=_POLY_EDGE_THICKNESS,
        min=0.0,
        max=0.25,
        precision=3,
        update=_update_polyhedron_style,
    )


def _active_root(scene):
    """Return the active structure collection if this scene actually shows it.

    ``scene.cgn_active_structure`` holds only a name, so it still resolves after
    the user unlinks the collection from the scene (or moves it to another
    scene) while the datablock survives in ``bpy.data.collections``.  A plain
    lookup would let operators and panels act on invisible data, so check that
    the collection really is under this scene's collection tree.
    """
    name = scene.cgn_active_structure
    root = bpy.data.collections.get(name) if name else None
    if root is None or root.get("cgn_role") != "structure":
        return None
    if not _collection_in_scene(scene, root):
        return None
    return root


def _collection_in_scene(scene, collection):
    """True when *collection* is linked somewhere under *scene* (any depth)."""
    target = collection.name
    return any(
        child.name == target for child in scene.collection.children_recursive
    )


def _unlinked_active_structure(scene):
    """The active structure datablock, but only when *scene* does not show it."""
    name = scene.cgn_active_structure
    root = bpy.data.collections.get(name) if name else None
    if root is None or root.get("cgn_role") != "structure":
        return None
    return None if _collection_in_scene(scene, root) else root


def _no_structure_message(scene, action):
    """Error text for an operator that needs an active structure in *scene*.

    Saying "import a CIF" is wrong when the structure exists but was unlinked
    from the scene, so name that case explicitly.
    """
    unlinked = _unlinked_active_structure(scene)
    if unlinked is not None:
        return (
            f"The active structure '{unlinked.name}' is not linked to this "
            f"scene; link it in the Outliner before {action}."
        )
    return f"Import a CIF before {action}."


def _socket_identifier(node_group, name, in_out="INPUT"):
    for item in node_group.interface.items_tree:
        if item.item_type == "SOCKET" and item.in_out == in_out and item.name == name:
            return item.identifier
    raise RuntimeError(f"Geometry Nodes socket '{name}' was not created.")


def _ensure_shade_smooth(node_group):
    """Add a Set Shade Smooth node to an atom node group if it is missing."""
    for node in node_group.nodes:
        if node.bl_idname == "GeometryNodeSetShadeSmooth":
            return
    instance = next(
        (
            node
            for node in node_group.nodes
            if node.bl_idname == "GeometryNodeInstanceOnPoints"
        ),
        None,
    )
    if instance is None:
        return
    link = next(
        (
            link
            for link in node_group.links
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
        # Re-apply the scene's quality preset instead of hardcoding the
        # defaults: this helper is called on every rebuild (import, model
        # change, grow, selection highlight), and it must never silently
        # override LOW/HIGH/CUSTOM values chosen by the user.
        _apply_quality_settings(getattr(bpy.context, "scene", None))
        return node_group

    node_group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    geometry_in = node_group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    radius_in = node_group.interface.new_socket(
        name="Radius", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    radius_in.default_value = (
        _DEFAULT_BOND_RADIUS if is_bond_group else 0.5 * _DISPLAY_SCALE
    )
    material_in = node_group.interface.new_socket(
        name="Material", in_out="INPUT", socket_type="NodeSocketMaterial"
    )
    geometry_out = node_group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
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
        profile.inputs["Resolution"].default_value = 12
        links.new(
            group_in.outputs[geometry_in.identifier], mesh_to_curve.inputs["Mesh"]
        )
        links.new(mesh_to_curve.outputs["Curve"], curve_to_mesh.inputs["Curve"])
        links.new(profile.outputs["Curve"], curve_to_mesh.inputs["Profile Curve"])
        links.new(group_in.outputs[radius_in.identifier], profile.inputs["Radius"])
        links.new(curve_to_mesh.outputs["Mesh"], set_material.inputs["Geometry"])
        links.new(
            group_in.outputs[material_in.identifier], set_material.inputs["Material"]
        )
        links.new(
            set_material.outputs["Geometry"], group_out.inputs[geometry_out.identifier]
        )
    else:
        ico = nodes.new("GeometryNodeMeshIcoSphere")
        set_material = nodes.new("GeometryNodeSetMaterial")
        smooth = nodes.new("GeometryNodeSetShadeSmooth")
        instance = nodes.new("GeometryNodeInstanceOnPoints")
        ico.location = (-350, -150)
        set_material.location = (-100, -150)
        smooth.location = (0, -150)
        instance.location = (150, 100)
        ico.inputs["Subdivisions"].default_value = 3
        links.new(group_in.outputs[geometry_in.identifier], instance.inputs["Points"])
        links.new(group_in.outputs[radius_in.identifier], ico.inputs["Radius"])
        links.new(ico.outputs["Mesh"], set_material.inputs["Geometry"])
        links.new(
            group_in.outputs[material_in.identifier], set_material.inputs["Material"]
        )
        links.new(set_material.outputs["Geometry"], smooth.inputs["Geometry"])
        links.new(smooth.outputs["Geometry"], instance.inputs["Instance"])
        links.new(
            instance.outputs["Instances"], group_out.inputs[geometry_out.identifier]
        )
    _apply_quality_settings(getattr(bpy.context, "scene", None))
    return node_group


def _make_unit_cell_node_group():
    node_group = bpy.data.node_groups.get("CGN_GN_UnitCell")
    if node_group is not None:
        return node_group
    node_group = bpy.data.node_groups.new("CGN_GN_UnitCell", "GeometryNodeTree")
    geometry_in = node_group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    face_material_in = node_group.interface.new_socket(
        name="Face Material", in_out="INPUT", socket_type="NodeSocketMaterial"
    )
    edge_material_in = node_group.interface.new_socket(
        name="Edge Material", in_out="INPUT", socket_type="NodeSocketMaterial"
    )
    edge_thickness_in = node_group.interface.new_socket(
        name="Edge Thickness", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    edge_thickness_in.default_value = _UNIT_CELL_EDGE_THICKNESS * _DISPLAY_SCALE
    marker_radius_in = node_group.interface.new_socket(
        name="Marker Radius", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    marker_radius_in.default_value = _UNIT_CELL_EDGE_THICKNESS * _DISPLAY_SCALE
    geometry_out = node_group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    nodes = node_group.nodes
    links = node_group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    group_in.location = (-900, 0)
    group_out.location = (900, 0)

    # Branch 1: faces (transparent)
    set_face_mat = nodes.new("GeometryNodeSetMaterial")
    set_face_mat.location = (-300, 200)
    links.new(group_in.outputs[geometry_in.identifier], set_face_mat.inputs["Geometry"])
    links.new(
        group_in.outputs[face_material_in.identifier], set_face_mat.inputs["Material"]
    )

    # Branch 2: beveled edges
    mesh_to_curve = nodes.new("GeometryNodeMeshToCurve")
    profile = nodes.new("GeometryNodeCurvePrimitiveCircle")
    curve_to_mesh = nodes.new("GeometryNodeCurveToMesh")
    set_edge_mat = nodes.new("GeometryNodeSetMaterial")
    mesh_to_curve.location = (-400, -100)
    profile.location = (-400, -250)
    curve_to_mesh.location = (-150, -100)
    set_edge_mat.location = (100, -100)
    profile.inputs["Resolution"].default_value = 8
    links.new(group_in.outputs[geometry_in.identifier], mesh_to_curve.inputs["Mesh"])
    links.new(mesh_to_curve.outputs["Curve"], curve_to_mesh.inputs["Curve"])
    links.new(profile.outputs["Curve"], curve_to_mesh.inputs["Profile Curve"])
    links.new(group_in.outputs[edge_thickness_in.identifier], profile.inputs["Radius"])
    links.new(curve_to_mesh.outputs["Mesh"], set_edge_mat.inputs["Geometry"])
    links.new(
        group_in.outputs[edge_material_in.identifier], set_edge_mat.inputs["Material"]
    )

    # Branch 3: corner markers (icospheres on vertices)
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico_mat = nodes.new("GeometryNodeSetMaterial")
    smooth = nodes.new("GeometryNodeSetShadeSmooth")
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    ico.location = (-400, -400)
    ico_mat.location = (-250, -400)
    smooth.location = (-100, -400)
    instance.location = (150, -300)
    realize.location = (350, -300)
    ico.inputs["Subdivisions"].default_value = 3
    links.new(group_in.outputs[marker_radius_in.identifier], ico.inputs["Radius"])
    links.new(ico.outputs["Mesh"], ico_mat.inputs["Geometry"])
    links.new(group_in.outputs[edge_material_in.identifier], ico_mat.inputs["Material"])
    links.new(ico_mat.outputs["Geometry"], smooth.inputs["Geometry"])
    links.new(smooth.outputs["Geometry"], instance.inputs["Instance"])
    links.new(group_in.outputs[geometry_in.identifier], instance.inputs["Points"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])

    # Join
    join = nodes.new("GeometryNodeJoinGeometry")
    join.location = (700, 0)
    links.new(set_face_mat.outputs["Geometry"], join.inputs["Geometry"])
    links.new(set_edge_mat.outputs["Geometry"], join.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], join.inputs["Geometry"])
    links.new(join.outputs["Geometry"], group_out.inputs[geometry_out.identifier])
    _apply_quality_settings(getattr(bpy.context, "scene", None))
    return node_group


def _make_polyhedra_node_group():
    """Shared Geometry Nodes graph: instance one site prototype on its images.

    A polyhedron is a single canonical mesh (faces + edge tubes, two material
    slots) instanced once per image of the crystallographic site.  The points
    scaffold fed into this group holds one vertex per image, so growing a site
    only appends a point instead of duplicating the whole polyhedron geometry.

    Instances are realized at the end of the graph: that keeps the object
    editable (Apply converts it to an ordinary mesh) and lets every exporter
    see the faces without special-casing instances.  The memory/file-size win
    remains — the prototype is stored once per site and the scaffold is one
    point per image.
    """
    node_group = bpy.data.node_groups.get(_NODE_GROUP_POLYHEDRA)
    if node_group is not None:
        return node_group

    node_group = bpy.data.node_groups.new(_NODE_GROUP_POLYHEDRA, "GeometryNodeTree")
    geometry_in = node_group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    node_group.interface.new_socket(
        name="Prototype", in_out="INPUT", socket_type="NodeSocketObject"
    )
    geometry_out = node_group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    nodes = node_group.nodes
    links = node_group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    info = nodes.new("GeometryNodeObjectInfo")
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    group_in.location = (-600, 0)
    info.location = (-350, -150)
    instance.location = (0, 100)
    realize.location = (300, 0)
    group_out.location = (550, 0)

    info.transform_space = "ORIGINAL"
    links.new(group_in.outputs["Geometry"], instance.inputs["Points"])
    links.new(group_in.outputs["Prototype"], info.inputs["Object"])
    links.new(info.outputs["Geometry"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], group_out.inputs[geometry_out.identifier])
    return node_group


def _current_model(scene=None):
    """Return the active rendering model name."""
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    model = getattr(scene, "cgn_model", "BALL_STICK") if scene else "BALL_STICK"
    if model not in {item[0] for item in _MODEL_ITEMS}:
        model = "BALL_STICK"
    return model


def _atom_radius(vdw_radius, model=None):
    """Display-space atom radius for the active model."""
    if model is None:
        model = _current_model()
    if model == "WIRE_STICKS":
        scene = getattr(bpy.context, "scene", None)
        wire_radius = getattr(scene, "cgn_wire_sticks_radius", _WIRE_RADIUS_DEFAULT)
        return float(wire_radius) * _DISPLAY_SCALE
    if model == "SPACE_FILLING":
        factor = _SPACE_FILLING_ATOM_VDW_FACTOR
    else:
        factor = _BALL_ATOM_VDW_FACTOR
    return float(vdw_radius) * factor * _DISPLAY_SCALE


def _bond_radius(vdw_a, vdw_b, model=None):
    """Display-space bond radius for the active model."""
    if model is None:
        model = _current_model()
    if model == "WIRE_STICKS":
        scene = getattr(bpy.context, "scene", None)
        wire_radius = getattr(scene, "cgn_wire_sticks_radius", _WIRE_RADIUS_DEFAULT)
        return float(wire_radius) * _DISPLAY_SCALE
    mean_vdw = 0.5 * (float(vdw_a) + float(vdw_b))
    if model == "ELLIPSOID":
        factor = _ELLIPSOID_BOND_VDW_FACTOR
    else:
        factor = _BOND_VDW_FACTOR
    return mean_vdw * factor * _DISPLAY_SCALE


def _ellipsoid_probability_factor(probability):
    """Scale ADP axes to the requested 3D Gaussian probability surface."""
    probability = max(0.001, min(99.999, float(probability))) / 100.0

    def chi_square_3_cdf(value):
        root = math.sqrt(value / 2.0)
        return math.erf(root) - math.sqrt(2.0 * value / math.pi) * math.exp(
            -value / 2.0
        )

    lower = 0.0
    upper = 1.0
    while chi_square_3_cdf(upper) < probability:
        upper *= 2.0
    for _iteration in range(64):
        middle = 0.5 * (lower + upper)
        if chi_square_3_cdf(middle) < probability:
            lower = middle
        else:
            upper = middle
    return math.sqrt(0.5 * (lower + upper) / _ELLIPSOID_CHI2_50)


def _ortep_template_geometry(ellipse_thickness=_ORTEP_ELLIPSE_THICKNESS):
    """Return (vertices, faces, face_types, smooth_faces, tube_radius).

    Only the three ellipse tubes depend on ``ellipse_thickness``; the sphere
    shell and the clipped inner fan are fixed.  The topology is therefore
    stable while the thickness slider moves, which lets the caller refresh the
    mesh in place instead of rebuilding it for every tick.
    """
    latitude_steps = 32
    longitude_steps = 64
    vertices = [(0.0, 0.0, 1.0)]

    def ring_index(latitude, longitude):
        return 1 + (latitude - 1) * longitude_steps + longitude % longitude_steps

    for latitude in range(1, latitude_steps):
        theta = math.pi * latitude / latitude_steps
        for longitude in range(longitude_steps):
            phi = 2.0 * math.pi * longitude / longitude_steps
            vertices.append(
                (
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                )
            )
    south = len(vertices)
    vertices.append((0.0, 0.0, -1.0))
    faces = []
    face_types = []
    smooth_faces = []

    def add_sphere_face(face):
        centroid = tuple(
            sum(vertices[index][axis] for index in face) / len(face)
            for axis in range(3)
        )
        if all(component > 1e-8 for component in centroid):
            return
        faces.append(face)
        face_types.append((True, False, False))
        smooth_faces.append(True)

    for longitude in range(longitude_steps):
        add_sphere_face(
            (
                0,
                ring_index(1, longitude),
                ring_index(1, longitude + 1),
            )
        )
    for latitude in range(1, latitude_steps - 1):
        for longitude in range(longitude_steps):
            current = ring_index(latitude, longitude)
            next_longitude = ring_index(latitude, longitude + 1)
            next_latitude = ring_index(latitude + 1, longitude)
            next_both = ring_index(latitude + 1, longitude + 1)
            add_sphere_face((current, next_latitude, next_both))
            add_sphere_face((current, next_both, next_longitude))
    for longitude in range(longitude_steps):
        add_sphere_face(
            (
                ring_index(latitude_steps - 1, longitude),
                south,
                ring_index(latitude_steps - 1, longitude + 1),
            )
        )

    inner_center = len(vertices)
    vertices.append((0.0, 0.0, 0.0))

    def add_inner_fan(boundary, reverse=False):
        for index in range(len(boundary) - 1):
            first, second = boundary[index : index + 2]
            face = (
                (inner_center, second, first)
                if reverse
                else (inner_center, first, second)
            )
            faces.append(face)
            face_types.append((False, True, False))
            smooth_faces.append(False)

    half_latitude = latitude_steps // 2
    quarter_longitude = longitude_steps // 4
    x_cut = [0] + [
        ring_index(latitude, quarter_longitude)
        for latitude in range(1, half_latitude + 1)
    ]
    y_cut = [0] + [ring_index(latitude, 0) for latitude in range(1, half_latitude + 1)]
    z_cut = [
        ring_index(half_latitude, longitude)
        for longitude in range(quarter_longitude + 1)
    ]
    add_inner_fan(x_cut, reverse=True)
    add_inner_fan(y_cut)
    add_inner_fan(z_cut)

    tube_segments = 48
    tube_sides = 6
    tube_radius = max(0.001, float(ellipse_thickness))
    ellipse_planes = (
        lambda angle: (
            (math.cos(angle), math.sin(angle), 0.0),
            (math.cos(angle), math.sin(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        lambda angle: (
            (math.cos(angle), 0.0, math.sin(angle)),
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
        ),
        lambda angle: (
            (0.0, math.cos(angle), math.sin(angle)),
            (0.0, math.cos(angle), math.sin(angle)),
            (1.0, 0.0, 0.0),
        ),
    )
    for plane_index, ellipse in enumerate(ellipse_planes):
        start = len(vertices)
        for segment in range(tube_segments):
            angle = 2.0 * math.pi * segment / tube_segments
            point, radial, normal = ellipse(angle)
            for side in range(tube_sides):
                cross_angle = 2.0 * math.pi * side / tube_sides
                vertices.append(
                    tuple(
                        point[axis] * 1.004
                        + tube_radius
                        * (
                            math.cos(cross_angle) * radial[axis]
                            + math.sin(cross_angle) * normal[axis]
                        )
                        for axis in range(3)
                    )
                )
        for segment in range(tube_segments):
            for side in range(tube_sides):
                current = start + segment * tube_sides + side
                next_side = start + segment * tube_sides + (side + 1) % tube_sides
                next_segment = (
                    start + ((segment + 1) % tube_segments) * tube_sides + side
                )
                next_both = (
                    start
                    + ((segment + 1) % tube_segments) * tube_sides
                    + (side + 1) % tube_sides
                )
                if plane_index == 1:
                    faces.extend(
                        (
                            (current, next_both, next_segment),
                            (current, next_side, next_both),
                        )
                    )
                else:
                    faces.extend(
                        (
                            (current, next_segment, next_both),
                            (current, next_both, next_side),
                        )
                    )
                face_types.extend(((False, False, True),) * 2)
                smooth_faces.extend((True, True))

    return vertices, faces, face_types, smooth_faces, tube_radius


def _build_ortep_template_mesh(ellipse_thickness=_ORTEP_ELLIPSE_THICKNESS):
    """Create a clipped ellipsoid template with cut faces and three rings."""
    vertices, faces, face_types, smooth_faces, tube_radius = _ortep_template_geometry(
        ellipse_thickness
    )
    mesh = bpy.data.meshes.new("CGN_ORTEP_Template_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for name, flag_index in (
        ("cgn_ortep_surface", 0),
        ("cgn_ortep_inner", 1),
        ("cgn_ortep_ellipse", 2),
    ):
        attribute = mesh.attributes.new(name, "BOOLEAN", "FACE")
        for polygon_index, face_type in enumerate(face_types):
            attribute.data[polygon_index].value = face_type[flag_index]
    for polygon, smooth in zip(mesh.polygons, smooth_faces):
        polygon.use_smooth = smooth
    mesh["cgn_ortep_template_version"] = 3
    mesh["cgn_ortep_ellipse_thickness"] = tube_radius
    return mesh


def _update_ortep_template_mesh(mesh, ellipse_thickness):
    """Refresh an existing template mesh in place.

    Returns ``False`` when the mesh cannot be updated in place (missing or
    stale attributes, different topology), in which case the caller must
    rebuild it.  ``foreach_set`` avoids reallocating the mesh and recreating
    the faces/attributes on every thickness tick.
    """
    if mesh is None or mesh.get("cgn_ortep_template_version") != 3:
        return False
    vertices, faces, _face_types, _smooth_faces, tube_radius = _ortep_template_geometry(
        ellipse_thickness
    )
    if len(mesh.vertices) != len(vertices) or len(mesh.polygons) != len(faces):
        return False
    for name in ("cgn_ortep_surface", "cgn_ortep_inner", "cgn_ortep_ellipse"):
        if mesh.attributes.get(name) is None:
            return False
    coordinates = [component for vertex in vertices for component in vertex]
    mesh.vertices.foreach_set("co", coordinates)
    mesh.update()
    mesh["cgn_ortep_ellipse_thickness"] = tube_radius
    return True


def _ensure_ortep_template():
    obj = bpy.data.objects.get(_ORTEP_TEMPLATE_OBJECT)
    if obj is None:
        scene = bpy.context.scene
        obj = bpy.data.objects.new(
            _ORTEP_TEMPLATE_OBJECT,
            _build_ortep_template_mesh(scene.cgn_ortep_ellipse_thickness),
        )
    elif (
        obj.type != "MESH"
        or obj.data.get("cgn_ortep_template_version") != 3
        or not math.isclose(
            float(obj.data.get("cgn_ortep_ellipse_thickness", -1.0)),
            bpy.context.scene.cgn_ortep_ellipse_thickness,
            rel_tol=1e-6,
        )
    ):
        old_mesh = obj.data
        obj.data = _build_ortep_template_mesh(
            bpy.context.scene.cgn_ortep_ellipse_thickness
        )
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    collection = bpy.data.collections.get("CGN_ORTEP_Assets")
    if collection is None:
        collection = bpy.data.collections.new("CGN_ORTEP_Assets")
    if collection.name not in {
        child.name for child in bpy.context.scene.collection.children
    }:
        bpy.context.scene.collection.children.link(collection)
    if obj.name not in {item.name for item in collection.objects}:
        collection.objects.link(obj)
    obj.hide_render = True
    obj.hide_set(True)
    return obj


def _make_ortep_materials(scene):
    inner = _make_material(
        "CGN_ORTEP_Inner_Faces",
        scene.cgn_ortep_inner_color,
        scene.cgn_ortep_inner_roughness,
    )
    ellipses = _make_material(
        "CGN_ORTEP_Ellipse_Lines",
        scene.cgn_ortep_ellipse_color,
        scene.cgn_ortep_ellipse_roughness,
    )
    return inner, ellipses


def _make_ellipsoid_node_group():
    """Build the ORTEP cutaway ellipsoid instancing node group."""
    node_group = bpy.data.node_groups.get(_NODE_GROUP_ELLIPSOIDS)
    if node_group is not None:
        # The group may already exist while the ORTEP template object was
        # deleted; refresh the Object Info pointer so it self-heals.
        template = _ensure_ortep_template()
        for node in node_group.nodes:
            if node.bl_idname == "GeometryNodeObjectInfo":
                node.inputs["Object"].default_value = template
        return node_group

    node_group = bpy.data.node_groups.new(_NODE_GROUP_ELLIPSOIDS, "GeometryNodeTree")
    sockets = {}
    sockets["Geometry"] = node_group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    for name, socket_type in (
        ("Material", "NodeSocketMaterial"),
        ("Inner Face Material", "NodeSocketMaterial"),
        ("Ellipse Material", "NodeSocketMaterial"),
        ("Probability Scale", "NodeSocketFloat"),
    ):
        sockets[name] = node_group.interface.new_socket(
            name=name,
            in_out="INPUT",
            socket_type=socket_type,
        )
    output_geometry = node_group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    sockets["Probability Scale"].default_value = 1.0

    nodes = node_group.nodes
    links = node_group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    group_in.location = (-1050, 100)
    group_out.location = (800, 100)
    object_info = nodes.new("GeometryNodeObjectInfo")
    object_info.transform_space = "ORIGINAL"
    object_info.inputs["Object"].default_value = _ensure_ortep_template()
    object_info.inputs["As Instance"].default_value = False
    object_info.location = (-800, -220)
    scale_attribute = nodes.new("GeometryNodeInputNamedAttribute")
    rotation_attribute = nodes.new("GeometryNodeInputNamedAttribute")
    scale_attribute.data_type = "FLOAT_VECTOR"
    scale_attribute.inputs["Name"].default_value = "cgn_scale"
    scale_attribute.location = (-800, 240)
    rotation_attribute.data_type = "FLOAT_VECTOR"
    rotation_attribute.inputs["Name"].default_value = "cgn_rotation"
    rotation_attribute.location = (-800, 450)
    vector_scale = nodes.new("ShaderNodeVectorMath")
    vector_scale.operation = "SCALE"
    vector_scale.location = (-350, 240)
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.location = (500, 100)

    material_nodes = []
    for index, (attribute_name, material_name) in enumerate(
        (
            ("cgn_ortep_surface", "Material"),
            ("cgn_ortep_inner", "Inner Face Material"),
            ("cgn_ortep_ellipse", "Ellipse Material"),
        )
    ):
        named_attribute = nodes.new("GeometryNodeInputNamedAttribute")
        named_attribute.data_type = "BOOLEAN"
        named_attribute.inputs["Name"].default_value = attribute_name
        set_material = nodes.new("GeometryNodeSetMaterial")
        set_material.location = (-400 + index * 220, -220)
        named_attribute.location = (-600 + index * 220, -400)
        links.new(
            named_attribute.outputs["Attribute"],
            set_material.inputs["Selection"],
        )
        links.new(
            group_in.outputs[sockets[material_name].identifier],
            set_material.inputs["Material"],
        )
        material_nodes.append(set_material)

    links.new(object_info.outputs["Geometry"], material_nodes[0].inputs["Geometry"])
    links.new(
        material_nodes[0].outputs["Geometry"],
        material_nodes[1].inputs["Geometry"],
    )
    links.new(
        material_nodes[1].outputs["Geometry"],
        material_nodes[2].inputs["Geometry"],
    )
    links.new(material_nodes[2].outputs["Geometry"], instance.inputs["Instance"])
    links.new(
        group_in.outputs[sockets["Geometry"].identifier], instance.inputs["Points"]
    )
    links.new(scale_attribute.outputs["Attribute"], vector_scale.inputs[0])
    links.new(
        group_in.outputs[sockets["Probability Scale"].identifier],
        vector_scale.inputs["Scale"],
    )
    links.new(vector_scale.outputs["Vector"], instance.inputs["Scale"])
    links.new(rotation_attribute.outputs["Attribute"], instance.inputs["Rotation"])
    links.new(
        instance.outputs["Instances"],
        group_out.inputs[output_geometry.identifier],
    )
    return node_group


def _euler_xyz_rotation_matrices(euler_values):
    """Return (N, 3, 3) rotation matrices for XYZ Euler triples.

    Matches ``mathutils.Euler(values, "XYZ").to_matrix()``, i.e. the
    composition ``R = Rz @ Ry @ Rx``.
    """
    angles = np.asarray(euler_values, dtype=np.float64).reshape(-1, 3)
    cos = np.cos(angles)
    sin = np.sin(angles)
    cx, cy, cz = cos[:, 0], cos[:, 1], cos[:, 2]
    sx, sy, sz = sin[:, 0], sin[:, 1], sin[:, 2]
    matrices = np.empty((angles.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = cz * cy
    matrices[:, 0, 1] = cz * sy * sx - sz * cx
    matrices[:, 0, 2] = cz * sy * cx + sz * sx
    matrices[:, 1, 0] = sz * cy
    matrices[:, 1, 1] = sz * sy * sx + cz * cx
    matrices[:, 1, 2] = sz * sy * cx - cz * sx
    matrices[:, 2, 0] = -sy
    matrices[:, 2, 1] = cy * sx
    matrices[:, 2, 2] = cy * cx
    return matrices


def _matrix_to_euler_xyz(matrices):
    """Return (N, 3) XYZ Euler angles for (N, 3, 3) rotation matrices.

    Inverse of :func:`_euler_xyz_rotation_matrices`, following the same
    convention as ``mathutils.Matrix.to_euler("XYZ")``.
    """
    matrix = np.asarray(matrices, dtype=np.float64)
    cosine_y = np.hypot(matrix[:, 0, 0], matrix[:, 1, 0])
    x = np.arctan2(matrix[:, 2, 1], matrix[:, 2, 2])
    y = np.arctan2(-matrix[:, 2, 0], cosine_y)
    z = np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])
    singular = cosine_y < 1e-8
    if np.any(singular):
        # Gimbal lock (y = ±90°): the X/Z split is ambiguous, so fold the
        # rotation into X and set Z to zero.
        negative = matrix[:, 2, 0] < 0.0
        gimbal_x = np.arctan2(
            np.where(negative, matrix[:, 0, 1], -matrix[:, 0, 1]),
            np.where(negative, matrix[:, 0, 2], -matrix[:, 0, 2]),
        )
        gimbal_y = np.where(negative, 0.5 * np.pi, -0.5 * np.pi)
        x = np.where(singular, gimbal_x, x)
        y = np.where(singular, gimbal_y, y)
        z = np.where(singular, 0.0, z)
    return np.stack((x, y, z), axis=1)


def _add_ellipsoid_attributes(mesh, atoms):
    """Store per-point semi-axes and rotations on an atom point mesh.

    All displacement tensors are decomposed in one batched ``eigh`` call and
    both attributes are written with ``foreach_set``.  The previous per-atom
    ``eigh`` plus per-element attribute writes dominated import time on large
    structures.
    """
    scene = getattr(bpy.context, "scene", None)
    hydrogen_scale = scene.cgn_hydrogen_scale if scene is not None else 1.0
    count = len(atoms)
    scales = np.empty((count, 3), dtype=np.float64)
    rotations = np.zeros((count, 3), dtype=np.float64)

    tensors = []
    tensor_indices = []
    tensor_size_scales = []
    for index, atom in enumerate(atoms):
        size_scale = (
            hydrogen_scale if str(atom.get("element", "")).upper() == "H" else 1.0
        )
        values = atom.get("u_cart")
        if values and len(values) == 6:
            numbers = [float(value) for value in values]
            u11, u22, u33 = numbers[0], numbers[1], numbers[2]
            if (
                all(math.isfinite(value) for value in numbers)
                and u11 > 0.0
                and u22 > 0.0
                and u33 > 0.0
            ):
                u12, u13, u23 = numbers[3], numbers[4], numbers[5]
                tensors.append(((u11, u12, u13), (u12, u22, u23), (u13, u23, u33)))
                tensor_indices.append(index)
                tensor_size_scales.append(size_scale)
                continue
        # No usable ADP: keep the atom spherical on the ball-and-stick radius.
        radius = _atom_radius(atom.get("vdw_radius", _DEFAULT_VDW_RADIUS), "BALL_STICK")
        scales[index] = radius * size_scale

    if tensors:
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.asarray(tensors, dtype=np.float64)
        )
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        axes = np.sqrt(eigenvalues * _ELLIPSOID_CHI2_50)
        axes = np.maximum(
            axes
            * _DISPLAY_SCALE
            * np.asarray(tensor_size_scales, dtype=np.float64)[:, np.newaxis],
            1e-4,
        )
        flip = np.linalg.det(eigenvectors) < 0.0
        eigenvectors[flip, :, 2] *= -1.0
        indices = np.asarray(tensor_indices, dtype=np.intp)
        scales[indices] = axes
        rotations[indices] = _matrix_to_euler_xyz(eigenvectors)

    scale_attribute = mesh.attributes.new("cgn_scale", "FLOAT_VECTOR", "POINT")
    rotation_attribute = mesh.attributes.new("cgn_rotation", "FLOAT_VECTOR", "POINT")
    scale_attribute.data.foreach_set("vector", scales.astype(np.float32).ravel())
    rotation_attribute.data.foreach_set("vector", rotations.astype(np.float32).ravel())


def _orient_ortep_openings(scene, view_direction=None, camera_position=None):
    """Point each ellipsoid's open quadrant at the viewer/camera.

    Works on the whole ``cgn_scale`` attribute at once (``foreach_get`` /
    ``foreach_set``) instead of building a ``mathutils.Euler`` and writing one
    element at a time per vertex.
    """
    root = _active_root(scene)
    if root is None or scene.cgn_model != "ELLIPSOID":
        return
    if camera_position is not None:
        camera_vector = np.array(tuple(camera_position), dtype=np.float64)
        direction_world = None
    elif view_direction is not None:
        camera_vector = None
        direction_world = np.array(tuple(view_direction), dtype=np.float64)
    else:
        return

    for obj in root.objects:
        if (
            obj.get("cgn_role") != "atoms"
            or str(obj.get("cgn_element", "")).upper() == "H"
        ):
            continue
        data = obj.data
        scale_attribute = data.attributes.get("cgn_scale")
        rotation_attribute = data.attributes.get("cgn_rotation")
        if scale_attribute is None or rotation_attribute is None:
            continue
        count = len(data.vertices)
        if count == 0:
            continue
        scales = np.empty(count * 3, dtype=np.float32)
        rotations = np.empty(count * 3, dtype=np.float32)
        scale_attribute.data.foreach_get("vector", scales)
        rotation_attribute.data.foreach_get("vector", rotations)
        scales = scales.reshape(-1, 3).astype(np.float64)
        rotations = rotations.reshape(-1, 3)

        world_to_object = obj.matrix_world.to_3x3().inverted_safe()
        to_object = np.array(
            [[world_to_object[row][column] for column in range(3)] for row in range(3)],
            dtype=np.float64,
        )
        if camera_vector is None:
            direction = to_object @ direction_world
            directions = np.broadcast_to(direction, scales.shape)
        else:
            coordinates = np.empty(count * 3, dtype=np.float32)
            data.vertices.foreach_get("co", coordinates)
            coordinates = coordinates.reshape(-1, 3).astype(np.float64)
            world_matrix = obj.matrix_world
            to_world = np.array(
                [
                    [world_matrix[row][column] for column in range(3)]
                    for row in range(3)
                ],
                dtype=np.float64,
            )
            translation = np.array(tuple(world_matrix.translation), dtype=np.float64)
            points = coordinates @ to_world.T + translation
            directions = (camera_vector - points) @ to_object.T

        matrices = _euler_xyz_rotation_matrices(rotations)
        local_directions = np.einsum("nji,nj->ni", matrices, directions)
        oriented = np.abs(scales) * np.where(local_directions >= 0.0, 1.0, -1.0)
        if np.any(oriented != scales):
            scale_attribute.data.foreach_set(
                "vector", oriented.astype(np.float32).ravel()
            )
            data.update()
            obj.update_tag(refresh={"OBJECT", "DATA"})


@persistent
def _orient_ortep_for_render(scene):
    # Slider changes are debounced; apply any pending one now so the rendered
    # frame never shows stale ellipsoid thickness or bonds.
    _flush_debounce()
    if scene.camera is not None:
        _orient_ortep_to_camera(scene, scene.camera)
    _ORTEP_VIEW_SIGNATURES.pop(scene.as_pointer(), None)


def _orient_ortep_to_camera(scene, camera):
    if camera.data.type == "ORTHO":
        camera_direction = camera.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
        _orient_ortep_openings(scene, view_direction=camera_direction)
    else:
        _orient_ortep_openings(
            scene, camera_position=camera.matrix_world.translation.copy()
        )


def _ortep_view_signature(view_direction=None, camera=None):
    if camera is not None:
        if camera.data.type == "ORTHO":
            direction = camera.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
            return ("ortho", *(round(value, 4) for value in direction))
        position = camera.matrix_world.translation
        return ("camera", *(round(value, 4) for value in position))
    direction = view_direction.normalized()
    return ("view", *(round(value, 4) for value in direction))


_ORTEP_VIEW_TIMER_INTERVAL = 0.25


def _start_ortep_view_timer():
    if not bpy.app.timers.is_registered(_update_ortep_view_orientation):
        bpy.app.timers.register(
            _update_ortep_view_orientation,
            first_interval=_ORTEP_VIEW_TIMER_INTERVAL,
            persistent=True,
        )


def _stop_ortep_view_timer():
    if bpy.app.timers.is_registered(_update_ortep_view_orientation):
        bpy.app.timers.unregister(_update_ortep_view_orientation)


def _sync_ortep_view_timer(scene=None):
    """Run the ORTEP view timer only while some scene is in ELLIPSOID.

    The ellipsoids have to follow the viewport, but outside ORTEP mode there is
    nothing to orient: keeping the 4 Hz timer registered forever woke Blender
    up for no reason.  Every scene is considered so switching between scenes
    cannot leave the timer stopped while an ORTEP scene is still open.
    """
    scenes = list(getattr(bpy.data, "scenes", ()))
    if scene is not None and scene.name not in {item.name for item in scenes}:
        scenes.append(scene)
    if any(_current_model(item) == "ELLIPSOID" for item in scenes):
        _start_ortep_view_timer()
    else:
        _stop_ortep_view_timer()


@persistent
def _sync_ortep_view_timer_after_load(_unused=None):
    """Re-evaluate the ELLIPSOID-only timer after a .blend is loaded."""
    _sync_ortep_view_timer()


def _update_ortep_view_orientation():
    if not any(_current_model(item) == "ELLIPSOID" for item in bpy.data.scenes):
        # Safety net: if the model changed without going through
        # _update_model, stop instead of polling forever.
        return None
    scene = bpy.context.scene
    if scene is None or _current_model(scene) != "ELLIPSOID":
        # Another scene is in ORTEP mode; keep the timer alive for it.
        return _ORTEP_VIEW_TIMER_INTERVAL
    view_direction = None
    camera = None
    window_manager = getattr(bpy.context, "window_manager", None)
    windows = getattr(window_manager, "windows", ())
    for window in windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = area.spaces.active.region_3d
            if region.view_perspective == "CAMERA" and scene.camera is not None:
                camera = scene.camera
            else:
                view_direction = region.view_rotation @ Vector((0.0, 0.0, 1.0))
            break
        if view_direction is not None or camera is not None:
            break
    if view_direction is None and camera is None:
        if scene.camera is not None:
            camera = scene.camera
        else:
            view_direction = Vector((1.0, 1.0, 1.0))
    signature = _ortep_view_signature(view_direction, camera)
    scene_key = scene.as_pointer()
    if _ORTEP_VIEW_SIGNATURES.get(scene_key) != signature:
        if camera is not None:
            _orient_ortep_to_camera(scene, camera)
        else:
            _orient_ortep_openings(scene, view_direction=view_direction)
        _ORTEP_VIEW_SIGNATURES[scene_key] = signature
    return _ORTEP_VIEW_TIMER_INTERVAL


def _set_modifier_input(modifier, node_group, socket_name, value):
    identifier = _socket_identifier(node_group, socket_name)
    inputs = getattr(getattr(modifier, "properties", None), "inputs", None)
    input_value = getattr(inputs, identifier, None)
    if input_value is not None and hasattr(input_value, "value"):
        input_value.value = value
        return
    try:
        modifier[identifier] = value
    except TypeError:
        if input_value is None:
            raise RuntimeError(
                f"Geometry Nodes modifier input '{socket_name}' is unavailable."
            )
        input_value.value = value


def _make_material(name, color, roughness=0.25):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    # An existing material of this name may have nodes disabled, which would
    # make material.node_tree None.
    material.use_nodes = True
    material.diffuse_color = color
    shader = _principled_shader(material)
    if shader is not None:
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = roughness
    return material


def _atom_style(scene, element):
    if int(scene.get("cgn_atom_style_version", 0)) < 1:
        for existing in scene.cgn_atom_styles:
            if math.isclose(existing.roughness, 0.35, abs_tol=1e-6):
                existing.roughness = 0.25
        scene["cgn_atom_style_version"] = 1
    for style in scene.cgn_atom_styles:
        if style.element == element:
            return style
    color = _ELEMENT_COLORS.get(element, _DEFAULT_COLOR)
    material = bpy.data.materials.get(f"CGN_Mat_{element}")
    if material is not None:
        color = tuple(material.diffuse_color)
    style = scene.cgn_atom_styles.add()
    style.element = element
    style.color = color
    style.roughness = 0.25
    return style


def _existing_atom_style(scene, element):
    """Read-only style lookup, safe to call from panel ``draw``.

    Mutating scene data during redraw (the legacy migration and style creation
    in ``_atom_style``) is not allowed, so panels must never call that helper.
    Styles are created by the import/bond-rebuild paths instead; elements with
    no style yet are drawn as a hint until the next structural operation.
    """
    for style in scene.cgn_atom_styles:
        if style.element == element:
            return style
    return None


def _add_geometry_nodes_modifier(
    obj,
    node_group,
    radius=None,
    material=None,
    probability=None,
    inner_material=None,
    ellipse_material=None,
):
    modifier = obj.modifiers.new("Geometry Nodes", "NODES")
    modifier.node_group = node_group
    if radius is not None:
        _set_modifier_input(modifier, node_group, "Radius", radius)
    if material is not None:
        _set_modifier_input(modifier, node_group, "Material", material)
    if probability is not None:
        _set_modifier_input(modifier, node_group, "Probability Scale", probability)
    if inner_material is not None:
        _set_modifier_input(modifier, node_group, "Inner Face Material", inner_material)
    if ellipse_material is not None:
        _set_modifier_input(modifier, node_group, "Ellipse Material", ellipse_material)


def _remove_object(obj):
    """Delete an object and the mesh/curve datablock it leaves orphaned.

    Removing an object never frees its data; without this the session keeps
    every atom, bond and unit-cell mesh that was ever generated.
    """
    data = getattr(obj, "data", None)
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is None or data.users != 0:
        return
    if isinstance(data, bpy.types.Curve):
        bpy.data.curves.remove(data)
    elif isinstance(data, bpy.types.Mesh):
        bpy.data.meshes.remove(data)


def _clear_collection(collection):
    """Delete a structure collection and all of its object datablocks.

    Children are cleared recursively *before* being unlinked, so their own
    meshes/curves are purged as well instead of being left behind in
    ``bpy.data``.
    """
    for child in list(collection.children):
        _clear_collection(child)
        bpy.data.collections.remove(child, do_unlink=True)
    for obj in list(collection.objects):
        _remove_object(obj)


def _purge_structure_roots(scene=None, keep=None):
    """Delete every structure collection except *keep*.

    The add-on drives a single active structure, but an import only removed
    the collection referenced by ``scene.cgn_active_structure``.  Any older
    structure that had lost that pointer (renamed collection, stale scene
    property, import that failed after creating its collection) therefore
    stayed in the file with all of its meshes.  Purging by ``cgn_role``
    instead of by that pointer removes those leftovers.
    """
    removed = 0
    for collection in list(bpy.data.collections):
        if collection is keep or collection.get("cgn_role") != "structure":
            continue
        _clear_collection(collection)
        bpy.data.collections.remove(collection, do_unlink=True)
        removed += 1
    if scene is not None:
        active = scene.cgn_active_structure
        if active and bpy.data.collections.get(active) is None:
            scene.cgn_active_structure = keep.name if keep is not None else ""
    return removed


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
        frac = [float(site["frac"][axis]) + translation[axis] for axis in range(3)]
        cart = [
            frac[0] * cell_matrix[axis]
            + frac[1] * cell_matrix[3 + axis]
            + frac[2] * cell_matrix[6 + axis]
            for axis in range(3)
        ]
        atom = dict(site)
        atom.update(
            {
                "frac": frac,
                "cart": cart,
                "p1_index": site_index,
                "growth_translation": list(translation),
            }
        )
        if index >= initial_count:
            atom["label"] = (
                f"{site.get('source_label', site.get('label', 'X'))} "
                f"[P1:{site_index}; {translation[0]},"
                f"{translation[1]},{translation[2]}]"
            )
        atoms.append(atom)
    return atoms


def _structure_cache_key(root):
    """Identity + revision key for the parsed-tables cache.

    ``cgn_revision`` is bumped by every operation that rewrites the stored
    tables, so a matching key means the cached sites/instances/atoms are already
    current and the JSON blobs need not be parsed again.
    """
    return (root.get("cgn_instance_id", root.name), int(root.get("cgn_revision", 0)))


def _invalidate_structure_cache():
    """Drop the parsed-structure cache (e.g. on import/unregister)."""
    global _STRUCTURE_CACHE, _POLY_SEARCH_CACHE
    _STRUCTURE_CACHE = None
    _POLY_SEARCH_CACHE = None


def _bump_structure_revision(root):
    """Mark the stored structure tables as changed for the cache."""
    root["cgn_revision"] = int(root.get("cgn_revision", 0)) + 1


def _structure_tables(root):
    """Return (sites, instances, cell_matrix, initial_count) for a structure.

    The parsed tables are cached per (structure id, revision).  Delete, grow,
    fill and the polyhedron paths call this several times per operation, and
    re-running ``json.loads`` over the full instance table each time dominated
    the cost on large CIFs (e.g. ``Ca_autored.cif``).  The returned lists are
    shared with the cache: callers must not mutate them (copy first if needed).
    """
    global _STRUCTURE_CACHE
    key = _structure_cache_key(root)
    cached = _STRUCTURE_CACHE
    if cached is not None and cached["key"] == key:
        return (
            cached["sites"],
            cached["instances"],
            cached["cell_matrix"],
            cached["initial_count"],
        )
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
    _STRUCTURE_CACHE = {
        "key": key,
        "sites": sites,
        "instances": instances,
        "cell_matrix": cell_matrix,
        "initial_count": initial_count,
        "atoms": None,
    }
    return sites, instances, cell_matrix, initial_count


def _structure_atoms(root):
    """Return the materialized atom list, cached per (structure id, revision)."""
    global _STRUCTURE_CACHE
    key = _structure_cache_key(root)
    cached = _STRUCTURE_CACHE
    if cached is not None and cached["key"] == key and cached["atoms"] is not None:
        return cached["atoms"]
    sites, instances, cell_matrix, initial_count = _structure_tables(root)
    atoms = _materialize_atoms(sites, cell_matrix, instances, initial_count)
    if _STRUCTURE_CACHE is not None and _STRUCTURE_CACHE["key"] == key:
        _STRUCTURE_CACHE["atoms"] = atoms
    return atoms


def _structure_atom_count(root):
    """Number of atom instances, read from the cached scalar when available.

    Panel draws call this on every redraw, so avoid json.loads on the whole
    instance table: ``cgn_atom_count`` is written alongside ``cgn_instances``
    by :func:`_set_instances`. Structures stored before that scalar existed fall
    back to counting the JSON once.
    """
    count = root.get("cgn_atom_count")
    if count is not None:
        return int(count)
    instances_json = root.get("cgn_instances", "")
    if not instances_json:
        return 0
    return len(json.loads(instances_json)) // 4


def _set_instances(root, instances):
    """Store the instance table and its derived atom count in one place.

    Bumping ``cgn_revision`` invalidates the parsed-tables cache so the next
    read re-parses only when the underlying data actually changed.
    """
    root["cgn_instances"] = json.dumps(instances)
    root["cgn_atom_count"] = len(instances) // 4
    _bump_structure_revision(root)


def _snapshot_structure_state(root, pair_settings=None):
    """Capture the data custom-properties (plus per-site polyhedron centers)
    that fill/grow mutate, so a mid-rebuild failure can be rolled back
    atomically.  Atoms/bonds are fully derived from ``cgn_instances``; the
    polyhedron centers are read back from each face object's ``cgn_poly_map``.

    ``pair_settings`` (the connectivity pairs used to build the current bonds)
    is copied so a rollback can reproduce the exact pre-operation bond set even
    when the user toggled the pairs in the meantime.
    """
    keys = [
        "cgn_instances",
        "cgn_atom_count",
        "cgn_initial_count",
        "cgn_selected_indices",
        "cgn_growth_shells",
        "cgn_fill_mode",
        "cgn_cell_atoms",
        "cgn_fill_repeats",
        "cgn_bond_settings_signature",
    ]
    props = {key: root.get(key) for key in keys}
    pair_records = [
        {
            "element_a": str(item.element_a),
            "element_b": str(item.element_b),
            "enabled": bool(item.enabled),
            "min_distance": float(item.min_distance),
            "max_distance": float(item.max_distance),
        }
        for item in (pair_settings or ())
    ]
    centers_by_style = {}
    for obj in root.objects:
        if obj.get("cgn_role") != "polyhedron":
            continue
        element = str(obj.get("cgn_poly_element", "")).upper()
        site = str(obj.get("cgn_poly_site_label", "") or "")
        try:
            poly_map = json.loads(obj.get("cgn_poly_map", "{}"))
        except (TypeError, ValueError):
            continue
        indices = {int(key) for key in poly_map}
        style_key = (element, site)
        centers_by_style.setdefault(style_key, set()).update(indices)
    return {
        "props": props,
        "pairs": pair_records,
        "polyhedra": {key: sorted(value) for key, value in centers_by_style.items()},
    }


def _rollback_structure_state(root, snapshot, scene, pair_settings):
    """Restore a structure to a captured pre-operation state after a failed
    fill/grow.  Restores the stored ``cgn_*`` props, drops every generated
    display object, and rebuilds atoms/bonds/polyhedra/unit-cell from the
    restored instances so the viewport and stored data stay consistent.
    """
    props = snapshot["props"]
    for key, value in props.items():
        if value is None:
            if key in root:
                del root[key]
        else:
            root[key] = value
    # The instances were restored directly (bypassing _set_instances), so bump
    # the revision to force the parsed-tables cache to be rebuilt.
    _bump_structure_revision(root)
    # Rebuild bonds with the pairs that produced the captured bond signature so
    # the rollback is faithful even if the user toggled pairs in the meantime.
    rollback_pairs = _pairs_from_records(snapshot.get("pairs")) or pair_settings
    for obj in list(root.all_objects):
        role = obj.get("cgn_role")
        if role in {
            "atoms",
            "bonds",
            "growth_selection",
            "polyhedron",
            "polyhedron_prototype",
        }:
            _remove_object(obj)
    for child in list(root.children):
        if child.get("cgn_role") == "polyhedron_prototypes" and not child.all_objects:
            bpy.data.collections.remove(child)
    try:
        atoms = _structure_atoms(root)
    except (ValueError, TypeError):
        return
    try:
        selected = json.loads(props.get("cgn_selected_indices", "[]"))
    except (TypeError, ValueError):
        selected = []
    _make_atom_objects(root, atoms)
    _update_growth_selection(root, atoms, selected)
    _bond_objects(root, atoms, rollback_pairs)
    _apply_unit_cell_settings(scene)
    poly_centers = []
    for centers in snapshot["polyhedra"].values():
        poly_centers.extend(centers)
    if poly_centers:
        _build_polyhedra_objects(root, atoms, poly_centers, scene)


def _pairs_from_records(records):
    """Rebuild a list of minimal connectivity-pair objects from captured
    records, or return ``[]`` when there are no records to restore from."""
    if not records:
        return []
    pairs = []
    for record in records:
        pair = _ConnectivityRecord()
        pair.element_a = record["element_a"]
        pair.element_b = record["element_b"]
        pair.enabled = record["enabled"]
        pair.min_distance = record["min_distance"]
        pair.max_distance = record["max_distance"]
        pairs.append(pair)
    return pairs


class _ConnectivityRecord:
    """Minimal stand-in for ``CGN_ConnectivityPair`` used to replay captured
    connectivity settings during rollback without touching Blender data."""

    __slots__ = ("element_a", "element_b", "enabled", "min_distance", "max_distance")

    def __init__(self):
        self.element_a = ""
        self.element_b = ""
        self.enabled = True
        self.min_distance = 0.1
        self.max_distance = 3.0


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
    _apply_hydrogen_visibility(_active_root(scene), scene.cgn_show_hydrogens)


def _update_model(scene, context):
    """Rebuild atoms and bonds when the rendering model changes."""
    root = _active_root(scene)
    if root is None:
        return
    try:
        sites, instances, cell_matrix, initial_count = _structure_tables(root)
        atoms = _materialize_atoms(sites, cell_matrix, instances, initial_count)
        _rebuild_atom_objects(root, atoms)
        _bond_objects(root, atoms, scene.cgn_connectivity_pairs)
        _apply_hydrogen_visibility(root, scene.cgn_show_hydrogens)
        context.view_layer.update()
    except Exception:
        traceback.print_exc()


def _update_hydrogen_scale(scene, context):
    if _current_model(scene) != "ELLIPSOID":
        return
    _debounce("hydrogen_scale", lambda: _apply_hydrogen_scale(scene))


def _apply_hydrogen_scale(scene):
    """Resize the ellipsoid-mode hydrogen spheres without rebuilding atoms.

    In ORTEP (ELLIPSOID) the hydrogen atoms are plain spheres produced by the
    atom node group, so only that modifier's ``Radius`` input has to change.
    Non-hydrogen atoms use per-point ``cgn_scale`` attributes and are
    unaffected by the hydrogen scale.
    """
    root = _active_root(scene)
    if root is None or _current_model(scene) != "ELLIPSOID":
        return
    scale = float(scene.cgn_hydrogen_scale)
    for obj in root.objects:
        if (
            obj.get("cgn_role") != "atoms"
            or str(obj.get("cgn_element", "")).upper() != "H"
        ):
            continue
        radius = (
            float(obj.get("cgn_vdw_radius", _DEFAULT_VDW_RADIUS))
            * _BALL_ATOM_VDW_FACTOR
            * scale
            * _DISPLAY_SCALE
        )
        updated = False
        for modifier in obj.modifiers:
            if (
                modifier.type == "NODES"
                and modifier.node_group is not None
                and modifier.node_group.name == _NODE_GROUP_ATOMS
            ):
                _set_modifier_input(modifier, modifier.node_group, "Radius", radius)
                updated = True
        if updated:
            obj.update_tag(refresh={"OBJECT", "DATA"})


def _update_wire_sticks_radius(scene, context):
    if scene.cgn_model != "WIRE_STICKS":
        return
    root = _active_root(scene)
    if root is None:
        return
    radius = float(scene.cgn_wire_sticks_radius) * _DISPLAY_SCALE
    for obj in root.objects:
        if obj.get("cgn_role") not in {"atoms", "bonds"}:
            continue
        for modifier in obj.modifiers:
            if modifier.type == "NODES" and modifier.node_group is not None:
                _set_modifier_input(modifier, modifier.node_group, "Radius", radius)
                obj.update_tag(refresh={"OBJECT", "DATA"})
    if context.view_layer is not None:
        context.view_layer.update()


def _update_ellipsoid_probability(scene, context):
    """Rescale existing ellipsoids without rebuilding geometry."""
    root = _active_root(scene)
    if root is None or _current_model(scene) != "ELLIPSOID":
        return
    if any(
        modifier.type == "NODES"
        and modifier.node_group is not None
        and modifier.node_group.name == _LEGACY_NODE_GROUP_ELLIPSOIDS
        for obj in root.objects
        if obj.get("cgn_role") == "atoms"
        for modifier in obj.modifiers
    ):
        _update_model(scene, context)
        return
    factor = _ellipsoid_probability_factor(scene.cgn_ellipsoid_probability)
    for obj in root.objects:
        if obj.get("cgn_role") != "atoms":
            continue
        is_hydrogen = str(obj.get("cgn_element", "")).upper() == "H"
        atom_factor = 1.0 if is_hydrogen else factor
        updated = False
        for modifier in obj.modifiers:
            if (
                modifier.type == "NODES"
                and modifier.node_group is not None
                and modifier.node_group.name == _NODE_GROUP_ELLIPSOIDS
            ):
                try:
                    _set_modifier_input(
                        modifier,
                        modifier.node_group,
                        "Probability Scale",
                        atom_factor,
                    )
                    updated = True
                except Exception:
                    traceback.print_exc()
        if updated:
            obj.update_tag(refresh={"OBJECT", "DATA"})
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
    ellipsoids = model == "ELLIPSOID"
    scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        _ORTEP_VIEW_SIGNATURES.pop(scene.as_pointer(), None)
    if ellipsoids:
        node_group = _make_ellipsoid_node_group()
        hydrogen_node_group = _make_node_group(_NODE_GROUP_ATOMS, is_bond_group=False)
        scene = getattr(bpy.context, "scene", None)
        inner_material, ellipse_material = _make_ortep_materials(scene)
        probability = _ellipsoid_probability_factor(
            getattr(scene, "cgn_ellipsoid_probability", 50.0)
        )
    else:
        node_group = _make_node_group(_NODE_GROUP_ATOMS, is_bond_group=False)
        hydrogen_node_group = node_group
        probability = None
        inner_material = None
        ellipse_material = None
    for element, element_atoms in sorted(by_element.items()):
        scene = bpy.context.scene
        style = _atom_style(scene, element)
        size_scale = (
            scene.cgn_hydrogen_scale if element == "H" and model == "ELLIPSOID" else 1.0
        )
        is_ortep_ellipsoid = ellipsoids and element != "H"
        mesh = bpy.data.meshes.new(f"CGN_Atoms_{element}_Points")
        mesh.from_pydata(
            [_display_point(atom["cart"]) for atom in element_atoms],
            [],
            [],
        )
        if is_ortep_ellipsoid:
            _add_ellipsoid_attributes(mesh, element_atoms)
        mesh.update()
        obj = bpy.data.objects.new(f"CGN_Atoms_{element}", mesh)
        obj["cgn_role"] = "atoms"
        obj["cgn_element"] = element
        obj["cgn_vdw_radius"] = float(
            element_atoms[0].get("vdw_radius", _DEFAULT_VDW_RADIUS)
        )
        root.objects.link(obj)
        if element == "H" and not _show_hydrogens():
            obj.hide_viewport = True
            obj.hide_render = True
        material = _make_material(f"CGN_Mat_{element}", style.color, style.roughness)
        if is_ortep_ellipsoid:
            _add_geometry_nodes_modifier(
                obj,
                node_group,
                material=material,
                probability=1.0 if element == "H" else probability,
                inner_material=inner_material,
                ellipse_material=ellipse_material,
            )
        else:
            if ellipsoids:
                node_group_for_atom = hydrogen_node_group
                radius = (
                    obj["cgn_vdw_radius"]
                    * _BALL_ATOM_VDW_FACTOR
                    * size_scale
                    * _DISPLAY_SCALE
                )
            elif model == "BALL_STICK":
                node_group_for_atom = node_group
                radius = obj["cgn_vdw_radius"] * style.size * _DISPLAY_SCALE
            else:
                node_group_for_atom = node_group
                radius = _atom_radius(obj["cgn_vdw_radius"], model)
                radius *= size_scale
            _add_geometry_nodes_modifier(obj, node_group_for_atom, radius, material)
    _sync_ortep_view_timer(scene)


def _rebuild_atom_objects(root, atoms):
    for obj in list(root.objects):
        if obj.get("cgn_role") == "atoms":
            _remove_object(obj)
    _make_atom_objects(root, atoms)


def _update_growth_selection(root, atoms, indices):
    """Persist the selected-atom indices and refresh the highlight object.

    Used at modal boundaries and after structural operations, where persistence
    is the point.  The highlight mesh itself is reused in place via
    ``_set_growth_selection_mesh`` so repeated updates do not rebuild the
    object/material/modifier graph.
    """
    root["cgn_selected_indices"] = json.dumps(sorted(set(indices)))
    _set_growth_selection_mesh(root, atoms, indices)


def _refresh_growth_selection(root, atoms, indices):
    """Refresh only the on-screen highlight, without touching persisted state.

    The selection modal already keeps the live set in memory during the drag,
    so re-serializing ``cgn_selected_indices`` on every click/box release is
    wasted work; the value is only read back at a modal boundary.
    """
    _set_growth_selection_mesh(root, atoms, indices)


def _set_growth_selection_mesh(root, atoms, indices):
    """Update the highlight point object for the selected atoms.

    Reuses the existing ``growth_selection`` object (and its mesh, material and
    Geometry-Nodes modifier) when present, only rewriting the mesh vertices;
    it is created once and removed when nothing is selected.
    """
    selected_atoms = [
        atoms[index] for index in sorted(set(indices)) if 0 <= index < len(atoms)
    ]
    obj = next(
        (o for o in root.objects if o.get("cgn_role") == "growth_selection"),
        None,
    )
    if not selected_atoms:
        if obj is not None:
            _remove_object(obj)
        return
    if obj is None:
        mesh = bpy.data.meshes.new("CGN_GrowthSelection_Points")
        obj = bpy.data.objects.new("CGN_GrowthSelection", mesh)
        obj["cgn_role"] = "growth_selection"
        root.objects.link(obj)
        material = _make_material("CGN_GrowthSelection_Material", (1.0, 0.48, 0.02, 1.0))
        _add_geometry_nodes_modifier(
            obj,
            _make_node_group(_NODE_GROUP_ATOMS, is_bond_group=False),
            0.9 * _DISPLAY_SCALE,
            material,
        )
    # ``from_pydata`` on a mesh that already has vertices cannot resize it
    # (the co buffer length is fixed), so drop the old geometry first and
    # rewrite in place.  The object, material and Geometry-Nodes modifier are
    # still reused --- only the mesh data is rewritten, not the object graph.
    obj.data.clear_geometry()
    obj.data.from_pydata(
        [_display_point(atom["cart"]) for atom in selected_atoms], [], []
    )
    obj.data.update()


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
            color[0],
            color[1],
            color[2],
            1.0,
        )
        shader.inputs["Roughness"].default_value = 0.5
        if "Alpha" in shader.inputs:
            shader.inputs["Alpha"].default_value = alpha
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    elif hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
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
        fx * cell_matrix[axis] + fy * cell_matrix[3 + axis] + fz * cell_matrix[6 + axis]
        for axis in range(3)
    )


def _clear_unit_cell(root):
    for obj in list(root.objects):
        if obj.get("cgn_role") not in {
            "unit_cell",
            "unit_cell_edge",
            "unit_cell_face",
            "unit_cell_marker",
        }:
            continue
        _remove_object(obj)


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


def _show_unit_cell(root, cell_matrix, settings=None, repeats=(1, 1, 1)):
    """Draw one or more unit cells as a single Geometry-Nodes object.

    Faces, beveled edges and corner markers all live in one mesh; the
    single GN modifier rebuilds the edges and markers from the mesh, so the
    panel can still adjust face/edge color, transparency and thickness.
    """
    _clear_unit_cell(root)
    _face, _alpha, _edge, thickness = _unit_cell_values(settings)
    edge_corners = [
        (index, index ^ (1 << bit))
        for index in range(8)
        for bit in range(3)
        if index < (index ^ (1 << bit))
    ]
    face_corners = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    # Deduplicate grid corners so markers instance once per unique corner.
    corner_index = {}
    corner_positions = []

    def corner(i, j, k, tx, ty, tz):
        key = (i + tx, j + ty, k + tz)
        if key in corner_index:
            return corner_index[key]
        position = _display_point(_cell_corner(cell_matrix, i + tx, j + ty, k + tz))
        index = len(corner_positions)
        corner_index[key] = index
        corner_positions.append(position)
        return index

    edges = []
    faces = []
    for tx in range(repeats[0]):
        for ty in range(repeats[1]):
            for tz in range(repeats[2]):
                base_indices = [
                    corner(i, j, k, tx, ty, tz)
                    for i in range(2)
                    for j in range(2)
                    for k in range(2)
                ]
                # base_indices order matches the original 0..7 corner order:
                # i outer, j middle, k inner -> index = i*4 + j*2 + k
                for first, second in edge_corners:
                    edges.append((base_indices[first], base_indices[second]))
                for face in face_corners:
                    faces.append(tuple(base_indices[index] for index in face))

    mesh = bpy.data.meshes.new("CGN_UnitCell")
    mesh.from_pydata(corner_positions, edges, faces)
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = False

    node_group = _make_unit_cell_node_group()
    obj = bpy.data.objects.new("CGN_UnitCell", mesh)
    obj["cgn_role"] = "unit_cell"
    # Remember the inputs this mesh was built from, so the refresh guard can
    # compare the actual matrix/grid instead of guessing from the vertex count.
    obj["cgn_unit_cell_matrix"] = json.dumps([float(value) for value in cell_matrix])
    obj["cgn_unit_cell_repeats"] = json.dumps([int(value) for value in repeats])
    root.objects.link(obj)

    modifier = obj.modifiers.new("Geometry Nodes", "NODES")
    modifier.node_group = node_group
    face_material = _make_unit_cell_face_material(settings)
    edge_material = _make_unit_cell_edge_material(settings)
    _set_modifier_input(modifier, node_group, "Face Material", face_material)
    _set_modifier_input(modifier, node_group, "Edge Material", edge_material)
    _set_modifier_input(
        modifier, node_group, "Edge Thickness", thickness * _DISPLAY_SCALE
    )
    _set_modifier_input(
        modifier, node_group, "Marker Radius", thickness * _DISPLAY_SCALE
    )
    obj.update_tag(refresh={"OBJECT", "DATA"})


def _update_unit_cell_objects(obj, cell_matrix, repeats, settings):
    _face, _alpha, _edge, thickness = _unit_cell_values(settings)
    node_group = _make_unit_cell_node_group()
    if obj.modifiers and obj.modifiers[0].type == "NODES":
        modifier = obj.modifiers[0]
    else:
        modifier = obj.modifiers.new("Geometry Nodes", "NODES")
        modifier.node_group = node_group
    modifier.node_group = node_group
    face_material = _make_unit_cell_face_material(settings)
    edge_material = _make_unit_cell_edge_material(settings)
    _set_modifier_input(modifier, node_group, "Face Material", face_material)
    _set_modifier_input(modifier, node_group, "Edge Material", edge_material)
    _set_modifier_input(
        modifier, node_group, "Edge Thickness", thickness * _DISPLAY_SCALE
    )
    _set_modifier_input(
        modifier, node_group, "Marker Radius", thickness * _DISPLAY_SCALE
    )
    obj.update_tag(refresh={"OBJECT", "DATA"})


def _unit_cell_signature(obj):
    """Inputs the current box mesh was built from, or ``None`` if unknown.

    Boxes created before this property existed (or manually edited meshes)
    return ``None`` so the caller rebuilds them once from the real settings.
    """
    if obj is None:
        return None
    try:
        matrix = json.loads(obj.get("cgn_unit_cell_matrix", ""))
        repeats = json.loads(obj.get("cgn_unit_cell_repeats", ""))
    except (TypeError, ValueError):
        return None
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 9:
        return None
    if not isinstance(repeats, (list, tuple)) or len(repeats) != 3:
        return None
    return (
        [float(value) for value in matrix],
        [int(value) for value in repeats],
    )


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
    try:
        repeats = tuple(
            max(1, min(_MAX_FILL_REPEATS, int(value)))
            for value in json.loads(root.get("cgn_cell_pack_repeats", "[1, 1, 1]"))
        )
    except (TypeError, ValueError):
        repeats = (1, 1, 1)
    if len(repeats) != 3:
        repeats = (1, 1, 1)
    obj = next((o for o in root.objects if o.get("cgn_role") == "unit_cell"), None)
    # Legacy objects from older builds: remove them and rebuild as one.
    legacy = [
        o
        for o in root.objects
        if o.get("cgn_role") in {"unit_cell_edge", "unit_cell_face", "unit_cell_marker"}
    ]
    if legacy:
        _clear_unit_cell(root)
        obj = None
    signature = (
        [float(value) for value in cell_matrix],
        [int(value) for value in repeats],
    )
    if obj is None or _unit_cell_signature(obj) != signature:
        _show_unit_cell(root, cell_matrix, settings, repeats)
        return
    _update_unit_cell_objects(obj, cell_matrix, repeats, settings)


# --------------------------------------------------------------------------
# Render quality: sphere/bond resolution presets.
# --------------------------------------------------------------------------

_APPLYING_QUALITY = False


def _quality_preset_values(preset):
    if preset == "LOW":
        return (2, 8)
    if preset == "HIGH":
        return (4, 16)
    return (3, 12)


def _apply_quality_settings(scene):
    global _APPLYING_QUALITY
    if scene is None:
        return
    preset = getattr(scene, "cgn_quality_preset", "MEDIUM")
    if preset in ("LOW", "MEDIUM", "HIGH"):
        atom_subdiv, bond_res = _quality_preset_values(preset)
    else:
        atom_subdiv = int(getattr(scene, "cgn_atom_subdivisions", 3))
        bond_res = int(getattr(scene, "cgn_bond_resolution", 12))
    for name, bl_id, attr, val in (
        (_NODE_GROUP_ATOMS, "GeometryNodeMeshIcoSphere", "Subdivisions", atom_subdiv),
        (_NODE_GROUP_BONDS, "GeometryNodeCurvePrimitiveCircle", "Resolution", bond_res),
    ):
        node_group = bpy.data.node_groups.get(name)
        if node_group is None:
            continue
        for node in node_group.nodes:
            if node.bl_idname == bl_id:
                node.inputs[attr].default_value = val
    unit_cell_group = bpy.data.node_groups.get("CGN_GN_UnitCell")
    if unit_cell_group is not None:
        for node in unit_cell_group.nodes:
            if node.bl_idname == "GeometryNodeMeshIcoSphere":
                node.inputs["Subdivisions"].default_value = atom_subdiv


def _update_quality_preset(self, context):
    global _APPLYING_QUALITY
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    preset = str(getattr(self, "cgn_quality_preset", "MEDIUM"))
    if preset in ("LOW", "MEDIUM", "HIGH"):
        atom_subdiv, bond_res = _quality_preset_values(preset)
        _APPLYING_QUALITY = True
        try:
            scene.cgn_atom_subdivisions = atom_subdiv
            scene.cgn_bond_resolution = bond_res
        finally:
            _APPLYING_QUALITY = False
    _apply_quality_settings(scene)


def _update_quality_manual(self, context):
    global _APPLYING_QUALITY
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    if _APPLYING_QUALITY:
        return
    if getattr(scene, "cgn_quality_preset", "MEDIUM") in ("LOW", "MEDIUM", "HIGH"):
        scene.cgn_quality_preset = "CUSTOM"
    _apply_quality_settings(scene)


# --------------------------------------------------------------------------
# Coordination polyhedra: convex hulls around connected ligand atoms.
# Faces and edges reuse the unit-cell rendering (transparent face mesh plus
# beveled edge curves) so both stay visually consistent.
# --------------------------------------------------------------------------

_POLY_ROLE_ROLES = ("polyhedron",)
_POLY_DELETE_ROLES = ("polyhedron", "polyhedron_prototype")


def _enabled_pair_windows(pair_settings):
    """Map sorted element-pair keys to enabled (min, max) distance windows."""
    windows = {}
    for setting in pair_settings:
        if setting.enabled and setting.max_distance >= setting.min_distance:
            windows[_pair_key(setting.element_a, setting.element_b)] = (
                float(setting.min_distance),
                float(setting.max_distance),
            )
    return windows


def _p1_pool(root):
    """P1 site list as atom dicts (cart in Angstrom), for coordination search."""
    try:
        sites = json.loads(root.get("cgn_sites", "[]"))
    except (TypeError, ValueError):
        return [], []
    cell_matrix = _stored_cell_matrix(root)
    if not cell_matrix:
        return [], []
    pool = []
    for site in sites:
        try:
            frac = [float(value) for value in site["frac"]]
        except (TypeError, ValueError, KeyError):
            continue
        cart = [
            frac[0] * cell_matrix[axis]
            + frac[1] * cell_matrix[3 + axis]
            + frac[2] * cell_matrix[6 + axis]
            for axis in range(3)
        ]
        pool.append(
            {
                "element": str(site.get("element", "X")).upper(),
                "frac": frac,
                "cart": cart,
                "covalent_radius": float(site.get("covalent_radius", 0.0))
                or _DEFAULT_VDW_RADIUS * 0.5,
                "index": len(pool),
            }
        )
    return pool, cell_matrix


def _gemmi_cell(cell_matrix):
    """gemmi.UnitCell built from the column-major orthogonalized cell matrix.

    The matrix holds the cell vectors as COLUMNS (that is how ``_p1_pool`` and
    the worker read it), so a is ``(m0, m1, m2)``.  Reading it row-wise instead
    gives a sheared cell for every triclinic structure (Ag2Cr2O7:
    a/b/c = 7.004/7.520/6.072 instead of 6.968/7.148/6.544), which produced
    wrong coordination spheres there.
    """
    worker = _gemmi_worker()
    a_vec = (cell_matrix[0], cell_matrix[1], cell_matrix[2])
    b_vec = (cell_matrix[3], cell_matrix[4], cell_matrix[5])
    c_vec = (cell_matrix[6], cell_matrix[7], cell_matrix[8])

    def _vec_len(v):
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    def _vec_angle(u, v):
        lu = _vec_len(u)
        lv = _vec_len(v)
        if lu <= 1e-12 or lv <= 1e-12:
            return 90.0
        dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (lu * lv)))))

    return worker.gemmi.UnitCell(
        _vec_len(a_vec),
        _vec_len(b_vec),
        _vec_len(c_vec),
        _vec_angle(b_vec, c_vec),
        _vec_angle(a_vec, c_vec),
        _vec_angle(a_vec, b_vec),
    )


class _PolyhedronSearch:
    """Coordination search for one structure revision, shared by every pass.

    The eligibility scan (``_cached_polyhedron_elements``) and the build pass
    (``_build_polyhedra_objects``) ask for the same ligand lists, so one gemmi
    ``NeighborSearch`` is built per revision, ligand lists are memoised per
    (element, centre position), and only the atoms gemmi reports near a centre
    are asked for exact periodic images.  Before this, every centre rebuilt its
    own ``gemmi.UnitCell`` and rescanned every P1 site in Python.
    """

    def __init__(self, pool, cell_matrix):
        self.pool = pool
        self.cell_matrix = cell_matrix
        self.windows = {}
        self.windows_key = ()
        self.radius = 0.0
        self._reach = {}
        self.cell = None
        self._index = None
        self._ligands = {}
        self._eligible = None
        if pool and cell_matrix:
            try:
                self.cell = _gemmi_cell(cell_matrix)
            except Exception:
                self.cell = None

    def update_windows(self, windows):
        """Adopt the enabled windows; drops the memo only if they changed."""
        key = tuple(sorted(windows.items()))
        if key == self.windows_key:
            return
        self.windows_key = key
        self.windows = dict(windows)
        # Widest window involving each element: a ligand of a centre can never
        # be further away, so the spatial index query can stop there.
        reach = {}
        for (element_a, element_b), (_minimum, maximum) in windows.items():
            reach[element_a] = max(reach.get(element_a, 0.0), maximum)
            reach[element_b] = max(reach.get(element_b, 0.0), maximum)
        self._reach = reach
        self.radius = max(reach.values(), default=0.0)
        self._index = None
        self._ligands.clear()
        self._eligible = None

    def _neighbor_index(self):
        """gemmi ``NeighborSearch`` over the P1 pool, built once per revision.

        Returns ``(search, structure, query)`` or ``None`` when no index can be
        built (no cell, no window, or a gemmi error), in which case the caller
        scans the whole pool.  The structure must outlive the search, and the
        query site is reused for every centre.
        """
        if self._index is not None:
            return self._index or None
        if self.cell is None or self.radius <= 0.0:
            self._index = ()
            return None
        worker = _gemmi_worker()
        gemmi = worker.gemmi
        structure = gemmi.SmallStructure()
        structure.cell = self.cell
        sites = []
        for atom in self.pool:
            site = gemmi.SmallStructure.Site()
            frac = atom["frac"]
            site.fract = gemmi.Fractional(frac[0], frac[1], frac[2])
            site.label = str(atom["index"])
            sites.append(site)
        structure.sites = sites
        try:
            search = gemmi.NeighborSearch(structure, self.radius + 1e-4)
            for position, site in enumerate(structure.sites):
                search.add_site(site, position)
        except Exception:
            self._index = ()
            return None
        self._index = (search, structure, gemmi.SmallStructure.Site())
        return self._index

    def eligible_elements(self):
        """Elements with at least one site carrying enough ligands."""
        if self._eligible is None:
            eligible = set()
            elements = {site["element"] for site in self.pool}
            for site in self.pool:
                element = site["element"]
                if element in eligible:
                    continue
                ligands = self.ligands(
                    site["cart"], element, limit=_POLY_MIN_LIGANDS
                )
                if len(ligands) >= _POLY_MIN_LIGANDS:
                    eligible.add(element)
                    if len(eligible) == len(elements):
                        break
            self._eligible = sorted(eligible)
        return self._eligible

    def ligands(self, center_cart, center_element, limit=None):
        """Ligand display positions for one centre, including periodic images.

        ``limit`` stops the scan once that many ligands are found, for callers
        that only ask "is this centre eligible?"; such a partial answer is not
        memoised, so a later full request recomputes it.
        """
        if self.cell is None or not self.windows:
            return ()
        key = (
            center_element,
            round(float(center_cart[0]), 6),
            round(float(center_cart[1]), 6),
            round(float(center_cart[2]), 6),
        )
        cached = self._ligands.get(key)
        if cached is not None:
            return cached
        ligands = self._search(center_cart, center_element, limit)
        if limit is None or len(ligands) < limit:
            self._ligands[key] = ligands
        return ligands

    def _search(self, center_cart, center_element, limit):
        reach = self._reach.get(center_element)
        if reach is None:
            return ()
        worker = _gemmi_worker()
        cell = self.cell
        center_frac = cell.fractionalize(worker.gemmi.Position(*center_cart))
        index = self._neighbor_index()
        if index is None:
            candidates = range(len(self.pool))
        else:
            search, _structure, query = index
            query.fract = center_frac
            candidates = sorted(
                {
                    mark.atom_idx
                    for mark in search.find_site_neighbors(query, 0.0, reach)
                }
            )
        ligands = []
        seen = set()
        for position in candidates:
            atom = self.pool[position]
            bounds = self.windows.get(
                _pair_key(center_element, atom["element"])
            )
            if bounds is None:
                continue
            minimum, maximum = bounds
            if maximum <= 0.0:
                continue
            pj = atom["frac"]
            site_index = atom["index"]
            pj_gemmi = worker.gemmi.Fractional(pj[0], pj[1], pj[2])
            for im in cell.find_nearest_pbc_images(
                center_frac, maximum, pj_gemmi, 0
            ):
                f_img = cell.fract_image(im, pj_gemmi)
                tx = int(round(f_img.x - pj[0]))
                ty = int(round(f_img.y - pj[1]))
                tz = int(round(f_img.z - pj[2]))
                c_img = cell.orthogonalize(f_img)
                delta = (
                    center_cart[0] - c_img.x,
                    center_cart[1] - c_img.y,
                    center_cart[2] - c_img.z,
                )
                d2 = delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2
                if d2 <= 1e-12:
                    continue
                distance = math.sqrt(d2)
                if distance < minimum or distance > maximum:
                    continue
                pos = _display_point(c_img)
                key = tuple(round(v, 6) for v in pos)
                if key in seen:
                    continue
                seen.add(key)
                ligands.append((pos, site_index, (tx, ty, tz)))
                if limit is not None and len(ligands) >= limit:
                    break
            if limit is not None and len(ligands) >= limit:
                break
        return tuple(ligands)


_POLY_SEARCH_CACHE = None


def _polyhedron_search(root, scene):
    """The shared ligand search for this structure revision and scene."""
    global _POLY_SEARCH_CACHE
    if root is None or not root.get("cgn_sites"):
        return None
    key = _structure_cache_key(root)
    cached = _POLY_SEARCH_CACHE
    if cached is not None and cached[0] == key:
        search = cached[1]
    else:
        pool, cell_matrix = _p1_pool(root)
        if not pool:
            return None
        search = _PolyhedronSearch(pool, cell_matrix)
        _POLY_SEARCH_CACHE = (key, search)
    search.update_windows(_enabled_pair_windows(scene.cgn_connectivity_pairs))
    return search


def _polyhedron_style_values(style):
    """Return (face_rgba, edge_rgba, thickness) from a per-element style."""
    face_color = tuple(float(channel) for channel in style.face_color)
    face_alpha = max(0.0, min(1.0, 1.0 - float(style.face_transparency)))
    face_rgba = (*face_color[:3], face_alpha)
    edge_color = tuple(float(channel) for channel in style.edge_color)
    edge_alpha = max(0.0, min(1.0, 1.0 - float(style.edge_transparency)))
    edge_rgba = (*edge_color[:3], edge_alpha)
    thickness = max(0.0, min(0.25, float(style.edge_width)))
    return face_rgba, edge_rgba, thickness


def _polyhedron_style(scene, element, site_label=""):
    """Look up or create the style for an element (and optionally a site).

    Defaults to the atom colour.  When ``site_label`` is non-empty the style is
    per crystallographic site (e.g. ``AG``/``Ag1`` vs ``AG``/``Ag2``) so each
    site's polyhedron keeps its own editable style.
    """
    site_label = str(site_label or "")
    for style in scene.cgn_polyhedron_styles:
        if style.element == element and str(style.site_label or "") == site_label:
            return style
    color = _ELEMENT_COLORS.get(element, _DEFAULT_COLOR)
    style = scene.cgn_polyhedron_styles.add()
    style.element = element
    style.site_label = site_label
    style.face_color = color
    style.face_transparency = 1.0 - _POLY_FACE_ALPHA
    style.edge_color = _POLY_EDGE_COLOR
    style.edge_transparency = 0.0
    style.edge_width = _POLY_EDGE_THICKNESS
    return style


def _existing_polyhedron_style(scene, element, site_label=""):
    site_label = str(site_label or "")
    for style in scene.cgn_polyhedron_styles:
        if style.element == element and str(style.site_label or "") == site_label:
            return style
    return None


def _make_polyhedron_face_material(element, style, site_label=""):
    suffix = f"_{site_label}" if site_label else ""
    material = bpy.data.materials.get(f"CGN_Polyhedron_Face_Material_{element}{suffix}")
    if material is None:
        material = bpy.data.materials.new(
            f"CGN_Polyhedron_Face_Material_{element}{suffix}"
        )
    material.use_nodes = True
    face_rgba, _edge, _thickness = _polyhedron_style_values(style)
    color, alpha = face_rgba[:3], face_rgba[3]
    material.diffuse_color = (*color, alpha)
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = (*color, 1.0)
        shader.inputs["Roughness"].default_value = 0.4
        if "Alpha" in shader.inputs:
            shader.inputs["Alpha"].default_value = alpha
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    elif hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = True
    material.use_backface_culling = False
    return material


def _make_polyhedron_edge_material(element, style, site_label=""):
    suffix = f"_{site_label}" if site_label else ""
    material = bpy.data.materials.get(f"CGN_Polyhedron_Edge_Material_{element}{suffix}")
    if material is None:
        material = bpy.data.materials.new(
            f"CGN_Polyhedron_Edge_Material_{element}{suffix}"
        )
    material.use_nodes = True
    _face, edge_rgba, _thickness = _polyhedron_style_values(style)
    color, alpha = edge_rgba[:3], edge_rgba[3]
    material.diffuse_color = (*color, alpha)
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = (*color, 1.0)
        shader.inputs["Roughness"].default_value = 0.5
        if "Alpha" in shader.inputs:
            shader.inputs["Alpha"].default_value = alpha
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    elif hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    material.use_backface_culling = False
    return material


def _cached_polyhedron_elements(root, scene):
    """Eligible center elements based on P1 sites + periodic images.

    Backed by the structure's shared ligand search, so a panel redraw asks the
    memoised ligand lists rather than rescanning every site.
    """
    search = _polyhedron_search(root, scene)
    if search is None:
        return []
    return search.eligible_elements()


def _clear_polyhedra(root, element=None):
    """Remove polyhedron scaffolds and their prototypes (all, or one element)."""
    for obj in list(root.all_objects):
        if obj.get("cgn_role") not in _POLY_DELETE_ROLES:
            continue
        if element is not None and str(obj.get("cgn_poly_element", "")) != element:
            continue
        _remove_object(obj)
    # Drop the helper collection once its last prototype is gone; leaving empty
    # child collections behind clutters the outliner on every clear/rebuild.
    for child in list(root.children):
        if child.get("cgn_role") == "polyhedron_prototypes" and not child.all_objects:
            bpy.data.collections.remove(child)


def _polyhedron_geometry(atoms, centers, search, removed_quads=None):
    """Face/edge geometry for *centers*.

    ``search`` is the structure's shared ``_PolyhedronSearch``; the ligand
    lists it already memoised are reused.  Returns
    ``(vertices, faces, edges, built_centers)`` or ``None`` when no centre
    still has enough ligands to enclose a polyhedron.

    ``removed_quads`` lists the ``(site, tx, ty, tz)`` images the user just
    erased.  If any of them is a ligand of *center*, the whole polyhedron is
    dropped instead of being rebuilt around the remaining neighbours.
    """
    vertices = []
    faces = []
    edges = []
    built = []
    for center in centers:
        if not 0 <= center < len(atoms):
            continue
        ligands = search.ligands(
            atoms[center]["cart"],
            str(atoms[center].get("element", "X")).upper(),
        )
        if removed_quads and any(
            (site_index, tx, ty, tz) in removed_quads
            for _pos, site_index, (tx, ty, tz) in ligands
        ):
            # A ligand was erased: remove this polyhedron entirely rather
            # than re-enclosing the surviving neighbours.
            continue
        if len(ligands) < _POLY_MIN_LIGANDS:
            continue
        topology = _polyhedron_topology([item[0] for item in ligands])
        if topology is None:
            continue
        local_vertices, _kept, local_faces, local_edges = topology
        base = len(vertices)
        vertices.extend(local_vertices)
        faces.extend(tuple(base + index for index in loop) for loop in local_faces)
        edges.extend((base + first, base + second) for first, second in local_edges)
        built.append(center)
    if not vertices or not faces:
        return None
    return vertices, faces, edges, built


def _remap_polyhedra(root, removed_indices, atoms=None, scene=None, removed_quads=None):
    """Rebuild polyhedra after atoms were deleted.

    Polyhedra are baked per crystallographic site into one mesh object, so
    incremental remapping of the baked geometry is not feasible.  Instead all
    existing polyhedra are cleared and rebuilt for every element that
    previously had them, keeping the displayed coordination consistent with
    the remaining atoms.
    """
    removed = sorted({int(index) for index in removed_indices})
    if not removed:
        return
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    if atoms is None:
        try:
            atoms = _structure_atoms(root)
        except (ValueError, TypeError, KeyError):
            return
    if not atoms:
        return

    # Elements that had polyhedra before the deletion.
    elements_with_poly = {
        str(obj.get("cgn_poly_element", "")).upper()
        for obj in root.objects
        if obj.get("cgn_role") == "polyhedron"
    }
    _clear_polyhedra(root)

    for element in elements_with_poly:
        centers = [
            i
            for i, a in enumerate(atoms)
            if str(a.get("element", "X")).upper() == element
        ]
        if centers:
            _build_polyhedra_objects(
                root,
                atoms,
                centers,
                scene,
                clear_existing=False,
                extend_structure=False,
            )


def _vec_sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def _vec_add(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def _vec_scale(u, factor):
    return (u[0] * factor, u[1] * factor, u[2] * factor)


def _vec_cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _vec_dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _vec_distance_squared(u, v):
    d = _vec_sub(u, v)
    return _vec_dot(d, d)


def _polygon_newell(loop, vertices):
    """Newell normal and plane offset for a polygon index loop."""
    normal = (0.0, 0.0, 0.0)
    count = len(loop)
    for position in range(count):
        u = vertices[loop[position]]
        v = vertices[loop[(position + 1) % count]]
        normal = _vec_add(
            normal,
            (
                (u[1] - v[1]) * (u[2] + v[2]),
                (u[2] - v[2]) * (u[0] + v[0]),
                (u[0] - v[0]) * (u[1] + v[1]),
            ),
        )
    length = math.sqrt(_vec_dot(normal, normal))
    if length <= 1e-12:
        return (0.0, 0.0, 0.0), 0.0
    unit = _vec_scale(normal, 1.0 / length)
    offset = _vec_dot(unit, vertices[loop[0]])
    return unit, offset


def _convex_hull_faces(vertices):
    """Convex hull faces of deduplicated display-unit positions.

    Returns index triples wound outward. The hull grows from an initial
    tetrahedron; every inserted point flips the visible faces over its
    horizon edges. All faces are re-oriented against the current interior
    centroid after each insertion, so winding never depends on input order.
    """
    n = len(vertices)
    if n < 4:
        return []
    index0 = 0
    index1 = max(
        range(n), key=lambda i: _vec_distance_squared(vertices[i], vertices[index0])
    )
    if _vec_distance_squared(vertices[index1], vertices[index0]) <= 1e-12:
        return []
    axis = _vec_sub(vertices[index1], vertices[index0])
    index2 = max(
        range(n),
        key=lambda i: _vec_dot(
            _vec_cross(_vec_sub(vertices[i], vertices[index0]), axis),
            _vec_cross(_vec_sub(vertices[i], vertices[index0]), axis),
        ),
    )
    if index2 == index0 or index2 == index1:
        if _vec_distance_squared(vertices[index2], vertices[index0]) <= 1e-12:
            return []
    plane_normal = _vec_cross(
        _vec_sub(vertices[index1], vertices[index0]),
        _vec_sub(vertices[index2], vertices[index0]),
    )
    plane_length = math.sqrt(_vec_dot(plane_normal, plane_normal))
    if plane_length <= 1e-12:
        return []
    index3 = max(
        range(n),
        key=lambda i: abs(
            _vec_dot(_vec_sub(vertices[i], vertices[index0]), plane_normal)
        ),
    )
    if (
        abs(_vec_dot(_vec_sub(vertices[index3], vertices[index0]), plane_normal))
        <= 1e-9
    ):
        return []  # all positions coplanar; no volume to enclose

    seeds = (index0, index1, index2, index3)
    faces = [
        (index0, index1, index2),
        (index0, index1, index3),
        (index0, index2, index3),
        (index1, index2, index3),
    ]
    tetra_centroid = _vec_scale(
        _vec_add(
            _vec_add(vertices[index0], vertices[index1]),
            _vec_add(vertices[index2], vertices[index3]),
        ),
        0.25,
    )
    faces = [
        (
            face
            if _vec_dot(
                _vec_cross(
                    _vec_sub(vertices[face[1]], vertices[face[0]]),
                    _vec_sub(vertices[face[2]], vertices[face[0]]),
                ),
                _vec_sub(
                    _vec_scale(
                        _vec_add(
                            _vec_add(vertices[face[0]], vertices[face[1]]),
                            vertices[face[2]],
                        ),
                        1.0 / 3.0,
                    ),
                    tetra_centroid,
                ),
            )
            > 0.0
            else (face[0], face[2], face[1])
        )
        for face in faces
    ]
    eps = 1e-6
    processed = list(seeds)
    for point_index in range(n):
        if point_index in seeds:
            continue
        point = vertices[point_index]
        visible = []
        for face in faces:
            a, b, c = (vertices[i] for i in face)
            if (
                _vec_dot(
                    _vec_cross(_vec_sub(b, a), _vec_sub(c, a)),
                    _vec_sub(point, a),
                )
                > eps
            ):
                visible.append(face)
        if not visible:
            processed.append(point_index)
            continue
        edge_counts = {}
        for face in visible:
            for position in range(3):
                edge = tuple(sorted((face[position], face[(position + 1) % 3])))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        horizon = [edge for edge, count in edge_counts.items() if count == 1]
        if not horizon:  # defensive; cannot happen for a proper horizon
            continue
        visible_set = set(visible)
        faces = [face for face in faces if face not in visible_set] + [
            (edge[0], edge[1], point_index) for edge in horizon
        ]
        # Pivot strictly inside the current hull: the centroid of every point
        # processed so far (each lies in, or on, the current hull).
        pivot = _vec_scale(
            (
                sum(vertices[i][0] for i in processed) / len(processed),
                sum(vertices[i][1] for i in processed) / len(processed),
                sum(vertices[i][2] for i in processed) / len(processed),
            ),
            1.0,
        )
        oriented = []
        for face in faces:
            a, b, c = (vertices[i] for i in face)
            face_center = _vec_scale(_vec_add(_vec_add(a, b), c), 1.0 / 3.0)
            if (
                _vec_dot(
                    _vec_cross(_vec_sub(b, a), _vec_sub(c, a)),
                    _vec_sub(face_center, pivot),
                )
                > 0.0
            ):
                oriented.append(face)
            else:
                oriented.append((face[0], face[2], face[1]))
        faces = oriented
        processed.append(point_index)
    return faces


def _merge_coplanar_faces(faces, vertices):
    """Merge coplanar triangles into n-gons so square faces lose diagonals.

    Returns (polygon_loops, undirected_edge_set) over the input vertex list.
    """
    groups = {}
    for face in faces:
        unit, offset = _polygon_newell(face, vertices)
        key = (
            round(unit[0], 4),
            round(unit[1], 4),
            round(unit[2], 4),
            round(offset, 4),
        )
        groups.setdefault(key, []).append(face)
    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        # Coplanar triangles are wound consistently, so shared interior edges
        # appear in both directions; boundary edges chain into polygon loops.
        directed = []
        for face in group:
            for position in range(3):
                directed.append((face[position], face[(position + 1) % 3]))
        directed_set = set(directed)
        boundary = [
            (start, end)
            for (start, end) in directed
            if (end, start) not in directed_set
        ]
        next_vertex = {}
        for start, end in boundary:
            existing = next_vertex.get(start)
            # two boundary edges may share a start vertex in non-simple
            # groups; keep pairing unambiguous by falling back
            if existing is not None and existing != end:
                next_vertex = None
                break
            next_vertex[start] = end
        if next_vertex is None:
            merged.extend(group)
            continue
        remaining = set(boundary)
        loops = []
        while remaining:
            start = next(iter(remaining))[0]
            loop = [start]
            while True:
                follower = next_vertex.get(loop[-1])
                if follower is None or (loop[-1], follower) not in remaining:
                    loop = None
                    break
                remaining.discard((loop[-1], follower))
                if follower == start:
                    break
                loop.append(follower)
            if loop is None or len(loop) < 3:
                loops = None
                break
            loops.append(loop)
        if loops is None:
            merged.extend(group)
            continue
        merged.extend(loops)
    return merged, {
        tuple(sorted((loop[i], loop[(i + 1) % len(loop)])))
        for loop in merged
        for i in range(len(loop))
    }


def _polyhedron_topology(points):
    """Return (vertices, faces, edges) for coordination polyhedron faces.

    ``points`` are display-unit positions. Coplanar hull triangles are merged
    into n-gons; the returned edges are the true polyhedron creases (no
    diagonals inside planar faces). Returns None when the positions do not
    enclose a volume.
    """
    deduplicated = []
    kept = []
    for index, point in enumerate(points):
        if any(
            _vec_distance_squared(point, previous) <= 1e-9 for previous in deduplicated
        ):
            continue
        deduplicated.append(point)
        kept.append(index)
    faces = _convex_hull_faces(deduplicated)
    if not faces:
        return None
    faces, edges = _merge_coplanar_faces(faces, deduplicated)
    if not faces or not edges:
        return None
    pivot = _vec_scale(
        (
            sum(point[0] for point in deduplicated) / len(deduplicated),
            sum(point[1] for point in deduplicated) / len(deduplicated),
            sum(point[2] for point in deduplicated) / len(deduplicated),
        ),
        1.0,
    )
    oriented = []
    for loop in faces:
        unit, _offset = _polygon_newell(loop, deduplicated)
        center = _vec_scale(
            (
                sum(deduplicated[i][0] for i in loop) / len(loop),
                sum(deduplicated[i][1] for i in loop) / len(loop),
                sum(deduplicated[i][2] for i in loop) / len(loop),
            ),
            1.0,
        )
        if _vec_dot(unit, _vec_sub(center, pivot)) > 0.0:
            oriented.append(loop)
        else:
            oriented.append(tuple(reversed(loop)))
    edges = [
        (first, second)
        for first, second in edges
        if first < len(deduplicated) and second < len(deduplicated)
    ]
    # Drop vertices that never appear in a face or edge (interior points).
    used = sorted(
        {index for loop in oriented for index in loop}
        | {index for first, second in edges for index in (first, second)}
    )
    if not used:
        return None
    index_map = {old: new for new, old in enumerate(used)}
    compact_vertices = [deduplicated[index] for index in used]
    compact_kept = [kept[index] for index in used]
    compact_faces = [tuple(index_map[index] for index in loop) for loop in oriented]
    compact_edges = [(index_map[first], index_map[second]) for first, second in edges]
    return (compact_vertices, compact_kept, compact_faces, compact_edges)


def _polyhedron_edge_tube_geometry(vertices, edges, radius, resolution=8):
    """Return ``(vertices, faces)`` for thin cylinders along ``edges``.

    Polyhedron edges used to be a separate bevelled curve object; baking them
    into the site's face mesh keeps one editable object per site.  A zero
    radius (edge width 0) returns empty lists so the edge material simply
    stays unused.
    """
    if radius <= 0.0 or resolution < 3:
        return [], []
    tube_vertices = []
    tube_faces = []
    for first, second in edges:
        start = vertices[first]
        end = vertices[second]
        direction = (
            end[0] - start[0],
            end[1] - start[1],
            end[2] - start[2],
        )
        length = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        if length <= 1e-12:
            continue
        axis = (
            direction[0] / length,
            direction[1] / length,
            direction[2] / length,
        )
        helper = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0)
        side_u = (
            axis[1] * helper[2] - axis[2] * helper[1],
            axis[2] * helper[0] - axis[0] * helper[2],
            axis[0] * helper[1] - axis[1] * helper[0],
        )
        side_length = math.sqrt(side_u[0] ** 2 + side_u[1] ** 2 + side_u[2] ** 2)
        if side_length <= 1e-12:
            continue
        side_u = (
            side_u[0] / side_length,
            side_u[1] / side_length,
            side_u[2] / side_length,
        )
        side_v = (
            axis[1] * side_u[2] - axis[2] * side_u[1],
            axis[2] * side_u[0] - axis[0] * side_u[2],
            axis[0] * side_u[1] - axis[1] * side_u[0],
        )
        base = len(tube_vertices)
        for point in (start, end):
            for step in range(resolution):
                angle = 2.0 * math.pi * step / resolution
                cosine = math.cos(angle)
                sine = math.sin(angle)
                tube_vertices.append(
                    (
                        point[0] + radius * (cosine * side_u[0] + sine * side_v[0]),
                        point[1] + radius * (cosine * side_u[1] + sine * side_v[1]),
                        point[2] + radius * (cosine * side_u[2] + sine * side_v[2]),
                    )
                )
        for step in range(resolution):
            following = (step + 1) % resolution
            tube_faces.append(
                (
                    base + step,
                    base + following,
                    base + resolution + following,
                    base + resolution + step,
                )
            )
    return tube_vertices, tube_faces


def _polyhedron_prototype_collection(root):
    """Hidden child collection holding the per-site polyhedron prototypes."""
    for child in root.children:
        if child.get("cgn_role") == "polyhedron_prototypes":
            return child
    collection = bpy.data.collections.new(_POLY_PROTOTYPE_COLLECTION)
    collection["cgn_role"] = "polyhedron_prototypes"
    collection.hide_viewport = True
    collection.hide_render = True
    collection.hide_select = True
    root.children.link(collection)
    return collection


def _make_polyhedron_prototype_mesh(
    name,
    local_vertices,
    local_faces,
    local_edges,
    thickness,
    face_material,
    edge_material,
):
    """Build one canonical polyhedron (faces + edge tubes) at the origin.

    The mesh is stored once per crystallographic site and instanced on every
    image by the ``CGN_GN_Polyhedra`` node group.  Material slot 0 is the face
    material, slot 1 the edge-tube material, matching the old baked layout.
    """
    tube_vertices, tube_faces = _polyhedron_edge_tube_geometry(
        local_vertices, local_edges, thickness * _DISPLAY_SCALE
    )
    vertices = [tuple(vertex) for vertex in local_vertices] + list(tube_vertices)
    base = len(local_vertices)
    faces = [tuple(loop) for loop in local_faces] + [
        tuple(base + index for index in loop) for loop in tube_faces
    ]
    material_indices = [0] * len(local_faces) + [1] * len(tube_faces)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for polygon, material_index in zip(mesh.polygons, material_indices):
        polygon.use_smooth = False
        polygon.material_index = material_index
    mesh.materials.append(face_material)
    mesh.materials.append(edge_material)
    return mesh


def _polyhedron_prototype_for(obj):
    """Return the prototype object of a polyhedron scaffold, if it still exists."""
    name = str(obj.get("cgn_poly_prototype", "") or "")
    if not name:
        return None
    prototype = bpy.data.objects.get(name)
    if prototype is None or prototype.get("cgn_role") != "polyhedron_prototype":
        return None
    return prototype


def _rebuild_polyhedron_prototype(prototype, thickness):
    """Recompute the edge-tube geometry of one prototype at a new width.

    Only the prototype changes: every image of the site is an instance of it,
    so the scaffold points and their transforms are untouched.
    """
    try:
        local_vertices = json.loads(prototype.get("cgn_poly_local_vertices", "[]"))
        local_faces = json.loads(prototype.get("cgn_poly_local_faces", "[]"))
        local_edges = json.loads(prototype.get("cgn_poly_local_edges", "[]"))
    except (TypeError, ValueError):
        return False
    if not local_vertices or not local_faces:
        return False
    tube_vertices, tube_faces = _polyhedron_edge_tube_geometry(
        local_vertices, local_edges, thickness * _DISPLAY_SCALE
    )
    vertices = [tuple(vertex) for vertex in local_vertices] + list(tube_vertices)
    base = len(local_vertices)
    faces = [tuple(loop) for loop in local_faces] + [
        tuple(base + index for index in loop) for loop in tube_faces
    ]
    material_indices = [0] * len(local_faces) + [1] * len(tube_faces)
    mesh = prototype.data
    mesh.clear_geometry()
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for polygon, material_index in zip(mesh.polygons, material_indices):
        polygon.use_smooth = False
        polygon.material_index = material_index
    prototype["cgn_poly_edge_width"] = float(thickness)
    return True


def _append_polyhedron_points(obj, points):
    """Append image centres to a polyhedron scaffold (points-only mesh).

    Appending vertices alone is safe; the Blender 5.2 ``Mesh.edges.add`` crash
    only affects meshes that already hold edges, and this scaffold never has
    any.  The prototype geometry is shared, so a grow is O(new images).
    """
    if not points:
        return 0
    mesh = obj.data
    base = len(mesh.vertices)
    mesh.vertices.add(len(points))
    for offset, point in enumerate(points):
        mesh.vertices[base + offset].co = tuple(point)
    mesh.update()
    obj.update_tag(refresh={"OBJECT", "DATA"})
    return len(points)


def _pair_key(element_a, element_b):
    return tuple(sorted((str(element_a).upper(), str(element_b).upper())))


def _default_pair_window(element_a, element_b, radii, charges, metals, ionic=None):
    """Default (min, max) connectivity distance for one element pair.

    *radii* maps uppercase element symbols to covalent radii, *charges* holds the
    oxidation numbers the CIF declares, *metals* is the set of metallic symbols
    and *ionic* maps the charged symbols to their Shannon radius.

    Cation-anion pairs use the Shannon radii and the plain ``[0.6, 1.2]``
    multiples (Diamond's model): Ag(I)-O in Ag2Cr2O7 needs 2.55 A as the
    reference, and the covalent sum (2.11 A) misses half the coordination
    sphere.  Every other pair keeps the covalent radii, with the upper bound
    capped at ``r_a + r_b + 0.4`` so it never exceeds the old additive default,
    and a metal-metal pair whose two elements carry same-sign non-zero formal
    charges is capped at ``0.85 * (r_a + r_b)`` so cation-cation contacts in
    ionic frameworks are not drawn.
    """
    ionic = ionic or {}
    charge_a = float(charges.get(element_a, 0.0))
    charge_b = float(charges.get(element_b, 0.0))
    if (
        charge_a
        and charge_b
        and (charge_a > 0.0) != (charge_b > 0.0)
        and element_a in ionic
        and element_b in ionic
    ):
        reference = ionic[element_a] + ionic[element_b]
        return _BOND_MIN_SCALE * reference, min(
            _MAX_BOND_WINDOW, _BOND_MAX_SCALE * reference
        )
    reference = radii[element_a] + radii[element_b]
    maximum = min(_BOND_MAX_SCALE * reference, reference + _BOND_MAX_SLACK)
    if element_a in metals and element_b in metals:
        if charge_a and charge_b and (charge_a > 0.0) == (charge_b > 0.0):
            maximum = _BOND_METAL_CAP_SCALE * reference
    minimum = min(_BOND_MIN_SCALE * reference, maximum)
    return minimum, max(minimum, min(_MAX_BOND_WINDOW, maximum))


def _bond_point_key(point):
    return tuple(round(float(value), 4) for value in point)


def _bond_edge_key(first, second):
    return tuple(sorted((_bond_point_key(first), _bond_point_key(second))))


def _chemdraw_bond_orders(segments, covalent_radii):
    """Infer conservative multiple-bond candidates with valence limits."""
    degrees = {}
    elements = {}
    candidates = []
    seen_edges = set()
    for first, second, element_a, element_b in segments:
        first_key = _bond_point_key(first)
        second_key = _bond_point_key(second)
        edge_key = tuple(sorted((first_key, second_key)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        element_a = str(element_a).upper()
        element_b = str(element_b).upper()
        elements[first_key] = element_a
        elements[second_key] = element_b
        if element_a in _CHEMDRAW_MAX_VALENCE:
            degrees[first_key] = degrees.get(first_key, 0) + 1
        if element_b in _CHEMDRAW_MAX_VALENCE:
            degrees[second_key] = degrees.get(second_key, 0) + 1

        radius_a = float(covalent_radii.get(first_key, 0.0))
        radius_b = float(covalent_radii.get(second_key, 0.0))
        if (
            element_a not in _CHEMDRAW_MAX_VALENCE
            or element_b not in _CHEMDRAW_MAX_VALENCE
            or radius_a <= 0.0
            or radius_b <= 0.0
        ):
            continue
        distance = math.sqrt(
            sum((float(first[axis]) - float(second[axis])) ** 2 for axis in range(3))
        )
        normalized_length = distance / (radius_a + radius_b)
        order = (
            3 if normalized_length <= 0.83 else 2 if normalized_length <= 0.90 else 1
        )
        if order > 1:
            confidence = (
                0.83 - normalized_length if order == 3 else 0.90 - normalized_length
            )
            candidates.append((confidence, edge_key, order, element_a, element_b))

    remaining = {
        node: max(
            0,
            _CHEMDRAW_MAX_VALENCE.get(element, 0) - degrees.get(node, 0),
        )
        for node, element in elements.items()
    }
    orders = {}
    for _confidence, edge_key, order, _element_a, _element_b in sorted(
        candidates, reverse=True
    ):
        extra_valence = min(
            order - 1,
            remaining.get(edge_key[0], 0),
            remaining.get(edge_key[1], 0),
        )
        if extra_valence:
            orders[edge_key] = extra_valence + 1
            remaining[edge_key[0]] -= extra_valence
            remaining[edge_key[1]] -= extra_valence
    return orders


def _classify_aromatic_rings(segments, covalent_radii, bond_orders):
    """Find planar conjugated rings with equalized bond lengths."""
    ring_elements = {"B", "C", "N", "O", "S"}
    nodes = {}
    elements = {}
    degrees = {}
    edges = []
    neighbors = {}
    all_neighbors = {}
    seen_edges = set()
    for first, second, element_a, element_b in segments:
        first_key = _bond_point_key(first)
        second_key = _bond_point_key(second)
        edge_key = tuple(sorted((first_key, second_key)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        element_a = str(element_a).upper()
        element_b = str(element_b).upper()
        nodes[first_key] = tuple(float(value) for value in first)
        nodes[second_key] = tuple(float(value) for value in second)
        elements[first_key] = element_a
        elements[second_key] = element_b
        degrees[first_key] = degrees.get(first_key, 0) + 1
        degrees[second_key] = degrees.get(second_key, 0) + 1
        all_neighbors.setdefault(first_key, []).append((second_key, element_b))
        all_neighbors.setdefault(second_key, []).append((first_key, element_a))
        distance = math.sqrt(
            sum((float(first[axis]) - float(second[axis])) ** 2 for axis in range(3))
        )
        radius_a = float(covalent_radii.get(first_key, 0.0))
        radius_b = float(covalent_radii.get(second_key, 0.0))
        normalized_length = (
            distance / (radius_a + radius_b) if radius_a + radius_b > 0.0 else 0.0
        )
        if (
            element_a not in ring_elements
            or element_b not in ring_elements
            or not 0.85 <= normalized_length <= 0.99
        ):
            continue
        edge_index = len(edges)
        edges.append((first_key, second_key, normalized_length))
        neighbors.setdefault(first_key, []).append((second_key, edge_index))
        neighbors.setdefault(second_key, []).append((first_key, edge_index))

    aromatic_edges = set()
    aromatic_rings = []
    seen_rings = set()
    for edge_index, (start, end, _distance) in enumerate(edges):
        if degrees[start] > 3 or degrees[end] > 3:
            continue
        path = [start]
        path_edges = []

        def find_ring(current):
            if len(path_edges) >= 5 or degrees[current] > 3:
                return
            for neighbor, next_edge in neighbors.get(current, ()):
                if next_edge == edge_index:
                    continue
                if neighbor != end and degrees[neighbor] > 3:
                    continue
                if neighbor == end and len(path_edges) in (3, 4):
                    ring_edges = tuple(sorted((*path_edges, next_edge, edge_index)))
                    if ring_edges in seen_rings:
                        continue
                    seen_rings.add(ring_edges)
                    ring_distances = [edges[index][2] for index in ring_edges]
                    ring_nodes = {
                        node for index in ring_edges for node in edges[index][:2]
                    }
                    ring_edge_keys = {
                        _bond_edge_key(edges[index][0], edges[index][1])
                        for index in ring_edges
                    }
                    if (
                        max(ring_distances) - min(ring_distances)
                        > _AROMATIC_BOND_LENGTH_SPREAD_MAX
                        or not 0.90 <= sum(ring_distances) / len(ring_distances) <= 0.96
                        or any(
                            degrees[node] > 3 or elements[node] not in ring_elements
                            for node in ring_nodes
                        )
                    ):
                        continue
                    pi_electrons = 0
                    lone_pair_nodes = set()
                    unsubstituted_nitrogens = []
                    ring_has_exocyclic_multiple = False
                    for node in ring_nodes:
                        element = elements[node]
                        for other, _other_element in all_neighbors.get(node, ()):
                            edge_key = tuple(sorted((node, other)))
                            if (
                                edge_key not in ring_edge_keys
                                and bond_orders.get(edge_key, 1) > 1
                            ):
                                ring_has_exocyclic_multiple = True
                                break
                        if ring_has_exocyclic_multiple:
                            break
                        if element == "C":
                            pi_electrons += 1
                        elif element == "N":
                            has_substituent = any(
                                other not in ring_nodes
                                and (
                                    other_element in _CHEMDRAW_MAX_VALENCE
                                    or other_element == "H"
                                )
                                for other, other_element in all_neighbors.get(node, ())
                            )
                            pi_electrons += 2 if has_substituent else 1
                            if has_substituent:
                                lone_pair_nodes.add(node)
                            else:
                                unsubstituted_nitrogens.append(node)
                        elif element in {"O", "S"}:
                            pi_electrons += 2
                    if ring_has_exocyclic_multiple:
                        continue
                    if pi_electrons < 2 or (pi_electrons - 2) % 4 != 0:
                        if (
                            pi_electrons >= 1
                            and (pi_electrons - 1) % 4 == 0
                            and unsubstituted_nitrogens
                        ):

                            def mean_nitrogen_bond_length(node):
                                lengths = [
                                    math.dist(
                                        nodes[node],
                                        nodes[
                                            edges[index][1]
                                            if edges[index][0] == node
                                            else edges[index][0]
                                        ],
                                    )
                                    for index in ring_edges
                                    if node in edges[index][:2]
                                ]
                                return sum(lengths) / len(lengths)

                            lone_pair_node = max(
                                unsubstituted_nitrogens,
                                key=lambda node: (
                                    mean_nitrogen_bond_length(node),
                                    node,
                                ),
                            )
                            pi_electrons += 1
                            lone_pair_nodes.add(lone_pair_node)
                        if pi_electrons < 2 or (pi_electrons - 2) % 4 != 0:
                            continue
                    pi_bond_nodes = {
                        node
                        for index in ring_edges
                        if edges[index][2] <= 0.96
                        for node in edges[index][:2]
                    }
                    if any(
                        elements[node] in {"B", "C"} and node not in pi_bond_nodes
                        for node in ring_nodes
                    ):
                        continue
                    coordinates = np.asarray(
                        [nodes[node] for node in ring_nodes], dtype=float
                    )
                    centroid_array = coordinates.mean(axis=0)
                    _left, singular_values, plane_axes = np.linalg.svd(
                        coordinates - centroid_array, full_matrices=False
                    )
                    if (
                        len(singular_values) < 3
                        or singular_values[-1] / math.sqrt(len(ring_nodes)) > 0.12
                    ):
                        continue
                    centroid = tuple(float(value) for value in centroid_array)
                    ring_elements_in = [elements[node] for node in ring_nodes]
                    ring_element = min(
                        set(ring_elements_in),
                        key=lambda element: (
                            -ring_elements_in.count(element),
                            element != "C",
                            element,
                        ),
                    )
                    inner_radius = float(
                        np.linalg.norm(coordinates - centroid_array, axis=1).mean()
                    )
                    aromatic_rings.append(
                        (
                            centroid,
                            tuple(float(value) for value in plane_axes[0]),
                            tuple(float(value) for value in plane_axes[1]),
                            inner_radius,
                            ring_element,
                        )
                    )
                    aromatic_edges.update(
                        _bond_edge_key(nodes[edges[index][0]], nodes[edges[index][1]])
                        for index in ring_edges
                    )
                    continue
                if neighbor in path or len(path_edges) >= 4:
                    continue
                path.append(neighbor)
                path_edges.append(next_edge)
                find_ring(neighbor)
                path_edges.pop()
                path.pop()

        find_ring(start)
    return aromatic_edges, aromatic_rings


def _aromatic_torus_segments(
    aromatic_rings,
    radius_scale=_AROMATIC_TORUS_SIZE_DEFAULT,
    sample_count=48,
):
    rings = []
    for center, axis_u, axis_v, radius, element in aromatic_rings:
        points = []
        for sample in range(sample_count):
            angle = 2.0 * math.pi * sample / sample_count
            points.append(
                tuple(
                    center[axis]
                    + radius
                    * radius_scale
                    * (math.cos(angle) * axis_u[axis] + math.sin(angle) * axis_v[axis])
                    for axis in range(3)
                )
            )
        rings.append(
            [
                (
                    first,
                    points[(index + 1) % sample_count],
                    element,
                    element,
                )
                for index, first in enumerate(points)
            ]
        )
    return rings


def _multiple_bond_plane_directions(segments, bond_orders):
    """Find each multiple-bond offset direction from its neighboring atoms."""
    points = {}
    neighbors = {}
    seen_edges = set()
    for first, second, _element_a, _element_b in segments:
        first_key = _bond_point_key(first)
        second_key = _bond_point_key(second)
        edge_key = tuple(sorted((first_key, second_key)))
        points[first_key] = tuple(float(value) for value in first)
        points[second_key] = tuple(float(value) for value in second)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        neighbors.setdefault(first_key, set()).add(second_key)
        neighbors.setdefault(second_key, set()).add(first_key)

    directions = {}
    for edge_key in bond_orders:
        first_key, second_key = edge_key
        first = points[first_key]
        second = points[second_key]
        delta = tuple(second[axis] - first[axis] for axis in range(3))
        length = math.sqrt(sum(value * value for value in delta))
        if length <= 1e-8:
            continue
        axis = tuple(value / length for value in delta)
        midpoint = tuple(0.5 * (first[index] + second[index]) for index in range(3))
        neighbor_vectors = []
        for node, opposite in ((first_key, second_key), (second_key, first_key)):
            for neighbor in neighbors.get(node, ()):
                if neighbor == opposite:
                    continue
                vector = tuple(
                    points[neighbor][index] - midpoint[index] for index in range(3)
                )
                projection = sum(vector[index] * axis[index] for index in range(3))
                vector = tuple(
                    vector[index] - projection * axis[index] for index in range(3)
                )
                if sum(value * value for value in vector) > 1e-12:
                    neighbor_vectors.append(vector)
        if neighbor_vectors:
            _left, _singular_values, right = np.linalg.svd(
                np.asarray(neighbor_vectors, dtype=float),
                full_matrices=False,
            )
            perpendicular = tuple(float(value) for value in right[0])
        else:
            reference_axis = min(range(3), key=lambda index: abs(axis[index]))
            reference = tuple(
                1.0 if index == reference_axis else 0.0 for index in range(3)
            )
            perpendicular = (
                axis[1] * reference[2] - axis[2] * reference[1],
                axis[2] * reference[0] - axis[0] * reference[2],
                axis[0] * reference[1] - axis[1] * reference[0],
            )
        perpendicular_length = math.sqrt(sum(value * value for value in perpendicular))
        if perpendicular_length > 1e-8:
            directions[edge_key] = tuple(
                value / perpendicular_length for value in perpendicular
            )
    return directions


def _chemdraw_segments(
    segments,
    bond_orders,
    aromatic_edges,
    multiple_bond_directions=None,
    bond_spacing=_CHEMDRAW_BOND_SPACING_DEFAULT,
):
    multiple_bond_directions = multiple_bond_directions or {}
    rendered = []
    for first, second, element_a, element_b in segments:
        delta = tuple(float(second[axis]) - float(first[axis]) for axis in range(3))
        length = math.sqrt(sum(value * value for value in delta))
        if length <= 1e-8:
            rendered.append((first, second, element_a, element_b, 1))
            continue
        direction = tuple(value / length for value in delta)
        edge_key = _bond_edge_key(first, second)
        order = bond_orders.get(edge_key, 1)
        if edge_key in aromatic_edges:
            rendered.append((first, second, element_a, element_b, 1))
            continue

        if order == 1:
            rendered.append((first, second, element_a, element_b, order))
            continue
        perpendicular = multiple_bond_directions.get(edge_key)
        if perpendicular is None:
            reference_axis = min(range(3), key=lambda axis: abs(direction[axis]))
            reference = tuple(
                1.0 if axis == reference_axis else 0.0 for axis in range(3)
            )
            perpendicular = (
                direction[1] * reference[2] - direction[2] * reference[1],
                direction[2] * reference[0] - direction[0] * reference[2],
                direction[0] * reference[1] - direction[1] * reference[0],
            )
            perpendicular_length = math.sqrt(
                sum(value * value for value in perpendicular)
            )
            perpendicular = tuple(
                value / perpendicular_length for value in perpendicular
            )
        spacing = float(bond_spacing)
        offsets = (
            (-spacing / 2.0, spacing / 2.0) if order == 2 else (-spacing, 0.0, spacing)
        )
        for offset in offsets:
            rendered.append(
                (
                    tuple(
                        float(first[axis]) + perpendicular[axis] * offset
                        for axis in range(3)
                    ),
                    tuple(
                        float(second[axis]) + perpendicular[axis] * offset
                        for axis in range(3)
                    ),
                    element_a,
                    element_b,
                    order,
                )
            )
    return rendered


def _connectivity_settings_signature(pair_settings):
    settings = [
        {
            "element_a": str(item.element_a).upper(),
            "element_b": str(item.element_b).upper(),
            "enabled": bool(item.enabled),
            "min_distance": float(item.min_distance),
            "max_distance": float(item.max_distance),
        }
        for item in pair_settings
    ]
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def _cached_atoms_for_growth(root, pair_settings, atom_count, new_atoms):
    cached = _BOND_SPATIAL_CACHE
    if (
        cached is None
        or cached["root_id"] != root.get("cgn_instance_id", root.name)
        or cached["settings_signature"]
        != _connectivity_settings_signature(pair_settings)
        or cached["atom_count"] != atom_count
    ):
        return None
    return cached["atoms"] + new_atoms


def _bond_group_objects(root, pair_name, order, element=None):
    """Return the bond objects that all represent one pair/order(/element) group.

    ``element`` selects the two-colour half-bond object of that element; when
    ``None`` the single-colour object (which has no ``cgn_bond_element``) is
    returned.  A healthy structure has exactly one match per group, but files
    grown with older builds can contain several (``CGN_Bonds_C_O.001``…); the
    caller merges them back into one.
    """
    matches = []
    for obj in root.objects:
        if obj.get("cgn_role") != "bonds" or obj.get("cgn_aromatic_torus"):
            continue
        if obj.get("cgn_pair") != pair_name:
            continue
        if int(obj.get("cgn_bond_order", 1)) != int(order):
            continue
        if element is None:
            if "cgn_bond_element" in obj:
                continue
        elif str(obj.get("cgn_bond_element", "")) != str(element):
            continue
        matches.append(obj)
    return matches


def _bond_object_segments(obj):
    """Read an existing bond mesh back as display-space ``(start, end)`` pairs."""
    mesh = obj.data
    coordinates = [vertex.co for vertex in mesh.vertices]
    return [
        (coordinates[edge.vertices[0]], coordinates[edge.vertices[1]])
        for edge in mesh.edges
    ]


def _bond_mesh_edge_keys(obj):
    """Return the display-space edge keys already present in a bond mesh."""
    mesh = obj.data
    coordinates = [vertex.co for vertex in mesh.vertices]
    return {
        _bond_edge_key(
            coordinates[edge.vertices[0]],
            coordinates[edge.vertices[1]],
        )
        for edge in mesh.edges
    }


def _append_bond_segments(obj, segments, dedupe=False):
    """Append display-space segments to an existing bond edge mesh.

    Blender 5.2 crashes when ``Mesh.edges.add()`` is called on a mesh that
    already holds edges (``ED_mesh_edges_add``), so in-place edge growth is not
    an option.  The new segments are added through a bmesh copy instead: that
    keeps the per-edge work in optimized C and is roughly 8x cheaper than
    rebuilding Python vertex/edge lists and calling ``from_pydata``.

    ``segments`` is deduped against itself.  ``dedupe`` additionally compares
    against the mesh's existing edges; it is only used when folding duplicate
    bond objects left behind by older builds, because the grow path never
    re-derives a bond it already drew.
    """
    if not segments:
        return 0
    mesh = obj.data
    seen = _bond_mesh_edge_keys(obj) if dedupe else set()
    new_segments = []
    for first, second in segments:
        key = _bond_edge_key(first, second)
        if key in seen:
            continue
        seen.add(key)
        new_segments.append((tuple(first), tuple(second)))
    if not new_segments:
        return 0
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for first, second in new_segments:
        start = bm.verts.new(first)
        end = bm.verts.new(second)
        bm.edges.new((start, end))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj.update_tag(refresh={"OBJECT", "DATA"})
    return len(new_segments)


def _merge_into_existing_bond_object(root, pair_name, order, element, segments):
    """Extend the existing bond object for a group instead of creating one.

    Returns ``True`` when an existing object was found (and therefore nothing
    new should be created).  Any duplicate objects left behind by older builds
    are folded into the first match and deleted, so the outliner converges back
    to one object per pair/order(/element) instead of growing on every grow.
    """
    matches = _bond_group_objects(root, pair_name, order, element)
    if not matches:
        return False
    target = matches[0]
    duplicates = matches[1:]
    if not duplicates:
        # Fast grow path: append the new bonds only.
        _append_bond_segments(target, segments)
        return True
    # One-off repair of a legacy .blend: collect the duplicate geometry, fold
    # it into the survivor and remove the extra objects.
    merged = list(segments)
    for duplicate in duplicates:
        merged.extend(_bond_object_segments(duplicate))
        _remove_object(duplicate)
    _append_bond_segments(target, merged, dedupe=True)
    return True


def _canonical_atom_indices(atoms):
    """Index of the representative atom for every element/position duplicate.

    Disordered CIFs can list the same element twice at one site, and the P1
    expansion then contains two atoms at the same point.  For bonding they are
    one atom: only the first occurrence is bucketed and receives bonds, so a
    disordered site yields one drawn bond instead of a pair of overlapping
    ones.  Mirrors the worker's canonical base-atom rule (``element`` plus the
    rounded position), which is what stops growth adding the copies.
    """
    groups = {}
    canonical = []
    for index, atom in enumerate(atoms):
        key = (
            str(atom.get("element", "X")).upper(),
            _bond_point_key(atom["cart"]),
        )
        canonical.append(groups.setdefault(key, index))
    return canonical


def _bond_objects(root, atoms, pair_settings, start_index=0, append_only=False):
    """Build (or, with ``append_only``, extend) the structure's bond objects.

    ``start_index`` is the first atom index whose bonds are computed; buckets
    are built from the earlier atoms so bonds among pre-existing atoms are
    never re-derived.  With ``append_only`` the existing bond objects are left
    untouched and only bonds for atoms ``[start_index, len(atoms))`` are added
    — this is the grow path, which must never tear down bonds that were built
    under earlier connectivity settings.  Connectivity toggles therefore only
    *add* to the scene; re-deriving everything from the current windows is the
    explicit job of the "Rebuild Bonds" operator (``append_only=False``).

    Duplicate atoms — same element at the same point, e.g. a disordered CIF
    that lists one site twice — are bonded as a single atom: the copies get no
    bonds of their own, so the pair cannot produce a zero-length bond (and a
    doubled set of bonds to its neighbours) when ``min_distance`` is 0.
    """
    global _BOND_SPATIAL_CACHE
    start_index = max(0, min(int(start_index), len(atoms)))
    scene = getattr(bpy.context, "scene", None)
    chemdraw = bool(getattr(scene, "cgn_chemdraw", False))
    model = _current_model()
    enabled = _enabled_pair_windows(pair_settings)
    settings_signature = _connectivity_settings_signature(pair_settings)
    max_distance = max((pair[1] for pair in enabled.values()), default=0.0)

    if append_only:
        # Grow: never remove or rebuild existing bonds.  If there is nothing to
        # bond (no enabled windows, not enough atoms, space-filling, or no new
        # atoms to append to), leave the current bonds exactly as they are.
        if (
            model == "SPACE_FILLING"
            or not enabled
            or len(atoms) < 2
            or max_distance <= 0
            or start_index <= 0
        ):
            return 0
    else:
        # Full rebuild: re-derive all bonds from the current windows.  Any of
        # these invalidation conditions drops the existing bonds.
        if (
            model == "SPACE_FILLING"
            or not enabled
            or len(atoms) < 2
            or max_distance <= 0
        ):
            _BOND_SPATIAL_CACHE = None
            root["cgn_bond_settings_signature"] = settings_signature
            return 0
        start_index = 0

    cell_width = max(max_distance, 1e-6)
    root_id = root.get("cgn_instance_id", root.name)
    # Duplicate atoms (same element, same point) collapse onto their
    # representative: only it is bucketed and only it receives bonds.
    canonical = _canonical_atom_indices(atoms)
    cached = _BOND_SPATIAL_CACHE
    cache_matches = (
        cached is not None
        and cached["root_id"] == root_id
        and cached["settings_signature"] == settings_signature
        and cached["cell_width"] == cell_width
        and cached["atom_count"] == start_index
    )
    if start_index == 0:
        for obj in list(root.objects):
            if obj.get("cgn_role") == "bonds":
                _remove_object(obj)

    # One lookup entry per (element, neighbour element): the enabled window and
    # its canonical pair key.  Rebuilding that key (two ``str.upper`` plus
    # ``sorted``) for every spatial candidate used to dominate the search on
    # packed supercells, although only a few percent of candidates are bonds.
    # ``reach[element]`` is that element's widest enabled window, which bounds
    # how far away any of its bonds can lie.  It is only an early-rejection
    # distance: the bucket geometry stays at the global ``cell_width`` with a
    # one-cell halo (identical coverage to before), but a centre rejects far
    # candidates against its *own* reach instead of the largest global window,
    # so one permissive pair never inflates every centre's distance tests --
    # only centres of that element pay for it.
    pair_entries = {}
    reach = {}
    for (element_a, element_b), (low, high) in enabled.items():
        entry = (low, high, (element_a, element_b))
        pair_entries.setdefault(element_a, {})[element_b] = entry
        pair_entries.setdefault(element_b, {})[element_a] = entry
        reach[element_a] = max(reach.get(element_a, 0.0), high)
        reach[element_b] = max(reach.get(element_b, 0.0), high)

    if cache_matches:
        buckets = cached["buckets"]
        vdw_by_element = cached["vdw_by_element"]
        elements = list(cached["elements"])
        elements.extend([None] * (len(atoms) - len(elements)))
    else:
        buckets = {}
        vdw_by_element = {}
        elements = [None] * len(atoms)
        for index in range(start_index):
            if canonical[index] != index:
                continue
            atom = atoms[index]
            element = str(atom.get("element", "X")).upper()
            elements[index] = element
            vdw_by_element.setdefault(
                element, float(atom.get("vdw_radius", _DEFAULT_VDW_RADIUS))
            )
            x, y, z = atom["cart"]
            cell = (
                math.floor(x / cell_width),
                math.floor(y / cell_width),
                math.floor(z / cell_width),
            )
            buckets.setdefault(cell, []).append(index)
    by_pair = {}
    for index in range(start_index, len(atoms)):
        if canonical[index] != index:
            continue
        atom = atoms[index]
        atom_cart = atom["cart"]
        element = str(atom.get("element", "X")).upper()
        elements[index] = element
        element_entries = pair_entries.get(element, {})
        element_reach = reach.get(element, 0.0)
        if element_reach <= 0.0:
            buckets.setdefault(
                (
                    math.floor(atom_cart[0] / cell_width),
                    math.floor(atom_cart[1] / cell_width),
                    math.floor(atom_cart[2] / cell_width),
                ),
                [],
            ).append(index)
            continue
        vdw_by_element.setdefault(
            element, float(atom.get("vdw_radius", _DEFAULT_VDW_RADIUS))
        )
        x, y, z = atom_cart
        cell = (
            math.floor(x / cell_width),
            math.floor(y / cell_width),
            math.floor(z / cell_width),
        )
        # ``reach_squared`` is this element's own widest window squared -- the
        # safe upper bound for every bond it can form (each enabled window
        # involving the element has ``high <= reach``).  Far candidates are
        # rejected against it rather than the global maximum distance, so a
        # single permissive pair no longer forces every other centre to test
        # out to that answer.
        reach_squared = element_reach * element_reach
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for other_index in buckets.get(neighbor_cell, ()):
                        # Cheapest rejections first: the distance needs no string
                        # work, and most candidates fall outside this element's
                        # own reach (not the largest global window).
                        other = atoms[other_index]
                        other_cart = other["cart"]
                        delta_x = x - other_cart[0]
                        delta_y = y - other_cart[1]
                        delta_z = z - other_cart[2]
                        distance_squared = (
                            delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
                        )
                        if distance_squared > reach_squared:
                            continue
                        if distance_squared <= 1e-12:
                            continue
                        other_element = elements[other_index]
                        entry = element_entries.get(other_element)
                        if entry is None:
                            continue
                        distance = math.sqrt(distance_squared)
                        if entry[0] <= distance <= entry[1]:
                            by_pair.setdefault(entry[2], []).append(
                                (
                                    other_cart,
                                    atom_cart,
                                    other_element,
                                    element,
                                )
                            )
        buckets.setdefault(cell, []).append(index)
    _BOND_SPATIAL_CACHE = {
        "root_id": root_id,
        "settings_signature": settings_signature,
        "cell_width": cell_width,
        "atom_count": len(atoms),
        "atoms": atoms,
        "buckets": buckets,
        "elements": elements,
        "vdw_by_element": vdw_by_element,
    }

    node_group = _make_node_group(_NODE_GROUP_BONDS, is_bond_group=True)
    bond_count = 0
    all_segments = [segment for segments in by_pair.values() for segment in segments]
    covalent_radii = {
        _bond_point_key(atom["cart"]): float(atom.get("covalent_radius", 0.0))
        for atom in atoms
    }
    bond_orders = (
        _chemdraw_bond_orders(all_segments, covalent_radii) if chemdraw else {}
    )
    aromatic_edges, aromatic_rings = (
        _classify_aromatic_rings(all_segments, covalent_radii, bond_orders)
        if chemdraw
        else ({}, [])
    )
    multiple_bond_directions = (
        _multiple_bond_plane_directions(all_segments, bond_orders) if chemdraw else {}
    )
    two_color = getattr(scene, "cgn_bond_style", "SINGLE_COLOR") == "TWO_COLOR"
    for key, segments in sorted(by_pair.items()):
        rendered_segments = (
            _chemdraw_segments(
                segments,
                bond_orders,
                aromatic_edges,
                multiple_bond_directions,
                getattr(
                    scene,
                    "cgn_bond_spacing",
                    _CHEMDRAW_BOND_SPACING_DEFAULT,
                ),
            )
            if chemdraw
            else segments
        )
        pair_name = "_".join(key)
        vdw_a = vdw_by_element.get(key[0], _DEFAULT_VDW_RADIUS)
        vdw_b = vdw_by_element.get(key[1], _DEFAULT_VDW_RADIUS)
        bond_radius = _bond_radius(vdw_a, vdw_b, model)
        segments_by_order = {}
        for segment in rendered_segments:
            first, second, first_element, second_element = segment[:4]
            order = int(segment[4]) if len(segment) > 4 else 1
            segments_by_order.setdefault(order, []).append(
                (first, second, first_element, second_element)
            )
        for order, order_segments in sorted(segments_by_order.items()):
            order_scale = _CHEMDRAW_MULTIPLE_BOND_RADIUS_SCALE.get(order, 1.0)
            order_bond_radius = bond_radius * order_scale
            order_base_radius = bond_radius * order_scale
            if order == 1:
                radius_scale = 1.0
                if model != "WIRE_STICKS":
                    radius_scale = getattr(
                        scene,
                        "cgn_bond_radius_scale",
                        _BOND_RADIUS_SCALE_DEFAULT,
                    )
                order_bond_radius *= radius_scale
            else:
                order_bond_radius *= getattr(
                    scene,
                    "cgn_multiple_bond_radius_scale",
                    _MULTIPLE_BOND_RADIUS_SCALE_DEFAULT,
                )
            order_suffix = f"_Order_{order}" if order > 1 else ""
            if two_color:
                segments_by_element = {}
                for first, second, first_element, second_element in order_segments:
                    midpoint = tuple(
                        0.5 * (first[axis] + second[axis]) for axis in range(3)
                    )
                    segments_by_element.setdefault(first_element, []).append(
                        (first, midpoint)
                    )
                    segments_by_element.setdefault(second_element, []).append(
                        (midpoint, second)
                    )
                for element, element_segments in sorted(segments_by_element.items()):
                    if append_only and _merge_into_existing_bond_object(
                        root,
                        "|".join(key),
                        order,
                        element,
                        [
                            (_display_point(first), _display_point(second))
                            for first, second in element_segments
                        ],
                    ):
                        continue
                    vertices = []
                    edges = []
                    for first, second in element_segments:
                        start = len(vertices)
                        vertices.extend((_display_point(first), _display_point(second)))
                        edges.append((start, start + 1))
                    obj_name = f"CGN_Bonds_{pair_name}_{element}{order_suffix}"
                    mesh = bpy.data.meshes.new(f"{obj_name}_Edges")
                    mesh.from_pydata(vertices, edges, [])
                    mesh.update()
                    obj = bpy.data.objects.new(obj_name, mesh)
                    style = _atom_style(scene, element)
                    material = _make_material(
                        f"CGN_Mat_{element}", style.color, style.roughness
                    )
                    mesh.materials.append(material)
                    obj["cgn_role"] = "bonds"
                    obj["cgn_pair"] = "|".join(key)
                    obj["cgn_bond_element"] = element
                    obj["cgn_bond_order"] = order
                    obj["cgn_base_radius"] = order_base_radius
                    root.objects.link(obj)
                    if "H" in key and not _show_hydrogens():
                        obj.hide_viewport = True
                        obj.hide_render = True
                    _add_geometry_nodes_modifier(
                        obj, node_group, order_bond_radius, material
                    )
            else:
                if append_only and _merge_into_existing_bond_object(
                    root,
                    "|".join(key),
                    order,
                    None,
                    [
                        (_display_point(first), _display_point(second))
                        for first, second, _first_element, _second_element in order_segments
                    ],
                ):
                    continue
                vertices = []
                edges = []
                for first, second, _first_element, _second_element in order_segments:
                    start = len(vertices)
                    vertices.extend((_display_point(first), _display_point(second)))
                    edges.append((start, start + 1))
                obj_name = f"CGN_Bonds_{pair_name}{order_suffix}"
                mesh = bpy.data.meshes.new(f"{obj_name}_Edges")
                mesh.from_pydata(vertices, edges, [])
                mesh.update()
                obj = bpy.data.objects.new(obj_name, mesh)
                obj["cgn_role"] = "bonds"
                obj["cgn_pair"] = "|".join(key)
                obj["cgn_bond_order"] = order
                obj["cgn_base_radius"] = order_base_radius
                root.objects.link(obj)
                if "H" in key and not _show_hydrogens():
                    obj.hide_viewport = True
                    obj.hide_render = True
                material = _make_material(
                    f"CGN_Bond_Mat_{pair_name}",
                    getattr(scene, "cgn_bond_color", _BOND_COLOR_DEFAULT),
                )
                _add_geometry_nodes_modifier(
                    obj, node_group, order_bond_radius, material
                )
        bond_count += len(segments)
    torus_width = getattr(
        scene, "cgn_aromatic_torus_width", _AROMATIC_TORUS_WIDTH_DEFAULT
    )
    torus_rings = _aromatic_torus_segments(
        aromatic_rings,
        getattr(
            scene,
            "cgn_aromatic_torus_size",
            _AROMATIC_TORUS_SIZE_DEFAULT,
        ),
    )
    # Group every aromatic ring into a single object so the scene does not
    # spawn one CGN_Bonds_Aromatic_<n> per ring.  Each ring is an independent
    # closed polyline; all rings share one bevel radius and one material.
    if aromatic_rings:
        vertices = []
        edges = []
        base_radii = []
        for ring_segments in torus_rings:
            first_segment = ring_segments[0]
            element = first_segment[2]
            vdw_radius = vdw_by_element.get(element, _DEFAULT_VDW_RADIUS)
            base_radii.append(_bond_radius(vdw_radius, vdw_radius, model))
            base = len(vertices)
            for first, _second, _first_element, _second_element in ring_segments:
                vertices.append(_display_point(first))
            count = len(vertices) - base
            edges.extend(
                (base + index, base + (index + 1) % count) for index in range(count)
            )
        # One shared bevel radius: use the largest so no ring is under-drawn.
        torus_base_bond_radius = max(base_radii)
        torus_bond_radius = torus_base_bond_radius * torus_width
        existing_torus = next(
            (
                obj
                for obj in root.objects
                if obj.get("cgn_role") == "bonds" and obj.get("cgn_aromatic_torus")
            ),
            None,
        )
        if append_only and existing_torus is not None:
            # Fold the rings detected during this grow into the single aromatic
            # object: creating a new one per grow would add
            # CGN_Bonds_Aromatic.001, .002, ... to the outliner forever.
            try:
                entries = list(
                    json.loads(existing_torus.get("cgn_aromatic_rings", "[]"))
                )
            except (TypeError, ValueError):
                entries = []
            known = {
                (tuple(_bond_point_key(entry[0])), round(float(entry[3]), 4))
                for entry in entries
            }
            for center, axis_u, axis_v, radius, element in aromatic_rings:
                key = (tuple(_bond_point_key(center)), round(float(radius), 4))
                if key in known:
                    continue
                known.add(key)
                entries.append(
                    list(map(list, (center, axis_u, axis_v))) + [radius, element]
                )
            existing_torus["cgn_aromatic_rings"] = json.dumps(entries)
            existing_torus["cgn_aromatic_base_bond_radius"] = max(
                float(existing_torus.get("cgn_aromatic_base_bond_radius", 0.0)),
                torus_base_bond_radius,
            )
            try:
                _update_aromatic_torus_style(scene, bpy.context)
            except Exception:
                traceback.print_exc()
        else:
            obj_name = "CGN_Bonds_Aromatic"
            mesh = bpy.data.meshes.new(f"{obj_name}_Edges")
            mesh.from_pydata(vertices, edges, [])
            mesh.update()
            obj = bpy.data.objects.new(obj_name, mesh)
            obj["cgn_role"] = "bonds"
            obj["cgn_aromatic_torus"] = True
            # Store the raw ring definitions so the style updater can rebuild
            # every ring polyline in the single mesh from a size/width/color
            # change.
            obj["cgn_aromatic_rings"] = json.dumps(
                [
                    list(map(list, (center, axis_u, axis_v))) + [radius, element]
                    for center, axis_u, axis_v, radius, element in aromatic_rings
                ]
            )
            obj["cgn_aromatic_base_bond_radius"] = torus_base_bond_radius
            obj["cgn_base_radius"] = torus_bond_radius
            root.objects.link(obj)
            material = _make_material(
                "CGN_Aromatic_Torus_Mat",
                getattr(
                    scene,
                    "cgn_aromatic_torus_color",
                    _ELEMENT_COLORS["C"],
                ),
            )
            _add_geometry_nodes_modifier(obj, node_group, torus_bond_radius, material)
    root["cgn_bond_settings_signature"] = settings_signature
    return bond_count


def _update_bond_style(scene, context):
    _debounce("bond_style", lambda: _rebuild_bonds(scene))


def _rebuild_bonds(scene):
    """Rebuild all bond objects for the active structure (the heavy core)."""
    root = _active_root(scene)
    if root is None:
        return
    try:
        atoms = _structure_atoms(root)
        _bond_objects(root, atoms, scene.cgn_connectivity_pairs)
    except Exception:
        traceback.print_exc()


def _update_aromatic_torus_style(scene, context):
    root = _active_root(scene)
    if root is None:
        return
    material = _make_material(
        "CGN_Aromatic_Torus_Mat",
        getattr(scene, "cgn_aromatic_torus_color", _ELEMENT_COLORS["C"]),
    )
    size_scale = getattr(scene, "cgn_aromatic_torus_size", _AROMATIC_TORUS_SIZE_DEFAULT)
    for obj in root.objects:
        if not obj.get("cgn_aromatic_torus"):
            continue
        try:
            rings = [
                (
                    tuple(item[0]),
                    tuple(item[1]),
                    tuple(item[2]),
                    float(item[3]),
                    str(item[4]),
                )
                for item in json.loads(obj.get("cgn_aromatic_rings", "[]"))
            ]
        except (TypeError, ValueError, IndexError):
            continue
        if not rings:
            continue
        torus_rings = _aromatic_torus_segments(rings, size_scale)
        vertices = []
        edges = []
        for ring_segments in torus_rings:
            for first, _second, _first_element, _second_element in ring_segments:
                vertices.append(_display_point(first))
            count = len(ring_segments)
            base = len(vertices) - count
            edges.extend(
                (base + index, base + (index + 1) % count) for index in range(count)
            )
        mesh = obj.data
        if len(mesh.vertices) != len(vertices):
            mesh.clear_geometry()
            mesh.from_pydata(vertices, edges, [])
        else:
            for vertex, position in zip(mesh.vertices, vertices):
                vertex.co = position
            if len(mesh.edges) != len(edges):
                mesh.edges.clear()
                for first, second in edges:
                    mesh.edges.new((first, second))
        mesh.update()
        base_radius = (
            float(obj["cgn_aromatic_base_bond_radius"]) * scene.cgn_aromatic_torus_width
        )
        obj["cgn_base_radius"] = base_radius
        radius = base_radius
        for modifier in obj.modifiers:
            if modifier.type == "NODES" and modifier.node_group is not None:
                _set_modifier_input(modifier, modifier.node_group, "Radius", radius)
                _set_modifier_input(modifier, modifier.node_group, "Material", material)
        obj.update_tag(refresh={"OBJECT", "DATA"})
    if context.view_layer is not None:
        context.view_layer.update()


def _update_bond_radius_scale(scene, context):
    if scene.cgn_model in {"WIRE_STICKS", "SPACE_FILLING"}:
        return
    root = _active_root(scene)
    if root is None:
        return
    bonds = [
        obj
        for obj in root.objects
        if obj.get("cgn_role") == "bonds"
        and not obj.get("cgn_aromatic_torus")
        and int(obj.get("cgn_bond_order", 1)) == 1
    ]
    if any("cgn_base_radius" not in obj for obj in bonds):
        _rebuild_bonds(scene)
        root = _active_root(scene)
        bonds = [
            obj
            for obj in root.objects
            if obj.get("cgn_role") == "bonds"
            and not obj.get("cgn_aromatic_torus")
            and int(obj.get("cgn_bond_order", 1)) == 1
        ]
    for obj in bonds:
        radius = float(obj["cgn_base_radius"]) * scene.cgn_bond_radius_scale
        for modifier in obj.modifiers:
            if modifier.type == "NODES" and modifier.node_group is not None:
                _set_modifier_input(modifier, modifier.node_group, "Radius", radius)
                obj.update_tag(refresh={"OBJECT", "DATA"})
    if context.view_layer is not None:
        context.view_layer.update()


def _update_multiple_bond_radius_scale(scene, context):
    root = _active_root(scene)
    if root is None:
        return
    bonds = [
        obj
        for obj in root.objects
        if obj.get("cgn_role") == "bonds" and int(obj.get("cgn_bond_order", 1)) > 1
    ]
    if any("cgn_base_radius" not in obj for obj in bonds):
        _rebuild_bonds(scene)
        root = _active_root(scene)
        bonds = [
            obj
            for obj in root.objects
            if obj.get("cgn_role") == "bonds" and int(obj.get("cgn_bond_order", 1)) > 1
        ]
    for obj in bonds:
        radius = float(obj["cgn_base_radius"]) * scene.cgn_multiple_bond_radius_scale
        for modifier in obj.modifiers:
            if modifier.type == "NODES" and modifier.node_group is not None:
                _set_modifier_input(modifier, modifier.node_group, "Radius", radius)
                obj.update_tag(refresh={"OBJECT", "DATA"})
    if context.view_layer is not None:
        context.view_layer.update()


def _update_bond_color(scene, context):
    if scene.cgn_bond_style != "SINGLE_COLOR":
        return
    root = _active_root(scene)
    if root is None:
        return
    for obj in root.objects:
        if (
            obj.get("cgn_role") != "bonds"
            or obj.get("cgn_bond_element")
            or obj.get("cgn_aromatic_torus")
        ):
            continue
        pair_name = str(obj.get("cgn_pair", "")).replace("|", "_")
        if not pair_name:
            continue
        material = _make_material(f"CGN_Bond_Mat_{pair_name}", scene.cgn_bond_color)
        for modifier in obj.modifiers:
            if modifier.type == "NODES" and modifier.node_group is not None:
                _set_modifier_input(modifier, modifier.node_group, "Material", material)
                obj.update_tag(refresh={"OBJECT", "DATA"})
    if context.view_layer is not None:
        context.view_layer.update()


def _set_scene_units(scene):
    """Display the scene at 1 Å = 1 cm (1 Blender unit = 10 Å)."""
    units = scene.unit_settings
    units.system = "METRIC"
    units.scale_length = 0.1
    units.length_unit = "CENTIMETERS"


def _import_result(scene, result, cif_path):
    # A new structure is about to replace the active one; drop any parsed tables
    # cached for the previous structure so they do not linger in memory.
    _invalidate_structure_cache()
    sites = result.get("sites", [])
    instances = result.get("instances", [])
    cell_matrix = result.get("cell_matrix", [])
    if len(cell_matrix) != 9:
        raise ValueError("The CIF reader returned an invalid cell matrix.")
    initial_count = len(instances) // 4
    atoms = _materialize_atoms(sites, cell_matrix, instances, initial_count)
    if not atoms:
        raise ValueError("The selected CIF contains no atoms.")

    root = bpy.data.collections.new("CrystalViz_Structure")
    root["cgn_role"] = "structure"
    root["cgn_instance_id"] = uuid.uuid4().hex
    root["cgn_space_group"] = result.get("space_group", "unknown")
    root["cgn_space_group_ops"] = json.dumps(result.get("space_group_ops", []))
    root["cgn_unit_cell"] = json.dumps(result.get("unit_cell", []))
    root["cgn_unit_cell_display"] = json.dumps(result.get("unit_cell_display", []))
    root["cgn_cif_block"] = result.get("label", "")
    root["cgn_cif_path"] = cif_path
    root["cgn_sites"] = json.dumps(sites)
    _set_instances(root, instances)
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
            pair.element_a,
            pair.element_b,
            pair.enabled,
            pair.min_distance,
            pair.max_distance,
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
        charges = {
            str(key).upper(): float(value)
            for key, value in (result.get("atom_type_charges") or {}).items()
        }
        metals = {str(element).upper() for element in result.get("metal_elements", [])}
        ionic = {
            str(key).upper(): float(value)
            for key, value in (result.get("ionic_radii") or {}).items()
        }
        for index, element_a in enumerate(elements):
            for element_b in elements[index:]:
                item = scene.cgn_connectivity_pairs.add()
                item.element_a = element_a
                item.element_b = element_b
                item.enabled = True
                minimum, maximum = _default_pair_window(
                    element_a, element_b, radii, charges, metals, ionic
                )
                item.min_distance = minimum
                item.max_distance = maximum
        _bond_objects(root, atoms, scene.cgn_connectivity_pairs)
        # Single-structure view: drop every previous structure (not just the
        # one the scene pointer happens to name) so repeated imports cannot
        # accumulate hidden collections and their meshes.
        _purge_structure_roots(scene, keep=root)
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
    """Return one whole image of each finite molecule plus periodic contents.

    A finite molecule is unwrapped once and shifted so its center is near the
    unit-cell center. Periodic components cannot be represented as finite
    molecules, so their original unit-cell contents are retained.
    """
    windows = _enabled_pair_windows(pair_settings)
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
        matrix = Matrix(
            (
                (cell_matrix[0], cell_matrix[3], cell_matrix[6]),
                (cell_matrix[1], cell_matrix[4], cell_matrix[7]),
                (cell_matrix[2], cell_matrix[5], cell_matrix[8]),
            )
        )
        inverse = matrix.inverted()
    except (ValueError, ZeroDivisionError):
        return None

    # A Cartesian move of max_distance cannot change a fractional coordinate
    # by more than max_distance times the length of that row of the inverse.
    reach = [max_distance * inverse.row[axis].length for axis in range(3)]
    cells = [max(1, int(math.ceil(value))) for value in reach]

    # The image bank costs len(sites) * prod(2*cells[i]+1) entries and the
    # adjacency pass visits every one of them, so bound the product *before*
    # allocating it.  A small cell with a wide bond window explodes here: the
    # 10 A ceiling in a 1 A cell is 21^3 = 9261 images per site.
    image_count = len(sites)
    for axis in range(3):
        image_count *= 2 * cells[axis] + 1
    if image_count > max_instances:
        raise ValueError(
            f"The connectivity window spans {image_count} cell images "
            f"({2 * cells[0] + 1}x{2 * cells[1] + 1}x{2 * cells[2] + 1} per "
            f"site); reduce the max bond distance or use a larger unit cell."
        )

    fracs = [Vector(site["frac"]) for site in sites]
    carts = [matrix @ frac for frac in fracs]
    elements = [str(site.get("element", "X")).upper() for site in sites]
    width = max(max_distance, 1e-6)

    bank = {}
    for index, frac in enumerate(fracs):
        for tx in range(-cells[0], cells[0] + 1):
            for ty in range(-cells[1], cells[1] + 1):
                for tz in range(-cells[2], cells[2] + 1):
                    image_frac = Vector(
                        (
                            frac.x + tx,
                            frac.y + ty,
                            frac.z + tz,
                        )
                    )
                    image_cart = matrix @ image_frac
                    key = (
                        math.floor(image_cart.x / width),
                        math.floor(image_cart.y / width),
                        math.floor(image_cart.z / width),
                    )
                    bank.setdefault(key, []).append((index, (tx, ty, tz), image_cart))

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
                    for other_index, translation, image_cart in bank.get(neighbour, ()):
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
                        if other_index == index and translation == (0, 0, 0):
                            continue
                        edge = (index, other_index, translation)
                        if edge in seen:
                            continue
                        reverse = (
                            other_index,
                            index,
                            (-translation[0], -translation[1], -translation[2]),
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

    instances = set()
    for component_index, component in enumerate(components):
        if component_index in periodic:
            instances.update((index, 0, 0, 0) for index in component)
            continue
        anchors = {
            (-translations[index][0], -translations[index][1], -translations[index][2])
            for index in component
        }
        center = tuple(
            sum(
                float(sites[index]["frac"][axis]) + translations[index][axis]
                for index in component
            )
            / len(component)
            for axis in range(3)
        )
        anchor = min(
            anchors,
            key=lambda value: (
                sum((center[axis] + value[axis] - 0.5) ** 2 for axis in range(3)),
                value,
            ),
        )
        for index in component:
            translation = translations[index]
            instances.add(
                (
                    index,
                    translation[0] + anchor[0],
                    translation[1] + anchor[1],
                    translation[2] + anchor[2],
                )
            )
            if len(instances) > max_instances:
                raise ValueError(
                    f"Molecule completion exceeds the {max_instances}-atom fill limit."
                )

    return [value for instance in sorted(instances) for value in instance]


def _extend_cell_boundary_instances(
    sites, instances, pack_repeats=None, tolerance=1e-4, max_instances=None
):
    """Extend P1 cell instances so atoms appear at both [0] and [1] cell faces.

    For CELL fill mode, periodic frameworks (diamond, MOFs, etc.) keep each
    site only at translation (0, 0, 0). An atom at fractional coordinate 0.0
    therefore appears at one corner only. This function adds the equivalent
    boundary images so corner atoms appear at all 8 corners, edge atoms on all
    edges, and face atoms on both faces — giving a complete cell in [0, 1]
    range.

    For PACK fill mode, every cell extends frac≈0 atoms by +1 on every axis,
    giving a ``[0,1)[0,1)[0,1]...[0,1]`` scheme along each axis.  Atoms on
    shared internal faces are deduplicated by the ``seen`` set, so no atom
    appears twice.

    ``pack_repeats`` is (ra, rb, rc) for PACK mode, or None for CELL mode.
    It is accepted for interface compatibility but all cells are extended
    identically; ``seen`` handles deduplication.

    Returns an extended flat quadruple list with boundary images appended.

    When *max_instances* is given the atom count is checked while the images
    are generated, so an over-limit fill is abandoned instead of allocated.
    """
    if not instances:
        return instances

    expanded = []
    seen = set()
    count = len(instances) - len(instances) % 4

    # Keep all original instances.
    for offset in range(0, count, 4):
        key = (
            int(instances[offset]),
            int(instances[offset + 1]),
            int(instances[offset + 2]),
            int(instances[offset + 3]),
        )
        expanded.extend(key)
        seen.add(key)
    if max_instances is not None and len(seen) > max_instances:
        raise ValueError(
            f"This fill would create more than {max_instances} atoms; "
            "reduce the repeats."
        )

    # Add boundary images where local frac is near 0 on any axis.  Every cell
    # extends by +1 on all axes; duplicates at internal faces are skipped
    # by ``seen``.
    for offset in range(0, count, 4):
        site_index = int(instances[offset])
        tx0, ty0, tz0 = (
            int(instances[offset + 1]),
            int(instances[offset + 2]),
            int(instances[offset + 3]),
        )
        frac = sites[site_index]["frac"]  # LOCAL fractional position in [0, 1)

        # Cumulatively build +1 boundary translations.
        translations = [(0, 0, 0)]
        for axis, coord in enumerate(frac):
            new_translations = []
            for t in translations:
                if coord < tolerance:
                    t_list = list(t)
                    t_list[axis] += 1
                    new_translations.append(tuple(t_list))
            translations.extend(new_translations)

        for t in translations:
            if t == (0, 0, 0):
                continue
            key = (site_index, tx0 + t[0], ty0 + t[1], tz0 + t[2])
            if key not in seen:
                seen.add(key)
                expanded.extend(key)
                if max_instances is not None and len(seen) > max_instances:
                    raise ValueError(
                        f"This fill would create more than {max_instances} "
                        "atoms; reduce the repeats."
                    )

    return expanded


def _fill_structure(operator, context, mode, repeats=(1, 1, 1)):
    """Replace displayed atoms with the ASU, completed cell, or packed cells."""
    root = _active_root(context.scene)
    if root is None:
        operator.report({"ERROR"}, _no_structure_message(context.scene, "filling the structure"))
        return {"CANCELLED"}
    try:
        sites, _instances, cell_matrix, _initial_count = _structure_tables(root)
    except (ValueError, TypeError) as exc:
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}
    if not sites:
        operator.report({"ERROR"}, "The stored site table is empty.")
        return {"CANCELLED"}

    if mode == "ASU":
        baseline = root.get("cgn_asu_instances", "")
        if not baseline:
            operator.report(
                {"ERROR"},
                "Re-import this CIF once to enable the Fill menu.",
            )
            return {"CANCELLED"}
        try:
            new_instances = json.loads(baseline)
        except (ValueError, TypeError) as exc:
            operator.report(
                {"ERROR"}, f"Stored asymmetric-unit table is invalid: {exc}"
            )
            return {"CANCELLED"}
        if not new_instances:
            operator.report(
                {"ERROR"},
                "Re-import this CIF once to enable the Fill menu.",
            )
            return {"CANCELLED"}
        repeats = (1, 1, 1)
        description = "asymmetric unit"
    else:
        if mode == "PACK":
            repeats = tuple(
                max(1, min(_MAX_FILL_REPEATS, int(value))) for value in repeats
            )
            description = f"{repeats[0]}×{repeats[1]}×{repeats[2]} unit cells"
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
            operator.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
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
                                {"ERROR"},
                                f"This fill would create more than "
                                f"{_MAX_FILL_ATOMS} atoms; reduce the repeats.",
                            )
                            return {"CANCELLED"}
                        new_instances.extend(key)

        # Extend P1 cell instances to [0, 1] range so atoms on cell boundaries
        # appear on all faces/corners for periodic frameworks (diamond, MOFs).
        # Scheme: [0,1)[0,1)[0,1]...[0,1] along each axis; seen() prevents
        # duplicates at shared internal faces.
        pack_repeats = repeats if mode == "PACK" else None
        try:
            new_instances = _extend_cell_boundary_instances(
                sites,
                new_instances,
                pack_repeats=pack_repeats,
                max_instances=_MAX_FILL_ATOMS,
            )
        except ValueError as exc:
            operator.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        # The raise above stops the extension mid-loop; this final check keeps
        # the guarantee explicit for callers that pass no max_instances.
        if len(new_instances) // 4 > _MAX_FILL_ATOMS:
            operator.report(
                {"ERROR"},
                f"This fill would create more than "
                f"{_MAX_FILL_ATOMS} atoms; reduce the repeats.",
            )
            return {"CANCELLED"}
        cell_atom_count = len(new_instances) // 4

    atoms = _materialize_atoms(
        sites,
        cell_matrix,
        new_instances,
        len(new_instances) // 4,
    )
    if not atoms:
        operator.report({"ERROR"}, "The fill produced no atoms.")
        return {"CANCELLED"}
    snapshot = _snapshot_structure_state(root, context.scene.cgn_connectivity_pairs)
    try:
        _rebuild_atom_objects(root, atoms)
        _update_growth_selection(root, atoms, [])
        _bond_objects(root, atoms, context.scene.cgn_connectivity_pairs)
        _clear_polyhedra(root)
        _apply_unit_cell_settings(context.scene)
    except Exception as exc:
        traceback.print_exc()
        # A mid-rebuild failure must not leave the displayed atoms inconsistent
        # with the stored data: restore the pre-operation state atomically.
        _rollback_structure_state(
            root, snapshot, context.scene, context.scene.cgn_connectivity_pairs
        )
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}

    _set_instances(root, new_instances)
    root["cgn_initial_count"] = len(atoms)
    root["cgn_fill_mode"] = mode
    if mode != "ASU":
        root["cgn_cell_atoms"] = cell_atom_count
    if mode == "PACK":
        root["cgn_fill_repeats"] = json.dumps(list(repeats))
    root["cgn_growth_shells"] = 0
    operator.report({"INFO"}, f"Filled {len(atoms)} atoms ({description}).")
    return {"FINISHED"}


class CGN_OT_import_cif(Operator):
    bl_idname = "cgn.import_cif"
    bl_label = "Import CIF"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH")
    block_name: StringProperty(
        name="CIF Block",
        description="Optional exact data-block name for multi-structure CIF files",
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if not os.path.isfile(self.filepath):
            self.report({"ERROR"}, "The selected CIF file does not exist.")
            return {"CANCELLED"}
        try:
            worker = _gemmi_worker()
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        try:
            result = worker.load_structure(
                bpy.path.abspath(self.filepath), self.block_name or None
            )
            root = _import_result(context.scene, result, self.filepath)
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Imported {len(result.get('instances', [])) // 4} ASU atoms "
            f"({root.get('cgn_space_group')}).",
        )
        return {"FINISHED"}


def _default_connectivity_pairs(scene, atoms):
    """Repopulate ``scene.cgn_connectivity_pairs`` with the default windows for
    the elements present in ``atoms`` (covalent-radius defaults, as import
    computes for the common case)."""
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
            minimum, maximum = _default_pair_window(
                element_a, element_b, radii, {}, set(), {}
            )
            item.min_distance = minimum
            item.max_distance = maximum


class CGN_OT_rebuild_connectivity(Operator):
    bl_idname = "cgn.rebuild_connectivity"
    bl_label = "Reset Default"
    bl_description = (
        "Restore the default connectivity windows for the next growth. "
        "Already-rendered bonds are kept; the restored settings steer future "
        "growing of selected atoms."
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        root = _active_root(context.scene)
        if root is None:
            self.report({"ERROR"}, _no_structure_message(context.scene, "resetting connectivity"))
            return {"CANCELLED"}
        try:
            atoms = _structure_atoms(root)
        except (ValueError, TypeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        # Non-destructive: restore the default pair windows but DO NOT tear down
        # the bonds already on screen.  Under Option B the restored settings are
        # used the next time the user grows selected atoms.
        _default_connectivity_pairs(context.scene, atoms)
        self.report(
            {"INFO"},
            "Connectivity reset to defaults. Existing bonds kept; the next grow "
            "uses these windows.",
        )
        return {"FINISHED"}


class CGN_OT_fill_asu(Operator):
    bl_idname = "cgn.fill_asu"
    bl_label = "Fill Asymmetric Unit-cell"
    bl_description = "Show only the atoms of the CIF asymmetric unit"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        return _fill_structure(self, context, "ASU")


class CGN_OT_fill_cell(Operator):
    bl_idname = "cgn.fill_cell"
    bl_label = "Fill Unit-Cell"
    bl_description = (
        "Show one unit cell and complete finite molecules across its boundaries"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        return _fill_structure(self, context, "CELL")


class CGN_OT_fill_pack(Operator):
    bl_idname = "cgn.fill_pack"
    bl_label = "Pack Unit-cells"
    bl_description = (
        "Show repeated unit cells with finite molecules completed at the edges"
    )
    bl_options = {"REGISTER", "UNDO"}

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
                    int(value)
                    for value in json.loads(root.get("cgn_fill_repeats", "[]"))
                ]
            except (ValueError, TypeError):
                repeats = []
            if len(repeats) == 3:
                self.repeat_a = max(1, min(_MAX_FILL_REPEATS, repeats[0]))
                self.repeat_b = max(1, min(_MAX_FILL_REPEATS, repeats[1]))
                self.repeat_c = max(1, min(_MAX_FILL_REPEATS, repeats[2]))
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
                count = cell_atoms * self.repeat_a * self.repeat_b * self.repeat_c
                layout.label(text=f"Atoms (approx.): {count}")

    def execute(self, context):
        return _fill_structure(
            self,
            context,
            "PACK",
            (self.repeat_a, self.repeat_b, self.repeat_c),
        )


class CGN_OT_pack_cell_boxes(Operator):
    bl_idname = "cgn.pack_cell_boxes"
    bl_label = "Pack Unit-Cells (Faces & Edges)"
    bl_description = "Show a packed unit-cell grid without adding or changing atoms"
    bl_options = {"REGISTER", "UNDO"}

    repeat_a: IntProperty(name="a", default=2, min=1, max=_MAX_FILL_REPEATS)
    repeat_b: IntProperty(name="b", default=2, min=1, max=_MAX_FILL_REPEATS)
    repeat_c: IntProperty(name="c", default=2, min=1, max=_MAX_FILL_REPEATS)

    def invoke(self, context, event):
        root = _active_root(context.scene)
        if root is not None:
            try:
                repeats = [
                    int(value)
                    for value in json.loads(
                        root.get("cgn_cell_pack_repeats", "[2, 2, 2]")
                    )
                ]
            except (ValueError, TypeError):
                repeats = []
            if len(repeats) == 3:
                self.repeat_a, self.repeat_b, self.repeat_c = (
                    max(1, min(_MAX_FILL_REPEATS, value)) for value in repeats
                )
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Number of unit-cell boxes along each axis:")
        row = layout.row(align=True)
        row.prop(self, "repeat_a")
        row.prop(self, "repeat_b")
        row.prop(self, "repeat_c")
        layout.label(text="Draws faces and edges only; atoms are unchanged.")

    def execute(self, context):
        root = _active_root(context.scene)
        if root is None:
            self.report({"ERROR"}, _no_structure_message(context.scene, "packing unit cells"))
            return {"CANCELLED"}
        cell_matrix = _stored_cell_matrix(root)
        if cell_matrix is None:
            self.report({"ERROR"}, "The structure has no valid unit-cell data.")
            return {"CANCELLED"}
        repeats = tuple(
            max(1, min(_MAX_FILL_REPEATS, int(value)))
            for value in (self.repeat_a, self.repeat_b, self.repeat_c)
        )
        root["cgn_cell_pack_repeats"] = json.dumps(list(repeats))
        settings = context.scene.cgn_unit_cell_settings
        settings.show_unit_cell = True
        _show_unit_cell(root, cell_matrix, settings, repeats)
        self.report(
            {"INFO"},
            f"Displayed {repeats[0]}×{repeats[1]}×{repeats[2]} unit-cell boxes.",
        )
        return {"FINISHED"}


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
    inside = (x >= 0.0) & (x < width) & (y >= 0.0) & (y < height)
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
    best = candidates[np.lexsort((distance[candidates], depths[candidates]))[0]]
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
    select_action = "selecting atoms"
    status_text = ""
    keep_stored_selection = True
    # The grow operator keeps its seed sphere visible until the deferred grow
    # runs; delete and polyhedra operators do not, so their confirm (and any
    # early-return) path must drop the selection sphere instead of leaving an
    # orphan behind.
    keep_selection_sphere = True

    def invoke(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"ERROR"}, "Run atom selection from the 3D Viewport.")
            return {"CANCELLED"}
        root = _active_root(context.scene)
        if root is None:
            self.report({"ERROR"}, _no_structure_message(context.scene, self.select_action))
            return {"CANCELLED"}
        self._region = next(
            (region for region in context.area.regions if region.type == "WINDOW"),
            None,
        )
        if self._region is None:
            self.report({"ERROR"}, "The 3D Viewport window region is unavailable.")
            return {"CANCELLED"}

        self._root_name = root.name
        self._root_id = root.get("cgn_instance_id")
        try:
            self._atoms = _structure_atoms(root)
        except (ValueError, TypeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        try:
            self._initial_selection = set(
                json.loads(root.get("cgn_selected_indices", "[]"))
            )
        except (ValueError, TypeError) as exc:
            self.report({"ERROR"}, f"Stored atom selection is invalid: {exc}")
            return {"CANCELLED"}
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
            self._draw_box, (), "WINDOW", "POST_PIXEL"
        )
        context.workspace.status_text_set(self.status_text)
        context.window.cursor_set("CROSSHAIR")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish(self, context, cancel=False):
        if cancel:
            final = set(self._initial_selection)
        elif not self.keep_selection_sphere:
            final = set()
        else:
            final = set(self._selected)
        root = bpy.data.collections.get(self._root_name)
        if root is not None and root.get("cgn_instance_id") == self._root_id:
            # Persist the final selection exactly once, at the modal boundary.
            # The live set is kept in memory during the modal, so ``_apply``
            # only refreshes the highlight and never re-serializes here.
            _update_growth_selection(root, self._atoms, final)
        if getattr(self, "_draw_handle", None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, "WINDOW")
            self._draw_handle = None
        context.workspace.status_text_set(None)
        context.window.cursor_set("DEFAULT")
        if context.area is not None:
            context.area.tag_redraw()

    def _region_mouse(self, event, clamp=False):
        point = Vector(
            (
                event.mouse_x - self._region.x,
                event.mouse_y - self._region.y,
            )
        )
        if clamp:
            point.x = min(max(point.x, 0), self._region.width - 1)
            point.y = min(max(point.y, 0), self._region.height - 1)
            return point
        if 0 <= point.x < self._region.width and 0 <= point.y < self._region.height:
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
        batch = batch_for_shader(shader, "LINE_STRIP", {"pos": points})
        gpu.state.blend_set("ALPHA")
        gpu.state.line_width_set(2.0)
        try:
            shader.bind()
            shader.uniform_float("color", (1.0, 0.55, 0.05, 0.9))
            batch.draw(shader)
        finally:
            gpu.state.line_width_set(1.0)
            gpu.state.blend_set("NONE")

    def _apply_selection(self, context):
        if self._drag_start is None or self._drag_end is None:
            return
        root = bpy.data.collections.get(self._root_name)
        if root is None or root.get("cgn_instance_id") != self._root_id:
            self.report({"ERROR"}, "The active structure changed during selection.")
            return

        if self._dragging:
            if not self._drag_additive:
                self._selected.clear()
            x_min, x_max = sorted((self._drag_start.x, self._drag_end.x))
            y_min, y_max = sorted((self._drag_start.y, self._drag_end.y))
            indices, points, _depths, visible = self._project_atoms(context)
            boxed = (
                (points[:, 0] >= x_min)
                & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] <= y_max)
            )
            # Hidden atoms (e.g. hydrogens) are projected too; keep the box
            # selection consistent with click selection and skip them.
            self._selected.update(int(index) for index in indices[boxed & visible])
        else:
            indices, points, depths, visible = self._project_atoms(context)
            nearest = _click_candidate(
                indices,
                points,
                depths,
                visible,
                self._drag_start,
                self.max_pixel_distance,
            )
            _select_click(self._selected, nearest, self._drag_additive)

        _refresh_growth_selection(root, self._atoms, self._selected)
        context.area.tag_redraw()

    def modal(self, context, event):
        if event.type == "ESC":
            self._finish(context, cancel=True)
            return {"CANCELLED"}
        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            return self._confirm(context)
        if event.type == "RIGHTMOUSE" and event.value == "PRESS":
            return self._confirm(context)

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            point = self._region_mouse(event)
            if point is not None:
                self._drag_start = point
                self._drag_end = point.copy()
                self._drag_additive = event.shift
                self._dragging = False
            return {"RUNNING_MODAL"}
        if (
            event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}
            and self._drag_start is not None
        ):
            point = self._region_mouse(event, clamp=True)
            self._drag_end = point
            if (self._drag_end - self._drag_start).length >= self.box_drag_threshold:
                self._dragging = True
            if context.area is not None:
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if self._drag_start is not None:
                point = self._region_mouse(event, clamp=True)
                if point is not None:
                    self._drag_end = point
                self._apply_selection(context)
                self._drag_start = None
                self._drag_end = None
                self._dragging = False
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


def _grow_from_selection():
    """Run the grow operator after the selection modal has fully finished."""
    try:
        bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
    except RuntimeError as exc:
        print(f"CrystalViz: grow failed: {exc}")
    # grow_further clears the seed sphere on its success and "grew_by == 0"
    # paths, but its CANCELLED early returns leave the seed selection sphere
    # on screen.  Drop any leftover sphere so nothing lingers regardless of
    # what the grow reported.
    try:
        root = _active_root(bpy.context.scene)
        if root is not None and root.get("cgn_selected_indices") not in ("[]", ""):
            _update_growth_selection(root, _structure_atoms(root), [])
    except (ValueError, TypeError):
        pass
    return None


class CGN_OT_select_growth_atoms(_CGN_OT_select_atoms):
    bl_idname = "cgn.select_growth_atoms"
    bl_label = "Grow Selected Atoms"
    bl_description = (
        "Select atoms in the viewport, then confirm to grow around them; "
        "click a selected atom to remove it, Shift adds to the selection"
    )
    # NOTE: this REGISTER/UNDO modal, together with the grow operator it defers
    # to a bpy.app.timers callback, MERGES into a single undo step in Blender
    # 5.2 (verified empirically). One Ctrl+Z reverts the whole select-and-grow
    # back to the pre-modal baseline -- it does NOT restore an intermediate
    # "seeds still selected, growth reverted" state. The old comment claiming
    # two separate undo steps was wrong; don't rely on that.
    bl_options = {"REGISTER", "UNDO"}

    select_action = "selecting growth atoms"
    status_text = (
        "Click atoms to add/remove, drag a box to select; Shift adds. "
        "Enter/right-click grows around them, Esc cancels."
    )

    def _confirm(self, context):
        if not self._selected:
            self._finish(context)
            self.report({"ERROR"}, "Select one or more atoms before growing.")
            return {"CANCELLED"}
        self._finish(context)
        # The grow operator is modal on the cctbx build, and starting a modal
        # operator from inside this modal does not receive events. Defer it
        # until the selection operator has returned. (On this gemmi build the
        # grow is synchronous, so the deferral only buys a 10 ms window with a
        # stale seed sphere on screen -- harmless, kept for cctbx parity.)
        bpy.app.timers.register(_grow_from_selection, first_interval=0.01)
        return {"FINISHED"}


class CGN_OT_delete_atoms(_CGN_OT_select_atoms):
    bl_idname = "cgn.delete_atoms"
    bl_label = "Delete Atoms"
    bl_description = (
        "Click atoms or drag a box around them, then confirm to delete; "
        "click a selected atom to remove it, Shift adds to the selection"
    )
    bl_options = {"REGISTER", "UNDO"}

    select_action = "deleting atoms"
    status_text = (
        "Click atoms to add/remove, drag a box to select; Shift adds. "
        "Enter/right-click deletes them, Esc cancels."
    )
    keep_stored_selection = False
    keep_selection_sphere = False

    def _confirm(self, context):
        if not self._selected:
            self._finish(context)
            self.report({"ERROR"}, "Select one or more atoms before deleting.")
            return {"CANCELLED"}
        root = bpy.data.collections.get(self._root_name)
        if root is None or root.get("cgn_instance_id") != self._root_id:
            self._finish(context)
            self.report({"ERROR"}, "The active structure changed during selection.")
            return {"CANCELLED"}
        try:
            sites, instances, cell_matrix, initial_count = _structure_tables(root)
        except (ValueError, TypeError) as exc:
            self._finish(context)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        atom_count = len(instances) // 4
        removed = sorted(
            {
                index
                for index in self._selected
                if isinstance(index, int) and 0 <= index < atom_count
            }
        )
        if not removed:
            self._finish(context)
            self.report({"ERROR"}, "Select one or more atoms before deleting.")
            return {"CANCELLED"}
        if len(removed) >= atom_count:
            self._finish(context)
            self.report(
                {"ERROR"},
                "Deleting every atom is not supported; re-import the CIF instead.",
            )
            return {"CANCELLED"}

        removed_offsets = {index * 4 for index in removed}
        removed_quads = {
            (
                int(instances[index * 4]),
                int(instances[index * 4 + 1]),
                int(instances[index * 4 + 2]),
                int(instances[index * 4 + 3]),
            )
            for index in removed
        }
        new_instances = [
            value
            for offset, value in enumerate(instances)
            if (offset - offset % 4) not in removed_offsets
        ]
        removed_below_initial = sum(1 for index in removed if index < initial_count)
        new_initial_count = max(0, initial_count - removed_below_initial)
        atoms = _materialize_atoms(sites, cell_matrix, new_instances, new_initial_count)

        self._finish(context)
        snapshot = _snapshot_structure_state(root, context.scene.cgn_connectivity_pairs)
        _set_instances(root, new_instances)
        root["cgn_initial_count"] = new_initial_count
        try:
            _rebuild_atom_objects(root, atoms)
            _update_growth_selection(root, atoms, [])
            _bond_objects(root, atoms, context.scene.cgn_connectivity_pairs)
            _remap_polyhedra(root, removed, atoms, context.scene, removed_quads)
        except Exception as exc:
            traceback.print_exc()
            # Roll the stored state back so the viewport and data stay
            # consistent even when a rebuild fails mid-way (e.g. a polyhedron
            # regen error) instead of leaving half-mutated instances behind.
            try:
                _rollback_structure_state(
                    root,
                    snapshot,
                    context.scene,
                    context.scene.cgn_connectivity_pairs,
                )
            except Exception as rollback_exc:  # pragma: no cover - defensive
                traceback.print_exc()
                self.report(
                    {"ERROR"},
                    f"Delete failed and rollback also failed: {rollback_exc}",
                )
                return {"CANCELLED"}
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Deleted {len(removed)} atoms; structure now contains {len(atoms)} atoms.",
        )
        return {"FINISHED"}


class CGN_OT_grow_further(Operator):
    bl_idname = "cgn.grow_further"
    bl_label = "Grow Structure"
    bl_description = "Add one shell around all atoms or the selected growth atoms"
    bl_options = {"REGISTER", "UNDO"}

    all_atoms: BoolProperty(
        name="Grow from all atoms",
        description="Use every atom as a growth seed",
        default=True,
        options={"HIDDEN"},
    )

    def execute(self, context):
        root = _active_root(context.scene)
        if root is None:
            self.report({"ERROR"}, _no_structure_message(context.scene, "growing the structure"))
            return {"CANCELLED"}
        enabled_pairs = [
            item
            for item in context.scene.cgn_connectivity_pairs
            if item.enabled and item.max_distance >= item.min_distance
        ]
        if not enabled_pairs:
            self.report({"ERROR"}, "Enable at least one valid connectivity pair.")
            return {"CANCELLED"}
        try:
            sites, instances, cell_matrix, initial_count = _structure_tables(root)
        except (ValueError, TypeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        try:
            selected_indices = json.loads(root.get("cgn_selected_indices", "[]"))
        except (ValueError, TypeError) as exc:
            self.report({"ERROR"}, f"Stored growth selection is invalid: {exc}")
            return {"CANCELLED"}
        atom_count = len(instances) // 4
        if not atom_count:
            self.report({"ERROR"}, "The active structure contains no atoms.")
            return {"CANCELLED"}
        if self.all_atoms:
            selected_indices = list(range(atom_count))
        else:
            selected_indices = sorted(
                {
                    index
                    for index in selected_indices
                    if isinstance(index, int) and 0 <= index < atom_count
                }
            )
        if not selected_indices:
            self.report(
                {"ERROR"},
                "Select one or more atoms before growing the structure.",
            )
            return {"CANCELLED"}
        seed_instances = []
        for index in selected_indices:
            offset = index * 4
            seed_instances.extend(instances[offset : offset + 4])

        try:
            worker = _gemmi_worker()
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        try:
            result = worker.grow_connectivity_shell(
                json.loads(root.get("cgn_unit_cell", "[]")),
                [{"element": site["element"], "frac": site["frac"]} for site in sites],
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
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        grew_by = int(result.get("grew_by", 0))
        if grew_by == 0:
            # Nothing matched, so the growth is finished; drop the lingering
            # selection sphere that was shown while choosing the seed atoms.
            _update_growth_selection(root, [], [])
            self.report(
                {"INFO"},
                "No additional atoms match the current connectivity settings.",
            )
            return {"FINISHED"}
        new_instances = result.get("new_instances", [])
        if len(new_instances) < grew_by * 4:
            self.report(
                {"ERROR"}, "The gemmi worker returned an invalid growth result."
            )
            return {"CANCELLED"}
        snapshot = _snapshot_structure_state(root, context.scene.cgn_connectivity_pairs)
        try:
            old_atom_count = len(instances) // 4
            instances = instances + new_instances
            new_atoms = _materialize_atoms(
                sites,
                cell_matrix,
                new_instances,
                initial_count + old_atom_count,
            )
            # Re-derive the full atom list once and make the per-element atom
            # objects reflect it.  A grow must not spawn a fresh CGN_Atoms_*
            # object per element per shell (the outliner would accumulate one
            # per grow); instead rebuild the single per-element objects in
            # place so the outliner stays flat while growth still works.
            atoms = (
                _cached_atoms_for_growth(
                    root,
                    context.scene.cgn_connectivity_pairs,
                    old_atom_count,
                    new_atoms,
                )
                if old_atom_count > 0
                else None
            )
            if atoms is None:
                atoms = _materialize_atoms(sites, cell_matrix, instances, initial_count)
            _rebuild_atom_objects(root, atoms)
            _update_growth_selection(root, [], [])
            # Option B: growing always APPENDS bonds for the newly grown atoms
            # and never tears down bonds that were built under earlier
            # connectivity settings.  The current pair toggles act as a query
            # filter for this growth only; re-deriving all bonds from them is
            # the explicit "Rebuild Bonds" operator.
            if _current_model(context.scene) != "SPACE_FILLING":
                _bond_objects(
                    root,
                    atoms,
                    context.scene.cgn_connectivity_pairs,
                    start_index=old_atom_count,
                    append_only=True,
                )
            _set_instances(root, instances)
            root["cgn_growth_shells"] = int(root.get("cgn_growth_shells", 0)) + 1
        except Exception as exc:
            traceback.print_exc()
            # A mid-rebuild failure must not leave the displayed atoms
            # inconsistent with the stored data: restore the pre-operation
            # state atomically.
            _rollback_structure_state(
                root,
                snapshot,
                context.scene,
                context.scene.cgn_connectivity_pairs,
            )
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Added {grew_by} atoms; structure now contains {len(instances) // 4} atoms.",
        )
        return {"FINISHED"}


def _build_polyhedra_objects(
    root, atoms, center_indices, scene, clear_existing=True, extend_structure=True
):
    """Build one Geometry Nodes polyhedron object per crystallographic site.

    Centers are grouped by ``p1_index`` so a single canonical prototype is
    built per site.  Each object holds a points scaffold (one vertex per image)
    and instances the prototype with the ``CGN_GN_Polyhedra`` node group, which
    keeps the outliner flat (one entry per site, not one per image) while
    storing the face/edge geometry only once per site.

    When ``clear_existing`` is True (default), existing polyhedra of the
    element(s) being built are removed first — other elements are left
    untouched.  When False, new images of an already-built site are appended to
    its scaffold, which is O(new images): the prototype is shared.

    When ``extend_structure`` is False, newly discovered ligand images are
    not added to the instance table — used by ``_remap_polyhedra`` after
    atom deletion, where the structure is already correct.
    """
    if scene is None:
        scene = bpy.context.scene
    search = _polyhedron_search(root, scene)
    if search is None or not search.pool or not search.windows:
        return 0, len(center_indices)

    if clear_existing:
        # Only replace the polyhedra of the element(s) being built: asking for
        # Cr polyhedra must not wipe the Ag ones.
        for element in {
            str(atoms[center].get("element", "X")).upper()
            for center in center_indices
            if 0 <= center < len(atoms)
        }:
            _clear_polyhedra(root, element)

    # Group centers by crystallographic site (p1_index).
    site_groups = {}  # p1_index -> list of atom indices
    site_rep = {}  # p1_index -> representative atom index (home cell first)
    for center in center_indices:
        atom = atoms[center]
        p1_index = atom.get("p1_index", center)
        site_groups.setdefault(p1_index, []).append(center)
        trans = tuple(atom.get("growth_translation", (0, 0, 0)))
        if p1_index not in site_rep or trans == (0, 0, 0):
            site_rep[p1_index] = center

    # Existing site objects.  When adding (clear_existing=False) the new images
    # of an existing site are appended to its scaffold, otherwise a freshly
    # grown image could never get a polyhedron ("Create from Selected" reported
    # a bogus skip).  A scaffold whose prototype vanished is rebuilt instead.
    existing_sites = {}
    if not clear_existing:
        for obj in list(root.objects):
            if obj.get("cgn_role") != "polyhedron":
                continue
            p1 = obj.get("cgn_p1_index")
            if p1 is None:
                continue
            if _polyhedron_prototype_for(obj) is None:
                _remove_object(obj)
                continue
            try:
                previous = [
                    int(index)
                    for index in json.loads(obj.get("cgn_poly_centers", "[]"))
                ]
            except (TypeError, ValueError):
                previous = []
            existing_sites[int(p1)] = (obj, previous)

    new_ligand_keys = set()
    new_ligand_quads = []
    built_total = 0
    skipped_total = 0

    for p1_index, centers in site_groups.items():
        p1_index = int(p1_index)
        existing = existing_sites.get(p1_index)
        previous = []
        if existing is not None:
            existing_obj, previous = existing
            valid_previous = [index for index in previous if 0 <= index < len(atoms)]
            merged = list(dict.fromkeys(valid_previous + centers))
            if len(merged) == len(valid_previous):
                # Every requested image is already represented by the scaffold.
                skipped_total += len(centers)
                continue
            known = set(valid_previous)
            centers_to_add = [center for center in merged if center not in known]
            previous = valid_previous
        else:
            existing_obj = None
            merged = list(dict.fromkeys(centers))
            centers_to_add = merged

        rep_center = site_rep[p1_index]
        rep_atom = atoms[rep_center]
        element = str(rep_atom.get("element", "X")).upper()

        # Canonical site label: strip the [P1:...] translation suffix so all
        # images of the same site share one style and one panel entry.
        raw_label = str(
            rep_atom.get("source_label", rep_atom.get("label", "")) or ""
        ).strip()
        site_label = raw_label.split(" [")[0] if " [" in raw_label else raw_label

        style = _polyhedron_style(scene, element, site_label)

        # Canonical geometry at the representative (home-cell) center.
        center_cart = rep_atom["cart"]
        ligands = search.ligands(center_cart, element)
        if len(ligands) < _POLY_MIN_LIGANDS:
            skipped_total += len(centers_to_add)
            continue

        # Collect ligand quads for the images being added so the structure can
        # be extended with them.  The coordination search runs once around the
        # representative image; every other image of the same site has the same
        # shell shifted by its integer cell translation.  Translating the quads
        # (and not just using the representative's) is what makes a box-selected
        # batch complete the edge atoms of *every* polyhedron.
        rep_translation = tuple(
            int(value) for value in rep_atom.get("growth_translation", (0, 0, 0))
        )
        for center in centers_to_add:
            translation = tuple(
                int(value)
                for value in atoms[center].get("growth_translation", (0, 0, 0))
            )
            delta = tuple(
                translation[axis] - rep_translation[axis] for axis in range(3)
            )
            for _pos, site_index, (tx, ty, tz) in ligands:
                quad = (
                    int(site_index),
                    int(tx) + delta[0],
                    int(ty) + delta[1],
                    int(tz) + delta[2],
                )
                if quad not in new_ligand_keys:
                    new_ligand_keys.add(quad)
                    new_ligand_quads.append(quad)

        if existing_obj is not None:
            # Growing an existing site only appends one vertex per new image;
            # the canonical prototype (and its materials) is already there.
            _append_polyhedron_points(
                existing_obj,
                [_display_point(atoms[center]["cart"]) for center in centers_to_add],
            )
            existing_obj["cgn_poly_centers"] = json.dumps(
                [int(center) for center in merged]
            )
            existing_obj["cgn_poly_map"] = json.dumps(
                {str(int(center)): [] for center in merged}
            )
            built_total += 1
            continue

        topology = _polyhedron_topology([item[0] for item in ligands])
        if topology is None:
            skipped_total += len(centers_to_add)
            continue
        local_vertices, _kept, local_faces, local_edges = topology

        # Shift canonical vertices so the site centre is at the origin.
        # ``local_vertices`` are already in display units (the ligand search
        # returns ``_display_point`` positions), so only the display-space
        # centre may be subtracted here — no second scale conversion.
        cx, cy, cz = _display_point(center_cart)
        local = [(v[0] - cx, v[1] - cy, v[2] - cz) for v in local_vertices]

        _face, _edge, thickness = _polyhedron_style_values(style)
        face_material = _make_polyhedron_face_material(element, style, site_label)
        edge_material = _make_polyhedron_edge_material(element, style, site_label)

        prototype = bpy.data.objects.new(
            f"CGN_Polyhedron_Prototype_{element}_P1{p1_index}",
            _make_polyhedron_prototype_mesh(
                f"CGN_Polyhedron_Prototype_{element}_P1{p1_index}",
                local,
                local_faces,
                local_edges,
                thickness,
                face_material,
                edge_material,
            ),
        )
        prototype["cgn_role"] = "polyhedron_prototype"
        prototype["cgn_poly_element"] = element
        prototype["cgn_poly_site_label"] = site_label
        prototype["cgn_p1_index"] = p1_index
        prototype["cgn_poly_edge_width"] = float(thickness)
        prototype["cgn_poly_local_vertices"] = json.dumps(
            [list(vertex) for vertex in local]
        )
        prototype["cgn_poly_local_faces"] = json.dumps(
            [list(loop) for loop in local_faces]
        )
        prototype["cgn_poly_local_edges"] = json.dumps(
            [list(edge) for edge in local_edges]
        )
        _polyhedron_prototype_collection(root).objects.link(prototype)

        # The scaffold is a points-only mesh: one vertex per image.  The node
        # group instances the prototype on those points and realizes the
        # instances so the object stays editable and exports without special
        # cases.
        mesh = bpy.data.meshes.new(f"CGN_Polyhedron_{element}_P1{p1_index}_Points")
        mesh.from_pydata(
            [_display_point(atoms[center]["cart"]) for center in merged], [], []
        )
        mesh.update()

        obj = bpy.data.objects.new(f"CGN_Polyhedron_{element}_P1{p1_index}", mesh)
        obj["cgn_role"] = "polyhedron"
        obj["cgn_poly_element"] = element
        obj["cgn_poly_site_label"] = site_label
        obj["cgn_p1_index"] = p1_index
        obj["cgn_poly_centers"] = json.dumps([int(center) for center in merged])
        obj["cgn_poly_map"] = json.dumps({str(int(center)): [] for center in merged})
        obj["cgn_poly_edge_width"] = float(thickness)
        obj["cgn_poly_prototype"] = prototype.name
        root.objects.link(obj)

        modifier = obj.modifiers.new("Geometry Nodes", "NODES")
        modifier.node_group = _make_polyhedra_node_group()
        _set_modifier_input(modifier, modifier.node_group, "Prototype", prototype)

        built_total += 1

    # Add the ligand atoms (polyhedron vertices) to the displayed structure
    # so the polyhedron edges connect to real atoms.
    bonds_rebuilt = False
    if new_ligand_quads and extend_structure:
        try:
            sites, instances, cell_matrix, initial_count = _structure_tables(root)
        except (ValueError, TypeError):
            sites, instances, cell_matrix, initial_count = None, None, None, 0
        if sites is not None:
            # _structure_tables returns shared (cached) lists; copy before the
            # extend below so the parsed-tables cache is not mutated in place.
            instances = list(instances)
            existing = set()
            count = len(instances) - len(instances) % 4
            for offset in range(0, count, 4):
                existing.add(
                    (
                        int(instances[offset]),
                        int(instances[offset + 1]),
                        int(instances[offset + 2]),
                        int(instances[offset + 3]),
                    )
                )
            added = False
            for quad in new_ligand_quads:
                if quad in existing:
                    continue
                instances.extend(quad)
                existing.add(quad)
                added = True
            if added:
                atoms = _materialize_atoms(sites, cell_matrix, instances, initial_count)
                try:
                    _rebuild_atom_objects(root, atoms)
                    _update_growth_selection(root, atoms, [])
                    _bond_objects(root, atoms, scene.cgn_connectivity_pairs)
                    bonds_rebuilt = True
                    _apply_unit_cell_settings(scene)
                except Exception:
                    traceback.print_exc()
                _set_instances(root, instances)
                root["cgn_initial_count"] = initial_count

    # A selected atom whose ligand atoms are already displayed adds no new
    # instance, so the extension branch above does not touch the bonds.  The
    # polyhedron vertices still need centre–ligand bonds (same result as the
    # full "Create Polyhedra" path, which always rebuilds them while
    # extending), so redraw them here.
    if built_total > 0 and extend_structure and not bonds_rebuilt:
        try:
            current_atoms = _structure_atoms(root)
            _bond_objects(root, current_atoms, scene.cgn_connectivity_pairs)
        except (ValueError, TypeError):
            traceback.print_exc()
    return built_total, skipped_total


class CGN_OT_create_polyhedra(Operator):
    bl_idname = "cgn.create_polyhedra"
    bl_label = "Create Polyhedra"
    bl_description = (
        "Build convex polyhedra around every atom of the selected element "
        "that has four or more bonded neighbours"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def execute(self, context):
        scene = context.scene
        root = _active_root(scene)
        if root is None:
            self.report({"ERROR"}, _no_structure_message(context.scene, "creating polyhedra"))
            return {"CANCELLED"}
        try:
            atoms = _structure_atoms(root)
        except (ValueError, TypeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        element = str(getattr(scene.cgn_polyhedron_settings, "element", ""))
        if not element:
            self.report(
                {"ERROR"},
                "No eligible element is available; enable connectivity "
                "windows so an element gains four or more neighbours.",
            )
            return {"CANCELLED"}
        # All matching atoms are centers; _build_polyhedra_objects groups them
        # by crystallographic site (p1_index) and bakes one polyhedron object
        # per site covering every corresponding atom image.
        centers = [
            index
            for index, atom in enumerate(atoms)
            if str(atom.get("element", "X")).upper() == element
        ]
        if not centers:
            self.report({"ERROR"}, f"The structure contains no {element} atoms.")
            return {"CANCELLED"}
        try:
            built, skipped = _build_polyhedra_objects(root, atoms, centers, scene)
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if built == 0:
            self.report(
                {"ERROR"},
                f"No {element} atom has four or more bonded neighbours to "
                "enclose a polyhedron.",
            )
            return {"CANCELLED"}
        message = f"Built {built} {element} polyhedra."
        if skipped:
            message += f" {skipped} site(s) skipped (too few ligands or planar)."
        self.report({"INFO"}, message)
        return {"FINISHED"}


class CGN_OT_create_polyhedra_selected(_CGN_OT_select_atoms):
    bl_idname = "cgn.create_polyhedra_selected"
    bl_label = "Create from Selected"
    bl_description = (
        "Select atoms in the viewport, then confirm to build polyhedra "
        "around them and their connected ligands"
    )
    bl_options = {"REGISTER", "UNDO"}

    select_action = "selecting polyhedron centers"
    status_text = (
        "Click atoms to add/remove, drag a box to select; Shift adds. "
        "Enter/right-click creates the polyhedra, Esc cancels."
    )
    keep_stored_selection = False
    keep_selection_sphere = False

    def _confirm(self, context):
        if not self._selected:
            self._finish(context)
            self.report({"ERROR"}, "Select one or more atoms before building.")
            return {"CANCELLED"}
        root = bpy.data.collections.get(self._root_name)
        if root is None or root.get("cgn_instance_id") != self._root_id:
            self._finish(context)
            self.report({"ERROR"}, "The active structure changed during selection.")
            return {"CANCELLED"}
        try:
            atoms = _structure_atoms(root)
        except (ValueError, TypeError) as exc:
            self._finish(context)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        centers = sorted(
            {
                index
                for index in self._selected
                if isinstance(index, int) and 0 <= index < len(atoms)
            }
        )
        self._finish(context)
        try:
            built, skipped = _build_polyhedra_objects(
                root,
                atoms,
                centers,
                context.scene,
                clear_existing=False,
            )
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        # The viewport highlight wrote the transient selection into the
        # growth-selection storage; restore what was there before the modal.
        try:
            _update_growth_selection(root, atoms, self._initial_selection)
        except Exception:
            traceback.print_exc()
        if built == 0:
            self.report(
                {"ERROR"},
                "Skipped every selected atom: its site already has a "
                "polyhedron covering it, or the atom has fewer than four "
                "bonded neighbours / a planar coordination.",
            )
            return {"CANCELLED"}
        message = f"Built {built} polyhedra."
        if skipped:
            message += f" {skipped} selected atom(s) skipped."
        self.report({"INFO"}, message)
        return {"FINISHED"}


class CGN_OT_clear_polyhedra(Operator):
    bl_idname = "cgn.clear_polyhedra"
    bl_label = "Clear Polyhedra"
    bl_description = "Remove polyhedra faces and edges (all, or one element)"
    bl_options = {"REGISTER", "UNDO"}

    element: StringProperty(default="", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        root = _active_root(context.scene)
        return root is not None and any(
            obj.get("cgn_role") in _POLY_ROLE_ROLES for obj in root.objects
        )

    def execute(self, context):
        root = _active_root(context.scene)
        if root is None:
            self.report({"ERROR"}, _no_structure_message(context.scene, "clearing polyhedra"))
            return {"CANCELLED"}
        target = str(self.element).upper() if self.element else None
        _clear_polyhedra(root, target)
        if target:
            self.report({"INFO"}, f"Removed {target} polyhedra.")
        else:
            self.report({"INFO"}, "Removed every polyhedron.")
        return {"FINISHED"}


def _format_cell_parameter(value, cif_value="", angle=False):
    """Keep CIF standard uncertainties visible and avoid trailing angle zeros."""
    token = str(cif_value or "").strip()
    if token not in {"", ".", "?"} and _CIF_CELL_VALUE_RE.fullmatch(token):
        if angle and "(" not in token:
            numeric = float(token)
            if numeric.is_integer():
                return str(int(numeric))
        return token
    numeric = float(value)
    if angle and math.isclose(numeric, round(numeric), abs_tol=1e-9):
        return str(int(round(numeric)))
    return f"{numeric:.3f}" if angle else f"{numeric:.4f}"


class CGN_PT_panel(Panel):
    bl_label = "CrystalViz"
    bl_idname = "CGN_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CrystalViz"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        layout.operator("cgn.import_cif", icon="IMPORT")
        root = _active_root(context.scene)
        if root is None:
            unlinked = _unlinked_active_structure(context.scene)
            layout.label(
                text=(
                    f"'{unlinked.name}' is not linked to this scene."
                    if unlinked is not None
                    else "Import a CIF to begin."
                )
            )
            return

        cif_path = root.get("cgn_cif_path", "")
        filename = os.path.splitext(os.path.basename(cif_path))[0]
        if filename:
            layout.label(text=filename)
        layout.label(text=f"Space group: {root.get('cgn_space_group', 'unknown')}")
        try:
            parameters = json.loads(root.get("cgn_unit_cell", "[]"))
        except (ValueError, TypeError):
            parameters = []
        try:
            cell_display = json.loads(root.get("cgn_unit_cell_display", "[]"))
        except (ValueError, TypeError):
            cell_display = []
        if len(parameters) >= 6:
            a, b, c, alpha, beta, gamma = (float(value) for value in parameters[:6])
            displayed = [
                _format_cell_parameter(
                    value,
                    cell_display[index] if index < len(cell_display) else "",
                    angle=index >= 3,
                )
                for index, value in enumerate((a, b, c, alpha, beta, gamma))
            ]
            grid = layout.grid_flow(
                row_major=True, columns=2, even_columns=True, align=True
            )
            grid.label(text=f"a = {displayed[0]} Å")
            grid.label(text=f"α = {displayed[3]}°")
            grid.label(text=f"b = {displayed[1]} Å")
            grid.label(text=f"β = {displayed[4]}°")
            grid.label(text=f"c = {displayed[2]} Å")
            grid.label(text=f"γ = {displayed[5]}°")
        has_count = root.get("cgn_atom_count", None) is not None
        if has_count or root.get("cgn_instances", ""):
            layout.label(
                text=f"Atoms: {_structure_atom_count(root)}",
                icon="MESH_UVSPHERE",
            )
        else:
            layout.label(
                text="Re-import this CIF (old structure format).",
                icon="ERROR",
            )


class CGN_PT_build(Panel):
    bl_label = "Build"
    bl_idname = "CGN_PT_build"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CrystalViz"
    bl_parent_id = "CGN_PT_panel"
    bl_order = 1

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def draw(self, context):
        layout = self.layout
        layout.menu("CGN_MT_fill", text="Fill", icon="MESH_GRID")
        layout.operator(
            "cgn.grow_further", text="Grow Structure", icon="ADD"
        ).all_atoms = True
        layout.operator(
            "cgn.select_growth_atoms",
            text="Grow Selected Atoms",
            icon="RESTRICT_SELECT_OFF",
        )
        layout.separator()
        layout.operator("cgn.delete_atoms", text="Delete Atoms", icon="X")
        layout.separator()
        layout.prop(
            context.scene,
            "cgn_show_hydrogens",
            text="Show Hydrogens",
            icon="HIDE_OFF" if context.scene.cgn_show_hydrogens else "HIDE_ON",
        )


class CGN_PT_connectivity(Panel):
    bl_label = "Connectivity"
    bl_idname = "CGN_PT_connectivity"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CrystalViz"
    bl_parent_id = "CGN_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
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
        layout.operator(
            "cgn.rebuild_connectivity", text="Reset Default", icon="LOOP_BACK"
        )


class CGN_PT_unit_cell(Panel):
    bl_label = "Unit Cell"
    bl_idname = "CGN_PT_unit_cell"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CrystalViz"
    bl_parent_id = "CGN_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def draw(self, context):
        layout = self.layout
        settings = context.scene.cgn_unit_cell_settings
        layout.prop(settings, "show_unit_cell", text="Show Unit Cell", icon="MESH_CUBE")
        layout.operator(
            "cgn.pack_cell_boxes",
            text="Pack Unit-Cells (Faces & Edges)...",
            icon="MESH_CUBE",
        )
        if not settings.show_unit_cell:
            return
        column = layout.column(align=True)
        column.prop(settings, "face_color", text="Face Color")
        column.prop(settings, "face_transparency", text="Face Transparency")
        column.prop(settings, "edge_color", text="Edge Color")
        column.prop(settings, "edge_thickness", text="Edge Thickness")


class CGN_PT_model_style(Panel):
    bl_label = "Atom Model & Style"
    bl_idname = "CGN_PT_model_style"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CrystalViz"
    bl_parent_id = "CGN_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "cgn_model", text="Model")
        if scene.cgn_model == "WIRE_STICKS":
            layout.prop(
                scene,
                "cgn_wire_sticks_radius",
                text="Atom & Bond Radius (Å)",
                slider=True,
            )
        if scene.cgn_model == "ELLIPSOID":
            layout.prop(
                scene,
                "cgn_ellipsoid_probability",
                text="Probability (%)",
                slider=True,
            )
            ortep_box = layout.box()
            ortep_box.label(text="ORTEP Cutaway")
            ortep_box.prop(scene, "cgn_ortep_inner_color", text="Inner Faces")
            ortep_box.prop(scene, "cgn_ortep_inner_roughness", text="Inner Roughness")
            ortep_box.prop(scene, "cgn_ortep_ellipse_color", text="Ellipses")
            ortep_box.prop(
                scene, "cgn_ortep_ellipse_roughness", text="Ellipse Roughness"
            )
            ortep_box.prop(
                scene,
                "cgn_ortep_ellipse_thickness",
                text="Ellipse Thickness",
                slider=True,
            )
        if scene.cgn_model == "ELLIPSOID":
            layout.prop(
                scene,
                "cgn_hydrogen_scale",
                text="Hydrogen Size",
                slider=True,
            )
        elements = sorted(
            {
                str(obj.get("cgn_element", "")).upper()
                for obj in _active_root(scene).objects
                if obj.get("cgn_role") == "atoms"
            }
        )
        if elements:
            box = layout.box()
            box.label(text="Atom Appearance")
            for element in elements:
                style = _existing_atom_style(scene, element)
                if style is None:
                    # No data mutation during draw; styles are created on
                    # import / bond rebuild. Hint instead of drawing controls.
                    box.label(
                        text=f"{element}: import or rebuild bonds to configure.",
                        icon="INFO",
                    )
                    continue
                row = box.row(align=True)
                row.label(text=element)
                row.prop(style, "color", text="")
                row.prop(style, "roughness", text="Roughness")
                if scene.cgn_model == "BALL_STICK":
                    box.prop(
                        style,
                        "size",
                        text=f"{element} Size (× VdW)",
                        slider=True,
                    )
        if scene.cgn_model == "SPACE_FILLING":
            layout.label(
                text="Full van der Waals radii; bonds are hidden.",
                icon="INFO",
            )


class CGN_PT_bond_style(Panel):
    bl_label = "Bond Style"
    bl_idname = "CGN_PT_bond_style"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CrystalViz"
    bl_parent_id = "CGN_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "cgn_bond_style", text="Style")
        if scene.cgn_bond_style == "SINGLE_COLOR":
            layout.prop(scene, "cgn_bond_color", text="Color")
        if scene.cgn_model == "WIRE_STICKS":
            layout.label(
                text="Bond radius follows the Wire-sticks atom radius.",
                icon="INFO",
            )
        elif scene.cgn_model != "SPACE_FILLING":
            layout.prop(
                scene,
                "cgn_bond_radius_scale",
                text="Bond Radius",
                slider=True,
            )
        layout.prop(scene, "cgn_chemdraw", text="ChemDraw")
        if scene.cgn_chemdraw:
            if scene.cgn_model != "SPACE_FILLING":
                layout.prop(
                    scene,
                    "cgn_multiple_bond_radius_scale",
                    text="Multiple Bond Radius",
                    slider=True,
                )
            layout.prop(
                scene,
                "cgn_bond_spacing",
                text="Bond Spacing",
                slider=True,
            )
            layout.prop(scene, "cgn_aromatic_torus_size", slider=True)
            layout.prop(scene, "cgn_aromatic_torus_width", slider=True)
            layout.prop(scene, "cgn_aromatic_torus_color", text="Torus Color")


class CGN_PT_polyhedra(Panel):
    bl_label = "Polyhedra"
    bl_idname = "CGN_PT_polyhedra"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CrystalViz"
    bl_parent_id = "CGN_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 6

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        root = _active_root(scene)
        settings = scene.cgn_polyhedron_settings
        col = layout.column(align=True)
        col.prop(settings, "element", text="Element")
        col.operator("cgn.create_polyhedra", text="Create Polyhedra", icon="MESH_GRID")
        col.operator(
            "cgn.create_polyhedra_selected",
            text="Create from Selected",
            icon="RESTRICT_SELECT_OFF",
        )
        # Per-element delete buttons
        present_elements = sorted(
            {
                str(obj.get("cgn_poly_element", "")).upper()
                for obj in root.objects
                if obj.get("cgn_role") in _POLY_ROLE_ROLES
                and str(obj.get("cgn_poly_element", "")).strip()
            }
        )
        if present_elements:
            box = layout.box()
            box.label(text="Delete polyhedra")
            for element in present_elements:
                op = box.operator(
                    "cgn.clear_polyhedra",
                    text=f"Delete {element} polyhedra",
                    icon="X",
                )
                op.element = element
        layout.operator("cgn.clear_polyhedra", text="Clear All Polyhedra", icon="TRASH")
        self._draw_styles(layout, scene, root)

    def _draw_styles(self, layout, scene, root):
        # Collect the styles we actually own: one per (element, site_label)
        # that has at least one built polyhedron object, plus the set of
        # element-level defaults for sites that have no per-site style yet.
        present = set()
        for obj in root.objects if root is not None else []:
            if obj.get("cgn_role") not in _POLY_ROLE_ROLES:
                continue
            el = str(obj.get("cgn_poly_element", "")).upper()
            site = str(obj.get("cgn_poly_site_label", ""))
            if el:
                present.add((el, site))
        shown = False
        for element, site_label in sorted(present):
            style = _polyhedron_style(scene, element, site_label)
            shown = True
            title = f"{element} · {site_label}" if site_label else element
            box = layout.box()
            head = box.row()
            head.prop(
                style,
                "expanded",
                text="",
                emboss=False,
                icon="DOWNARROW_HLT" if style.expanded else "RIGHTARROW",
            )
            head.label(text=f"{title} polyhedron")
            if style.expanded:
                box.prop(style, "face_color", text="Face")
                box.prop(
                    style, "face_transparency", text="Face Transparency", slider=True
                )
                box.prop(style, "edge_color", text="Edge")
                box.prop(
                    style, "edge_transparency", text="Edge Transparency", slider=True
                )
                box.prop(style, "edge_width", text="Edge Width", slider=True)
        if not shown:
            layout.label(
                text="Create polyhedra above to configure their style.",
                icon="INFO",
            )


class CGN_PT_quality(Panel):
    bl_label = "Render Quality"
    bl_idname = "CGN_PT_quality"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CrystalViz"
    bl_parent_id = "CGN_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 7

    @classmethod
    def poll(cls, context):
        return _active_root(context.scene) is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "cgn_quality_preset", text="Preset")
        if scene.cgn_quality_preset == "CUSTOM":
            layout.prop(scene, "cgn_atom_subdivisions", text="Atom Subdivisions")
            layout.prop(scene, "cgn_bond_resolution", text="Bond Resolution")


class CGN_MT_viewport(Menu):
    bl_idname = "CGN_MT_viewport"
    bl_label = "CrystalViz"

    def draw(self, context):
        layout = self.layout
        layout.operator("cgn.import_cif", text="Import CIF...", icon="IMPORT")
        if _active_root(context.scene) is None:
            return
        layout.separator()
        layout.menu("CGN_MT_fill", text="Fill", icon="MESH_GRID")
        layout.operator(
            "cgn.grow_further", text="Grow Structure", icon="ADD"
        ).all_atoms = True
        layout.operator(
            "cgn.select_growth_atoms",
            text="Grow Selected Atoms",
            icon="RESTRICT_SELECT_OFF",
        )
        layout.operator("cgn.delete_atoms", text="Delete Atoms", icon="X")
        layout.separator()
        layout.operator(
            "cgn.rebuild_connectivity", text="Reset Default", icon="LOOP_BACK"
        )


def _draw_viewport_menu(self, context):
    self.layout.menu("CGN_MT_viewport")


def _draw_file_import(self, context):
    self.layout.operator("cgn.import_cif", text="CIF (.cif)", icon="IMPORT")


_CLASSES = (
    CGN_AddonPreferences,
    CGN_ConnectivityPair,
    CGN_AtomStyle,
    CGN_UnitCellSettings,
    CGN_PolyhedronSettings,
    CGN_PolyhedronStyle,
    CGN_OT_import_cif,
    CGN_OT_rebuild_connectivity,
    CGN_OT_fill_asu,
    CGN_OT_fill_cell,
    CGN_OT_fill_pack,
    CGN_OT_pack_cell_boxes,
    CGN_OT_select_growth_atoms,
    CGN_OT_delete_atoms,
    CGN_OT_grow_further,
    CGN_OT_create_polyhedra,
    CGN_OT_create_polyhedra_selected,
    CGN_OT_clear_polyhedra,
    CGN_MT_fill,
    CGN_MT_viewport,
    CGN_PT_panel,
    CGN_PT_build,
    CGN_PT_connectivity,
    CGN_PT_unit_cell,
    CGN_PT_model_style,
    CGN_PT_bond_style,
    CGN_PT_polyhedra,
    CGN_PT_quality,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.cgn_connectivity_pairs = CollectionProperty(
        type=CGN_ConnectivityPair
    )
    bpy.types.Scene.cgn_atom_styles = CollectionProperty(type=CGN_AtomStyle)
    bpy.types.Scene.cgn_active_structure = StringProperty(options={"HIDDEN"})
    bpy.types.Scene.cgn_show_hydrogens = BoolProperty(
        name="Show Hydrogens",
        description="Show hydrogen atoms and their bonds",
        default=True,
        update=_update_show_hydrogens,
    )
    bpy.types.Scene.cgn_unit_cell_settings = PointerProperty(type=CGN_UnitCellSettings)
    bpy.types.Scene.cgn_polyhedron_settings = PointerProperty(
        type=CGN_PolyhedronSettings
    )
    bpy.types.Scene.cgn_polyhedron_styles = CollectionProperty(type=CGN_PolyhedronStyle)
    bpy.types.Scene.cgn_quality_preset = EnumProperty(
        name="Quality",
        description="Sphere and bond tube resolution preset",
        items=(
            ("LOW", "Low", "Icosphere 2, bond circle 8"),
            ("MEDIUM", "Medium", "Icosphere 3, bond circle 12"),
            ("HIGH", "High", "Icosphere 4, bond circle 16"),
            ("CUSTOM", "Custom", "Use the manual values below"),
        ),
        default="MEDIUM",
        update=_update_quality_preset,
    )
    bpy.types.Scene.cgn_atom_subdivisions = IntProperty(
        name="Atom Subdivisions",
        description="Icosphere subdivisions for atom spheres",
        default=3,
        min=1,
        max=6,
        update=_update_quality_manual,
    )
    bpy.types.Scene.cgn_bond_resolution = IntProperty(
        name="Bond Resolution",
        description="Circle resolution for bond tubes",
        default=12,
        min=4,
        max=32,
        update=_update_quality_manual,
    )
    bpy.types.Scene.cgn_model = EnumProperty(
        name="Model",
        description="Rendering model for atoms and bonds",
        items=_MODEL_ITEMS,
        default="BALL_STICK",
        update=_update_model,
    )
    bpy.types.Scene.cgn_bond_style = EnumProperty(
        name="Bond Style",
        description="Use one shared color or color each half by its bonded atom",
        items=(
            ("SINGLE_COLOR", "Single Color", "Use one color for every bond"),
            ("TWO_COLOR", "Two Colors", "Color each bond half by its atom"),
        ),
        default="SINGLE_COLOR",
        update=_update_bond_style,
    )
    bpy.types.Scene.cgn_chemdraw = BoolProperty(
        name="ChemDraw",
        description=("Show localized multiple bonds and delocalized aromatic rings"),
        default=False,
        update=_update_bond_style,
    )
    bpy.types.Scene.cgn_aromatic_torus_size = FloatProperty(
        name="Torus Size",
        description="Inner torus radius relative to the ring atom radius",
        default=_AROMATIC_TORUS_SIZE_DEFAULT,
        min=0.25,
        max=0.9,
        precision=2,
        update=_update_aromatic_torus_style,
    )
    bpy.types.Scene.cgn_aromatic_torus_width = FloatProperty(
        name="Torus Width",
        description="Torus tube radius relative to a normal bond radius",
        default=_AROMATIC_TORUS_WIDTH_DEFAULT,
        min=0.1,
        max=1.5,
        precision=2,
        update=_update_aromatic_torus_style,
    )
    bpy.types.Scene.cgn_aromatic_torus_color = FloatVectorProperty(
        name="Torus Color",
        description="Color for aromatic inner tori",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=_ELEMENT_COLORS["C"],
        update=_update_aromatic_torus_style,
    )
    bpy.types.Scene.cgn_bond_color = FloatVectorProperty(
        name="Bond Color",
        description="Color for bonds when using the single-color style",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=_BOND_COLOR_DEFAULT,
        update=_update_bond_color,
    )
    bpy.types.Scene.cgn_bond_radius_scale = FloatProperty(
        name="Bond Radius",
        description="Scale single-bond radii without changing multiple bonds",
        default=_BOND_RADIUS_SCALE_DEFAULT,
        min=0.1,
        max=2.0,
        precision=2,
        update=_update_bond_radius_scale,
    )
    bpy.types.Scene.cgn_multiple_bond_radius_scale = FloatProperty(
        name="Multiple Bond Radius",
        description="Scale the radius of double and triple bonds",
        default=_MULTIPLE_BOND_RADIUS_SCALE_DEFAULT,
        min=0.1,
        max=2.0,
        precision=2,
        update=_update_multiple_bond_radius_scale,
    )
    bpy.types.Scene.cgn_bond_spacing = FloatProperty(
        name="Bond Spacing",
        description="Distance between parallel strokes in multiple bonds",
        default=_CHEMDRAW_BOND_SPACING_DEFAULT,
        min=0.05,
        max=1.0,
        precision=2,
        update=_update_bond_style,
    )
    bpy.types.Scene.cgn_ellipsoid_probability = FloatProperty(
        name="Probability",
        description="Ellipsoid probability level in percent",
        default=50.0,
        min=0.0,
        max=100.0,
        update=_update_ellipsoid_probability,
    )
    bpy.types.Scene.cgn_hydrogen_scale = FloatProperty(
        name="Hydrogen Size",
        description=(
            "Scale hydrogen atoms in the ORTEP/ellipsoid model; "
            "space-filling always uses the full van der Waals radius"
        ),
        default=1.0,
        min=0.1,
        max=3.0,
        update=_update_hydrogen_scale,
    )
    bpy.types.Scene.cgn_wire_sticks_radius = FloatProperty(
        name="Atom & Bond Radius",
        description="Shared atom and bond radius in Wire-sticks mode",
        default=_WIRE_RADIUS_DEFAULT,
        min=0.01,
        max=0.5,
        precision=2,
        update=_update_wire_sticks_radius,
    )
    bpy.types.Scene.cgn_ortep_ellipse_thickness = FloatProperty(
        name="Ellipse Thickness",
        description="Tube thickness of the three ORTEP ellipses",
        default=_ORTEP_ELLIPSE_THICKNESS,
        min=0.003,
        max=0.1,
        precision=3,
        update=_update_ortep_ellipse_thickness,
    )
    bpy.types.Scene.cgn_ortep_inner_color = FloatVectorProperty(
        name="Inner Face Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=_ORTEP_INNER_COLOR,
        update=_update_ortep_materials,
    )
    bpy.types.Scene.cgn_ortep_inner_roughness = FloatProperty(
        name="Inner Face Roughness",
        default=0.25,
        min=0.0,
        max=1.0,
        update=_update_ortep_materials,
    )
    bpy.types.Scene.cgn_ortep_ellipse_color = FloatVectorProperty(
        name="Ellipse Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=_ORTEP_ELLIPSE_COLOR,
        update=_update_ortep_materials,
    )
    bpy.types.Scene.cgn_ortep_ellipse_roughness = FloatProperty(
        name="Ellipse Roughness",
        default=0.25,
        min=0.0,
        max=1.0,
        update=_update_ortep_materials,
    )
    bpy.types.VIEW3D_MT_editor_menus.append(_draw_viewport_menu)
    bpy.types.TOPBAR_MT_file_import.append(_draw_file_import)
    if _orient_ortep_for_render not in bpy.app.handlers.render_pre:
        bpy.app.handlers.render_pre.append(_orient_ortep_for_render)
    if _sync_ortep_view_timer_after_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_sync_ortep_view_timer_after_load)
    _sync_ortep_view_timer()


def unregister():
    _cancel_all_debounce()
    _invalidate_structure_cache()
    _stop_ortep_view_timer()
    if _orient_ortep_for_render in bpy.app.handlers.render_pre:
        bpy.app.handlers.render_pre.remove(_orient_ortep_for_render)
    if _sync_ortep_view_timer_after_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_sync_ortep_view_timer_after_load)
    bpy.types.TOPBAR_MT_file_import.remove(_draw_file_import)
    bpy.types.VIEW3D_MT_editor_menus.remove(_draw_viewport_menu)
    del bpy.types.Scene.cgn_ortep_ellipse_thickness
    del bpy.types.Scene.cgn_wire_sticks_radius
    del bpy.types.Scene.cgn_bond_radius_scale
    del bpy.types.Scene.cgn_multiple_bond_radius_scale
    del bpy.types.Scene.cgn_bond_spacing
    del bpy.types.Scene.cgn_bond_color
    del bpy.types.Scene.cgn_aromatic_torus_color
    del bpy.types.Scene.cgn_aromatic_torus_width
    del bpy.types.Scene.cgn_aromatic_torus_size
    del bpy.types.Scene.cgn_chemdraw
    del bpy.types.Scene.cgn_bond_style
    del bpy.types.Scene.cgn_hydrogen_scale
    del bpy.types.Scene.cgn_ortep_ellipse_roughness
    del bpy.types.Scene.cgn_ortep_ellipse_color
    del bpy.types.Scene.cgn_ortep_inner_roughness
    del bpy.types.Scene.cgn_ortep_inner_color
    del bpy.types.Scene.cgn_ellipsoid_probability
    del bpy.types.Scene.cgn_model
    del bpy.types.Scene.cgn_unit_cell_settings
    del bpy.types.Scene.cgn_polyhedron_settings
    del bpy.types.Scene.cgn_polyhedron_styles
    del bpy.types.Scene.cgn_quality_preset
    del bpy.types.Scene.cgn_atom_subdivisions
    del bpy.types.Scene.cgn_bond_resolution
    del bpy.types.Scene.cgn_show_hydrogens
    del bpy.types.Scene.cgn_active_structure
    del bpy.types.Scene.cgn_atom_styles
    del bpy.types.Scene.cgn_connectivity_pairs
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
