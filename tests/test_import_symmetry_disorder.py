"""Import resolves space-group-generated disorder to one component.

``CIF/Ca.cif`` (local-only) showed the symptom: a *tert*-butyl group and a
methyl sit on a mirror plane, the CIF writes their atoms once at half
occupancy, and the P1 expansion therefore holds both orientations at once.
The load-time resolver keeps one chemically valid orientation per anchor,
promotes it to full occupancy and relocates the asymmetric unit onto it; the
default import must therefore give a single tetrahedral t-Bu while
"Keep All Disorder Components" keeps both.

The fixture is a synthetic ``P 1 2/m 1`` structure with the same defect (see
``_symmetry_disorder_fixture.py``) because the Ca CIF is gitignored.

Run:
  blender --background --factory-startup --python test_import_symmetry_disorder.py
"""

import json
import math
import os
import sys
import tempfile

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import _symmetry_disorder_fixture as fixture
import CrystalViz

CrystalViz.register()

METHYLS = ("C8", "C9", "C10")
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def label(atom):
    return atom.get("source_label", atom.get("label"))


def neighbours_by_point(segments):
    """Map ``_bond_point_key`` to the keys bonded to it."""
    lookup = {}
    for first, second, _element_a, _element_b in segments:
        lookup.setdefault(CrystalViz._bond_point_key(first), []).append(
            CrystalViz._bond_point_key(second)
        )
        lookup.setdefault(CrystalViz._bond_point_key(second), []).append(
            CrystalViz._bond_point_key(first)
        )
    return lookup


def main():
    path = fixture.write_cif(os.path.join(tempfile.mkdtemp(), "mirror_tbu.cif"))

    # ---------------------------------------------------------------- default
    bpy.ops.cgn.import_cif(filepath=path)
    scene = bpy.context.scene
    scene.cgn_chemdraw = True
    root = bpy.data.collections.get(scene.cgn_active_structure)

    labels = [site["label"] for site in json.loads(root["cgn_sites"])]
    check(
        "the import keeps one disorder component",
        len(labels) == 56,
        str(len(labels)),
    )
    check(
        "losing images were dropped from every methyl",
        all(labels.count(name) == 2 for name in METHYLS),
        str({name: labels.count(name) for name in METHYLS}),
    )
    check(
        "the half-occupied solvent was not resolved",
        all(labels.count(name) == 4 for name in ("K1", "O1")),
        str({name: labels.count(name) for name in ("K1", "O1")}),
    )

    bpy.ops.cgn.fill_asu()
    atoms = CrystalViz._structure_atoms(root)
    CrystalViz._default_connectivity_pairs(scene, atoms)
    CrystalViz._rebuild_bonds(scene)
    root_id = root.get("cgn_instance_id", root.name)
    segments = CrystalViz._drawn_segments_for(root_id)
    check("the filled structure drew bonds", bool(segments), str(len(segments)))

    key = CrystalViz._bond_point_key
    by_key = {key(atom["cart"]): atom for atom in atoms}
    neighbours = neighbours_by_point(segments)
    apexes = [atom for atom in atoms if label(atom) == "C7"]
    check("the asymmetric unit holds one t-Bu", len(apexes) == 1, str(len(apexes)))
    apex = apexes[0]
    carbons = [by_key[k] for k in neighbours.get(key(apex["cart"]), [])]
    carbons = [atom for atom in carbons if atom["element"] == "C"]
    methyls = [atom for atom in carbons if label(atom) in METHYLS]
    check(
        "the t-Bu carbon has four carbon neighbours",
        len(carbons) == 4,
        str(sorted(label(atom) for atom in carbons)),
    )
    check(
        "exactly one orientation of the three methyls is present",
        len(methyls) == 3 and len({label(atom) for atom in methyls}) == 3,
        str(sorted(label(atom) for atom in methyls)),
    )
    check(
        "the kept methyls are full occupancy",
        all(atom.get("occupancy") == 1.0 for atom in methyls),
        str([atom.get("occupancy") for atom in methyls]),
    )
    for methyl in methyls:
        hydrogens = [
            by_key[k]
            for k in neighbours.get(key(methyl["cart"]), [])
            if by_key[k]["element"] == "H"
        ]
        check(f"{label(methyl)} keeps three hydrogens", len(hydrogens) == 3, str(len(hydrogens)))
        check(
            f"{label(methyl)} is bonded to the t-Bu carbon",
            math.dist(methyl["cart"], apex["cart"]) < 1.7,
            f"{math.dist(methyl['cart'], apex['cart']):.3f} A",
        )
    for first in range(len(methyls)):
        for second in range(first + 1, len(methyls)):
            distance = math.dist(methyls[first]["cart"], methyls[second]["cart"])
            check(
                f"no methyl clash {label(methyls[first])}-{label(methyls[second])}",
                distance >= 2.3,
                f"{distance:.3f} A",
            )

    present = {label(atom) for atom in atoms}
    check(
        "the solvent atoms survived the import",
        {"K1", "O1"} <= present,
        str(sorted(label for label in present if label in ("K1", "O1"))),
    )

    radii = {key(atom["cart"]): atom["covalent_radius"] for atom in atoms}
    orders = CrystalViz._chemdraw_bond_orders(segments, radii)
    elements, edges, _lengths, _points = CrystalViz._chemdraw_graph(segments, radii)
    rings = CrystalViz._chemistry().aromatic_rings(elements, edges, orders, detail=True)
    check(
        "the benzene ring is still aromatic",
        len(rings) == 1 and len(next(iter(rings)).nodes) == 6,
        str([len(ring.nodes) for ring in rings]),
    )

    # ------------------------------------------------------------ keep them all
    bpy.ops.cgn.import_cif(filepath=path, keep_disorder=True)
    root = bpy.data.collections.get(scene.cgn_active_structure)
    labels = [site["label"] for site in json.loads(root["cgn_sites"])]
    check("keep-all keeps every image", len(labels) == 80, str(len(labels)))
    bpy.ops.cgn.fill_cell()
    atoms = CrystalViz._structure_atoms(root)
    CrystalViz._default_connectivity_pairs(scene, atoms)
    CrystalViz._rebuild_bonds(scene)
    segments = CrystalViz._drawn_segments_for(root.get("cgn_instance_id", root.name))
    by_key = {key(atom["cart"]): atom for atom in atoms}
    neighbours = neighbours_by_point(segments)
    cross = [
        (label(by_key[key(first)]), label(by_key[key(second)]))
        for first, second, _element_a, _element_b in segments
        if label(by_key[key(first)]) in METHYLS and label(by_key[key(second)]) in METHYLS
    ]
    check("alternate components are never bonded to each other", not cross, str(cross[:4]))
    apex = next(atom for atom in atoms if label(atom) == "C7")
    methyls = [
        by_key[k] for k in neighbours.get(key(apex["cart"]), []) if label(by_key[k]) in METHYLS
    ]
    check(
        "both orientations stay attached to the t-Bu carbon",
        len(methyls) == 6,
        str(len(methyls)),
    )
    components = {atom.get("component", 0) for atom in methyls}
    check(
        "the six methyls form two components",
        len(components) == 2 and 0 not in components,
        str(sorted(components)),
    )

    CrystalViz.unregister()
    print()
    print("TOTAL FAILURES:", len(FAILURES))
    print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
    sys.exit(1 if FAILURES else 0)


main()
