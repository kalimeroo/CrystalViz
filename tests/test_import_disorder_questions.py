"""Import omits ``?``-labelled disorder components (PEPQAG 4-selenopyran case).

PEPQAG (CSD) models part of the molecule in two orientations; the alternate
component is numbered on its own lines and flagged with a trailing question
mark (``H2?`` sits 0.06 A from ``H1``, ``S6?`` 0.10 A from ``C31``, ``C35?``
0.21 A from ``S2``, ``C32?`` 0.36 A from ``S1``).  There is no shared base name
for the prime pairing to see, so ``_disorder_question_labels`` treats every
``?``-suffixed site as the marked level: one coincident pair fires the gate and
the whole marked set is omitted -- including the flip hydrogens ``H30?``-``H33?``
that sit 0.6-0.9 A from any plain atom.  The loader drops all five ``?`` labels
of the subset, leaving the first component only.

The CSD export is gitignored, so this test embeds a subset of the same atoms
(real fractional coordinates and cell) in a P1 cell, like the AKUQAB import
test does for primes.

The second check pins the unconditional side of the rule: a ``?`` label on a
*distinct* atom, a full bond length from everything else, is still ignored --
every site carrying a trailing ``?`` is an alternate component.

Run:
  blender --background --factory-startup --python test_import_disorder_questions.py
"""

import json
import os
import sys
import tempfile

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz

CrystalViz.register()

CIF_TEXT = """\
data_pepqag_subset
_cell_length_a 9.39445
_cell_length_b 14.0312
_cell_length_c 14.33664
_cell_angle_alpha 105.6795
_cell_angle_beta 91.5817
_cell_angle_gamma 95.5451
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C31 C 1.303 1.256 0.484
H29 H 1.24180 1.29160 0.45450
S2 S 1.4125 1.1220 0.5425
C1 C 1.4993 1.2313 0.5727
H1 H 1.58740 1.24640 0.61040
S1 S -0.3555 1.10656 0.05790
S6? S 1.3026 1.2623 0.4891
C35? C 1.399 1.120 0.530
H2? H 1.58510 1.24350 0.61260
C32? C -0.332 1.1246 0.0754
H33? H 1.41390 1.05570 0.53700
"""

EXPECTED_ATOMS = 6  # 11 loop sites minus the five ``?`` alternates

# Unconditional control: a ``?`` on a distinct atom (2.5 A from anything) with
# no plain partner anywhere is still an alternate component and is ignored.
CIF_TEXT_LONE = """\
data_question_lone
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.20 0.30 0.40
H2? H 0.45 0.30 0.40
"""

EXPECTED_LONE_ATOMS = 1  # only C1: every ``?`` label is dropped
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def main():
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "pepqag_subset.cif")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(CIF_TEXT)

    bpy.ops.cgn.import_cif(filepath=path)
    scene = bpy.context.scene
    root = bpy.data.collections.get(scene.cgn_active_structure)

    labels = [site["label"] for site in json.loads(root["cgn_sites"])]
    marked = [label for label in labels if label.endswith("?")]
    check("no question-labelled site survives the import", not marked, str(marked))
    check(
        "ASU holds only the first disorder component",
        CrystalViz._structure_atom_count(root) == EXPECTED_ATOMS,
        str(CrystalViz._structure_atom_count(root)),
    )
    kept_labels = set(labels)
    check(
        "the plain component is present verbatim",
        {"C31", "S2", "C1", "H1", "S1"} <= kept_labels,
        str(sorted(kept_labels)),
    )

    # Fill (asymmetric unit) then bond, like the reported sequence.
    bpy.ops.cgn.fill_asu()
    atoms = CrystalViz._structure_atoms(root)
    CrystalViz._default_connectivity_pairs(scene, atoms)
    CrystalViz._rebuild_bonds(scene)
    check("fill keeps the first disorder component", len(atoms) == EXPECTED_ATOMS, str(len(atoms)))

    root_id = root.get("cgn_instance_id", root.name)
    segments = CrystalViz._drawn_segments_for(root_id)
    check("the filled structure drew bonds", bool(segments), str(len(segments)))

    # "Keep All Disorder Components" (operator toggle): the same CIF keeps
    # every ?-marked component alongside the plain one -- the operator forwards
    # the toggle to load_structure, which skips the label-based omission
    # entirely.
    bpy.ops.cgn.import_cif(filepath=path, keep_disorder=True)
    keep_root = bpy.data.collections.get(scene.cgn_active_structure)
    keep_labels = [site["label"] for site in json.loads(keep_root["cgn_sites"])]
    kept_marked = {label for label in keep_labels if label.endswith("?")}
    check(
        "keep_disorder keeps every question-marked component",
        len(keep_labels) == 11 and kept_marked == {"S6?", "C35?", "H2?", "C32?", "H33?"},
        f"{len(keep_labels)} sites, marked={sorted(kept_marked)}",
    )
    keep_atoms = CrystalViz._structure_atom_count(keep_root)
    check(
        "keep_disorder ASU holds all 11 components",
        keep_atoms == 11,
        str(keep_atoms),
    )

    # Unconditional ``?`` (distinct atom, no coincidence anywhere): the marked
    # label is still ignored instead of surviving.
    lone_path = os.path.join(directory, "question_lone.cif")
    with open(lone_path, "w", encoding="utf-8") as handle:
        handle.write(CIF_TEXT_LONE)
    bpy.ops.cgn.import_cif(filepath=lone_path)
    lone_root = bpy.data.collections.get(scene.cgn_active_structure)

    lone_labels = [site["label"] for site in json.loads(lone_root["cgn_sites"])]
    check(
        "a lone question-marked atom is ignored",
        len(lone_labels) == EXPECTED_LONE_ATOMS and "H2?" not in set(lone_labels),
        str(lone_labels),
    )
    check(
        "ASU holds only the unmarked atom",
        CrystalViz._structure_atom_count(lone_root) == EXPECTED_LONE_ATOMS,
        str(CrystalViz._structure_atom_count(lone_root)),
    )

    CrystalViz.unregister()
    print()
    print("TOTAL FAILURES:", len(FAILURES))
    print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
    sys.exit(1 if FAILURES else 0)


main()
