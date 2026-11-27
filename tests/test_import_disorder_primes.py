"""Import omits prime-labelled disorder components (AKUQAB 5-ring case).

AKUQAB.S_ring5.cif (CSD) models a thiophene ring in two orientations sharing
one exocyclic carbon; the second component is labelled with a trailing prime
(S1', C1', ... H4').  With both components loaded, the shared carbon and C1
each carry four non-metal neighbours, so the five-membered ring is never
enumerated and is drawn as a plain Kekulé ring (the "S_ring5_no_aromatic"
symptom).  The loader now omits the primed component, which restores the ring.

The CSD export is gitignored, so this test embeds a subset of the same atoms
(real fractional coordinates and cell) in a P1 cell.

The second check pins the ANUBUI.N_ring5.cif side of the same rule: a P 2 1
entry where the trailing primes (C1'', C3', O1'', ...) are just the author's
numbering of the second molecule.  Its bases never overlap -- C3/C3' sit a
full C-C bond (1.42 A) apart, the rest farther -- so no level is dropped:
primes are only alternate *positions* (drop them) when a base name appears at
two places that coincide (AKUQAB keeps C1/C1' 0.08 A apart).

Run:
  blender --background --factory-startup --python test_import_disorder_primes.py
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
data_akquab_subset
_cell_length_a 11.0315
_cell_length_b 9.8892
_cell_length_c 19.0275
_cell_angle_alpha 90.0
_cell_angle_beta 95.457
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
S1 S 0.3205 0.6842 0.66138
C1 C 0.3928 0.707 0.587
C2 C 0.1805 0.685 0.6092
H2 H 0.1048 0.6771 0.6289
C3 C 0.188 0.698 0.5401
H3 H 0.1217 0.6958 0.5045
C4 C 0.3132 0.715 0.5286
H4 H 0.3389 0.7299 0.483
S1' S 0.3084 0.728 0.508
C1' C 0.3976 0.706 0.584
C2' C 0.179 0.702 0.5513
H2' H 0.0993 0.7087 0.5276
C3' C 0.198 0.672 0.6202
H3' H 0.1381 0.651 0.6515
C4' C 0.3271 0.678 0.6363
H4' H 0.3628 0.6635 0.6831
C5 C 0.5257 0.7214 0.5927
C6 C 0.6072 0.7262 0.5454
O1 O 0.5749 0.7361 0.66156
"""

EXPECTED_ATOMS = 11  # 19 loop sites minus the eight primed alternates

# ANUBUI.N_ring5.cif: the same prime suffix used as a pure naming convention.
# Real cell (P 2 1 -> P1 here) and coordinates; every base's components are
# *distinct* atoms a bond length or more apart (C3/C3' 1.42 A, C1/C1" 5.78 A),
# so the block must keep every label.  No occupancy column, like ANUBUI.
CIF_TEXT_NAMING = """\
data_anubui_naming
_cell_length_a 13.8821
_cell_length_b 4.5502
_cell_length_c 14.6589
_cell_angle_alpha 90.0
_cell_angle_beta 109.8086
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.2008 0.7204 0.9550
C1" C 0.38484 0.6798 0.65562
C3 C -0.0345 1.0723 0.7078
C3' C -0.1329 1.1896 0.6941
O1 O 0.13648 0.5681 0.87114
O1" O 0.31018 0.8806 0.60287
H4 H 0.0184 1.3317 0.5492
H4" H 0.6331 0.7614 0.7531
N2' N -0.1756 1.1091 0.7584
"""

EXPECTED_NAMING_ATOMS = 9  # every loop site: primes are names, not disorder
FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def main():
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "akquab_subset.cif")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(CIF_TEXT)

    bpy.ops.cgn.import_cif(filepath=path)
    scene = bpy.context.scene
    scene.cgn_chemdraw = True
    root = bpy.data.collections.get(scene.cgn_active_structure)

    labels = [site["label"] for site in json.loads(root["cgn_sites"])]
    primed = [label for label in labels if label.endswith("'") or label.endswith('"')]
    check("no prime-labelled site survives the import", not primed, str(primed))
    check(
        "ASU holds only the first disorder component",
        CrystalViz._structure_atom_count(root) == EXPECTED_ATOMS,
        str(CrystalViz._structure_atom_count(root)),
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

    radii = {CrystalViz._bond_point_key(atom["cart"]): atom["covalent_radius"] for atom in atoms}
    orders = CrystalViz._chemdraw_bond_orders(segments, radii)
    elements, edges, _lengths, _points = CrystalViz._chemdraw_graph(segments, radii)
    rings = CrystalViz._chemistry().aromatic_rings(elements, edges, orders, detail=True)
    check(
        "the thiophene five-ring is perceived as aromatic",
        len(rings) == 1 and len(next(iter(rings)).nodes) == 5,
        str([len(ring.nodes) for ring in rings]),
    )

    ring_atoms = set(next(iter(rings)).nodes)
    ring_doubles = [
        edge
        for edge, order in orders.items()
        if int(order) == 2 and edge[0] in ring_atoms and edge[1] in ring_atoms
    ]
    check(
        "the Kekulé ring carries two double bonds", len(ring_doubles) == 2, str(len(ring_doubles))
    )

    # Naming-convention primes (ANUBUI): nothing overlaps any base, so the
    # import must keep every primed label instead of omitting it.
    naming_path = os.path.join(directory, "anubui_naming.cif")
    with open(naming_path, "w", encoding="utf-8") as handle:
        handle.write(CIF_TEXT_NAMING)
    bpy.ops.cgn.import_cif(filepath=naming_path)
    naming_root = bpy.data.collections.get(scene.cgn_active_structure)

    naming_labels = [site["label"] for site in json.loads(naming_root["cgn_sites"])]
    naming_counts = {label: naming_labels.count(label) for label in set(naming_labels)}
    check(
        "every primed site of a naming convention survives the import",
        len(naming_labels) == EXPECTED_NAMING_ATOMS,
        str(naming_counts),
    )
    kept_labels = set(naming_labels)
    check(
        "the naming primes are present verbatim",
        {'C1"', "C3'", 'O1"', 'H4"', "N2'"} <= kept_labels,
        str(sorted(kept_labels)),
    )
    check(
        "ASU holds every naming-convention atom",
        CrystalViz._structure_atom_count(naming_root) == EXPECTED_NAMING_ATOMS,
        str(CrystalViz._structure_atom_count(naming_root)),
    )

    CrystalViz.unregister()
    print()
    print("TOTAL FAILURES:", len(FAILURES))
    print("RESULT:", "FAIL" if FAILURES else "ALL PASS")
    sys.exit(1 if FAILURES else 0)


main()
