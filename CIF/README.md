# Example CIFs

Every file here is an entry from the
[Crystallography Open Database](https://www.crystallography.net/cod/) (COD).
The COD identifier is the one stored in `_cod_database_code`; it is also listed
below so the provenance of each fixture is checkable.

| File | COD ID | Formula | Used for |
| --- | --- | --- | --- |
| `Ag2Cr2O7.cif` | 1007030 | Ag2 Cr2 O7 | Polyhedra, grow, multi-pair connectivity (ionic Ag-O) |
| `Brookite.cif` | 8104269 | Ti O2 | TiO6 polyhedra, spoke masking, ligand deletion |
| `Chalco.cif` | 1010940 | Cu Fe S2 | Declared oxidation numbers, ionic Cu-S / Fe-S windows |
| `Cu.cif` | 4105040 | Cu | Elemental metal (wide Cu-Cu window) |
| `Cu2O.cif` | 1000063 | Cu2 O | Ionic oxide, metal-metal cap |
| `aromatic.cif` | 4025500 | C9 H9 N3 | Aromatic ChemDraw: Kekulé triazole (methyl+phenyl 1H-1,2,4-triazole) |
| `Diamond.cif` | 1541404 | C | Fill/bank guards (32 P1 sites per cell) |
| `Ferrocene.cif` | 1100887 | C26 H18 Fe | Organometallic finite molecule, grow after fill |
| `Graphite.cif` | 1200017 | C | Layered carbon, oxidation numbers |
| `multiple_bonds.cif` | 4029335 | C21 H17 N O4 | Organic molecule with ADPs and a bond table |
| `Nano.cif` | 4118445 | C120 H136 | Large structure (2048 P1 sites) for cache/perf tests |

`tests/test_gemmi_worker.py` pins the atom/site counts of `multiple_bonds.cif`,
`Ferrocene.cif`, `Brookite.cif` and `Nano.cif`; update that table (and
`tests/README.md` if a test scenario changes) when a fixture is replaced.

Some COD entries omit `_atom_type_oxidation_number`. The default connectivity
windows then fall back to covalent radii, with metal-metal pairs capped in
compounds (see `_default_pair_window` in `__init__.py`) so that, for example,
Ti(IV) is not bonded to Ti(IV) across a shared octahedron edge.
