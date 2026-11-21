# Example CIFs

Most files here are entries from the
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
| `Cyanoferrocene.cif` | 2005233 | C11 H9 Fe N | Dummy `Cg` sites (`_atom_site_calc_flag` = `dum`) skipped; haptic Cp + C≡N |
| `aromatic.cif` | 4025500 | C9 H9 N3 | Aromatic ChemDraw: Kekulé triazole (methyl+phenyl 1H-1,2,4-triazole) |
| `BN.cif` | 1010602 | B N | Hexagonal boron nitride: grown sheet draws every closed B3N3 hexagon as a torus |
| `Diamond.cif` | 1541404 | C | Fill/bank guards (32 P1 sites per cell) |
| `Ferrocene.cif` | 1100887 | C26 H18 Fe | Organometallic finite molecule, grow after fill |
| `Graphite.cif` | 1200017 | C | Layered carbon, oxidation numbers |
| `multiple_bonds.cif` | 4029335 | C21 H17 N O4 | Organic molecule with ADPs and a bond table |
| `Nano.cif` | 4118445 | C120 H136 | Large structure (2048 P1 sites) for cache/perf tests |
| `Nitrobenzene.cif` | 5000155 | C6 H5 N O2 | Nitro group drawn N(=O)₂, aromatic ring |
| `Pyrazine_dioxide.cif` | 2201315 | C4 H4 N2 O2 | Two N-oxides: dative N-O bonds, aromatic ring |
| `Pyridine_N_oxide.cif` | 2105865 | C5 H5 N O | N-oxide: four-valent N, dative N-O, aromatic ring |

The heavy main-group fixtures extend the ChemDraw perception: selenium and
tellurium follow sulfur, arsenic and antimony follow phosphorus, and a
three-connected boron has an empty p orbital:

| File | COD ID | Formula | Used for |
| --- | --- | --- | --- |
| `Arsabenzene.cif` | 1573224 | C9 H5 As S2 | As=C ring double: arsenic ring torus, two Kekulé thiophenes |
| `Arsole.cif` | 1555319 | C22 H17 As | Five-ring As keeps its lone pair (Kekulé), three phenyl tori |
| `Borazine.cif` | 2216154 | C3 H9 B3 Cl3 N3 | B3N3 aromatic from the N lone pairs; no B=N is drawn, so both rings keep their torus |
| `Diazastibolene.cif` | 7111438 | C10 H20 Cl N2 Sb | Sb ring is found but its 8π electron count is not aromatic |
| `Selenochromenylium.cif` | 7243860 | C13 H15 B F4 Se | Se=C ring double, fused bicycle drawn as two tori |
| `Selenophene.cif` | 1553554 | C18 H15 I O Se | Five-ring Se keeps its lone pair (Kekulé), two phenyl tori |
| `Tellurochromenylium.cif` | 7243859 | C13 H15 B F4 Te | Te=C ring double, fused bicycle drawn as two tori |
| `Tellurophene.cif` | 1558536 | C18 H16 O2 Te | Five-ring Te keeps its lone pair (Kekulé), two phenyl tori |

The two sulfur fixtures are Cambridge Structural Database entries instead
(their CIFs carry `_database_code_CSD`):

| File | CSD refcode | CCDC | Formula | Used for |
| --- | --- | --- | --- | --- |
| `S1.cif` | ATBOXM | 1103749 | C9 H14 O3 S | Thiabenzene 1-oxide: S=O double, shallow-boat S ring drawn as an aromatic torus |
| `S2.cif` | BIZTAF | 1112440 | C23 H17 Cu2 I3 S | Thiopyrylium: S=C ring double, four aromatic tori |

`tests/test_gemmi_worker.py` pins the atom/site counts of `multiple_bonds.cif`,
`Ferrocene.cif`, `Brookite.cif`, `Nano.cif` and `Cyanoferrocene.cif`; update
that table (and `tests/README.md` if a test scenario changes) when a fixture is
replaced.

Some COD entries omit `_atom_type_oxidation_number`. The default connectivity
windows then fall back to covalent radii, with metal-metal pairs capped in
compounds (see `_default_pair_window` in `__init__.py`) so that, for example,
Ti(IV) is not bonded to Ti(IV) across a shared octahedron edge.
