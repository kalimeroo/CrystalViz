#!/usr/bin/env python
"""cctbx worker: read a CIF, expose its P1 cell, grow coordination shells.

The worker is stateless and runs in the cctbx interpreter. A structure is sent
back and forth as a small site table (the P1 cell contents) plus instances,
where an instance is the quadruple (site index, tx, ty, tz). One atom therefore
costs four small integers instead of a full dictionary.
"""

import json
import math
import sys
import traceback

from cctbx import uctbx
from cctbx.eltbx import covalent_radii, van_der_waals_radii
from iotbx import cif


def _radius_tables(element, label):
    try:
        vdw = van_der_waals_radii.vdw.table[element]
    except KeyError:
        vdw = 1.70
    try:
        covalent = covalent_radii.table(label or element).radius()
    except Exception:
        try:
            covalent = covalent_radii.table(element).radius()
        except Exception:
            covalent = 0.75
    return float(vdw), float(covalent)


def _read_structure(cif_path, block_name=None):
    structures = cif.reader(file_path=cif_path).build_crystal_structures()
    if not structures:
        raise ValueError("No crystal structures were found in this CIF.")
    if block_name:
        if block_name not in structures:
            raise ValueError(
                f"CIF block '{block_name}' was not found. "
                f"Available blocks: {', '.join(structures.keys())}"
            )
        key = block_name
    elif len(structures) == 1:
        key = next(iter(structures))
    else:
        raise ValueError(
            "This CIF contains multiple structures. Enter a CIF block name: "
            + ", ".join(structures.keys())
        )
    return structures[key], key


def _p1_sites(structure):
    """Cell contents in P1 as a small table of sites."""
    sites = []
    unit_cell = structure.unit_cell()
    p1_structure = structure.expand_to_p1(sites_mod_positive=True)
    for scatterer in p1_structure.scatterers():
        symbol = scatterer.element_symbol()
        element = symbol.upper()
        vdw_radius, covalent_radius = _radius_tables(symbol, scatterer.label)
        # Cartesian ADP tensor (U11, U22, U33, U12, U13, U23) in A^2.
        # cctbx already folds Uiso into a diagonal tensor and returns
        # non-positive values when the atom has no displacement parameters.
        u_cart = [
            float(value)
            for value in scatterer.u_cart_plus_u_iso(unit_cell)
        ]
        site = {
            "label": scatterer.label,
            "source_label": scatterer.label,
            "element": element,
            "frac": list(scatterer.site),
            "occupancy": float(scatterer.occupancy),
            "vdw_radius": vdw_radius,
            "covalent_radius": covalent_radius,
        }
        if u_cart[0] > 0.0 and u_cart[1] > 0.0 and u_cart[2] > 0.0:
            site["u_cart"] = u_cart
        sites.append(site)
    return sites


def _cell_columns(unit_cell):
    """Orthogonalized basis vectors: cart = sum(frac[i] * columns[i])."""
    columns = []
    for axis in range(3):
        basis = [0.0, 0.0, 0.0]
        basis[axis] = 1.0
        columns.extend(float(value) for value in unit_cell.orthogonalize(basis))
    return columns


def _enabled_windows(pair_settings):
    """Return {sorted element pair: (min_distance, max_distance)}."""
    windows = {}
    for setting in pair_settings:
        element_a = str(setting.get("element_a", "")).upper()
        element_b = str(setting.get("element_b", "")).upper()
        minimum = float(setting.get("min_distance", 0.0))
        maximum = float(setting.get("max_distance", 0.0))
        if setting.get("enabled", False) and 0.0 <= minimum <= maximum:
            windows[tuple(sorted((element_a, element_b)))] = (minimum, maximum)
    return windows


def _wrapped_fractional(frac):
    """Wrap fractional coordinates into [0, 1)."""
    wrapped = []
    for value in frac:
        value = float(value) % 1.0
        wrapped.append(0.0 if value >= 1.0 else value)
    return wrapped


def _fractional_reach(unit_cell, max_distance):
    """Per-axis fractional displacement equivalent to max_distance (Angstrom).

    A Cartesian move of at most max_distance can never move a site further
    than this in fractional coordinates, so it bounds both the periodic cell
    translations that must be inspected and the size of the search window.
    """
    matrix = list(unit_cell.fractionalization_matrix())
    reach = []
    for axis in range(3):
        row = matrix[axis * 3:axis * 3 + 3]
        reach.append(
            max_distance * math.sqrt(sum(value * value for value in row))
        )
    cells = [max(1, int(math.ceil(value))) for value in reach]
    return reach, cells


_CART_TOLERANCE = 1e-3  # Angstrom, position tolerance when matching atoms


def _fractional_tolerance(unit_cell, cart_tolerance):
    """Per-axis fractional tolerance equivalent to cart_tolerance (Angstrom)."""
    matrix = list(unit_cell.fractionalization_matrix())
    return [
        cart_tolerance * math.sqrt(
            sum(value * value for value in matrix[axis * 3:axis * 3 + 3])
        )
        for axis in range(3)
    ]


def _canonical_base_indices(base_atoms):
    """Index of the representative atom for every (element, site) group.

    Disordered CIFs can list the same element twice at one site (the P1
    expansion then contains exact duplicates). Growth treats those as a single
    atom, like the position based duplicate check of the original
    implementation did.
    """
    groups = {}
    canonical = []
    for index, atom in enumerate(base_atoms):
        key = (
            atom["element"],
            *(round(float(value), 5) for value in atom["cart"]),
        )
        canonical.append(groups.setdefault(key, index))
    return canonical


def _base_atom_lookup(unit_cell, base_atoms):
    """Bucket the P1 cell contents for periodic fractional lookups.

    Buckets wrap around the cell, so the 27 neighbouring buckets of a query
    always contain every base atom within the matching tolerance, no matter
    where in the cell the atom sits.
    """
    tolerance = _fractional_tolerance(unit_cell, _CART_TOLERANCE)
    buckets = max(1, int(math.ceil(1.0 / max(tolerance))))
    width = 1.0 / buckets
    grid = {}
    for index, atom in enumerate(base_atoms):
        wrapped = _wrapped_fractional(atom["frac"])
        cell = tuple(
            int(math.floor(value / width)) % buckets for value in wrapped
        )
        grid.setdefault(cell, []).append((index, wrapped))
    return {
        "tolerance": tolerance,
        "buckets": buckets,
        "width": width,
        "grid": grid,
    }


def _match_base_atom(unit_cell, base_atoms, lookup, frac, cart, element=None,
                     label=None):
    """Return (base atom index, integer cell translation) or None.

    Disordered CIFs can list different elements at the same site, so a
    position alone is ambiguous. Candidates of the same label win, then
    candidates of the same element, and only then anything close enough.
    """
    tolerance = lookup["tolerance"]
    buckets = lookup["buckets"]
    width = lookup["width"]
    wrapped = _wrapped_fractional(frac)
    cell = tuple(
        int(math.floor(value / width)) % buckets for value in wrapped
    )
    best = None
    best_other_element = None
    best_label = None
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbour = (
                    (cell[0] + da) % buckets,
                    (cell[1] + db) % buckets,
                    (cell[2] + dc) % buckets,
                )
                for candidate, base_wrapped in lookup["grid"].get(neighbour, ()):
                    delta = [
                        wrapped[axis] - base_wrapped[axis] for axis in range(3)
                    ]
                    if any(
                        abs(value - round(value)) > tolerance[axis]
                        for axis, value in enumerate(delta)
                    ):
                        continue
                    base_frac = base_atoms[candidate]["frac"]
                    offset = tuple(
                        int(round(float(frac[axis]) - base_frac[axis]))
                        for axis in range(3)
                    )
                    if cart is None:
                        return candidate, offset
                    expected = unit_cell.orthogonalize([
                        base_frac[axis] + offset[axis] for axis in range(3)
                    ])
                    distance_squared = sum(
                        (expected[axis] - cart[axis]) ** 2 for axis in range(3)
                    )
                    if distance_squared > _CART_TOLERANCE ** 2:
                        continue
                    found = (distance_squared, candidate, offset)
                    if (
                        label is not None
                        and base_atoms[candidate]["label"] == label
                    ):
                        if best_label is None or distance_squared < best_label[0]:
                            best_label = found
                        continue
                    if (
                        element is not None
                        and base_atoms[candidate]["element"] != element
                    ):
                        if (
                            best_other_element is None
                            or distance_squared < best_other_element[0]
                        ):
                            best_other_element = found
                        continue
                    if best is None or distance_squared < best[0]:
                        best = found
    if best_label is not None:
        best = best_label
    if best is None:
        best = best_other_element
    if best is None:
        return None
    return best[1], best[2]


def _base_atoms(unit_cell, sites):
    """Worker-side view of the site table: element, fractional and cartesian."""
    atoms = []
    for site in sites:
        frac = [float(value) for value in site["frac"]]
        atoms.append({
            "label": site.get("label"),
            "element": str(site.get("element", "X")).upper(),
            "frac": frac,
            "cart": list(unit_cell.orthogonalize(frac)),
        })
    return atoms


def _instances_from_atoms(unit_cell, base_atoms, lookup, atoms):
    """Match atom dictionaries to (site, translation) instances, keeping order."""
    instances = []
    for atom in atoms:
        frac = atom.get("frac")
        cart = atom.get("cart")
        if frac is None and cart is not None:
            frac = unit_cell.fractionalize(cart)
        if frac is None:
            continue
        match = _match_base_atom(
            unit_cell, base_atoms, lookup, frac, cart,
            str(atom.get("element", "X")).upper(), atom.get("label"),
        )
        if match is None:
            continue
        instances.extend((
            match[0], match[1][0], match[1][1], match[1][2],
        ))
    return instances


def load_structure(cif_path, block_name=None):
    """Return the P1 site table and the instances that make up the ASU."""
    structure, key = _read_structure(cif_path, block_name)
    unit_cell = structure.unit_cell()
    sites = _p1_sites(structure)
    base_atoms = _base_atoms(unit_cell, sites)
    lookup = _base_atom_lookup(unit_cell, base_atoms)
    asu_atoms = []
    for scatterer in structure.scatterers():
        frac = list(scatterer.site)
        asu_atoms.append({
            "label": scatterer.label,
            "element": scatterer.element_symbol().upper(),
            "frac": frac,
            "cart": list(unit_cell.orthogonalize(frac)),
        })
    instances = _instances_from_atoms(unit_cell, base_atoms, lookup, asu_atoms)
    if not instances:
        raise ValueError("The CIF atoms could not be placed in the unit cell.")
    return {
        "label": key,
        "space_group": structure.space_group().info().symbol_and_number(),
        "space_group_ops": [
            operation.as_xyz()
            for operation in structure.space_group().all_ops()
        ],
        "unit_cell": list(unit_cell.parameters()),
        "cell_matrix": _cell_columns(unit_cell),
        "sites": sites,
        "instances": instances,
    }


def _instance_keys(instances, canonical):
    """Turn a flat [site, tx, ty, tz, ...] list into exact growth keys."""
    keys = set()
    count = len(instances) - len(instances) % 4
    for offset in range(0, count, 4):
        site = int(instances[offset])
        if not 0 <= site < len(canonical):
            continue
        keys.add((
            canonical[site],
            (
                int(instances[offset + 1]),
                int(instances[offset + 2]),
                int(instances[offset + 3]),
            ),
        ))
    return keys


def _build_image_bank(unit_cell, base_atoms, base_elements, active_elements,
                      reach, cells, canonical=None):
    """Bucket every (base atom, neighbouring cell) image in fractional space."""
    bucket_width = [max(value, 1e-6) for value in reach]
    bank = {}
    for base_index, base_atom in enumerate(base_atoms):
        if canonical is not None and canonical[base_index] != base_index:
            continue
        if base_elements[base_index] not in active_elements:
            continue
        frac = base_atom["frac"]
        for ta in range(-cells[0], cells[0] + 1):
            for tb in range(-cells[1], cells[1] + 1):
                for tc in range(-cells[2], cells[2] + 1):
                    relative = (frac[0] + ta, frac[1] + tb, frac[2] + tc)
                    cart = tuple(unit_cell.orthogonalize(relative))
                    cell = tuple(
                        int(math.floor(relative[axis] / bucket_width[axis]))
                        for axis in range(3)
                    )
                    bank.setdefault(cell, []).append(
                        (base_index, ta, tb, tc, cart[0], cart[1], cart[2])
                    )
    return bank, bucket_width


def _periodic_neighbour_table(unit_cell, base_atoms, base_elements,
                              windows_by_element, bank, bucket_width,
                              canonical=None):
    """For each cell atom, list its bonded (base atom, cell offset) pairs.

    This depends only on the unit-cell contents and the distance windows, not
    on how large the displayed structure has become. A bond between cell atom
    i and cell atom j + t is expressed once as (j, t); translating the whole
    structure leaves the table valid.
    """
    table = []
    for base_index, base_atom in enumerate(base_atoms):
        pair_windows = windows_by_element.get(base_elements[base_index])
        if not pair_windows or (
            canonical is not None and canonical[base_index] != base_index
        ):
            table.append(())
            continue
        frac = base_atom["frac"]
        origin = unit_cell.orthogonalize(frac)
        bucket = tuple(
            int(math.floor(frac[axis] / bucket_width[axis])) for axis in range(3)
        )
        neighbours = []
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbour = (
                        bucket[0] + da, bucket[1] + db, bucket[2] + dc
                    )
                    for candidate, ta, tb, tc, x, y, z in bank.get(
                        neighbour, ()
                    ):
                        bounds = pair_windows.get(base_elements[candidate])
                        if bounds is None:
                            continue
                        delta_x = x - origin[0]
                        delta_y = y - origin[1]
                        delta_z = z - origin[2]
                        distance_squared = (
                            delta_x * delta_x
                            + delta_y * delta_y
                            + delta_z * delta_z
                        )
                        if distance_squared <= 1e-12:
                            continue
                        maximum = bounds[1]
                        if distance_squared > maximum * maximum:
                            continue
                        if math.sqrt(distance_squared) >= bounds[0]:
                            neighbours.append((candidate, ta, tb, tc))
        table.append(tuple(neighbours))
    return table


def _grow_shell_keys(unit_cell, base_atoms, canonical, visited, seed_keys,
                     windows):
    """Return the instance keys of the next coordination shell.

    A new atom must lie inside one enabled distance window of a seed atom, so
    growth is expressed as a graph walk instead of a bounding box over the
    whole structure:

    1. build the periodic neighbour table of one unit cell (depends only on
       the cell contents and the windows),
    2. walk one step from every seed key, skipping keys already present.

    Cost is O(cell atoms * local density + seed atoms * coordination number).
    It does not depend on the extent of the displayed structure, so repeated
    growth no longer re-enumerates the volume around it.
    """
    if not windows:
        raise ValueError("Enable at least one valid element-pair distance window.")
    max_distance = max(bound[1] for bound in windows.values())
    if max_distance <= 0.0:
        raise ValueError("Enabled connectivity maximum distances must be positive.")

    reach, cells = _fractional_reach(unit_cell, max_distance)

    windows_by_element = {}
    for (element_a, element_b), bounds in windows.items():
        windows_by_element.setdefault(element_a, {})[element_b] = bounds
        windows_by_element.setdefault(element_b, {})[element_a] = bounds

    base_elements = [atom["element"] for atom in base_atoms]
    bank, bucket_width = _build_image_bank(
        unit_cell, base_atoms, base_elements, windows_by_element, reach, cells,
        canonical,
    )
    table = _periodic_neighbour_table(
        unit_cell, base_atoms, base_elements, windows_by_element,
        bank, bucket_width, canonical,
    )

    new_keys = []
    for base_index, translation in sorted(seed_keys):
        for candidate, ta, tb, tc in table[base_index]:
            key = (
                candidate,
                (
                    translation[0] + ta,
                    translation[1] + tb,
                    translation[2] + tc,
                ),
            )
            if key in visited:
                continue
            visited.add(key)
            new_keys.append(key)
    return new_keys


def grow_connectivity_shell(
    unit_cell_parameters, sites, current_instances, seed_instances, pair_settings
):
    """Return the instance keys of the next shell around the seed atoms."""
    if not current_instances:
        raise ValueError("There are no current atoms to grow from.")
    if not seed_instances:
        raise ValueError("Select at least one atom to grow from.")

    unit_cell = uctbx.unit_cell(
        [float(value) for value in unit_cell_parameters]
    )
    base_atoms = _base_atoms(unit_cell, sites)
    if not base_atoms:
        raise ValueError("The site table is empty.")

    canonical = _canonical_base_indices(base_atoms)
    visited = _instance_keys(current_instances, canonical)
    seed_keys = _instance_keys(seed_instances, canonical)
    if not seed_keys:
        raise ValueError("The selected atoms are not part of the structure.")

    new_keys = _grow_shell_keys(
        unit_cell, base_atoms, canonical, visited, seed_keys,
        _enabled_windows(pair_settings),
    )

    new_instances = []
    for site_index, translation in sorted(new_keys):
        new_instances.extend((
            site_index, translation[0], translation[1], translation[2],
        ))
    return {"new_instances": new_instances, "grew_by": len(new_keys)}


def main():
    try:
        request = json.loads(sys.stdin.read())
        if request.get("op") == "grow_connectivity_shell":
            result = grow_connectivity_shell(
                request["unit_cell"],
                request.get("sites", []),
                request.get("current_instances", []),
                request.get("seed_instances", []),
                request.get("pair_settings", []),
            )
        else:
            result = load_structure(
                request["cif_path"], request.get("block_name") or None
            )
        sys.stdout.write(json.dumps({"ok": True, "result": result}))
    except Exception as exc:
        sys.stdout.write(json.dumps({
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }))


if __name__ == "__main__":
    main()
