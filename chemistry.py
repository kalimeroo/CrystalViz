"""ChemDraw-style bond-order and aromatic-ring perception.

This module turns a plain connectivity graph (atoms + bonds, no bond orders)
into a chemically sensible Lewis structure: multiple-bond orders from valence
bookkeeping, then aromatic rings from the Kekulé structure that results.

It is deliberately pure standard library — no ``bpy``, no ``gemmi``, no numpy —
so it stays unit testable outside Blender and outside the crystallography
worker (``gemmi_worker.py`` owns everything crystallographic).

Contract
--------
Every function takes the graph as *plain Python* structures:

* ``elements``: ``{node: "C"}`` — node keys are any mutually comparable,
  hashable objects (CrystalViz uses rounded cartesian tuples).
* ``edges``: iterable of ``(node_a, node_b)`` pairs.
* ``lengths``: optional ``{(node_a, node_b): normalised_length}`` where the
  length is divided by the sum of the two covalent radii (1.0 = "as long as
  the radii predict").  Missing lengths are treated as 1.0.

Edge keys are always normalised with :func:`edge_key`, so callers can pass
either orientation.
"""

# Highest bond order this module will ever assign.
MAX_BOND_ORDER = 3

# Valence ladders: the valences an element can show, ascending.  The target
# valence of an atom is the smallest rung at or above its connectivity degree
# (so a two-connected nitrogen aims at 3, a four-connected one at 4), which is
# what makes a Kekulé repair possible.  Elements missing from this table are
# left untouched: metals, noble gases and anything exotic keep single bonds.
VALENCE_LADDERS = {
    "H": (1,),
    "B": (3,),
    "C": (4,),
    "N": (3, 4),
    "O": (2,),
    "F": (1,),
    "SI": (4,),
    "P": (3, 5),
    "S": (2, 4, 6),
    "CL": (1,),
    "BR": (1,),
    "I": (1,),
}

# Only these elements take part in multiple bonds (H and the halogens never
# do: their single rung is already full once they are connected).
MULTIPLE_BOND_ELEMENTS = frozenset({"B", "C", "N", "O", "P", "S", "SI"})

# Ring perception only walks atoms that can be part of a conjugated ring.
RING_ELEMENTS = frozenset({"B", "C", "N", "O", "P", "S"})

# Elements contributing a lone pair (2 pi electrons) when they carry no double
# bond of their own inside the ring — pyrrole N, furan O, thiophene S.
LONE_PAIR_ELEMENTS = frozenset({"N", "P", "O", "S"})

# A candidate multiple bond may not be longer than the covalent radii predict
# (normalised length 1.0).  Real double/triple bonds sit well below that.
_CANDIDATE_MAX_LENGTH_SCALE = 1.0

# Ring sizes ChemDraw depicts as rings.  5, 6 and 7 cover the overwhelming
# majority of organic structures; 3- and 4-rings are handled by the ring-atom
# valence cap below instead of being drawn as aromatic.
_RING_SIZES = (5, 6, 7)
_RING_MAX_SIZE = max(_RING_SIZES)
_RING_STEP_BUDGET = 400_000
_RING_MAX_COUNT = 2048

# Bounded backtracking budgets.  These keep a pathological or incomplete
# structure (no hydrogens, radicals, metal clusters) from stalling the UI.
_REPAIR_STEP_BUDGET = 20_000
_REPAIR_NODE_LIMIT = 12
_REPAIR_DEFICIT_LIMIT = 12
_CYCLE_STEP_BUDGET = 40_000


def edge_key(first, second):
    """Return the orientation-independent key of an edge."""
    return (first, second) if first <= second else (second, first)


def _other(key, node):
    """Return the endpoint of *key* that is not *node*."""
    return key[1] if key[0] == node else key[0]


def _normalise_elements(elements):
    return {node: str(symbol).strip().upper() for node, symbol in elements.items()}


def _graph(elements, edges):
    """Build the adjacency lists and the deduplicated edge list."""
    adjacency = {node: [] for node in elements}
    edge_keys = []
    seen = set()
    for first, second in edges:
        if first == second or first not in elements or second not in elements:
            continue
        key = edge_key(first, second)
        if key in seen:
            continue
        seen.add(key)
        edge_keys.append(key)
        adjacency[first].append(second)
        adjacency[second].append(first)
    return adjacency, edge_keys


def _in_cycle(adjacency, node, budget):
    """True when *node* lies on a cycle of at least three atoms.

    Used to recognise "this two-connected ring atom is missing an implicit
    hydrogen" so it is not over-bonded into an allene-like arrangement.
    """
    neighbours = adjacency.get(node, ())
    if len(neighbours) < 2:
        return False
    for start in neighbours:
        stack = [(start, {node, start})]
        while stack:
            current, visited = stack.pop()
            budget[0] -= 1
            if budget[0] <= 0:
                return False
            for nxt in adjacency.get(current, ()):
                if nxt == node:
                    if len(visited) >= 3:
                        return True
                    continue
                if nxt in visited or len(visited) >= _RING_MAX_SIZE + 1:
                    continue
                stack.append((nxt, visited | {nxt}))
    return False


def _initial_deficit(element, degree, adjacency, node, budget):
    """How many extra bond orders *node* wants, or None when unconstrained."""
    ladder = VALENCE_LADDERS.get(element)
    if not ladder:
        return None
    for rung in ladder:
        if rung >= degree:
            target = rung
            break
    else:
        # Already past the highest valence (hypervalent or over-coordinated):
        # leave it alone instead of guessing.
        return None
    deficit = target - degree
    if deficit > 1 and _in_cycle(adjacency, node, budget):
        # A two-connected ring atom is one substituent short (usually an
        # implicit hydrogen): one extra order is the most it can carry.
        deficit = 1
    return deficit


def _candidates(elements, edge_keys, deficits, lengths):
    """Edges able to carry a multiple bond, shortest first."""
    candidates = []
    for key in edge_keys:
        first, second = key
        if first not in deficits or second not in deficits:
            continue
        if deficits[first] <= 0 and deficits[second] <= 0:
            continue
        if (
            elements[first] not in MULTIPLE_BOND_ELEMENTS
            or elements[second] not in MULTIPLE_BOND_ELEMENTS
        ):
            continue
        length = lengths.get(key)
        length = 1.0 if length is None else float(length)
        if length > _CANDIDATE_MAX_LENGTH_SCALE:
            continue
        candidates.append((length, key))
    candidates.sort()
    by_node = {}
    for _length, key in candidates:
        for node in key:
            by_node.setdefault(node, []).append(key)
    return [key for _length, key in candidates], by_node


def _greedy(deficits, extras, candidates):
    """Give every pair of neighbouring deficits one extra order each."""
    for key in candidates:
        first, second = key
        if deficits[first] > 0 and deficits[second] > 0:
            extras[key] = 1
            deficits[first] -= 1
            deficits[second] -= 1


def _augment_once(node, deficits, extras, by_node):
    """Satisfy *node* with a direct match or a 3-edge augmenting path."""
    for key in by_node.get(node, ()):
        if extras.get(key):
            continue
        middle = _other(key, node)
        if deficits.get(middle, 0) <= 0:
            continue
        extras[key] = 1
        deficits[node] -= 1
        deficits[middle] -= 1
        return True
    # node - middle (single) - partner (already matched) - free atom:
    # move the match one step further and give node the freed partner.
    for key in by_node.get(node, ()):
        if extras.get(key):
            continue
        middle = _other(key, node)
        if deficits.get(middle, 0) > 0:
            continue
        for matched in by_node.get(middle, ()):
            if matched == key or extras.get(matched) != 1:
                continue
            partner = _other(matched, middle)
            if partner in (node, middle):
                continue
            for far_key in by_node.get(partner, ()):
                if far_key == matched:
                    continue
                free = _other(far_key, partner)
                if free in (node, middle, partner) or deficits.get(free, 0) <= 0:
                    continue
                extras[key] = 1
                del extras[matched]
                extras[far_key] = 1
                deficits[node] -= 1
                deficits[free] -= 1
                return True
    return False


def _augment(deficits, extras, by_node):
    """Repair greedy dead ends with augmenting paths (bounded by degree)."""
    changed = True
    while changed:
        changed = False
        for node in sorted(deficits, key=repr):
            if deficits[node] <= 0:
                continue
            if _augment_once(node, deficits, extras, by_node):
                changed = True


def _escalate(deficits, extras, by_node):
    """Turn a double bond into a triple for atoms with a single partner.

    Restricting escalation to atoms that have exactly one *useful* neighbour
    keeps alkynes and nitriles right (H-C#C-H, N#C-C) while leaving a
    hydrogen-less aromatic ring at its alternating Kekulé pattern instead of
    tripling every ring bond.
    """
    for key in sorted(extras, key=repr):
        if extras[key] != 1:
            continue
        first, second = key
        if deficits[first] <= 0 or deficits[second] <= 0:
            continue
        for node in key:
            useful = [
                candidate
                for candidate in by_node.get(node, ())
                if deficits.get(_other(candidate, node), 0) > 0
            ]
            if len(useful) != 1:
                break
        else:
            extras[key] = 2
            deficits[first] -= 1
            deficits[second] -= 1


def _repair(deficits, extras, by_node, budget, depth=0):
    """Bounded backtracking for deficits greedy and augmenting could not meet.

    Only ever *adds* an order, so every step lowers the total remaining deficit
    by at least one: the recursion is bounded by that total, never by the size
    of the graph.  A dead end simply returns False and leaves the greedy
    structure alone.
    """
    pending = sorted((node for node, value in deficits.items() if value > 0), key=repr)
    if not pending:
        return True
    budget[0] -= 1
    if budget[0] <= 0 or depth >= _REPAIR_DEFICIT_LIMIT:
        return False
    node = min(pending, key=lambda item: (len(by_node.get(item, ())), repr(item)))
    for key in by_node.get(node, ()):
        other = _other(key, node)
        if deficits.get(other, 0) <= 0:
            continue
        current = extras.get(key, 0)
        for step in (1, 2):
            if current + step > MAX_BOND_ORDER - 1:
                continue
            if deficits[node] < step or deficits[other] < step:
                continue
            extras[key] = current + step
            deficits[node] -= step
            deficits[other] -= step
            if _repair(deficits, extras, by_node, budget, depth + 1):
                return True
            deficits[node] += step
            deficits[other] += step
            extras[key] = current
    return False


def bond_orders(elements, edges, lengths=None):
    """Return ``{edge_key: order}`` for every bond that is not a single bond.

    The algorithm is a valence repair, in four bounded stages:

    1. greedy: give one extra order to every edge joining two atoms that both
       still want one;
    2. augmenting paths: rescue atoms the greedy pass stranded even though a
       valid assignment exists;
    3. escalation: promote a double bond to a triple when an atom has exactly
       one possible partner left (alkynes, nitriles);
    4. bounded backtracking for whatever is still unsatisfied.

    Atoms whose element is unknown, or whose degree already exceeds its highest
    valence, are left unconstrained rather than guessed at.
    """
    elements = _normalise_elements(elements)
    lengths = {edge_key(*key): float(value) for key, value in (lengths or {}).items()}
    adjacency, edge_keys = _graph(elements, edges)
    budget = [_CYCLE_STEP_BUDGET]

    deficits = {}
    for node, element in elements.items():
        deficit = _initial_deficit(element, len(adjacency[node]), adjacency, node, budget)
        if deficit is not None:
            deficits[node] = deficit

    candidates, by_node = _candidates(elements, edge_keys, deficits, lengths)
    extras = {}
    _greedy(deficits, extras, candidates)
    _augment(deficits, extras, by_node)
    _escalate(deficits, extras, by_node)

    unsatisfied = sum(1 for value in deficits.values() if value > 0)
    if unsatisfied and unsatisfied <= _REPAIR_NODE_LIMIT:
        if sum(value for value in deficits.values() if value > 0) <= _REPAIR_DEFICIT_LIMIT:
            _repair(deficits, extras, by_node, [_REPAIR_STEP_BUDGET])

    return {key: extra + 1 for key, extra in extras.items() if extra > 0}


def find_rings(elements, edges, sizes=_RING_SIZES, max_rings=_RING_MAX_COUNT):
    """Return the simple cycles of *sizes* as frozensets of node keys.

    Only atoms that can sit in a conjugated ring are walked (``RING_ELEMENTS``
    with two or three neighbours), so hydrogens, metals and substituents never
    appear in the result.  Each ring is reported once, whichever direction the
    walk took.
    """
    elements = _normalise_elements(elements)
    adjacency, _edge_keys = _graph(elements, edges)
    allowed_sizes = tuple(sorted(int(size) for size in sizes))
    smallest = min(allowed_sizes)
    largest = max(allowed_sizes)

    def eligible(node):
        return elements.get(node) in RING_ELEMENTS and 2 <= len(adjacency[node]) <= 3

    rank = {node: index for index, node in enumerate(sorted(adjacency, key=repr))}
    neighbours = {
        node: [item for item in adjacency[node] if eligible(item)]
        for node in adjacency
        if eligible(node)
    }

    rings = []
    seen = set()
    steps = [_RING_STEP_BUDGET]

    for start in sorted(neighbours, key=lambda node: rank[node]):
        if len(rings) >= max_rings or steps[0] <= 0:
            break
        path = [start]
        on_path = {start}

        def walk(current, path=path, on_path=on_path, start=start):
            if steps[0] <= 0 or len(rings) >= max_rings:
                return
            for nxt in neighbours.get(current, ()):
                steps[0] -= 1
                if nxt == start:
                    if len(path) in allowed_sizes:
                        ring = frozenset(path)
                        if ring not in seen:
                            seen.add(ring)
                            rings.append(ring)
                    continue
                if nxt in on_path or rank[nxt] < rank[start]:
                    continue
                if len(path) >= largest:
                    continue
                path.append(nxt)
                on_path.add(nxt)
                walk(nxt)
                on_path.discard(nxt)
                path.pop()

        if smallest <= largest:
            walk(start)
    return rings


def _ring_pi_electrons(ring, ring_atoms, elements, ring_edges, adjacency, orders):
    """Hückel pi electron count of *ring*, or None when it cannot be aromatic.

    *ring_atoms* is the union of every ring in the molecule: a multiple bond to
    one of those atoms is a ring-fusion bond (shared with a neighbouring ring),
    not an exocyclic double bond, so only bonds leading out of the ring system
    veto the ring.
    """
    pi_electrons = 0
    for node in ring:
        element = elements.get(node, "")
        if element not in RING_ELEMENTS:
            return None
        incident = adjacency.get(node, ())
        # An exocyclic multiple bond (C=O, C=CH2, ...) breaks the conjugation
        # of this ring.
        for key in incident:
            if key in ring_edges or orders.get(key, 1) <= 1:
                continue
            if _other(key, node) not in ring_atoms:
                return None
        if any(orders.get(key, 1) > 1 for key in incident if key in ring_edges):
            pi_electrons += 1
        elif element in LONE_PAIR_ELEMENTS:
            pi_electrons += 2
        elif any(orders.get(key, 1) > 1 for key in incident):
            # A fusion atom whose double bond sits on the shared bond of the
            # neighbouring ring (a Clar structure): still one pi electron here.
            pi_electrons += 1
        else:
            # A carbon (or boron) with no double bond at all has an empty p
            # orbital: the ring is not conjugated.
            return None
    if pi_electrons >= 2 and (pi_electrons - 2) % 4 == 0:
        return pi_electrons
    return None


def aromatic_rings(elements, edges, orders):
    """Return ``[(ring_nodes, ring_edges)]`` for every aromatic ring.

    ``orders`` is the mapping returned by :func:`bond_orders`; rings are judged
    on the Kekulé structure it produced, so a fused system is described ring by
    ring (naphthalene yields two aromatic six-rings) instead of relying on bond
    length equalisation.
    """
    elements = _normalise_elements(elements)
    orders = {edge_key(*key): order for key, order in (orders or {}).items()}
    adjacency, edge_keys = _graph(elements, edges)
    by_node = {}
    for key in edge_keys:
        for node in key:
            by_node.setdefault(node, []).append(key)

    result = []
    rings = find_rings(elements, edges)
    ring_atoms = frozenset(node for ring in rings for node in ring)
    for ring in rings:
        ring_edges = frozenset(
            key
            for node in ring
            for key in by_node.get(node, ())
            if key[0] in ring and key[1] in ring
        )
        if _ring_pi_electrons(ring, ring_atoms, elements, ring_edges, by_node, orders) is None:
            continue
        result.append((frozenset(ring), ring_edges))
    result.sort(key=lambda item: min(repr(node) for node in item[0]))
    return result
