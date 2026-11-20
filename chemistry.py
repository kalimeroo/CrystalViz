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

from typing import NamedTuple

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
    # Group 15: a two-connected ring atom aims at 3, so arsabenzene and
    # stibabenzene take their As=C/Sb=C double from the ladder alone, exactly
    # as phosphinine does.  A five-membered arsole/stibole has an odd number of
    # unsatisfied orders, so the heteroatom keeps its lone pair instead.
    "P": (3, 5),
    "AS": (3, 5),
    "SB": (3, 5),
    # Group 16: a two-connected ring atom aims at 2, so the conjugated
    # six-membered rings need the explicit rule below (selenopyrylium,
    # telluropyrylium), exactly like thiopyrylium.
    "S": (2, 4, 6),
    "SE": (2, 4, 6),
    "TE": (2, 4, 6),
    "CL": (1,),
    "BR": (1,),
    "I": (1,),
}

# Only these elements take part in multiple bonds (H and the halogens never
# do: their single rung is already full once they are connected).
MULTIPLE_BOND_ELEMENTS = frozenset({"B", "C", "N", "O", "P", "S", "SI", "AS", "SB", "SE", "TE"})

# Ring perception only walks atoms that can be part of a conjugated ring.
RING_ELEMENTS = frozenset({"B", "C", "N", "O", "P", "S", "AS", "SB", "SE", "TE"})

# Elements contributing a lone pair (2 pi electrons) when they carry no double
# bond of their own inside the ring — pyrrole N, furan O, thiophene S, and
# their heavier analogues (phosphole/arsole/stibole, selenophene/tellurophene).
LONE_PAIR_ELEMENTS = frozenset({"N", "P", "O", "S", "AS", "SB", "SE", "TE"})

# Metals that carry an eta5/eta6/eta7 pi ligand (metallocenes, half-sandwiches,
# arene complexes).  A coordinated cyclopentadienyl ring is the 6 pi Huckel
# aromatic Cp- ligand, and IUPAC GR-1.7.2 requires such a coordinated
# contiguous pi system to be drawn with a delocalisation curve because no
# alternating single/double bond arrangement describes it.
HAPTIC_METALS = frozenset(
    {
        # Group 2 (Cp2Mg, Cp2Ca) and group 12 (Cp2Zn, Cp2Hg).
        "MG",
        "CA",
        "SR",
        "BA",
        "ZN",
        "CD",
        "HG",
        # d block.
        "SC",
        "TI",
        "V",
        "CR",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "Y",
        "ZR",
        "NB",
        "MO",
        "TC",
        "RU",
        "RH",
        "PD",
        "AG",
        "LA",
        "HF",
        "TA",
        "W",
        "RE",
        "OS",
        "IR",
        "PT",
        "AU",
        # f block (lanthanocenes, uranocene and relatives).
        "CE",
        "PR",
        "ND",
        "PM",
        "SM",
        "EU",
        "GD",
        "TB",
        "DY",
        "HO",
        "ER",
        "TM",
        "YB",
        "LU",
        "TH",
        "PA",
        "U",
        "NP",
        "PU",
    }
)

# A candidate multiple bond may not be longer than the covalent radii predict
# (normalised length 1.0).  Real double/triple bonds sit well below that, but
# pairs differ: the gap between a stretched C=C and a single C-C is smaller
# than the gap between a sulfone S=O and a long S-O contact, so the gate is
# per pair.  Pairs missing from the table keep the uniform 1.0 gate.
_DEFAULT_CANDIDATE_MAX_LENGTH_SCALE = 1.0
_CANDIDATE_MAX_LENGTH_BY_PAIR = {
    frozenset(("C", "C")): 0.98,
    frozenset(("C", "N")): 0.98,
    frozenset(("C", "O")): 0.98,
    frozenset(("C", "S")): 0.98,
    frozenset(("C", "SE")): 0.98,
    frozenset(("C", "TE")): 0.98,
    frozenset(("N", "N")): 0.98,
    frozenset(("N", "O")): 0.98,
    frozenset(("P", "O")): 0.96,
    frozenset(("S", "O")): 0.96,
    frozenset(("SE", "O")): 0.96,
    frozenset(("TE", "O")): 0.96,
}


def _candidate_max_length_scale(element_a, element_b):
    """Largest normalised length at which an edge may carry a multiple bond."""
    return _CANDIDATE_MAX_LENGTH_BY_PAIR.get(
        frozenset((element_a, element_b)), _DEFAULT_CANDIDATE_MAX_LENGTH_SCALE
    )


# A six-membered ring of a heavy chalcogen takes one ring double bond of its
# own only when its ring bonds are clearly shortened towards a double bond
# (thiopyrylium, selenopyrylium, telluropyrylium: ~0.92-0.96 of the
# covalent-radii sum).  A saturated C-S/C-Se/C-Te bond sits at ~1.0, and a
# five-membered thiophene/selenophene/tellurophene keeps its lone pair instead.
# A ring fused through a hydrogen-free carbon is left alone too: its
# benzologue (a benzothio-/benzoseleno-/benzotelluropyrylium) has an
# alternating Kekule structure with the positive charge on the fusion carbon,
# which is the form its geometry shows and the one the CCDC tools assign.
# Aromaticity itself is still decided by the Huckel count on the resulting
# Kekule structure, never by this threshold.
_RING_CHALCOGEN_MAX_LENGTH_SCALE = 0.97

# The chalcogens that take a genuine ring double bond in a six-membered ring,
# and whose terminal oxide (S=O, Se=O, Te=O) is then outside the ring pi system.
_SIX_RING_CHALCOGENS = frozenset({"S", "SE", "TE"})

# Tier 1 oxo-pattern centers (see :func:`_oxo_double_bonds`): an element whose
# terminal oxide oxygens are drawn as genuine X=O doubles — sulfone/sulfonate/
# sulfate S(=O)2 (and Se/Te analogues), nitro/nitrate N(=O)2, phosphate P=O.
# The spares come from the valence ladder (hexavalent chalcogen, pentavalent
# N/P); the actual double-bond count never exceeds _OXO_MAX_DOUBLES.
_OXO_TARGET_VALENCE = {"S": 6, "SE": 6, "TE": 6, "N": 5, "P": 5}
_OXO_MIN_TERMINAL_OXYGENS = {"S": 2, "SE": 2, "TE": 2, "N": 2, "P": 1}
_OXO_MAX_DOUBLES = 2

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
_AUGMENT_STEP_BUDGET = 200_000


class AromaticRing(NamedTuple):
    """One aromatic ring together with the electrons that make it aromatic.

    ``lone_pair_electrons`` is the part of ``pi_electrons`` that a heteroatom
    donates; a ring that needs such a donation has only one arrangement of
    alternating single and double bonds (IUPAC GR-6.3, furan/pyrrole), so the
    depiction layer draws it Kekule instead of with a delocalisation curve.
    ``kekule`` says whether every ring atom that does not donate a lone pair
    carries a drawn multiple bond, i.e. whether that Kekulé depiction exists at
    all; borazine and an h-BN sheet have none (their borons carry no B=N), so
    they keep the curve, while a carbocation ring centre (an empty p orbital on
    a three-connected carbon) counts as satisfied.  ``haptic_metal`` names the
    metal the whole ring coordinates to, when the ring is a pi ligand (eta5-Cp,
    eta6-arene) rather than a Kekule ring.
    """

    nodes: frozenset
    edges: frozenset
    pi_electrons: int
    lone_pair_electrons: int
    haptic_metal: str | None = None
    kekule: bool = False


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


def _haptic_metal(ring, elements, adjacency):
    """Return the one haptic metal every atom of *ring* is bonded to, or None.

    A metallocene cyclopentadienyl ring coordinates all five carbons to the same
    metal.  Requiring a *common* metal keeps a pair of unrelated metal
    contacts (or a bridging ligand) from passing as a pi complex.
    """
    common = None
    for node in ring:
        bonded = {
            elements.get(neighbour)
            for neighbour in adjacency.get(node, ())
            if elements.get(neighbour) in HAPTIC_METALS
        }
        common = bonded if common is None else common & bonded
        if not common:
            return None
    return min(common)


def _ring_eligible(node, elements, adjacency):
    """True when *node* can be walked as an atom of a conjugated ring.

    Two or three neighbours is the normal case.  A ring atom of a coordinated
    pi ligand also carries its metal bond (eta5-Cp: two ring carbons, one
    hydrogen and the metal), so the metal bonds are not counted: the ring
    itself only has the non-metal neighbours.

    A sulfoxide/selenoxide/telluroxide chalcogen carries one terminal oxygen
    that is not a ring atom either (thiabenzene 1-oxide: two ring carbons, a
    methyl and the O).  Its non-oxygen neighbours are counted the same way, so
    the six-membered ring can still be found.  A sulfone/selenone chalcogen
    (two terminal oxygens) stays out: it is not part of the ring pi system.
    """
    if elements.get(node) not in RING_ELEMENTS:
        return False
    neighbours = adjacency.get(node, ())
    if 2 <= len(neighbours) <= 3:
        return True
    if (
        elements.get(node) in _SIX_RING_CHALCOGENS
        and len(_terminal_oxygens(node, elements, adjacency)) == 1
    ):
        non_oxygen = sum(1 for item in neighbours if elements.get(item) != "O")
        return 2 <= non_oxygen <= 3
    if not any(elements.get(item) in HAPTIC_METALS for item in neighbours):
        return False
    non_metal = sum(1 for item in neighbours if elements.get(item) not in HAPTIC_METALS)
    return 2 <= non_metal <= 3


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


def _terminal_oxygens(node, elements, adjacency):
    """Terminal oxidic oxygens bonded to *node* (N-oxide/nitro O, sulfoxide O,
    sulfone S=O, ...).

    An oxygen is terminal when *node* is its only *covalent* neighbour.
    Cationic coordination contacts do not stop it being the exocyclic oxide of
    the heteroatom: a metal salt of a sulfone coordinates every S=O oxygen
    (Ca-O/K-O), so such an oxygen has graph degree two but still carries the
    S=O double.  A hydroxyl, ester or ether oxygen (a second covalent
    neighbour such as H or C) is not terminal.
    """
    result = []
    for other in adjacency.get(node, ()):
        if elements.get(other) != "O":
            continue
        for neighbour in adjacency.get(other, ()):
            if neighbour != node and elements.get(neighbour) in VALENCE_LADDERS:
                break
        else:
            result.append(other)
    return result


def _initial_deficit(element, degree, adjacency, node, budget, elements):
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
    oxygens = (
        _terminal_oxygens(node, elements, adjacency)
        if element in {"N", *_SIX_RING_CHALCOGENS}
        else ()
    )
    if element == "N" and target == 3 and degree == 3:
        if len(oxygens) >= 2:
            # Nitro (and nitrate) nitrogen: the depiction shows both N-O bonds
            # as genuine double bonds, N(=O)2, so the nitrogen is five-valent.
            target = 5
        elif oxygens:
            # An N-oxide (pyridine N-oxide, pyrazine N,N'-dioxide) is
            # four-valent: its N-O bond is dative, so the fourth bond is a real
            # multiple bond to a carbon.  Without it the nitrogen contributes
            # no pi bond, the ring is left with an odd number of unsatisfied
            # carbons and cannot close a Kekule structure at all.
            target = 4
    deficit = target - degree
    if element == "O":
        # An exocyclic oxide oxygen of nitrogen or a chalcogen (nitro N=O,
        # sulfone/sulfoxide S=O, ...) carries one extra order even when it
        # also coordinates a cation: a metal salt of a sulfone picks up every
        # S=O oxygen, so each has graph degree two and its ladder target is
        # already full — but the second S-O order still needs drawing.
        if any(
            elements.get(other) in {"N", *_SIX_RING_CHALCOGENS}
            and node in _terminal_oxygens(other, elements, adjacency)
            for other in adjacency.get(node, ())
        ):
            deficit = 1
    if deficit > 1 and _in_cycle(adjacency, node, budget):
        # A two-connected ring atom is one substituent short (usually an
        # implicit hydrogen): one extra order is the most it can carry.
        deficit = 1
    return deficit


def _dative_bonds(elements, adjacency):
    """N-O bonds that must stay single: the N-oxide bond.

    A three-connected nitrogen carrying exactly one terminal oxygen is an
    N-oxide nitrogen: that N-O bond is dative (N+ - O-), and its fourth valence
    is a real multiple bond into the ring or the substituent.  A nitro or
    nitrate nitrogen has two or three terminal oxygens: it is drawn N(=O)2, so
    neither N-O bond is excluded here.  A two-connected N-O (nitroso) is a
    genuine double bond as well.
    """
    single = set()
    for node, element in elements.items():
        if element != "N" or len(adjacency.get(node, ())) != 3:
            continue
        oxygens = _terminal_oxygens(node, elements, adjacency)
        if len(oxygens) == 1:
            single.add(edge_key(node, oxygens[0]))
    return single


def _candidates(elements, edge_keys, deficits, lengths, single_only=frozenset()):
    """Edges able to carry a multiple bond, shortest first."""
    candidates = []
    for key in edge_keys:
        first, second = key
        if key in single_only:
            continue
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
        if length > _candidate_max_length_scale(elements[first], elements[second]):
            continue
        candidates.append((length, key))
    candidates.sort()
    by_node = {}
    for _length, key in candidates:
        for node in key:
            by_node.setdefault(node, []).append(key)
    return [key for _length, key in candidates], by_node


def _force_triples(elements, deficits, by_node, extras):
    """Triple the only usable bond of an atom that has nowhere else to go.

    A two-connected alkyne carbon carries no hydrogen, so it needs a triple
    bond to reach its valence.  When one of its two neighbours is already
    saturated (a haptic metallocene carbon whose fourth bond is the metal, an
    aromatic ring atom whose extra order is spent in the ring), the remaining
    bond must carry both extra orders.  Seeding that triple *before* the greedy
    pass keeps the rest of the graph's Kekule repair intact.  Escalating after
    the greedy pass cannot: the partner is by then satisfied by another double
    bond, and moving that order away would strand a ring atom that the bounded
    repair is not allowed to reopen.
    """
    changed = True
    while changed:
        changed = False
        for node in sorted(deficits, key=repr):
            if deficits[node] != 2 or elements.get(node) not in {"C", "N"}:
                continue
            useful = [
                key for key in by_node.get(node, ()) if deficits.get(_other(key, node), 0) > 0
            ]
            if len(useful) != 1:
                continue
            key = useful[0]
            other = _other(key, node)
            if deficits.get(other, 0) < 2:
                continue
            extras[key] = 2
            deficits[node] -= 2
            deficits[other] -= 2
            changed = True


def _greedy(deficits, extras, candidates):
    """Give every pair of neighbouring deficits one extra order each."""
    for key in candidates:
        first, second = key
        if deficits[first] > 0 and deficits[second] > 0:
            extras[key] = 1
            deficits[first] -= 1
            deficits[second] -= 1


def _matched_key(node, extras, by_node):
    """Return the edge carrying *node*'s extra order, or None."""
    for key in by_node.get(node, ()):
        if extras.get(key) == 1:
            return key
    return None


def _find_augmenting_path(node, deficits, extras, by_node, path, visited, budget):
    """Extend the alternating *path* from *node* to a still-deficient atom.

    *path* holds an even number of edges (unmatched, matched, ..., matched) and
    *node* is the atom it reaches.  Returns the free endpoint that closes the
    path, or None.  The walk is depth-first over simple paths and bounded by
    *budget*, so a pathological graph cannot stall it.
    """
    budget[0] -= 1
    if budget[0] <= 0:
        return None
    for key in by_node.get(node, ()):
        if extras.get(key):
            continue
        other = _other(key, node)
        if other in visited:
            continue
        if deficits.get(other, 0) > 0:
            path.append(key)
            return other
        matched = _matched_key(other, extras, by_node)
        if matched is None:
            continue
        partner = _other(matched, other)
        if partner in visited:
            continue
        path.append(key)
        path.append(matched)
        visited.update((other, partner))
        free = _find_augmenting_path(partner, deficits, extras, by_node, path, visited, budget)
        if free is not None:
            return free
        visited.difference_update((other, partner))
        del path[-2:]
    return None


def _augment_once(node, deficits, extras, by_node, budget):
    """Satisfy *node* with a direct match or an augmenting path."""
    for key in by_node.get(node, ()):
        if extras.get(key):
            continue
        other = _other(key, node)
        if deficits.get(other, 0) <= 0:
            continue
        extras[key] = 1
        deficits[node] -= 1
        deficits[other] -= 1
        return True
    # node - x1 (single) - x2 (already matched) - ... - free atom: flip the
    # whole alternating path so *node* and the free atom become matched.
    path = []
    free = _find_augmenting_path(node, deficits, extras, by_node, path, {node}, budget)
    if free is None:
        return False
    for index, key in enumerate(path):
        if index % 2:
            del extras[key]
        else:
            extras[key] = 1
    deficits[node] -= 1
    deficits[free] -= 1
    return True


def _augment(deficits, extras, by_node):
    """Repair greedy dead ends with augmenting paths."""
    budget = [_AUGMENT_STEP_BUDGET]
    changed = True
    while changed and budget[0] > 0:
        changed = False
        for node in sorted(deficits, key=repr):
            if deficits[node] <= 0:
                continue
            if _augment_once(node, deficits, extras, by_node, budget):
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


def _hydrogen_free_carbon(node, elements, neighbours):
    """True for a carbon with exactly three neighbours, none of them hydrogen.

    That is the shape of a carbocation centre (or of an unsaturated fusion
    carbon): three sigma bonds and an empty p orbital.  A saturated CH or CH2
    has a hydrogen, so it never matches.
    """
    return (
        elements.get(node) == "C"
        and len(neighbours) == 3
        and all(elements.get(neighbour) != "H" for neighbour in neighbours)
    )


def _six_ring_chalcogen_deficits(elements, edges, lengths, adjacency):
    """Extra bond orders wanted by a sulfur, selenium or tellurium ring atom.

    A thiopyrylium cation (BIZTAF), a selenopyrylium/telluropyrylium cation, or
    a thiabenzene 1-oxide (ATBOXM) draws a genuine C=S/C=Se/C=Te double bond
    inside its six-membered ring, exactly as pyridine draws N=C: the chalcogen
    belongs to the pi system, so one of its ring bonds carries one more order,
    plus one more when it also carries a terminal oxygen drawn as S=O/Se=O/Te=O.
    The rule is deliberately narrow:

    * only a six-membered ring qualifies.  A five-membered thiophene,
      selenophene or tellurophene cannot hold an all-double-bond Kekule
      structure (chalcogen plus four carbons is an odd number of orders), so
      its chalcogen keeps its lone pair instead;
    * the ring must not be fused through a hydrogen-free carbon.  A
      benzothio-/benzoseleno-/benzotelluropyrylium has an alternating Kekule
      structure that leaves that fusion carbon without a double bond, i.e. as
      the ring-system carbocation (tellurochromenylium: ``C1=C9, C2=C3, C4=C5,
      C6=C7`` with C8 positive).  That localized form is what the
      crystallographic geometry shows -- the chalcogen ring bonds stay long --
      and it is what the CCDC tools assign, so the chalcogen does not take the
      ring double there.  A fusion through a saturated (hydrogen-carrying)
      carbon keeps the rule: that ring system is no pyrylium;
    * every chalcogen ring bond must already be shortened towards a double
      bond (below ``_RING_CHALCOGEN_MAX_LENGTH_SCALE`` of the covalent-radii
      sum).  A saturated thiacyclohexane/selenane/tellurane sits at ~1.0 and is
      left alone.

    Returns ``{node: deficit}``.
    """
    rings = find_rings(elements, edges)
    result = {}
    for ring in rings:
        if len(ring) != 6:
            continue
        if not any(elements.get(node) in _SIX_RING_CHALCOGENS for node in ring):
            continue
        if any(
            other is not ring
            and len(ring & other) >= 2
            and any(
                _hydrogen_free_carbon(node, elements, adjacency.get(node, ()))
                for node in ring & other
            )
            for other in rings
        ):
            continue
        for node in ring:
            if elements.get(node) not in _SIX_RING_CHALCOGENS:
                continue
            ring_lengths = [
                lengths.get(edge_key(node, other), 1.0)
                for other in adjacency.get(node, ())
                if other in ring
            ]
            if not ring_lengths or max(ring_lengths) >= _RING_CHALCOGEN_MAX_LENGTH_SCALE:
                continue
            result[node] = 1 + len(_terminal_oxygens(node, elements, adjacency))
    return result


def _oxo_double_bonds(elements, adjacency, lengths, deficits, extras, single_only):
    """Tier-1 pass: seed the genuine terminal X=O double bonds of oxo acids.

    A centre with enough terminal (oxide) oxygens is drawn with its chalcogen/
    group-15 ``X=O`` doubles decided by the oxo pattern itself instead of by
    deficit bookkeeping: a sulfone/sulfonate/triflate draws ``S(=O)2``, a
    selenone/tellurone likewise, a nitro or nitrate nitrogen ``N(=O)2``, and a
    phosphate/phosphonate phosphorus one ``P=O``.  The count is
    ``min(target valence - covalent degree, terminal oxygens, _OXO_MAX_DOUBLES)``
    drawn shortest-oxygen-first, and the centre's degree counts only ladder
    neighbours (``VALENCE_LADDERS``), so the cation contacts of a salt cannot
    inflate it and hide a double: a metal-coordinated sulfonate still draws its
    two S=O bonds.  ``B/Cl/Br/I`` never qualify, and a chalcogen already in
    the ring pi system (``_six_ring_chalcogen_deficits``) never has two
    terminal oxygens, so the two tiers cannot collide.

    Each seeded edge is written into *extras*, its endpoints' deficits are
    decremented (never below zero) and the edge is added to *single_only*, so
    the candidate engine never sees or overturns it.  Terminal oxide oxygens
    beyond the count keep an unmet deficit and are drawn single (charged).
    """
    for node, element in elements.items():
        target = _OXO_TARGET_VALENCE.get(element)
        if target is None:
            continue
        oxygens = _terminal_oxygens(node, elements, adjacency)
        if len(oxygens) < _OXO_MIN_TERMINAL_OXYGENS.get(element, 2):
            continue
        covalent_degree = sum(
            1 for other in adjacency.get(node, ()) if elements.get(other) in VALENCE_LADDERS
        )
        budget = min(target - covalent_degree, len(oxygens), _OXO_MAX_DOUBLES)
        if budget <= 0:
            continue
        gate = _candidate_max_length_scale(element, "O")
        ordered = sorted(
            oxygens,
            key=lambda oxygen: (lengths.get(edge_key(node, oxygen), 1.0), repr(oxygen)),
        )
        for oxygen in ordered[:budget]:
            if lengths.get(edge_key(node, oxygen), 1.0) > gate:
                continue
            key = edge_key(node, oxygen)
            extras[key] = 1
            single_only.add(key)
            deficits[node] = max(0, deficits.get(node, 0) - 1)
            deficits[oxygen] = max(0, deficits.get(oxygen, 0) - 1)


def bond_orders(elements, edges, lengths=None):
    """Return ``{edge_key: order}`` for every bond that is not a single bond.

    The algorithm is a valence repair, in six bounded stages:

    0. oxo seeding: a centre whose terminal oxidic oxygens call for a genuine
       ``X=O`` double (sulfone, sulfonate, triflate, selenone/tellurone, nitro/
       nitrate, phosphate) gets those doubles seeded up front and reserved, so
       no later stage can overturn them (see :func:`_oxo_double_bonds`);
    1. forced triples: an atom with two extra orders and only one usable bond
       (an alkyne or nitrile whose other neighbour is saturated) gets that bond
       tripled up front, so the rest of the graph can still find a clean Kekule
       structure.  An N-oxide nitrogen counts as four-valent and its dative N-O
       bond is never a candidate, so its fourth bond goes into the ring (see
       :func:`_initial_deficit` and :func:`_dative_bonds`);
    2. greedy: give one extra order to every edge joining two atoms that both
       still want one;
    3. augmenting paths: rescue atoms the greedy pass stranded even though a
       valid assignment exists;
    4. escalation: promote a double bond to a triple when an atom has exactly
       one possible partner left (alkynes, nitriles);
    5. bounded backtracking for whatever is still unsatisfied.

    Atoms whose element is unknown, or whose degree already exceeds its highest
    valence, are left unconstrained rather than guessed at.
    """
    elements = _normalise_elements(elements)
    lengths = {edge_key(*key): float(value) for key, value in (lengths or {}).items()}
    adjacency, edge_keys = _graph(elements, edges)
    budget = [_CYCLE_STEP_BUDGET]

    deficits = {}
    for node, element in elements.items():
        deficit = _initial_deficit(element, len(adjacency[node]), adjacency, node, budget, elements)
        if deficit is not None:
            deficits[node] = deficit
    if not _SIX_RING_CHALCOGENS.isdisjoint(elements.values()):
        # Overrides the rung chosen above: a conjugated six-membered chalcogen
        # ring needs one (or two) extra orders that the plain valence ladder
        # does not ask for (see :func:`_six_ring_chalcogen_deficits`).
        deficits.update(_six_ring_chalcogen_deficits(elements, edges, lengths, adjacency))

    single_only = _dative_bonds(elements, adjacency)
    extras = {}
    _oxo_double_bonds(elements, adjacency, lengths, deficits, extras, single_only)

    candidates, by_node = _candidates(elements, edge_keys, deficits, lengths, single_only)
    _force_triples(elements, deficits, by_node, extras)
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
    with two or three non-metal neighbours), so hydrogens, metals and
    substituents never appear in the result.  A pi ligand atom keeps its metal
    bond out of that count: the five carbons of a ferrocene cyclopentadienyl
    ring each have four neighbours (two carbons, one hydrogen, the iron) but
    only three non-metal ones, so the ring is found.  Each ring is reported
    once, whichever direction the walk took.
    """
    elements = _normalise_elements(elements)
    adjacency, _edge_keys = _graph(elements, edges)
    allowed_sizes = tuple(sorted(int(size) for size in sizes))
    smallest = min(allowed_sizes)
    largest = max(allowed_sizes)

    def eligible(node):
        return _ring_eligible(node, elements, adjacency)

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


def _carbocation(node, elements, by_node):
    """True when *node* is the carbocation centre of a ring system.

    A benzopyrylium fusion carbon (tellurochromenylium C8) is bonded to three
    atoms, carries no hydrogen, and has no valence left for a double bond: the
    ring system is a cation and that carbon is its positive centre.  Its empty
    p orbital contributes no pi electron but does not interrupt the conjugation,
    exactly like a three-connected boron.

    Three neighbours and no hydrogen is what keeps a saturated ring carbon out:
    a CH or CH2 that merely failed to find a double bond still rejects the ring.
    """
    return _hydrogen_free_carbon(
        node, elements, [_other(key, node) for key in by_node.get(node, ())]
    )


def _ring_pi_electrons(ring, ring_atoms, elements, ring_edges, by_node, orders, ring_count):
    """``(pi_electrons, lone_pair_electrons, donors)`` for *ring*, or None.

    *by_node* maps every atom to the edge keys around it (the shape
    :func:`_graph` yields, keyed by edges rather than neighbours).

    *ring_atoms* is the union of every ring in the molecule: a multiple bond to
    one of those atoms is a ring-fusion bond (shared with a neighbouring ring),
    not an exocyclic double bond, so only bonds leading out of the ring system
    veto the ring.

    *ring_count* maps every ring atom to the number of rings it belongs to.  A
    lone pair is only donated to a ring by a heteroatom of that ring alone: the
    bridgehead nitrogen of an indolizine (or any heteroatom shared by two
    rings) holds a lone pair that belongs to the fused pi system, so no single
    ring may claim it as its own sextet.  Both rings are then depicted Kekulé,
    which is the honest picture of an either/or delocalisation.

    A ring containing boron is exempt from that restriction: borazine, boroxine
    and an h-BN sheet are aromatic networks rather than a finite fused pi
    system, so a shared nitrogen or oxygen still supplies its share of the
    sextet (the boron itself contributes nothing).

    A three-connected ring carbon with no multiple bond and no hydrogen is the
    ring-system carbocation (the fusion carbon of a benzopyrylium): its empty p
    orbital contributes nothing either, so the localized Kekulé structure the
    valence repair produced stays aromatic.

    The second value counts how much of the sextet comes from such a donation,
    and the third names the atoms that donated it.  A ring drawn Kekulé is one
    where every other atom carries a drawn multiple bond.

    A double bond to a *terminal* carbon is exempt from the exocyclic veto: a
    graphite or graphene patch that has been grown by whole shells ends in
    dangling carbons, and a Kekule structure gives some of them a double bond to
    the ring.  That is a truncated lattice, not a C=CH2 substituent, so the ring
    stays aromatic.  With hydrogens present the CH2 carbon has a second
    neighbour and the veto still fires; the exemption only reaches a carbon that
    is bonded to nothing else.
    """
    boron_ring = any(elements.get(atom) == "B" for atom in ring)
    pi_electrons = 0
    lone_pair_electrons = 0
    donors = set()
    for node in ring:
        element = elements.get(node, "")
        if element not in RING_ELEMENTS:
            return None
        incident = by_node.get(node, ())
        ring_double = any(orders.get(key, 1) > 1 for key in incident if key in ring_edges)
        # An exocyclic multiple bond (C=O, C=CH2, ...) breaks the conjugation
        # of this ring.
        for key in incident:
            if key in ring_edges or orders.get(key, 1) <= 1:
                continue
            other = _other(key, node)
            if other in ring_atoms:
                continue
            if (
                element == "C"
                and elements.get(other) == "C"
                and orders.get(key, 1) == 2
                and len(by_node.get(other, ())) == 1
            ):
                # A dangling carbon at the edge of a truncated lattice: the
                # double bond is an artefact of growing a finite patch out of
                # an infinite sheet, not an exocyclic C=CH2.  A real methylene
                # carbon carries its hydrogens, so it is never degree one; a
                # terminal alkyne stays a triple and is still vetoed.
                continue
            if (
                element in _SIX_RING_CHALCOGENS
                and ring_double
                and elements.get(other) == "O"
                and len(by_node.get(other, ())) == 1
            ):
                # A thiopyrylium/thiabenzene (or Se/Te analogue) exocyclic
                # oxide: the chalcogen is part of the ring pi system through
                # its ring double bond, and the terminal oxygen is outside it,
                # so the S=O does not break conjugation.
                continue
            return None
        if ring_double:
            pi_electrons += 1
        elif element in LONE_PAIR_ELEMENTS and (ring_count.get(node, 1) == 1 or boron_ring):
            pi_electrons += 2
            lone_pair_electrons += 2
            donors.add(node)
        elif any(orders.get(key, 1) > 1 for key in incident):
            # A fusion atom whose double bond sits on the shared bond of the
            # neighbouring ring (a Clar structure): still one pi electron here.
            pi_electrons += 1
        elif element == "B":
            # A boron with no multiple bond has an empty p orbital: it
            # contributes no pi electron but does not break conjugation
            # (borazine, boroxine, boratabenzene).  A carbon in the same
            # position is left with an unsatisfied valence, so it still
            # rejects the ring.
            pass
        elif _carbocation(node, elements, by_node):
            # A three-connected ring carbon with no multiple bond and no
            # hydrogen is the carbocation centre of a benzopyrylium (or a
            # related fused ring cation): its empty p orbital carries no
            # electron but leaves the ring conjugated, so the localized
            # Kekule structure the valence repair produced stays aromatic.
            pass
        else:
            # A carbon (or any other element) with no double bond at all has
            # an empty p orbital: the ring is not conjugated.
            return None
    if pi_electrons >= 2 and (pi_electrons - 2) % 4 == 0:
        return pi_electrons, lone_pair_electrons, frozenset(donors)
    return None


def _haptic_pi_electrons(ring, ring_atoms, elements, ring_edges, adjacency, by_node, orders):
    """6 pi electrons when *ring* is a coordinated pi ligand, else None.

    *adjacency* maps an atom to its neighbour atoms, *by_node* to its edge keys.

    A metallocene cyclopentadienyl ring has no valence left for a Kekule double
    bond: every carbon is already bonded to its two ring neighbours, its
    substituent and the metal, so :func:`_ring_pi_electrons` finds an empty p
    orbital and rejects the ring.  The ligand is nevertheless the 6 pi
    aromatic Cp- anion (eta5-Cp, and likewise eta6-arene or eta7-tropylium),
    so a ring whose atoms all coordinate the same haptic metal counts as six pi
    electrons.  An exocyclic multiple bond still vetoes it, exactly as for a
    Kekule ring, so a coordinated quinone stays non-aromatic.
    """
    if _haptic_metal(ring, elements, adjacency) is None:
        return None
    for node in ring:
        for key in by_node.get(node, ()):
            if key in ring_edges or orders.get(key, 1) <= 1:
                continue
            if _other(key, node) not in ring_atoms:
                return None
    return 6


def _aromatic_ring_details(elements, edges, orders):
    """Perception of every aromatic ring, with its electron bookkeeping."""
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
    ring_count = {}
    for ring in rings:
        for node in ring:
            ring_count[node] = ring_count.get(node, 0) + 1
    for ring in rings:
        ring_edges = frozenset(
            key
            for node in ring
            for key in by_node.get(node, ())
            if key[0] in ring and key[1] in ring
        )
        counts = _ring_pi_electrons(
            ring, ring_atoms, elements, ring_edges, by_node, orders, ring_count
        )
        haptic_metal = None
        donors = frozenset()
        if counts is None:
            haptic = _haptic_pi_electrons(
                ring, ring_atoms, elements, ring_edges, adjacency, by_node, orders
            )
            if haptic is None:
                continue
            pi_electrons, lone_pair_electrons = haptic, 0
            haptic_metal = _haptic_metal(ring, elements, adjacency)
        else:
            pi_electrons, lone_pair_electrons, donors = counts

        # A Kekule depiction exists when every atom that does not donate its
        # lone pair already carries a drawn multiple bond -- inside the ring or
        # on a fusion bond -- or holds an empty p orbital instead (the
        # carbocation centre of a benzopyrylium).  Borazine and the h-BN sheet
        # have none (their borons carry no B=N), so they keep the
        # delocalisation curve.
        kekule = all(
            node in donors
            or any(orders.get(key, 1) > 1 for key in by_node.get(node, ()))
            or _carbocation(node, elements, by_node)
            for node in ring
        )
        result.append(
            AromaticRing(
                frozenset(ring),
                ring_edges,
                pi_electrons,
                lone_pair_electrons,
                haptic_metal,
                kekule,
            )
        )
    result.sort(key=lambda ring: min(repr(node) for node in ring.nodes))
    return result


def aromatic_rings(elements, edges, orders, detail=False):
    """Return the aromatic rings, with their electron bookkeeping on request.

    ``orders`` is the mapping returned by :func:`bond_orders`; rings are judged
    on the Kekulé structure it produced, so a fused system is described ring by
    ring (naphthalene yields two aromatic six-rings) instead of relying on bond
    length equalisation.

    With ``detail=False`` the result is ``[(ring_nodes, ring_edges)]``; with
    ``detail=True`` it is a list of :class:`AromaticRing` records, which also
    carry the lone-pair share of the sextet and the haptic metal of a pi
    ligand.
    """
    rings = _aromatic_ring_details(elements, edges, orders)
    if detail:
        return rings
    return [(ring.nodes, ring.edges) for ring in rings]
