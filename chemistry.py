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
        # Group 1: the alkali pi salts (NaCp, K pyrrolide/indolide) where the
        # cation contacts every atom of the ring with a short Pi bond.  The
        # haptic path only fires when the Kekule count has already failed, so
        # an alkali cation that merely bridges a ring nitrogen (CEMGAD's Na)
        # never triggers it -- the common-metal test needs the whole ring.
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
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

# Elements that bond covalently in these graphs: ring pi participants, the
# hydrogens that ride on them, and the halogens.  A ring atom with more than
# three neighbours is still walkable when the extra neighbours are *metal
# contacts* — a pyridine nitrogen bridging two sodium cations still donates
# its lone pair — so only the non-metal neighbours count toward the 2-3
# limit that decides whether an atom is ring material.  Unlike HAPTIC_METALS
# this tests the element itself, so alkali and alkaline-earth cations
# (Na/K/Li/Ca...) that never form an eta5/6/7 pi ligand still let their
# coordinated ring atom be enumerated; HAPTIC_METALS keeps its separate job
# of recognising genuine pi ligands in :func:`_haptic_pi_electrons`.
NONMETALS = frozenset(
    {
        "H",
        "D",
        "B",
        "C",
        "N",
        "O",
        "F",
        "SI",
        "P",
        "S",
        "CL",
        "AS",
        "SE",
        "BR",
        "TE",
        "I",
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
# Aromaticity itself is still decided by the Huckel count on the resulting
# Kekule structure, never by this threshold.
_RING_CHALCOGEN_MAX_LENGTH_SCALE = 0.97

# The chalcogens that take a genuine ring double bond in a six-membered ring,
# and whose terminal oxide (S=O, Se=O, Te=O) is then outside the ring pi system.
_SIX_RING_CHALCOGENS = frozenset({"S", "SE", "TE"})

# Tier 1 oxo-pattern centers (see :func:`_oxo_double_bonds`): an element whose
# terminal oxide oxygens are drawn as genuine X=O doubles — sulfone/sulfonate/
# sulfate S(=O)2 (and Se/Te analogues), nitro/nitrate N(=O)2, phosphate P=O,
# and a terminal C=O carbonyl (ketone/aldehyde/carboxylate/ester/amide).  The
# spares come from the valence ladder (hexavalent chalcogen, pentavalent N/P,
# tetravalent carbon); the actual double-bond count never exceeds
# _OXO_MAX_DOUBLES.
_OXO_TARGET_VALENCE = {"C": 4, "S": 6, "SE": 6, "TE": 6, "N": 5, "P": 5}
_OXO_MIN_TERMINAL_OXYGENS = {"S": 2, "SE": 2, "TE": 2, "N": 2, "P": 1, "C": 1}
_OXO_MAX_DOUBLES = 2

# A carbonyl C=O is recognised as such only when it is short enough: in the CSD
# the genuine C=O of ketones, aldehydes, carboxylates, esters and amides
# clusters at 0.85-0.90 of the covalent-radii sum, while a real C-O single
# whose proton is missing from the graph (enol C-OH, phenoxide Ar-O-, ester
# C-OR) sits at 0.92-1.00.  The length separates the oxide, whose double must
# be reserved in the tier-1 pass so a shorter ring/chain C-C cannot steal it,
# from the single bond of a deprotonated or protonated oxygen.
_C_CARBONYL_MAX_LENGTH_SCALE = 0.90

# The shortest P-C bond a four-coordinate group-15 centre may take while still
# being a saturated phosphonium: a genuine ylide P=C sits at 0.93-0.96 of the
# covalent-radii sum, a phosphonium P-C single at 0.98-1.00, so the threshold
# lets the ladder keep its 5th rung for a real double and drop it for the salt.
_P_YLIDE_MAX_LENGTH_SCALE = 0.97

# An N-O bond is recognised as an N-oxide bond only when it is short enough to
# be the charge-separated N+(O-) bond: in the CSD the (formally single) N-O of
# a pyridine N-oxide clusters at 0.975 of the covalent-radii sum (up to about
# 1.0), while the N-O(H) of a hydroxylamine sits at about 1.03 and an aliphatic
# amine N-oxide (whose ring is not aromatic) at ~1.02-1.04.  The length tells
# an N-oxide, whose fourth valence goes into the ring, from an N-OH, whose ring
# must stay non-aromatic even when the graph drops the hydroxyl hydrogen
# (which real CIF graphs commonly do).
_N_OXIDE_MAX_LENGTH_SCALE = 1.0

# The terminal N-N chain of a diazo (R2C=N+=N-) or azide (R-N=N+=N-) group is
# a cumulene: the terminal N- is two-valent (one double bond) and the middle
# N+ is four-valent (two double bonds), so both the C=N and the N=N are drawn
# double.  The terminal N-N bond is very short (~1.10-1.20 A, 0.78-0.85 of the
# covalent-radii sum) -- far below a single N-N (~1.45 A, 1.02) -- which
# separates the chain from an ordinary hydrazine/hydrazone N-N even when the
# hydrogens are missing from the graph.  The middle nitrogen must also reach
# its other neighbour (the diazo carbon, or the outer azide nitrogen) through
# a bond short enough to be a genuine double; a diazonium R-N+=N has a long
# C-N single bond there and keeps its current depiction.
_DIAZO_MAX_LENGTH_SCALE = 0.90
_DIAZO_PARTNER_MAX_LENGTH_SCALE = 0.92

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

# Clar re-placement: a fused ring system is re-Kekule-ed so that as many rings
# as possible get a full sextet, preferring terminal rings (Clar's rule).  The
# matching search is bounded by cluster size and a step budget so a large fused
# system cannot stall an import; a molecule with more rings than the cap (a
# grown graphite/graphene patch) is left alone.
_CLAR_MAX_RINGS = 8
_CLAR_MAX_ATOMS = 24
_CLAR_MAX_TOTAL_RINGS = 400
_CLAR_STEP_BUDGET = 120_000


class AromaticRing(NamedTuple):
    """One aromatic ring together with the electrons that make it aromatic.

    ``lone_pair_electrons`` is the part of ``pi_electrons`` that a heteroatom
    donates; a ring that needs such a donation has only one arrangement of
    alternating single and double bonds (IUPAC GR-6.3, furan/pyrrole), so the
    depiction layer draws it Kekule instead of with a delocalisation curve.
    ``kekule`` says whether every ring atom that does not donate a lone pair
    carries a drawn multiple bond, i.e. whether that Kekule depiction exists at
    all; borazine and an h-BN sheet have none (their borons carry no B=N), so
    they keep the curve.  ``haptic_metal`` names the metal the whole ring
    coordinates to, when the ring is a pi ligand (eta5-Cp, eta6-arene) rather
    than a Kekule ring.
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

    Any atom with more than three neighbours is walked only when the excess
    neighbours are metal contacts: a coordinated pi ligand carbon (eta5-Cp:
    two ring carbons, one hydrogen and the metal) and equally a pyridine
    nitrogen bridging one or two cations (Na/K/Mg... are not HAPTIC_METALS but
    their contact does not consume the lone pair).  Only the non-metal
    neighbours (``NONMETALS``) are counted toward the 2-3 limit.
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
    non_metal = sum(1 for item in neighbours if elements.get(item) in NONMETALS)
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


def _terminal_oxygens(node, elements, adjacency, lengths=None):
    """Terminal oxidic oxygens bonded to *node* (N-oxide/nitro O, sulfoxide O,
    sulfone S=O, ...).

    An oxygen is terminal when *node* is its only *covalent* neighbour.
    Cationic coordination contacts do not stop it being the exocyclic oxide of
    the heteroatom: a metal salt of a sulfone coordinates every S=O oxygen
    (Ca-O/K-O), so such an oxygen has graph degree two but still carries the
    S=O double.  A hydroxyl, ester or ether oxygen (a second covalent
    neighbour such as H or C) is not terminal.

    For a nitrogen centre the N-O bond must also be short enough to be an
    oxide bond (``_N_OXIDE_MAX_LENGTH_SCALE`` of the covalent-radii sum): an
    N-OH whose hydroxyl hydrogen is missing from the graph would otherwise
    look terminal and the ring would be promoted to a pyridine N-oxide it is
    not.  *lengths* is the map of normalised lengths; the check is skipped
    when it is None or when the centre is not nitrogen (a chalcogen oxide and
    a chalcogen hydroxide are too close in length to separate this way).
    """
    result = []
    for other in adjacency.get(node, ()):
        if elements.get(other) != "O":
            continue
        if (
            lengths is not None
            and elements.get(node) == "N"
            and lengths.get(edge_key(node, other), 1.0) > _N_OXIDE_MAX_LENGTH_SCALE
        ):
            continue
        for neighbour in adjacency.get(other, ()):
            if neighbour != node and elements.get(neighbour) in VALENCE_LADDERS:
                break
        else:
            result.append(other)
    return result


def _diazo_terminal(node, elements, adjacency, lengths):
    """True when *node* is the terminal N- of a diazo or azide group.

    A degree-one nitrogen bonded to another nitrogen through a very short N-N
    bond is the anionic end of ``R2C=N+=N-`` (diazo) or ``R-N=N+=N-`` (azide),
    which is two-valent -- not the three-valent N of a nitrile.  The geometry
    is required: without it the group cannot be told from an ordinary terminal
    N-H whose hydrogens are missing from the graph.
    """
    if lengths is None or elements.get(node) != "N":
        return False
    neighbours = adjacency.get(node, ())
    if len(neighbours) != 1 or elements.get(neighbours[0]) != "N":
        return False
    return lengths.get(edge_key(node, neighbours[0]), 1.0) <= _DIAZO_MAX_LENGTH_SCALE


def _diazo_middle(node, elements, adjacency, lengths):
    """True when *node* is the four-valent N+ of a diazo or azide chain.

    The middle nitrogen bonds the terminal N- through the short N-N bond and a
    carbon (diazo) or the outer nitrogen (azide) through a bond short enough to
    be a genuine double.  A diazonium ``R-N+=N`` has a long C-N single bond
    there, so it is not promoted and keeps its current depiction.
    """
    if lengths is None or elements.get(node) != "N":
        return False
    neighbours = adjacency.get(node, ())
    if len(neighbours) != 2:
        return False
    terminal = [
        other for other in neighbours if _diazo_terminal(other, elements, adjacency, lengths)
    ]
    if len(terminal) != 1:
        return False
    partner = next(other for other in neighbours if other != terminal[0])
    return lengths.get(edge_key(node, partner), 1.0) <= _DIAZO_PARTNER_MAX_LENGTH_SCALE


def _initial_deficit(element, degree, adjacency, node, budget, elements, lengths=None):
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
        _terminal_oxygens(node, elements, adjacency, lengths)
        if element in {"N", "P", "AS", "SB", *_SIX_RING_CHALCOGENS}
        else ()
    )
    if element in {"P", "AS", "SB"} and degree == 4:
        # A four-connected group-15 centre is either P(=X)R3 (X a terminal
        # chalcogen or imine nitrogen, drawn double through the 5th rung), a
        # genuine ylide P=C, or a saturated cation/complex.  The (3, 5) ladder
        # cannot tell them apart, so the geometry does:
        #  * a metal neighbour means a phosphine/arsine complex (Ar3P.M or an
        #    ylide-metalolate): the CSD types every P-C of such a centre
        #    Single -- the dative metal bond consumes the 5th rung, so even a
        #    short P-C (ADAKIB 0.965, a gold-bound ylide) stays single;
        #  * otherwise (all sigma) an all-carbon substituent set whose bonds
        #    are all too long to be a P=C ylide double stays at rung 4 and
        #    draws single bonds, exactly as the CSD types a phosphonium (P-C
        #    at 0.98-1.00 of the radii sum).  A short P-C (BEDLON 0.95,
        #    HINVAB 0.93) is a genuine ylide and keeps the 5th rung.
        metal = any(elements.get(other) not in VALENCE_LADDERS for other in adjacency.get(node, ()))
        if metal or (
            all(elements.get(other) in {"C", "H"} for other in adjacency.get(node, ()))
            and not any(
                lengths.get(edge_key(node, other), 1.0) <= _P_YLIDE_MAX_LENGTH_SCALE
                for other in adjacency.get(node, ())
            )
        ):
            target = 4
    if element == "N":
        if _diazo_terminal(node, elements, adjacency, lengths):
            # The anionic end of a diazo (R2C=N+=N-) or azide (R-N=N+=N-)
            # group is two-valent: one double bond, not the three-valent N of
            # a nitrile.
            target = 2
        elif _diazo_middle(node, elements, adjacency, lengths):
            # The central N+ of the same chain is four-valent: one double bond
            # to the terminal N- and one to its other neighbour.  That is what
            # lets the adjacent diazo carbon reach its own four bonds.
            target = 4
        elif target == 3 and degree == 3:
            if len(oxygens) >= 2:
                # Nitro (and nitrate) nitrogen: the depiction shows both N-O
                # bonds as genuine double bonds, N(=O)2, so the nitrogen is
                # five-valent.
                target = 5
            elif oxygens:
                # An N-oxide (pyridine N-oxide, pyrazine N,N'-dioxide) is
                # four-valent: its N-O bond is dative, so the fourth bond is a
                # real multiple bond to a carbon.  Without it the nitrogen
                # contributes no pi bond, the ring is left with an odd number
                # of unsatisfied carbons and cannot close a Kekule structure at
                # all.  Only a short N-O qualifies: an N-OH whose hydrogen is
                # missing from the graph has a long N-O (see
                # _N_OXIDE_MAX_LENGTH_SCALE in :func:`_terminal_oxygens`) and
                # stays three-valent, non-aromatic.
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
            and node in _terminal_oxygens(other, elements, adjacency, lengths)
            for other in adjacency.get(node, ())
        ):
            deficit = 1
    if element in {"S", "SE", "TE", "N"} and deficit <= 0:
        # A phosphine sulfide/selenide or iminophosphorane whose chalcogen or
        # nitrogen also coordinates a metal has graph degree two (or three)
        # but still carries its genuine P=X double: the CSD types the P-S bond
        # of a Ce thiophosphoryl complex (LEVQUZ) or the P=N of a
        # metal-bound phosphinimine (AVEYAE, BEFNUW) Double even though the
        # S/N hosts the cation.  The double sits on the phosphorus side, so
        # only a partner whose ladder neighbour(s) are group-15 pnictogens
        # counts -- a thiolate R-S-P or thiocyanate S-C bridging a metal
        # stays single.
        ladder = [o for o in adjacency.get(node, ()) if elements.get(o) in VALENCE_LADDERS]
        metal = [o for o in adjacency.get(node, ()) if elements.get(o) not in VALENCE_LADDERS]
        if metal and len(ladder) == 1 and elements.get(ladder[0]) in {"P", "AS", "SB"}:
            deficit = 1
        elif (
            element == "N"
            and metal
            and any(
                elements.get(other) in {"P", "AS", "SB"}
                and not any(
                    elements.get(m) not in VALENCE_LADDERS for m in adjacency.get(other, ())
                )
                for other in ladder
            )
        ):
            # A three-connected phosphinimine nitrogen (R3P=N-R'-M, degree
            # three: the (3, 4) ladder is full) still draws P=N when it
            # bridges a metal -- BEFNUW.  Only while the pnictogen is a plain
            # R3P= centre itself: a P that already holds a metal (UJITAI's
            # Zr-bound phosphine) keeps its P-N single.
            deficit = 1
    if deficit > 1 and _in_cycle(adjacency, node, budget):
        # A two-connected ring atom is one substituent short (usually an
        # implicit hydrogen): one extra order is the most it can carry.
        deficit = 1
    return deficit


def _dative_bonds(elements, adjacency, lengths=None):
    """N-O bonds that must stay single: the N-oxide bond.

    A three-connected nitrogen carrying exactly one terminal oxygen is an
    N-oxide nitrogen: that N-O bond is dative (N+ - O-), and its fourth valence
    is a real multiple bond into the ring or the substituent.  A nitro or
    nitrate nitrogen has two or three terminal oxygens: it is drawn N(=O)2, so
    neither N-O bond is excluded here.  A two-connected N-O (nitroso) is a
    genuine double bond as well.  Only a short N-O is an oxide bond (the
    length gate in :func:`_terminal_oxygens`): an N-OH is never excluded, so
    its ring stays non-aromatic.

    A two-connected nitrogen whose other neighbour is a metal is the anionic
    end of a metal-bound nitroxide or the ``N-O-`` of a hydroxylamido ligand:
    its one terminal oxygen is the dative ``N-O`` single of the anion, not the
    double of a nitroso (a nitroso nitrogen bonds a carbon, never a cation).
    The metal bond is dative either way, so the N-O stays single and the
    nitrogen does not reach its ladder valence -- which is what the CSD types.
    """
    single = set()
    for node, element in elements.items():
        if element != "N":
            continue
        neighbours = adjacency.get(node, ())
        oxygens = _terminal_oxygens(node, elements, adjacency, lengths)
        if len(neighbours) == 3 and len(oxygens) == 1:
            single.add(edge_key(node, oxygens[0]))
        elif len(neighbours) == 2 and len(oxygens) == 1:
            other = next(neighbour for neighbour in neighbours if neighbour != oxygens[0])
            if elements.get(other) not in VALENCE_LADDERS:
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


def _force_triples(elements, deficits, by_node, extras, adjacency=None, lengths=None):
    """Triple the only bond able to carry an atom's two spare orders.

    A two-connected alkyne carbon carries no hydrogen, so it needs a triple
    bond to reach its valence.  When one of its two neighbours is already
    saturated (a haptic metallocene carbon whose fourth bond is the metal, an
    aromatic ring atom whose extra order is spent in the ring), the remaining
    bond must carry both extra orders.  Seeding that triple *before* the greedy
    pass keeps the rest of the graph's Kekule repair intact.  Escalating after
    the greedy pass cannot: the partner is by then satisfied by another double
    bond, and moving that order away would strand a ring atom that the bounded
    repair is not allowed to reopen.

    Two relaxations on the "only usable bond" rule, both rooted in CSD typing:

    * the partner is selected among atoms that want *two* spare orders, not
      one.  An alkyne carbon whose other neighbour is a ring carbon with a
      deficit of its own (an enyne: ring C-C#C-C) still triples its C#C bond:
      the ring carbon can pair inside its ring, the chain carbon cannot, so
      preferring the tripled partner keeps the double out of the exocyclic
      bond.  Before this the ring carbon's deficit counted as a second usable
      bond and the greedy pass drew C=C=C instead;
    * a metal carbonyl (M-C#O, the carbon bonded only to a dative metal and a
      terminal oxygen) triples its C-O bond even though the oxygen only wants
      one spare order: the oxygen is short (``_C_CARBONYL_MAX_LENGTH_SCALE``)
      and the positive charge lands on the metal.  The valence ladder alone
      cannot ask for this -- it gives the oxygen a deficit of one -- so the
      geometry and the all-metal neighbourhood make the decision.
    """
    changed = True
    while changed:
        changed = False
        for node in sorted(deficits, key=repr):
            if deficits[node] != 2 or elements.get(node) not in {"C", "N"}:
                continue
            strong = [
                key for key in by_node.get(node, ()) if deficits.get(_other(key, node), 0) >= 2
            ]
            if len(strong) == 1:
                key = strong[0]
                other = _other(key, node)
                extras[key] = 2
                deficits[node] -= 2
                deficits[other] -= 2
                changed = True
                continue
            # Metal carbonyl C#O: the oxygen wants only one extra order, but
            # the carbon's every other neighbour is a dative metal, so the C-O
            # bond carries both spares while the charge sits on the metal.
            edges = by_node.get(node, ())
            if len(edges) != 1 or adjacency is None:
                continue
            key = edges[0]
            other = _other(key, node)
            if (
                elements.get(other) == "O"
                and deficits.get(other, 0) == 1
                and lengths.get(key, 1.0) <= _C_CARBONYL_MAX_LENGTH_SCALE
            ):
                others = [neighbour for neighbour in adjacency.get(node, ()) if neighbour != other]
                if others and all(
                    elements.get(neighbour) not in VALENCE_LADDERS for neighbour in others
                ):
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
    * every chalcogen ring bond must already be shortened towards a double
      bond (below ``_RING_CHALCOGEN_MAX_LENGTH_SCALE`` of the covalent-radii
      sum).  A saturated thiacyclohexane/selenane/tellurane sits at ~1.0 and is
      left alone.

    Returns ``{node: deficit}``.
    """
    result = {}
    for ring in find_rings(elements, edges, sizes=(6,)):
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
    selenone/tellurone likewise, a nitro or nitrate nitrogen ``N(=O)2``, a
    phosphate/phosphonate phosphorus one ``P=O``, and a carbon one ``C=O``
    (ketone, aldehyde, carboxylate, ester, amide).  The count is
    ``min(target valence - covalent degree, terminal oxygens, _OXO_MAX_DOUBLES)``
    drawn shortest-oxygen-first, and the centre's degree counts only ladder
    neighbours (``VALENCE_LADDERS``), so the cation contacts of a salt cannot
    inflate it and hide a double: a metal-coordinated sulfonate still draws its
    two S=O bonds.  ``B/Cl/Br/I`` never qualify, and a chalcogen already in
    the ring pi system (``_six_ring_chalcogen_deficits``) never has two
    terminal oxygens, so the two tiers cannot collide.

    A carbon centre bonded to a metal is left to :func:`_force_triples` instead
    (a metal carbonyl needs its C#O triple, which a plain C=O seed would block
    by eating the carbon's second spare order).  A carbonyl C=O must also be
    short enough to be an oxide (``_C_CARBONYL_MAX_LENGTH_SCALE``): a phenoxide
    or enol whose proton is missing from the graph looks terminal but keeps its
    single bond, so the ring keeps its aromaticity.

    Each seeded edge is written into *extras*, its endpoints' deficits are
    decremented (never below zero) and the edge is added to *single_only*, so
    the candidate engine never sees or overturns it.  Terminal oxide oxygens
    beyond the count keep an unmet deficit and are drawn single (charged).
    """
    for node, element in elements.items():
        target = _OXO_TARGET_VALENCE.get(element)
        if target is None:
            continue
        oxygens = _terminal_oxygens(node, elements, adjacency, lengths)
        if len(oxygens) < _OXO_MIN_TERMINAL_OXYGENS.get(element, 2):
            continue
        if element == "C" and any(
            elements.get(other) not in VALENCE_LADDERS for other in adjacency.get(node, ())
        ):
            # Metal carbonyl/acyl: the C=O/C#O is decided by the forced-triple
            # pass (for M-C#O) or the greedy pass (for acyl M-C(=O)R).
            continue
        covalent_degree = sum(
            1 for other in adjacency.get(node, ()) if elements.get(other) in VALENCE_LADDERS
        )
        budget = min(target - covalent_degree, len(oxygens), _OXO_MAX_DOUBLES)
        if budget <= 0:
            continue
        gate = (
            _C_CARBONYL_MAX_LENGTH_SCALE
            if element == "C"
            else _candidate_max_length_scale(element, "O")
        )
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


def delocalised_oxo_edges(elements, edges, lengths=None):
    """Terminal X-O bonds that carry a resonance (1.5) stroke.

    A centre whose oxo budget leaves at least one terminal oxygen undoubled is
    a resonance oxo group rather than a set of localized doubles: a
    carboxylate or carbonate (C), nitrate (N), phosphate/phosphonate (P), or
    sulfate/sulfonate (S, Se, Te) has more terminal oxygens than the valence
    ladder can draw as genuine ``X=O`` doubles.  Every terminal X-O bond of
    such a centre is returned so the depiction can draw it as a ChemDraw
    delocalised bond -- one solid and one dashed stroke -- instead of the
    localized ``C=O`` + ``C-O-`` pair.

    The count mirrors :func:`_oxo_double_bonds`: ``min(target valence -
    covalent degree, terminal oxygens, _OXO_MAX_DOUBLES)``.  When that budget
    is smaller than the number of terminal oxygens the leftover oxygens are
    the ones the oxo pass draws single, and the whole group is delocalised.
    A nitro ``N(=O)2`` or a sulfone ``S(=O)2`` spends its entire budget, so it
    keeps its genuine doubles; a lone carbonyl, sulfoxide or N-oxide (a single
    terminal oxygen) is never delocalised.

    The length gate is the general candidate gate
    (:func:`_candidate_max_length_scale`), not the tight carbonyl gate: the
    "two or more terminal oxygens" condition already separates a carboxylate
    or carbonate from a phenoxide/enol, and a carbonate C-O sits at ~0.90 of
    the radii sum, right at the carbonyl threshold.  A metal-bound oxygen
    still counts (a salt of a sulfonate coordinates every S=O oxygen), but a
    metal carbonyl carbon is skipped exactly as in the oxo pass.
    """
    elements = _normalise_elements(elements)
    lengths = {edge_key(*key): float(value) for key, value in (lengths or {}).items()}
    adjacency, _edge_keys = _graph(elements, edges)
    delocalised = set()
    for node, element in elements.items():
        target = _OXO_TARGET_VALENCE.get(element)
        if target is None:
            continue
        oxygens = _terminal_oxygens(node, elements, adjacency, lengths)
        if len(oxygens) < 2:
            continue
        if element == "C" and any(
            elements.get(other) not in VALENCE_LADDERS for other in adjacency.get(node, ())
        ):
            # Metal carbonyl/acyl: its C=O/C#O belongs to the forced-triple or
            # greedy pass, exactly as in _oxo_double_bonds.
            continue
        gate = _candidate_max_length_scale(element, "O")
        oxygens = [oxygen for oxygen in oxygens if lengths.get(edge_key(node, oxygen), 1.0) <= gate]
        if len(oxygens) < 2:
            continue
        covalent_degree = sum(
            1 for other in adjacency.get(node, ()) if elements.get(other) in VALENCE_LADDERS
        )
        budget = min(target - covalent_degree, len(oxygens), _OXO_MAX_DOUBLES)
        if budget <= 0 or len(oxygens) <= budget:
            continue
        for oxygen in oxygens:
            delocalised.add(edge_key(node, oxygen))
    return frozenset(delocalised)


def _ring_link_bonds(rings, adjacency):
    """Edges joining two ring atoms without belonging to any ring.

    A biaryl (or ring-to-ring) link is a single bond: the CSD types 88.8% of
    them Single and only 1.3% Double, and the short ones sit at the same
    normalised length as a ring double bond.  Sorting candidates by length
    alone therefore lets a link steal the double bond that belongs inside the
    ring: in EDOBEG the pyrrole ``C34=N2`` double was lost to the shorter
    phenyl link ``C34-C21``, which also stranded a phenyl carbon at three
    bonds.  The link is only forced single when a ring double is available;
    :func:`bond_orders` relaxes it again for a ring that still has an
    unsatisfied atom (the central C=C of a bifluorenylidene).
    """
    ring_atoms = frozenset(node for ring in rings for node in ring)
    ring_edges = set()
    for ring in rings:
        for node in ring:
            for other in adjacency.get(node, ()):
                if other in ring:
                    ring_edges.add(edge_key(node, other))
    links = set()
    for node in ring_atoms:
        for other in adjacency.get(node, ()):
            if other in ring_atoms:
                key = edge_key(node, other)
                if key not in ring_edges:
                    links.add(key)
    return links


def _metal_bound_ring_pnictogens(elements, adjacency, rings):
    """Ring group-15 atoms (N, P, As, Sb) that also coordinate a metal.

    The metal bond is dative: it does not consume a covalent valence, so the
    pyridine nitrogen of a metal complex is still three-valent with one ring
    double, exactly like the free ligand (the CSD types 92.7% of the N-C ring
    bonds of a metal-bound six-ring nitrogen Aromatic), and a metal-bound
    phosphabenzene/arsabenzene ring takes its P=C/As=C double the same way.  A
    pyrrole/pyrrolide nitrogen (or a phosphole/arsole/stibole) keeps its lone
    pair instead, so the metal bond alone does not say which of the two is
    present; :func:`_promote_metal_ring_pnictogens` decides from the ring's own
    valence needs.  Although the group-15 (3, 5) ladder leaves a degree-three
    P/As/Sb with no deficit of its own, the ring carbon left over by five
    carbons that cannot pair up is what the promotion responds to.
    """
    ring_atoms = frozenset(node for ring in rings for node in ring)
    return {
        node
        for node in ring_atoms
        if elements.get(node) in {"N", "P", "AS", "SB"}
        and any(elements.get(other) not in VALENCE_LADDERS for other in adjacency.get(node, ()))
    }


def _promote_metal_ring_pnictogens(elements, adjacency, rings, deficits, promoted):
    """Metal-bound ring group-15 atoms that must take a ring double after all.

    A pyrrole/pyrrolide nitrogen keeps its lone pair and its ring is drawn with
    the four carbons paired among themselves; a pyridine nitrogen carries one
    ring double.  Both are three-connected once the dative metal bond is
    discounted, so the nitrogen is only promoted when one of the carbons of
    its ring is still short of four bonds.  A pyrrole ring never leaves one
    (four carbons pair up), a pyridine ring always does (five carbons cannot),
    and so does a five-ring whose conjugation is broken by an exocyclic double
    (the dipyrrin of EDOBEG, where C37=C19 leaves C34 without a partner).
    The group-15 ring pnictogens follow the same two cases: a phosphinine/
    arsabenzene/stibabenzene ring has five carbons that cannot pair up, while a
    phosphole/arsole/stibole has four that can -- so the promotion turns a
    metal-coordinated six-membered ring aromatic and leaves a five-membered one
    with its lone pair.
    """
    result = set()
    for node in _metal_bound_ring_pnictogens(elements, adjacency, rings) - promoted:
        if deficits.get(node, 0) > 0:
            continue
        for ring in rings:
            if node not in ring:
                continue
            if any(elements.get(other) == "C" and deficits.get(other, 0) > 0 for other in ring):
                result.add(node)
                break
    return result


def _link_bonds_to_relax(link_bonds, rings, deficits, relaxed):
    """Biaryl links that may take a double bond after all.

    A link is only relaxed when one of its endpoints sits in a ring that still
    has an unsatisfied atom: the ring cannot pair its own atoms, so the link
    has to carry the order (the central C=C of a bifluorenylidene).  A link
    next to a ring that is already satisfied stays single.
    """
    result = set()
    for key in link_bonds - relaxed:
        for node in key:
            if any(
                node in ring and any(deficits.get(other, 0) > 0 for other in ring) for ring in rings
            ):
                result.add(key)
                break
    return result


def _solve_bond_orders(
    elements, edge_keys, adjacency, base_deficits, lengths, link_bonds, relaxed_links, promoted
):
    """Run the valence repair once with the current relaxations.

    Returns ``(extras, remaining)``: the extra orders placed on each edge, and
    the deficits still unsatisfied afterwards.
    """
    deficits = dict(base_deficits)
    for node in promoted:
        deficits[node] = max(deficits.get(node, 0), 1)
    single_only = _dative_bonds(elements, adjacency, lengths)
    single_only |= link_bonds - relaxed_links
    extras = {}
    _oxo_double_bonds(elements, adjacency, lengths, deficits, extras, single_only)

    candidates, by_node = _candidates(elements, edge_keys, deficits, lengths, single_only)
    _force_triples(elements, deficits, by_node, extras, adjacency, lengths)
    _greedy(deficits, extras, candidates)
    _augment(deficits, extras, by_node)
    _escalate(deficits, extras, by_node)

    unsatisfied = sum(1 for value in deficits.values() if value > 0)
    if unsatisfied and unsatisfied <= _REPAIR_NODE_LIMIT:
        if sum(value for value in deficits.values() if value > 0) <= _REPAIR_DEFICIT_LIMIT:
            _repair(deficits, extras, by_node, [_REPAIR_STEP_BUDGET])
    return extras, deficits


def bond_orders(elements, edges, lengths=None):
    """Return ``{edge_key: order}`` for every bond that is not a single bond.

    The algorithm is a valence repair, in six bounded stages:

    0. oxo seeding: a centre whose terminal oxidic oxygens call for a genuine
       ``X=O`` double (sulfone, sulfonate, triflate, selenone/tellurone, nitro/
       nitrate, phosphate, carbonyl C=O) gets those doubles seeded up front and
       reserved, so no later stage can overturn them (see
       :func:`_oxo_double_bonds`);
    1. forced triples: an atom with two extra orders and only one usable bond
       (an alkyne or nitrile whose other neighbour is saturated, or a ring
       carbon whose own deficit must stay in its ring; a metal carbonyl whose
       only other neighbours are dative metals) gets that bond tripled up
       front, so the rest of the graph can still find a clean Kekule
       structure.  An N-oxide nitrogen counts as four-valent and its dative
       N-O bond is never a candidate, so its fourth bond goes into the ring
       (see :func:`_initial_deficit` and :func:`_dative_bonds`).  Only a short
       N-O is an oxide bond (:func:`_terminal_oxygens`); an N-OH ring is left
       non-aromatic even when its hydroxyl hydrogen is absent from the graph.
       A diazo or azide terminal N-N chain books its terminal N as two-valent
       and its middle N as four-valent, so R2C=N+=N- is drawn with both bonds
       double and the diazo carbon reaches four (:func:`_diazo_terminal`);
    2. greedy: give one extra order to every edge joining two atoms that both
       still want one;
    3. augmenting paths: rescue atoms the greedy pass stranded even though a
       valid assignment exists;
    4. escalation: promote a double bond to a triple when an atom has exactly
       one possible partner left (alkynes, nitriles);
    5. bounded backtracking for whatever is still unsatisfied;
    6. Clar re-placement: a fused ring system whose Kekule structure leaves a
       ring without a sextet is re-Kekule-ed so that as many rings as possible
       become aromatic, preferring terminal rings (see
       :func:`_clar_ring_placement`).

    Two structural rules feed the deficits and the candidate set:

    * a metal bond is dative, so a ring group-15 atom (N, P, As, Sb) that
      coordinates a metal may still take the ring double of a pyridine or
      phosphabenzene (or of a pyrrole/phosphole whose conjugation an exocyclic
      double has broken).  It is only promoted when a carbon of its ring is
      otherwise left with three bonds, which is what tells a pyridine (five
      carbons cannot pair up) from a pyrrole (four carbons can) -- see
      :func:`_promote_metal_ring_pnictogens`;
    * a bond between two ring atoms that belongs to no ring is a biaryl link
      and stays single while a ring double is available, so it cannot steal
      the double bond of the ring (see :func:`_ring_link_bonds`).  It is
      relaxed for a ring that still has an unsatisfied atom, which keeps a
      genuine inter-ring C=C (bifluorenylidene) double.

    Atoms whose element is unknown, or whose degree already exceeds its highest
    valence, are left unconstrained rather than guessed at.
    """
    elements = _normalise_elements(elements)
    lengths = {edge_key(*key): float(value) for key, value in (lengths or {}).items()}
    adjacency, edge_keys = _graph(elements, edges)
    rings = find_rings(elements, edges)
    link_bonds = _ring_link_bonds(rings, adjacency)
    budget = [_CYCLE_STEP_BUDGET]

    deficits = {}
    for node, element in elements.items():
        deficit = _initial_deficit(
            element, len(adjacency[node]), adjacency, node, budget, elements, lengths
        )
        if deficit is not None:
            deficits[node] = deficit
    if not _SIX_RING_CHALCOGENS.isdisjoint(elements.values()):
        # Overrides the rung chosen above: a conjugated six-membered chalcogen
        # ring needs one (or two) extra orders that the plain valence ladder
        # does not ask for (see :func:`_six_ring_chalcogen_deficits`).
        deficits.update(_six_ring_chalcogen_deficits(elements, edges, lengths, adjacency))

    promoted = set()
    relaxed_links = set()
    extras, remaining = _solve_bond_orders(
        elements, edge_keys, adjacency, deficits, lengths, link_bonds, relaxed_links, promoted
    )
    # A metal-bound ring pnictogen (pyridine/phosphinine N/P/As/Sb) is promoted
    # before a biaryl link is relaxed: the ring double is the chemically
    # correct answer, the link double the fallback for an atom that has nothing
    # else.  Each round is bounded.
    for _ in range(3):
        new_promoted = _promote_metal_ring_pnictogens(
            elements, adjacency, rings, remaining, promoted
        )
        if new_promoted:
            promoted |= new_promoted
        else:
            new_links = _link_bonds_to_relax(link_bonds, rings, remaining, relaxed_links)
            if not new_links:
                break
            relaxed_links |= new_links
        extras, remaining = _solve_bond_orders(
            elements, edge_keys, adjacency, deficits, lengths, link_bonds, relaxed_links, promoted
        )

    extras = _clar_ring_placement(elements, edge_keys, adjacency, rings, extras, remaining, lengths)
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


def _fused_ring_clusters(rings):
    """Group rings that share a fusion bond (two or more atoms)."""
    parent = {ring: ring for ring in rings}
    size = {ring: 1 for ring in rings}

    def find(ring):
        while parent[ring] != ring:
            parent[ring] = parent[parent[ring]]
            ring = parent[ring]
        return ring

    ring_list = list(rings)
    for index, first in enumerate(ring_list):
        for second in ring_list[index + 1 :]:
            if len(first & second) < 2:
                continue
            root_a, root_b = find(first), find(second)
            if root_a == root_b:
                continue
            if size[root_a] < size[root_b]:
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            size[root_a] += size[root_b]

    clusters = {}
    for ring in rings:
        clusters.setdefault(find(ring), []).append(ring)
    return list(clusters.values())


def _clar_hydroxyl_nitrogen(node, elements, adjacency, lengths):
    """True when *node* (a ring nitrogen) binds a hydroxyl-like oxygen.

    ``_terminal_oxygens`` refuses an N-oxide in two ways: an N-O bond too
    long to be an oxide (``_N_OXIDE_MAX_LENGTH_SCALE``) -- an N-OH whose
    hydroxyl hydrogen is missing from the graph looks terminal but leaves the
    nitrogen three-valent -- or an oxygen carrying a second covalent
    neighbour (the hydroxyl hydrogen present).  The Clar re-placement's fresh
    promotion of a required-zero single-ring heteroatom must honour the same
    gate, otherwise an N-OH ring would be promoted to a pyridinium-type
    double it is not and become a bogus aromatic sextet (the N-oxide length
    tests rely on this).  The check is nitrogen-only, exactly like
    ``_terminal_oxygens``: ring nitrogen is promoted only while every oxygen
    it binds is a genuine terminal oxide.
    """
    if elements.get(node) != "N" or not adjacency.get(node):
        return False
    oxides = set(_terminal_oxygens(node, elements, adjacency, lengths))
    return any(
        elements.get(other) == "O" and other not in oxides for other in adjacency.get(node, ())
    )


def _clar_cluster_placement(
    cluster,
    elements,
    adjacency,
    ring_atoms,
    ring_count,
    ring_edges,
    by_node,
    orders,
    remaining,
    single_only,
    lengths,
    budget,
):
    """Best set of ring doubles for one fused *cluster*, or None.

    None means the cluster is too large, holds a haptic-metal ring, or has a
    cumulated atom; the caller then leaves the drawn orders untouched.
    """
    atoms = frozenset(node for ring in cluster for node in ring)
    if len(cluster) > _CLAR_MAX_RINGS or len(atoms) > _CLAR_MAX_ATOMS:
        return None
    if any(
        _haptic_pi_electrons(
            ring, ring_atoms, elements, ring_edges[ring], adjacency, by_node, orders
        )
        is not None
        for ring in cluster
    ):
        return None
    movable = frozenset(key for ring in cluster for key in ring_edges[ring])
    current_doubles = {
        node: sum(1 for key in by_node.get(node, ()) if key in movable and orders.get(key, 1) > 1)
        for node in atoms
    }
    if any(count > 1 for count in current_doubles.values()):
        # A cumulated atom: flattening it would change its valence.
        return None
    # The valence repair already placed one double on each atom that needs one,
    # and left a residual deficit on the atoms it could not satisfy.  A carbon
    # with either is mandatory: the re-placement must cover it, otherwise it
    # would strand the atom at three bonds.  A lone-pair heteroatom is not: its
    # double may be the repair's way of reaching the third rung of a (3, 4)
    # ladder when the CIF omits the N-H, and the pyrrole/phosphole depiction is
    # the heteroatom single with its lone pair (GEXRAC's N-H five-ring).  The
    # Huckel count then decides: a 5-ring's four carbons pair among themselves,
    # so the lone pair is preferred, while a 6-ring's five cannot and keeps the
    # heteroatom double.  A nitrogen that belongs to a single ring may also
    # take a double the repair did not ask for -- a pyridinium or N-oxide
    # nitrogen is four-valent even when its three sigma bonds already fill the
    # third rung.  The fresh promotion is nitrogen-only: a chalcogen or
    # pnictogen (an S, Se, Te, As, Sb ring atom) keeps its lone pair, so a
    # diazastibolene cannot promote its antimony into an Sb=N that would fake
    # an aromatic sextet out of an 8-pi ring.  And it applies to at most one
    # ring atom: a single pyridinium completes a genuine sextet, while two
    # promoted ring nitrogens (plus a third atom's lone pair) would let a
    # non-aromatic ring -- two N and an Sb, each with its lone pair -- re-read
    # as 6 pi when it is really an 8-pi antiaromatic system that must stay
    # single-bonded C=C.  A bridgehead heteroatom keeps its lone pair for the
    # fused system (indolizine) unless the repair already asked for a double,
    # and a saturated carbon (four sigma orders: sp3 ring carbon, exocyclic
    # C=O) has no room for one.
    required = {node: current_doubles[node] + remaining.get(node, 0) for node in atoms}
    mandatory = frozenset(
        node
        for node in atoms
        if required[node] >= 1 and elements.get(node) not in LONE_PAIR_ELEMENTS
    )
    capacity = {}
    for node in atoms:
        if required[node] >= 1:
            capacity[node] = 1
        elif elements.get(node) == "C" or ring_count.get(node, 0) > 1:
            capacity[node] = 0
        elif _clar_hydroxyl_nitrogen(node, elements, adjacency, lengths):
            # A long N-O means N-OH, not a pyridinium the Clar pass may
            # promote: the fresh double would fabricate an aromatic sextet
            # where the N-oxide length gate says there is none.
            capacity[node] = 0
        elif elements.get(node) == "N":
            # Fresh promotion is a pyridinium/N-oxide nitrogen: the only
            # ring atom the search may give a double the repair did not ask
            # for.  Every other lone-pair heteroatom (a ring S, Se, Te, As,
            # Sb) keeps its lone pair, so an Sb ring atom can never be
            # promoted into an Sb=N.
            capacity[node] = 1
        else:
            capacity[node] = 0
    double_capable = set()
    for key in movable:
        first, second = key
        if key in single_only:
            continue
        if capacity.get(first, 0) < 1 or capacity.get(second, 0) < 1:
            continue
        length = lengths.get(key)
        length = 1.0 if length is None else float(length)
        if length > _candidate_max_length_scale(elements[first], elements[second]):
            continue
        double_capable.add(key)
    if not double_capable:
        return None

    current_moves = frozenset(key for key in movable if orders.get(key, 1) > 1)
    atom_edges = {node: [] for node in atoms}
    for key in double_capable:
        atom_edges[key[0]].append(key)
        atom_edges[key[1]].append(key)
    # Most-constrained atom first (fewest usable edges), so a fruitless branch
    # is pruned before it can waste the shared budget; and within an atom try
    # the already-drawn doubles before the fresh ones, so the first all-valid
    # arrangement found is the one closest to the current depiction.
    for node in atom_edges:
        atom_edges[node].sort(
            key=lambda key: (
                0 if key in current_moves else 1,
                lengths.get(key, 1.0),
            )
        )
    nodes = sorted(atoms, key=lambda node: (len(atom_edges.get(node, ())), repr(node)))
    best = {"score": None, "moves": current_moves}
    # Every aromatic ring of an all-valid cluster is a terminal ring that
    # shares a fusion bond with exactly one other ring of the cluster -- a
    # property of the cluster's topology, not of the chosen placement -- so
    # once every ring is aromatic no later matching can score higher and the
    # search stops, leaving the shared budget for the remaining clusters.
    done = [False]

    def evaluate(moves):
        covered = set()
        for key in moves:
            covered.update(key)
        if not mandatory <= covered:
            # The re-placement may not strand an atom the repair had satisfied.
            return
        candidate = dict(orders)
        for key in moves:
            candidate[key] = 2
        for key in movable - moves:
            candidate[key] = 1
        valid = 0
        terminal = 0
        for ring in cluster:
            fresh = [
                node
                for node in ring
                if required.get(node, 0) == 0
                and elements.get(node) in LONE_PAIR_ELEMENTS
                and any(
                    key in ring_edges[ring] and candidate.get(key, 1) > 1
                    for key in by_node.get(node, ())
                )
            ]
            if len(fresh) > 1:
                # Two freshly promoted ring heteroatoms (plus a third's lone
                # pair) are how an intrinsically non-aromatic ring passes for
                # a sextet: the diazastibolene, with two N and an Sb in the
                # ring, would re-read as 6 pi when it is 8.  A single
                # pyridinium completes a genuine sextet; keep that reading.
                continue
            if (
                _ring_pi_electrons(
                    ring,
                    ring_atoms,
                    elements,
                    ring_edges[ring],
                    by_node,
                    candidate,
                    ring_count,
                )
                is None
            ):
                continue
            valid += 1
            if sum(1 for other in cluster if other != ring and len(other & ring) >= 2) == 1:
                terminal += 1
        changes = len(moves ^ current_moves)
        score = (valid, terminal, -changes)
        if best["score"] is None or score > best["score"]:
            best["score"] = score
            best["moves"] = frozenset(moves)
            if valid == len(cluster):
                done[0] = True

    def choose(index, moves, matched):
        budget[0] -= 1
        if budget[0] <= 0 or done[0]:
            return
        if index >= len(nodes):
            evaluate(moves)
            return
        node = nodes[index]
        if node in matched:
            choose(index + 1, moves, matched)
            return
        for key in atom_edges.get(node, ()):
            if key in matched:
                continue
            other = _other(key, node)
            if other in matched:
                continue
            matched.add(node)
            matched.add(other)
            moves.add(key)
            choose(index + 1, moves, matched)
            if done[0]:
                return
            moves.discard(key)
            matched.discard(other)
            matched.discard(node)
        # no double for this atom
        choose(index + 1, moves, matched)

    # Seed with the current placement so the search can never regress.
    evaluate(current_moves)
    choose(0, set(), set())
    return best["moves"]


def _clar_ring_placement(elements, edge_keys, adjacency, rings, extras, remaining, lengths):
    """Re-Kekule ring systems so that as many rings as possible are aromatic.

    The valence repair walks the candidate bonds shortest first, so in a fused
    system it can spend a ring atom's double on the fusion bond and leave the
    neighbouring ring with an empty p orbital (or an over-filled sextet), even
    though another arrangement of the same cluster's double bonds closes both.
    A lone ring can mis-place its doubles the same way without any fusion: a
    metal-bound pyrazolide whose two metal-promoted N=C doubles strand the
    opposite carbon at three bonds (ALIGUA) has a four-pi placement where a
    C=C-instead-of-N=C arrangement reaches six with one nitrogen as the lone
    pair holder, so singleton clusters are re-placed exactly like fused ones.
    Clar's rule prefers the arrangement with the most disjoint sextets, and a
    terminal ring's sextet over an internal one.

    Only bonds that belong to the cluster's rings are rewritten.  Every atom
    keeps at most one ring double; a bond may only carry one when it passes the
    same length gate as the valence repair; and an atom only takes a double when
    its valence still allows it.  The atoms the repair already satisfied or left
    deficient must all stay covered, so no atom is stranded at three bonds.  A
    carbon already saturated by its other bonds (an sp3 ring carbon, an
    exocyclic C=O) is never given one; a heteroatom belonging to a single ring
    may be promoted (pyridinium, N-oxide), while a bridgehead heteroatom keeps
    its lone pair for the fused system unless the repair asked for a double.
    The search is a bounded matching over the cluster atoms and stops after
    :data:`_CLAR_STEP_BUDGET` steps.  A ring with a haptic metal is left alone
    (its delocalised ligand must stay a circle, not a Kekule pattern).
    """
    if not rings or len(rings) > _CLAR_MAX_TOTAL_RINGS:
        return extras
    orders = {key: extra + 1 for key, extra in extras.items() if extra > 0}
    by_node = {}
    for key in edge_keys:
        for node in key:
            by_node.setdefault(node, []).append(key)
    ring_atoms = frozenset(node for ring in rings for node in ring)
    ring_count = {}
    for ring in rings:
        for node in ring:
            ring_count[node] = ring_count.get(node, 0) + 1
    ring_edges = {}
    for ring in rings:
        ring_edges[ring] = frozenset(
            key
            for node in ring
            for key in by_node.get(node, ())
            if key[0] in ring and key[1] in ring
        )
    single_only = _dative_bonds(elements, adjacency, lengths)
    budget = [_CLAR_STEP_BUDGET]

    clusters = _fused_ring_clusters(rings)
    # Fused clusters first: they hold every aromatic-ring miss in the corpus,
    # and a singleton ring is cheap to re-place afterwards (its seed evaluate
    # closes an already-aromatic ring in one step), so the shared budget is
    # spent where the fused search needs it most.  A metal-bound pyrazolide
    # whose doubles strand a carbon (ALIGUA) is a singleton that only aromatic
    # rings may fix, and the search below does exactly that.
    clusters.sort(key=lambda cluster: len(cluster) < 2)
    for cluster in clusters:
        if budget[0] <= 0:
            continue
        chosen = _clar_cluster_placement(
            cluster,
            elements,
            adjacency,
            ring_atoms,
            ring_count,
            ring_edges,
            by_node,
            orders,
            remaining,
            single_only,
            lengths,
            budget,
        )
        if chosen is None:
            continue
        movable = frozenset(key for ring in cluster for key in ring_edges[ring])
        for key in movable:
            if key in chosen:
                extras[key] = 1
            else:
                extras.pop(key, None)
    return extras


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
        # on a fusion bond.  Borazine and the h-BN sheet have none (their
        # borons carry no B=N), so they keep the delocalisation curve.
        kekule = all(
            node in donors or any(orders.get(key, 1) > 1 for key in by_node.get(node, ()))
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
