"""Rigorous Option-B (issue #14) validation on a MULTI-PAIR structure.

Uses Ag2Cr2O7.cif: a periodic oxide that (a) renders two distinct bond pairs
(Ag|O, Cr|O) and (b) really grows from a seed selection, so the preservation
assertions below are not vacuous.  The scenario mimics the user's report:
  1. Build the full bond set (both pairs).
  2. Disable ALL connectivity pairs, then re-enable only ONE of them.
  3. Grow from a seed selection.
  4. Assert the previously-built OTHER bond pairs are NOT lost, the total bond
     object count did not decrease, and the structure gained atoms.
"""
import sys, os, json, bpy

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _env  # noqa: F401  (gemmi + add-on parent on sys.path)
import CrystalViz
CrystalViz.register()

CIF = _env.CIF_DIR
scene = bpy.context.scene
scene.cgn_chemdraw = False

FAILURES = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)

def bond_pairs(root):
    return sorted({
        str(o.get("cgn_pair", ""))
        for o in root.objects
        if o.get("cgn_role") == "bonds" and not o.get("cgn_aromatic_torus")
    })

def bond_object_count(root):
    return sum(1 for o in root.objects if o.get("cgn_role") == "bonds")

def atom_count(root):
    return len(json.loads(root.get("cgn_instances", "[]"))) // 4

# --- import multi-pair structure; build full bond set ---
scene.cgn_connectivity_pairs.clear()
bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "Ag2Cr2O7.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)
bpy.ops.cgn.rebuild_connectivity("INVOKE_DEFAULT")
bpy.context.view_layer.update()

full_pairs = bond_pairs(root)
full_bond_count = bond_object_count(root)
full_atoms = atom_count(root)
print("full bond pairs:", full_pairs, "| bond objects:", full_bond_count, "| atoms:", full_atoms)
check("setup: MULTIPLE distinct bond pairs present", len(full_pairs) > 1, str(full_pairs))

# --- find a re-enabled pair that actually grows new atoms from seeds ---
grew_any = False
for pair_def in [
    {"AG", "O"},
    {"CR", "O"},
]:
    # restore full bond set first so each attempt starts from the same base
    for p in scene.cgn_connectivity_pairs:
        p.enabled = False
    re_enabled = None
    for p in scene.cgn_connectivity_pairs:
        if {str(p.element_a).upper(), str(p.element_b).upper()} == pair_def:
            p.enabled = True
            re_enabled = p
            break
    root["cgn_selected_indices"] = json.dumps(list(range(atom_count(root))))
    res = bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
    bpy.context.view_layer.update()
    a = atom_count(root)
    key = f"{re_enabled.element_a}|{re_enabled.element_b}".upper() if re_enabled else None
    print(f"  grow with only {key}: atoms {full_atoms} -> {a}, result {res}")
    if a > full_atoms:
        grew_any = True
        grew_key = key
        pairs_after = bond_pairs(root)
        count_after = bond_object_count(root)
        break
check("setup: at least one pair grows new atoms from seeds", grew_any)
if not grew_any:
    print("No pair grew atoms; asserting preservation on the last reduced-pair state.")
    pairs_after = bond_pairs(root)
    count_after = bond_object_count(root)
    grew_key = re_enabled.element_a + "|" + re_enabled.element_b

after_atoms = atom_count(root)
print("after grow -> atoms:", after_atoms, "| bond objects:", count_after, "| pairs:", pairs_after)

# Core Option-B assertions:
lost = sorted(set(full_pairs) - set(pairs_after))
check("grow preserved the OTHER (non-re-enabled) bond pairs",
      set(full_pairs) <= set(pairs_after),
      f"lost={lost}")
check("grow did not remove any bond objects", count_after >= full_bond_count,
      f"before={full_bond_count} after={count_after}")

print()
print("TOTAL FAILURES:", len(FAILURES))
CrystalViz.unregister()
if FAILURES:
    print("RESULT: FAIL")
    sys.exit(1)
print("RESULT: ALL PASS")
