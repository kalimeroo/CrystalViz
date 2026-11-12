"""Validate Option B for issue #14: grow must APPEND bonds and never tear down
bonds built under earlier connectivity settings.

Scenario (mimics the user's report):
  1. Build a fuller structure and a full set of bonds.
  2. Disable all connectivity pairs, then re-enable only a subset.
  3. Grow from a seed selection.
  4. Assert the previously-built bond objects/pairs are NOT lost (Option B fix)
     and the structure gained atoms.

Fills build no bonds, so the test builds the full bond set explicitly with
``_bond_objects`` (the same call the grow path uses) before exercising the
append-only grow behaviour.  "Reset Default" only restores the connectivity
windows; it never tears down bonds either.
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

def atom_object_count(root):
    return sum(1 for o in root.objects if o.get("cgn_role") == "atoms")

# --- fresh import; fill to a fuller cell; build the full bond set ---
scene.cgn_connectivity_pairs.clear()
bpy.ops.cgn.import_cif(filepath=os.path.join(CIF, "Brookite.cif"))
root = bpy.data.collections.get(scene.cgn_active_structure)
bpy.ops.cgn.fill_cell("INVOKE_DEFAULT")
root = bpy.data.collections.get(scene.cgn_active_structure)
bpy.ops.cgn.rebuild_connectivity("INVOKE_DEFAULT")
# A fill displays atoms only; build the bond set explicitly so the test can
# prove grow appends to bonds that already existed.
CrystalViz._bond_objects(
    root, CrystalViz._structure_atoms(root), scene.cgn_connectivity_pairs
)
bpy.context.view_layer.update()

full_pairs = bond_pairs(root)
full_bond_count = bond_object_count(root)
full_atoms = atom_count(root)
print("full bond pairs:", full_pairs, "| bond objects:", full_bond_count, "| atoms:", full_atoms)
check("setup: bonds exist", full_bond_count > 0, f"count={full_bond_count}")

# --- simulate the user: disable all, re-enable only one pair (Ti|O) ---
for p in scene.cgn_connectivity_pairs:
    p.enabled = False
re_enabled = None
for p in scene.cgn_connectivity_pairs:
    if {str(p.element_a).upper(), str(p.element_b).upper()} == {"TI", "O"}:
        p.enabled = True
        re_enabled = p
        break
check("setup: found a Ti|O pair to re-enable", re_enabled is not None)
print("re-enabled pair:", re_enabled.element_a, re_enabled.element_b if re_enabled else None)

# --- grow from a seed selection under the reduced pair set ---
root["cgn_selected_indices"] = json.dumps(list(range(5)))
res = bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
print("grow result:", res)
bpy.context.view_layer.update()

after_atoms = atom_count(root)
pairs_after = bond_pairs(root)
count_after = bond_object_count(root)
print("after grow -> atoms:", after_atoms, "| bond objects:", count_after, "| pairs:", pairs_after)

check("grow added atoms", after_atoms > full_atoms, f"{full_atoms} -> {after_atoms}")
check(
    "grow preserved the full original bond-pair set (Option B)",
    set(full_pairs) <= set(pairs_after),
    f"lost={sorted(set(full_pairs) - set(pairs_after))}",
)
check(
    "grow did not remove bond objects",
    count_after >= full_bond_count,
    f"before={full_bond_count} after={count_after}",
)

# --- flat outliner: repeated grows must not accumulate atom objects ---
atom_objs_before = atom_object_count(root)
for _ in range(3):
    root["cgn_selected_indices"] = json.dumps(list(range(5)))
    bpy.ops.cgn.grow_further("INVOKE_DEFAULT", all_atoms=False)
    bpy.context.view_layer.update()
check(
    "atom object count stays flat across repeated grows",
    atom_object_count(root) == atom_objs_before,
    f"before={atom_objs_before} after={atom_object_count(root)}",
)

# --- Reset Default: restore default pairs, KEEP existing bonds (non-destructive) ---
bond_objs_before_reset = bond_object_count(root)
pairs_before_reset = bond_pairs(root)
bpy.ops.cgn.rebuild_connectivity("INVOKE_DEFAULT")
bpy.context.view_layer.update()
print("after Reset Default -> config pairs:")
for p in scene.cgn_connectivity_pairs:
    print("  ", p.element_a, p.element_b, "enabled=", p.enabled)
check(
    "Reset Default re-enabled multiple default pairs",
    sum(1 for p in scene.cgn_connectivity_pairs if p.enabled) > 1,
    f"enabled={sum(1 for p in scene.cgn_connectivity_pairs if p.enabled)}",
)
check(
    "Reset Default kept existing bond objects (no teardown)",
    bond_object_count(root) >= bond_objs_before_reset,
    f"before={bond_objs_before_reset} after={bond_object_count(root)}",
)
check(
    "Reset Default did not lose existing bond pairs",
    set(pairs_before_reset) <= set(bond_pairs(root)),
    f"lost={sorted(set(pairs_before_reset) - set(bond_pairs(root)))}",
)

print()
print("TOTAL FAILURES:", len(FAILURES))
CrystalViz.unregister()
if FAILURES:
    print("RESULT: FAIL")
    sys.exit(1)
print("RESULT: ALL PASS")
