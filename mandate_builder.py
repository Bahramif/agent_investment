"""
mandate_builder.py
===================
Interactive wizard that lets the END USER build their own investment mandate
by walking a category tree: pick which asset classes to fund, how much of the portfolio to weight each, then optionally
drill into sub-categories (e.g. equities -> stocks vs. options; stocks ->
which themes) before landing on the exact same mandate schema
mandate_allocator.load_mandate()/allocate() already consume. This is just an
interactive way to produce mandate.yaml instead of hand-editing it.

Division of labor is unchanged from mandate.yaml's own header: the user only
ever locks BUCKET weights here — instrument selection stays the agents' job
(mandate_allocator.py). Anything the user doesn't drill into, or leaves
under-allocated within a bucket, is OPEN space the agents fill by conviction
— exactly how forex/crypto and the un-drilled part of equities already work
today.

Run it directly:
    python3 mandate_builder.py

CATEGORY_TREE just below is the one thing you're expected to edit as the
firm's actual taxonomy grows (new themes, a real options desk, etc).

NOTE ON PATHS: whatever bucket path a sub-category ends up at (e.g.
"equities.stocks.energy") has to match the `path` your UniverseConfig
(data_ingestion.py) tags that theme's instruments with — this wizard prints
the exact paths it produced (render_summary()) so you can wire the two
together.
"""

from __future__ import annotations

from typing import Any, Dict, List

DEFAULT_BAND = 0.05          # 'band' isn't read anywhere yet (see mandate.yaml's
                             # own header comment) — it's schema-carried for
                             # future rebalance-drift logic, so the wizard just
                             # fills in a sane constant instead of prompting for it.

# =============================================================================
# CATEGORY_TREE: the selectable taxonomy. Each node is a dict that may have:
#   "children": {name: node, ...}   -- sub-categories to optionally drill into
#   "limits":   {...}               -- bucket-level overrides (merged over
#                                      mandate default_limits by the allocator)
#   "note":     "..."                -- shown to the user before they pick it
# A leaf (no "children" key) means: lock this bucket's weight, agents pick
# every instrument inside it freely — exactly how forex/crypto work today.
# =============================================================================
CATEGORY_TREE: Dict[str, Any] = {
    "equities": {
        "children": {
            "stocks": {
                "children": {
                    "tech_ai":     {},
                    "energy":      {},
                    "industrials": {},
                    "healthcare":  {},
                    "financials":  {},
                },
            },
            "options": {
                "note": "Not tradable yet — no options broker is wired into "
                        "alpaca_clients.py/execution_workflow.py. Locking a "
                        "weight here only affects the allocation math until "
                        "that execution path exists.",
            },
        },
    },
}

THEME_LABELS = {   # cosmetic only — anything missing falls back to title-casing
    "tech_ai": "AI / Tech",
}


def _label(name: str) -> str:
    return THEME_LABELS.get(name, name.replace("_", " ").title())


def _prompt_select(options: List[str], prompt: str, allow_none: bool = False) -> List[str]:
    """Numbered multi-select. Returns the subset of `options` the user picked,
    in the options' own order (not typed order)."""
    for i, name in enumerate(options, 1):
        print(f"    {i}) {_label(name)}")
    hint = "comma-separated numbers" + (", or Enter for none" if allow_none else "")
    while True:
        raw = input(f"  {prompt} ({hint}): ").strip()
        if not raw:
            if allow_none:
                return []
            print("  Pick at least one.")
            continue
        try:
            idxs = {int(tok) for tok in raw.replace(" ", "").split(",")}
        except ValueError:
            print("  Enter numbers separated by commas, e.g. 1,3")
            continue
        if not idxs.issubset(set(range(1, len(options) + 1))):
            print(f"  Valid choices are 1-{len(options)}.")
            continue
        return [options[i - 1] for i in sorted(idxs)]


def _prompt_weight(prompt: str, max_pct: float) -> float:
    """Returns a fraction of total (0-1), capped at max_pct."""
    while True:
        raw = input(f"  {prompt} (% of total, up to {max_pct * 100:.1f}%): ").strip().rstrip("%")
        try:
            pct = float(raw)
        except ValueError:
            print("  Enter a number, e.g. 25")
            continue
        if pct < 0 or pct / 100 > max_pct + 1e-9:
            print(f"  Must be between 0 and {max_pct * 100:.1f}.")
            continue
        return round(pct / 100, 6)


def _build_node(name: str, spec: Dict[str, Any], budget: float, depth: int) -> Dict[str, Any]:
    """Recursively builds one bucket dict. `budget` is the weight the parent
    prompt already locked in for this node."""
    node: Dict[str, Any] = {"set_by": "user", "weight": budget, "band": DEFAULT_BAND}
    if "limits" in spec:
        node["limits"] = spec["limits"]

    children_spec = spec.get("children")
    if not children_spec:
        return node                          # leaf — whole budget stays open for agents

    indent = "  " * depth
    print(f"\n{indent}{_label(name)} is {budget:.1%} of the portfolio.")
    drill = input(f"{indent}Split it into sub-categories now? [y/N]: ").strip().lower()
    if drill != "y":
        return node                          # declined — whole budget stays open

    names = list(children_spec.keys())
    for n in names:
        note = children_spec[n].get("note")
        if note:
            print(f"{indent}  ! {_label(n)}: {note}")
    picked = _prompt_select(
        names, f"Which {_label(name)} sub-categories do you want to lock a weight for?",
        allow_none=True)
    if not picked:
        return node

    children: Dict[str, Any] = {}
    remaining = budget
    for n in picked:
        w = _prompt_weight(f"{indent}{_label(n)} weight", remaining)
        children[n] = _build_node(n, children_spec[n], w, depth + 1)
        remaining = round(remaining - w, 6)
    if remaining > 1e-6:
        print(f"{indent}{remaining:.1%} of {_label(name)} stays open — agents place it "
              f"across any {_label(name)} names they have conviction in.")
    node["children"] = children
    return node


def build_mandate_interactive() -> Dict[str, Any]:
    print("=" * 70)
    print("MANDATE BUILDER — pick your asset classes, lock their weights,")
    print("then optionally drill into sub-categories. Anything you leave")
    print("open (unpicked, or budget left over) is the AGENTS' call —")
    print("instrument selection is always theirs, never yours.")
    print("=" * 70)

    top_level = list(CATEGORY_TREE.keys())
    print("\nHigh-level categories:")
    picked = _prompt_select(top_level, "Which do you want to invest in?")

    buckets: Dict[str, Any] = {}
    remaining_total = 1.0
    for name in picked:
        w = _prompt_weight(f"\n{_label(name)} weight", remaining_total)
        buckets[name] = _build_node(name, CATEGORY_TREE[name], w, depth=1)
        remaining_total = round(remaining_total - w, 6)

    if remaining_total > 1e-6:
        add_cash = input(f"\n{remaining_total:.1%} of the portfolio is unallocated — "
                         f"hold it as cash? [Y/n]: ").strip().lower()
        if add_cash != "n":
            buckets["cash"] = {"set_by": "user", "weight": remaining_total, "band": 0.03}
            remaining_total = 0.0

    mandate = {
        "total": 1.0,
        "default_limits": {"max_per_name": 0.10, "min_names": 3},
        "buckets": buckets,
    }
    if remaining_total > 1e-6:
        print(f"\n! {remaining_total:.1%} left fully unallocated (not even cash) — "
              f"that capital sits outside this mandate until you re-run the "
              f"wizard or hand-edit the result.")
    return mandate


def render_summary(mandate: Dict[str, Any]) -> str:
    """Flat 'bucket.path  weight%' listing — use it to line up UniverseConfig
    paths (data_ingestion.py) with what this wizard produced."""
    lines = []

    def walk(nodes: Dict[str, Any], prefix: str) -> None:
        for name, node in nodes.items():
            path = f"{prefix}.{name}" if prefix else name
            lines.append(f"  {path:30s} {node['weight']:6.1%}")
            if "children" in node:
                walk(node["children"], path)

    walk(mandate["buckets"], "")
    return "\n".join(lines)


def save_mandate(mandate: Dict[str, Any], path: str = "mandate.yaml") -> None:
    """Replaces the mandate's own keys and keeps every other block in the file
    (rulebook, day_trading, ...), which the wizard knows nothing about."""
    import os
    import yaml
    existing: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as fh:
            existing = yaml.safe_load(fh) or {}
    with open(path, "w") as fh:
        yaml.safe_dump({**existing, **mandate}, fh, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    import yaml

    mandate = build_mandate_interactive()

    print("\n" + "=" * 70)
    print("YOUR MANDATE")
    print("=" * 70)
    print(render_summary(mandate))

    dest = input("\nSave to mandate.yaml? This overwrites the existing file. [y/N]: ").strip().lower()
    if dest == "y":
        save_mandate(mandate)
        print("Saved to mandate.yaml")
    else:
        print("\n--- mandate.yaml contents (not saved) ---")
        print(yaml.safe_dump(mandate, sort_keys=False, default_flow_style=False))
