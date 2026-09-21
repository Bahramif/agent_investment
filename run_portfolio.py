"""
run_portfolio.py
=================
Ties the three pieces together for one allocation cycle:

  mandate.yaml (user locks)  +  strategist theme tilts  +  analyst instrument
  signals  --allocate-->  final book

Division of labor, enforced by the code, not just described:
  * User-locked weights (equities 50%, tech_ai 25%, forex 20%, crypto 15%,
    cash 15%) are hard bounds — never crossed.
  * The OPEN 25% of equities (not locked to tech_ai) is split across themes
    by the STRATEGIST's two-horizon tilts (strategic anchor + tactical
    deviation, both capped) — this is the "AI vs energy" decision.
  * Inside each theme, ANALYST signals pick and size the actual names.
"""

from __future__ import annotations

from mandate_allocator import Signal, allocate, load_mandate
from strategist import HorizonView, ThemeStrategy, TiltCaps, combine


def main() -> None:
    mandate = {
        "total": 1.0,
        "default_limits": {"max_per_name": 0.10, "min_names": 3},
        "buckets": {
            "equities": {"set_by": "user", "weight": 0.50, "band": 0.05,
                        "children": {"tech_ai": {"set_by": "user", "weight": 0.25,
                                                 "band": 0.05, "limits": {"max_per_name": 0.08}}}},
            "forex":  {"set_by": "user", "weight": 0.20, "band": 0.04},
            "crypto": {"set_by": "user", "weight": 0.15, "band": 0.05,
                      "limits": {"max_per_name": 0.075}},
            "cash":   {"set_by": "user", "weight": 0.15, "band": 0.03},
        },
    }
    # In production: load_mandate("mandate.yaml") reads the identical structure.

    # --- 1. Strategist: two-horizon view on the THEMES competing for the open
    #        25% of equities (tech_ai is already user-locked, so it sits outside
    #        this contest — the strategist can only steer capital the user left
    #        undecided).
    caps = TiltCaps(strategic_cap=0.15, tactical_cap=0.06, total_cap=0.15)
    strategies = [
        ThemeStrategy("energy",
                     HorizonView(raw=+0.10, confidence=0.50, age_days=20, half_life_days=365),
                     HorizonView(raw=+0.35, confidence=0.70, age_days=4,  half_life_days=14)),
        ThemeStrategy("industrials",
                     HorizonView(raw=-0.05, confidence=0.40, age_days=15, half_life_days=365),
                     HorizonView(raw=+0.05, confidence=0.30, age_days=10, half_life_days=14)),
    ]
    theme_tilts = {t.theme: combine(t, caps) for t in strategies}

    # --- 2. Analyst signals: instrument picks, tagged with bucket/theme path.
    #        tech_ai names go straight to the locked bucket; energy/industrials
    #        names land in the open space the strategist just tilted.
    signals = [
        Signal("NVDA",  "equities.tech_ai", 0.9), Signal("MSFT", "equities.tech_ai", 0.7),
        Signal("GOOGL", "equities.tech_ai", 0.6), Signal("AVGO", "equities.tech_ai", 0.5),
        Signal("XOM",   "equities.energy",  0.6), Signal("FSLR", "equities.energy",  0.5),
        Signal("CAT",   "equities.industrials", 0.4),
        Signal("EURUSD", "forex", 0.7), Signal("USDJPY", "forex", 0.5), Signal("GBPUSD", "forex", 0.3),
        Signal("BTC", "crypto", 0.9), Signal("ETH", "crypto", 0.7), Signal("SOL", "crypto", 0.5),
    ]

    # --- 3. Allocate: strategist tilts steer the open-space theme split;
    #        analyst scores pick names within each theme; user locks untouched.
    book = allocate(mandate, signals, theme_tilts=theme_tilts)

    print("STRATEGIST THEME TILTS (feeding only the OPEN equity space)")
    for th, t in theme_tilts.items():
        print(f"  {th:12s} strategic {t.strategic_tilt:+.1%}  tactical {t.tactical_tilt:+.1%}"
              f"  -> net {t.net_tilt:+.1%}")

    print("\nFINAL BOOK (instrument -> % of total)")
    for name, w in sorted(book.weights.items(), key=lambda kv: -kv[1]):
        print(f"  {name:8s} {w:6.2%}")
    print(f"  {'TOTAL':8s} {sum(book.weights.values()):6.2%}")

    print("\nBUCKET ROLLUP")
    for b in book.buckets:
        tgt = f"{b.target:.0%}" if b.target is not None else "  —"
        print(f"  [{b.set_by:11s}] {b.path:30s} target {tgt:>4s}  allocated {b.allocated:6.2%}  {b.note}")

    if book.warnings:
        print("\nWARNINGS")
        for w in book.warnings:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
