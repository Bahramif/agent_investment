"""
strategist.py
=============
The top-down strategist, on two horizons.

For each theme it holds two views:
  * strategic  — the long-term thesis. Slow-decaying, sets the ANCHOR.
  * tactical   — the short-term read. Fast-decaying, a bounded DEVIATION.

Each view carries a confidence (0..1) and a decay half-life. A view's live
strength = raw * confidence * (0.5 ** age/half_life), so a stale two-week
momentum call fades on its own while a multi-year thesis barely moves.

Net tilt = clip(strategic, strategic_cap) + clip(tactical, tactical_cap),
the whole thing clipped to the mandate's total_cap. The anchor can't be
flipped by a short-term signal; the short-term signal can only shade it.

`theme_budgets` turns those net tilts into a split of an OPEN bucket (e.g. the
25% of equities the user left to the agents) — tilting toward favored themes
around a neutral baseline, never all-in. That output is what replaces the raw
score-sum split in mandate_allocator's open space; instrument signals then fill
inside each theme. Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class HorizonView:
    raw: float                 # signed intent in tilt units (+0.30 = strongly favor)
    confidence: float          # 0..1
    age_days: float = 0.0      # how old the call is
    half_life_days: float = 30.0
    max_age_days: Optional[float] = None   # hard refresh deadline (None = decay-only)

    def strength(self) -> float:
        decay = 0.5 ** (self.age_days / self.half_life_days) if self.half_life_days > 0 else 1.0
        return self.raw * self.confidence * decay

    def is_stale(self) -> bool:
        """Decay alone never hits exactly zero — a thesis can sit at a faint
        but nonzero strength forever if nobody revisits it. This is the hard
        cutoff: past max_age_days the view is EXPIRED, not just quiet, and
        must not be trusted until an agent re-affirms it."""
        return self.max_age_days is not None and self.age_days > self.max_age_days


@dataclass
class ThemeStrategy:
    theme: str
    strategic: HorizonView     # long-term
    tactical: HorizonView      # short-term


@dataclass
class TiltCaps:
    strategic_cap: float = 0.15   # max long-term anchor deviation from neutral
    tactical_cap: float = 0.06    # max short-term deviation around the anchor
    total_cap: float = 0.15       # hard cap on the net tilt (from the mandate)


@dataclass
class ThemeTilt:
    theme: str
    strategic_tilt: float
    tactical_tilt: float
    net_tilt: float
    strategic_stale: bool = False
    tactical_stale: bool = False

    @property
    def is_stale(self) -> bool:
        return self.strategic_stale or self.tactical_stale


# A view's raw tilt is in [-1, 1]. These turn it into tilt units so that only the
# strongest, most confident view reaches its cap, so tilts keep their information
# instead of saturating. Used by BOTH the LLM and the heuristic reasoners.
STRATEGIC_VIEW_SCALE = 0.15   # = TiltCaps.strategic_cap
TACTICAL_VIEW_SCALE = 0.06    # = TiltCaps.tactical_cap


def _clip(x: float, cap: float) -> float:
    return max(-cap, min(cap, x))


def combine(strategy: ThemeStrategy, caps: TiltCaps) -> ThemeTilt:
    strat_stale = strategy.strategic.is_stale()
    tact_stale = strategy.tactical.is_stale()
    # Stale views contribute NOTHING — not "whatever decay left them at."
    # A view that's silently gone neutral because nobody refreshed it is a
    # governance event, so it's flagged on the tilt, not just quietly dropped.
    anchor = 0.0 if strat_stale else _clip(strategy.strategic.strength(), caps.strategic_cap)
    deviation = 0.0 if tact_stale else _clip(strategy.tactical.strength(), caps.tactical_cap)
    net = _clip(anchor + deviation, caps.total_cap)
    return ThemeTilt(strategy.theme, anchor, deviation, net, strat_stale, tact_stale)


def _apply_share_cap(shares: Dict[str, float], cap: float) -> Dict[str, float]:
    """Ensure no theme exceeds `cap` of the bucket; redistribute overflow."""
    out = dict(shares)
    for _ in range(len(out)):
        over = {t: w for t, w in out.items() if w > cap + 1e-12}
        if not over:
            break
        spill = sum(out[t] - cap for t in over)
        for t in over:
            out[t] = cap
        room = {t: w for t, w in out.items() if w < cap - 1e-12}
        base = sum(room.values()) or 1.0
        for t in room:
            out[t] += spill * room[t] / base
    return out


def theme_budgets(tilts: List[ThemeTilt], open_budget: float,
                  neutral: Optional[Dict[str, float]] = None,
                  max_theme_share: float = 0.60) -> Dict[str, float]:
    """Split `open_budget` across themes: neutral baseline, shifted by net tilts."""
    themes = [t.theme for t in tilts]
    if not themes or open_budget <= 0:
        return {}
    if neutral is None:
        neutral = {th: 1.0 / len(themes) for th in themes}     # equal baseline
    tilted = {t.theme: max(neutral[t.theme] * (1 + t.net_tilt), 0.0) for t in tilts}
    total = sum(tilted.values()) or 1.0
    raw_shares = {th: tilted[th] / total for th in themes}
    # BUG FIX: capping one theme's share only makes sense when there's a
    # SECOND theme to redistribute the spillover to — _apply_share_cap's
    # redistribution loop hands overflow to themes with headroom, and with
    # only one theme there IS no such headroom, so the capped-away amount
    # was silently discarded. With a single open theme it's getting the
    # whole open_budget regardless of tilt sign; capping it just strands
    # budget nobody else can claim. Confirmed live: a single-sector
    # recommendation came back with 0 names funded once the strategist
    # produced a real tilt for that lone theme — the allocator computed
    # 60% (max_theme_share), lost the other 40%, and risk_check correctly
    # rejected "total book != 100%" every time, exhausting retries.
    shares = raw_shares if len(themes) <= 1 else _apply_share_cap(raw_shares, max_theme_share)
    return {th: shares[th] * open_budget for th in themes}


@dataclass
class FreshnessViolation:
    theme: str
    horizon: str            # "strategic" | "tactical"
    prior_budget: float      # what this theme was getting BEFORE the view expired
    message: str


def freshness_gate(tilts: List[ThemeTilt], prior_budgets: Optional[Dict[str, float]] = None,
                   material_threshold: float = 0.03) -> List[FreshnessViolation]:
    """Turn stale flags into governance events, not just silently-changed numbers.

    combine() already zeroes a stale view's contribution, so the tilt itself is
    safe by default. This gate answers the separate question: did an expired
    view use to matter? If a theme was drawing real capital (>= material_threshold
    of the open budget) on the strength of a view that has now expired, that's a
    step-change the system should surface for a human or the owning agent to
    re-affirm — not something that should just quietly reallocate on its own.
    """
    violations: List[FreshnessViolation] = []
    prior = prior_budgets or {}
    for t in tilts:
        prior_b = prior.get(t.theme, 0.0)
        if t.strategic_stale and prior_b >= material_threshold:
            violations.append(FreshnessViolation(
                t.theme, "strategic", prior_b,
                f"'{t.theme}' strategic thesis expired (past max_age_days) while holding "
                f"{prior_b:.1%} of the open budget — needs re-affirmation before next rebalance"))
        if t.tactical_stale and prior_b >= material_threshold:
            violations.append(FreshnessViolation(
                t.theme, "tactical", prior_b,
                f"'{t.theme}' tactical read expired while holding {prior_b:.1%} "
                f"of the open budget — refresh or let it fall back to the strategic anchor"))
    return violations


if __name__ == "__main__":
    caps = TiltCaps(strategic_cap=0.15, tactical_cap=0.06, total_cap=0.15)

    strategies = [
        # AI: the long-term thesis hasn't been re-affirmed in 400 days, past its
        # 180-day refresh deadline -> EXPIRED, even though decay alone would have
        # left it a faint but nonzero (and easy-to-miss) residual strength.
        ThemeStrategy("AI",
                      HorizonView(raw=+0.30, confidence=0.80, age_days=400,
                                  half_life_days=365, max_age_days=180),
                      HorizonView(raw=-0.20, confidence=0.60, age_days=6,  half_life_days=14)),
        # Energy: mild long-term view, strong fresh short-term momentum. Fresh.
        ThemeStrategy("energy",
                      HorizonView(raw=+0.10, confidence=0.50, age_days=20,
                                  half_life_days=365, max_age_days=180),
                      HorizonView(raw=+0.35, confidence=0.70, age_days=4,  half_life_days=14)),
        # Financials: long-term underweight, negligible short-term view. Fresh.
        ThemeStrategy("financials",
                      HorizonView(raw=-0.15, confidence=0.60, age_days=10,
                                  half_life_days=365, max_age_days=180),
                      HorizonView(raw=+0.05, confidence=0.40, age_days=8,  half_life_days=14)),
    ]

    tilts = [combine(s, caps) for s in strategies]

    print("THEME TILTS (anchor + bounded deviation, all capped; STALE views zeroed)")
    print(f"  {'theme':12s} {'strategic':>10s} {'tactical':>10s} {'net':>8s}   flags")
    for t in tilts:
        flags = ("STRATEGIC EXPIRED " if t.strategic_stale else "") + \
                ("TACTICAL EXPIRED" if t.tactical_stale else "")
        print(f"  {t.theme:12s} {t.strategic_tilt:>+10.1%} {t.tactical_tilt:>+10.1%} "
              f"{t.net_tilt:>+8.1%}   {flags}")

    open_budget = 0.25   # the open slice of equities the user left to the agents
    budgets = theme_budgets(tilts, open_budget)
    print(f"\nOPEN-SPACE SPLIT (agents distribute the open {open_budget:.0%} of equities)")
    for th, b in sorted(budgets.items(), key=lambda kv: -kv[1]):
        print(f"  {th:12s} {b:6.2%} of total")
    print(f"  {'TOTAL':12s} {sum(budgets.values()):6.2%} of total")

    # What AI was drawing LAST cycle, before its thesis expired — this is what
    # the freshness gate checks against to decide whether the expiry mattered.
    prior_budgets = {"AI": 0.14, "energy": 0.07, "financials": 0.04}
    violations = freshness_gate(tilts, prior_budgets)

    print("\nFRESHNESS GATE")
    if violations:
        for v in violations:
            print(f"  ⚠ {v.message}")
    else:
        print("  no violations")

    print("\nRead: AI's long-term thesis is 400 days old against a 180-day refresh")
    print("deadline, so combine() zeroed its strategic contribution entirely —")
    print("no quiet 'faint but nonzero' anchor left over. Because AI was drawing a")
    print("material 14% of the open budget on the strength of that view, the")
    print("freshness gate raises it as a violation: someone must re-affirm the AI")
    print("thesis before its capital is reallocated on the agent's say-so alone.")
