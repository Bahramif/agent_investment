"""
mandate_allocator.py
====================
Turns a user mandate + agent conviction scores into a final target book.

The contract:
  * User-locked bucket weights are HARD bounds. The allocator never crosses them.
  * Whatever the user leaves open, the agents fill.
  * Inside every bucket the agents distribute by conviction, but always within
    the sub-limits (per-name cap, diversification floor).
  * Instrument selection is entirely the agents' — the user never names a ticker.

Within an already-selected bucket of names, sizing tries a REAL mean-variance
optimizer first (portfolio_optimizer.py — genuinely covariance-aware, sees
when top picks would move in lockstep), falling back to a transparent,
capped, score-weighted "water-fill" (stdlib only) whenever the optimizer
can't run (no scipy, too little return history, solver failure) — see
_allocate_within_bucket below. The constraint plumbing around either one
(locked buckets, per-name caps, theme tilts) is the part that stays fixed.

`load_mandate` needs PyYAML (`pip install pyyaml`); everything else is stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from strategist import ThemeTilt, theme_budgets


@dataclass
class Signal:
    """One agent conviction on one instrument, tagged with its bucket path."""
    name: str          # ticker / pair, e.g. "NVDA", "EURUSD", "BTC"
    path: str          # bucket path the agent assigns it to, e.g. "equities.tech_ai"
    score: float       # conviction, higher = more
    rationale: str = ""    # one sentence: why this conviction — surfaced to the end user,
                           # not just an internal number (see claude_reasoning.py /
                           # analyst_reasoning.py, where this is actually generated)
    risk_note: str = ""    # one sentence: the main risk to this call
    # The four raw inputs the conviction score was built from (see
    # analyst_reasoning.InstrumentSnapshot) — carried through here purely for
    # UI transparency ("what actually drove this pick"), not consumed by the
    # allocation math itself.
    momentum_1m: float = 0.0
    news_sentiment: float = 0.0
    quality_score: float = 0.0
    volatility: float = 0.0
    # Analyst forecast: expected_return is a decimal (0.08 = +8%); rating is
    # a Buy/Hold/Sell tier consistent with it. REAL Wall Street analyst
    # consensus (fundamentals.get_analyst_forecast) when coverage exists,
    # falling back to a heuristic/Claude-generated estimate when it
    # doesn't — see analyst_reasoning.py / claude_reasoning.py.
    expected_return: float = 0.0
    rating: str = ""
    analyst_count: int = 0   # 0 when expected_return/rating came from the heuristic/Claude
                            # fallback rather than real analyst coverage — see analyst_reasoning.py
    # Real daily returns over analyst_reasoning's lookback window, carried
    # through from InstrumentSnapshot.returns_window so a real optimizer
    # (see _optimize_weights below) can build an ACTUAL covariance matrix
    # across whichever names survive top-N selection, instead of only ever
    # seeing each name's own volatility in isolation.
    returns_window: List[float] = field(default_factory=list)
    # Buy-the-dip read (analyst_reasoning.dip_opportunity): 0 = not a dip
    # opportunity; >0 = a quality name trading well below its recent high
    # with an intact outlook. drawdown_6m is the distance from that high.
    dip_score: float = 0.0
    drawdown_6m: float = 0.0


@dataclass
class BucketReport:
    path: str
    set_by: str
    target: float | None
    allocated: float
    note: str = ""


@dataclass
class Book:
    weights: Dict[str, float] = field(default_factory=dict)   # instrument -> weight of total
    buckets: List[BucketReport] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # name -> one-line explanation of the SIZING decision (why this % and
    # not equal-weight), populated by _allocate_within_bucket at the moment
    # weight is actually decided — a separate question from "why was this
    # name picked at all" (that's Signal.rationale, driven by conviction).
    weight_rationale: Dict[str, str] = field(default_factory=dict)
    # name -> {"rationale", "risk_note", "momentum_1m", "news_sentiment",
    # "quality_score", "volatility", "conviction", "expected_return", "rating"}
    # — populated by allocate() below, not by add()/water-fill (those don't
    # know about any of this, only weight).
    notes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add(self, name: str, w: float) -> None:
        if w > 1e-9:
            self.weights[name] = self.weights.get(name, 0.0) + w


def _under(path: str, prefix: str) -> bool:
    return prefix == "" or path == prefix or path.startswith(prefix + ".")


def _top_n(items: List[Tuple[str, float]], n: Optional[float]) -> List[Tuple[str, float]]:
    """Keep only the N highest-conviction candidates before water-filling.
    Without this, a large candidate pool with closely-clustered scores (a
    real symptom of the heuristic scorer at S&P-500 scale — see
    universe.py) spreads the budget near-equally across dozens of names
    instead of concentrating on the agents' actual top picks.

    BUG FIX #1: a mandate arriving through a real deployed query() call has
    already round-tripped through JSON/proto, where EVERY number becomes a
    float — confirmed live: a literal int 12 came back as 12.0. list[:12.0]
    raises "slice indices must be integers" — int()-coerce here rather than
    trusting the caller's type, since this is exactly the kind of mandate
    value (like min_names, max_per_name) that legitimately arrives this way.

    BUG FIX #2, confirmed live: ranking among ALL items (including
    zero-or-negative-scored ones) can let a candidate with score <= 0 take
    one of the N slots. _water_fill assigns that candidate EXACTLY zero
    (it's a proportional split — 0 conviction gets 0 share of the budget,
    every time) and then filters it out entirely (`w > 1e-9`), so the
    caller silently ends up funding FEWER than N names even when N genuinely
    positive-scoring candidates existed — e.g. "top 5" requested, one of the
    raw top-5-by-rank has conviction 0.0, only 4 actually get capital.
    Ranking among positive-scored candidates only means a weaker-but-real
    candidate takes that slot instead, so N slots go to N fundable names
    whenever that many exist. Only when FEWER than N candidates have any
    positive conviction at all do we fall back to the full pool — matching
    _water_fill's own equal-weight-when-nobody-has-conviction behavior
    rather than returning nothing to allocate."""
    if not n or n <= 0:
        return items
    n = int(n)
    positive = [item for item in items if item[1] > 0]
    pool = positive if positive else items
    if len(pool) <= n:
        return pool
    return sorted(pool, key=lambda x: -x[1])[:n]


RISK_AVERSION = 2.0   # tunable: how hard per-name volatility discounts water-fill share


def _risk_adjusted_score(s: "Signal") -> float:
    """WEIGHT OPTIMIZATION: an inverse-volatility tilt on top of raw
    conviction, applied only at the water-fill stage (i.e. how much of the
    ALREADY-SELECTED budget each name gets) — NOT at _top_n, which still
    ranks/selects by raw conviction alone. Two names with equal conviction
    split budget in inverse proportion to their volatility, so the calmer
    one gets the bigger slice of the same conviction.

    Deliberately simple, same honesty caveat as analyst_reasoning.py's own
    per-instrument volatility penalty and this module's docstring: this is
    risk-AWARE sizing using each name's OWN volatility, not a real
    portfolio-level optimizer that weighs correlations between holdings
    (mean-variance/Black-Litterman remains the flagged production upgrade)."""
    return s.score / (1.0 + max(s.volatility, 0.0) * RISK_AVERSION)


def _weight_rationale(sig: "Signal", weight: float, budget: float, cap: float,
                      peers: List["Signal"], equal_weight_won: bool = False) -> str:
    """One-line answer to "why THIS weight" — not "why this name" (that's
    Signal.rationale). Benchmarks the actual weight against an equal-weight
    baseline (what this name would get if conviction/return/risk didn't
    matter at all) and names the driver: expected return relative to the
    other names it was weighed against, then volatility as a secondary
    factor — the same two inputs portfolio_optimizer.py's objective
    actually optimizes over (mu, Sigma), stated in plain language instead
    of left implicit in a percentage.

    CAP-BOUND CASE, handled explicitly rather than left to produce a
    misleading message: when the per-name cap binds (uncapped equal-weight
    would exceed it), EVERY name in a small/undiversified group can land at
    exactly the same capped weight regardless of how much better one name's
    numbers are — confirmed live: a 3-name group under a 10% cap put NVDA
    (the best expected return) at the same 10% as the others, and the
    naive "below baseline" framing read as if its strong numbers were a
    NEGATIVE, backwards from what actually happened. Report the cap as the
    reason instead of a fabricated relative-attractiveness story.

    equal_weight_won: True when _allocate_within_bucket's guardrail
    overrode the optimizer/water-fill result because a plain equal split
    projected a HIGHER weighted return for this group than the sizing
    method's own output — said outright rather than left to look like an
    ordinary "in line with baseline" coincidence."""
    n = len(peers)
    if n == 0 or budget <= 1e-12:
        return ""
    uncapped_equal_weight = budget / n
    equal_weight = min(uncapped_equal_weight, cap)
    avg_return = sum(p.expected_return for p in peers) / n
    avg_vol = sum(p.volatility for p in peers) / n
    ret_diff = sig.expected_return - avg_return
    vol_diff = sig.volatility - avg_vol

    drivers = []
    if abs(ret_diff) >= 0.01:
        drivers.append(f"expected return of {sig.expected_return:+.1%} vs. this group's {avg_return:+.1%} average")
    if abs(vol_diff) >= 0.005:
        cmp_word = "higher" if vol_diff > 0 else "lower"
        drivers.append(f"{cmp_word} volatility ({sig.volatility:.1%} vs. {avg_vol:.1%} avg)")
    driver_text = ", and ".join(drivers) if drivers else "roughly average expected return and risk within this group"

    if equal_weight_won:
        return (f"Sized at {weight:.1%}, an equal split across this {n}-name group — the optimizer's own "
                f"tilted weights would have projected a LOWER combined expected return than just splitting "
                f"evenly, so the equal split was used instead; {driver_text}.")

    at_cap = weight >= cap - 1e-9 and cap < uncapped_equal_weight - 1e-9
    if at_cap:
        return (f"Sized at {weight:.1%} — hit the {cap:.0%} per-name position cap "
                f"(this group's uncapped equal-weight baseline would be {uncapped_equal_weight:.1%}); "
                f"{driver_text}.")

    tilt_pp = (weight - equal_weight) * 100
    if abs(tilt_pp) < 0.5:
        return (f"Sized at {weight:.1%} — in line with the {equal_weight:.1%} equal-weight baseline "
                f"for this {n}-name group; {driver_text}.")
    direction = "above" if tilt_pp > 0 else "below"
    return (f"Sized at {weight:.1%} — {abs(tilt_pp):.1f}pp {direction} the {equal_weight:.1%} "
            f"equal-weight baseline for this {n}-name group, driven mainly by {driver_text}.")


# DIVERSIFICATION FLOOR: every SELECTED name gets at least this fraction of an
# equal-weight share (0.5 -> with 12 names, >= 4.2% each). Without it the
# mean-variance optimizer filled the top names to the per-name cap and gave the
# rest zero, so a portfolio set to "12 stocks" held 6 (and "6" held 3). 0 turns
# the floor off (the old behaviour).
MIN_WEIGHT_FRACTION = 0.5


def _water_fill_with_floor(items: List[Tuple[str, float]], budget: float, cap: float,
                           floor: float) -> Dict[str, float]:
    """Everyone gets `floor` first; the rest is water-filled by score with
    each name's remaining headroom (cap - floor)."""
    if floor <= 1e-12 or not items:
        return _water_fill(items, budget, cap)
    base = {n: floor for n, _ in items}
    extra = _water_fill(items, budget - floor * len(items), max(cap - floor, 0.0))
    return {n: base[n] + extra.get(n, 0.0) for n in base}


def _allocate_within_bucket(signals_subset: List["Signal"], budget: float, cap: float,
                            book: "Book") -> Dict[str, float]:
    """WEIGHT OPTIMIZATION entry point for an already-SELECTED set of names
    (selection itself — _top_n, theme grouping — is untouched by this).
    Tries the real mean-variance optimizer first (portfolio_optimizer.py) —
    genuinely covariance-aware, so two "top pick" names that move in
    lockstep don't both get full-size weight the way a per-name-only risk
    tilt would allow — falling back to the pre-existing risk-adjusted
    water-fill whenever the optimizer can't run (scipy missing, too little
    return history, fewer than 2 usable names, solver non-convergence — see
    portfolio_optimizer.optimize_weights's own docstring). Callers don't
    need to know which path ran: on any failure this reduces to EXACTLY
    what shipped before this function existed.

    Also records WHY each weight is what it is (book.weight_rationale) —
    computed the same way regardless of which path produced the weight,
    since the question "how does this compare to equal-weight, and why"
    is meaningful either way.

    GUARDRAIL, always applied regardless of which path produced `w`: never
    hand back a weighting whose OWN projected return (the weighted sum of
    each name's expected_return — plain portfolio-expected-return math, no
    risk term) is worse than simply splitting the same budget equally
    across the same names. Mean-variance legitimately trades some return
    for lower correlation/volatility, and the risk-adjusted water-fill
    fallback sizes by conviction, not raw expected return either — both
    can, in principle, land on a weighting an equal split would have beaten
    on return alone. Comparing against that plain baseline and taking
    whichever is actually better keeps the promise every per-name
    "Sized at X% ... equal-weight baseline ..." rationale already makes:
    that equal-weight is the thing being measured against, not just a
    number quoted in a sentence."""
    n_sel = len(signals_subset)
    floor = min(cap, MIN_WEIGHT_FRACTION * budget / n_sel) if n_sel else 0.0
    w = None
    try:
        from portfolio_optimizer import optimize_weights
        w = optimize_weights(signals_subset, budget, cap, floor=floor)
    except Exception:
        w = None
    if w is None:
        w = _water_fill_with_floor([(s.name, _risk_adjusted_score(s)) for s in signals_subset], budget, cap, floor)

    by_name = {s.name: s for s in signals_subset}
    n = len(signals_subset)
    equal_weight_won = False
    if n > 0 and budget > 1e-12:
        equal_w = min(budget / n, cap)
        equal_weights = {s.name: equal_w for s in signals_subset}
        opt_return = sum(w.get(name, 0.0) * sig.expected_return for name, sig in by_name.items())
        equal_return = sum(equal_weights[name] * sig.expected_return for name, sig in by_name.items())
        if equal_return > opt_return + 1e-9:
            w = equal_weights
            equal_weight_won = True

    for name, wt in w.items():
        sig = by_name.get(name)
        if sig is not None:
            book.weight_rationale[name] = _weight_rationale(sig, wt, budget, cap, signals_subset,
                                                             equal_weight_won=equal_weight_won)
    return w


def _water_fill(items: List[Tuple[str, float]], budget: float, cap) -> Dict[str, float]:
    """Distribute `budget` across scored items, each capped at `cap` — a
    single float applied uniformly to every item (every existing call site
    inside this module), OR a dict mapping item name -> its own cap, used
    by _alloc_children's cross-theme redistribution below, where each
    "item" is actually a whole THEME and its true ceiling is however many
    of its survivors times the per-name cap — a number that differs theme
    to theme, not one shared constant.

    BUG FIX, confirmed live: "No recommendation — reject: ['total book
    93.33% != 100%']" — risk_check_node correctly vetoing a proposal where
    a theme's tilt/pro-rata-derived budget share exceeded what its (post
    global-top-N-cut) survivor count could legally absorb under the
    per-name cap. The excess was silently STRANDED — never offered to
    other themes that still had spare room under THEIR OWN cap — dropping
    the book below 100% for no economic reason. The zero-conviction branch
    below had the exact same latent bug (`min(budget/n, cap)` with no
    redistribution of the excess when budget/n exceeded cap) — both are the
    same problem: SOME cap binds before the whole budget is placed, and the
    leftover must cascade to whoever still has room, possibly through
    several rounds as each additional cap binds in turn. Equal weight when
    nobody has conviction is now just this same iteration run with every
    score treated as equal, instead of a separate shortcut that skipped it."""
    if budget <= 1e-12 or not items:
        return {}
    cap_of = cap if isinstance(cap, dict) else {n: cap for n, _ in items}
    raw_scores = {n: max(s, 0.0) for n, s in items}
    scores = raw_scores if sum(raw_scores.values()) > 0 else {n: 1.0 for n in raw_scores}
    weights = {n: 0.0 for n in scores}
    active = set(scores)
    remaining = budget
    while active and remaining > 1e-12:
        tot = sum(scores[n] for n in active)
        proposed = {n: remaining * scores[n] / tot for n in active}
        newly_capped = [n for n in active if weights[n] + proposed[n] >= cap_of[n] - 1e-12]
        if newly_capped:
            for n in newly_capped:
                weights[n] = cap_of[n]
                active.discard(n)
            remaining = budget - sum(weights.values())
        else:
            for n in active:
                weights[n] += proposed[n]
            remaining = 0.0
    return {n: w for n, w in weights.items() if w > 1e-9}


def allocate(mandate: Dict[str, Any], signals: List[Signal],
            theme_tilts: Optional[Dict[str, ThemeTilt]] = None) -> Book:
    """theme_tilts: strategist output keyed by theme name (e.g. "energy", "AI").
    Used to split OPEN space (no named children) across themes before the
    per-instrument water-fill runs inside each theme. Bucket paths the user
    already locked (e.g. tech_ai) are untouched by this — tilts only steer
    capital the user left undecided."""
    book = Book()
    defaults = mandate.get("default_limits", {})
    _alloc_children(mandate["buckets"], mandate.get("total", 1.0), "", signals,
                    defaults, book, theme_tilts or {})
    # Rationale/risk/scores/forecast aren't part of the weight math at all —
    # attach them after the fact, for whichever names actually made it into
    # the book, so the UI can show WHY a name was picked, not just its %.
    for s in signals:
        if s.name in book.weights:
            book.notes[s.name] = {
                "rationale": s.rationale, "risk_note": s.risk_note,
                "momentum_1m": s.momentum_1m, "news_sentiment": s.news_sentiment,
                "quality_score": s.quality_score, "volatility": s.volatility,
                "conviction": s.score,
                "expected_return": s.expected_return, "rating": s.rating,
                "analyst_count": s.analyst_count,
                "dip_score": s.dip_score, "drawdown_6m": s.drawdown_6m,
                "weight_rationale": book.weight_rationale.get(s.name, ""),
            }
    return book


def _alloc_children(children: Dict[str, Any], budget: float, parent_path: str,
                    signals: List[Signal], defaults: Dict[str, Any], book: Book,
                    theme_tilts: Dict[str, ThemeTilt]) -> None:
    locked = {n: node for n, node in children.items()
              if node.get("set_by") == "user" and "weight" in node}
    named_agent = {n: node for n, node in children.items() if n not in locked}

    locked_sum = sum(node["weight"] for node in locked.values())
    open_rem = round(budget - locked_sum, 12)
    if open_rem < -1e-9:
        book.warnings.append(
            f"{parent_path or 'portfolio'}: locked children sum {locked_sum:.2%} "
            f"exceed budget {budget:.2%} — mandate over-allocated")
        open_rem = 0.0

    for name, node in locked.items():
        _alloc_bucket(name, node, node["weight"], _join(parent_path, name),
                      signals, defaults, book, theme_tilts)

    if named_agent:                                     # split open space by agent conviction
        shares = {}
        for name in named_agent:
            p = _join(parent_path, name)
            shares[name] = sum(s.score for s in signals if _under(s.path, p))
        total = sum(shares.values())
        for name, node in named_agent.items():
            p = _join(parent_path, name)
            b = open_rem * (shares[name] / total) if total > 0 else open_rem / len(named_agent)
            _alloc_bucket(name, node, b, p, signals, defaults, book, theme_tilts)
    elif open_rem > 1e-9:                                # open space with no named children:
        locked_paths = [_join(parent_path, n) for n in locked]   # agents place it directly
        pool = [s for s in signals
                if _under(s.path, parent_path) and not any(_under(s.path, lp) for lp in locked_paths)]
        cap = defaults.get("max_per_name", 1.0)
        top_n = defaults.get("top_n")
        open_path = _join(parent_path, "*open*")

        # GLOBAL top-N, confirmed as the intended behavior: "max # of
        # stocks" caps the TOTAL number of positions across every open
        # theme/sector COMBINED, not per theme — applying _top_n() inside
        # each theme's own water-fill below (the previous behavior) meant
        # selecting 3 sectors at top_n=5 funded 5 PER sector, 15 total,
        # instead of 5 total. Cut the pool to its global top-N ONCE here,
        # before grouping by theme, so every theme below competes for
        # budget using only names that already survived the one shared cut
        # — a theme whose stocks don't crack the global top-N simply gets
        # none of the open budget, regardless of its macro tilt.
        if top_n:
            survivors = {name for name, _ in _top_n([(s.name, s.score) for s in pool], top_n)}
            pool = [s for s in pool if s.name in survivors]

        themes = _group_by_theme(pool, parent_path)
        tilted_themes = {th: theme_tilts[th] for th in themes if th in theme_tilts}
        if tilted_themes:                                # strategist steers the theme split
            tilts = list(tilted_themes.values())
            neutral = {t.theme: len(themes[t.theme]) for t in tilts}
            tot = sum(neutral.values()) or 1.0
            neutral = {th: n / tot for th, n in neutral.items()}
            budgets = theme_budgets(tilts, open_rem, neutral=neutral)
            for th in themes:                            # untilted themes: leftover pro-rata
                budgets.setdefault(th, 0.0)
            untilted = [th for th in themes if th not in tilted_themes]
            if untilted:
                spare = open_rem - sum(budgets.values())
                n_inst = sum(len(themes[th]) for th in untilted) or 1
                for th in untilted:
                    budgets[th] = spare * len(themes[th]) / n_inst

            # BUG FIX, confirmed live ("total book 93.33% != 100%", tripping
            # risk_check_node's veto on an otherwise valid recommendation):
            # a theme's tilt/pro-rata share above can exceed what its own
            # (post global-top-N) survivor count can legally absorb under
            # the per-name cap — e.g. a theme with just 1 survivor and a 20%
            # tilt share under a 13.3% per-name cap can only ever place
            # 13.3%, stranding the other 6.7% with nowhere to go. Re-run
            # theme_budgets' targets through the SAME capped water-fill
            # every per-name allocation already uses, but with each THEME's
            # own true ceiling (its survivor count times the per-name cap)
            # instead of one shared cap — any theme that can't absorb its
            # full share now cascades the excess to whichever other themes
            # still have room, so the full open_rem gets placed whenever
            # the aggregate capacity across every theme allows it.
            capacities = {th: len(insts) * cap for th, insts in themes.items()}
            final_budgets = _water_fill([(th, budgets.get(th, 0.0)) for th in themes],
                                        open_rem, capacities)

            w: Dict[str, float] = {}
            for th in themes:
                b = final_budgets.get(th, 0.0)
                # top_n already applied globally above — NOT re-applied per
                # theme here, or a theme with few survivors could still
                # water-fill down to fewer names than its own count anyway.
                fill = _allocate_within_bucket(themes[th], b, cap, book)
                w.update(fill)
                tag = tilted_themes.get(th)
                note = (f"tilt net {tag.net_tilt:+.1%} "
                        f"(strat {tag.strategic_tilt:+.1%} / tact {tag.tactical_tilt:+.1%})"
                        if tag else "no strategist view — neutral")
                if abs(b - budgets.get(th, 0.0)) > 1e-6:
                    note += f" — redistributed from {budgets.get(th, 0.0):.1%} target (survivor cap)"
                book.buckets.append(BucketReport(
                    _join(open_path, th), "strategist" if tag else "agent",
                    b, sum(fill.values()), f"{len(fill)} names — {note}"))
        else:                                            # no strategist input: flat pool fill
            w = _allocate_within_bucket(pool, open_rem, cap, book)
            book.buckets.append(BucketReport(
                open_path, "agent", open_rem, sum(w.values()), f"{len(w)} agent-chosen names"))

        for n, wt in w.items():
            book.add(n, wt)
        _check_min_names(len(w), defaults, book, open_path)


def _group_by_theme(pool: List["Signal"], parent_path: str) -> Dict[str, List["Signal"]]:
    """Group pooled open-space signals by their first path segment past parent_path."""
    groups: Dict[str, List["Signal"]] = {}
    prefix_len = len(parent_path) + 1 if parent_path else 0
    for s in pool:
        rest = s.path[prefix_len:] if parent_path else s.path
        theme = rest.split(".", 1)[0]
        groups.setdefault(theme, []).append(s)
    return groups


def _alloc_bucket(name: str, node: Dict[str, Any], budget: float, path: str,
                  signals: List[Signal], defaults: Dict[str, Any], book: Book,
                  theme_tilts: Dict[str, ThemeTilt]) -> None:
    set_by = node.get("set_by", "agent")
    if "children" in node:
        book.buckets.append(BucketReport(path, set_by, node.get("weight"), budget,
                                         "locked, has sub-buckets"))
        _alloc_children(node["children"], budget, path, signals, defaults, book, theme_tilts)
        return

    limits = {**defaults, **node.get("limits", {})}
    insts = [(s.name, s.score) for s in signals if _under(s.path, path)]
    if not insts:                                        # e.g. cash: no instruments to pick
        book.add(name.upper(), budget)
        book.buckets.append(BucketReport(path, set_by, node.get("weight"), budget, "no instruments"))
        return

    top = _top_n(insts, limits.get("top_n"))            # selection: ranked by raw conviction
    by_name = {s.name: s for s in signals}
    subset = [by_name[n] for n, _ in top]
    w = _allocate_within_bucket(subset, budget, limits.get("max_per_name", 1.0), book)   # sizing: real optimizer w/ fallback
    for n, wt in w.items():
        book.add(n, wt)
    book.buckets.append(BucketReport(path, set_by, node.get("weight"), sum(w.values()),
                                     f"{len(w)} agent-chosen names"))
    _check_min_names(len(w), limits, book, path)


def _check_min_names(n_funded: int, limits: Dict[str, Any], book: Book, path: str) -> None:
    floor = limits.get("min_names", 1)
    if 0 < n_funded < floor:
        book.warnings.append(f"{path}: only {n_funded} names funded, below min_names={floor}")


def _join(a: str, b: str) -> str:
    return b if not a else f"{a}.{b}"


def load_mandate(path: str = "mandate.yaml") -> Dict[str, Any]:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    # Inline demo (no YAML needed) — mirrors mandate.yaml.
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

    # What the agents produced this cycle: convictions + their chosen bucket tags.
    signals = [
        Signal("NVDA",  "equities.tech_ai", 0.9), Signal("MSFT", "equities.tech_ai", 0.7),
        Signal("GOOGL", "equities.tech_ai", 0.6), Signal("AVGO", "equities.tech_ai", 0.5),
        Signal("XOM",   "equities.energy",  0.6), Signal("FSLR", "equities.energy",  0.5),
        Signal("CAT",   "equities.industrials", 0.4),
        Signal("EURUSD", "forex", 0.7), Signal("USDJPY", "forex", 0.5), Signal("GBPUSD", "forex", 0.3),
        Signal("BTC", "crypto", 0.9), Signal("ETH", "crypto", 0.7), Signal("SOL", "crypto", 0.5),
    ]

    book = allocate(mandate, signals)

    print("FINAL BOOK (instrument -> % of total)")
    for name, w in sorted(book.weights.items(), key=lambda kv: -kv[1]):
        print(f"  {name:8s} {w:6.2%}")
    print(f"  {'TOTAL':8s} {sum(book.weights.values()):6.2%}")

    print("\nBUCKET ROLLUP (user targets must be met exactly; open space is agents')")
    for b in book.buckets:
        tgt = f"{b.target:.0%}" if b.target is not None else "  —"
        print(f"  [{b.set_by:5s}] {b.path:22s} target {tgt:>4s}  allocated {b.allocated:6.2%}  {b.note}")

    if book.warnings:
        print("\nWARNINGS")
        for w in book.warnings:
            print(f"  ! {w}")

    print("\n" + "=" * 70)
    print("TEST: diversification floor — a '12 stocks' book holds 12, not 6")
    print("=" * 70)
    import random as _random
    import mandate_allocator as _ma
    _rng = _random.Random(7)
    sigs12 = [Signal(f"N{i:02d}", "equities.tech_ai", 0.9 - 0.03 * i, expected_return=0.30 - 0.02 * i,
                     returns_window=[_rng.gauss(0.001, 0.02) for _ in range(20)]) for i in range(12)]
    cap12 = min(1.0, max(0.10, 2.0 / 12))                      # web_ui._max_per_name_for(12) = 16.7%
    def funded(frac):
        _ma.MIN_WEIGHT_FRACTION = frac
        return _ma._allocate_within_bucket(sigs12, 1.0, cap12, _ma.Book())
    old = funded(0.0)
    new = funded(0.5)
    _ma.MIN_WEIGHT_FRACTION = 0.5
    print(f"  floor off: {len(old)} names funded {sorted(round(v, 3) for v in old.values())[-3:]}…")
    print(f"  floor 0.5: {len(new)} names, min {min(new.values()):.1%}, max {max(new.values()):.1%}")
    assert len(old) < 12, "reproduces the corner solution the floor fixes"
    assert len(new) == 12 and min(new.values()) >= 0.5 / 12 - 1e-6 and max(new.values()) <= cap12 + 1e-6
    assert abs(sum(new.values()) - 1.0) < 1e-6
    assert new["N00"] > new["N11"], "the optimizer still tilts toward its favourites"
    wf = _ma._water_fill_with_floor([(s.name, s.score) for s in sigs12], 1.0, cap12, 0.5 / 12)
    assert len(wf) == 12 and abs(sum(wf.values()) - 1.0) < 1e-9 and min(wf.values()) >= 0.5 / 12 - 1e-12
    assert max(wf.values()) <= cap12 + 1e-12
    print("✓ every selected name funded at >= half an equal share; still tilted; fallback path floored too")
