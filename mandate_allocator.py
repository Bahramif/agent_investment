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

Which names and how much of each follow rulebook.py, the same rules the news
watch and the backtest use: selection with a keep band and re-entry blocks,
inverse-volatility sizing, and sub-industry and beta limits. What a limit won't
let the book hold stays as CASH.

`default_limits` may carry the portfolio's context: "held" (its current stocks,
for the keep band), "blocked" (stocks that must not be bought) and "rulebook"
(a config override, used by the backtest).

`load_mandate` needs PyYAML (`pip install pyyaml`); everything else is stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import rulebook
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
    # DISPLAY ONLY: neither ranks nor sizes anything.
    expected_return: float = 0.0
    rating: str = ""
    analyst_count: int = 0   # 0 when expected_return/rating came from the heuristic/Claude
                            # fallback rather than real analyst coverage — see analyst_reasoning.py
    # Pullback credit (rulebook.pullback_credit): 0 = none; up to 1 = a meaningful,
    # steadying pullback in an intact uptrend with clean news.
    dip_score: float = 0.0
    drawdown_6m: float = 0.0
    # Rulebook inputs
    momentum_12_1: Optional[float] = None
    earnings_yield: Optional[float] = None
    industry: str = ""
    beta: Optional[float] = None
    earnings_date: Optional[str] = None
    eligible: bool = True
    ineligible_reason: str = ""


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


def _cfg(defaults: Dict[str, Any]) -> rulebook.RulebookConfig:
    return rulebook.config_from_dict(defaults["rulebook"]) if defaults.get("rulebook") else rulebook.default_config()


def _select(signals: List["Signal"], n: Optional[float], defaults: Dict[str, Any]) -> List["Signal"]:
    """rulebook.select_names over these signals, for the N a mandate asks for
    (it may arrive as a float: every number that crosses JSON does). With no N,
    every eligible, unblocked name is selected."""
    cfg = _cfg(defaults)
    blocked = set(defaults.get("blocked") or [])
    pool = [s for s in signals if s.eligible and s.name not in blocked]
    if not n or n <= 0:
        return pool
    today = rulebook._iso_date(defaults.get("as_of")) or date.today()
    names = rulebook.select_names(pool, int(n), defaults.get("held") or [], blocked, cfg, today)
    by_name = {s.name: s for s in pool}
    return [by_name[x] for x in names]


def _weight_rationale(sig: "Signal", weight: float, cap: float, peers: List["Signal"]) -> str:
    """One line on why THIS weight (Signal.rationale says why this name)."""
    vols = [p.volatility for p in peers if p.volatility]
    if weight >= cap - 1e-9:
        return f"Sized at {weight:.1%}, the {cap:.0%} per-stock cap."
    if not sig.volatility or not vols:
        return f"Sized at {weight:.1%}: a typical share, since its volatility isn't known."
    avg = sum(vols) / len(vols)
    word = "calmer" if sig.volatility < avg else "more volatile"
    return (f"Sized at {weight:.1%} by volatility: {word} than this group "
            f"({sig.volatility:.1%} daily vs. {avg:.1%} average), so less risky stocks get more.")


def _allocate_within_bucket(signals_subset: List["Signal"], budget: float, cap: float,
                            book: "Book") -> Dict[str, float]:
    """Sizes an already-selected set of names: inverse volatility, capped per
    name. Deliberately no return forecast and no covariance estimate: analyst
    upside rises automatically when a price falls, and a covariance matrix for a
    dozen stocks can't be estimated reliably. See rulebook.inverse_vol_weights."""
    w = rulebook.inverse_vol_weights({s.name: s.volatility for s in signals_subset}, budget, cap)
    by_name = {s.name: s for s in signals_subset}
    for name, wt in w.items():
        book.weight_rationale[name] = _weight_rationale(by_name[name], wt, cap, signals_subset)
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
    _apply_limits(book, signals, defaults, mandate.get("total", 1.0))
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
                "momentum_12_1": s.momentum_12_1, "earnings_yield": s.earnings_yield,
                "industry": s.industry, "beta": s.beta, "earnings_date": s.earnings_date,
                "weight_rationale": book.weight_rationale.get(s.name, ""),
            }
    return book


def _apply_limits(book: Book, signals: List[Signal], defaults: Dict[str, Any], total: float) -> None:
    """Sub-industry and beta limits on the finished book; whatever the book can't
    hold, for these or any other reason, is CASH, so it always adds up."""
    stocks = {n: w for n, w in book.weights.items() if n != rulebook.CASH}
    by_name = {s.name: s for s in signals}
    limited, notes = rulebook.apply_portfolio_limits(
        stocks, {n: by_name[n].industry for n in stocks if n in by_name},
        {n: by_name[n].beta for n in stocks if n in by_name}, defaults.get("max_per_name", 1.0), _cfg(defaults))
    book.warnings.extend(notes)
    cash = total - sum(limited.values())
    book.weights = dict(limited)
    if cash > 1e-9:
        book.weights[rulebook.CASH] = cash


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
        open_path = _join(parent_path, "*open*")

        # GLOBAL selection, confirmed as the intended behavior: "max # of
        # stocks" caps the TOTAL number of positions across every open
        # theme/sector COMBINED, not per theme. Scores are within-sector
        # percentiles, so stocks from different sectors compare fairly here.
        pool = _select(pool, defaults.get("top_n"), defaults)

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
                # selection already applied globally above, not per theme
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
    insts = [s for s in signals if _under(s.path, path)]
    if not insts:                                        # e.g. cash: no instruments to pick
        book.add(name.upper(), budget)
        book.buckets.append(BucketReport(path, set_by, node.get("weight"), budget, "no instruments"))
        return

    subset = _select(insts, limits.get("top_n"), limits)
    w = _allocate_within_bucket(subset, budget, limits.get("max_per_name", 1.0), book)
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
    print("TEST: a '12 stocks' book holds 12, sized by inverse volatility")
    print("=" * 70)
    sigs12 = [Signal(f"N{i:02d}", "equities.tech_ai", 0.9 - 0.03 * i, volatility=0.01 + 0.002 * i) for i in range(14)]
    m12 = {"total": 1.0, "default_limits": {"max_per_name": 1 / 6, "top_n": 12}, "buckets": {"equities": {"set_by": "user", "weight": 1.0}}}
    b12 = allocate(m12, sigs12)
    assert len(b12.weights) == 12 and abs(sum(b12.weights.values()) - 1.0) < 1e-9
    assert b12.weights["N00"] > b12.weights["N11"] and max(b12.weights.values()) <= 1 / 6 + 1e-9
    print(f"✓ 12 names, calmest {b12.weights['N00']:.1%}, wildest {b12.weights['N11']:.1%}")

    print("TEST: ineligible and blocked names are out; holdings stay inside the keep band; 'blocked' beats 'held'")
    sigs = [Signal(f"S{i}", "equities.tech_ai", 1.0 - i * 0.05, volatility=0.02) for i in range(10)]
    sigs[0].eligible = False
    m = {"total": 1.0, "default_limits": {"max_per_name": 0.5, "top_n": 3, "held": ["S5", "S1"], "blocked": ["S1", "S2"]},
         "buckets": {"equities": {"set_by": "user", "weight": 1.0}}}
    assert set(allocate(m, sigs).weights) == {"S5", "S3", "S4"}, allocate(m, sigs).weights
    print("✓ ok")

    print("TEST: limits leave the excess as CASH and the book still adds up")
    beta = [Signal(f"B{i}", "equities.tech_ai", 0.9, volatility=0.02, beta=2.0) for i in range(4)]
    bk = allocate({"total": 1.0, "default_limits": {"max_per_name": 0.5, "top_n": 4}, "buckets": {"equities": {"set_by": "user", "weight": 1.0}}}, beta)
    assert abs(bk.weights[rulebook.CASH] - 0.25) < 1e-9 and abs(sum(bk.weights.values()) - 1.0) < 1e-9 and bk.warnings
    print(f"✓ beta 2.0 book holds {1 - bk.weights[rulebook.CASH]:.0%} stocks, {bk.weights[rulebook.CASH]:.0%} cash")
