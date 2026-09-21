"""
analyst_reasoning.py
=====================
Shared home for InstrumentSnapshot and the DEFAULT (deterministic,
network-free) per-instrument conviction scorer — the arithmetic stand-in
for what should be an LLM call. Mirrors macro_reasoning.py's role for the
strategist: this module has no dependency on firm_graph.py or
claude_reasoning.py, so swapping in Claude-backed analysis is dependency
injection at graph-build time, not a code change to the graph itself.
See firm_graph.make_analyst_node() and
claude_reasoning.make_claude_analyst_reasoner().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InstrumentSnapshot:
    """What would feed an LLM's per-instrument assessment, reduced to
    structured numbers so the default scorer can run without a model call."""
    name: str              # ticker/pair, e.g. "NVDA", "EURUSD", "BTC"
    path: str               # bucket path this instrument is tagged with, e.g. "equities.tech_ai"
    momentum_1m: float       # trailing 1-month return, e.g. +0.08
    news_sentiment: float    # -1..+1
    quality_score: float     # -1..+1, fundamentals/quality proxy (balance sheet, moat, etc.)
    volatility: float = 0.0  # stdev of daily returns over the lookback window — see
                            # data_ingestion.compute_volatility().
    # REAL Wall Street analyst data (fundamentals.get_analyst_forecast) —
    # None when yfinance has no analyst coverage for this name, in which
    # case the scorer falls back to its own heuristic estimate. Genuinely
    # independent third-party numbers, unlike the heuristic/Claude
    # "forecast" below, which is built from these same momentum/sentiment/
    # quality/volatility inputs and so isn't new information at all.
    analyst_expected_return: Optional[float] = None
    analyst_rating: Optional[str] = None
    analyst_count: int = 0
    # Daily returns over the SAME lookback window volatility is computed
    # from (data_ingestion.compute_returns) — real per-instrument history,
    # carried through so mandate_allocator can build an ACTUAL covariance
    # matrix across the names it's choosing between, instead of only ever
    # seeing each instrument's own volatility in isolation (which is what
    # both this module's and mandate_allocator.py's docstrings flagged as
    # the missing piece for real mean-variance optimization).
    returns_window: List[float] = field(default_factory=list)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _heuristic_rationale(snap: InstrumentSnapshot) -> str:
    """Plain-language gloss of the same three numbers the score itself is
    built from — not a separate judgment, just making the arithmetic legible."""
    parts = []
    if snap.momentum_1m > 0.02:
        parts.append(f"positive 1-month momentum ({snap.momentum_1m:+.1%})")
    elif snap.momentum_1m < -0.02:
        parts.append(f"negative 1-month momentum ({snap.momentum_1m:+.1%})")
    if snap.news_sentiment > 0.15:
        parts.append("favorable recent news sentiment")
    elif snap.news_sentiment < -0.15:
        parts.append("unfavorable recent news sentiment")
    if snap.quality_score > 0.5:
        parts.append("above-average quality score")
    if snap.volatility > 0.03:
        parts.append(f"elevated volatility ({snap.volatility:.1%} daily) discounted this some")
    if not parts:
        parts.append("no strong signal in either direction — a neutral, data-light read")
    return "Heuristic scorer: " + ", ".join(parts) + "."


def _heuristic_risk_note(snap: InstrumentSnapshot) -> str:
    """Flags whichever input to the score looks weakest — the point isn't
    precision, it's not letting a high conviction score hide an obvious
    caveat in the same data that produced it. Returns "" (no risk div
    rendered at all) rather than generic filler when nothing in the data
    actually looks weak — a boilerplate "no red flag" line under every
    single name added noise without adding information."""
    if snap.volatility > 0.045:
        return f"High volatility ({snap.volatility:.1%} daily stdev) — a choppier, riskier position than most."
    if snap.quality_score < 0.3:
        return "Below-average quality score — thinner fundamental support for this call."
    if snap.momentum_1m < -0.03:
        return "Currently in a downtrend; conviction here is a reversal bet."
    if snap.news_sentiment < -0.2:
        return "Recent news flow has been negative."
    return ""


def _rating_from_conviction(conviction: float) -> str:
    """Buy/Hold/Sell tier consistent with the conviction score — same
    thresholds an eventual human reviewer would sanity-check against."""
    if conviction >= 0.65:
        return "Buy"
    if conviction >= 0.40:
        return "Hold"
    return "Sell"


def _heuristic_expected_return(snap: InstrumentSnapshot) -> float:
    """FORECAST: a naive forward-return estimate — partial momentum
    persistence, a small sentiment/quality tilt, a volatility drag — built
    from the same four numbers as the conviction score, so it's legible
    arithmetic rather than a black box. This is a heuristic estimate, NOT a
    real forecast or guarantee (same honesty caveat web_ui.py already
    applies to its own momentum-only projection)."""
    raw = (0.4 * snap.momentum_1m + 0.05 * snap.news_sentiment
          + 0.08 * (snap.quality_score - 0.5) - 0.15 * snap.volatility)
    return _clip(raw, -0.30, 0.30)


def reason_instrument_conviction(snap: InstrumentSnapshot):
    """One instrument in, one Signal out — the default (non-Claude) scorer.
    Treats each instrument independently; a real LLM call can reason about
    a whole desk jointly instead (see claude_reasoning.py).

    RISK ADJUSTMENT: conviction is penalized by volatility (a simple,
    transparent risk/reward tradeoff, roughly analogous in spirit to a
    Sharpe-style adjustment) — this is what actually makes stock SELECTION
    risk-aware, not just a cosmetic display metric. It is deliberately
    simple: a per-instrument penalty, not a real portfolio-level optimizer
    weighing correlations between holdings (mandate_allocator.py's own
    docstring already flags that as the production upgrade path)."""
    from mandate_allocator import Signal   # deferred: mandate_allocator stays independent
    raw = 0.5 * snap.momentum_1m * 10 + 0.3 * snap.news_sentiment + 0.2 * snap.quality_score
    risk_penalty = min(snap.volatility * 3.0, 0.3)   # capped so one wild name doesn't hit zero outright
    conviction = _clip(0.5 + 0.5 * raw - risk_penalty, 0.0, 1.0)
    # Prefer REAL analyst data (fundamentals.get_analyst_forecast, wired in
    # by data_ingestion.py) over the self-referential heuristic estimate
    # below — the heuristic is built from the SAME four numbers already
    # driving conviction, so it's not independent information; real analyst
    # coverage is. Falls back to the heuristic only when yfinance has no
    # analyst coverage for this name.
    if snap.analyst_expected_return is not None:
        expected_return = snap.analyst_expected_return
        rating = snap.analyst_rating or _rating_from_conviction(conviction)
        analyst_count = snap.analyst_count
    else:
        expected_return, rating, analyst_count = _heuristic_expected_return(snap), _rating_from_conviction(conviction), 0
    return Signal(snap.name, snap.path, conviction,
                 rationale=_heuristic_rationale(snap), risk_note=_heuristic_risk_note(snap),
                 momentum_1m=snap.momentum_1m, news_sentiment=snap.news_sentiment,
                 quality_score=snap.quality_score, volatility=snap.volatility,
                 expected_return=expected_return, rating=rating, analyst_count=analyst_count,
                 returns_window=snap.returns_window)


def reason_desk(snapshots: List[InstrumentSnapshot]):
    """A whole desk's worth of candidate instruments -> conviction signals."""
    return [reason_instrument_conviction(s) for s in snapshots]
