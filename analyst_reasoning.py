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
    # Longer-horizon context for buy-the-dip (data_ingestion fetches ~6 months):
    drawdown_6m: float = 0.0   # latest price vs. the ~6-month high, <= 0 (-0.18 = 18% below the high)
    pullback_5d: float = 0.0   # last 5 trading days' return


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------------------------------------------------------
# Buy-the-dip: a QUALITY name that has pulled back well below its recent high,
# WITHOUT thesis-breaking news, is an opportunity — not a momentum loser.
# Every gate below exists to tell that apart from a falling knife.
# -----------------------------------------------------------------------------
DIP_MIN_DRAWDOWN = 0.10       # at least 10% below the ~6-month high
DIP_FULL_DRAWDOWN = 0.30      # depth that earns the full dip score
DIP_MAX_DRAWDOWN = 0.35       # deeper than this is a crash / trend change, not a discount
DIP_MIN_5D = -0.04            # still falling faster than this over 5 days = wait for it to stabilize
DIP_MIN_QUALITY = 0.50        # fundamentals.score_from_info (0..1): margins, ROE, leverage, growth
DIP_MAX_BAD_NEWS = -0.40      # news sentiment at/below this = something actually broke
DIP_MIN_ANALYST_UPSIDE = 0.05 # if covered, the Street must still see upside


def dip_opportunity(snap: InstrumentSnapshot) -> float:
    """0 = not a dip opportunity; 0.4..1.0 = a quality name on sale, deeper
    pullbacks scoring higher (10% below the high -> 0.4, 30-35% -> 1.0).
    Calibrated on live data (Oct 2026 tech correction): without the depth
    cap and the 5-day stabilization check, 22 of 59 tech names "qualified",
    including 40-54% crashes still falling — exactly the names that had just
    cost a portfolio 11.5%."""
    dd = -min(0.0, snap.drawdown_6m)
    if dd < DIP_MIN_DRAWDOWN or dd > DIP_MAX_DRAWDOWN:
        return 0.0                                  # not a dip yet / a crash, not a discount
    if snap.pullback_5d < DIP_MIN_5D:
        return 0.0                                  # still in free fall: don't catch the knife
    if snap.quality_score < DIP_MIN_QUALITY:
        return 0.0                                  # cheap for a reason: weak fundamentals
    if snap.news_sentiment <= DIP_MAX_BAD_NEWS:
        return 0.0                                  # falling knife: the news explains the drop
    if snap.analyst_expected_return is not None and snap.analyst_expected_return < DIP_MIN_ANALYST_UPSIDE:
        return 0.0                                  # the analysts who follow it see no rebound
    depth = _clip((dd - DIP_MIN_DRAWDOWN) / (DIP_FULL_DRAWDOWN - DIP_MIN_DRAWDOWN), 0.0, 1.0)
    return round(0.4 + 0.6 * depth, 3)


def _heuristic_rationale(snap: InstrumentSnapshot) -> str:
    """Plain-language gloss of the same three numbers the score itself is
    built from — not a separate judgment, just making the arithmetic legible."""
    parts = []
    dip = dip_opportunity(snap)
    if dip:
        parts.append(f"BUY-THE-DIP: quality name {abs(snap.drawdown_6m):.0%} below its 6-month high "
                     f"with no thesis-breaking news")
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
    dip = dip_opportunity(snap)
    raw = (0.4 * snap.momentum_1m * (1.0 - dip) + 0.05 * snap.news_sentiment
          + 0.08 * (snap.quality_score - 0.5) - 0.15 * snap.volatility
          + 0.25 * dip * abs(snap.drawdown_6m))          # partial recovery toward the recent high
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
    # A qualifying dip REPLACES the momentum penalty (the drop is the opportunity,
    # not the signal) and adds conviction in proportion to its depth.
    dip = dip_opportunity(snap)
    raw = (0.5 * snap.momentum_1m * 10 * (1.0 - dip) + 0.3 * snap.news_sentiment + 0.2 * snap.quality_score
           + 0.5 * dip)
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
                 returns_window=snap.returns_window, dip_score=dip, drawdown_6m=snap.drawdown_6m)


def reason_desk(snapshots: List[InstrumentSnapshot]):
    """A whole desk's worth of candidate instruments -> conviction signals."""
    return [reason_instrument_conviction(s) for s in snapshots]


if __name__ == "__main__":
    def snap(name="Q", momentum=-0.08, sentiment=0.0, quality=0.75, vol=0.02, dd=-0.20,
             upside=0.25, count=20):
        return InstrumentSnapshot(name, "equities.tech_ai", momentum_1m=momentum, news_sentiment=sentiment,
                                  quality_score=quality, volatility=vol, analyst_expected_return=upside,
                                  analyst_rating="Buy" if upside and upside > 0.05 else "Hold",
                                  analyst_count=count, drawdown_6m=dd)

    print("TEST 1: a quality name 20% below its high, no bad news, analysts see upside -> dip opportunity")
    d = dip_opportunity(snap())
    assert 0.4 <= d <= 1.0 and abs(d - (0.4 + 0.6 * 0.5)) < 1e-9, d
    assert dip_opportunity(snap(dd=-0.10)) == 0.4 and dip_opportunity(snap(dd=-0.32)) == 1.0
    print(f"✓ -20% -> {d}; -10% -> 0.4; -32% -> 1.0")

    print("TEST 2: falling knives are NOT dips")
    falling = snap()
    falling.pullback_5d = -0.07
    for label, s in (("only 6% off the high", snap(dd=-0.06)),
                     ("a 45% crash, not a discount", snap(dd=-0.45)),
                     ("still falling 7% in 5 days", falling),
                     ("weak fundamentals", snap(quality=0.35)),
                     ("the news explains the drop", snap(sentiment=-0.5)),
                     ("analysts see no rebound", snap(upside=0.02)),
                     ("analysts see downside", snap(upside=-0.10))):
        assert dip_opportunity(s) == 0.0, label
    assert dip_opportunity(snap(upside=None, count=0)) > 0, "no coverage is not a veto — the other gates still apply"
    print("✓ shallow / crash / still-falling / low-quality / bad-news / no-upside all rejected; uncovered judged on the rest")

    print("TEST 3: the dip replaces the momentum penalty — a quality name on sale outranks itself without the discount")
    on_sale = reason_instrument_conviction(snap(name="ONSALE"))
    no_dip = reason_instrument_conviction(snap(name="NODIP", dd=-0.05))           # same -8% month, not a dip
    knife = reason_instrument_conviction(snap(name="KNIFE", sentiment=-0.6))     # same drop, bad news
    winner = reason_instrument_conviction(snap(name="MOMO", momentum=0.06, dd=-0.01))
    assert on_sale.score > no_dip.score and on_sale.score > knife.score, (on_sale.score, no_dip.score, knife.score)
    assert on_sale.dip_score > 0 and no_dip.dip_score == 0 and on_sale.drawdown_6m == -0.20
    assert "BUY-THE-DIP" in on_sale.rationale and "BUY-THE-DIP" not in knife.rationale
    assert on_sale.score >= winner.score * 0.9, "a deep quality dip competes with a momentum winner"
    print(f"✓ on sale {on_sale.score:.2f} vs same name no-dip {no_dip.score:.2f} vs bad-news knife {knife.score:.2f} "
          f"vs momentum winner {winner.score:.2f}")

    print("TEST 4: snapshots without the new fields (older callers) behave exactly as before")
    old = InstrumentSnapshot("OLD", "equities.tech_ai", momentum_1m=0.03, news_sentiment=0.2, quality_score=0.6)
    assert dip_opportunity(old) == 0.0
    expected = max(0.0, min(1.0, 0.5 + 0.5 * (0.5 * 0.03 * 10 + 0.3 * 0.2 + 0.2 * 0.6) - 0.0))
    assert abs(reason_instrument_conviction(old).score - expected) < 1e-9
    print("✓ ok")
    print("\nAll analyst_reasoning tests passed.")
