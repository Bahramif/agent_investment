"""
analyst_reasoning.py
=====================
Shared home for InstrumentSnapshot and the deterministic desk scorer. The score
itself lives in rulebook.py (one score for live and backtest); this module turns
a desk of snapshots into Signals with a plain-language rationale. The LLM path
(claude_reasoning.py) feeds its read into the SAME score. No dependency on
firm_graph.py or claude_reasoning.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import rulebook


@dataclass
class InstrumentSnapshot:
    """Everything one stock's assessment reads, as structured numbers."""
    name: str               # ticker
    path: str               # bucket path, e.g. "equities.tech_ai"
    momentum_1m: float      # latest month's return (shown to people and the LLM; not a score factor)
    news_sentiment: float   # -1..+1, from the news classification (0 until classified)
    quality_score: float    # 0..1, fundamentals.score_from_info
    volatility: float = 0.0  # daily volatility over up to a year
    # Analyst consensus: shown to people and the LLM, never used to size a position
    # (target upside rises automatically when a price falls).
    analyst_expected_return: Optional[float] = None
    analyst_rating: Optional[str] = None
    analyst_count: int = 0
    drawdown_6m: float = 0.0   # latest price vs. the ~6-month high, <= 0
    pullback_5d: float = 0.0   # last 5 trading days' return
    # Score inputs
    momentum_12_1: Optional[float] = None    # 12 months ago to 1 month ago
    earnings_yield: Optional[float] = None   # forward EPS / price (negative earnings rank last)
    # Portfolio-limit and eligibility inputs
    industry: str = ""
    beta: Optional[float] = None
    earnings_date: Optional[str] = None      # next report, ISO date
    history_bars: Optional[int] = None       # None = unknown, not held against it
    avg_dollar_volume: Optional[float] = None
    range_position: float = 0.5              # where the price sits in its own 1-year range (0..1)
    # News: raw headlines, and their classification
    headlines: List[str] = field(default_factory=list)
    news_status: str = "unknown"             # clean | negative | thesis_break | unknown


def rating_from_score(score: float) -> str:
    """Buy/Hold/Sell tier, shown when there is no analyst consensus."""
    if score >= 0.65:
        return "Buy"
    if score >= 0.40:
        return "Hold"
    return "Sell"


def eligibility_of(snap: InstrumentSnapshot, cfg: Optional[rulebook.RulebookConfig] = None):
    return rulebook.eligibility(snap.history_bars, snap.avg_dollar_volume, snap.news_status, cfg)


def heuristic_rationale(snap: InstrumentSnapshot, credit: float) -> str:
    """Plain-language gloss of the numbers behind the score, not a separate judgment."""
    parts = []
    if credit > 0:
        parts.append(f"a pullback in an uptrend, {abs(snap.drawdown_6m):.0%} below its 6-month high")
    if snap.momentum_12_1 is not None:
        parts.append(f"{snap.momentum_12_1:+.0%} over the year to last month")
    if snap.quality_score > 0.5:
        parts.append("above-average quality")
    if snap.earnings_yield is not None:
        parts.append(f"{snap.earnings_yield:.1%} forward earnings yield")
    if not parts:
        parts.append("no strong signal in either direction")
    return "Rules score: " + ", ".join(parts) + "."


def heuristic_risk_note(snap: InstrumentSnapshot) -> str:
    """Flags whichever input looks weakest. "" when nothing does: a boilerplate
    line under every name adds noise, not information."""
    if snap.volatility > 0.045:
        return f"High volatility ({snap.volatility:.1%} daily), a choppier, riskier position than most."
    if snap.quality_score < 0.3:
        return "Below-average quality score: thinner fundamental support for this call."
    if snap.momentum_12_1 is not None and snap.momentum_12_1 < 0:
        return "The long-term trend is down; this call rests on its other factors."
    if snap.news_status == "negative":
        return "Recent news flow has been negative."
    return ""


def heuristic_expected_return(snap: InstrumentSnapshot) -> float:
    """DISPLAY ONLY, for a name with no analyst coverage: a naive estimate from
    the same inputs as the score, so not independent information. Never sizes
    or ranks anything."""
    return max(-0.30, min(0.30, 0.15 * (snap.momentum_12_1 or 0.0) + 0.08 * (snap.quality_score - 0.5)
                          - 0.15 * snap.volatility))


def make_signal(snap: InstrumentSnapshot, score: float, credit: float, cfg: Optional[rulebook.RulebookConfig] = None,
                rationale: str = "", risk_note: str = "", rating: str = "", expected_return: Optional[float] = None):
    """The one place a Signal is built, for both the rules-only and the LLM path."""
    from mandate_allocator import Signal   # deferred: mandate_allocator stays independent
    if snap.analyst_expected_return is not None:
        expected_return = snap.analyst_expected_return
        rating = snap.analyst_rating or rating or rating_from_score(score)
        analyst_count = snap.analyst_count
    else:
        expected_return = heuristic_expected_return(snap) if expected_return is None else expected_return
        rating = rating or rating_from_score(score)
        analyst_count = 0
    eligible, why = eligibility_of(snap, cfg)
    return Signal(snap.name, snap.path, score,
                  rationale=rationale or heuristic_rationale(snap, credit),
                  risk_note=risk_note or heuristic_risk_note(snap),
                  momentum_1m=snap.momentum_1m, news_sentiment=snap.news_sentiment,
                  quality_score=snap.quality_score, volatility=snap.volatility,
                  expected_return=expected_return, rating=rating, analyst_count=analyst_count,
                  dip_score=credit, drawdown_6m=snap.drawdown_6m,
                  momentum_12_1=snap.momentum_12_1, earnings_yield=snap.earnings_yield,
                  industry=snap.industry, beta=snap.beta, earnings_date=snap.earnings_date,
                  eligible=eligible, ineligible_reason=why)


def reason_desk(snapshots: List[InstrumentSnapshot], cfg: Optional[rulebook.RulebookConfig] = None,
                llm_pct: Optional[Dict[str, float]] = None):
    """A desk's candidates -> Signals, scored by rulebook.composite_scores."""
    scores = rulebook.composite_scores(snapshots, llm_pct, cfg)
    return [make_signal(s, *scores[s.name], cfg) for s in snapshots]


if __name__ == "__main__":
    def snap(name="Q", mom_long=0.25, quality=0.75, vol=0.02, dd=-0.20, p5=0.0, news="clean", ey=0.05):
        return InstrumentSnapshot(name, "equities.tech_ai", momentum_1m=-0.05, news_sentiment=0.0, quality_score=quality,
                                  volatility=vol, drawdown_6m=dd, pullback_5d=p5, momentum_12_1=mom_long,
                                  earnings_yield=ey, news_status=news)

    print("TEST 1: scores rank peers; every signal is 0..1 and carries what the portfolio limits need")
    desk = [snap("A", mom_long=0.5, quality=0.9, ey=0.08), snap("B", mom_long=0.1, quality=0.5, ey=0.05),
            snap("C", mom_long=-0.3, quality=0.2, ey=0.01)]
    desk[0].industry, desk[0].beta = "Semiconductors", 1.4
    sigs = {s.name: s for s in reason_desk(desk)}
    assert sigs["A"].score > sigs["B"].score > sigs["C"].score
    assert all(0 <= s.score <= 1 for s in sigs.values()) and sigs["A"].industry == "Semiconductors" and sigs["A"].beta == 1.4
    print(f"✓ A {sigs['A'].score:.2f} > B {sigs['B'].score:.2f} > C {sigs['C'].score:.2f}")

    print("TEST 2: real analyst data is displayed when present and never changes the score")
    a, b = snap("X"), snap("Y")
    a.analyst_expected_return, a.analyst_rating, a.analyst_count = 0.40, "Buy", 30
    sa, sb = {s.name: s for s in reason_desk([a, b])}.values()
    assert sa.expected_return == 0.40 and sa.rating == "Buy" and sa.analyst_count == 30 and sb.analyst_count == 0
    assert sa.score == sb.score, "same factors, same score, whatever the analysts' target says"
    print("✓ target shown; scores identical")

    print("TEST 3: ineligible names say why")
    young = snap("NEW")
    young.history_bars = 100
    assert not reason_desk([young])[0].eligible and "100 trading days" in reason_desk([young])[0].ineligible_reason
    print("✓ ok")
    print("\nAll analyst_reasoning tests passed.")
