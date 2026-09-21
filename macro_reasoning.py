"""
macro_reasoning.py
===================
Shared home for MacroSnapshot and the DEFAULT reasoning functions — the
deterministic arithmetic stand-ins for what should be an LLM call.

This lives in its own module, separate from firm_graph.py and
claude_reasoning.py, specifically so neither has to import the other:
swapping heuristic reasoning for Claude-backed reasoning is dependency
injection at graph-build time, not a code change to the graph itself.
See firm_graph.py's make_strategist_reasoning_node() and
claude_reasoning.py's make_claude_reasoners().
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MacroSnapshot:
    """What would feed an LLM's theme assessment, reduced to structured
    numbers so the default reasoning below can run without a model call."""
    theme: str
    momentum_1m: float           # trailing 1-month theme return, e.g. +0.08
    news_sentiment: float        # -1..+1, e.g. headline sentiment score
    valuation_percentile: float  # 0..1, where this trades vs. its own history
    secular_score: float         # -1..+1, structural tailwind (capex, TAM, policy)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def reason_strategic_view(snap: MacroSnapshot):
    from strategist import HorizonView   # deferred: strategist.py stays a leaf module
    raw = _clip(0.5 * snap.secular_score - 0.3 * (snap.valuation_percentile - 0.5), -1, 1) * 0.30
    confidence = _clip(0.4 + 0.4 * abs(snap.secular_score), 0.0, 0.9)
    return HorizonView(raw=raw, confidence=confidence, age_days=0.0,
                       half_life_days=365, max_age_days=180)


def reason_tactical_view(snap: MacroSnapshot):
    from strategist import HorizonView
    raw = _clip(0.6 * snap.momentum_1m * 10 + 0.4 * snap.news_sentiment, -1, 1) * 0.35
    confidence = _clip(0.3 + 0.5 * abs(snap.news_sentiment), 0.0, 0.85)
    return HorizonView(raw=raw, confidence=confidence, age_days=0.0,
                       half_life_days=14, max_age_days=21)
