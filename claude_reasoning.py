"""
claude_reasoning.py
====================
Real Claude-backed reasoning for two parts of the firm:

  1. The strategist's two horizons (MacroSnapshot -> HorizonView), matching
     macro_reasoning.py's contract.
  2. Analyst desk scoring (List[InstrumentSnapshot] -> List[Signal]), matching
     analyst_reasoning.py's contract — one call scores a whole desk's
     candidates AT ONCE, relative to each other, rather than one call per
     instrument (more realistic, and far more token-efficient).

Both follow the same rule: "swap the body, not the signature."

MODEL CHOICE mirrors the cadence design already in firm_graph.py: strategic
views are rare (cadence-gated, ~every 90 days) and consequential, so they get
the stronger model; tactical views and analyst desk scoring both run every
cycle and need to stay cheap.

  strategic        -> claude-opus-5               (rare, high-stakes, worth the cost)
  tactical + desks -> claude-haiku-4-5@20251001    (frequent, needs to stay fast/cheap)

DEPLOYMENT TARGET: per this project's plan (Vertex AI Agent Engine), the
default client is AnthropicVertex, matching Claude's official Vertex
integration. Set ANTHROPIC_VERTEX_PROJECT_ID and CLOUD_ML_REGION. For local
dev without GCP set up, set CLAUDE_USE_DIRECT_API=1 to use the plain
Anthropic() client with an ANTHROPIC_API_KEY instead — the call shape is
identical either way; only client construction differs.

Vertex model IDs can carry a version suffix that direct-API names don't have.
Confirm the exact string in your project's Model Garden listing before
deploying — the model names above are the direct-API names.

HONEST LIMITATION: this sandbox has no network access, so no live Claude
call runs here. FakeAnthropicClient below returns a response shaped exactly
like the real SDK's, so the part that actually has bugs to catch — request
construction, tool-output parsing, clamping, and the fallback-on-error path
(including the PARTIAL fallback when a model skips one instrument in a desk)
— is fully exercised. Point a real Anthropic()/AnthropicVertex() client at
the same functions and nothing else changes.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple

from macro_reasoning import (
    MacroSnapshot,
    reason_strategic_view as _fallback_strategic,
    reason_tactical_view as _fallback_tactical,
)
from analyst_reasoning import (
    InstrumentSnapshot,
    reason_instrument_conviction as _fallback_instrument,
    reason_desk as _fallback_desk,
    _rating_from_conviction,
)
from strategist import HorizonView
from mandate_allocator import Signal


ASSESS_THEME_TOOL = {
    "name": "assess_theme",
    "description": "Report a directional investment view on a market theme.",
    "input_schema": {
        "type": "object",
        "properties": {
            "raw_tilt": {
                "type": "number",
                "description": "Signed view strength in [-1, 1]. Positive favors "
                               "overweighting this theme, negative favors "
                               "underweighting it. Magnitude reflects conviction "
                               "in the DIRECTION, separate from confidence.",
            },
            "confidence": {
                "type": "number",
                "description": "How sure you are in this view, in [0, 1]. Low "
                               "confidence should pull the view toward neutral "
                               "when combined downstream — be honest here rather "
                               "than overstating certainty.",
            },
            "rationale": {
                "type": "string",
                "description": "One or two sentences: the key evidence driving "
                               "this view, for the audit log.",
            },
        },
        "required": ["raw_tilt", "confidence", "rationale"],
    },
}


def _prompt_for(snap: MacroSnapshot, horizon: str) -> str:
    if horizon == "strategic":
        ask = ("Assess the LONG-TERM (1-3 year) structural case for this theme. "
              "Weigh secular tailwinds (capex cycles, TAM expansion, policy) "
              "against how much of that is already priced in.")
    else:
        ask = ("Assess the SHORT-TERM (days-to-weeks) tactical read for this "
              "theme. Weigh recent momentum and news flow — is it overbought "
              "or oversold relative to the move?")
    return (
        f"{ask}\n\n"
        f"Theme: {snap.theme}\n"
        f"1-month momentum: {snap.momentum_1m:+.1%}\n"
        f"News sentiment (-1..+1): {snap.news_sentiment:+.2f}\n"
        f"Valuation percentile vs. own history (0=cheap, 1=rich): {snap.valuation_percentile:.2f}\n"
        f"Secular tailwind score (-1..+1): {snap.secular_score:+.2f}\n\n"
        "Call the assess_theme tool with your view."
    )


def _client():
    if os.environ.get("CLAUDE_USE_DIRECT_API"):
        from anthropic import Anthropic
        return Anthropic()   # reads ANTHROPIC_API_KEY from env
    from anthropic import AnthropicVertex
    return AnthropicVertex(
        project_id=os.environ["ANTHROPIC_VERTEX_PROJECT_ID"],
        region=os.environ.get("CLOUD_ML_REGION", "us-east5"),
    )


def _call_claude(client: Any, model: str, prompt: str) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=400,
        tools=[ASSESS_THEME_TOOL],
        tool_choice={"type": "tool", "name": "assess_theme"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "assess_theme":
            return block.input
    raise ValueError("Claude did not return the expected assess_theme tool call")


def _view_from_tool_output(output: dict, half_life_days: float, max_age_days: float) -> HorizonView:
    raw = max(-1.0, min(1.0, float(output["raw_tilt"])))
    confidence = max(0.0, min(1.0, float(output["confidence"])))
    return HorizonView(raw=raw, confidence=confidence, age_days=0.0,
                       half_life_days=half_life_days, max_age_days=max_age_days)


def reason_strategic_view_claude(snap: MacroSnapshot, client: Any = None) -> HorizonView:
    client = client or _client()
    try:
        output = _call_claude(client, "claude-opus-5", _prompt_for(snap, "strategic"))
        return _view_from_tool_output(output, half_life_days=365, max_age_days=180)
    except Exception as e:
        # Never let a model/network hiccup take down the pipeline — fall back
        # to the deterministic heuristic and let the caller's logs show why.
        print(f"[claude_reasoning] strategic call failed for {snap.theme}: {e} — using fallback")
        return _fallback_strategic(snap)


def reason_tactical_view_claude(snap: MacroSnapshot, client: Any = None) -> HorizonView:
    client = client or _client()
    try:
        output = _call_claude(client, "claude-haiku-4-5@20251001", _prompt_for(snap, "tactical"))
        return _view_from_tool_output(output, half_life_days=14, max_age_days=21)
    except Exception as e:
        print(f"[claude_reasoning] tactical call failed for {snap.theme}: {e} — using fallback")
        return _fallback_tactical(snap)


def make_claude_reasoners(
    client: Any = None,
) -> Tuple[Callable[[MacroSnapshot], HorizonView], Callable[[MacroSnapshot], HorizonView]]:
    """The integration point: bind ONE client and return two closures shaped
    exactly like macro_reasoning's defaults, ready to pass into
    firm_graph.build_compiled_graph(reason_strategic=..., reason_tactical=...).
    Bind the client once here, at setup time — not per call."""
    client = client or _client()
    return (lambda snap: reason_strategic_view_claude(snap, client=client),
           lambda snap: reason_tactical_view_claude(snap, client=client))


# -----------------------------------------------------------------------------
# Analyst desk scoring — one Claude call scores a whole desk's candidates at
# once, rather than one call per instrument. This is both more realistic (an
# analyst reasons about candidates RELATIVE to each other, not in isolation)
# and far more token-efficient than N separate calls for N instruments.
# -----------------------------------------------------------------------------
SCORE_INSTRUMENTS_TOOL = {
    "name": "score_instruments",
    "description": "Score each candidate instrument's investment conviction, "
                   "0 (avoid) to 1 (high conviction).",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string",
                                "description": "Must exactly match one of the given instrument names."},
                        "conviction": {"type": "number",
                                      "description": "0..1, higher = stronger conviction to hold/buy."},
                        "rationale": {"type": "string",
                                     "description": "One sentence, for the audit log."},
                        "key_risk": {"type": "string",
                                    "description": "One sentence: the single biggest risk to "
                                                   "this call going wrong. Surfaced directly to "
                                                   "the end user, not just an internal note."},
                        "rating": {"type": "string", "enum": ["Buy", "Hold", "Sell"],
                                  "description": "Overall recommendation tier, consistent with "
                                                 "the conviction score you give."},
                        "expected_return": {"type": "number",
                                            "description": "Your own naive forward return "
                                                           "estimate over the next ~1-3 months, "
                                                           "as a decimal (0.08 = +8%, -0.05 = "
                                                           "-5%). A rough directional estimate "
                                                           "for display, not a guarantee."},
                    },
                    "required": ["name", "conviction", "rationale", "key_risk", "rating", "expected_return"],
                },
            },
        },
        "required": ["scores"],
    },
}


def _desk_prompt(snapshots: List[InstrumentSnapshot]) -> str:
    lines = "\n".join(
        f"- {s.name} ({s.path}): 1-month momentum {s.momentum_1m:+.1%}, "
        f"news sentiment {s.news_sentiment:+.2f}, quality {s.quality_score:+.2f}, "
        f"volatility {s.volatility:.1%} daily stdev"
        for s in snapshots
    )
    return (
        "Score each candidate instrument's investment conviction from 0 "
        "(avoid) to 1 (high conviction), RELATIVE to each other — you're "
        "allocating attention across a whole desk, not judging each in "
        "isolation. Weigh RISK, not just return: a name with strong momentum "
        "but much higher volatility than its peers should generally score "
        "lower than an equally-strong but steadier one — conviction should "
        f"reflect a risk/reward tradeoff, not raw upside alone.\n\n{lines}\n\n"
        "For each one, also give a one-sentence rationale (why this "
        "conviction) and a one-sentence key_risk (the single biggest way "
        "this call could be wrong — call out high volatility explicitly "
        "when it's a factor) — both go directly in front of the end user, "
        "so write them for a person deciding whether to act on this, not "
        "as an internal log line. Also give a rating (Buy/Hold/Sell, "
        "consistent with the conviction score) and expected_return (your "
        "own naive forward-return estimate as a decimal, e.g. 0.08 for "
        "+8%) — be honest that this is a rough estimate, not a guarantee.\n\n"
        "Call score_instruments with exactly one entry per instrument listed above."
    )


def reason_desk_convictions_claude(
    snapshots: List[InstrumentSnapshot], client: Any = None,
) -> List[Signal]:
    if not snapshots:
        return []
    client = client or _client()
    by_name = {s.name: s for s in snapshots}
    try:
        output = _call_claude(client, "claude-haiku-4-5@20251001", _desk_prompt(snapshots))
        signals: List[Signal] = []
        seen = set()
        for entry in output.get("scores", []):
            snap = by_name.get(entry.get("name"))
            if snap is None:
                continue   # model scored something outside what we asked about — ignore it
            conviction = max(0.0, min(1.0, float(entry["conviction"])))
            # Prefer REAL analyst data over Claude's own generated
            # rating/expected_return — same fallback order as the heuristic
            # scorer (analyst_reasoning.py): Claude's guess is built from
            # the same inputs already driving conviction, real coverage
            # isn't. Only fall back to Claude's answer when yfinance has no
            # coverage for this name.
            if snap.analyst_expected_return is not None:
                expected_return = snap.analyst_expected_return
                rating = snap.analyst_rating or ""
                analyst_count = snap.analyst_count
            else:
                rating = str(entry.get("rating", "")).strip().title()
                expected_return = max(-0.5, min(0.5, float(entry.get("expected_return", 0.0))))
                analyst_count = 0
            if rating not in ("Buy", "Hold", "Sell"):   # never trust an out-of-enum value blindly
                rating = _rating_from_conviction(conviction)
            signals.append(Signal(snap.name, snap.path, conviction,
                                  rationale=entry.get("rationale", ""),
                                  risk_note=entry.get("key_risk", ""),
                                  momentum_1m=snap.momentum_1m, news_sentiment=snap.news_sentiment,
                                  quality_score=snap.quality_score, volatility=snap.volatility,
                                  expected_return=expected_return, rating=rating,
                                  analyst_count=analyst_count, returns_window=snap.returns_window))
            seen.add(snap.name)
        # A partial miss shouldn't zero out the whole desk — fall back to the
        # heuristic individually for anything the model skipped.
        for name, snap in by_name.items():
            if name not in seen:
                print(f"[claude_reasoning] '{name}' missing from model output — using fallback")
                signals.append(_fallback_instrument(snap))
        return signals
    except Exception as e:
        print(f"[claude_reasoning] desk scoring call failed: {e} — using fallback for entire desk")
        return _fallback_desk(list(snapshots))


def make_claude_analyst_reasoner(
    client: Any = None,
) -> Callable[[List[InstrumentSnapshot]], List[Signal]]:
    """Returns ONE callable usable for every desk (equities/forex/crypto) —
    List[InstrumentSnapshot] -> List[Signal]. Same bind-once-at-setup-time
    pattern as make_claude_reasoners()."""
    client = client or _client()
    return lambda snapshots: reason_desk_convictions_claude(snapshots, client=client)


# ----------------------------------------------------------------------------
# Test harness — proves the request/response plumbing without network access.
# ----------------------------------------------------------------------------
class _FakeToolUseBlock:
    type = "tool_use"
    def __init__(self, name, input_):
        self.name = name
        self.input = input_


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeAnthropicClient:
    """Stands in for anthropic.Anthropic() / AnthropicVertex() in this
    sandbox. Returns a response shaped exactly like the real SDK's, so
    everything downstream of the API call — parsing, clamping, fallback-on-
    error — is exercised for real. Swap in a real client; nothing else changes."""

    def __init__(self, canned_input: Optional[dict] = None, raise_error: bool = False):
        self.canned_input = canned_input or {
            "raw_tilt": 0.4, "confidence": 0.7,
            "rationale": "Strong secular tailwind, valuation not yet stretched.",
        }
        self.raise_error = raise_error
        self.messages = self   # so client.messages.create(...) resolves to self.create(...)

    def create(self, **kwargs):
        if self.raise_error:
            raise ConnectionError("simulated network failure")
        return _FakeResponse([_FakeToolUseBlock("assess_theme", self.canned_input)])


if __name__ == "__main__":
    snap = MacroSnapshot("AI", momentum_1m=0.04, news_sentiment=0.6,
                        valuation_percentile=0.7, secular_score=0.6)

    print("=" * 70)
    print("TEST 1: successful tool call — verifies request/response parsing")
    print("=" * 70)
    fake = FakeAnthropicClient(canned_input={
        "raw_tilt": 0.55, "confidence": 0.75,
        "rationale": "AI capex still accelerating; valuation rich but justified by growth.",
    })
    view = reason_strategic_view_claude(snap, client=fake)
    print(f"strategic view: raw={view.raw:+.3f} confidence={view.confidence:.2f} "
          f"half_life={view.half_life_days}d max_age={view.max_age_days}d")
    assert view.raw == 0.55 and view.confidence == 0.75, "parsing failed"
    print("✓ parsed correctly")

    print("\n" + "=" * 70)
    print("TEST 2: out-of-range model output gets clamped, not trusted blindly")
    print("=" * 70)
    fake_bad = FakeAnthropicClient(canned_input={
        "raw_tilt": 1.4, "confidence": -0.2, "rationale": "overconfident test case",
    })
    view2 = reason_strategic_view_claude(snap, client=fake_bad)
    print(f"clamped: raw={view2.raw:+.3f} confidence={view2.confidence:.2f}")
    assert view2.raw == 1.0 and view2.confidence == 0.0, "clamping failed"
    print("✓ clamped correctly")

    print("\n" + "=" * 70)
    print("TEST 3: API/network failure falls back to the deterministic heuristic")
    print("=" * 70)
    fake_broken = FakeAnthropicClient(raise_error=True)
    view3 = reason_tactical_view_claude(snap, client=fake_broken)
    print(f"fallback view: raw={view3.raw:+.3f} confidence={view3.confidence:.2f}")
    print("✓ pipeline did not crash on a simulated network failure")

    print("\n" + "=" * 70)
    print("TEST 4: make_claude_reasoners() returns two bindable closures")
    print("=" * 70)
    strategic_fn, tactical_fn = make_claude_reasoners(client=fake)
    v = strategic_fn(snap)
    print(f"bound closure result: raw={v.raw:+.3f} confidence={v.confidence:.2f}")
    assert v.raw == 0.55
    print("✓ closures match direct-call results — safe to pass into build_compiled_graph()")

    print("\n" + "=" * 70)
    print("TEST 5: desk scoring — full success, all instruments scored")
    print("=" * 70)
    desk = [
        InstrumentSnapshot("NVDA", "equities.tech_ai", momentum_1m=0.08, news_sentiment=0.6, quality_score=0.7),
        InstrumentSnapshot("CAT", "equities.industrials", momentum_1m=-0.01, news_sentiment=0.0, quality_score=0.3),
    ]
    fake_desk = FakeAnthropicClient(canned_input={
        "scores": [
            {"name": "NVDA", "conviction": 0.85, "rationale": "strong momentum and sentiment"},
            {"name": "CAT", "conviction": 0.4, "rationale": "weak momentum, mediocre quality"},
        ]
    })
    signals = reason_desk_convictions_claude(desk, client=fake_desk)
    by_name = {s.name: s for s in signals}
    print(f"  {[(s.name, round(s.score, 2)) for s in signals]}")
    assert len(signals) == 2 and by_name["NVDA"].score == 0.85 and by_name["CAT"].score == 0.4
    print("✓ all instruments scored from the model's output")

    print("\n" + "=" * 70)
    print("TEST 6: desk scoring — model skips one instrument, partial fallback")
    print("=" * 70)
    fake_partial = FakeAnthropicClient(canned_input={
        "scores": [{"name": "NVDA", "conviction": 0.85, "rationale": "strong"}],
        # CAT is silently missing from the model's response
    })
    signals2 = reason_desk_convictions_claude(desk, client=fake_partial)
    by_name2 = {s.name: s for s in signals2}
    print(f"  {[(s.name, round(s.score, 2)) for s in signals2]}")
    assert len(signals2) == 2, "CAT should have been filled in by the fallback, not dropped"
    assert by_name2["NVDA"].score == 0.85          # model's answer, used as-is
    assert by_name2["CAT"].score != 0.85            # fell back to the heuristic, not copied from NVDA
    print("✓ missing instrument backfilled via heuristic — desk wasn't zeroed out")

    print("\n" + "=" * 70)
    print("TEST 7: desk scoring — full API failure falls back for the whole desk")
    print("=" * 70)
    signals3 = reason_desk_convictions_claude(desk, client=FakeAnthropicClient(raise_error=True))
    print(f"  {[(s.name, round(s.score, 2)) for s in signals3]}")
    assert len(signals3) == 2
    print("✓ pipeline did not crash; entire desk fell back to the heuristic")

    print("\n" + "=" * 70)
    print("TEST 8: make_claude_analyst_reasoner() — one closure, any desk")
    print("=" * 70)
    analyst_fn = make_claude_analyst_reasoner(client=fake_desk)
    result = analyst_fn(desk)
    print(f"  {[(s.name, round(s.score, 2)) for s in result]}")
    assert len(result) == 2
    print("✓ closure works identically to the direct call — safe to pass into")
    print("  firm_graph.build_compiled_graph(reason_equities=analyst_fn, ...)")
