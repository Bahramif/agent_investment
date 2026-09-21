"""
firm_graph.py
=============
Assembles everything built in this conversation into one compiled LangGraph:

  START
    ├→ equities_analyst  ┐
    ├→ forex_analyst     ├→ strategist_reasoning → combine_tilts → freshness_gate ─┬→ (clean) portfolio_construction
    └→ crypto_analyst    ┘                                                          └→ (stale, material) human_approval_freshness
                                                                                            ├→ re_affirm/revise → strategist_reasoning (loop)
                                                                                            └→ override → portfolio_construction
                                                    portfolio_construction → risk_check ─┬→ approve → execute → END
                                                                                          └→ reject  → portfolio_construction (retry, capped)

strategist_reasoning derives each theme's ThemeStrategy from incoming macro
data + carried-forward memory (cadence-gated: strategic views are expensive
and only re-derived periodically or when data arrives and enough time has
passed; tactical views refresh whenever fresh data shows up). combine_tilts
then applies strategist.py's two-horizon cap/decay math, unchanged.

Analyst nodes (equities/forex/crypto) are built the same way: a factory
(make_analyst_node) binds a scoring function — heuristic by default, or
Claude-backed via claude_reasoning.make_claude_analyst_reasoner() — once at
graph-build time, and each desk's candidate data arrives as state input
(state["<asset_class>_snapshots"]) rather than being hardcoded in the node.

Every node function below is PURE LOGIC — it takes a state dict, returns a
partial update, and imports nothing from langgraph at module scope. That's
deliberate: it means every node can be unit-tested with plain Python calls,
independent of whether langgraph itself is installed. Only build_compiled_graph()
touches the langgraph API, via a deferred import.

HONEST LIMITATION: this sandbox has no network access, so `pip install
langgraph` cannot run here and the actual StateGraph/compile()/interrupt()
calls are UNTESTED in this environment. Every node's business logic below
(macro reasoning, tilt combination, freshness gate, mandate allocation, risk
check) IS tested here, via the multi-cycle walk-through in __main__ that
calls the node functions directly in sequence — the same functions
build_compiled_graph() wires into the real graph. Install langgraph in your
own environment and build_compiled_graph() should work as written; verify
against current LangGraph docs before relying on it, since the API evolves.
"""

from __future__ import annotations

import operator
from dataclasses import asdict
from typing import Annotated, Any, Callable, Dict, List, Optional, Set, TypedDict

from mandate_allocator import Signal, allocate, Book
from strategist import (
    HorizonView, ThemeStrategy, TiltCaps, ThemeTilt,
    combine, freshness_gate, FreshnessViolation,
)
from macro_reasoning import (
    MacroSnapshot, reason_strategic_view, reason_tactical_view,
)
from analyst_reasoning import InstrumentSnapshot, reason_desk


# -----------------------------------------------------------------------------
# Shared state schema
# -----------------------------------------------------------------------------
class FirmState(TypedDict, total=False):
    mandate: dict
    equities_snapshots: List[InstrumentSnapshot]   # this cycle's candidate data, per desk
    forex_snapshots: List[InstrumentSnapshot]
    crypto_snapshots: List[InstrumentSnapshot]
    macro_snapshots: List["MacroSnapshot"]      # this cycle's incoming data, per theme
    theme_memory: Dict[str, ThemeStrategy]      # carried-forward views, ages included
    theme_strategies: List[ThemeStrategy]       # this cycle's resolved views (memory, updated)
    elapsed_days: float                         # time since the previous cycle ran
    force_refresh_themes: Set[str]              # themes a human explicitly told us to re-derive
    tilt_caps: TiltCaps
    prior_theme_budgets: Dict[str, float]      # what themes drew last cycle, for freshness materiality
    material_threshold: float
    risk_limits: dict

    signals: Annotated[List[Signal], operator.add]   # parallel-safe merge from analysts
    theme_tilts: Dict[str, ThemeTilt]
    freshness_violations: List[FreshnessViolation]
    needs_human_freshness: bool
    human_response_freshness: Optional[str]

    proposal: Book
    # Book.notes (name -> {rationale, risk_note, momentum_1m, news_sentiment,
    # quality_score, volatility, conviction, expected_return, rating})
    # mirrored at top level for the same reason final_weights is: only
    # declared FirmState keys survive the JSON round-trip through a real
    # deployed query() call — a nested dataclass field buried inside
    # `proposal` isn't something a caller can rely on the same way. See the
    # final_weights bug fix below.
    book_notes: Dict[str, Dict[str, Any]]
    risk_verdict: str
    risk_reasons: List[str]
    retries: int
    status: str
    # BUG FIX: make_execute_node()'s return value has always included these two
    # keys, but neither was declared here — StateGraph(FirmState) only threads
    # through channels declared in this schema, so an undeclared key returned
    # by a node is silently dropped rather than erroring. Confirmed live: a
    # real deployed query() came back with status="executed" (declared) but
    # final_weights=None (wasn't declared), even though execute_node's own
    # code unconditionally sets it.
    final_weights: Dict[str, float]
    execution_workflow_id: str


# -----------------------------------------------------------------------------
# Analyst nodes (parallel fan-out from START)
#
# make_analyst_node() is a factory for the same reason
# make_strategist_reasoning_node() is: the scoring function (heuristic, or
# Claude-backed via claude_reasoning.make_claude_analyst_reasoner()) gets
# bound ONCE at graph-build time, not read from state on every call.
#
# Each desk's candidate data (InstrumentSnapshot list) is now an INPUT — via
# state["<asset_class>_snapshots"] — rather than hardcoded in the node, same
# move as macro_snapshots was for the strategist.
# -----------------------------------------------------------------------------
def make_analyst_node(
    state_key: str,
    reason_fn: Callable[[List[InstrumentSnapshot]], List[Signal]] = reason_desk,
):
    def _node(state: FirmState) -> dict:
        snapshots = state.get(state_key, [])
        return {"signals": reason_fn(snapshots) if snapshots else []}
    return _node


# Module-level defaults — heuristic scoring, used by the __main__ demo below
# and by build_compiled_graph() when no reasoners are explicitly supplied.
equities_analyst_node = make_analyst_node("equities_snapshots")
forex_analyst_node = make_analyst_node("forex_snapshots")
crypto_analyst_node = make_analyst_node("crypto_snapshots")


# -----------------------------------------------------------------------------
# Cadence & memory helpers.
#
# The actual reasoning (MacroSnapshot -> HorizonView) lives in
# macro_reasoning.py by default, or claude_reasoning.py for a real model call.
# Neither is hardcoded here — see make_strategist_reasoning_node() below.
# Strategic reasoning is expensive (a real LLM call weighing years of
# structural evidence) and doesn't need re-running every cycle just because
# data arrived — re-derive on a cadence, not on every tick.
# -----------------------------------------------------------------------------
STRATEGIC_REVIEW_CADENCE_DAYS = 90.0


def _age_view(view: HorizonView, elapsed_days: float) -> HorizonView:
    """Carry a view forward without re-deriving it — just let it get older."""
    return HorizonView(raw=view.raw, confidence=view.confidence,
                       age_days=view.age_days + elapsed_days,
                       half_life_days=view.half_life_days, max_age_days=view.max_age_days)


def _strategic_due(prior: Optional[ThemeStrategy], elapsed_days: float, forced: bool) -> bool:
    if prior is None or forced:
        return True
    return prior.strategic.age_days + elapsed_days >= STRATEGIC_REVIEW_CADENCE_DAYS


def make_strategist_reasoning_node(
    reason_strategic: Callable[[MacroSnapshot], HorizonView] = reason_strategic_view,
    reason_tactical: Callable[[MacroSnapshot], HorizonView] = reason_tactical_view,
):
    """Factory, not a plain function — deliberately.

    The reasoning functions (heuristic, or Claude-backed via
    claude_reasoning.make_claude_reasoners()) get bound ONCE here, at
    graph-build time — analogous to Vertex Agent Engine's set_up(), which
    runs once per container. They are NOT read from graph state on every
    call, because a durable (Postgres-backed) checkpointer needs state to
    stay plain, JSON-serializable data — a bound closure over a live Claude
    client is exactly the kind of thing that can't round-trip through a
    checkpoint. Swap reasoners by calling this factory differently, not by
    passing callables through state.
    """
    def _node(state: FirmState) -> dict:
        """Resolves this cycle's ThemeStrategy per theme from macro data + memory.

        Three cases per theme, and this is where staleness actually comes from:
          1. Fresh data + strategic review due (cadence, or human forced it)
             -> re-derive both horizons, age resets to 0.
          2. Fresh data but strategic not due yet
             -> tactical re-derives (cheap, always refreshed); strategic just ages.
          3. NO fresh data this cycle (a real data-pipeline gap)
             -> nothing can be re-derived; both horizons simply age. Enough
                missed cycles in a row and the strategic view crosses its
                max_age_days on its own — this is the realistic path to a
                freshness-gate violation, not an arbitrarily hand-set one.
        """
        snapshots = {s.theme: s for s in state.get("macro_snapshots", [])}
        prior_memory = state.get("theme_memory", {})
        elapsed = state.get("elapsed_days", 1.0)
        forced = state.get("force_refresh_themes", set())

        updated: Dict[str, ThemeStrategy] = {}
        all_themes = set(prior_memory) | set(snapshots)
        for theme in all_themes:
            prior = prior_memory.get(theme)
            snap = snapshots.get(theme)
            if snap is None:                                   # case 3: data gap
                if prior is None:
                    continue                                    # no data, no memory — nothing to do
                updated[theme] = ThemeStrategy(theme, _age_view(prior.strategic, elapsed),
                                              _age_view(prior.tactical, elapsed))
                continue
            tactical = reason_tactical(snap)                     # cheap — refresh whenever data arrives
            if _strategic_due(prior, elapsed, theme in forced):  # case 1
                strategic = reason_strategic(snap)
            else:                                                # case 2
                strategic = _age_view(prior.strategic, elapsed)
            updated[theme] = ThemeStrategy(theme, strategic, tactical)

        return {"theme_strategies": list(updated.values()), "theme_memory": updated}
    return _node


# Module-level default — heuristic reasoning, used by the __main__ demo below
# and by build_compiled_graph() when no reasoners are explicitly supplied.
strategist_reasoning_node = make_strategist_reasoning_node()


# -----------------------------------------------------------------------------
# Combine: two-horizon tilts (from strategist.py, unmodified)
# -----------------------------------------------------------------------------
def combine_tilts_node(state: FirmState) -> dict:
    caps = state["tilt_caps"]
    tilts = {s.theme: combine(s, caps) for s in state["theme_strategies"]}
    return {"theme_tilts": tilts}


# -----------------------------------------------------------------------------
# Freshness gate: catches stale views that were drawing material capital
# -----------------------------------------------------------------------------
def freshness_gate_node(state: FirmState) -> dict:
    tilts = list(state["theme_tilts"].values())
    violations = freshness_gate(
        tilts,
        prior_budgets=state.get("prior_theme_budgets", {}),
        material_threshold=state.get("material_threshold", 0.03),
    )
    return {"freshness_violations": violations, "needs_human_freshness": len(violations) > 0}


def freshness_router(state: FirmState) -> str:
    return "human_approval_freshness" if state.get("needs_human_freshness") else "portfolio_construction"


def human_approval_freshness_node(state: FirmState) -> dict:
    """Pauses the run and asks a human to re-affirm, override, or revise.
    Uses LangGraph's interrupt() — only reachable when langgraph is installed
    and this node is actually invoked inside a compiled, checkpointed graph."""
    from langgraph.types import interrupt
    violations = state["freshness_violations"]
    prompt = "Stale strategist views need attention:\n" + "\n".join(
        f"  • {v.message}" for v in violations
    ) + "\n\nRespond: re_affirm:{theme} | override | revise:{theme}:{guidance}"
    response = interrupt(prompt)
    return _parse_freshness_response(response)


def _parse_freshness_response(response: str) -> dict:
    """Split out so the manual walk-through can test this without interrupt()."""
    out: dict = {"human_response_freshness": response}
    if response.startswith("re_affirm:"):
        out["force_refresh_themes"] = {response.split(":", 1)[1].strip()}
    elif response.startswith("revise:"):
        parts = response.split(":", 2)
        theme = parts[1].strip() if len(parts) > 1 else ""
        # In production: the guidance text (parts[2]) would be passed into the
        # reasoning prompt for this theme, not just force a bare re-derivation.
        out["force_refresh_themes"] = {theme} if theme else set()
    return out


def post_freshness_approval_router(state: FirmState) -> str:
    resp = (state.get("human_response_freshness") or "")
    if resp.startswith("re_affirm") or resp.startswith("revise"):
        return "strategist_reasoning"    # loop back to re-derive, not just recombine
    return "portfolio_construction"      # "override" (or anything else) → proceed


# -----------------------------------------------------------------------------
# Portfolio construction: mandate + strategist tilts + analyst signals → book
# (from mandate_allocator.py, unmodified)
# -----------------------------------------------------------------------------
def portfolio_construction_node(state: FirmState) -> dict:
    book = allocate(state["mandate"], state["signals"], theme_tilts=state.get("theme_tilts"))
    return {"proposal": book, "book_notes": book.notes}


# -----------------------------------------------------------------------------
# Risk check: deterministic, no LLM. Vetoes anything past the hard limits.
# -----------------------------------------------------------------------------
def _deterministic_risk_check(book: Book, limits: dict) -> tuple[str, List[str]]:
    max_name = limits.get("max_single_name", 0.10)
    exempt = set(n.upper() for n in limits.get("concentration_exempt", ["CASH"]))
    reasons = [f"{n} at {w:.1%} exceeds max_single_name {max_name:.1%}"
              for n, w in book.weights.items()
              if n.upper() not in exempt and w > max_name + 1e-9]
    total = sum(book.weights.values())
    if abs(total - 1.0) > 1e-6:
        reasons.append(f"total book {total:.2%} != 100%")
    return ("reject" if reasons else "approve"), reasons


def risk_check_node(state: FirmState) -> dict:
    verdict, reasons = _deterministic_risk_check(state["proposal"], state.get("risk_limits", {}))
    update: dict = {"risk_verdict": verdict, "risk_reasons": reasons}
    if verdict != "approve":
        # BUG FIX: risk_router's retry cap reads state["retries"], but nothing
        # incremented it — a rejected proposal looped portfolio_construction
        # <-> risk_check forever (no re-scoring happens between retries either,
        # so a genuine rejection can never self-correct), only stopping when
        # LangGraph's own recursion_limit killed the run. Confirmed live: hit
        # the default 10007-iteration ceiling on a real deployed query().
        update["retries"] = state.get("retries", 0) + 1
    return update


def risk_router(state: FirmState) -> str:
    verdict = state.get("risk_verdict")
    if verdict == "approve":
        return "execute"
    retries = state.get("retries", 0)
    return "portfolio_construction" if retries < 3 else "END"


# -----------------------------------------------------------------------------
# Execution handoff — factory, same pattern as everywhere else in this file.
# Deliberately thin: NO order logic lives here, only a handoff to whatever
# durable execution layer trigger_fn points at (see execution_workflow.py).
# -----------------------------------------------------------------------------
def make_execute_node(trigger_fn: Optional[Callable[[Book], Any]] = None):
    """trigger_fn: Book -> an execution-workflow ID (or similar handle) —
    either a plain sync callable, or an async one (e.g. from
    execution_workflow.make_execution_trigger(), which returns Book ->
    Awaitable[str]). Left as None (the default used below and in the
    __main__ demo), this just reports the final weights directly — there's
    no real execution layer to hand off to in a demo.

    If trigger_fn is async, this returns an ASYNC node function instead of
    a sync one. LangGraph natively supports mixing sync and async nodes in
    the same graph when invoked via .ainvoke() — this is exactly the async
    path vertex_deploy.py's async_query() method uses, so a real Temporal
    handoff never blocks the request. This node must never await order
    fills — only the trigger call itself, which returns as soon as the
    workflow is STARTED."""
    import asyncio

    if trigger_fn is not None and asyncio.iscoroutinefunction(trigger_fn):
        async def _async_node(state: FirmState) -> dict:
            workflow_id = await trigger_fn(state["proposal"])
            return {"status": "dispatched", "execution_workflow_id": workflow_id,
                   "final_weights": dict(state["proposal"].weights)}
        return _async_node

    def _node(state: FirmState) -> dict:
        if trigger_fn is None:
            return {"status": "executed", "final_weights": dict(state["proposal"].weights)}
        workflow_id = trigger_fn(state["proposal"])
        return {"status": "dispatched", "execution_workflow_id": workflow_id,
               "final_weights": dict(state["proposal"].weights)}
    return _node


# Module-level default — used by the __main__ demo and by build_compiled_graph()
# when no execute_trigger is explicitly supplied.
execute_node = make_execute_node()


def _traced(name: str, fn):
    """Wraps a node function in an OpenTelemetry span named after the node —
    so once a TracerProvider + exporter is configured (see vertex_deploy.py's
    set_up(), which points it at Cloud Trace), each node's real execution
    shows up as a span in the Agent Engine console's Traces tab under that
    exact node name, matching web_ui.py's /about page topology diagram
    one-for-one. A plain no-op wrapper — falls back to the unwrapped fn — if
    opentelemetry isn't installed, so the __main__ demo below and any test
    that calls build_compiled_graph() directly never need it as a hard
    dependency."""
    try:
        from opentelemetry import trace
    except ImportError:
        return fn
    import asyncio
    tracer = trace.get_tracer("firm_graph")
    if asyncio.iscoroutinefunction(fn):
        async def _wrapped(state):
            with tracer.start_as_current_span(name):
                return await fn(state)
        return _wrapped

    def _wrapped(state):
        with tracer.start_as_current_span(name):
            return fn(state)
    return _wrapped


# -----------------------------------------------------------------------------
# Graph assembly — the only function that touches the langgraph API
# -----------------------------------------------------------------------------
def build_compiled_graph(
    checkpointer=None,
    reason_strategic: Optional[Callable[[MacroSnapshot], HorizonView]] = None,
    reason_tactical: Optional[Callable[[MacroSnapshot], HorizonView]] = None,
    reason_equities: Optional[Callable[[List[InstrumentSnapshot]], List[Signal]]] = None,
    reason_forex: Optional[Callable[[List[InstrumentSnapshot]], List[Signal]]] = None,
    reason_crypto: Optional[Callable[[List[InstrumentSnapshot]], List[Signal]]] = None,
    ingestion_node: Optional[Callable[[dict], dict]] = None,
    execute_trigger: Optional[Callable[[Book], Any]] = None,
):
    """Pass reason_strategic/reason_tactical and/or reason_equities/
    reason_forex/reason_crypto to swap in Claude-backed reasoning — e.g.
    from claude_reasoning.make_claude_reasoners() and
    claude_reasoning.make_claude_analyst_reasoner() (the same callable works
    for all three desks). Left as None, the graph uses the deterministic
    heuristics from macro_reasoning.py / analyst_reasoning.py.

    Pass ingestion_node (e.g. from data_ingestion.make_data_ingestion_node())
    to have the graph fetch its own snapshot data each run, inserted as
    START -> ingestion_node -> the three analyst nodes. Left as None, the
    three analyst nodes read state["<asset_class>_snapshots"] directly from
    whatever was passed into invoke() — which is how the __main__ demo below
    supplies data, so that demo's behavior is unaffected either way.

    Pass execute_trigger to hand off to a real execution workflow (see
    make_execute_node's docstring and execution_workflow.py); left as None,
    the execute node just reports final weights with nothing dispatched."""
    from langgraph.graph import StateGraph, START, END

    strategist_node_fn = make_strategist_reasoning_node(
        reason_strategic or reason_strategic_view,
        reason_tactical or reason_tactical_view,
    )
    equities_node_fn = make_analyst_node("equities_snapshots", reason_equities or reason_desk)
    forex_node_fn = make_analyst_node("forex_snapshots", reason_forex or reason_desk)
    crypto_node_fn = make_analyst_node("crypto_snapshots", reason_crypto or reason_desk)
    execute_node_fn = make_execute_node(execute_trigger)

    g = StateGraph(FirmState)
    for name, fn in [
        ("equities_analyst", equities_node_fn),
        ("forex_analyst", forex_node_fn),
        ("crypto_analyst", crypto_node_fn),
        ("strategist_reasoning", strategist_node_fn),
        ("combine_tilts", combine_tilts_node),
        ("freshness_gate", freshness_gate_node),
        ("human_approval_freshness", human_approval_freshness_node),
        ("portfolio_construction", portfolio_construction_node),
        ("risk_check", risk_check_node),
        ("execute", execute_node_fn),
    ]:
        g.add_node(name, _traced(name, fn))

    if ingestion_node is not None:
        g.add_node("data_ingestion", _traced("data_ingestion", ingestion_node))
        g.add_edge(START, "data_ingestion")
        for a in ("equities_analyst", "forex_analyst", "crypto_analyst"):
            g.add_edge("data_ingestion", a)
            g.add_edge(a, "strategist_reasoning")
    else:
        for a in ("equities_analyst", "forex_analyst", "crypto_analyst"):
            g.add_edge(START, a)
            g.add_edge(a, "strategist_reasoning")

    g.add_edge("strategist_reasoning", "combine_tilts")
    g.add_edge("combine_tilts", "freshness_gate")
    g.add_conditional_edges("freshness_gate", freshness_router, {
        "human_approval_freshness": "human_approval_freshness",
        "portfolio_construction": "portfolio_construction",
    })
    g.add_conditional_edges("human_approval_freshness", post_freshness_approval_router, {
        "strategist_reasoning": "strategist_reasoning",
        "portfolio_construction": "portfolio_construction",
    })
    g.add_edge("portfolio_construction", "risk_check")
    g.add_conditional_edges("risk_check", risk_router, {
        "execute": "execute",
        "portfolio_construction": "portfolio_construction",
        "END": END,
    })
    g.add_edge("execute", END)

    return g.compile(checkpointer=checkpointer)


# =============================================================================
# Manual walk-through — exercises every node's REAL logic without needing
# langgraph installed. This is not a graph traversal; it's a hand-driven
# simulation of the same sequence build_compiled_graph() would execute,
# so you can verify the business logic end-to-end in this sandbox.
# =============================================================================
if __name__ == "__main__":
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
    caps = TiltCaps(strategic_cap=0.15, tactical_cap=0.06, total_cap=0.15)

    def run_one_cycle(state: FirmState, label: str) -> FirmState:
        """One pass through the graph, manually walked."""
        state["signals"] = []
        for analyst in (equities_analyst_node, forex_analyst_node, crypto_analyst_node):
            state["signals"] = state["signals"] + analyst(state)["signals"]

        state.update(strategist_reasoning_node(state))
        state.update(combine_tilts_node(state))
        state.update(freshness_gate_node(state))
        route = freshness_router(state)
        print(f"[{label}] freshness_gate route → {route}")

        if route == "human_approval_freshness":
            for v in state["freshness_violations"]:
                print(f"    ⚠ {v.message}")
            return state  # caller decides how to respond before continuing

        state.pop("force_refresh_themes", None)
        state.update(portfolio_construction_node(state))
        state.update(risk_check_node(state))
        route = risk_router(state)
        print(f"[{label}] risk_check route → {route} (verdict={state['risk_verdict']}, "
              f"reasons={state['risk_reasons'] or 'none'})")
        if route == "execute":
            state.update(execute_node(state))
            print(f"[{label}] status: {state['status']}")
        return state

    print("=" * 70)
    print("MULTI-CYCLE SIMULATION: staleness emerges from a real data gap,")
    print("not a hand-set age. Same scenario, three cycles over time.")
    print("=" * 70)

    state: FirmState = {
        "mandate": mandate, "tilt_caps": caps, "theme_memory": {},
        "material_threshold": 0.03, "risk_limits": {"max_single_name": 0.10},
        "prior_theme_budgets": {},
        # Analyst desk data — now an INPUT the analyst nodes score, not
        # hardcoded inside them. Held constant across cycles here purely to
        # keep the demo focused on strategist staleness; a real deployment
        # would refresh these every cycle same as macro_snapshots.
        "equities_snapshots": [
            InstrumentSnapshot("NVDA", "equities.tech_ai", momentum_1m=0.08, news_sentiment=0.6, quality_score=0.7),
            InstrumentSnapshot("MSFT", "equities.tech_ai", momentum_1m=0.04, news_sentiment=0.4, quality_score=0.6),
            InstrumentSnapshot("GOOGL", "equities.tech_ai", momentum_1m=0.03, news_sentiment=0.3, quality_score=0.6),
            InstrumentSnapshot("AVGO", "equities.tech_ai", momentum_1m=0.02, news_sentiment=0.2, quality_score=0.5),
            InstrumentSnapshot("XOM", "equities.energy", momentum_1m=0.03, news_sentiment=0.3, quality_score=0.4),
            InstrumentSnapshot("FSLR", "equities.energy", momentum_1m=0.02, news_sentiment=0.2, quality_score=0.4),
            InstrumentSnapshot("CAT", "equities.industrials", momentum_1m=-0.01, news_sentiment=0.0, quality_score=0.3),
        ],
        "forex_snapshots": [
            InstrumentSnapshot("EURUSD", "forex", momentum_1m=0.02, news_sentiment=0.3, quality_score=0.2),
            InstrumentSnapshot("USDJPY", "forex", momentum_1m=0.01, news_sentiment=0.1, quality_score=0.1),
            InstrumentSnapshot("GBPUSD", "forex", momentum_1m=-0.01, news_sentiment=-0.1, quality_score=0.0),
        ],
        "crypto_snapshots": [
            InstrumentSnapshot("BTC", "crypto", momentum_1m=0.10, news_sentiment=0.6, quality_score=0.5),
            InstrumentSnapshot("ETH", "crypto", momentum_1m=0.06, news_sentiment=0.4, quality_score=0.4),
            InstrumentSnapshot("SOL", "crypto", momentum_1m=0.04, news_sentiment=0.2, quality_score=0.2),
        ],
    }

    # --- Cycle 1 (day 0): both themes have data. Fresh derive, no prior memory.
    state["elapsed_days"] = 0.0
    state["macro_snapshots"] = [
        MacroSnapshot("energy", momentum_1m=0.03, news_sentiment=0.4,
                     valuation_percentile=0.4, secular_score=0.3),
        MacroSnapshot("industrials", momentum_1m=-0.01, news_sentiment=0.0,
                     valuation_percentile=0.5, secular_score=-0.1),
    ]
    state = run_one_cycle(state, "cycle 1, day 0")
    for th, ts in state["theme_memory"].items():
        print(f"    {th}: strategic age={ts.strategic.age_days:.0f}d, tactical age={ts.tactical.age_days:.0f}d")
    state["prior_theme_budgets"] = {"energy": 0.09, "industrials": 0.07}  # what cycle 1 actually funded

    # --- Cycle 2 (30 days later): both themes still have data, but strategic
    #     review isn't due yet (30 < 90-day cadence) — only tactical refreshes.
    print()
    state["elapsed_days"] = 30.0
    state["macro_snapshots"] = [
        MacroSnapshot("energy", momentum_1m=0.05, news_sentiment=0.5,
                     valuation_percentile=0.42, secular_score=0.3),
        MacroSnapshot("industrials", momentum_1m=0.01, news_sentiment=0.1,
                     valuation_percentile=0.5, secular_score=-0.1),
    ]
    state = run_one_cycle(state, "cycle 2, day 30")
    for th, ts in state["theme_memory"].items():
        print(f"    {th}: strategic age={ts.strategic.age_days:.0f}d, tactical age={ts.tactical.age_days:.0f}d")

    # --- Cycle 3 (160 days later — a real gap, e.g. the data vendor dropped
    #     "energy" theme coverage). energy gets NO snapshot this cycle, so it
    #     just ages: 30 + 160 = 190 days, past its 180-day max_age_days. This
    #     is the freshness gate catching a REAL failure mode, not a contrived one.
    print()
    state["elapsed_days"] = 160.0
    state["macro_snapshots"] = [
        MacroSnapshot("industrials", momentum_1m=0.02, news_sentiment=0.2,
                     valuation_percentile=0.48, secular_score=-0.05),
        # energy: no snapshot this cycle — simulates a data-feed gap
    ]
    state["prior_theme_budgets"] = {"energy": 0.10, "industrials": 0.08}
    state = run_one_cycle(state, "cycle 3, day 190 (energy data gap)")

    if state.get("needs_human_freshness"):
        # Simulate a human re-affirming energy: force a fresh derive next cycle.
        response = "re_affirm:energy"
        print(f"\n[human] responding: '{response}'")
        state.update(_parse_freshness_response(response))
        route = post_freshness_approval_router(state)
        print(f"[human] → route {route}")
        assert route == "strategist_reasoning"

        # --- Cycle 4: same day, but now WITH energy data arriving and the
        #     forced re-derivation in effect — this is what re_affirm actually does.
        print()
        state["macro_snapshots"] = [
            MacroSnapshot("energy", momentum_1m=0.02, news_sentiment=0.1,
                         valuation_percentile=0.55, secular_score=0.15),
            MacroSnapshot("industrials", momentum_1m=0.02, news_sentiment=0.2,
                         valuation_percentile=0.48, secular_score=-0.05),
        ]
        state["elapsed_days"] = 0.0   # same moment — this is the re-affirmation, not a new tick
        state = run_one_cycle(state, "cycle 4, re-affirmed energy")
        for th, ts in state["theme_memory"].items():
            print(f"    {th}: strategic age={ts.strategic.age_days:.0f}d (re-derived)")

    print("\n" + "=" * 70)
    print("TEST: wiring CLAUDE-backed reasoning into the graph via the factory")
    print("(uses claude_reasoning's fake client — no network call is made)")
    print("=" * 70)
    try:
        from claude_reasoning import (
            make_claude_reasoners, make_claude_analyst_reasoner, FakeAnthropicClient,
        )

        fake_client = FakeAnthropicClient(canned_input={
            "raw_tilt": 0.5, "confidence": 0.8,
            "rationale": "test rationale for wiring proof",
        })
        claude_strategic, claude_tactical = make_claude_reasoners(client=fake_client)
        claude_backed_node = make_strategist_reasoning_node(claude_strategic, claude_tactical)

        test_state: FirmState = {
            "macro_snapshots": [MacroSnapshot("AI", momentum_1m=0.05, news_sentiment=0.5,
                                             valuation_percentile=0.6, secular_score=0.5)],
            "theme_memory": {}, "elapsed_days": 0.0,
        }
        result = claude_backed_node(test_state)
        ai_strategy = result["theme_strategies"][0]
        print(f"Claude-backed strategic view for AI: raw={ai_strategy.strategic.raw:+.3f} "
              f"confidence={ai_strategy.strategic.confidence:.2f}")
        assert ai_strategy.strategic.raw == 0.5 and ai_strategy.strategic.confidence == 0.8
        print("✓ strategist reasoner correctly wired through make_strategist_reasoning_node()")

        fake_desk_client = FakeAnthropicClient(canned_input={
            "scores": [{"name": "NVDA", "conviction": 0.8, "rationale": "wiring proof"}]
        })
        claude_analyst = make_claude_analyst_reasoner(client=fake_desk_client)
        equities_claude_node = make_analyst_node("equities_snapshots", claude_analyst)
        analyst_result = equities_claude_node({
            "equities_snapshots": [InstrumentSnapshot("NVDA", "equities.tech_ai",
                                                       momentum_1m=0.08, news_sentiment=0.6, quality_score=0.7)],
        })
        print(f"Claude-backed analyst signal: {analyst_result['signals'][0]}")
        assert analyst_result["signals"][0].score == 0.8
        print("✓ analyst reasoner correctly wired through make_analyst_node()")
        print("  In production: build_compiled_graph(reason_strategic=claude_strategic,")
        print("                                       reason_tactical=claude_tactical,")
        print("                                       reason_equities=claude_analyst,")
        print("                                       reason_forex=claude_analyst,")
        print("                                       reason_crypto=claude_analyst)")
    except ImportError as e:
        print(f"(claude_reasoning module or `anthropic` package not available here: {e})")

    print("\n" + "=" * 70)
    print("TEST: wiring the data-ingestion node in — START -> ingestion -> analysts")
    print("=" * 70)
    try:
        from data_ingestion import (
            UniverseConfig, make_data_ingestion_node, FakeMarketDataClient, FakeNewsClient,
        )
        universe = UniverseConfig(
            equities=[("NVDA", "equities.tech_ai", 0.7), ("CAT", "equities.industrials", 0.3)],
            forex=[], crypto=[], themes=[],
        )
        fake_market = FakeMarketDataClient({"NVDA": [100, 105, 112], "CAT": [100, 98, 96]})
        fake_news = FakeNewsClient({"NVDA": ["NVDA surges on record demand"]})
        ingestion = make_data_ingestion_node(universe, fake_market, fake_news)

        ingested = ingestion({})
        equities_node = make_analyst_node("equities_snapshots")   # default heuristic
        signals_out = equities_node(ingested)
        print(f"  ingested {len(ingested['equities_snapshots'])} equities snapshots -> "
              f"{[(s.name, round(s.score, 3)) for s in signals_out['signals']]}")
        assert len(signals_out["signals"]) == 2
        print("✓ ingestion node's output feeds directly into the analyst node —")
        print("  build_compiled_graph(ingestion_node=ingestion) wires START -> data_ingestion")
        print("  -> {equities,forex,crypto}_analyst automatically")
    except ImportError as e:
        print(f"(data_ingestion module not available here: {e})")

    print("\n" + "=" * 70)
    print("TEST: wiring the execution handoff — execute node dispatches, doesn't block")
    print("=" * 70)
    dispatched_books = []

    def fake_trigger(book: Book) -> str:
        dispatched_books.append(book)
        return "workflow-id-abc123"

    handoff_node = make_execute_node(fake_trigger)
    fake_state: FirmState = {"proposal": Book(weights={"NVDA": 0.3, "CASH": 0.7})}
    handoff_result = handoff_node(fake_state)
    print(f"  sync: {handoff_result}")
    assert handoff_result["status"] == "dispatched"
    assert handoff_result["execution_workflow_id"] == "workflow-id-abc123"
    assert len(dispatched_books) == 1
    print("✓ sync execute node calls the trigger and returns immediately with a workflow ID —")
    print("  it does NOT wait for fills.")

    try:
        import asyncio
        from execution_workflow import make_execution_trigger, ExecutionWorkflowInput

        async def fake_async_trigger_execution(wf_input: ExecutionWorkflowInput, target: str) -> str:
            return "async-workflow-id-xyz"

        async_trigger = make_execution_trigger(
            temporal_target="fake:7233", total_capital=100_000.0,
            get_current_positions=lambda: {"NVDA": 0.10, "CASH": 0.90},
            _trigger_execution_fn=fake_async_trigger_execution,
        )
        async_handoff_node = make_execute_node(async_trigger)
        async_state: FirmState = {"proposal": Book(weights={"NVDA": 0.3, "CASH": 0.7})}
        async_result = asyncio.run(async_handoff_node(async_state))
        print(f"  async: {async_result}")
        assert async_result["execution_workflow_id"] == "async-workflow-id-xyz"
        print("✓ ASYNC execute node (auto-detected via asyncio.iscoroutinefunction) awaits the")
        print("  real trigger adapter correctly — this is the exact path vertex_deploy.py's")
        print("  async_query() uses via graph.ainvoke(), so the Temporal handoff never blocks")
        print("  the request.")
        print("  In production: build_compiled_graph(execute_trigger=make_execution_trigger(...))")
    except ImportError as e:
        print(f"(execution_workflow module not available here: {e})")

    print("\n" + "=" * 70)
    print("ATTEMPTING build_compiled_graph() — requires `pip install langgraph`")
    print("=" * 70)
    try:
        build_compiled_graph()
        print("✓ Graph compiled successfully.")
    except ImportError as e:
        print(f"✗ langgraph not installed in this environment ({e}).")
        print("  All node logic above was tested directly and passed.")
        print("  Install langgraph in your own environment to compile the real graph;")
        print("  the node functions and edges above are what it will wire together.")
