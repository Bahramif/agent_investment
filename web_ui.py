"""
web_ui.py
=========
Local web UI for the investment firm: pick an equity sector, get real
recommended names + weights back from the deployed InvestmentFirmAgent on
Vertex AI Agent Engine, enriched with live prices, buy/sell actions against
your real (paper) Alpaca positions, a naive expected-return projection, and
a chat box to ask follow-up questions. Same trust model as test_agent.py —
runs on YOUR machine, uses YOUR gcloud ADC + Alpaca credentials, never
exposes them to a browser or a public page.

Run it:
    source .venv/bin/activate
    python3 web_ui.py
Then open http://127.0.0.1:5050 in your browser.

HOW THE RECOMMENDATION WORKS: picking a sector builds a mandate that locks
100% of a notional portfolio into that one equities.<sector> bucket, then
calls the deployed agent's query() with no *_snapshots supplied. The graph's
own live Alpaca ingestion (data_ingestion.py, driven by the UniverseConfig
baked into vertex_deploy.py's set_up()) fetches real price/news data for the
WHOLE configured universe; the mandate lock filters the result down to just
the chosen sector's names, scored by the real analyst reasoner (Claude-
backed, falling back to heuristic scoring if Claude quota/model access
isn't available — see claude_reasoning.py). This is a deliberately
CONCENTRATED single-sector view (top picks within one sector), not a
diversified portfolio — default_limits/risk_limits here are intentionally
looser than the full-portfolio mandates used elsewhere in this project.

WHAT'S HONEST vs. WHAT'S A GUESS, so nothing here overstates itself:
  - Weights, rationale, risk notes, live price: real, from the actual
    deployed pipeline and a live Alpaca quote.
  - Buy/sell action: real math (execution_workflow.compute_order_deltas)
    against your REAL current Alpaca paper positions — but nothing here
    actually places an order; it only tells you what an order WOULD be.
  - "Expected return": a NAIVE extrapolation of each name's trailing
    1-month momentum, weighted by its recommended allocation. This is not
    a forecast, a backtest, or a guarantee — trailing momentum continuing
    is one of the weakest assumptions in finance. Labeled as such in the UI.
  - Chat answers: a real Claude call, given the current recommendation as
    context — but Claude can still be wrong, and this isn't investment advice.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from flask import Flask, render_template_string, request, Response, stream_with_context, redirect

import vertexai
from vertexai.preview import reasoning_engines

from alpaca_clients import apply_default_timeout

PROJECT = os.environ.get("MERIDIAN_PROJECT", "meridian-508414")
LOCATION = os.environ.get("MERIDIAN_LOCATION", "us-central1")
# Update this if you redeploy — vertex_deploy.py's build_and_deploy() prints
# the current resource name every time it runs.
RESOURCE_NAME = os.environ.get(
    "MERIDIAN_RESOURCE_NAME",
    "projects/497590571361/locations/us-central1/reasoningEngines/3593236435164987392",
)

# SECTORS/UNIVERSE come from universe.py — the SAME file vertex_deploy.py
# bundles into the deployed agent, so there's exactly one candidate list to
# keep accurate rather than two copies that can drift apart.
from universe import SECTORS, EQUITIES as UNIVERSE
import rulebook

def _max_per_name_for(top_n: int) -> float:
    """Per-name allocation cap, SCALED to whatever "max # of stocks" the
    user actually asked for.

    BUG FIX, confirmed live with real market data: a FIXED cap here
    (previously 0.50, regardless of top_n) meant the "max # of stocks"
    setting barely mattered past 2 names — the allocator
    routinely concentrated the whole budget onto the top 2 convictions
    (50% each, hitting the cap) and funded nothing else, whether top_n was
    set to 6, 8, or 12. This wasn't a hardcoded "always 3 stocks" anywhere
    in the code, but a fixed 50% ceiling has the same practical effect: at
    most 2 names can ever hit a 50% cap before the budget is spent, so
    top_n above ~3 was doing nothing.

    Scaling the cap down as top_n grows lets the diversification setting
    actually diversify. Scaling it UP for small top_n keeps the allocator
    ABLE to deploy the full budget without stranding it (a top_n=1 mandate
    needs a 100% cap; the smallest real sector in universe.py has 16 names,
    needing at least 1/16=6.25% of headroom per name to avoid permanently
    stranding budget there and forcing a guaranteed "total book != 100%"
    rejection — see firm_graph._deterministic_risk_check). 2.0/top_n gives
    roughly 2x headroom over the bare equal-weight floor across the whole
    practical range, with a 10% floor so a very large top_n never forces
    near-equal-weighting regardless of conviction.

    MUST stay in lockstep with whatever risk_limits.max_single_name the
    risk-check node is given (_build_recommend_state below) — if the
    allocator's own cap can legitimately exceed the risk ceiling, a
    perfectly valid, intentional allocation gets vetoed by risk_check_node
    and the recommendation silently fails after burning its 3 retries (no
    re-scoring happens between them, so an identical mandate/signals just
    produces the identical rejection three times — see
    firm_graph.risk_check_node's own bug-fix comment)."""
    cap = min(1.0, max(0.10, 2.0 / max(top_n, 1)))
    # Never propose what the pre-trade reviewer would block (rec_evaluator hard
    # check: > max_single_weight in one name when the book holds >=
    # min_names_for_cap names). Without this a 3-stock book got a 67% cap and
    # its biggest buy was vetoed every time. Still >= an equal share (1/n), so
    # the budget can always be fully placed.
    try:
        from rec_evaluator import load_config
        ev = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandate.yaml"))
        limit, min_names = ev.max_single_weight, ev.min_names_for_cap
    except Exception:
        limit, min_names = 0.40, 3
    if top_n >= min_names:
        cap = max(1.0 / top_n, min(cap, limit))
    return cap
# With 223 real candidates, the heuristic scorer's convictions cluster too
# closely together to differentiate meaningfully — without a cap, water_fill
# spreads the budget near-equally across 50-70+ names instead of
# concentrating on the agents' actual top picks. This is a GLOBAL cap across
# every selected sector combined (confirmed as the intended behavior — see
# mandate_allocator._alloc_children's global top-N cut), not per sector.
TOP_N_TOTAL = 12
DEFAULT_CAPITAL = 10_000.0
MOMENTUM_LOOKBACK_DAYS = 21   # matches data_ingestion.py's own default
REBALANCE_OPTIONS = {"5": "Weekly", "21": "Monthly", "63": "Quarterly"}
SYMBOL_TO_SECTOR = {sym: path.split(".")[-1] for sym, path, _ in UNIVERSE}

app = Flask(__name__)
vertexai.init(project=PROJECT, location=LOCATION)
_remote_agent = reasoning_engines.ReasoningEngine(RESOURCE_NAME)


def _alpaca_credentials() -> tuple[str, str]:
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    api_key = client.access_secret_version(
        name=f"projects/{PROJECT}/secrets/alpaca-api-key/versions/latest"
    ).payload.data.decode("utf-8")
    secret_key = client.access_secret_version(
        name=f"projects/{PROJECT}/secrets/alpaca-secret-key/versions/latest"
    ).payload.data.decode("utf-8")
    return api_key, secret_key


_ALPACA_API_KEY, _ALPACA_SECRET_KEY = _alpaca_credentials()

_ANTHROPIC_KEY: Optional[str] = None
_ANTHROPIC_KEY_CHECKED = -1e9


def _anthropic_api_key() -> Optional[str]:
    """Anthropic API key from Secret Manager (`anthropic-api-key`), or None.
    A found key is cached; a missing one is re-checked at most every 5
    minutes, so adding the secret takes effect without a redeploy."""
    import time
    global _ANTHROPIC_KEY, _ANTHROPIC_KEY_CHECKED
    if _ANTHROPIC_KEY or time.monotonic() - _ANTHROPIC_KEY_CHECKED < 300:
        return _ANTHROPIC_KEY
    _ANTHROPIC_KEY_CHECKED = time.monotonic()
    try:
        from google.cloud import secretmanager
        _ANTHROPIC_KEY = secretmanager.SecretManagerServiceClient().access_secret_version(
            name=f"projects/{PROJECT}/secrets/anthropic-api-key/versions/latest"
        ).payload.data.decode("utf-8").strip() or None
    except Exception:
        _ANTHROPIC_KEY = None
    return _ANTHROPIC_KEY


def _claude_client():
    """The ONE place this app builds its AI client. Gemini on Vertex by
    default (llm.GeminiClaudeShim — same messages.create() surface, thinking
    on). With MERIDIAN_LLM_PROVIDER=anthropic: Anthropic's own API when an
    `anthropic-api-key` secret exists, else Claude on Vertex AI."""
    import llm
    if llm.provider() == "gemini":
        return llm.GeminiClaudeShim(project=PROJECT)
    key = _anthropic_api_key()
    if key:
        from anthropic import Anthropic
        return Anthropic(api_key=key)
    from anthropic import AnthropicVertex
    return AnthropicVertex(project_id=PROJECT, region="global")

# In-memory cache of the last recommendation, so /chat has context without
# needing a real session store — fine for a single-user local tool.
_LAST_CONTEXT: Dict[str, Any] = {}


def build_mandate(selected_sectors: List[str], top_n: int = TOP_N_TOTAL, context: Optional[Dict[str, Any]] = None) -> dict:
    """Locks OUT every non-selected sector at 0% (so its names never enter
    the allocator's pool) and leaves every SELECTED sector fully OPEN. Open
    equity space with no locked children is exactly what
    mandate_allocator._alloc_children's theme-tilt branch was built for —
    it groups open-pool signals by theme and splits the budget by
    strategist.theme_budgets() when real theme_tilts are available (AI-
    tilted by conviction), falling back to pro-rata-by-name-count only if
    tilts are missing entirely. This is real allocator code, not a special
    case written for this UI.

    `context` is what the rulebook needs to know about an existing portfolio:
    {"held": [...], "blocked": [...]} (rulebook.select_names)."""
    excluded = [s for s in SECTORS if s not in selected_sectors]
    mandate: Dict[str, Any] = {
        "total": 1.0,
        "default_limits": {"max_per_name": _max_per_name_for(top_n), "min_names": 1, "top_n": top_n,
                           **{k: list(v) for k, v in (context or {}).items() if v}},
        "buckets": {"equities": {"set_by": "user", "weight": 1.0}},
    }
    if excluded:
        mandate["buckets"]["equities"]["children"] = {
            s: {"set_by": "user", "weight": 0.0} for s in excluded
        }
    return mandate


def _call_with_timeout(fn, *args, timeout: float = 300.0, **kwargs):
    """CRITICAL, confirmed by reading vertexai's own source
    (_wrap_query_operation's _method calls
    self.execution_api_client.query_reasoning_engine(request=...) with NO
    timeout kwarg at all) — the dynamically-attached query()/get_final_state()
    methods on a ReasoningEngine handle have no built-in timeout whatsoever.
    Confirmed live: a stuck connection there hangs FOREVER — same root-cause
    class as the Alpaca timeout bug fixed earlier (alpaca_clients.
    apply_default_timeout), just in a client we can't mount a requests
    adapter on (it's a gapic/grpc-backed client, not plain requests). Runs
    the call in a worker thread and enforces a hard wall-clock ceiling
    instead, since there's no cooperative way to cancel it from here."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    # Not a `with` block: leaving one waits for the worker, so a hung call would
    # still hold this request after its time was up.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        raise TimeoutError(f"{getattr(fn, '__name__', fn)} did not respond within {timeout:.0f}s")
    finally:
        pool.shutdown(wait=False)


def _strategist_memory() -> tuple:
    """(saved sector views, days since they were saved). Starting from nothing
    only costs an extra strategic review, so a store failure isn't an error."""
    from datetime import datetime, timezone
    try:
        import portfolio_store
        saved = portfolio_store.get_strategist_memory()
        memory = saved.get("theme_memory") or {}
        at = datetime.fromisoformat(saved["saved_at"]) if saved.get("saved_at") else None
        elapsed = (datetime.now(timezone.utc) - at).total_seconds() / 86400.0 if at else 0.0
        return memory, max(0.0, elapsed)
    except Exception as e:
        print(f"[web_ui] strategist memory unavailable ({e}): starting fresh")
        return {}, 0.0


def _finish_recommendation(result: dict) -> dict:
    """The agent's answer, ready for this app: its sector views saved for next
    time, and CASH (what the rulebook's limits left uninvested) taken out of the
    weights, since it isn't a stock to price or order."""
    memory = result.get("theme_memory") if isinstance(result, dict) else None
    if memory and isinstance(memory, dict):
        try:
            import portfolio_store
            portfolio_store.save_strategist_memory(memory)
        except Exception as e:
            print(f"[web_ui] could not save strategist memory: {e}")
    weights = result.get("final_weights") if isinstance(result, dict) else None
    if weights and rulebook.CASH in weights:
        result = {**result, "final_weights": {n: w for n, w in weights.items() if n != rulebook.CASH}}
    return result


def _build_recommend_state(selected_sectors: List[str], top_n: int, context: Optional[Dict[str, Any]] = None) -> tuple:
    """Shared by recommend() and the SSE streaming route below so the
    mandate/state-building logic (and the thread_id convention) lives in
    exactly one place. Returns (thread_id, state)."""
    theme_memory, elapsed_days = _strategist_memory()
    state = {
        "mandate": build_mandate(selected_sectors, top_n, context),
        "tilt_caps": {"strategic_cap": 0.15, "tactical_cap": 0.06, "total_cap": 0.15},
        "theme_memory": theme_memory,
        "elapsed_days": elapsed_days,
        "material_threshold": 0.03,
        # +0.05 buffer above the allocator's own cap (same relationship the
        # old fixed 0.50/0.55 pair had) so ordinary floating-point solver
        # slack never trips this ceiling on a legitimately at-cap weight.
        "risk_limits": {"max_single_name": min(1.0, _max_per_name_for(top_n) + 0.05)},
        "prior_theme_budgets": {},
    }
    thread_id = f"web-ui-{'-'.join(selected_sectors)}-{uuid.uuid4().hex[:8]}"
    return thread_id, state


# Allows for the agent's own budgets: about 90s of data, then at most 120s of desk
# scoring and 90s of sector views.
RECOMMEND_TIMEOUT_S = 360.0


def recommend(selected_sectors: List[str], top_n: int = TOP_N_TOTAL, context: Optional[Dict[str, Any]] = None) -> dict:
    """Asks the deployed agent to run its real pipeline (live ingestion ->
    analyst scoring -> strategist tilts -> risk check) over the selected
    sectors, AI-tilting the split between them by conviction. top_n caps how
    many of each sector's highest-conviction names the allocator will
    actually fund — the user-facing "max number of stocks" setting."""
    thread_id, state = _build_recommend_state(selected_sectors, top_n, context)
    return _finish_recommendation(_call_with_timeout(_remote_agent.query, thread_id=thread_id, state=state,
                                                     timeout=RECOMMEND_TIMEOUT_S))


# Human-friendly labels for each real LangGraph node name, shown live while
# /recommend_stream drains stream_query() — every label corresponds to an
# ACTUAL node in firm_graph.py's compiled graph (see build_compiled_graph
# and vertex_deploy.py's ingestion_node wiring), so this is genuine
# execution progress, not a synthetic/fake progress bar.
NODE_LABELS = {
    "data_ingestion": "📡 Fetching live prices & news for the full universe...",
    "equities_analyst": "📊 Scoring equity instruments...",
    "strategist_reasoning": "🧠 Building macro strategic & tactical tilts...",
    "combine_tilts": "⚖️ Combining desk tilts into theme budgets...",
    "freshness_gate": "🕒 Checking data freshness...",
    "portfolio_construction": "🏗️ Constructing portfolio allocation...",
    "risk_check": "🛡️ Running risk checks...",
}
# forex_analyst/crypto_analyst are hidden from the live progress feed —
# UniverseConfig currently ships forex=[]/crypto=[] (see vertex_deploy.py's
# set_up()), so these nodes run over an empty candidate list and never
# actually contribute anything to a recommendation; showing "Scoring forex
# instruments..." for a desk with zero forex instruments configured was
# misleading, not just noisy.
HIDDEN_NODES = {"forex_analyst", "crypto_analyst"}


def _live_prices(symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    sc = apply_default_timeout(StockHistoricalDataClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY))
    trades = sc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbols))
    return {sym: float(t.price) for sym, t in trades.items()}


def _momentum(symbols: List[str]) -> Dict[str, float]:
    """Trailing N-day momentum per symbol — same calc data_ingestion.py uses
    internally, recomputed here independently so the UI can show an
    'expected return' without needing to smuggle it out of the deployed
    agent's internal state."""
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca_clients import AlpacaMarketDataClient
    from data_ingestion import compute_momentum
    market = AlpacaMarketDataClient(
        StockHistoricalDataClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY),   # timeout applied inside AlpacaMarketDataClient
        CryptoHistoricalDataClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY),
    )
    out = {}
    for sym in symbols:
        try:
            out[sym] = compute_momentum(market.get_price_history(sym, MOMENTUM_LOOKBACK_DAYS))
        except Exception:
            out[sym] = 0.0
    return out


def _current_positions(capital: float) -> Dict[str, float]:
    from alpaca.trading.client import TradingClient
    from alpaca_clients import AlpacaBrokerClient
    tc = TradingClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY, paper=True)   # timeout applied inside AlpacaBrokerClient
    return AlpacaBrokerClient(tc).get_current_positions(capital)


def enrich(result: dict, selected_sectors: List[str], capital: float) -> Dict[str, Any]:
    """Combines the deployed agent's recommendation with live prices, real
    current-position-relative buy/sell actions, and a naive expected-return
    projection. Returns a dict ready for the template, rows grouped by
    sector so a multi-sector pick reads as sections, not one flat list."""
    weights: Dict[str, float] = result.get("final_weights") or {}
    notes: Dict[str, Dict[str, str]] = result.get("book_notes") or {}
    tilts: Dict[str, Dict[str, Any]] = result.get("theme_tilts") or {}
    symbols = list(weights.keys())

    prices = _live_prices(symbols)
    momentum = _momentum(symbols)

    # Informational only — see note below. NOT used to compute action/shares:
    # confirmed that execute()'s "Launch as New Portfolio" ignores current
    # positions entirely and always buys fresh from weight x capital, so
    # sizing shown here must match THAT, not a rebalance-style delta.
    current_positions = {}
    try:
        current_positions = _current_positions(capital)
    except Exception as e:
        print(f"[web_ui] current positions lookup failed: {e}")

    rows = []
    for i, (name, w) in enumerate(sorted(weights.items(), key=lambda kv: -kv[1])):
        price = prices.get(name)
        note = notes.get(name, {})
        # BUG FIX, confirmed live: this used to show a rebalance-style
        # BUY/SELL/HOLD delta against your combined current Alpaca
        # positions (execution_workflow.compute_order_deltas) — but
        # execute()'s "Launch as New Portfolio" button does something
        # completely different: a fresh whole-share buy sized from THIS
        # capital and weight, ignoring current positions entirely. The
        # two could disagree (e.g. a card showing "SELL" or "HOLD" for a
        # name execute() would actually buy), which is exactly the
        # confusing mismatch reported. Sizing here now matches execute()
        # exactly, line for line, so what you see is what gets ordered.
        shares = math.floor((capital * w) / price) if price else None
        if not shares:
            action, action_notional, shares = "SKIP", 0.0, None   # too small for 1 whole share
        else:
            action, action_notional = "BUY", shares * price
        rows.append({
            "name": name,
            "sector": SYMBOL_TO_SECTOR.get(name, "other"),
            "weight": w,
            "price": price,
            "momentum_1m": momentum.get(name, 0.0),
            "action": action,
            "action_notional": action_notional,
            "shares": shares,
            "already_held": current_positions.get(name, 0.0) > 1e-9,
            "rationale": note.get("rationale", ""),
            "risk_note": note.get("risk_note", ""),
            # WHY this weight, not just why this name: see mandate_allocator._weight_rationale.
            "weight_rationale": note.get("weight_rationale", ""),
            # Key metrics/score block — the "why this equity was selected"
            # transparency panel. All from the analyst's original scoring
            # pass (Book.notes, see mandate_allocator.py), NOT recomputed
            # here, so what's shown matches what actually drove the pick.
            "conviction": note.get("conviction"),
            "rating": note.get("rating", ""),
            "expected_return": note.get("expected_return"),
            # >0 means rating/expected_return are REAL Wall Street analyst
            # consensus (fundamentals.get_analyst_forecast); 0 means no
            # coverage existed for this name and they're the heuristic/
            # Claude-generated fallback instead — the UI badges this
            # distinction rather than presenting both as equally real.
            "analyst_count": note.get("analyst_count", 0),
            "quality_score": note.get("quality_score"),
            "news_sentiment": note.get("news_sentiment"),
            "volatility": note.get("volatility"),
            # Assigned by weight-sorted position (rows is already sorted by
            # -weight above) so the biggest slice always gets the same color
            # regardless of how many names are in the mandate. Shared by the
            # pie chart slice and the card's swatch dot — same dict, one value.
            "color": PIE_COLORS[i % len(PIE_COLORS)],
        })

    # Group into sections per selected sector, each with its own subtotal
    # weight and (if the strategist produced one) net tilt — this is the
    # "why did AI/Tech get more than Energy" narrative made visible.
    sections = []
    for s in selected_sectors:
        sector_rows = [r for r in rows if r["sector"] == s]
        tilt = tilts.get(s)
        sections.append({
            "sector": s,
            "label": SECTORS.get(s, s),
            "rows": sector_rows,
            "total_weight": sum(r["weight"] for r in sector_rows),
            "net_tilt": (tilt.get("net_tilt") if isinstance(tilt, dict) else None),
        })

    expected_return = sum(w * momentum.get(name, 0.0) for name, w in weights.items())
    # Renamed from total_action_buy/total_action_sell: every row is now
    # either BUY or SKIP (see the fix above), so a buy/sell split no longer
    # means anything — what actually matters is how much of the capital
    # gets deployed vs left as residual cash from whole-share rounding.
    total_to_deploy = sum(r["action_notional"] for r in rows if r["action"] == "BUY")
    already_held_count = sum(1 for r in rows if r.get("already_held") and r["action"] == "BUY")

    context = {
        "selected_sectors": selected_sectors, "capital": capital,
        "sections": sections, "rows": rows,
        "pie_chart": svg_pie_chart(rows),
        "expected_return": expected_return,
        "total_to_deploy": total_to_deploy,
        "residual_cash": max(0.0, capital - total_to_deploy),
        "already_held_count": already_held_count,
        "status": result.get("status"), "risk_verdict": result.get("risk_verdict"),
        "risk_reasons": result.get("risk_reasons"),
        # Stable per-recommendation ID so re-clicking "Accept & Execute" (a
        # page reload, a double-click) reuses the SAME client_order_id per
        # symbol — AlpacaBrokerClient.submit_and_await() then finds the
        # existing order via _find_existing() instead of submitting a real
        # second one. Regenerated only when a NEW recommendation is fetched.
        "batch_id": uuid.uuid4().hex[:10],
    }
    _LAST_CONTEXT.clear()
    _LAST_CONTEXT.update(context)
    return context


def ask_claude(question: str, context: Dict[str, Any]) -> str:
    """Answers a free-form question about the current recommendation. Fresh
    client per call (cheap; avoids holding a long-lived client across
    requests) — built by _claude_client() (Anthropic API, or Vertex)."""

    if not context:
        return "Get a recommendation first — there's nothing to discuss yet."

    sector_labels = ", ".join(SECTORS.get(s, s) for s in context.get("selected_sectors", []))
    lines = [f"{r['name']} ({SECTORS.get(r['sector'], r['sector'])}): weight {r['weight']:.1%}, "
            f"action {r['action']}, 1-month momentum {r['momentum_1m']:+.1%}, "
            f"rationale: {r['rationale']} weight rationale: {r['weight_rationale']} risk: {r['risk_note']}"
            for r in context["rows"]]
    system_context = (
        f"You are answering a question about a JUST-GENERATED equity recommendation "
        f"across these sectors: {sector_labels}. "
        f"Status: {context['status']}, risk verdict: {context['risk_verdict']}. "
        f"Naive expected 1-month portfolio return (trailing momentum extrapolation, "
        f"NOT a forecast): {context['expected_return']:+.1%}.\n\n"
        + "\n".join(lines) +
        "\n\nAnswer the user's question directly and concisely, grounded only in "
        "the data above. If asked to predict something not supported by this data, "
        "say so plainly rather than inventing false precision. This is not "
        "investment advice and you should say so if the question implies otherwise."
    )
    try:
        client = _claude_client()
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=500,
            system=system_context,
            messages=[{"role": "user", "content": question}],
        )
        return "".join(b.text for b in response.content if b.type == "text")
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return ("Claude is temporarily unavailable — this Google Cloud project's Claude "
                    "quota hasn't been approved yet, so chat can't answer right now. The "
                    "recommendation itself still worked (it falls back to a heuristic "
                    "scorer when Claude is unavailable); ask again once the quota request "
                    "in the Cloud Console is approved.")
        return f"Something went wrong answering that: {msg}"


def svg_multi_line_chart(dates: List[str], series: List[Tuple[str, List[float], str]],
                         width: int = 680, height: int = 300) -> str:
    """N-series line chart (strategy / equal-weight benchmark / SPY, etc.)
    with a labeled Y-axis and an interactive hover/drag crosshair — same
    interaction model as svg_price_chart, generalized to multiple series
    sharing one axis. series: [(label, values, color), ...]; any series
    with no values is skipped (e.g. SPY data unavailable for the window)."""
    series = [(label, vals, color) for label, vals, color in series if vals]
    if not series:
        return "<p>No simulation data available.</p>"
    global _CHART_COUNTER
    _CHART_COUNTER += 1
    chart_id = f"linechart{_CHART_COUNTER}"

    pad_left, pad_right, pad_top, pad_bottom = 64, 14, 28, 26
    all_vals = [v for _, vals, _ in series for v in vals]
    lo, hi = min(all_vals), max(all_vals)
    span = (hi - lo) or 1.0
    lo_padded, hi_padded = lo - span * 0.06, hi + span * 0.06
    span_padded = hi_padded - lo_padded
    n = max(len(vals) for _, vals, _ in series)

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def x_at(i: int) -> float:
        return pad_left + i * plot_w / max(n - 1, 1)

    def y_at(v: float) -> float:
        return pad_top + plot_h - (v - lo_padded) / span_padded * plot_h

    def points(vals: List[float]) -> str:
        return " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(vals))

    n_ticks = 4
    y_ticks = []
    for k in range(n_ticks + 1):
        val = lo_padded + span_padded * k / n_ticks
        y = y_at(val)
        y_ticks.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width-pad_right}" y2="{y:.1f}" '
            f'stroke="#eee"/><text x="{pad_left-8}" y="{y+3:.1f}" font-size="10" fill="#999" '
            f'text-anchor="end">${val:,.0f}</text>'
        )
    y_ticks_svg = "".join(y_ticks)

    lines_svg = "".join(
        f'<polyline points="{points(vals)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        for _, vals, color in series
    )
    legend_svg = "".join(
        f'<text x="{pad_left + i*150}" y="16" font-size="12" fill="{color}">■ {label}</text>'
        for i, (label, _, color) in enumerate(series)
    )
    dots_svg = "".join(
        f'<circle id="{chart_id}-dot{i}" r="3.5" fill="{color}" stroke="#fff" stroke-width="1.5" style="display:none;"/>'
        for i, (_, _, color) in enumerate(series)
    )

    return f"""
    <div style="position:relative;">
    <svg id="{chart_id}" viewBox="0 0 {width} {height}" style="width:100%;height:auto;background:#fafbff;
        border-radius:10px;border:1px solid #eee;cursor:crosshair;touch-action:none;display:block;">
      {y_ticks_svg}
      {lines_svg}
      {legend_svg}
      <text x="{pad_left}" y="{height-6}" font-size="11" fill="#999">{dates[0] if dates else ''}</text>
      <text x="{width-pad_right}" y="{height-6}" font-size="11" fill="#999" text-anchor="end">{dates[-1] if dates else ''}</text>
      <line id="{chart_id}-cross" x1="0" y1="{pad_top}" x2="0" y2="{height-pad_bottom}"
            stroke="#888" stroke-width="1" stroke-dasharray="3,3" style="display:none;"/>
      {dots_svg}
    </svg>
    <div id="{chart_id}-tip" style="position:absolute;top:6px;right:6px;background:#0d1117;color:#fff;
        padding:6px 12px;border-radius:6px;font-size:0.76rem;font-weight:600;display:none;
        pointer-events:none;white-space:nowrap;line-height:1.5;"></div>
    </div>
    <script>
    (function() {{
      var svg = document.getElementById("{chart_id}");
      var cross = document.getElementById("{chart_id}-cross");
      var tip = document.getElementById("{chart_id}-tip");
      var dates = {json.dumps(dates)};
      var series = {json.dumps([{"label": label, "values": vals, "color": color} for label, vals, color in series])};
      var padLeft = {pad_left}, plotW = {plot_w}, padTop = {pad_top}, plotH = {plot_h};
      var lo = {lo_padded}, span = {span_padded}, n = {n};
      var dots = series.map(function(_, i) {{ return document.getElementById("{chart_id}-dot" + i); }});

      function update(clientX) {{
        var rect = svg.getBoundingClientRect();
        var x = (clientX - rect.left) * ({width} / rect.width);
        var frac = Math.max(0, Math.min(1, (x - padLeft) / plotW));
        var idx = Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
        var px = padLeft + idx * plotW / Math.max(n - 1, 1);
        cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.style.display = "block";
        var lines = [dates[idx] || ""];
        series.forEach(function(s, i) {{
          var v = s.values[idx];
          if (v === undefined) {{ dots[i].style.display = "none"; return; }}
          var py = padTop + plotH - (v - lo) / span * plotH;
          dots[i].setAttribute("cx", px); dots[i].setAttribute("cy", py); dots[i].style.display = "block";
          lines.push(s.label + ": $" + v.toLocaleString(undefined, {{maximumFractionDigits: 0}}));
        }});
        tip.style.display = "block";
        tip.innerHTML = lines.join("<br>");
      }}
      function hide() {{
        cross.style.display = "none";
        dots.forEach(function(d) {{ d.style.display = "none"; }});
        tip.style.display = "none";
      }}
      svg.addEventListener("mousemove", function(e) {{ update(e.clientX); }});
      svg.addEventListener("mouseleave", hide);
      svg.addEventListener("touchstart", function(e) {{ update(e.touches[0].clientX); e.preventDefault(); }}, {{passive: false}});
      svg.addEventListener("touchmove", function(e) {{ update(e.touches[0].clientX); e.preventDefault(); }}, {{passive: false}});
      svg.addEventListener("touchend", hide);
    }})();
    </script>
    """


_CHART_COUNTER = 0


def svg_price_chart(dates: List[str], prices: List[float], width: int = 680, height: int = 260) -> str:
    """Single-series historical price/value chart with a labeled Y-axis and
    an interactive hover/drag crosshair — move the pointer (desktop) or
    drag a finger (mobile/touch) across the chart to read the exact
    date + value at that point. Plain SVG + vanilla JS, no charting
    library, consistent with the rest of this project's hand-rolled charts."""
    if not prices:
        return "<p>No price history available.</p>"
    global _CHART_COUNTER
    _CHART_COUNTER += 1
    chart_id = f"pricechart{_CHART_COUNTER}"

    pad_left, pad_right, pad_top, pad_bottom = 58, 14, 24, 26
    lo, hi = min(prices), max(prices)
    span = (hi - lo) or 1.0
    lo_padded, hi_padded = lo - span * 0.06, hi + span * 0.06
    span_padded = hi_padded - lo_padded
    n = len(prices)
    change = (prices[-1] - prices[0]) / prices[0] if prices[0] else 0.0
    color = "#00a340" if change >= 0 else "#d43f3f"

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def x_at(i: int) -> float:
        return pad_left + i * plot_w / max(n - 1, 1)

    def y_at(p: float) -> float:
        return pad_top + plot_h - (p - lo_padded) / span_padded * plot_h

    points = " ".join(f"{x_at(i):.1f},{y_at(p):.1f}" for i, p in enumerate(prices))

    n_ticks = 4
    y_ticks = []
    for k in range(n_ticks + 1):
        val = lo_padded + span_padded * k / n_ticks
        y = y_at(val)
        y_ticks.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width-pad_right}" y2="{y:.1f}" '
            f'stroke="#eee"/><text x="{pad_left-8}" y="{y+3:.1f}" font-size="10" fill="#999" '
            f'text-anchor="end">${val:,.2f}</text>'
        )
    y_ticks_svg = "".join(y_ticks)

    return f"""
    <div style="position:relative;">
    <svg id="{chart_id}" viewBox="0 0 {width} {height}" style="width:100%;height:auto;background:#fafbff;
        border-radius:10px;border:1px solid #eee;cursor:crosshair;touch-action:none;display:block;">
      {y_ticks_svg}
      <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>
      <text x="{pad_left}" y="16" font-size="12" fill="{color}">{'+' if change >= 0 else ''}{change:.1%} over this period</text>
      <text x="{pad_left}" y="{height-6}" font-size="11" fill="#999">{dates[0] if dates else ''}</text>
      <text x="{width-pad_right}" y="{height-6}" font-size="11" fill="#999" text-anchor="end">{dates[-1] if dates else ''}</text>
      <line id="{chart_id}-cross" x1="0" y1="{pad_top}" x2="0" y2="{height-pad_bottom}"
            stroke="#888" stroke-width="1" stroke-dasharray="3,3" style="display:none;"/>
      <circle id="{chart_id}-dot" r="4" fill="{color}" stroke="#fff" stroke-width="1.5" style="display:none;"/>
    </svg>
    <div id="{chart_id}-tip" style="position:absolute;top:6px;right:6px;background:#0d1117;color:#fff;
        padding:4px 10px;border-radius:6px;font-size:0.78rem;font-weight:600;display:none;
        pointer-events:none;white-space:nowrap;"></div>
    </div>
    <script>
    (function() {{
      var svg = document.getElementById("{chart_id}");
      var cross = document.getElementById("{chart_id}-cross");
      var dot = document.getElementById("{chart_id}-dot");
      var tip = document.getElementById("{chart_id}-tip");
      var dates = {json.dumps(dates)};
      var prices = {json.dumps(prices)};
      var padLeft = {pad_left}, plotW = {plot_w}, padTop = {pad_top}, plotH = {plot_h};
      var lo = {lo_padded}, span = {span_padded};

      function update(clientX) {{
        var rect = svg.getBoundingClientRect();
        var x = (clientX - rect.left) * ({width} / rect.width);
        var frac = Math.max(0, Math.min(1, (x - padLeft) / plotW));
        var idx = Math.max(0, Math.min(prices.length - 1, Math.round(frac * (prices.length - 1))));
        var px = padLeft + idx * plotW / Math.max(prices.length - 1, 1);
        var py = padTop + plotH - (prices[idx] - lo) / span * plotH;
        cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.style.display = "block";
        dot.setAttribute("cx", px); dot.setAttribute("cy", py); dot.style.display = "block";
        tip.style.display = "block";
        tip.textContent = dates[idx] + "  $" + prices[idx].toFixed(2);
      }}
      function hide() {{ cross.style.display = "none"; dot.style.display = "none"; tip.style.display = "none"; }}

      svg.addEventListener("mousemove", function(e) {{ update(e.clientX); }});
      svg.addEventListener("mouseleave", hide);
      svg.addEventListener("touchstart", function(e) {{ update(e.touches[0].clientX); e.preventDefault(); }}, {{passive: false}});
      svg.addEventListener("touchmove", function(e) {{ update(e.touches[0].clientX); e.preventDefault(); }}, {{passive: false}});
      svg.addEventListener("touchend", hide);
    }})();
    </script>
    """


# Distinct, reasonably colorblind-friendly categorical palette — cycles if
# there are more rows than colors. Assigned once in enrich() and carried on
# each row dict (row["color"]) so the pie chart and each card's swatch use
# the EXACT same color per stock, not just a matching index computed twice.
PIE_COLORS = [
    "#4a6cf7", "#00a340", "#d97706", "#d43f3f", "#8b5cf6", "#06b6d4",
    "#ec4899", "#84cc16", "#f59e0b", "#6366f1", "#14b8a6", "#f97316",
    "#a855f7", "#22c55e",
]


def _polar(cx: float, cy: float, r: float, angle: float) -> tuple:
    return cx + r * math.cos(angle), cy + r * math.sin(angle)


def _donut_slice_path(cx: float, cy: float, r_outer: float, r_inner: float,
                      start_angle: float, end_angle: float) -> str:
    x1o, y1o = _polar(cx, cy, r_outer, start_angle)
    x2o, y2o = _polar(cx, cy, r_outer, end_angle)
    x1i, y1i = _polar(cx, cy, r_inner, end_angle)
    x2i, y2i = _polar(cx, cy, r_inner, start_angle)
    large_arc = 1 if (end_angle - start_angle) > math.pi else 0
    return (f"M {x1o:.2f},{y1o:.2f} A {r_outer:.2f},{r_outer:.2f} 0 {large_arc} 1 {x2o:.2f},{y2o:.2f} "
           f"L {x1i:.2f},{y1i:.2f} A {r_inner:.2f},{r_inner:.2f} 0 {large_arc} 0 {x2i:.2f},{y2i:.2f} Z")


def svg_pie_chart(rows: List[Dict[str, Any]], size: int = 280) -> str:
    """Interactive donut chart of the recommended portfolio — click a slice
    to jump to that stock's card below (see the inline <script>'s
    selectStock()). Each row must already carry a 'color' field (set in
    enrich()) so a slice and its card's swatch always match exactly."""
    if not rows:
        return ""
    cx = cy = size / 2
    r_outer, r_inner = size / 2 - 4, (size / 2 - 4) * 0.55
    total = sum(r["weight"] for r in rows) or 1.0
    angle = -math.pi / 2   # start at 12 o'clock
    slices = []
    for r in rows:
        frac = r["weight"] / total
        end_angle = angle + frac * 2 * math.pi
        path = _donut_slice_path(cx, cy, r_outer, r_inner, angle, end_angle)
        slices.append(
            f'<path d="{path}" fill="{r["color"]}" class="pie-slice" '
            f'onclick="selectStock(\'{r["name"]}\')" stroke="#fff" stroke-width="1.5">'
            f'<title>{r["name"]}: {r["weight"]:.1%}</title></path>'
        )
        angle = end_angle
    return f"""
    <svg viewBox="0 0 {size} {size}" style="width:100%;max-width:280px;height:auto;display:block;margin:0 auto;">
      {''.join(slices)}
      <text x="{cx}" y="{cy-6}" text-anchor="middle" font-size="13" fill="#888">Portfolio</text>
      <text x="{cx}" y="{cy+14}" text-anchor="middle" font-size="18" font-weight="700" fill="#0d1117">{len(rows)} names</text>
    </svg>
    """


# Shared top tab bar, injected into every page's <body> right after it opens
# (see the `.replace("<body>", ...)` calls right after each page template is
# defined below) — one definition, one look, everywhere, instead of a
# different ad-hoc hyperlink footer per page. `active_page` (passed by every
# route) highlights the current section; pages that aren't one of the four
# primary tabs (stock detail, execution results) render it with nothing
# highlighted, which correctly signals "you're one level deeper than these."
TOP_NAV = """
<style>
  .topnav { display: flex; gap: 4px; padding: 4px 0 14px; margin-bottom: 10px;
           border-bottom: 1px solid #eee; overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .topnav a { display: inline-flex; align-items: center; gap: 6px; padding: 8px 14px;
             border-radius: 10px; text-decoration: none; color: #555; font-weight: 600;
             font-size: 0.86rem; white-space: nowrap; flex: none; }
  .topnav a:hover { background: #f6f8ff; }
  .topnav a.active { background: #0d1117; color: #fff; }
</style>
<nav class="topnav">
  <a href="/" class="{{ 'active' if active_page == 'home' else '' }}">🏠 Home</a>
  <a href="/portfolios" class="{{ 'active' if active_page == 'portfolios' else '' }}">📊 Portfolio</a>
  <a href="/daytrade" class="{{ 'active' if active_page == 'daytrade' else '' }}">⚡ Day Trading</a>
  <a href="/simulate" class="{{ 'active' if active_page == 'simulate' else '' }}">📈 Simulate</a>
  <a href="/evaluation" class="{{ 'active' if active_page == 'evaluation' else '' }}">🧪 Evaluation</a>
  <a href="/about" class="{{ 'active' if active_page == 'about' else '' }}">ℹ️ How This Works</a>
</nav>
"""


_RECONCILE_LAST: Dict[str, float] = {}


def _reconcile_portfolio_orders(portfolio_id: str, min_interval: float = 0.0) -> bool:
    """Fold any orders that were still IN FLIGHT when this portfolio's ledger
    was written (a launch or recalibration placed after hours, or slower than
    the few-second poll window) into its holdings and cash, now that Alpaca
    knows how they ended. Without this the Portfolio page shows the
    PRE-recalibration holdings until the end of time, and the next
    recalibration sizes its trades off that stale ledger. See
    portfolio_store.resolve_pending for the accounting rules.

    Returns True if the ledger changed (caller should reload the document).
    NEVER raises — a broker hiccup must leave the page working, showing the
    ledger as it was. `min_interval` throttles repeat checks per portfolio for
    the callers that run on every page load (the dropdown/list)."""
    import time
    import portfolio_store
    now = time.monotonic()
    if min_interval and now - _RECONCILE_LAST.get(portfolio_id, -1e9) < min_interval:
        return False
    _RECONCILE_LAST[portfolio_id] = now
    try:
        from alpaca.trading.client import TradingClient
        tc = apply_default_timeout(TradingClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY, paper=True))

        def fetch(client_order_id: str) -> Dict[str, Any]:
            o = tc.get_order_by_client_id(client_order_id)     # raises if not found -> stays pending
            v = lambda x: str(getattr(x, "value", x)).lower()
            return {"side": v(o.side), "status": v(o.status),
                    "filled_qty": float(o.filled_qty or 0), "filled_avg_price": float(o.filled_avg_price or 0)}

        applied = portfolio_store.reconcile_portfolio(portfolio_id, fetch)
        if applied:
            print(f"[web_ui] reconciled {len(applied)} in-flight order(s) into portfolio {portfolio_id}: "
                  f"{[(a['symbol'], a['side'], a['filled_qty']) for a in applied]}")
        return bool(applied)
    except Exception as e:
        print(f"[web_ui] could not reconcile pending orders for {portfolio_id}: {e}")
        return False


def _all_portfolios_enriched() -> List[Dict[str, Any]]:
    """Every tracked portfolio, mark-to-market against REAL current Alpaca
    prices (_live_prices -> Alpaca's latest-trade endpoint) — the single
    source both the compare list and the selector dropdown draw their
    dollar values from, so a value shown in the dropdown always matches
    what the page you land on shows too."""
    import portfolio_store
    raw = portfolio_store.list_portfolios()
    for i, row in enumerate(raw):
        # Same reconciliation the detail page does, so the dropdown/list never
        # disagrees with it. Throttled: this runs on nearly every page load.
        if row.get("status") == "open" and row.get("pending_order_ids") \
                and _reconcile_portfolio_orders(row["id"], min_interval=30.0):
            raw[i] = portfolio_store.get_portfolio(row["id"]) or row
    all_symbols = sorted({sym for p in raw for sym in (p.get("holdings") or {})})
    live = _live_prices(all_symbols)
    return [_mark_to_market(p, live) for p in raw]


ABOUT_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian — How This Works</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 820px;
        margin: 0 auto; padding: 32px 16px 64px; color: #0d1117; line-height: 1.5; background: #fff; }
  h1 { font-size: 1.35rem; font-weight: 700; margin-bottom: 4px; }
  h2 { font-size: 1.15rem; margin: 36px 0 6px; color: #0d1117; padding-top: 10px; border-top: 1px solid #eee; }
  h2:first-of-type { border-top: none; }
  h3 { font-size: 0.95rem; margin: 20px 0 8px; color: #222; }
  .about-body p { margin: 0 0 12px; font-size: 0.9rem; color: #333; }
  .about-body ul, .about-body ol { margin: 0 0 12px; padding-left: 20px; font-size: 0.88rem; color: #333; line-height: 1.6; }
  .about-body li { margin-bottom: 4px; }
  .lede { font-size: 0.98rem; color: #222; }
  .stage-flow { background: #f6f8ff; border: 1px solid #e8ecfb; border-radius: 12px; padding: 16px 18px;
               margin: 12px 0 20px; font-size: 0.85rem; color: #333; }
  .stage-flow ol { counter-reset: stage; list-style: none; padding-left: 0; margin: 0; }
  .stage-flow li { counter-increment: stage; position: relative; padding: 6px 0 6px 34px; }
  .stage-flow li::before { content: counter(stage); position: absolute; left: 0; top: 4px;
    background: #4a6cf7; color: #fff; width: 22px; height: 22px; border-radius: 50%;
    font-size: 0.72rem; font-weight: 700; display: flex; align-items: center; justify-content: center; }
  .stage-flow b { color: #0d1117; }
  .formula { background: #0d1117; color: #d6e0ff; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.78rem; padding: 10px 14px; border-radius: 8px; margin: 8px 0 14px; overflow-x: auto; }
  .about-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 4px 0 18px; }
  .about-grid ul { margin: 0; }
  .callout { background: #fff8e6; border-radius: 10px; padding: 10px 14px; font-size: 0.82rem;
            color: #4a3c00; margin: 10px 0 16px; }
  .honest { background: #fee; border-radius: 10px; padding: 10px 14px; font-size: 0.82rem;
           color: #7a2020; margin: 10px 0 16px; }
  .toc { font-size: 0.85rem; columns: 2; column-gap: 24px; margin: 10px 0 24px; }
  .toc a { color: #4a6cf7; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .badge { display: inline-block; font-size: 0.7rem; font-weight: 700; padding: 1px 8px; border-radius: 8px;
          background: #f0f0f5; color: #555; margin-left: 6px; }
  @media (max-width: 600px) {
    body { padding: 20px 14px 56px; }
    .about-grid { grid-template-columns: 1fr; gap: 14px; }
    .toc { columns: 1; }
  }
</style>
</head>
<body>
  <h1>How This Works</h1>
  <div class="about-body">
    <p class="lede">Meridian is a multi-agent AI pipeline that turns a sector pick and a dollar
      amount into a real, executable equity allocation — selecting which stocks to hold AND how
      much of the budget each one gets — then places real orders in a real (paper) brokerage
      account and tracks the result against the S&amp;P 500 over time. Every request runs the
      full pipeline fresh against live data; nothing shown is cached, backtested, or
      pre-computed unless explicitly labeled as historical.</p>

    <div class="toc">
      <a href="#highlevel">1. High-level architecture</a>
      <a href="#topology">Agent topology (raw graph)</a>
      <a href="#inputs">2. What you control</a>
      <a href="#ingestion">3. Data ingestion</a>
      <a href="#scoring">4. Per-stock scoring</a>
      <a href="#strategist">5. Sector-level strategist</a>
      <a href="#selection">6. Stock selection</a>
      <a href="#weights">7. Weight optimization</a>
      <a href="#risk">8. Risk check</a>
      <a href="#execution">9. Execution</a>
      <a href="#tracking">10. Ongoing tracking</a>
      <a href="#daytrading">11. Day trading</a>
    </div>

    <h2 id="highlevel">1. High-level architecture</h2>
    <p>Meridian is built as a LangGraph state machine — a directed graph of small, independent
      steps ("nodes") that each read some state and hand back an update, rather than one large
      function. Every node's logic is deterministic and independently testable; the two that
      call an LLM (scoring and sector-level strategy) can be swapped between a real Claude call
      and a transparent arithmetic fallback without changing anything else in the graph.</p>
    <div class="stage-flow">
      <ol>
        <li><b>Data ingestion</b> — a year of real prices, news, fundamentals, and analyst
          estimates for every candidate stock</li>
        <li><b>Per-stock scoring</b> — each stock gets one score, ranked against its own sector</li>
        <li><b>Sector-level strategist</b> — a long-term + short-term view per sector, which
          shades how the budget splits across your selected sectors</li>
        <li><b>Stock selection</b> — the best eligible names up to your "max # of stocks",
          with current holdings kept while they stay near the top</li>
        <li><b>Sizing and limits</b> — inverse-volatility weights, then sub-industry and
          portfolio-beta limits; any excess stays cash</li>
        <li><b>Risk check</b> — a deterministic, non-negotiable veto on every limit</li>
        <li><b>Execution</b> (only on "Launch as New Portfolio") — real whole-share market
          orders placed on Alpaca's paper trading API</li>
        <li><b>Ongoing tracking</b> — an independent ledger reconstructs the portfolio's real
          performance over time and compares it to a real SPY buy-and-hold</li>
      </ol>
    </div>
    <p>Stages 1-3 fan out and run in parallel per asset class/sector, then converge before
      stage 4. If the risk check in stage 6 ever vetoes a proposal, the graph loops back to
      stage 5 and retries (capped at 3 attempts) rather than ever showing you something that
      breaks its own limits.</p>

    <h3 id="topology">Agent topology (the actual compiled graph)</h3>
    <p>The eight stages above are the user-facing summary. Underneath, this is the real
      node/edge wiring of the compiled LangGraph (see <code>firm_graph.py</code>'s
      <code>build_compiled_graph()</code>) — every box below is a real node, not a
      simplification. Blue is the path an ordinary run takes; amber, dashed lines are the two
      conditional branches that only trigger in specific circumstances.</p>
    <div style="background: #fff; border: 1px solid #e8ecfb; border-radius: 12px;
                padding: 16px 8px; margin: 12px 0 6px; overflow-x: auto;">
      <svg viewBox="0 0 600 800" style="width: 100%; max-width: 560px; display: block; margin: 0 auto;">
        <defs>
          <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#4a6cf7"/>
          </marker>
          <marker id="arrowAmber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#d97706"/>
          </marker>
        </defs>

        <line x1="160" y1="50"  x2="160" y2="68"  stroke="#4a6cf7" stroke-width="1.75" marker-end="url(#arrowMain)"/>
        <line x1="160" y1="112" x2="160" y2="142" stroke="#4a6cf7" stroke-width="1.75" marker-end="url(#arrowMain)"/>
        <line x1="160" y1="186" x2="160" y2="216" stroke="#4a6cf7" stroke-width="1.75" marker-end="url(#arrowMain)"/>
        <line x1="160" y1="260" x2="160" y2="290" stroke="#4a6cf7" stroke-width="1.75" marker-end="url(#arrowMain)"/>
        <line x1="160" y1="334" x2="160" y2="350" stroke="#4a6cf7" stroke-width="1.75" marker-end="url(#arrowMain)"/>
        <line x1="160" y1="434" x2="160" y2="484" stroke="#4a6cf7" stroke-width="1.75" marker-end="url(#arrowMain)"/>
        <text x="172" y="462" font-size="11" font-style="italic" fill="#666" font-family="-apple-system,sans-serif">clean</text>
        <line x1="160" y1="528" x2="160" y2="548" stroke="#4a6cf7" stroke-width="1.75" marker-end="url(#arrowMain)"/>
        <line x1="160" y1="632" x2="160" y2="682" stroke="#4a6cf7" stroke-width="1.75" marker-end="url(#arrowMain)"/>
        <text x="172" y="660" font-size="11" font-style="italic" fill="#666" font-family="-apple-system,sans-serif">approve</text>
        <line x1="160" y1="726" x2="160" y2="744" stroke="#4a6cf7" stroke-width="1.75" marker-end="url(#arrowMain)"/>

        <line x1="255" y1="392" x2="320" y2="392" stroke="#d97706" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrowAmber)"/>
        <text x="287" y="384" font-size="10.5" font-style="italic" fill="#92620a" text-anchor="middle" font-family="-apple-system,sans-serif">stale + material</text>

        <path d="M420,364 L420,238 L270,238" fill="none" stroke="#d97706" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrowAmber)"/>
        <text x="345" y="230" font-size="10.5" font-style="italic" fill="#92620a" text-anchor="middle" font-family="-apple-system,sans-serif">re-affirm / revise</text>

        <path d="M420,420 L420,498 L270,498" fill="none" stroke="#d97706" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrowAmber)"/>
        <text x="345" y="490" font-size="10.5" font-style="italic" fill="#92620a" text-anchor="middle" font-family="-apple-system,sans-serif">override</text>

        <path d="M255,590 L480,590 L480,514 L270,514" fill="none" stroke="#d97706" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrowAmber)"/>
        <text x="488" y="555" font-size="10.5" font-style="italic" fill="#92620a" font-family="-apple-system,sans-serif">reject &rarr;</text>
        <text x="488" y="568" font-size="10.5" font-style="italic" fill="#92620a" font-family="-apple-system,sans-serif">retry (cap 3)</text>

        <circle cx="160" cy="30" r="20" fill="#4a6cf7"/>
        <text x="160" y="34" text-anchor="middle" font-size="11.5" fill="#fff" font-weight="700" font-family="-apple-system,sans-serif">START</text>

        <rect x="50" y="68" width="220" height="44" rx="10" fill="#f6f8ff" stroke="#4a6cf7" stroke-width="1.5"/>
        <text x="160" y="94" text-anchor="middle" font-size="12" fill="#0d1117" font-family="ui-monospace,Menlo,monospace">data_ingestion</text>

        <rect x="50" y="142" width="220" height="44" rx="10" fill="#f6f8ff" stroke="#4a6cf7" stroke-width="1.5"/>
        <text x="160" y="168" text-anchor="middle" font-size="12" fill="#0d1117" font-family="ui-monospace,Menlo,monospace">equities_analyst</text>

        <rect x="50" y="216" width="220" height="44" rx="10" fill="#f6f8ff" stroke="#4a6cf7" stroke-width="1.5"/>
        <text x="160" y="242" text-anchor="middle" font-size="12" fill="#0d1117" font-family="ui-monospace,Menlo,monospace">strategist_reasoning</text>

        <rect x="50" y="290" width="220" height="44" rx="10" fill="#f6f8ff" stroke="#4a6cf7" stroke-width="1.5"/>
        <text x="160" y="316" text-anchor="middle" font-size="12" fill="#0d1117" font-family="ui-monospace,Menlo,monospace">combine_tilts</text>

        <polygon points="160,350 255,392 160,434 65,392" fill="#fff8e6" stroke="#d97706" stroke-width="1.5"/>
        <text x="160" y="396" text-anchor="middle" font-size="11.5" fill="#4a3c00" font-family="ui-monospace,Menlo,monospace">freshness_gate</text>

        <rect x="320" y="364" width="200" height="56" rx="10" fill="#fff8e6" stroke="#d97706" stroke-width="1.5" stroke-dasharray="4 3"/>
        <text x="420" y="388" text-anchor="middle" font-size="11" fill="#4a3c00" font-family="ui-monospace,Menlo,monospace">human_approval_</text>
        <text x="420" y="402" text-anchor="middle" font-size="11" fill="#4a3c00" font-family="ui-monospace,Menlo,monospace">freshness</text>

        <rect x="50" y="484" width="220" height="44" rx="10" fill="#f6f8ff" stroke="#4a6cf7" stroke-width="1.5"/>
        <text x="160" y="510" text-anchor="middle" font-size="12" fill="#0d1117" font-family="ui-monospace,Menlo,monospace">portfolio_construction</text>

        <polygon points="160,548 255,590 160,632 65,590" fill="#fff8e6" stroke="#d97706" stroke-width="1.5"/>
        <text x="160" y="594" text-anchor="middle" font-size="11.5" fill="#4a3c00" font-family="ui-monospace,Menlo,monospace">risk_check</text>

        <rect x="50" y="682" width="220" height="44" rx="10" fill="#f6f8ff" stroke="#4a6cf7" stroke-width="1.5"/>
        <text x="160" y="708" text-anchor="middle" font-size="12" fill="#0d1117" font-family="ui-monospace,Menlo,monospace">execute</text>

        <circle cx="160" cy="764" r="20" fill="#0d1117"/>
        <text x="160" y="768" text-anchor="middle" font-size="12" fill="#fff" font-weight="700" font-family="-apple-system,sans-serif">END</text>
      </svg>
      <div style="display: flex; gap: 20px; justify-content: center; margin-top: 4px;
                  font-size: 0.78rem; color: #555;">
        <span><svg width="20" height="10" style="vertical-align: middle;"><line x1="0" y1="5" x2="20" y2="5" stroke="#4a6cf7" stroke-width="2"/></svg> normal path</span>
        <span><svg width="20" height="10" style="vertical-align: middle;"><line x1="0" y1="5" x2="20" y2="5" stroke="#d97706" stroke-width="2" stroke-dasharray="4 3"/></svg> conditional / rare branch</span>
      </div>
    </div>
    <p style="font-size: 0.85rem; color: #333;">Mapped to the numbered stages: <code>data_ingestion</code>
      is stage 1; <code>equities_analyst</code> is stage 2 (per-stock scoring);
      <code>strategist_reasoning</code> + <code>combine_tilts</code> together are stage 3;
      <code>freshness_gate</code> is a deterministic (no LLM) check for a strategist view that's
      gone stale while still drawing material capital &mdash; on ordinary runs it finds nothing
      and routes straight through to <code>portfolio_construction</code> (stages 4-5); <code>risk_check</code>
      is stage 6; <code>execute</code> is stage 7. (The compiled graph also has <code>forex_analyst</code>
      and <code>crypto_analyst</code> nodes feeding into <code>strategist_reasoning</code> the same way
      <code>equities_analyst</code> does — left off the diagram above because this deployment's
      universe configures zero forex/crypto instruments, so they run over an empty list and never
      actually contribute a signal.)</p>
    <div class="honest">Honest limitation: <code>human_approval_freshness</code> only fires when
      the freshness gate finds a real, material violation, and when it does it pauses the graph
      on a genuine LangGraph <code>interrupt()</code>, expecting a human response of
      <code>re_affirm</code>, <code>revise</code>, or <code>override</code>. This web app has no
      code path that supplies or resumes that response &mdash; so on the rare run where this
      branch actually fires, the request would simply run until its 300-second timeout rather
      than surfacing a prompt to approve or override.</div>

    <h2 id="inputs">2. What you control</h2>
    <p>Four inputs on the Home tab shape everything downstream:</p>
    <ul>
      <li><b>Sector(s)</b> — which of the 5 sectors' candidate pools to consider (locks every
        other sector to 0% so its names never enter scoring at all)</li>
      <li><b>Capital</b> — the dollar amount the recommendation and, if you launch it, the real
        order sizing are based on</li>
      <li><b>Max # of stocks</b> — the ceiling on how many distinct names can be selected across
        your chosen sector(s) combined (not per sector) — this also directly controls how
        concentrated any one stock is allowed to get (see §7)</li>
      <li><b>Rebalance cadence</b> — optional; if set when you launch a portfolio, a scheduled
        job re-runs this entire pipeline against it automatically (see §10)</li>
    </ul>

    <h2 id="ingestion">3. Data ingestion</h2>
    <p>The candidate pool is {{ universe_size }} names across {{ sectors|length }} sectors. Each
      stock's sector comes from the data provider, not from a hand-kept list. For every candidate,
      these real inputs are fetched fresh:</p>
    <div class="about-grid">
      <div>
        <ul>
          <li><b>12-1 month momentum</b> — the return from 12 months ago to 1 month ago, from
            Alpaca price bars. The latest month is skipped because returns over about a month
            tend to reverse.</li>
          <li><b>Volatility and beta</b> — daily volatility and beta to SPY over up to a year</li>
          <li><b>News</b> — the last 7 days of real headlines from Alpaca News, read by the LLM</li>
        </ul>
      </div>
      <div>
        <ul>
          <li><b>Quality score</b> — real fundamentals from Yahoo Finance: profit margin (35%
            weight), return on equity (35%), debt/equity (15%), revenue growth (15%)</li>
          <li><b>Value</b> — forward earnings yield (earnings per share / price)</li>
          <li><b>Analyst forecast</b> — Wall Street consensus target and rating. Shown to you and
            to the LLM, but it never sizes a position: target upside rises automatically when a
            price falls.</li>
        </ul>
      </div>
    </div>
    <p>Each sector's read for the strategist in §5 comes from all its stocks.</p>
    <div class="callout">Prices come from Alpaca's free IEX feed, not the full consolidated
      SIP tape — quotes can differ by a few cents (occasionally more for thinly-traded names)
      from a source using the full tape.</div>

    <h2 id="scoring">4. Per-stock scoring</h2>
    <p>Every rule for long-term portfolios lives in one place (<code>rulebook.py</code>, set in the
      <code>rulebook:</code> block of <code>mandate.yaml</code>), used by recommendations, the news
      watch and the simulation alike. No rule names a stock. Each stock gets one 0-1 <b>score</b>,
      from where it ranks among the stocks in its own sector:</p>
    <div class="formula">score = 0.35&middot;momentum + 0.25&middot;quality + 0.20&middot;value + 0.20&middot;LLM read</div>
    <ul>
      <li><b>LLM read</b> — the model compares a sector's stocks and classifies their headlines.
        Its scores are converted to ranks within each call, since raw scores aren't comparable
        between calls. If the model is unavailable its weight goes to the other factors and a
        stock's news counts as unknown.</li>
      <li><b>No dip buying</b> — a recent fall earns no extra credit. The backtest didn't support it,
        so that setting is off.</li>
      <li><b>Who can't be held</b> — a stock with under a year of trading history, under $10M of
        average daily dollar volume, or news the model judges thesis-breaking. The same test for
        every stock.</li>
    </ul>

    <h2 id="strategist">5. Sector-level strategist</h2>
    <p>Separately from individual stock scoring, each sector gets a top-down view on two
      horizons:</p>
    <ul>
      <li><b>Strategic</b> — a slow-moving thesis, reviewed every 90 days (365-day decay half-life,
        expires if not refreshed within 180 days), up to &plusmn;15% tilt</li>
      <li><b>Tactical</b> — a fast-moving read, refreshed on every run (14-day half-life, expires
        after 21 days), up to &plusmn;6% tilt</li>
    </ul>
    <p>Views are saved between runs. The two combine into one net tilt per sector (capped at
      &plusmn;15%), which shades how your budget splits across multiple selected sectors, never
      all-or-nothing.</p>

    <h2 id="selection">6. Stock selection</h2>
    <p>Scores are ranked across your selected sectors, and the best names up to your "max # of
      stocks" are selected: picking 3 sectors at "max 8" gives you 8 names total, not 24.</p>
    <ul>
      <li><b>Keeping a stock</b> — a holding stays while it ranks in the top 2&times;N, so a stock
        slipping a few places doesn't trade.</li>
      <li><b>Buying back</b> — anything sold, for any reason, can't be rebought for 30 days.</li>
      <li><b>Earnings</b> — no new buys within 3 days before a company reports.</li>
      <li><b>Sub-industry</b> — selection spreads across sub-industries where it can, so the limit
        in §7 doesn't have to leave money in cash.</li>
    </ul>

    <h2 id="weights">7. Sizing and portfolio limits</h2>
    <p>Selected names are sized by <b>inverse volatility</b>: calmer stocks get more, so each
      contributes more evenly to risk. There is deliberately no return forecast and no covariance
      estimate here, because neither can be estimated reliably for a dozen stocks.</p>
    <p>The per-name cap scales with your "max # of stocks" (roughly
      <span style="font-family:ui-monospace,monospace;font-size:0.85em;">2 &divide; max_stocks</span>,
      between 10% and 30% once a book holds 3 or more). Then two portfolio limits apply: at most
      35% in one sub-industry, and a portfolio beta of at most 1.5. Any excess stays in cash.</p>

    <h2 id="risk">8. Risk check</h2>
    <p>Before anything is ever shown to you, a separate, deterministic (no LLM) check runs
      against the proposed book: per-stock cap, sub-industry limit, beta limit, and that the
      book adds up to 100% including cash. A violation vetoes the proposal. This check cannot be
      argued with or reasoned around by either scoring path.</p>

    <h2 id="execution">9. Execution</h2>
    <p>"Launch as New Portfolio" is the only action that places real orders. Each selected name
      gets a real whole-share market order on Alpaca's PAPER trading API (never real money),
      tagged with a stable idempotency key so retrying a failed page load never double-buys.
      Orders are polled for their real fill price and quantity — including correctly handling a
      genuine partial fill (fewer shares filled than requested) rather than assuming the full
      requested amount went through.</p>

    <h2 id="tracking">10. Ongoing tracking</h2>
    <p>Every launch creates an independently-tracked ledger — its own entry prices, its own
      cash — even though real fills all land in one shared Alpaca paper account (Alpaca's paper
      product doesn't support multiple separate accounts natively). Opening its page in the
      Portfolio tab:</p>
    <ul>
      <li>Reconstructs its REAL performance day-by-day (or hour-by-hour for anything under 2
        days old) from actual historical Alpaca prices for exactly what it holds, compared
        against a real SPY buy-and-hold funded with the same starting capital — recomputed
        fresh on every page load, never simulated</li>
      <li>Marks every holding to a live current quote, shown alongside its real entry price</li>
      <li>If you set a rebalance cadence at launch, a real Cloud Scheduler job checks daily during
        market hours for portfolios that are due, re-runs this entire pipeline (stages 1-6) sized
        against the portfolio's CURRENT mark-to-market value, and places real orders for the share
        delta between what it holds now and the new target, sells first. Changes under 0.5% of the
        portfolio don't trade.</li>
      <li>If you turn on the news watch, it checks daily whether a holding must be sold early, for
        one of two reasons. The <b>stop</b>: below what was paid AND behind its sector, since
        purchase, by 3 of its own monthly sigmas (about 35% for a 40%-volatility stock, 22% for a
        25% one), so it only catches severe breakdowns. Or <b>news</b> the model judges thesis-breaking, at most twice in 30 days. There is one
        verdict per stock per day, shared by every portfolio holding it. Only that stock is sold, and
        only a stop can go past the limit. The sale happens first; its slot is filled afterwards, or
        stays cash until the next recalibration if no replacement can be found.</li>
      <li>The news watch also reads each sector you hold as a whole: every stock's news and moves. If a
        sector moves unusually for it (2 of its own daily sigmas, up or down) and the model judges that a
        real change in its outlook, the whole portfolio is re-ranked by the same rules as a scheduled
        recalibration. This counts toward the same limit of two early trades in 30 days.</li>
      <li>Every order is reviewed before it is placed: hard checks and the AI reviewer can block a
        buy, on every trigger, and never block a sell</li>
      <li>Closing a portfolio sells every real holding and removes it from the Portfolio tab's
        list and dropdown (it stays reachable by direct link, just no longer part of ongoing
        tracking)</li>
    </ul>

    <h2 id="daytrading">11. Day trading</h2>
    <p>A separate capability from everything above: a long-only intraday engine on US equities
      that opens and closes every position within the same trading day (flat by 3:45pm ET). It has
      its <b>own dedicated Alpaca paper account</b> — Alpaca allows up to three paper accounts per
      user, each with its own keys — so its constant buying and selling can never mix into the
      portfolios above or trip Alpaca's wash-trade protection against their orders. The
      <a href="/daytrade">Day Trading tab</a> shows it; a Cloud Run Job (not this web app) does
      the trading.</p>
    <ul>
      <li><b>No model in the trading loop.</b> Every minute, plain rules (opening-range breakout,
        VWAP pullback, gap-and-go) look for setups; a deterministic risk layer sizes each trade
        from cash (never margin), caps risk at ~1% of the sleeve per trade, and vetoes it if any
        limit is hit. Each entry is a <i>bracket order</i>, so its stop and target rest at the
        broker even if the runner crashes.</li>
      <li><b>Claude's one job</b> is a once-a-day pre-market playbook: risk on/off, a smaller risk
        multiplier, which strategies to allow, which names to watch. It can <i>only restrict</i> —
        its answer is clamped in code — so a bad or manipulated answer can make the system trade
        less, never more.</li>
      <li><b>Hard stops the model can't touch:</b> a daily-loss halt, a max-trades cap, an
        end-of-day flatten, and a kill switch on the tab (anyone can halt; resuming needs a token).</li>
      <li><b>Nothing goes live on hope.</b> Strategies are backtested with pessimistic fills and
        real costs against a random-entry null model, then must pass the same champion/challenger
        promotion gate as everything else — which deliberately can't pass until real paper results
        exist. Most intraday rules do not survive costs; the backtest exists to find that out
        before any infrastructure runs.</li>
      <li><b>Is the LLM step earning its keep?</b> Each day the runner also simulates the same
        engine with no playbook; the tab compares the two.</li>
    </ul>
  </div>
</body>
</html>
"""
ABOUT_PAGE = ABOUT_PAGE.replace("<body>", "<body>\n" + TOP_NAV, 1)


@app.route("/about")
def about_route():
    return render_template_string(ABOUT_PAGE, sectors=SECTORS, universe_size=len(UNIVERSE),
                                  active_page="about")


def _portfolio_selector(portfolios: List[Dict[str, Any]], current: str) -> str:
    """Shared dropdown injected into the compare-all list and every
    individual portfolio detail page — jump between your tracked
    portfolios, each labeled with its REAL current Alpaca-priced value
    (see _all_portfolios_enriched). Deliberately does NOT offer the raw
    combined Alpaca account here — this control is about YOUR named,
    independently-tracked portfolios, not the shared underlying account
    they all really settle into (see portfolio_store.py's module docstring).

    OPEN ONLY: closing/liquidating a portfolio removes it from this dropdown
    (and from the compare-all list) entirely — filtered here so every
    caller gets this for free rather than each needing to remember it.
    A closed portfolio is still reachable directly by its /portfolios/<id>
    URL, just no longer surfaced for ongoing navigation between portfolios."""
    portfolios = [p for p in portfolios if p.get("status") == "open"]

    def opt(value: str, label: str) -> str:
        sel = " selected" if value == current else ""
        return f'<option value="{value}"{sel}>{label}</option>'

    options = [opt("/portfolios?view=all", "📊 Compare All Tracked Portfolios")]
    for p in portfolios:
        value = p.get("mtm_value", p.get("capital", 0.0))
        options.append(opt(f"/portfolios/{p['id']}", f"📈 {p['name']} — ${value:,.0f} ({p['status']})"))
    return f"""
    <div style="margin-bottom:16px;">
      <label style="font-size:0.85rem;color:#666;">Viewing:
        <select onchange="window.location.href=this.value" style="margin-left:8px;padding:8px 12px;
                border-radius:8px;border:1.5px solid #ddd;font-family:inherit;font-size:0.9rem;
                max-width:100%;">
          {''.join(options)}
        </select>
      </label>
    </div>
    """


PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 780px;
        margin: 0 auto; padding: 32px 16px 64px; color: #0d1117; line-height: 1.45;
        background: #fff; }
  h1 { font-size: 1.3rem; font-weight: 700; margin-bottom: 4px; }
  .sub { color: #6a7280; font-size: 0.9rem; margin-bottom: 20px; }
  h2 { font-size: 1.05rem; margin: 28px 0 10px; color: #333; }
  .checks { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .chip { position: relative; }
  .chip input { position: absolute; opacity: 0; }
  .chip label { display: inline-block; padding: 8px 16px; border-radius: 20px; border: 1.5px solid #ddd;
               cursor: pointer; font-size: 0.92rem; font-weight: 600; color: #555; transition: all .12s; }
  .chip input:checked + label { background: #00c805; border-color: #00c805; color: white; }
  form.controls { display: flex; gap: 10px; margin: 14px 0 20px; flex-wrap: wrap; align-items: center; }
  input, button, textarea { font-size: 1rem; padding: 9px 14px; border-radius: 8px;
                   border: 1.5px solid #ddd; font-family: inherit; }
  input[type=number] { width: 120px; max-width: 100%; }
  button { background: #0d1117; color: white; border: none; cursor: pointer; font-weight: 600; }
  button:hover { background: #262d38; }
  .hero { background: linear-gradient(135deg,#f6fff7,#f4f7ff); border-radius: 16px; padding: 20px 22px;
         margin: 8px 0 20px; }
  .hero .value { font-size: 2.1rem; font-weight: 800; }
  .hero .delta { font-size: 1.05rem; font-weight: 700; margin-left: 8px; }
  .delta.pos { color: #00a340; } .delta.neg { color: #d43f3f; }
  .hero .label { color: #6a7280; font-size: 0.85rem; margin-top: 2px; }
  .sector-block { margin: 22px 0; }
  .sector-head { display: flex; justify-content: space-between; align-items: baseline;
                border-bottom: 2px solid #f0f0f5; padding-bottom: 6px; margin-bottom: 8px; }
  .sector-head .sname { font-weight: 700; font-size: 1rem; }
  .tilt { font-size: 0.82rem; padding: 2px 8px; border-radius: 10px; font-weight: 700; }
  .tilt.up { background: #e6f9ec; color: #00a340; }
  .tilt.down { background: #fdeaea; color: #d43f3f; }
  .tilt.flat { background: #eef0f5; color: #777; }
  .pie-wrap { display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
             gap: 24px; margin: 4px 0 22px; }
  .pie-slice { cursor: pointer; transition: opacity .1s; }
  .pie-slice:hover { opacity: 0.8; }
  .pie-legend { display: flex; flex-wrap: wrap; gap: 6px 14px; max-width: 320px; }
  .pie-legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.82rem;
                     color: #444; cursor: pointer; padding: 2px 4px; border-radius: 6px; }
  .pie-legend-item:hover { background: #f6f8ff; }
  .swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; display: inline-block; }
  .card { border: 1px solid #eee; border-radius: 12px; padding: 14px 16px; margin: 8px 0;
         transition: box-shadow .15s, border-color .15s; scroll-margin-top: 20px; }
  .card.selected { border-color: #4a6cf7; box-shadow: 0 0 0 3px rgba(74,108,247,0.15); }
  .card-top { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
  .name { font-size: 1.05rem; font-weight: 700; color: #0d1117; text-decoration: none;
         display: inline-flex; align-items: center; gap: 8px; }
  .name:hover { text-decoration: underline; }
  .detail-hint { font-size: 0.75rem; font-weight: 500; color: #4a6cf7; }
  .shares { color: #888; font-size: 0.85rem; }
  .weight { color: #0d1117; font-weight: 700; }
  .bar-track { background: #f0f0f5; height: 6px; border-radius: 3px; margin: 8px 0; }
  .bar { background: #00c805; height: 6px; border-radius: 3px; }
  .action { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;
           font-weight: 700; }
  .action.BUY { background: #e6f9ec; color: #00a340; }
  .action.SELL { background: #fdeaea; color: #d43f3f; }
  .action.HOLD { background: #eef0f5; color: #555; }
  .action.SKIP { background: #eef0f5; color: #888; }
  .score-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 10px; }
  .rating { font-size: 0.74rem; padding: 2px 10px; border-radius: 10px; font-weight: 700; }
  .rating.buy { background: #e6f9ec; color: #00a340; }
  .rating.hold { background: #eef0f5; color: #777; }
  .rating.sell { background: #fdeaea; color: #d43f3f; }
  .metric-pill { font-size: 0.74rem; padding: 2px 9px; border-radius: 10px; font-weight: 600;
                background: #f6f8ff; color: #556; border: 1px solid #e8ecfb; }
  .metric-pill .mv { font-weight: 700; color: #0d1117; }
  .forecast { font-size: 0.82rem; font-weight: 700; margin-left: auto; cursor: help; }
  .forecast.pos { color: #00a340; } .forecast.neg { color: #d43f3f; }
  .meta-row { color: #666; font-size: 0.85rem; margin-top: 4px; }
  .rationale { margin-top: 8px; font-size: 0.92rem; }
  .weight-rationale { margin-top: 4px; font-size: 0.88rem; color: #4a5568; }
  .risk { margin-top: 4px; font-size: 0.88rem; color: #a3312f; }
  .warn { color: #a3312f; background: #fee; padding: 12px 14px; border-radius: 10px; }
  .meta { color: #999; font-size: 0.82rem; margin-top: 28px; }
  .expected { background: #f6f8ff; border-radius: 12px; padding: 14px 16px; margin: 18px 0; }
  .chat-answer { background: #f6f8ff; border-radius: 12px; padding: 14px 16px; margin: 12px 0; white-space: pre-wrap; }
  .action-summary { display: flex; flex-direction: column; gap: 8px; margin: 8px 0 18px; }
  .action-summary div { border-radius: 10px; padding: 10px 14px; font-size: 0.88rem; }
  .action-summary .buy { background: #e6f9ec; color: #00873d; }
  .action-summary .sell { background: #fff8e6; color: #92650a; }
  .held-tag { display: inline-block; margin-left: 6px; padding: 1px 8px; border-radius: 10px;
             font-size: 0.72rem; font-weight: 700; background: #eef0f5; color: #555; }
  .execute-box { background: #fff8e6; border-radius: 12px; padding: 16px 18px; margin: 18px 0; }
  .execute-btn { margin-top: 10px; background: #d97706; font-size: 1rem; padding: 12px 20px; }
  .execute-btn:hover { background: #b45309; }
  textarea { width: 100%; height: 60px; margin-bottom: 8px; }
  nav a { color: #0d1117; text-decoration: none; font-weight: 600; font-size: 0.88rem; }
  .progress-box { background: #f6f8ff; border-radius: 12px; padding: 16px 18px; margin: 14px 0; }
  .progress-box .spinner { display: inline-block; width: 14px; height: 14px; border-radius: 50%;
                           border: 2px solid #c7d2fe; border-top-color: #4a6cf7;
                           animation: spin 0.8s linear infinite; margin-right: 8px; vertical-align: -2px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .progress-steps { list-style: none; padding: 0; margin: 10px 0 0; font-size: 0.88rem; color: #444; }
  .progress-steps li { padding: 3px 0; opacity: 0.55; }
  .progress-steps li.current { opacity: 1; font-weight: 600; color: #0d1117; }
  .progress-steps li.done::before { content: "✓ "; color: #00a340; font-weight: 700; }
  .progress-steps li.current::before { content: ""; }
  @media (max-width: 600px) {
    body { padding: 20px 14px 56px; }
    h1 { font-size: 1.2rem; }
    .hero .value { font-size: 1.7rem; }
    form.controls { flex-direction: column; align-items: stretch; }
    form.controls label { display: flex; flex-direction: column; gap: 4px; width: 100%; }
    input[type=number] { width: 100%; }
    form.controls button, .execute-btn { width: 100%; }
    .action-summary { flex-direction: column; }
    .execute-box form { display: flex; flex-direction: column; gap: 8px; align-items: stretch; }
    .execute-box label { display: flex; flex-direction: column; gap: 4px; }
  }
</style>
</head>
<body>
  <h1>Meridian</h1>
  <div class="sub">Pick one or more sectors — the AI decides how to split capital between them AND which names within each.</div>

  <form class="controls" id="reco-form" method="post" action="/">
    <div class="checks">
      {% for key, label in sectors.items() %}
        <div class="chip">
          <input type="checkbox" name="sectors" value="{{ key }}" id="s-{{ key }}"
                {{ 'checked' if key in selected else '' }}>
          <label for="s-{{ key }}">{{ label }}</label>
        </div>
      {% endfor %}
    </div>
    <label>Capital $ <input type="number" name="capital" value="{{ capital|int }}" step="any" min="0"></label>
    <label>Max # of stocks <input type="number" name="top_n" value="{{ top_n }}" step="1" min="1" max="{{ universe_size }}"
           title="Highest-conviction names the AI will actually fund, TOTAL across every sector you select combined — raise for more diversification, lower to concentrate on top picks."></label>
    <button type="submit">Get Recommendation</button>
  </form>

  <div class="progress-box" id="progress-box" hidden>
    <div><span class="spinner"></span><strong>Running the real pipeline live</strong> — this streams
      each step as the deployed agent actually completes it, usually 1-6 minutes.</div>
    <ul class="progress-steps" id="progress-steps"></ul>
  </div>

  {% if error %}
    <div class="warn">Request failed: {{ error }}</div>
  {% elif ctx %}
    {% set total_weighted = ctx.rows|sum(attribute='weight') %}
    <div class="hero">
      <div class="label">Recommended allocation of ${{ '{:,.0f}'.format(ctx.capital) }}
        across {{ ctx.selected_sectors|length }} sector{{ 's' if ctx.selected_sectors|length != 1 }}</div>
      <span class="value">${{ '{:,.0f}'.format(ctx.capital * total_weighted) }}</span>
      <span class="delta {{ 'pos' if ctx.expected_return >= 0 else 'neg' }}">
        {{ '%+.1f'|format(ctx.expected_return*100) }}% naive 1-mo projection</span>
      <div class="label">status: {{ ctx.status }} · risk check: {{ ctx.risk_verdict }}</div>
    </div>

    {% if ctx.rows %}
      <div class="pie-wrap">
        {{ ctx.pie_chart|safe }}
        <div class="pie-legend">
          {% for r in ctx.rows %}
          <div class="pie-legend-item" onclick="selectStock('{{ r.name }}')">
            <span class="swatch" style="background:{{ r.color }}"></span>
            {{ r.name }} <span style="color:#999">{{ '%.1f'|format(r.weight*100) }}%</span>
          </div>
          {% endfor %}
        </div>
      </div>
    {% endif %}

    {% if ctx.rows %}
      {% if ctx.total_to_deploy %}
      <div class="action-summary">
        <div class="buy"><strong>${{ '{:,.0f}'.format(ctx.total_to_deploy) }}</strong> would be deployed as
          {{ ctx.rows|selectattr('action','equalto','BUY')|list|length }} new whole-share BUY orders if you
          launch this as a portfolio — a fresh, independent allocation funded from the ${{ '{:,.0f}'.format(ctx.capital) }}
          capital above, <strong>not</strong> a rebalance of what you already hold.</div>
        {% if ctx.residual_cash >= 1 %}
        <div class="sell">${{ '{:,.0f}'.format(ctx.residual_cash) }} stays uninvested — the leftover from
          rounding each position down to whole shares.</div>
        {% endif %}
        {% if ctx.already_held_count %}
        <div class="sell">{{ ctx.already_held_count }} of these you already hold in your combined Alpaca
          account (see "already held" tags below) — buying more here adds to that position; it does not
          treat what you hold as already satisfying this recommendation.</div>
        {% endif %}
      </div>
      {% endif %}

      {% for sec in ctx.sections %}
        {% if sec.rows %}
        <div class="sector-block">
          <div class="sector-head">
            <span class="sname">{{ sec.label }} — {{ '%.1f'|format(sec.total_weight*100) }}% of portfolio</span>
            {% if sec.net_tilt is not none %}
              <span class="tilt {{ 'up' if sec.net_tilt > 0.005 else ('down' if sec.net_tilt < -0.005 else 'flat') }}">
                AI tilt {{ '%+.1f'|format(sec.net_tilt*100) }}%</span>
            {% endif %}
          </div>
          {% set top = sec.rows[0].weight %}
          {% for r in sec.rows %}
          <div class="card" id="card-{{ r.name }}">
            <div class="card-top">
              <a class="name" href="/stock/{{ r.name }}">
                <span class="swatch" style="background:{{ r.color }}"></span>{{ r.name }}
                <span class="detail-hint">details →</span></a>
              <span class="weight">{{ '%.1f'|format(r.weight*100) }}%</span>
            </div>
            <div class="bar-track"><div class="bar" style="width: {{ (r.weight/top*100)|round(0) }}%"></div></div>
            {% if r.rating or r.conviction is not none %}
            <div class="score-row">
              {% if r.rating %}
                <span class="rating {{ r.rating|lower }}"
                      title="{{ 'Real Wall Street analyst consensus (%d analysts)'|format(r.analyst_count) if r.analyst_count else 'No analyst coverage found — a heuristic/AI-generated estimate, not real analyst data' }}">
                  {{ r.rating }}{% if r.analyst_count %} <span class="mv">· {{ r.analyst_count }}A</span>{% endif %}</span>
              {% endif %}
              {% if r.conviction is not none %}<span class="metric-pill" title="Analyst conviction score (0-100)">Conviction <span class="mv">{{ '%.0f'|format(r.conviction*100) }}</span></span>{% endif %}
              <span class="metric-pill" title="Trailing 1-month momentum">Mtm <span class="mv">{{ '%+.0f'|format(r.momentum_1m*100) }}</span></span>
              {% if r.news_sentiment is not none %}<span class="metric-pill" title="News sentiment, -100 (negative) to +100 (positive)">Sentiment <span class="mv">{{ '%+.0f'|format(r.news_sentiment*100) }}</span></span>{% endif %}
              {% if r.quality_score is not none %}<span class="metric-pill" title="Fundamentals/quality proxy (0-100)">Quality <span class="mv">{{ '%.0f'|format(r.quality_score*100) }}</span></span>{% endif %}
              {% if r.volatility is not none %}<span class="metric-pill" title="Daily volatility (stdev of returns)">Vol <span class="mv">{{ '%.1f'|format(r.volatility*100) }}%</span></span>{% endif %}
              {% if r.expected_return is not none %}
              <span class="forecast {{ 'pos' if r.expected_return >= 0 else 'neg' }}"
                    title="{{ 'Real analyst consensus target price, implied forward return' if r.analyst_count else 'A rough heuristic/AI estimate, not real analyst data — no coverage found for this name' }}">
                {{ '%+.1f'|format(r.expected_return*100) }}% {{ 'target' if r.analyst_count else 'est.' }}</span>
              {% endif %}
            </div>
            {% endif %}
            <div class="meta-row">
              {% if r.price %}Price: ${{ '%.2f'|format(r.price) }} &middot; {% endif %}
              1-mo momentum: {{ '%+.1f'|format(r.momentum_1m*100) }}%
              &middot; <span class="action {{ r.action }}">{{ r.action }}{% if r.action_notional %} ${{ '{:,.0f}'.format(r.action_notional) }}{% endif %}</span>
              {% if r.shares %}<span class="shares">{{ r.shares }} share{{ 's' if r.shares != 1 }}</span>{% endif %}
              {% if r.already_held %}<span class="held-tag">already held</span>{% endif %}
            </div>
            {% if r.rationale %}<div class="rationale">💡 {{ r.rationale }}</div>{% endif %}
            {% if r.weight_rationale %}<div class="weight-rationale">⚖️ {{ r.weight_rationale }}</div>{% endif %}
            {% if r.risk_note %}<div class="risk">⚠ {{ r.risk_note }}</div>{% endif %}
          </div>
          {% endfor %}
        </div>
        {% endif %}
      {% endfor %}

      <div class="expected">
        <strong>Naive expected 1-month portfolio return: {{ '%+.1f'|format(ctx.expected_return*100) }}%</strong>
        <div class="meta-row">Extrapolates each name's trailing 1-month momentum forward,
        weighted by its recommended allocation. This is NOT a forecast, backtest, or
        guarantee — trailing momentum persisting is a weak assumption. Use the
        Simulation page to see how this approach would ACTUALLY have performed
        historically.</div>
      </div>

      {% if ctx.rows %}
      <div class="execute-box">
        <strong>Ready to act on this?</strong>
        <div class="meta-row">Launches this as a new, independently-tracked <strong>portfolio</strong> —
          real whole-share BUY orders on your Alpaca <strong>paper</strong> account (never live/real
          money) sized off the capital above and the weights shown. Every launch gets its own name
          and ledger, so you can run several strategies side by side and
          <a href="/portfolios">compare their performance →</a></div>
        <form method="post" action="/execute"
              onsubmit="return confirm('Launch a new tracked portfolio with ' + {{ ctx.rows|length }} + ' names on Alpaca now?');">
          {% for s in ctx.selected_sectors %}<input type="hidden" name="sectors" value="{{ s }}">{% endfor %}
          <input type="hidden" name="capital" value="{{ ctx.capital }}">
          <input type="hidden" name="top_n" value="{{ top_n }}">
          <input type="hidden" name="batch_id" value="{{ ctx.batch_id }}">
          <label>Portfolio name
            <input type="text" name="portfolio_name" placeholder="e.g. AI momentum #1" maxlength="60"></label>
          <label>Auto-recalibrate
            <select name="rebalance_cadence">
              <option value="0">Manual only — I'll rebalance it myself</option>
              {% for key, label in rebalance_options.items() %}
                <option value="{{ key }}">{{ label }}</option>
              {% endfor %}
            </select>
          </label>
          <label style="display:flex;gap:8px;align-items:flex-start;font-weight:400;">
            <input type="checkbox" name="news_watch" value="1" style="margin-top:3px;">
            <span>Mid-day news watch — check this portfolio's news every trading day at 12:30 ET and
              recalibrate early when it materially changes the picture. You can turn it on or off later.</span></label>
          <button type="submit" class="execute-btn">Launch as New Portfolio (Alpaca Paper)</button>
        </form>
      </div>
      {% endif %}

      <p class="meta">buy/sell actions and share counts are computed against your real
        current Alpaca paper positions, sized to the capital above. Sector tilts come
        from the same strategist logic the live deployed agent uses (Claude-backed
        when quota allows, heuristic fallback otherwise).</p>
    {% else %}
      <div class="warn">
        No recommendation — {{ ctx.risk_verdict }}: {{ ctx.risk_reasons }}
      </div>
    {% endif %}

    <h2>Ask a question about this recommendation</h2>
    <form method="post" action="/chat">
      <textarea name="question" placeholder="e.g. Why did AI/Tech get more capital than Energy? What's the biggest risk here?"></textarea>
      {% for s in ctx.selected_sectors %}<input type="hidden" name="sectors" value="{{ s }}">{% endfor %}
      <input type="hidden" name="capital" value="{{ ctx.capital }}">
      <input type="hidden" name="top_n" value="{{ top_n }}">
      <button type="submit">Ask</button>
    </form>
    {% if answer %}
      <div class="chat-answer"><strong>Q: {{ question }}</strong><br><br>{{ answer }}</div>
    {% endif %}
  {% endif %}

  <p class="meta">
    Calls the real deployed agent (Vertex AI Agent Engine) with live Alpaca
    market/news data. Paper-trading infrastructure only — nothing here places
    real orders. Not investment advice.
  </p>
<script>
function selectStock(symbol) {
  document.querySelectorAll('.card.selected').forEach(function(el) { el.classList.remove('selected'); });
  var card = document.getElementById('card-' + symbol);
  if (card) {
    card.classList.add('selected');
    card.scrollIntoView({behavior: 'smooth', block: 'center'});
  }
}

// Progressive enhancement: if EventSource isn't available, the form's
// normal method="post" action="/" submit still works exactly as before —
// just without live progress. When it IS available, intercept the submit
// and drive /recommend_stream instead, so the SAME real graph run's node
// completions show up live instead of a blank multi-minute wait.
(function() {
  var form = document.getElementById('reco-form');
  if (!form || !window.EventSource) return;
  form.addEventListener('submit', function(e) {
    e.preventDefault();
    var btn = form.querySelector('button[type=submit]');
    var box = document.getElementById('progress-box');
    var list = document.getElementById('progress-steps');
    list.innerHTML = '';
    box.hidden = false;
    box.scrollIntoView({behavior: 'smooth', block: 'center'});
    btn.disabled = true;
    btn.textContent = 'Working...';

    var params = new URLSearchParams();
    new FormData(form).forEach(function(v, k) { params.append(k, v); });

    var es = new EventSource('/recommend_stream?' + params.toString());
    var lastNode = null;

    function markPriorDone() {
      Array.prototype.forEach.call(list.querySelectorAll('li.current'), function(el) {
        el.classList.remove('current'); el.classList.add('done');
      });
    }

    es.onmessage = function(ev) {
      var data;
      try { data = JSON.parse(ev.data); } catch (err) { return; }
      if (data.error) {
        markPriorDone();
        var li = document.createElement('li');
        li.textContent = '⚠ ' + data.error;
        list.appendChild(li);
        es.close();
        btn.disabled = false;
        btn.textContent = 'Get Recommendation';
        return;
      }
      if (data.done) {
        markPriorDone();
        es.close();
        window.location.href = '/';
        return;
      }
      if (data.step) {
        if (data.node !== lastNode) {
          markPriorDone();
          var li = document.createElement('li');
          li.textContent = data.step;
          li.className = 'current';
          list.appendChild(li);
          lastNode = data.node;
        }
      }
    };
    es.onerror = function() {
      es.close();
      btn.disabled = false;
      btn.textContent = 'Get Recommendation';
      if (!list.querySelector('li')) {
        box.hidden = true;
      }
    };
  });
})();
</script>
</body>
</html>
"""
PAGE = PAGE.replace("<body>", "<body>\n" + TOP_NAV, 1)


@app.route("/", methods=["GET", "POST"])
def index():
    selected = request.form.getlist("sectors") or _LAST_CONTEXT.get("selected_sectors") or ["tech_ai"]
    capital = float(request.form.get("capital", DEFAULT_CAPITAL) or DEFAULT_CAPITAL)
    top_n = int(float(request.form.get("top_n", TOP_N_TOTAL) or TOP_N_TOTAL))
    top_n = max(1, min(top_n, len(UNIVERSE)))   # user can pick ANY value; just can't exceed the universe size
    ctx, error = None, None
    if request.method == "POST":
        # Progressive-enhancement fallback ONLY — the page's JS intercepts
        # the form submit and drives /recommend_stream (below) via
        # EventSource instead, so this branch only runs for a client with
        # JS disabled (no live progress updates in that case, just a plain
        # long wait, same as before streaming existed).
        try:
            result = recommend(selected, top_n)
            ctx = enrich(result, selected, capital)
            _LAST_CONTEXT["top_n"] = top_n
        except Exception as e:
            error = str(e)
    elif _LAST_CONTEXT:
        # A GET after /recommend_stream finishes (JS reloads the page once
        # it sees {"done": true}) — show the result it just computed.
        ctx = _LAST_CONTEXT
        selected = _LAST_CONTEXT.get("selected_sectors", selected)
        capital = _LAST_CONTEXT.get("capital", capital)
        # BUG FIX: this reload used to fall back to the default (12), so every
        # portfolio was LAUNCHED with top_n=12 whatever was chosen — and every
        # recalibration then re-selected 12.
        top_n = int(_LAST_CONTEXT.get("top_n") or top_n)
    return render_template_string(PAGE, sectors=SECTORS, selected=selected, capital=capital,
                                  top_n=top_n, universe_size=len(UNIVERSE), active_page="home",
                                  rebalance_options=REBALANCE_OPTIONS,
                                  ctx=ctx, error=error, answer=None, question=None)


@app.route("/recommend_stream")
def recommend_stream():
    """Server-Sent Events endpoint: drains the deployed agent's
    stream_query() and relays each REAL graph node's completion as a
    progress line, then fetches the final accumulated state (get_final_state
    — no second graph run) and stashes it in _LAST_CONTEXT for the page's
    next GET / to render. GET (not POST) because browser EventSource can
    only issue GET requests."""
    import json
    selected = request.args.getlist("sectors") or ["tech_ai"]
    capital = float(request.args.get("capital", DEFAULT_CAPITAL) or DEFAULT_CAPITAL)
    top_n = int(float(request.args.get("top_n", TOP_N_TOTAL) or TOP_N_TOTAL))
    top_n = max(1, min(top_n, len(UNIVERSE)))

    def generate():
        thread_id, state = _build_recommend_state(selected, top_n)
        try:
            for chunk in _remote_agent.stream_query(thread_id=thread_id, state=state):
                if not isinstance(chunk, dict):
                    continue
                for node_name in chunk.keys():
                    if node_name in HIDDEN_NODES:
                        continue
                    label = NODE_LABELS.get(node_name, f"Running {node_name}...")
                    yield f"data: {json.dumps({'step': label, 'node': node_name})}\n\n"
            final_state = _call_with_timeout(_remote_agent.get_final_state, thread_id=thread_id, timeout=60.0)
            enrich(_finish_recommendation(final_state), selected, capital)   # populates _LAST_CONTEXT
            _LAST_CONTEXT["top_n"] = top_n           # the launch form must carry THIS, not the default
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/chat", methods=["POST"])
def chat():
    question = request.form.get("question", "").strip()
    selected = request.form.getlist("sectors") or ["tech_ai"]
    capital = float(request.form.get("capital", DEFAULT_CAPITAL) or DEFAULT_CAPITAL)
    top_n = int(float(request.form.get("top_n", TOP_N_TOTAL) or TOP_N_TOTAL))
    answer = ask_claude(question, _LAST_CONTEXT) if question else None
    return render_template_string(PAGE, sectors=SECTORS, selected=selected, capital=capital,
                                  top_n=top_n, universe_size=len(UNIVERSE), active_page="home",
                                  rebalance_options=REBALANCE_OPTIONS,
                                  ctx=_LAST_CONTEXT or None, error=None,
                                  answer=answer, question=question)


EXECUTE_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian — Execution Results</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 680px;
        margin: 0 auto; padding: 32px 16px 64px; color: #0d1117; line-height: 1.45; }
  h1 { font-size: 1.3rem; }
  .paper-banner { background: #fff8e6; border-radius: 10px; padding: 10px 14px; font-size: 0.85rem;
                 margin-bottom: 20px; font-weight: 600; }
  .card { border: 1px solid #eee; border-radius: 12px; padding: 14px 16px; margin: 10px 0; }
  .card-top { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 6px; }
  .name { font-weight: 700; }
  .result { font-size: 0.8rem; font-weight: 700; padding: 2px 10px; border-radius: 12px; }
  .result.FILLED { background: #e6f9ec; color: #00a340; }
  .result.PARTIAL { background: #fff8e6; color: #b45309; }
  .result.REJECTED { background: #fdeaea; color: #d43f3f; }
  .result.TIMEOUT { background: #eef0f5; color: #555; }
  .result.ERROR { background: #fdeaea; color: #d43f3f; }
  .meta-row { color: #666; font-size: 0.85rem; margin-top: 4px; }
  .warn { color: #a3312f; background: #fee; padding: 12px 14px; border-radius: 10px; margin: 12px 0; }
  .ok { color: #00a340; background: #e6f9ec; padding: 12px 14px; border-radius: 10px; margin: 12px 0; font-weight: 600; }
  .meta { color: #999; font-size: 0.82rem; margin-top: 28px; }
  @media (max-width: 600px) { body { padding: 20px 14px 56px; } }
</style>
</head>
<body>
  <h1>Execution Results{% if portfolio_name %}: {{ portfolio_name }}{% endif %}</h1>
  <div class="paper-banner">📝 PAPER TRADING — these were real orders on your Alpaca
    paper account, not real money.</div>

  {% if error %}<div class="warn">{{ error }}</div>{% endif %}
  {% if portfolio_id %}
    <div class="ok">Tracked as a new portfolio ({{ fills|selectattr("result", "in", ["FILLED","PARTIAL"])|list|length }}
      of {{ fills|length }} orders filled). <a href="/portfolios">Compare it against your other portfolios →</a></div>
  {% endif %}

  {% if not fills %}
    <p>Nothing to execute — no name had both a live price and a whole-share allocation.</p>
  {% endif %}
  {% for f in fills %}
  <div class="card">
    <div class="card-top">
      <span class="name">{{ f.symbol }}</span>
      <span class="result {{ f.result }}">{{ f.result }}</span>
    </div>
    <div class="meta-row">{{ f.side }} {{ f.shares }} share{{ 's' if f.shares != 1 }}
      {% if f.filled_notional %}· filled ${{ '{:,.2f}'.format(f.filled_notional) }}{% endif %}
      {% if f.reason %}· {{ f.reason }}{% endif %}</div>
  </div>
  {% endfor %}

  <p class="meta">TIMEOUT means the order was still in flight when this page stopped
    polling — it may still fill; check your Alpaca account directly to confirm. ERROR means that one
    order failed (including after automatic retries on transient Alpaca gateway timeouts) — every
    other order in this batch was still attempted independently, and re-launching is safe to retry
    since each order has a stable idempotency key.</p>
</body>
</html>
"""
EXECUTE_PAGE = EXECUTE_PAGE.replace("<body>", "<body>\n" + TOP_NAV, 1)


@app.route("/execute", methods=["POST"])
def execute():
    selected = request.form.getlist("sectors") or ["tech_ai"]
    capital = float(request.form.get("capital", DEFAULT_CAPITAL) or DEFAULT_CAPITAL)
    top_n = int(float(request.form.get("top_n", TOP_N_TOTAL) or TOP_N_TOTAL))
    batch_id = request.form.get("batch_id", "") or uuid.uuid4().hex[:10]
    portfolio_name = request.form.get("portfolio_name", "").strip()
    rebalance_every_days = int(float(request.form.get("rebalance_cadence", 0) or 0))
    news_watch = request.form.get("news_watch") == "1"
    fills, error, portfolio_id = [], None, None

    if not _LAST_CONTEXT or not _LAST_CONTEXT.get("rows"):
        error = "No recommendation to execute — get a recommendation first."
    else:
        from alpaca.trading.client import TradingClient
        from alpaca_clients import AlpacaBrokerClient
        from execution_workflow import Order, Side
        from datetime import datetime, timezone
        import portfolio_store

        launch_date = datetime.now(timezone.utc).date().isoformat()
        tc = TradingClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY, paper=True)   # PAPER ONLY, hardcoded
        broker = AlpacaBrokerClient(tc)

        # Every "Launch" creates a FRESH, independently-tracked portfolio —
        # sized straight off THIS portfolio's own capital and the
        # recommended weights (not a delta against combined Alpaca
        # positions), so two portfolios launched back to back are directly
        # comparable rather than one silently treating the other's fills as
        # "already held." See portfolio_store.py's module docstring for why
        # real fills still land in one shared Alpaca paper account either way.
        holdings: Dict[str, Dict[str, float]] = {}
        errors: List[Dict[str, str]] = []
        # Orders that were placed and ACCEPTED by Alpaca but hadn't resolved
        # to FILLED/PARTIAL by the time we stopped polling (e.g. placed
        # after market hours) — confirmed live: closing a portfolio before
        # these land would otherwise leave a real, un-tracked position
        # sitting on Alpaca once they eventually fill. Cancelled explicitly
        # in close_portfolio_route() below, not just left dangling.
        pending_order_ids: Dict[str, str] = {}
        spent = 0.0
        # BUG FIX, confirmed against real Alpaca order history: _LAST_CONTEXT
        # is a process-wide global set whenever /recommend_stream last ran —
        # it can be minutes, hours, or (on a long-lived Cloud Run instance)
        # even longer old by the time "Launch as New Portfolio" is actually
        # clicked, or overwritten by someone else's concurrent recommendation
        # in between. Sizing shares off a stale cached price meant one real
        # launched portfolio bought the WRONG number of shares for its
        # intended dollar weight, and recorded an entry price that matched
        # neither the real fill nor the real market — a fresh quote right
        # before ordering fixes both at once. Falls back to the cached price
        # only if a fresh quote can't be had for some reason.
        fresh_prices = _live_prices([r["name"] for r in _LAST_CONTEXT["rows"]])
        # batch_id ties every order in THIS launch to a single idempotency
        # namespace — retrying the whole page after a partial failure reuses
        # the same client_order_ids, so already-filled orders are found via
        # get_order_by_client_id() instead of being placed a second time.
        for r in _LAST_CONTEXT["rows"]:
            price = fresh_prices.get(r["name"]) or r.get("price")
            weight = r.get("weight") or 0.0
            if not price or weight <= 0:
                continue
            shares = math.floor((capital * weight) / price)
            if shares < 1:
                continue
            client_order_id = f"webui-{batch_id}-{r['name']}-BUY"[:48]
            order = Order(symbol=r["name"], side=Side.BUY, notional=shares * price,
                         qty=shares, client_order_id=client_order_id)
            try:
                fill = broker.submit_and_await(order)
                result = fill.result.value.upper()
                fills.append({
                    "symbol": r["name"], "side": "BUY", "shares": shares,
                    "result": result, "filled_notional": fill.filled_notional,
                    "reason": fill.reason,
                })
                if result in ("FILLED", "PARTIAL"):
                    # BUG FIX: a genuine PARTIAL fill (order.qty shares
                    # requested, fewer actually bought) means REAL owned
                    # shares is fill.filled_qty, not the originally-intended
                    # `shares` — recording the intended count here would
                    # claim MORE shares than were actually purchased, and
                    # dividing the (smaller) real notional by that inflated
                    # count would understate entry_price, inflating this
                    # portfolio's apparent return. Falls back to `shares`
                    # only if filled_qty is unavailable (older BrokerClient
                    # implementations, or a FILLED result where it's
                    # expected to equal `shares` anyway).
                    real_shares = int(fill.filled_qty) if fill.filled_qty else shares
                    entry_price = (fill.filled_notional / real_shares) if fill.filled_notional and real_shares else price
                    holdings[r["name"]] = {"shares": real_shares, "entry_price": entry_price,
                                           "cost": fill.filled_notional or real_shares * price,
                                           "entry_date": launch_date}
                    spent += holdings[r["name"]]["cost"]
                elif result == "TIMEOUT":
                    pending_order_ids[r["name"]] = client_order_id
            except Exception as e:
                # ISOLATED per order — one failure (e.g. Alpaca's gateway
                # timing out, code 50410000) no longer aborts every order
                # after it in the batch. Each order already retries
                # transient errors internally (alpaca_clients._submit); this
                # catches whatever survives that retry.
                fills.append({"symbol": r["name"], "side": "BUY", "shares": shares,
                             "result": "ERROR", "filled_notional": 0.0, "reason": str(e)})
                errors.append({"symbol": r["name"], "error": str(e)})

        if fills:
            portfolio_id = portfolio_store.create_portfolio(
                name=portfolio_name, sectors=selected, capital=capital, top_n=top_n,
                holdings=holdings, cash_remaining=max(0.0, capital - spent), errors=errors,
                pending_order_ids=pending_order_ids, rebalance_every_days=rebalance_every_days,
                n_stocks=len(set(holdings) | set(pending_order_ids)),
                # what it STARTED with — the Portfolio page tracks every change from here
                initial_holdings={**{f["symbol"]: {"shares": f["shares"], "entry_price": None, "pending": True}
                                     for f in fills if f.get("result") == "TIMEOUT"},
                                  **{s: {"shares": h["shares"], "entry_price": h["entry_price"]}
                                     for s, h in holdings.items()}},
                estimated_1mo_return=_LAST_CONTEXT.get("expected_return"),
                news_watch=news_watch,
            )
        elif not error:
            error = "Nothing to execute — no name had both a live price and a whole-share allocation."

    return render_template_string(EXECUTE_PAGE, fills=fills, error=error, active_page="home",
                                  portfolio_id=portfolio_id, portfolio_name=portfolio_name)


SIMULATE_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian — Historical Simulation</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 820px;
        margin: 0 auto; padding: 32px 16px 64px; color: #0d1117; line-height: 1.45; }
  h1 { font-size: 1.3rem; margin-bottom: 4px; } h2 { font-size: 1.05rem; margin: 26px 0 10px; }
  .sub { color: #6a7280; font-size: 0.9rem; margin-bottom: 18px; }
  .checks { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .chip { position: relative; }
  .chip input { position: absolute; opacity: 0; }
  .chip label { display: inline-block; padding: 7px 14px; border-radius: 20px; border: 1.5px solid #ddd;
               cursor: pointer; font-size: 0.88rem; font-weight: 600; color: #555; }
  .chip input:checked + label { background: #4a6cf7; border-color: #4a6cf7; color: white; }
  form.controls { display: flex; gap: 10px; margin: 12px 0 22px; flex-wrap: wrap; align-items: center; }
  select, input, button { font-size: 1rem; padding: 9px 14px; border-radius: 8px;
                          border: 1.5px solid #ddd; font-family: inherit; }
  input { width: 130px; max-width: 100%; }
  button { background: #0d1117; color: white; border: none; cursor: pointer; font-weight: 600; }
  button:hover { background: #262d38; }
  .stat-row { display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }
  .stat { background: #f6f8ff; border-radius: 12px; padding: 14px 18px; flex: 1; min-width: 200px; }
  .stat .big { font-size: 1.5rem; font-weight: 800; }
  .stat.good .big { color: #00a340; } .stat.bad .big { color: #d43f3f; }
  .warn { color: #a3312f; background: #fee; padding: 12px 14px; border-radius: 10px; margin: 12px 0; }
  .meta { color: #999; font-size: 0.82rem; margin-top: 8px; }
  .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; margin-top: 12px; }
  table { width: 100%; min-width: 560px; border-collapse: collapse; font-size: 0.87rem; }
  th, td { text-align: left; padding: 7px 6px; border-bottom: 1px solid #eee; vertical-align: top; }
  .tilt-tag { display: inline-block; padding: 1px 7px; border-radius: 8px; font-size: 0.78rem; font-weight: 700; margin-right: 4px; }
  .tilt-tag.up { background: #e6f9ec; color: #00a340; } .tilt-tag.down { background: #fdeaea; color: #d43f3f; }
  .act { font-size: 0.78rem; font-weight: 700; padding: 1px 6px; border-radius: 6px; }
  .act.BUY { background: #e6f9ec; color: #00a340; } .act.SELL { background: #fdeaea; color: #d43f3f; }
  .caveats { background: #fff8e6; border-radius: 12px; padding: 14px 18px; margin-top: 24px; font-size: 0.88rem; }
  @media (max-width: 600px) {
    body { padding: 20px 14px 56px; }
    form.controls { flex-direction: column; align-items: stretch; }
    form.controls input, form.controls select, form.controls button { width: 100%; }
  }
</style>
</head>
<body>
  <h1>Simulate: what if I'd followed this system historically?</h1>
  <div class="sub">Pick sectors, a date range, and a rebalance cadence — the AI re-scores and
  re-tilts between them at each rebalance using only data available up to that point, and
  every re-tilt shows up below as explicit BUY/SELL decisions.</div>
  <form class="controls" method="post" action="/simulate">
    <div class="checks">
      {% for key, label in sectors.items() %}
        <div class="chip">
          <input type="checkbox" name="sectors" value="{{ key }}" id="sim-{{ key }}"
                {{ 'checked' if key in selected else '' }}>
          <label for="sim-{{ key }}">{{ label }}</label>
        </div>
      {% endfor %}
    </div>
    <label>From <input type="date" name="start" value="{{ start }}"></label>
    <label>To <input type="date" name="end" value="{{ end }}"></label>
    <label>Capital $ <input type="number" name="capital" value="{{ capital|int }}" step="any" min="0"></label>
    <label>Max # of stocks <input type="number" name="top_n" value="{{ top_n }}" step="1" min="1" max="{{ universe_size }}"
           title="Highest-conviction names kept before allocating, TOTAL across every sector selected combined — same setting as the home page."></label>
    <select name="rebalance">
      {% for key, label in rebalance_options.items() %}
        <option value="{{ key }}" {{ 'selected' if key == rebalance else '' }}>{{ label }} rebalancing</option>
      {% endfor %}
    </select>
    <button type="submit">Run Simulation</button>
  </form>

  {% if error %}
    <div class="warn">Simulation failed: {{ error }}</div>
  {% elif result %}
    <h2>{{ selected|map('title')|join(' + ') }} · {{ start }} → {{ end }} · ${{ '{:,.0f}'.format(capital) }} start</h2>
    <div class="stat-row">
      <div class="stat {{ 'good' if result.total_return >= 0 else 'bad' }}">
        <div>AI strategy final value</div>
        <div class="big">${{ '{:,.0f}'.format(result.final_value) }}</div>
        <div>{{ '%+.1f'|format(result.total_return*100) }}% total return, after ${{ '{:,.0f}'.format(result.total_costs) }} of costs</div>
        <div>Sharpe {{ '%.2f'|format(result.stats.sharpe) if result.stats.sharpe is not none else '—' }} ·
          max drawdown {{ '%.1f'|format((result.stats.max_drawdown or 0)*100) }}% ·
          turnover {{ '%.1f'|format(result.annual_turnover) }}x/yr · {{ result.n_stops }} stops</div>
      </div>
      <div class="stat">
        <div>Equal-weight buy &amp; hold (no rebalancing)</div>
        <div class="big">${{ '{:,.0f}'.format(result.benchmark_final_value) }}</div>
        <div>{{ '%+.1f'|format(result.benchmark_return*100) }}% total return</div>
      </div>
      {% if result.spy_values %}
      <div class="stat">
        <div>S&amp;P 500 (SPY, buy &amp; hold)</div>
        <div class="big">${{ '{:,.0f}'.format(result.spy_final_value) }}</div>
        <div>{{ '%+.1f'|format(result.spy_return*100) }}% total return</div>
      </div>
      {% endif %}
    </div>

    {{ chart|safe }}

    <h2>Rebalance history — what the AI bought/sold and why ({{ result.rebalance_dates|length }} rebalances)</h2>
    <div class="table-wrap">
    <table>
      <tr><th>Date</th><th>Sector tilts</th><th>Actions taken</th>
        <th>AI strategy weight <span style="font-weight:400;color:#999;">(vs. equal-weight of the same picks)</span></th></tr>
      {% for date, weights, tilts, actions in rebalance_history %}
      <tr>
        <td>{{ date }}</td>
        <td>
          {% for sec, t in tilts.items() %}
            <span class="tilt-tag {{ 'up' if t > 0.01 else 'down' if t < -0.01 else '' }}">{{ sec }} {{ '%+.0f'|format(t*100) }}%</span>
          {% endfor %}
        </td>
        <td>
          {% for sym, (act, amt) in actions.items() %}
            <span class="act {{ act }}">{{ act }} {{ sym }}</span>{% if not loop.last %} {% endif %}
          {% endfor %}
        </td>
        <td>
          {% set eq_pct = 100.0 / weights|length if weights else 0 %}
          {% for name, w in weights.items()|sort(attribute='1', reverse=True) %}
            {{ name }} <strong>{{ '%.0f'|format(w*100) }}%</strong><span style="color:#999;"> (eq {{ '%.0f'|format(eq_pct) }}%)</span>{% if not loop.last %}, {% endif %}
          {% endfor %}
        </td>
      </tr>
      {% endfor %}
    </table>
    </div>
    {% if result.warnings %}
      <div class="warn">{{ result.warnings|join(' · ') }}</div>
    {% endif %}
  {% endif %}

  <div class="caveats">
    <strong>What this simulation is, and isn't:</strong> it runs the SAME rules as
    live portfolios (rulebook.py: score, keep band, stops, re-entry block, sizing,
    sub-industry and beta limits) on price history available at each date, net of
    trading costs at {{ cost_bps|int }} bps per side. It can't replay three things:
    the LLM's read and news verdicts (news is assumed clean, and the model's weight
    is shared across the other factors), the earnings block, and point-in-time
    fundamentals (quality, EPS and industry are today's, a mild lookahead). The
    stock list is today's, so companies that failed or were removed are missing
    (survivorship bias), which flatters results. No taxes. Not a guarantee, forecast
    or audited track record.
  </div>

</body>
</html>
"""
SIMULATE_PAGE = SIMULATE_PAGE.replace("<body>", "<body>\n" + TOP_NAV, 1)


@app.route("/simulate", methods=["GET", "POST"])
def simulate_route():
    from datetime import date, timedelta
    default_start = (date.today() - timedelta(days=365)).isoformat()
    default_end = date.today().isoformat()

    selected = request.form.getlist("sectors") or ["tech_ai"]
    start = request.form.get("start", default_start)
    end = request.form.get("end", default_end)
    capital = float(request.form.get("capital", DEFAULT_CAPITAL) or DEFAULT_CAPITAL)
    rebalance = request.form.get("rebalance", "21")
    top_n = int(float(request.form.get("top_n", TOP_N_TOTAL) or TOP_N_TOTAL))
    top_n = max(1, min(top_n, len(UNIVERSE)))

    result, chart, error, rebalance_history = None, "", None, []
    if request.method == "POST":
        try:
            from simulate import fetch_price_series, rulebook_lookback_days, run_simulation
            symbols = [s for s, path, _ in UNIVERSE if path.split(".")[-1] in selected]
            # SPY fetched alongside the sector symbols in the SAME call — the
            # real, tradable S&P 500 comparison baseline (see
            # SimulationResult.spy_values' docstring for why SPY, not the
            # raw index). run_simulation() skips it gracefully if this
            # window has no SPY history.
            # The rules need a year of history BEFORE the first decision.
            history_start = (date.fromisoformat(start) - timedelta(days=rulebook_lookback_days())).isoformat()
            series = fetch_price_series(symbols + ["SPY"], history_start, end, _ALPACA_API_KEY, _ALPACA_SECRET_KEY)
            result = run_simulation(UNIVERSE, selected, series, capital,
                                    all_sectors=list(SECTORS.keys()), top_n=top_n, start=start,
                                    rebalance_every_days=int(rebalance), max_per_name=_max_per_name_for(top_n))
            chart = svg_multi_line_chart(result.dates, [
                ("AI strategy", result.strategy_values, "#4a6cf7"),
                ("Equal-weight buy & hold", result.benchmark_values, "#999"),
                ("S&P 500 (SPY)", result.spy_values, "#d97706"),
            ])
            rebalance_history = list(zip(result.rebalance_dates, result.rebalance_weights,
                                        result.rebalance_tilts, result.rebalance_actions))
        except Exception as e:
            error = str(e)

    return render_template_string(
        SIMULATE_PAGE, sectors=SECTORS, selected=selected, start=start, end=end,
        capital=capital, rebalance=rebalance, rebalance_options=REBALANCE_OPTIONS,
        top_n=top_n, universe_size=len(UNIVERSE), active_page="simulate",
        result=result, chart=chart, error=error, rebalance_history=rebalance_history,
        cost_bps=rulebook.default_config().cost_bps,
    )


STOCK_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian — {{ symbol }}</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 720px;
        margin: 0 auto; padding: 32px 16px 64px; color: #0d1117; line-height: 1.5; }
  h1 { font-size: 1.6rem; margin-bottom: 2px; }
  .subtitle { color: #6a7280; margin-bottom: 20px; }
  .period-row { display: flex; gap: 10px; margin: 14px 0 20px; flex-wrap: wrap; }
  .period { background: #f6f8ff; border-radius: 10px; padding: 8px 14px; font-size: 0.85rem; }
  .period .val { font-weight: 700; }
  .period .pos { color: #00a340; } .period .neg { color: #d43f3f; }
  h2 { font-size: 1.05rem; margin: 26px 0 10px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 6px 4px; border-bottom: 1px solid #eee; font-size: 0.92rem; }
  th { color: #888; font-weight: 500; width: 45%; }
  .summary { font-size: 0.9rem; color: #333; margin-top: 8px; }
  .warn { color: #a3312f; background: #fee; padding: 12px 14px; border-radius: 10px; margin: 12px 0; }
  .meta { color: #999; font-size: 0.82rem; margin-top: 28px; }
  @media (max-width: 600px) { body { padding: 20px 14px 56px; } h1 { font-size: 1.3rem; } }
</style>
</head>
<body>
  <h1>{{ symbol }}{% if name and name != symbol %} — {{ name }}{% endif %}</h1>
  <div class="subtitle">{% if sector %}{{ sector }}{% endif %}{% if industry %} · {{ industry }}{% endif %}</div>

  {% if error %}<div class="warn">{{ error }}</div>{% endif %}

  {{ chart|safe }}

  {% if periods %}
  <div class="period-row">
    {% for label, val in periods.items() %}
      {% if val is not none %}
      <div class="period">{{ label }}: <span class="val {{ 'pos' if val >= 0 else 'neg' }}">{{ '%+.1f'|format(val*100) }}%</span></div>
      {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {% if fundamentals %}
  <h2>Fundamentals</h2>
  <table>
    {% for label, val in fundamentals.items() %}
      {% if val is not none %}
      <tr><th>{{ label }}</th><td>{{ val }}</td></tr>
      {% endif %}
    {% endfor %}
  </table>
  {% endif %}

  {% if summary %}
  <h2>About</h2>
  <div class="summary">{{ summary }}</div>
  {% endif %}

  <p class="meta">
    Price history from Alpaca (real, live) — via Alpaca's free IEX feed, not the full
    consolidated market tape, so quotes can differ slightly from other sources. Fundamentals
    from Yahoo Finance via yfinance — a free, unofficial data source with no formal SLA;
    treat figures as directional, not audited. Not investment advice.
  </p>
</body>
</html>
"""
STOCK_PAGE = STOCK_PAGE.replace("<body>", "<body>\n" + TOP_NAV, 1)


def _format_fundamental(key: str, value) -> Any:
    if value is None:
        return None
    if key in ("Market Cap",):
        return f"${value / 1e9:,.1f}B" if value >= 1e9 else f"${value / 1e6:,.1f}M"
    if key == "Dividend Yield":
        # VERIFIED against real live data: unlike most other yfinance ratio
        # fields, this one already comes back AS a percentage number (0.46
        # means 0.46%, confirmed against NVDA/JPM/XOM's actual known
        # yields) — multiplying by 100 here would misreport it 100x too
        # high. yfinance's units are genuinely inconsistent field-to-field;
        # this was checked live, not assumed.
        return f"{value:.2f}%" if isinstance(value, (int, float)) else value
    if key in ("Profit Margin", "Revenue Growth"):
        return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else value
    if key in ("Trailing P/E", "Forward P/E", "Beta"):
        return f"{value:.1f}" if isinstance(value, (int, float)) else value
    if key in ("52-Week High", "52-Week Low"):
        return f"${value:.2f}"
    return value


@app.route("/stock/<symbol>")
def stock_detail(symbol: str):
    symbol = symbol.upper()
    error = None
    chart = ""
    periods: Dict[str, Optional[float]] = {}
    fundamentals: Dict[str, Any] = {}
    name, sector, industry, summary = symbol, None, None, None

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment
        from datetime import datetime, timedelta, timezone

        sc = apply_default_timeout(StockHistoricalDataClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY))
        start = datetime.now(timezone.utc) - timedelta(days=190)
        # BUG FIX, confirmed live: without adjustment=ALL, a stock split
        # inside the window shows up as a fake single-day crash (CRWD:
        # ~75% "drop" on its real split date, a raw/unadjusted-price
        # artifact, not a real market move) — this is exactly the
        # discrepancy reported against this chart.
        bars = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                                  start=start, adjustment=Adjustment.ALL))
        series = [(b.timestamp.strftime("%Y-%m-%d"), float(b.close)) for b in bars[symbol]]
        dates, prices = [d for d, _ in series], [p for _, p in series]
        chart = svg_price_chart(dates, prices)

        def ret(n: int) -> Optional[float]:
            if len(prices) <= n or prices[-n - 1] == 0:
                return None
            return (prices[-1] - prices[-n - 1]) / prices[-n - 1]
        periods = {"1 week": ret(5), "1 month": ret(21), "3 months": ret(63),
                  "Full period": ret(len(prices) - 1) if prices else None}
    except Exception as e:
        error = f"Price history unavailable: {e}"

    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        name = info.get("shortName") or symbol
        sector = info.get("sector")
        industry = info.get("industry")
        summary = info.get("longBusinessSummary")
        raw_fundamentals = {
            "Sector": info.get("sector"), "Industry": info.get("industry"),
            "Market Cap": info.get("marketCap"), "Trailing P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"), "Dividend Yield": info.get("dividendYield"),
            "Beta": info.get("beta"), "52-Week High": info.get("fiftyTwoWeekHigh"),
            "52-Week Low": info.get("fiftyTwoWeekLow"), "Profit Margin": info.get("profitMargins"),
            "Revenue Growth": info.get("revenueGrowth"),
        }
        fundamentals = {k: _format_fundamental(k, v) for k, v in raw_fundamentals.items()}
    except Exception as e:
        if not error:
            error = f"Fundamentals unavailable: {e}"

    return render_template_string(
        STOCK_PAGE, symbol=symbol, name=name, sector=sector, industry=industry, active_page="stock",
        chart=chart, periods=periods, fundamentals=fundamentals, summary=summary, error=error,
    )


PORTFOLIO_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian — Portfolio</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 780px;
        margin: 0 auto; padding: 32px 16px 64px; color: #0d1117; line-height: 1.45; }
  h1 { font-size: 1.3rem; margin-bottom: 4px; }
  .paper-banner { background: #fff8e6; border-radius: 10px; padding: 8px 14px; font-size: 0.82rem;
                 margin-bottom: 20px; font-weight: 600; display: inline-block; }
  .stat-row { display: flex; gap: 16px; margin: 16px 0 24px; flex-wrap: wrap; }
  .stat { background: #f6f8ff; border-radius: 12px; padding: 14px 18px; flex: 1; min-width: 140px; }
  .stat .big { font-size: 1.4rem; font-weight: 800; }
  h2 { font-size: 1.05rem; margin: 26px 0 10px; }
  .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  table { width: 100%; min-width: 560px; border-collapse: collapse; font-size: 0.9rem; }
  th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid #eee; white-space: nowrap; }
  th { color: #888; font-weight: 500; }
  .pos { color: #00a340; } .neg { color: #d43f3f; }
  .warn { color: #a3312f; background: #fee; padding: 12px 14px; border-radius: 10px; margin: 12px 0; }
  .meta { color: #999; font-size: 0.82rem; margin-top: 28px; }
  @media (max-width: 600px) { body { padding: 20px 14px 56px; } .stat { min-width: 100%; } }
</style>
</head>
<body>
  <h1>Your Portfolio</h1>
  {{ selector|safe }}
  <div class="paper-banner">📝 Alpaca PAPER account — not real money.</div>

  {% if error %}<div class="warn">{{ error }}</div>{% endif %}

  {% if account %}
  <div class="stat-row">
    <div class="stat"><div>Total Equity</div><div class="big">${{ '{:,.2f}'.format(account.equity) }}</div></div>
    <div class="stat"><div>Cash</div><div class="big">${{ '{:,.2f}'.format(account.cash) }}</div></div>
    <div class="stat"><div>Buying Power</div><div class="big">${{ '{:,.2f}'.format(account.buying_power) }}</div></div>
  </div>
  {% endif %}

  {% if chart %}
    <h2>Account value over time</h2>
    {{ chart|safe }}
  {% endif %}

  {% if positions %}
  <h2>Current positions ({{ positions|length }})</h2>
  <div class="table-wrap">
  <table>
    <tr><th>Symbol</th><th>Qty</th><th>Avg entry</th><th>Current</th><th>Market value</th><th>Unrealized P&amp;L</th></tr>
    {% for p in positions %}
    <tr>
      <td><a href="/stock/{{ p.symbol }}">{{ p.symbol }}</a></td>
      <td>{{ p.qty }}</td>
      <td>${{ '%.2f'|format(p.avg_entry_price) }}</td>
      <td>{% if p.current_price %}${{ '%.2f'|format(p.current_price) }}{% endif %}</td>
      <td>${{ '{:,.2f}'.format(p.market_value) }}</td>
      <td class="{{ 'pos' if p.unrealized_pl >= 0 else 'neg' }}">
        {{ '%+.2f'|format(p.unrealized_pl) }} ({{ '%+.1f'|format(p.unrealized_plpc*100) }}%)</td>
    </tr>
    {% endfor %}
    <tr style="font-weight:700;border-top:2px solid #ddd;">
      <td>Total</td><td></td><td></td><td></td>
      <td>${{ '{:,.2f}'.format(positions|sum(attribute='market_value')) }}</td>
      <td class="{{ 'pos' if positions|sum(attribute='unrealized_pl') >= 0 else 'neg' }}">
        {{ '%+.2f'|format(positions|sum(attribute='unrealized_pl')) }}</td>
    </tr>
  </table>
  </div>
  {% elif not error %}
  <p>No open positions yet — accept a recommendation from the home page to place your first trades.</p>
  {% endif %}

  <p class="meta">Account/position/history data is fetched live from Alpaca — this is not a
    separately-stored performance log, so figures reflect right now, and the chart reflects
    Alpaca's own portfolio history for your account.</p>
</body>
</html>
"""
PORTFOLIO_PAGE = PORTFOLIO_PAGE.replace("<body>", "<body>\n" + TOP_NAV, 1)


@app.route("/portfolio")
def portfolio_route():
    error = None
    account: Dict[str, float] = {}
    positions: List[Dict[str, Any]] = []
    chart = ""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        from datetime import datetime

        tc = apply_default_timeout(TradingClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY, paper=True))
        acct = tc.get_account()
        account = {
            "equity": float(acct.equity), "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
        }

        for p in tc.get_all_positions():
            positions.append({
                "symbol": p.symbol, "qty": p.qty,
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price else None,
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            })
        positions.sort(key=lambda p: -p["market_value"])

        hist = tc.get_portfolio_history(GetPortfolioHistoryRequest(period="1M", timeframe="1D"))
        pairs = [(t, e) for t, e in zip(hist.timestamp or [], hist.equity or []) if e]
        if pairs:
            dates = [datetime.fromtimestamp(t).strftime("%Y-%m-%d") for t, _ in pairs]
            chart = svg_price_chart(dates, [e for _, e in pairs])
    except Exception as e:
        error = f"Could not load portfolio: {e}"

    selector = _portfolio_selector(_all_portfolios_enriched(), "/portfolio")
    return render_template_string(PORTFOLIO_PAGE, account=account, positions=positions,
                                  chart=chart, error=error, active_page="portfolios", selector=selector)


PORTFOLIOS_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian — Compare Portfolios</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 860px;
        margin: 0 auto; padding: 32px 16px 64px; color: #0d1117; line-height: 1.45; }
  h1 { font-size: 1.3rem; margin-bottom: 4px; }
  .sub { color: #6a7280; font-size: 0.9rem; margin-bottom: 20px; }
  .paper-banner { background: #fff8e6; border-radius: 10px; padding: 8px 14px; font-size: 0.82rem;
                 margin-bottom: 20px; font-weight: 600; display: inline-block; }
  .warn { color: #a3312f; background: #fee; padding: 12px 14px; border-radius: 10px; margin: 12px 0; }
  .card { border: 1px solid #eee; border-radius: 12px; padding: 16px 18px; margin: 12px 0; }
  .card-top { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; }
  .pname { font-size: 1.05rem; font-weight: 700; }
  .status { font-size: 0.75rem; font-weight: 700; padding: 2px 9px; border-radius: 10px; margin-left: 8px; }
  .status.open { background: #e6f9ec; color: #00a340; }
  .status.closed { background: #eef0f5; color: #666; }
  .ret { font-size: 1.3rem; font-weight: 800; }
  .pos { color: #00a340; } .neg { color: #d43f3f; }
  .bar-track { background: #f0f0f5; height: 8px; border-radius: 4px; margin: 10px 0; overflow: hidden; }
  .bar { height: 8px; }
  .bar.pos { background: #00c805; } .bar.neg { background: #d43f3f; }
  .meta-row { color: #666; font-size: 0.85rem; margin-top: 4px; }
  .holdings { font-size: 0.85rem; color: #555; margin-top: 8px; }
  .holdings span { display: inline-block; margin: 2px 8px 2px 0; padding: 2px 8px;
                   background: #f6f8ff; border-radius: 8px; }
  .meta-row code { background: #f0f0f5; padding: 1px 6px; border-radius: 5px; font-size: 0.85em; }
  .close-form { margin-top: 10px; }
  button.close-btn { background: #fee; color: #a3312f; border: 1.5px solid #f3c; font-size: 0.8rem;
                     padding: 6px 12px; border-radius: 8px; cursor: pointer; font-weight: 600; }
  button.close-btn { border-color: #f3d4d4; }
  button.close-btn:hover { background: #fdd; }
  .meta { color: #999; font-size: 0.82rem; margin-top: 28px; }
  @media (max-width: 600px) {
    body { padding: 20px 14px 56px; }
    .card-top { flex-direction: column; align-items: flex-start; gap: 4px; }
  }
</style>
</head>
<body>
  <h1>Compare Your Portfolios</h1>
  {{ selector|safe }}
  <div class="sub">Every "Launch as New Portfolio" click from the home page creates one of these —
    each tracked independently by its own entry prices, so returns compare fairly even though real
    fills all land in one shared Alpaca paper account.</div>
  <div class="paper-banner">📝 Alpaca PAPER account — not real money.</div>

  {% if error %}<div class="warn">{{ error }}</div>{% endif %}

  {% if not portfolios %}
    <p>No tracked portfolios yet — go to the <a href="/">home page</a>, get a recommendation,
      and click "Launch as New Portfolio."</p>
  {% endif %}

  {% for p in portfolios %}
  <div class="card">
    <div class="card-top">
      <div><a class="pname" href="/portfolios/{{ p.id }}" style="text-decoration:none;color:inherit;">{{ p.name }}</a>
        <span class="status {{ p.status }}">{{ p.status }}</span></div>
      <span class="ret {{ 'pos' if p.return_pct >= 0 else 'neg' }}">{{ '%+.1f'|format(p.return_pct) }}%</span>
    </div>
    <div class="bar-track"><div class="bar {{ 'pos' if p.return_pct >= 0 else 'neg' }}"
         style="width: {{ p.bar_width }}%"></div></div>
    <div class="meta-row">
      Launched {{ p.created_at[:10] }} · {{ p.sectors|join(', ') }} · top {{ p.top_n }} names total
      · capital ${{ '{:,.0f}'.format(p.capital) }} · now worth ${{ '{:,.2f}'.format(p.mtm_value) }}
      {% if p.status == 'closed' %}· closed {{ p.closed_at[:10] }}{% endif %}
    </div>
    {% if p.holdings_display %}
    <div class="holdings">
      {% for h in p.holdings_display %}
      <span>{{ h.symbol }}: {{ h.shares }} sh @ ${{ '%.2f'|format(h.entry_price) }}
        {% if h.current_price %}→ ${{ '%.2f'|format(h.current_price) }}{% endif %}</span>
      {% endfor %}
    </div>
    <div class="meta-row">To verify these real fills yourself in Alpaca's own dashboard
      (<a href="https://app.alpaca.markets/paper/dashboard/orders" target="_blank" rel="noopener">app.alpaca.markets → Orders</a>),
      search for order IDs starting with <code>webui-{{ p.id }}-</code> — that prefix is unique to
      this portfolio, so every real order it placed shares it.</div>
    {% endif %}
    {% if p.errors %}
    <div class="meta-row" style="color:#a3312f;">{{ p.errors|length }} order error(s) at launch —
      this portfolio's cost basis reflects only what actually filled.</div>
    {% endif %}
    {% if p.pending_order_ids %}
    <div class="meta-row">{{ p.pending_order_ids|length }} order(s) still pending fill from launch
      (placed after market hours, or not yet matched) — closing will cancel them.</div>
    {% endif %}
    {% if p.status == 'open' %}
    <form class="close-form" method="post" action="/portfolios/{{ p.id }}/close"
          onsubmit="return confirm('Sell all {{ p.holdings_display|length }} holdings' + ({{ (p.pending_order_ids|length)|default(0) }} ? ' and cancel {{ p.pending_order_ids|length }} pending order(s)' : '') + ' in {{ p.name }} for real on Alpaca and close this portfolio?');">
      <button type="submit" class="close-btn">Close &amp; Liquidate</button>
    </form>
    {% endif %}
  </div>
  {% endfor %}

  <p class="meta">Returns are computed from each portfolio's OWN recorded entry prices and live
    market quotes — independent of every other portfolio and of the raw combined Alpaca account
    view. Alpaca itself only ever shows one combined position per symbol; this comparison is a
    ledger this app keeps on your behalf, not something Alpaca tracks natively.</p>
</body>
</html>
"""
PORTFOLIOS_PAGE = PORTFOLIOS_PAGE.replace("<body>", "<body>\n" + TOP_NAV, 1)


def _mark_to_market(p: Dict[str, Any], live: Dict[str, float]) -> Dict[str, Any]:
    holdings = p.get("holdings") or {}
    holdings_value = 0.0
    holdings_display = []
    for symbol, h in holdings.items():
        current = live.get(symbol)
        row_value = (current or h["entry_price"]) * h["shares"]
        holdings_value += row_value
        holdings_display.append({"symbol": symbol, "shares": h["shares"],
                                 "entry_price": h["entry_price"], "current_price": current,
                                 "value": row_value})
    holdings_display.sort(key=lambda h: h["symbol"])
    mtm_value = holdings_value + p.get("cash_remaining", 0.0)
    capital = p.get("capital") or 1.0
    return_pct = (mtm_value - capital) / capital * 100
    return {
        **p, "holdings_display": holdings_display, "mtm_value": mtm_value,
        "return_pct": return_pct,
        # Explicit .get(), not just whatever **p carries: portfolios created
        # before this field existed have NO "estimated_1mo_return" key at
        # all in Firestore, and Jinja's `p.estimated_1mo_return is not none`
        # would otherwise hit its Undefined fallback (which is NOT equal to
        # None) instead of cleanly skipping the comparison stat card.
        "estimated_1mo_return": p.get("estimated_1mo_return"),
        # Bar width scaled to +/-50% so typical moves are readable without
        # one outlier squashing every other bar to a sliver.
        "bar_width": min(100.0, abs(return_pct) / 50 * 100),
    }


@app.route("/portfolios")
def portfolios_route():
    error = None
    enriched = []
    try:
        enriched = _all_portfolios_enriched()
    except Exception as e:
        error = f"Could not load portfolios: {e}"

    # The Portfolio tab's job is to show ONE portfolio's real performance
    # over time vs SPY, not a summary list — landing here (from the top
    # nav, with no explicit ?view=all) redirects straight to the most
    # recently created OPEN portfolio's own time-series page. The
    # "Compare All" dropdown option links here WITH ?view=all specifically
    # to bypass this and reach the list below.
    #
    # Closing/liquidating a portfolio removes it from this tab entirely —
    # both this list and the dropdown selector below only ever show OPEN
    # portfolios now. A closed one is no longer "tracked" (nothing to
    # recalibrate, no ongoing SPY comparison you'd check day to day); it's
    # still reachable directly by its /portfolios/<id> URL if you have it,
    # just not surfaced here for ongoing navigation.
    open_ones = [p for p in enriched if p.get("status") == "open"]

    if request.args.get("view") != "all" and not error:
        if open_ones:
            return redirect(f"/portfolios/{open_ones[0]['id']}")
        # nothing open to redirect to — fall through to the (now empty,
        # closed-filtered) list below rather than landing on a closed
        # portfolio's detail page.

    selector = _portfolio_selector(open_ones, "/portfolios?view=all")
    return render_template_string(PORTFOLIOS_PAGE, portfolios=open_ones, error=error,
                                  active_page="portfolios", selector=selector)


@app.route("/portfolios/<portfolio_id>/close", methods=["POST"])
def close_portfolio_route(portfolio_id: str):
    import portfolio_store
    from alpaca.trading.client import TradingClient
    from alpaca_clients import AlpacaBrokerClient
    from execution_workflow import Order, Side

    p = portfolio_store.get_portfolio(portfolio_id)
    if p is None or p.get("status") != "open":
        selector = _portfolio_selector(_all_portfolios_enriched(), "/portfolios?view=all")
        return render_template_string(PORTFOLIOS_PAGE, portfolios=[], error="Portfolio not found or already closed.",
                                      active_page="portfolios", selector=selector)

    # Resolve anything that filled since the ledger was written BEFORE
    # deciding what to sell — otherwise this sells stale share counts.
    if p.get("pending_order_ids") and _reconcile_portfolio_orders(portfolio_id):
        p = portfolio_store.get_portfolio(portfolio_id) or p

    tc = TradingClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY, paper=True)
    broker = AlpacaBrokerClient(tc)

    # Cancel anything still pending from launch FIRST — otherwise an order
    # placed after hours (accepted but unfilled at launch) could fill later
    # into a real position this "closed" portfolio no longer tracks.
    for symbol, launch_client_order_id in (p.get("pending_order_ids") or {}).items():
        try:
            broker.cancel(Order(symbol=symbol, side=Side.BUY, notional=0.0,
                                client_order_id=launch_client_order_id))
        except Exception as e:
            print(f"[web_ui] could not cancel pending order {launch_client_order_id}: {e}")

    exit_holdings, proceeds = {}, 0.0
    for symbol, h in (p.get("holdings") or {}).items():
        client_order_id = f"webui-close-{portfolio_id}-{symbol}-SELL"[:48]
        order = Order(symbol=symbol, side=Side.SELL, notional=0.0,
                     qty=h["shares"], client_order_id=client_order_id)
        try:
            fill = broker.submit_and_await(order)
            # Same fix as execute()'s BUY side: a genuine PARTIAL sell means
            # fewer shares actually sold than held — dividing the (smaller)
            # real proceeds by the FULL original holding would understate
            # exit_price. A partial sell here also leaves a real residual
            # Alpaca position this now-"closed" portfolio no longer tracks;
            # that's a genuine limitation (same class as the pending-order
            # risk already handled above), not something fixable after the
            # fact, but at least exit_price/proceeds stay accurate for
            # whatever actually sold.
            real_sold = fill.filled_qty if fill.filled_qty else h["shares"]
            exit_price = (fill.filled_notional / real_sold) if fill.filled_notional and real_sold else None
            exit_holdings[symbol] = {"shares": h["shares"], "sold_shares": real_sold, "exit_price": exit_price,
                                     "result": fill.result.value.upper()}
            proceeds += fill.filled_notional or 0.0
        except Exception as e:
            exit_holdings[symbol] = {"shares": h["shares"], "exit_price": None, "result": "ERROR: " + str(e)}

    portfolio_store.close_portfolio(portfolio_id, exit_holdings, proceeds)
    return portfolios_route()


def _drop_and_resize(weights: Dict[str, float], exclude: Sequence[str], n: int, cap: float) -> Dict[str, float]:
    """`weights` without the `exclude` names, cut to the n largest and scaled
    back up to the same total, then re-capped at `cap` (any excess spread
    pro rata over the names still under it; what nobody can take stays cash)."""
    budget = sum(weights.values())
    keep = sorted(((w, s) for s, w in weights.items() if s not in set(exclude) and w > 0), reverse=True)[:n]
    total = sum(w for w, _ in keep)
    if total <= 0:
        return {}
    out = {s: w * budget / total for w, s in keep}
    capped: set = set()
    for _ in range(len(out)):
        over = {s: w - cap for s, w in out.items() if w > cap + 1e-12}
        if not over:
            break
        capped |= set(over)
        for s in over:
            out[s] = cap
        under = {s: w for s, w in out.items() if s not in capped}
        room = sum(under.values())
        if room <= 0:
            break
        for s, w in under.items():
            out[s] = w + sum(over.values()) * w / room
    return out


def _place_orders(portfolio_id: str, plan: List[Dict[str, Any]], vetoed: Dict[str, str], holdings: Dict[str, Dict[str, Any]],
                  rebalance_date: str, broker: Any) -> tuple:
    """Places `plan` in order. Returns (holdings afterwards, orders, errors, orders
    still pending). Anything not traded carries over untouched."""
    from execution_workflow import Order, Side
    new_holdings: Dict[str, Dict[str, Any]] = dict(holdings)
    orders_placed: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    # Orders accepted but not yet resolved — same tracking
    # close_portfolio_route() relies on for launch-time orders, so a future
    # recalibration doesn't place a duplicate of a still-pending order, and
    # closing the portfolio cancels these too instead of leaving them live.
    new_pending: Dict[str, str] = {}
    for o in plan:
        sym, qty, price, current_shares = o["symbol"], o["qty"], o["price"], o["current_shares"]
        if o["side"] == "BUY" and sym in vetoed:        # the evaluator blocked this buy
            continue
        side = Side.BUY if o["side"] == "BUY" else Side.SELL
        # Idempotency key includes the DATE, not just portfolio+symbol, so a
        # retried recalibration on the SAME day reuses this order, but the
        # NEXT scheduled recalibration (a different date) gets its own —
        # otherwise every future rebalance of this symbol would collide
        # with the very first one's client_order_id forever.
        client_order_id = f"webui-rc-{portfolio_id}-{rebalance_date}-{sym}-{side.value}"[:48]
        order = Order(symbol=sym, side=side, notional=qty * price, qty=qty, client_order_id=client_order_id)
        try:
            fill = broker.submit_and_await(order)
            fill_result = fill.result.value.upper()
            # BUG FIX: a genuine PARTIAL fill means fewer shares actually
            # traded than `qty` requested. Derive the real resulting share
            # count from fill.filled_qty rather than the intended target.
            real_qty = fill.filled_qty if fill.filled_qty else qty
            traded = fill_result in ("FILLED", "PARTIAL")
            orders_placed.append({"symbol": sym, "side": side.value.upper(), "qty": qty, "result": fill_result,
                                  "price": price,
                                  "fill_price": (fill.filled_notional / real_qty) if traded and fill.filled_notional and real_qty else None,
                                  "notional": (fill.filled_notional or real_qty * price) if traded else None})
            if not traded:
                if fill_result == "TIMEOUT":
                    new_pending[sym] = client_order_id
                continue
            actual_shares = current_shares + real_qty if side == Side.BUY else current_shares - real_qty
            if actual_shares <= 0:
                new_holdings.pop(sym, None)             # fully sold
                continue
            added_cost = fill.filled_notional or (real_qty * price)
            if side == Side.BUY and current_shares > 0:
                # Blend cost basis and entry date: kept shares' + newly bought shares'.
                old = holdings[sym]
                old_cost = current_shares * old["entry_price"]
                entry_price = (old_cost + added_cost) / actual_shares
                entry_date = rulebook.blended_entry_date(old.get("entry_date"), old_cost, rebalance_date, added_cost)
            elif side == Side.BUY:
                entry_price = added_cost / real_qty if real_qty else price
                entry_date = rebalance_date
            else:   # partial or full SELL, shares remain: cost basis per share is unchanged
                entry_price = holdings[sym]["entry_price"]
                entry_date = holdings[sym].get("entry_date")
            new_holdings[sym] = {"shares": actual_shares, "entry_price": entry_price,
                                 "cost": actual_shares * entry_price, "entry_date": entry_date}
        except Exception as e:
            print(f"[web_ui] order failed for {sym}: {e}")
            errors.append({"symbol": sym, "error": str(e)})
    return new_holdings, orders_placed, errors, new_pending


def _holdings_after(holdings: Dict[str, Dict[str, Any]], plan: List[Dict[str, Any]], pending: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Holdings right after trading (orders still filling counted at their size):
    the Portfolio page's "what changed" column reads this."""
    return {**{s: {"shares": h["shares"]} for s, h in holdings.items()},
            **{o["symbol"]: {"shares": o["current_shares"] + (o["qty"] if o["side"] == "BUY" else -o["qty"])}
               for o in plan if o["symbol"] in pending}}


def _signals_of(names: Sequence[str], notes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """What drove each pick; the scorecard attributes outcomes to these."""
    return {s: {k: (notes.get(s) or {}).get(k) for k in
                ("dip_score", "drawdown_6m", "momentum_1m", "momentum_12_1", "conviction", "rating", "quality_score")}
            for s in names}


def _targeted_exit(p: Dict[str, Any], exits: Sequence[str], trigger: str, reason: str, rules: rulebook.RulebookConfig,
                   today: Any, live: Dict[str, float], current_value: float, n_stocks: int) -> Dict[str, Any]:
    """A stop or the news review sells only `exits`, and everything else stays.

    THE SALE COMES FIRST and is recorded as soon as it happens: it needs nothing
    from the agent, so a slow or failed recommendation can never keep a stock the
    rules said to sell. Finding replacements comes second. If that fails, the money
    stays in cash until the next recalibration re-fills the slots."""
    import portfolio_store
    from datetime import datetime, timedelta, timezone
    from alpaca.trading.client import TradingClient
    from alpaca_clients import AlpacaBrokerClient

    portfolio_id = p["id"]
    old_holdings = p.get("holdings") or {}
    rebalance_date = today.isoformat()
    broker = AlpacaBrokerClient(TradingClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY, paper=True))

    def price_of(sym: str, h: Optional[Dict[str, Any]] = None) -> float:
        return float(prices.get(sym) or (h or {}).get("entry_price") or 0.0)

    def cash_for(holdings: Dict[str, Dict[str, Any]]) -> float:
        return max(0.0, current_value - sum(price_of(s, h) * h["shares"] for s, h in holdings.items()))

    prices = dict(live)
    sells = [{"symbol": s, "side": "SELL", "qty": old_holdings[s]["shares"], "price": price_of(s, old_holdings[s]),
              "current_shares": old_holdings[s]["shares"]} for s in exits if s in old_holdings]
    holdings, orders, errors, pending = _place_orders(portfolio_id, sells, {}, old_holdings, rebalance_date, broker)
    if not orders:
        return {"error": f"Could not sell {', '.join(exits)}: " + ("; ".join(e["error"] for e in errors) or "not held")}

    sold = [s for s in exits if s in old_holdings and s not in holdings]
    current = {s: float(h.get("shares") or 0) * price_of(s, h) / current_value
               for s, h in old_holdings.items()} if current_value > 0 else {}
    cadence = p.get("rebalance_every_days", 0) or 0
    next_date = (datetime.now(timezone.utc) + timedelta(days=cadence)).date().isoformat() if cadence > 0 else None
    portfolio_store.record_rebalance(
        portfolio_id, holdings, cash_for(holdings), pending,
        event={"date": rebalance_date, "at": datetime.now(timezone.utc).isoformat(), "orders": orders, "errors": errors,
               "portfolio_value_before": current_value,
               "target_weights": {s: w for s, w in current.items() if s in holdings},
               "trigger": trigger, "reason": reason[:1000], "review": {}, "excluded": list(exits),
               "holdings_after": _holdings_after(holdings, sells, pending), "signals": {},
               "replacement": "looking"},
        next_rebalance_date=next_date,
    )
    out: Dict[str, Any] = {"orders": orders, "errors": errors, "portfolio_value_before": current_value}

    # ---- replacements. Nothing below can undo the sale.
    try:
        if not sold:
            raise ValueError("the sale hasn't filled yet, so there's no money to reinvest")
        blocked = sorted(set(rulebook.reentry_blocked(p, today, rules)) | set(exits))
        result = recommend(p["sectors"], n_stocks, {"held": sorted(holdings), "blocked": blocked})
        notes = result.get("book_notes") or {}
        weights = {s: w for s, w in (result.get("final_weights") or {}).items() if s not in blocked}
        fresh = sorted((s for s in weights if s not in old_holdings),
                       key=lambda s: (-float((notes.get(s) or {}).get("conviction") or 0.0), -weights[s], s))
        if not fresh:
            raise ValueError(f"no replacement available (status: {result.get('status')}, risk: {result.get('risk_verdict')})")
        names = set(current) | set(fresh)
        target = rulebook.targeted_weights(current, sold, fresh, {s: (notes.get(s) or {}).get("industry") or "" for s in names},
                                           {s: (notes.get(s) or {}).get("beta") for s in names}, rules)
        prices.update(_live_prices(fresh))
        plan: List[Dict[str, Any]] = []
        for s in target:
            if s in old_holdings or not prices.get(s):
                continue
            qty = math.floor(current_value * target[s] / prices[s])
            if qty > 0 and not rulebook.below_min_trade(qty * prices[s], current_value, qty, rules):
                plan.append({"symbol": s, "side": "BUY", "qty": qty, "price": prices[s], "current_shares": 0})
        review = _review_plan(p, plan, target, notes, trigger, reason)          # never raises
        final, bought, buy_errors, buy_pending = _place_orders(portfolio_id, plan, review.get("vetoed") or {}, holdings,
                                                               rebalance_date, broker)
        orders, errors = orders + bought, errors + buy_errors
        pending = {**pending, **buy_pending}
        portfolio_store.update_last_rebalance(
            portfolio_id, final, cash_for(final), pending,
            {"orders": orders, "errors": errors, "target_weights": target, "review": review, "replacement": "done",
             "holdings_after": _holdings_after(final, plan + sells, pending), "signals": _signals_of(list(target), notes)})
        out.update(orders=orders, errors=errors)
    except Exception as e:
        why = f"{type(e).__name__}: {e}"[:300]
        print(f"[web_ui] replacement search failed for {portfolio_id}: {why}")
        portfolio_store.update_last_rebalance(portfolio_id, holdings, cash_for(holdings), pending,
                                              {"replacement": f"none: {why}",
                                               "reason": f"{reason} Sold; no replacement yet, so the slot stays cash "
                                                         f"until the next recalibration."[:1000]})
        out["replacement_error"] = why
    return out


def recalibrate_portfolio(portfolio_id: str, trigger: str = "manual", reason: str = "",
                          exclude: Sequence[str] = ()) -> Dict[str, Any]:
    """The actual recalibration, sized against the portfolio's CURRENT
    mark-to-market value, placing REAL Alpaca orders for the SHARE DELTA between
    what it holds and its new target. Callable manually
    (/portfolios/<id>/recalibrate), by the daily due-check
    (/portfolios/recalibrate_due) or by the news watch
    (/portfolios/proactive_check). `trigger` ("manual" | "scheduled" | "proactive")
    and `reason` are recorded on the history event.

    Follows rulebook.py on every trigger: holdings inside the keep band stay,
    anything sold in the last reentry_block_days isn't bought back, changes too
    small to be worth their cost don't trade, and sells go first. Only runs while
    the market is open, so orders fill at live prices rather than at the next
    open at a price nobody saw.

    `exclude` (the news watch's exits) makes it TARGETED: see _targeted_exit."""
    import portfolio_store
    from datetime import datetime, timedelta, timezone
    from alpaca.trading.client import TradingClient
    from alpaca_clients import AlpacaBrokerClient

    p = portfolio_store.get_portfolio(portfolio_id)
    if p is None or p.get("status") != "open":
        return {"error": "Portfolio not found or already closed."}
    try:
        market_open = _market_open()
    except Exception as e:
        return {"error": f"Could not check market hours ({e}), so nothing was traded."}
    if not market_open:
        return {"error": "The market is closed. Recalibrations only trade during market hours."}

    # Size this rebalance off what the portfolio ACTUALLY holds. An earlier
    # recalibration's orders may have filled since its ledger was written;
    # without this, deltas are computed against stale holdings and (order IDs
    # being date-stamped) a second, duplicate rebalance is placed on top.
    if p.get("pending_order_ids") and _reconcile_portfolio_orders(portfolio_id):
        p = portfolio_store.get_portfolio(portfolio_id) or p

    rules = rulebook.default_config()
    today = datetime.now(timezone.utc).date()
    old_holdings = p.get("holdings") or {}
    live = _live_prices(list(old_holdings.keys()))
    current_value = _mark_to_market(p, live)["mtm_value"]

    # Keep the number of stocks the portfolio was LAUNCHED with: recalibration
    # changes which stocks, never how many. Older portfolios without n_stocks
    # (see backfill_n_stocks.py) keep their current count.
    n_stocks = int(p.get("n_stocks") or len(old_holdings) or p.get("top_n") or TOP_N_TOTAL)
    exclude = sorted({str(s).upper() for s in exclude or ()})
    if exclude:
        return _targeted_exit(p, exclude, trigger, reason, rules, today, live, current_value, n_stocks)

    blocked = sorted(rulebook.reentry_blocked(p, today, rules))
    result = recommend(p["sectors"], n_stocks, {"held": sorted(old_holdings), "blocked": blocked})
    weights = result.get("final_weights") or {}
    if set(weights) & set(blocked):               # the rule also holds if the agent didn't apply it
        weights = _drop_and_resize(weights, blocked, n_stocks, _max_per_name_for(n_stocks))
    notes = result.get("book_notes") or {}
    if not weights:
        return {"error": f"Recalibration produced no recommendation "
                         f"(status: {result.get('status')}, risk: {result.get('risk_verdict')})."}

    all_symbols = sorted(set(old_holdings) | set(weights.keys()))
    new_prices = _live_prices(all_symbols)
    tc = TradingClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY, paper=True)
    broker = AlpacaBrokerClient(tc)

    # ---- the orders: sells first, so a buy never waits on proceeds. Reviewed by
    # the evaluation agent (rec_evaluator.py) before anything is placed.
    plan: List[Dict[str, Any]] = []
    for sym in all_symbols:
        px = new_prices.get(sym) or (old_holdings.get(sym) or {}).get("entry_price")
        cur = (old_holdings.get(sym) or {}).get("shares", 0)
        if not px:
            continue
        tgt = math.floor((current_value * weights.get(sym, 0.0)) / px) if weights.get(sym, 0.0) > 0 else 0
        if tgt == cur or rulebook.below_min_trade((tgt - cur) * px, current_value, tgt, rules):
            continue
        plan.append({"symbol": sym, "side": "BUY" if tgt > cur else "SELL", "qty": abs(tgt - cur),
                     "price": px, "current_shares": cur})
    plan.sort(key=lambda o: (o["side"] != "SELL", o["symbol"]))
    review = _review_plan(p, plan, weights, notes, trigger, reason)          # never raises

    rebalance_date = today.isoformat()
    new_holdings, orders_placed, errors, new_pending = _place_orders(portfolio_id, plan, review.get("vetoed") or {},
                                                                     old_holdings, rebalance_date, broker)

    # BUG FIX, confirmed live: cash must be current_value minus the CURRENT
    # MARKET VALUE of what's held, not what was paid for it, or unrealized gains
    # count as cash and compound on every recalibration.
    new_market_value = sum(
        (new_prices.get(sym) or old_holdings.get(sym, {}).get("entry_price", 0.0)) * h["shares"]
        for sym, h in new_holdings.items()
    )
    new_cash = max(0.0, current_value - new_market_value)
    cadence = p.get("rebalance_every_days", 0) or 0
    next_date = (datetime.now(timezone.utc) + timedelta(days=cadence)).date().isoformat() if cadence > 0 else None

    portfolio_store.record_rebalance(
        portfolio_id, new_holdings, new_cash, new_pending,
        event={"date": rebalance_date, "at": datetime.now(timezone.utc).isoformat(),
               "orders": orders_placed, "errors": errors,
              "portfolio_value_before": current_value, "target_weights": weights,
              "trigger": trigger, "reason": reason[:1000], "review": review, "excluded": [],
              "holdings_after": _holdings_after(new_holdings, plan, new_pending),
              "signals": _signals_of(list(weights), notes)},
        next_rebalance_date=next_date,
    )
    return {"orders": orders_placed, "errors": errors, "portfolio_value_before": current_value}


@app.template_filter("et_time")
def _et_time(iso: str) -> str:
    """ISO timestamp -> 'HH:MM' New York time."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        return datetime.fromisoformat(iso).astimezone(ZoneInfo("America/New_York")).strftime("%H:%M")
    except Exception:
        return ""


@app.route("/portfolios/<portfolio_id>/news_watch", methods=["POST"])
def news_watch_route(portfolio_id: str):
    """The owner's on/off switch for the mid-day news watch. Same trust model
    as "Recalibrate Now" and "Close & Liquidate" on this page (the app is
    deliberately public; the scheduled endpoints are what carry the token)."""
    import portfolio_store
    portfolio_store.set_news_watch(portfolio_id, request.form.get("enabled") == "1")
    return redirect(f"/portfolios/{portfolio_id}")


@app.route("/portfolios/<portfolio_id>/recalibrate", methods=["POST"])
def recalibrate_route(portfolio_id: str):
    """Manual "Recalibrate Now" trigger — the same real logic the scheduled
    daily due-check runs automatically, just invoked on demand."""
    result = recalibrate_portfolio(portfolio_id, trigger="manual")
    return portfolio_detail(portfolio_id, notice=result.get("error"))


_RECALIBRATE_TOKEN = None


def _recalibrate_token() -> Optional[str]:
    """Shared-secret check for the bulk due-check endpoint below. The rest
    of this app is deliberately public (see the session's earlier decision
    to keep it that way) — but that endpoint actually PLACES REAL ORDERS
    for every due portfolio when hit, so unlike every other page here, it
    cannot be left open to anyone who finds the URL. Cloud Scheduler is
    configured to send this same token as a header (see the gcloud scheduler
    job created for this). Returns None (meaning "no check configured") if
    the secret was never set — that's an explicit opt-out, not an accident,
    for local testing without Secret Manager access."""
    global _RECALIBRATE_TOKEN
    if _RECALIBRATE_TOKEN is None:
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            _RECALIBRATE_TOKEN = client.access_secret_version(
                name=f"projects/{PROJECT}/secrets/recalibrate-token/versions/latest"
            ).payload.data.decode("utf-8")
        except Exception as e:
            print(f"[web_ui] no recalibrate-token secret configured ({e}) — due-check endpoint is UNPROTECTED")
            _RECALIBRATE_TOKEN = ""
    return _RECALIBRATE_TOKEN or None


@app.route("/portfolios/recalibrate_due", methods=["POST"])
def recalibrate_due_route():
    """Hit once a day by a real Cloud Scheduler job — finds every OPEN
    portfolio whose next_rebalance_date has arrived and recalibrates each
    for real. This is what makes "prescheduled recalibration" actually
    automatic rather than something you have to remember to click."""
    import portfolio_store
    expected = _recalibrate_token()
    if expected and request.headers.get("X-Recalibrate-Token") != expected:
        return {"error": "unauthorized"}, 403

    due = portfolio_store.list_due_portfolios()
    results = []
    # Run this during market hours, e.g. --schedule "30 10 * * 1-5" --time-zone "America/New_York".
    # With the market closed nothing trades and the portfolio stays due for the next run.
    for p in due:
        try:
            r = recalibrate_portfolio(p["id"], trigger="scheduled")
            results.append({"id": p["id"], "name": p["name"], **r})
        except Exception as e:
            results.append({"id": p["id"], "name": p["name"], "error": str(e)})
    return {"checked": len(due), "results": results}


# =============================================================================
# Mid-day news watch — decides, per open portfolio, whether today's news calls
# for recalibrating NOW rather than waiting for the scheduled date. The logic
# (guardrails, evidence, Claude review, fallback, policy) lives in
# proactive_recal.py; this is only the wiring to Alpaca, Firestore and Claude.
#
# Schedule it once per trading day, mid-session (same token as the due-check):
#   gcloud scheduler jobs create http meridian-proactive-recal --location us-central1 \
#       --schedule "30 12 * * 1-5" --time-zone "America/New_York" \
#       --uri "https://<SERVICE_URL>/portfolios/proactive_check" --http-method POST \
#       --headers "X-Recalibrate-Token=<token>" --attempt-deadline 1800s
# A holiday is handled here, not in the schedule: with the market closed a
# "recalibrate" verdict is only flagged. POST ?dry_run=1 to see what it WOULD do.
# =============================================================================
# =============================================================================
# Evaluation agent — pre-trade review + track-record scorecard (rec_evaluator.py)
# =============================================================================
def _eval_cfg():
    from rec_evaluator import load_config
    return load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandate.yaml"))


def _review_plan(p: Dict[str, Any], plan: List[Dict[str, Any]], weights: Dict[str, float],
                 notes: Dict[str, Dict[str, Any]], trigger: str, reason: str) -> Dict[str, Any]:
    """Pre-trade review for one recalibration. NEVER raises: any failure
    means "no review" (nothing vetoed), never a blocked recalibration."""
    try:
        from rec_evaluator import review_plan, track_record_lines
        cfg = _eval_cfg()
        symbols = sorted({o["symbol"] for o in plan})
        heads: Dict[str, List[str]] = {}
        try:
            heads = _headlines_many(symbols, cfg.red_flag_days) if symbols else {}
        except Exception as e:
            print(f"[web_ui] reviewer headlines unavailable: {e}")
        track: List[str] = []
        try:
            track = track_record_lines(_scorecard()[1], min_trades=cfg.track_record_min_trades)
        except Exception as e:
            print(f"[web_ui] reviewer track record unavailable: {e}")
        client = None
        try:
            client = _claude_client()
        except Exception as e:
            print(f"[web_ui] reviewer model unavailable: {e}")
        return review_plan(p, plan, weights, notes, heads, cfg, client, trigger, reason, track)
    except Exception as e:
        print(f"[web_ui] pre-trade review failed: {e}")
        return {"grade": None, "summary": f"review failed: {e}", "orders": {}, "vetoed": {}, "source": "error"}


def _daily_closes(symbols: List[str], start: Any) -> Dict[str, Dict[Any, float]]:
    """{symbol: {date: close}} from `start` to today, batched."""
    from datetime import datetime, time as dtime, timezone
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment
    sc = apply_default_timeout(StockHistoricalDataClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY))
    out: Dict[str, Dict[Any, float]] = {}
    for i in range(0, len(symbols), 100):
        # split-adjusted: a split inside the window must not look like a crash
        bars = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbols[i:i + 100], timeframe=TimeFrame.Day,
                                                  adjustment=Adjustment.SPLIT,
                                                  start=datetime.combine(start, dtime(0), tzinfo=timezone.utc)))
        for sym, bs in (getattr(bars, "data", None) or {}).items():
            out[sym] = {b.timestamp.date(): float(b.close) for b in bs}
    return out


_SCORECARD: Dict[str, Any] = {"at": -1e9, "value": None}


def _scorecard(max_age: float = 1800.0):
    """(graded decisions, aggregate) over every portfolio's recalibration
    orders, cached for 30 minutes (one batched price request per refresh)."""
    import time
    if _SCORECARD["value"] is not None and time.monotonic() - _SCORECARD["at"] < max_age:
        return _SCORECARD["value"]
    import portfolio_store
    from datetime import date, timedelta
    from rec_evaluator import aggregate, decisions_from_portfolios, grade_decisions, portfolio_metrics
    sector_of = {sym: path.split(".")[-1] for sym, path, _ in UNIVERSE}
    pools: Dict[str, List[str]] = {}
    for sym, sector in sector_of.items():
        pools.setdefault(sector, []).append(sym)
    portfolios = portfolio_store.list_portfolios()
    decisions = decisions_from_portfolios(portfolios, sector_of)
    if not decisions:
        value = ([], aggregate([]))
    else:
        sectors = {d["sector"] for d in decisions}
        syms = {d["symbol"] for d in decisions} | {s for sec in sectors for s in pools.get(sec, [])} | {"SPY"}
        for p in portfolios:
            syms |= set(p.get("initial_holdings") or {}) | set(p.get("holdings") or {})
            for ev in p.get("rebalance_history") or []:
                syms |= set(ev.get("holdings_after") or {})
        starts = [d["date"] for d in decisions]
        for p in portfolios:
            created = rulebook._iso_date(p.get("created_at"))
            if created:
                starts.append(created)
        closes = _daily_closes(sorted(syms), min(starts) - timedelta(days=7))
        graded = grade_decisions(decisions, closes, pools)
        agg = aggregate(graded)
        today = date.today()
        agg["portfolios"] = [m for m in (portfolio_metrics(p, closes, today) for p in portfolios
                                         if p.get("status") == "open") if m]
        value = (graded, agg)
    _SCORECARD.update(at=time.monotonic(), value=value)
    return value


EVALUATION_PAGE = """<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian — Evaluation</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 860px;
        margin: 0 auto; padding: 32px 16px 64px; color: #0d1117; line-height: 1.45; }
  h1 { font-size: 1.3rem; } h2 { font-size: 1.05rem; margin: 26px 0 8px; }
  .meta { color: #6a7280; font-size: 0.86rem; }
  .table-wrap { overflow-x: auto; } table { width: 100%; min-width: 560px; border-collapse: collapse; font-size: 0.86rem; }
  th, td { text-align: left; padding: 6px 6px; border-bottom: 1px solid #eee; } th { color: #888; font-weight: 500; }
  .pos { color: #00a340; } .neg { color: #d43f3f; }
  .warn { color: #a3312f; background: #fee; padding: 12px; border-radius: 10px; }
  .card { background: #f6f8ff; border-radius: 12px; padding: 12px 16px; display: inline-block; margin: 0 8px 8px 0; min-width: 150px; }
  .card .big { font-size: 1.3rem; font-weight: 800; }
</style></head><body>
<h1>🧪 How good have the recommendations been?</h1>
<p class="meta">Every buy and sell placed by a recalibration, graded on what the market did NEXT — the stock vs. its
  own sector (all its universe stocks, equal-weighted) over 1, 5 and 20 trading days. A buy is a hit if the stock beat
  its sector; a sell is a hit if it lagged after we sold. "Value added" is that gap (for sells, the loss avoided).
  Hit rates and averages are weighted by trade size, graded from the fill price where recorded.
  Paper trading. Recalibrations made before signal tracking began show their signal as "unrecorded".
  <a href="/evaluation?refresh=1">Refresh now</a></p>
{% if error %}<div class="warn">{{ error }}</div>{% endif %}
{% macro cell(s) %}{% if s and s.n %}<td>{{ s.n }}</td><td>{{ '{:.0%}'.format(s.hit_rate) }}</td>
  <td class="{{ 'pos' if s.avg_value_added > 0 else 'neg' }}">{{ '{:+.1%}'.format(s.avg_value_added) }}</td>
  {% else %}<td>0</td><td>—</td><td>—</td>{% endif %}{% endmacro %}
{% macro table(title, groups) %}<h2>{{ title }}</h2><div class="table-wrap"><table>
  <tr><th></th>{% for h in horizons %}<th>n ({{ h }}d)</th><th>hit rate</th><th>value added</th>{% endfor %}</tr>
  {% for k, v in groups.items() %}<tr><td><b>{{ k }}</b></td>{% for h in horizons %}{{ cell(v[h]) }}{% endfor %}</tr>
  {% else %}<tr><td class="meta">nothing graded yet</td></tr>{% endfor %}</table></div>{% endmacro %}
{% if agg %}
  {% for h in horizons %}{% set s = agg.overall[h] %}<div class="card"><div class="meta">{{ h }} trading day{{ 's' if h > 1 }} after</div>
    <div class="big {{ 'pos' if s.n and s.avg_value_added > 0 else 'neg' if s.n else '' }}">{{ '{:+.1%}'.format(s.avg_value_added) if s.n else '—' }}</div>
    <div class="meta">{{ s.n }} calls{% if s.n %} · {{ '{:.0%}'.format(s.hit_rate) }} hit rate{% endif %}</div></div>{% endfor %}
  {% if agg.portfolios %}<h2>Portfolio record</h2>
  <p class="meta">The stock book's own record since launch, excluding cash, before costs. Cost is an estimate at the rulebook's
    cost per trade; a real account would also pay taxes on short-term gains.</p>
  <div class="table-wrap"><table>
    <tr><th></th><th>Return</th><th>SPY</th><th>Sharpe</th><th>Max drawdown</th><th>Turnover / yr</th><th>Est. cost</th></tr>
    {% for m in agg.portfolios %}<tr><td><b>{{ m.name }}</b> <span class="meta">{{ m.days }} days</span></td>
      <td class="{{ 'pos' if m['return'] >= m.spy_return else 'neg' }}">{{ '{:+.1%}'.format(m['return']) }}</td>
      <td>{{ '{:+.1%}'.format(m.spy_return) }}</td>
      <td>{{ '{:.2f}'.format(m.sharpe) if m.sharpe is not none else '—' }}</td>
      <td>{{ '{:.1%}'.format(m.max_drawdown) if m.max_drawdown is not none else '—' }}</td>
      <td>{{ '{:.1f}x'.format(m.annual_turnover) if m.annual_turnover is not none else '—' }}</td>
      <td>${{ '{:,.0f}'.format(m.estimated_cost) }}</td></tr>{% endfor %}
  </table></div>{% endif %}
  {{ table("By what triggered the recalibration", agg.by_trigger) }}
  {{ table("Buys, by the signal behind them", agg.by_signal) }}
  {{ table("Buys vs sells", agg.by_side) }}
  {{ table("By portfolio", agg.by_portfolio) }}
  <h2>Worst and best calls (5 trading days after)</h2><div class="table-wrap"><table>
    <tr><th>Date</th><th>Portfolio</th><th>Call</th><th>Trigger · signal</th><th>Stock</th><th>Sector</th><th>Value added</th></tr>
    {% for r in worst + best %}<tr><td>{{ r.date }}</td><td>{{ r.portfolio }}</td><td>{{ r.side }} {{ r.symbol }}</td>
      <td>{{ r.trigger }} · {{ r.signal }}</td><td>{{ '{:+.1%}'.format(r.ret_5) }}</td><td>{{ '{:+.1%}'.format(r.sector_5) }}</td>
      {% set va = r.excess_5 if r.side == 'BUY' else -r.excess_5 %}<td class="{{ 'pos' if va > 0 else 'neg' }}">{{ '{:+.1%}'.format(va) }}</td></tr>
    {% if loop.index == worst|length %}<tr><td colspan="7" style="border-bottom:2px solid #ddd;"></td></tr>{% endif %}{% endfor %}
  </table></div>
{% endif %}
</body></html>"""
EVALUATION_PAGE = EVALUATION_PAGE.replace("<body>", "<body>\n" + TOP_NAV, 1)


@app.route("/evaluation")
def evaluation_route():
    from rec_evaluator import HORIZONS, worst_and_best
    graded, agg, error = [], None, None
    try:
        graded, agg = _scorecard(max_age=0 if request.args.get("refresh") else 1800.0)
    except Exception as e:
        error = f"Could not compute the scorecard: {e}"
    worst, best = worst_and_best(graded) if graded else ([], [])
    return render_template_string(EVALUATION_PAGE, agg=agg, horizons=HORIZONS, worst=worst, best=best,
                                  error=error, active_page="evaluation")


def _intraday_quotes(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """{symbol: {"price": latest trade, "prev_close": previous session close}}
    from Alpaca's snapshot endpoint (one request for all symbols). A symbol
    Alpaca returns nothing for is simply absent."""
    if not symbols:
        return {}
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest
    sc = apply_default_timeout(StockHistoricalDataClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY))
    snaps: Dict[str, Any] = {}
    for i in range(0, len(symbols), 100):          # a sector pool can be 100+ names; keep each request modest
        snaps.update(sc.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbols[i:i + 100])) or {})
    out: Dict[str, Dict[str, float]] = {}
    for sym, sn in snaps.items():
        trade, prev = getattr(sn, "latest_trade", None), getattr(sn, "previous_daily_bar", None)
        if trade is not None and prev is not None and trade.price and prev.close:
            out[sym] = {"price": float(trade.price), "prev_close": float(prev.close)}
    return out


_NEWS_CLIENT = None


def _headlines(symbol: str, days: int) -> List[str]:
    global _NEWS_CLIENT
    if _NEWS_CLIENT is None:
        from alpaca.data.historical.news import NewsClient
        from alpaca_clients import AlpacaNewsClient
        _NEWS_CLIENT = AlpacaNewsClient(NewsClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY))
    return _NEWS_CLIENT.get_recent_headlines(symbol, days)


_NEWS_RAW = None


def _headlines_many(symbols: List[str], days: int) -> Dict[str, List[str]]:
    """{symbol: [headline, ...]} for many symbols in ONE batched Alpaca news
    request (the SDK pages internally; capped at 500 articles). Each article
    lists the tickers it covers, so it is filed under every watched one."""
    global _NEWS_RAW
    from datetime import datetime, timedelta, timezone
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    if _NEWS_RAW is None:
        _NEWS_RAW = apply_default_timeout(NewsClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY))
    wanted = set(symbols)
    out: Dict[str, List[str]] = {}
    news = _NEWS_RAW.get_news(NewsRequest(symbols=",".join(sorted(wanted)), limit=500,
                                          start=datetime.now(timezone.utc) - timedelta(days=days)))
    for a in (news.data.get("news", []) if news else []):
        headline = a["headline"] if isinstance(a, dict) else a.headline
        tickers = a["symbols"] if isinstance(a, dict) else a.symbols
        for t in tickers or []:
            if t in wanted and headline not in out.setdefault(t, []):
                out[t].append(headline)
    return out


def _market_open() -> bool:
    from alpaca.trading.client import TradingClient
    tc = apply_default_timeout(TradingClient(_ALPACA_API_KEY, _ALPACA_SECRET_KEY, paper=True))
    return bool(tc.get_clock().is_open)


def _proactive_cfg():
    from proactive_recal import load_config
    return load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandate.yaml"))


@app.route("/portfolios/proactive_check", methods=["POST"])
def proactive_check_route():
    """Hit once mid-day by Cloud Scheduler. PLACES REAL ORDERS for any
    portfolio the news watch decides to recalibrate, so it takes the same
    token as /portfolios/recalibrate_due. ?dry_run=1 assesses and logs only."""
    import portfolio_store
    from proactive_recal import ProactiveDeps, run_proactive_check
    expected = _recalibrate_token()
    if expected and request.headers.get("X-Recalibrate-Token") != expected:
        return {"error": "unauthorized"}, 403
    dry_run = request.args.get("dry_run") in ("1", "true", "yes")

    def recalibrate(pid: str, reason: str, exits: Sequence[str] = ()) -> Dict[str, Any]:
        return recalibrate_portfolio(pid, trigger="proactive", reason=reason, exclude=exits)

    def client():
        # Not claude_reasoning._client(): this service has no
        # ANTHROPIC_VERTEX_PROJECT_ID, which that factory requires.
        return _claude_client()

    cfg = _proactive_cfg()
    portfolios = []
    for p in portfolio_store.list_portfolios():
        # fold in fills from earlier rebalances first, so "orders still in
        # flight" reflects reality rather than a stale ledger
        if p.get("status") == "open" and p.get("pending_order_ids") and _reconcile_portfolio_orders(p["id"]):
            p = portfolio_store.get_portfolio(p["id"]) or p
        portfolios.append(p)
    deps = ProactiveDeps(quotes_fn=_intraday_quotes, news_fn=_headlines, recalibrate_fn=recalibrate,
                         record_fn=portfolio_store.record_proactive_check, client_fn=client,
                         market_open_fn=_market_open, news_many_fn=_headlines_many, series_fn=_daily_closes,
                         verdict_load_fn=portfolio_store.get_news_verdicts,
                         verdict_save_fn=portfolio_store.save_news_verdicts)
    results = run_proactive_check(portfolios, cfg, deps, dry_run=dry_run)
    return {"dry_run": dry_run or cfg.dry_run, "checked": len(results),
            "results": [{"id": r.portfolio_id, "name": r.name, "action": r.action, "reason": r.reason,
                         "source": r.source} for r in results]}


PORTFOLIO_DETAIL_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meridian — {{ p.name }}</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 780px;
        margin: 0 auto; padding: 32px 16px 64px; color: #0d1117; line-height: 1.45; }
  h1 { font-size: 1.3rem; margin-bottom: 4px; }
  h2 { font-size: 1.05rem; margin: 26px 0 10px; }
  .sub { color: #6a7280; font-size: 0.9rem; margin-bottom: 8px; }
  .status { font-size: 0.75rem; font-weight: 700; padding: 2px 9px; border-radius: 10px; margin-left: 8px; }
  .status.open { background: #e6f9ec; color: #00a340; }
  .status.closed { background: #eef0f5; color: #666; }
  .paper-banner { background: #fff8e6; border-radius: 10px; padding: 8px 14px; font-size: 0.82rem;
                 margin: 10px 0 20px; font-weight: 600; display: inline-block; }
  .stat-row { display: flex; gap: 16px; margin: 16px 0 24px; flex-wrap: wrap; }
  .stat { background: #f6f8ff; border-radius: 12px; padding: 14px 18px; flex: 1; min-width: 140px; }
  .stat .big { font-size: 1.4rem; font-weight: 800; }
  .pos { color: #00a340; } .neg { color: #d43f3f; }
  .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  table { width: 100%; min-width: 420px; border-collapse: collapse; font-size: 0.9rem; }
  th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid #eee; }
  th { color: #888; font-weight: 500; }
  .act { font-size: 0.78rem; font-weight: 700; padding: 1px 6px; border-radius: 6px; display: inline-block; margin: 1px 0; }
  .act.BUY { background: #e6f9ec; color: #00a340; } .act.SELL { background: #fdeaea; color: #d43f3f; }
  .warn { color: #a3312f; background: #fee; padding: 12px 14px; border-radius: 10px; margin: 12px 0; }
  .meta { color: #999; font-size: 0.82rem; margin-top: 28px; }
  .meta-row code { background: #f0f0f5; padding: 1px 6px; border-radius: 5px; font-size: 0.85em; }
  @media (max-width: 600px) { body { padding: 20px 14px 56px; } .stat { min-width: 100%; } }
</style>
</head>
<body>
  <h1>{{ p.name }}<span class="status {{ p.status }}">{{ p.status }}</span></h1>
  {{ selector|safe }}
  <div class="sub">Launched {{ p.created_at[:10] }} · {{ p.sectors|join(', ') }} · {{ p.n_stocks or p.holdings|length }} stocks (kept at every recalibration)
    {% if p.status == 'closed' %}· closed {{ p.closed_at[:10] }}
    {% elif p.next_rebalance_date %}· auto-recalibrates on {{ p.next_rebalance_date }}
    {% else %}· manual recalibration only
    {% endif %}</div>
  <div class="paper-banner">📝 Alpaca PAPER account — not real money.</div>
  {% if p.status == 'open' %}
  <form method="post" action="/portfolios/{{ p.id }}/news_watch" style="margin:0 0 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;"
        {% if not p.news_watch %}onsubmit="return confirm('Turn on the mid-day news watch for {{ p.name }}? Each trading day at 12:30 ET it may sell a holding (real paper orders) that hits its stop or has thesis-breaking news, and fill that slot.');"{% endif %}>
    <input type="hidden" name="enabled" value="{{ '0' if p.news_watch else '1' }}">
    <span style="font-size:0.9rem;">📰 Mid-day news watch:
      <b style="color:{{ '#00a340' if p.news_watch else '#888' }};">{{ 'ON' if p.news_watch else 'OFF' }}</b>
      <span style="color:#6a7280;">{{ '— checks daily at 12:30 ET and may sell a holding that hits its stop or has thesis-breaking news' if p.news_watch else '— recalibrates only on schedule or when you click' }}</span></span>
    <button type="submit" style="background:{{ '#fee' if p.news_watch else '#0d1117' }};color:{{ '#a3312f' if p.news_watch else '#fff' }};
            border:none;font-size:0.8rem;padding:6px 12px;border-radius:8px;cursor:pointer;font-weight:600;">
      {{ 'Turn off' if p.news_watch else 'Turn on' }}</button>
  </form>
  {% endif %}

  {% if error %}<div class="warn">{{ error }}</div>{% endif %}

  {% if p %}
  {{ chart|safe }}

  <div class="stat-row">
    <div class="stat {{ 'good' if p.return_pct >= 0 else 'bad' }}">
      <div>This portfolio</div>
      <div class="big {{ 'pos' if p.return_pct >= 0 else 'neg' }}">{{ '%+.1f'|format(p.return_pct) }}%</div>
      <div>${{ '{:,.0f}'.format(p.capital) }} → ${{ '{:,.2f}'.format(p.mtm_value) }}</div>
    </div>
    {% if spy_return is not none %}
    <div class="stat">
      <div>S&amp;P 500 (SPY) over the same window</div>
      <div class="big {{ 'pos' if spy_return >= 0 else 'neg' }}">{{ '%+.1f'|format(spy_return*100) }}%</div>
      <div>same ${{ '{:,.0f}'.format(p.capital) }} start</div>
    </div>
    {% endif %}
    {% if p.estimated_1mo_return is not none %}
    <div class="stat">
      <div>Naive 1-month projection (logged at launch)</div>
      <div class="big {{ 'pos' if p.estimated_1mo_return >= 0 else 'neg' }}">{{ '%+.1f'|format(p.estimated_1mo_return*100) }}%</div>
      <div>vs. actual {{ '%+.1f'|format(p.return_pct) }}% after {{ '%.1f'|format(p.age_days) }} day{{ 's' if p.age_days != 1 }}</div>
    </div>
    {% endif %}
  </div>
  {% if p.estimated_1mo_return is not none %}
  <p class="meta">The projection above is exactly what was shown on the recommendation page the
    moment you clicked "Launch as New Portfolio" — trailing 1-month momentum extrapolated
    forward, weighted by allocation, logged once and never recomputed. It's a 1-month-horizon
    estimate, so it's only a fair apples-to-apples comparison once this portfolio is itself
    about a month old; before that, "actual" simply covers a shorter window than what was
    projected.</p>
  {% endif %}

  {% if p.holdings_display %}
  <h2>Holdings</h2>
  <div class="table-wrap">
  <table>
    <tr><th>Symbol</th><th>Shares</th><th>Entry price</th><th>Current price</th><th>Value</th></tr>
    {% for h in p.holdings_display %}
    <tr>
      <td><a href="/stock/{{ h.symbol }}">{{ h.symbol }}</a></td>
      <td>{{ h.shares }}</td>
      <td>${{ '%.2f'|format(h.entry_price) }}</td>
      <td>{% if h.current_price %}${{ '%.2f'|format(h.current_price) }}{% endif %}</td>
      <td>${{ '{:,.2f}'.format(h.value) }}</td>
    </tr>
    {% endfor %}
    <tr style="font-weight:700;border-top:2px solid #ddd;">
      <td colspan="4">Holdings total</td>
      <td>${{ '{:,.2f}'.format(p.holdings_display|sum(attribute='value')) }}</td>
    </tr>
    <tr>
      <td colspan="4">Cash</td>
      <td>${{ '{:,.2f}'.format(p.cash_remaining or 0.0) }}</td>
    </tr>
    <tr style="font-weight:700;border-top:2px solid #ddd;">
      <td colspan="4">Total portfolio value</td>
      <td>${{ '{:,.2f}'.format(p.mtm_value) }}</td>
    </tr>
  </table>
  </div>
  {% endif %}

  <p class="meta-row">To verify these real fills yourself in Alpaca's own dashboard
    (<a href="https://app.alpaca.markets/paper/dashboard/orders" target="_blank" rel="noopener">app.alpaca.markets → Orders</a>),
    search for order IDs starting with <code>webui-{{ p.id }}-</code>.</p>

  <style>
    .chip { display:inline-block; font-size:0.75rem; font-weight:700; padding:1px 7px; border-radius:6px; margin:1px 2px 1px 0; }
    .chip.out { background:#fdeaea; color:#d43f3f; } .chip.in { background:#e6f9ec; color:#00a340; }
    .chip.rs { background:#eef0f5; color:#555; }
    .grade { display:inline-block; min-width:22px; text-align:center; font-weight:800; border-radius:6px; padding:1px 6px; }
    .grade.A, .grade.B { background:#e6f9ec; color:#00a340; } .grade.C { background:#fff4e5; color:#b45309; }
    .grade.D, .grade.F { background:#fdeaea; color:#d43f3f; }
    .hist td { vertical-align:top; } .hist .small { color:#6a7280; font-size:0.78rem; }
    details.more summary { cursor:pointer; color:#4a6cf7; font-size:0.78rem; list-style:none; }
    details.more summary::-webkit-details-marker { display:none; }
    details.more div { font-size:0.8rem; color:#555; margin-top:3px; max-width:340px; white-space:normal; }
  </style>

  {% if timeline and timeline.initial %}
  <h2>Started with</h2>
  <p class="meta-row">What this portfolio held at launch ({{ p.created_at[:10] }}) and what became of each position.</p>
  <div class="table-wrap"><table>
    <tr><th>Stock</th><th>Shares at launch</th><th>Launch price</th><th>Now</th></tr>
    {% for r in timeline.initial %}
    <tr><td><a href="/stock/{{ r.symbol }}">{{ r.symbol }}</a></td><td>{{ '%g'|format(r.shares) }}</td>
      <td>{% if r.entry_price %}${{ '%.2f'|format(r.entry_price) }}{% else %}<span class="small">filled later</span>{% endif %}</td>
      <td>{% if r.now_shares %}✓ held ({{ '%g'|format(r.now_shares) }} shares){% elif r.sold_on %}<span class="chip out">sold {{ r.sold_on }}</span>{% else %}<span class="small">not held</span>{% endif %}</td></tr>
    {% endfor %}
  </table></div>
  {% endif %}

  {% if p.rebalance_history %}
  <h2>Recalibration history</h2>
  {% set sm = timeline.summary %}
  <p class="meta-row">{{ sm.n }} recalibration{{ 's' if sm.n != 1 }} since launch
    {%- if sm.by_type %} ({% for k, v in sm.by_type|dictsort %}{{ v }} {{ {'scheduled': '📅 scheduled', 'proactive': '📰 mid-day', 'manual': '👆 manual'}.get(k, 'unlabelled') }}{{ ', ' if not loop.last }}{% endfor %}){% endif %}.
    {% if sm.started %}Started with {{ sm.started|join(', ') }} → now {{ sm.now|join(', ') }}{% if sm.kept %} (kept {{ sm.kept|join(', ') }}){% endif %}.{% endif %}</p>
  <div class="table-wrap">
  <table class="hist">
    <tr><th>Date</th><th>Type</th><th>What changed</th><th>Why</th><th>🧪 Reviewer</th><th>Value before</th></tr>
    {% for e in timeline.events %}
    <tr>
      <td style="white-space:nowrap;">{{ e.date }}{% if e.at %}<div class="small">{{ e.at|et_time }} ET</div>{% endif %}</td>
      <td style="white-space:nowrap;">{{ e.type }}</td>
      <td>
        {% if e.no_change %}<span class="small">no changes needed</span>
        {% else %}
          {% for s in e.exited %}<span class="chip out" title="sold all {{ '%g'|format(e.exited_qty[s]) }} shares">−{{ s }} {{ '%g'|format(e.exited_qty[s]) }}</span>{% endfor %}
          {% for s in e.entered %}<span class="chip in" title="bought {{ '%g'|format(e.entered_qty[s]) }} shares">+{{ s }} {{ '%g'|format(e.entered_qty[s]) }}</span>{% endfor %}
          {% for s in e.resized %}<span class="chip rs" title="{{ 'bought' if e.resized_delta[s] > 0 else 'sold' }} {{ '%g'|format(e.resized_delta[s]|abs) }} shares">⇅{{ s }} {{ ('%+g'|format(e.resized_delta[s]))|replace('-', '−') }}</span>{% endfor %}
          <div class="small">holds {{ e.n_holds }} · shares sold (−), bought (+), resized (⇅)</div>
        {% endif %}
        {% if e.not_filled %}<div style="color:#b45309;font-size:0.78rem;">{% for k, v in e.not_filled|dictsort %}{{ v }} order{{ 's' if v > 1 }} {{ k }}{{ ', ' if not loop.last }}{% endfor %}</div>{% endif %}
        {% if e.errors %}<div style="color:#a3312f;font-size:0.78rem;">{{ e.errors }} order error{{ 's' if e.errors > 1 }}</div>{% endif %}
      </td>
      <td style="white-space:normal;max-width:260px;font-size:0.84rem;">{{ e.why_short or '—' }}
        {% if e.why_full %}<details class="more"><summary>more ▸</summary><div>{{ e.why_full }}</div></details>{% endif %}</td>
      <td style="white-space:normal;max-width:260px;font-size:0.84rem;">
        {% if e.grade or e.blocked %}
          {% if e.grade %}<span class="grade {{ e.grade }}">{{ e.grade }}</span>{% endif %} {{ e.review_short }}
          {% if e.blocked %}<div style="color:#a3312f;font-size:0.78rem;">⛔ blocked {{ e.blocked|map(attribute=0)|join(', ') }}</div>{% endif %}
          {% if e.objections or e.concerns %}<div class="small">✋ {{ e.objections|length }} objection{{ 's' if e.objections|length != 1 }} · ⚠️ {{ e.concerns|length }} concern{{ 's' if e.concerns|length != 1 }}</div>{% endif %}
          {% if e.review_full or e.blocked or e.objections or e.concerns %}<details class="more"><summary>details ▸</summary><div>
            {% if e.review_full %}{{ e.review_full }}<br>{% endif %}
            {% for sym, why in e.blocked %}⛔ {{ sym }}: {{ why }}<br>{% endfor %}
            {% for sym, why in e.objections %}✋ {{ sym }}: {{ why }}<br>{% endfor %}
            {% for sym, why in e.concerns %}⚠️ {{ sym }}: {{ why }}<br>{% endfor %}
          </div></details>{% endif %}
        {% else %}<span class="small">—</span>{% endif %}
      </td>
      <td style="white-space:nowrap;">{% if e.value_before %}${{ '{:,.0f}'.format(e.value_before) }}{% else %}—{% endif %}</td>
    </tr>
    {% endfor %}
  </table>
  </div>
  {% endif %}

  {% if p.proactive_checks %}
  <h2>Mid-day news watch</h2>
  <p class="meta-row">Once each trading day the news watch reads this portfolio's headlines and price moves
    and decides whether to recalibrate ahead of schedule. Latest checks:</p>
  <div class="table-wrap">
  <table>
    <tr><th>When</th><th>Decision</th><th>Why</th></tr>
    {% for c in (p.proactive_checks|reverse|list)[:7] %}
    <tr>
      <td>{{ c.at[:10] }}</td>
      <td><b>{{ c.action }}</b>{% if c.source and c.source != 'rules' %}<div style="color:#888;font-size:0.78rem;">{{ (c.review.model if c.review and c.review.model else c.source) if c.source != 'fallback' else 'backup rule' }}{% if c.source != 'fallback' and c.review and c.review.confidence is not none %} · {{ '%.0f'|format(c.review.confidence * 100) }}%{% endif %}</div>{% endif %}</td>
      <td style="white-space:normal;">{{ c.reason }}{% if c.triggers %}<div style="color:#888;font-size:0.8rem;">{{ c.triggers|join('; ') }}</div>{% endif %}
        {% if c.coverage %}<div style="color:#888;font-size:0.78rem;">watched {{ c.coverage.holdings }} holdings + {{ c.coverage.sector_stocks }} other {{ p.sectors|join('/') }} stocks · {{ c.coverage.with_news }} with news{% if c.coverage.big_movers %} · {{ c.coverage.big_movers }} big movers{% endif %}</div>{% endif %}</td>
    </tr>
    {% endfor %}
  </table>
  </div>
  {% endif %}

  {% if p.status == 'open' %}
  <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px;">
    <form method="post" action="/portfolios/{{ p.id }}/recalibrate"
          onsubmit="return confirm('Re-run the recommendation for {{ p.name }} and adjust real Alpaca positions to the new target weights now?');">
      <button type="submit" style="background:#0d1117;color:#fff;border:none;font-size:0.85rem;
              padding:8px 16px;border-radius:8px;cursor:pointer;font-weight:600;">🔄 Recalibrate Now</button>
    </form>
    <form method="post" action="/portfolios/{{ p.id }}/close"
          onsubmit="return confirm('Sell all holdings in {{ p.name }} for real on Alpaca and close this portfolio?');">
      <button type="submit" style="background:#fee;color:#a3312f;border:1.5px solid #f3d4d4;font-size:0.85rem;
              padding:8px 16px;border-radius:8px;cursor:pointer;font-weight:600;">Close &amp; Liquidate</button>
    </form>
  </div>
  {% endif %}
  {% endif %}

  <p class="meta">This chart is reconstructed from real historical Alpaca prices for exactly what
    this portfolio bought at launch, held day by day since {{ p.created_at[:10] if p else '' }} —
    it is not a live-updating stream, but every point on it is real market data, recomputed fresh
    each time you load this page.</p>
  <p class="meta"><a href="/portfolios">← Back to all portfolios</a></p>
</body>
</html>
"""
PORTFOLIO_DETAIL_PAGE = PORTFOLIO_DETAIL_PAGE.replace("<body>", "<body>\n" + TOP_NAV, 1)


@app.route("/portfolios/<portfolio_id>")
def portfolio_detail(portfolio_id: str, notice: Optional[str] = None):
    """Real day-by-day reconstruction of ONE tracked portfolio's value since
    launch, from its actual entry holdings + real historical Alpaca prices
    — compared against a SPY buy-and-hold funded with the same capital over
    the same window. Recomputed fresh on every page load (not a stored,
    incrementally-updated series) so it always reflects today's real data,
    including whatever trading days have passed since you last looked."""
    import portfolio_store
    from simulate import fetch_price_series
    from datetime import datetime, timezone

    p = portfolio_store.get_portfolio(portfolio_id)
    if p is None:
        selector = _portfolio_selector(_all_portfolios_enriched(), "/portfolios?view=all")
        return render_template_string(PORTFOLIOS_PAGE, portfolios=[], error="Portfolio not found.",
                                      active_page="portfolios", selector=selector)

    # Show the LATEST holdings: fold in any launch/recalibration orders that
    # were still in flight when the ledger was last written (see
    # _reconcile_portfolio_orders), then re-read so the table below is current.
    if p.get("status") == "open" and p.get("pending_order_ids") and _reconcile_portfolio_orders(portfolio_id):
        p = portfolio_store.get_portfolio(portfolio_id) or p

    error, chart, spy_return = notice, "", None
    holdings = p.get("holdings") or {}
    symbols = list(holdings.keys())
    # Portfolios created before this field existed have NO
    # "estimated_1mo_return" key at all — default it here, before any
    # enrichment, so the template's `is not none` check behaves correctly
    # even on the path where mark-to-market enrichment itself fails below
    # (see _mark_to_market's own matching explicit default for the normal path).
    p.setdefault("estimated_1mo_return", None)

    # BUG FIX: mark-to-market (needed for the basic return%/value stats)
    # and the historical chart used to share ONE try/except — a chart-only
    # failure (confirmed live: this account's free-tier IEX data plan
    # rejects any query window edging into "now", a real, reachable error)
    # left `p` as the RAW, un-enriched portfolio dict, which the template
    # then crashed on (`p.return_pct` doesn't exist on that dict) — a 500
    # instead of the intended graceful `error` banner. Enrichment now
    # always runs; only the chart itself is allowed to fail independently.
    try:
        live = _live_prices(symbols)
        p = _mark_to_market(p, live)
    except Exception as e:
        error = f"Could not load current prices: {e}"

    # Computed here (not inside the chart's own try/except below) so it's
    # available for the estimated-vs-actual comparison even if the
    # historical chart fetch fails independently.
    created_at_iso = p["created_at"]
    end_at_iso = p.get("closed_at") or datetime.now(timezone.utc).isoformat()
    launch_dt = datetime.fromisoformat(created_at_iso)
    end_dt_for_age = datetime.fromisoformat(end_at_iso)
    p["age_days"] = (end_dt_for_age - launch_dt).total_seconds() / 86400.0

    try:
        # Always chart at hourly resolution — daily bars put at most 1-2
        # points on a chart for a portfolio that's only hours old (confirmed
        # live: a same-day portfolio's chart looked like a flat line or a
        # single dot), and hourly bars remain a more granular, truthful view
        # of the portfolio's actual path even once it's been running for
        # weeks or months.
        timeframe = "hour"
        launch_arg = created_at_iso
        end_arg = end_at_iso

        if symbols:
            series = fetch_price_series(symbols + ["SPY"], launch_arg, end_arg,
                                        _ALPACA_API_KEY, _ALPACA_SECRET_KEY, timeframe=timeframe)
            all_dates = sorted(set(d for s in symbols for d, _ in series.get(s, [])))

            def price_on_or_before(sym: str, date: str) -> Optional[float]:
                for d, price in reversed(series.get(sym, [])):
                    if d <= date:
                        return price
                return None

            cash = p.get("cash_remaining", 0.0)
            capital = p.get("capital") or 1.0
            portfolio_values = []
            for date in all_dates:
                val = cash
                for sym, h in holdings.items():
                    px = price_on_or_before(sym, date) or h["entry_price"]
                    val += px * h["shares"]
                portfolio_values.append(val)

            spy_values: List[float] = []
            spy_series = series.get("SPY", [])
            if spy_series and all_dates:
                entry_spy = price_on_or_before("SPY", all_dates[0])
                if entry_spy:
                    spy_shares = capital / entry_spy
                    spy_values = [(price_on_or_before("SPY", d) or entry_spy) * spy_shares for d in all_dates]
                    if spy_values:
                        spy_return = (spy_values[-1] - capital) / capital

            chart = svg_multi_line_chart(all_dates, [
                (p["name"], portfolio_values, "#4a6cf7"),
                ("S&P 500 (SPY)", spy_values, "#d97706"),
            ]) if all_dates else "<p>Not enough trading days have passed yet to chart this portfolio.</p>"
    except Exception as e:
        error = (error + " · " if error else "") + f"Could not load historical chart: {e}"

    selector = _portfolio_selector(_all_portfolios_enriched(), f"/portfolios/{portfolio_id}")
    from holdings_timeline import build_timeline
    timeline = build_timeline(p) if p else None
    return render_template_string(PORTFOLIO_DETAIL_PAGE, p=p, chart=chart, error=error, timeline=timeline,
                                  spy_return=spy_return, active_page="portfolios", selector=selector)


# =============================================================================
# Day Trading tab — the page itself lives in daytrade_ui.py (a Blueprint, so it
# can be tested without this module's import-time GCP calls). Everything here
# is lazy: a missing day-trading secret or Firestore hiccup degrades that ONE
# tab to a warning banner and never affects the portfolio pages above.
# The runner that actually trades is a separate Cloud Run Job
# (daytrade_runner.py); this tab only reads state and can HALT it.
# =============================================================================
_DAYTRADE_STORE = None
_DAYTRADE_BROKER = None
_DAYTRADE_TOKEN: Optional[str] = None


def _daytrade_store():
    global _DAYTRADE_STORE
    if _DAYTRADE_STORE is None:
        from daytrade_store import default_store
        _DAYTRADE_STORE = default_store()
    return _DAYTRADE_STORE


def _daytrade_cfg():
    from daytrade_config import load_day_trading_config
    return load_day_trading_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandate.yaml"))


def _daytrade_live() -> Dict[str, Any]:
    """Live view of the DEDICATED day-trading paper account (never the
    portfolio account). Raises when its keys aren't configured yet — the
    Blueprint turns that into a banner. A failed construction isn't cached,
    so fixing the secret takes effect on the next page load."""
    import dataclasses
    global _DAYTRADE_BROKER
    if _DAYTRADE_BROKER is None:
        from daytrade_broker import AlpacaDayTradeBroker, make_daytrade_trading_client
        _DAYTRADE_BROKER = AlpacaDayTradeBroker(make_daytrade_trading_client(), dedicated=True)
    b = _DAYTRADE_BROKER
    return {"account": b.account_snapshot(),
            "positions": [dataclasses.asdict(p) for p in b.positions().values()]}


def _daytrade_token() -> Optional[str]:
    """Shared secret required to RESUME trading after a halt (halting itself
    is open — it only reduces risk). Same opt-out semantics as
    _recalibrate_token(): no secret configured means unprotected, on purpose,
    for local use."""
    global _DAYTRADE_TOKEN
    if _DAYTRADE_TOKEN is None:
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            _DAYTRADE_TOKEN = client.access_secret_version(
                name=f"projects/{PROJECT}/secrets/daytrade-token/versions/latest"
            ).payload.data.decode("utf-8")
        except Exception as e:
            print(f"[web_ui] no daytrade-token secret configured ({e}) — RESUME is UNPROTECTED")
            _DAYTRADE_TOKEN = ""
    return _DAYTRADE_TOKEN or None


try:
    from daytrade_ui import make_daytrade_blueprint
    app.register_blueprint(make_daytrade_blueprint(
        TOP_NAV, _daytrade_store, _daytrade_live, _daytrade_token, _daytrade_cfg))
except ImportError as _e:      # a missing module must not take the portfolio app down
    print(f"[web_ui] Day Trading tab disabled: {_e}")


if __name__ == "__main__":
    print(f"Meridian sector-recommendation UI — using agent {RESOURCE_NAME}")
    print("Open http://127.0.0.1:5050 in your browser.")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
