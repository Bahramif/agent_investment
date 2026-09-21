"""
test_agent.py
==============
Reusable CLI for testing the deployed InvestmentFirmAgent on Vertex AI Agent
Engine — so testing a query doesn't mean hand-writing a python -c snippet
every time.

Two test modes:
  --mode snapshots (default)
      Supplies instrument snapshots directly in the request state. Exercises
      the full pipeline — analysts, strategist, freshness gate, portfolio
      construction, risk check, execute — WITHOUT needing live ingestion
      wired up. This is what's been used to verify every deploy so far; it
      always works regardless of UniverseConfig.

  --mode live
      Omits *_snapshots from the request state entirely, letting the
      deployed graph's data_ingestion node fetch REAL Alpaca market/news
      data instead. Requires TWO things to actually return non-empty
      signals:
        1. ALPACA_API_KEY/ALPACA_SECRET_KEY resolve on the container (via
           the alpaca_api_key_secret/alpaca_secret_key_secret passed to
           build_and_deploy() — already wired).
        2. UniverseConfig in vertex_deploy.py's set_up() is no longer the
           empty placeholder (equities=[], forex=[], crypto=[], themes=[])
           — it needs real (symbol, bucket_path, quality_score) tuples
           matching whatever mandate you pass in. Still a TODO as of this
           writing; --mode live will run without erroring but likely
           produce an all-cash book if the universe is empty.

Usage:
    python3 test_agent.py RESOURCE_NAME [--mode snapshots|live]
        [--mandate mandate.yaml] [--thread-id my-test-1] [--stream]

    # Using the currently-live deployment:
    python3 test_agent.py \\
        projects/497590571361/locations/us-central1/reasoningEngines/1931795200758251520
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict


# Default test mandate — verified to fully allocate to 100% with no risk
# rejections under the default per-name cap (see mandate_allocator.py's
# default_limits). Swap --mandate to point at mandate.yaml (the project's
# real mandate) or a mandate_builder.py-produced file instead.
DEFAULT_MANDATE: Dict[str, Any] = {
    "total": 1.0,
    "default_limits": {"max_per_name": 0.10, "min_names": 1},
    "buckets": {
        "equities": {"set_by": "user", "weight": 0.60, "band": 0.05},
        "crypto":   {"set_by": "user", "weight": 0.10, "band": 0.05},
        "cash":     {"set_by": "user", "weight": 0.30, "band": 0.03},
    },
}

# Matching test snapshots for --mode snapshots — enough names per bucket to
# actually fill it under default_limits.max_per_name (0.10), so a clean run
# reaches "approve" / "executed" rather than looping into a rejected book.
DEFAULT_EQUITIES_SNAPSHOTS = [
    {"name": "NVDA",  "path": "equities", "momentum_1m": 0.05,  "news_sentiment": 0.4, "quality_score": 0.6},
    {"name": "MSFT",  "path": "equities", "momentum_1m": 0.03,  "news_sentiment": 0.3, "quality_score": 0.5},
    {"name": "GOOGL", "path": "equities", "momentum_1m": 0.02,  "news_sentiment": 0.2, "quality_score": 0.5},
    {"name": "AVGO",  "path": "equities", "momentum_1m": 0.02,  "news_sentiment": 0.1, "quality_score": 0.4},
    {"name": "XOM",   "path": "equities", "momentum_1m": 0.01,  "news_sentiment": 0.1, "quality_score": 0.3},
    {"name": "CAT",   "path": "equities", "momentum_1m": -0.01, "news_sentiment": 0.0, "quality_score": 0.2},
]
DEFAULT_CRYPTO_SNAPSHOTS = [
    {"name": "BTC", "path": "crypto", "momentum_1m": 0.08, "news_sentiment": 0.3, "quality_score": 0.4},
]


def build_state(mandate: Dict[str, Any], mode: str) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "mandate": mandate,
        "tilt_caps": {"strategic_cap": 0.15, "tactical_cap": 0.06, "total_cap": 0.15},
        "theme_memory": {},
        "elapsed_days": 0.0,
        "material_threshold": 0.03,
        "risk_limits": {"max_single_name": 0.20},
        "prior_theme_budgets": {},
        "macro_snapshots": [],
    }
    if mode == "snapshots":
        state["equities_snapshots"] = DEFAULT_EQUITIES_SNAPSHOTS
        state["forex_snapshots"] = []
        state["crypto_snapshots"] = DEFAULT_CRYPTO_SNAPSHOTS
    else:  # "live" — omit *_snapshots entirely so the ingestion node fills them
        pass
    return state


def load_mandate(path: str) -> Dict[str, Any]:
    from mandate_allocator import load_mandate as _load
    return _load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("resource_name", help="Full ReasoningEngine resource name, "
                        "e.g. projects/P/locations/us-central1/reasoningEngines/ID")
    parser.add_argument("--project", default="meridian-508414")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--mode", choices=["snapshots", "live"], default="snapshots")
    parser.add_argument("--mandate", default=None, help="Path to a mandate.yaml-shaped file "
                        "(default: built-in test mandate, ignored --mode live universe caveats)")
    parser.add_argument("--thread-id", default="test-agent-cli")
    parser.add_argument("--stream", action="store_true", help="Use stream_query() instead of "
                        "query(), printing each node's output as it runs")
    args = parser.parse_args()

    import vertexai
    from vertexai.preview import reasoning_engines

    vertexai.init(project=args.project, location=args.location)
    remote_agent = reasoning_engines.ReasoningEngine(args.resource_name)

    mandate = load_mandate(args.mandate) if args.mandate else DEFAULT_MANDATE
    state = build_state(mandate, args.mode)

    print(f"Calling {'stream_query' if args.stream else 'query'}() "
          f"[mode={args.mode}, thread_id={args.thread_id}]...", flush=True)

    if args.stream:
        for chunk in remote_agent.stream_query(thread_id=args.thread_id, state=state):
            print(chunk)
        return

    result = remote_agent.query(thread_id=args.thread_id, state=state)
    print()
    print("status:      ", result.get("status"))
    print("risk_verdict:", result.get("risk_verdict"), result.get("risk_reasons") or "")
    print("retries:     ", result.get("retries"))
    print("final_weights:")
    weights = result.get("final_weights") or {}
    for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {name:10s} {w:6.2%}")
    if weights:
        print(f"  {'TOTAL':10s} {sum(weights.values()):6.2%}")
    else:
        print("  (none — check risk_verdict/reasons above; a rejected proposal "
              "never reaches execute)")


if __name__ == "__main__":
    main()
