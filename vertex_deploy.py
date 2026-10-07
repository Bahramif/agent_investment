"""
vertex_deploy.py
=================
Wraps the compiled LangGraph investment-firm graph for deployment on
Vertex AI Agent Engine (the managed runtime under Vertex AI Agent Builder /
Gemini Enterprise).

Agent Engine's custom-agent template is a plain Python class with methods
Google's runtime looks for:

  __init__()  -- config only (project, location, model names). Must stay
                 pickle-able, so NO live connections or unpickle-able
                 objects here.
  set_up()    -- runs once when the container starts. This is where you
                 build anything that can't be pickled: the compiled graph,
                 the checkpointer, DB clients, model clients.
  query() /
  stream_query() -- runs per request, synchronously. Invokes the graph and
                 returns state.
  async_query()  -- Agent Engine's documented convention for an async agent
                 method: define one, and the platform SDK exposes it as
                 agent.async_query(...). REQUIRED here whenever
                 temporal_target is configured, since the execute node
                 becomes async in that case (see make_execute_node in
                 firm_graph.py) — a sync .invoke() cannot await it.

This shape is exactly why it fits your architecture with almost no
rewrite: __init__/set_up/query maps directly onto "build the graph once,
invoke it many times," which is what compiled_graph.invoke(state) already
does locally.

NOTE ON NAMING: Google has been consolidating this surface under the
Gemini Enterprise brand and the exact import path has moved between
`vertexai.preview.reasoning_engines` and `vertexai.agent_engines` across
recent doc revisions. Check current docs before deploying — the
set_up()/query() contract shown here has been stable across that rename,
but the top-level entrypoint name may differ by the time you build this.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _coerce_firm_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """query()/async_query()/stream_query() arrive over a JSON RPC boundary —
    Agent Engine decodes the request body into plain dicts/lists before
    calling this class, so nested values land as plain dicts. But every node
    function downstream (analyst_reasoning.py, macro_reasoning.py,
    strategist.py, claude_reasoning.py) does attribute access on the actual
    dataclasses (InstrumentSnapshot, MacroSnapshot, TiltCaps, ThemeStrategy,
    HorizonView) — e.g. `snap.momentum_1m`, `caps.strategic_cap`. Without this,
    a real remote query() call fails with things like
    "AttributeError: 'dict' object has no attribute 'name'" deep inside a
    node, which is exactly what happened testing this end-to-end against the
    deployed agent. Coerce once, here, at the boundary, rather than pushing
    dict-vs-dataclass branching into every node module.

    `mandate` and `risk_limits` are NOT touched — mandate_allocator.py and
    risk_check_node read those via plain dict access by design."""
    from analyst_reasoning import InstrumentSnapshot
    from macro_reasoning import MacroSnapshot
    from strategist import TiltCaps, ThemeStrategy, HorizonView

    def _view(v: Any) -> HorizonView:
        return v if isinstance(v, HorizonView) else HorizonView(**v)

    state = dict(state)   # shallow copy — never mutate the caller's dict
    for key in ("equities_snapshots", "forex_snapshots", "crypto_snapshots"):
        if state.get(key):
            state[key] = [s if isinstance(s, InstrumentSnapshot) else InstrumentSnapshot(**s)
                         for s in state[key]]
    if state.get("macro_snapshots"):
        state["macro_snapshots"] = [s if isinstance(s, MacroSnapshot) else MacroSnapshot(**s)
                                    for s in state["macro_snapshots"]]
    if state.get("tilt_caps") is not None and not isinstance(state["tilt_caps"], TiltCaps):
        state["tilt_caps"] = TiltCaps(**state["tilt_caps"])
    if state.get("theme_memory"):
        state["theme_memory"] = {
            theme: ts if isinstance(ts, ThemeStrategy) else
                  ThemeStrategy(ts["theme"], _view(ts["strategic"]), _view(ts["tactical"]))
            for theme, ts in state["theme_memory"].items()
        }
    if isinstance(state.get("force_refresh_themes"), list):
        state["force_refresh_themes"] = set(state["force_refresh_themes"])   # JSON has no set type
    return state


def _resolve_secret(secret_name: str, secret_client: Optional[Any] = None) -> str:
    """Fetch a Secret Manager secret's latest payload as a string. Pulled out
    as a standalone function so it's swappable/testable with a fake client,
    same pattern the old _resolve_pg_conn_string used before the checkpointer
    moved to Firestore."""
    if secret_client is None:
        from google.cloud import secretmanager
        secret_client = secretmanager.SecretManagerServiceClient()
    response = secret_client.access_secret_version(name=secret_name)
    return response.payload.data.decode("utf-8")


class InvestmentFirmAgent:
    """Deployable wrapper around the firm's compiled LangGraph."""

    def __init__(
        self,
        project: str,
        location: str,
        claude_region: str = "global",
        use_direct_anthropic_api: bool = False,
        firestore_project: Optional[str] = None,
        checkpoints_collection: str = "checkpoints",
        alpaca_api_key_secret: Optional[str] = None,
        alpaca_secret_key_secret: Optional[str] = None,
        temporal_target: Optional[str] = None,
        total_capital: float = 0.0,
        enable_tracing: bool = False,
    ):
        # Config only — this object must stay pickle-able for deployment.
        # (Plain strings/floats are pickle-able; a live DB or Temporal
        # connection is not — that's why those get built in set_up(), not here.)
        self.project = project
        self.location = location
        self.claude_region = claude_region              # Claude on Vertex needs global/us/eu
        self.use_direct_anthropic_api = use_direct_anthropic_api
        # Checkpointer: Firestore (Native mode), not Postgres — no connection
        # string/secret to manage at all. Auth rides on the same ADC/service-
        # account identity Agent Engine already uses for everything else
        # (vertexai.init(), Secret Manager, etc), so there's no VPC/networking
        # setup either. firestore_project defaults to `project` — only pass a
        # different value if the Firestore database lives in another project.
        # NOTE: langgraph-checkpoint-firestore's FirestoreSaver only talks to
        # the "(default)" Firestore database in a project, not a named one.
        self.firestore_project = firestore_project or project
        self.checkpoints_collection = checkpoints_collection
        # Alpaca credentials: Secret Manager resource names (e.g.
        # "projects/P/secrets/alpaca-api-key/versions/latest"), NOT raw key
        # material — set_up() fetches the actual values at container start
        # (see _resolve_secret()) and sets them into os.environ there, so the
        # keys never sit in this object's pickled state or in the deployment
        # config. Leave both None to skip Alpaca entirely (matches the
        # existing "skip ingestion if creds missing" behavior below) — e.g.
        # for local/dev, just export ALPACA_API_KEY/ALPACA_SECRET_KEY instead.
        self.alpaca_api_key_secret = alpaca_api_key_secret
        self.alpaca_secret_key_secret = alpaca_secret_key_secret
        # temporal_target: host:port (self-hosted) or a Temporal Cloud
        # namespace address. None disables the execution handoff entirely —
        # the graph will just report final weights, dispatching nothing.
        self.temporal_target = temporal_target
        self.total_capital = total_capital
        # See set_up()'s tracer setup below for what this actually turns on.
        self.enable_tracing = enable_tracing

    def set_up(self) -> None:
        """Runs once per container start. Build everything un-pickle-able here."""
        import os
        import vertexai
        from langgraph_checkpoint_firestore import FirestoreSaver

        vertexai.init(project=self.project, location=self.location)

        # Cloud Trace: sends one span per real LangGraph node (see
        # firm_graph.py's _traced() wrapper, applied to every node in
        # build_compiled_graph()) to Cloud Trace, so the Agent Engine
        # console's Traces tab shows the exact graph topology in
        # web_ui.py's /about page, per-request, with real per-node timing.
        # Vertex AI Agent Engine's own env-var tracing toggle
        # (GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY) only auto-instruments
        # Google's own prebuilt agent templates (AdkApp/LanggraphAgent/
        # LangchainAgent) — this class is a fully custom agent, so tracing
        # has to be wired up by hand via OpenTelemetry, per Google's own
        # docs for custom agents. Requires the telemetry.googleapis.com and
        # cloudtrace.googleapis.com APIs enabled on `self.project`, plus
        # roles/cloudtrace.agent on Agent Engine's runtime service account.
        if self.enable_tracing:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

            provider = TracerProvider(
                resource=Resource.create({"service.name": "meridian-investment-firm"})
            )
            provider.add_span_processor(
                BatchSpanProcessor(CloudTraceSpanExporter(project_id=self.project))
            )
            trace.set_tracer_provider(provider)

        # Point claude_reasoning at this same GCP project by default — no
        # separate API key to manage, IAM already covers it. Set
        # use_direct_anthropic_api=True (+ ANTHROPIC_API_KEY in the
        # environment) instead if you'd rather call the direct API.
        if self.use_direct_anthropic_api:
            os.environ["CLAUDE_USE_DIRECT_API"] = "1"
        else:
            os.environ["ANTHROPIC_VERTEX_PROJECT_ID"] = self.project
            os.environ["CLOUD_ML_REGION"] = self.claude_region

        # Pull Alpaca credentials from Secret Manager if configured, rather
        # than requiring them already set in the container's environment —
        # keeps the actual key material out of the deployment config, IAM
        # audit logs, etc. setdefault() so an explicitly pre-set env var
        # (e.g. local/dev override) always wins. Left unconfigured, this is a
        # no-op and Alpaca stays skipped exactly as before. A fetch failure
        # (e.g. the secret exists but has no version yet) must NOT crash
        # set_up() — that would take the whole container down at startup over
        # a single optional integration; skip Alpaca and keep going instead,
        # same as the "creds missing" path below already does.
        if self.alpaca_api_key_secret and self.alpaca_secret_key_secret:
            try:
                os.environ.setdefault("ALPACA_API_KEY", _resolve_secret(self.alpaca_api_key_secret))
                os.environ.setdefault("ALPACA_SECRET_KEY", _resolve_secret(self.alpaca_secret_key_secret))
            except Exception as e:
                print(f"[vertex_deploy] Alpaca secret fetch failed ({e}) — continuing without Alpaca.")

        # Bundled via `extra_packages` at deploy time (see build_and_deploy()).
        from claude_reasoning import make_claude_reasoners, make_claude_analyst_reasoner
        from data_ingestion import UniverseConfig, make_data_ingestion_node
        from execution_workflow import make_execution_trigger
        from firm_graph import build_compiled_graph

        # Bind the Claude client ONCE here — this is exactly the "graph-build
        # time, not per-call" binding make_strategist_reasoning_node() and
        # make_analyst_node() were designed around, so these closures never
        # need to round-trip through checkpointed state.
        reason_strategic, reason_tactical = make_claude_reasoners()
        reason_analyst = make_claude_analyst_reasoner()   # one closure, all three desks

        # Market/news data: Alpaca covers equities + crypto (NOT forex — see
        # alpaca_clients.py's module docstring; a second vendor is needed
        # there). Reads ALPACA_API_KEY/ALPACA_SECRET_KEY from the
        # environment; if unset, ingestion is skipped entirely rather than
        # silently running on fake data.
        market_client = news_client = None
        if os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"):
            from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
            from alpaca.data.historical.news import NewsClient
            from alpaca_clients import AlpacaMarketDataClient, AlpacaNewsClient
            api_key, secret_key = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
            market_client = AlpacaMarketDataClient(
                StockHistoricalDataClient(api_key, secret_key),
                CryptoHistoricalDataClient(api_key, secret_key),
            )
            # AlpacaNewsClient's import path/response shape verified against
            # a real live call (alpaca-py 0.44.0) — see its docstring in
            # alpaca_clients.py for the two bugs that were fixed there.
            news_client = AlpacaNewsClient(NewsClient(api_key, secret_key))

        # Broad candidate universe — real S&P 500 constituents per sector, so
        # the analyst reasoner is actually picking "top opportunities" from a
        # real pool (per user request), not a hand-picked shortlist. Lives in
        # universe.py (bundled via extra_packages below) — the SAME file
        # web_ui.py/simulate.py import locally, so there's exactly one list
        # to keep accurate rather than two copies that can drift apart.
        from universe import EQUITIES, THEMES
        universe = UniverseConfig(equities=EQUITIES, forex=[], crypto=[], themes=THEMES)
        # max_workers=3: started at 4 (empirically safe for a single burst of
        # ~450 price+news calls at 223 symbols), but a day of heavy testing
        # against this same account pushed it into repeated rate-limit
        # retries even at 4 — a single request went from ~90s to 5m43s once
        # alpaca_clients.py's retry-with-backoff had to absorb several 429s.
        # 3 is a proactive step down to reduce how often that backoff fires
        # at all, rather than just retrying through it more. If real usage
        # still hits this, the fix is either fewer workers still or batching
        # requests (StockBarsRequest/NewsRequest both accept multiple
        # symbols per call — see fetch_price_series in simulate.py for the
        # batched pattern) — this file still does one call per symbol.
        ingestion_node = (
            make_data_ingestion_node(universe, market_client, news_client, max_workers=3)
            if market_client and news_client else None
        )

        # get_current_positions: real holdings, from the same Alpaca account
        # the broker trades through — equities/crypto only, same forex gap
        # as get_broker_client(). Falls back to "no positions" if Alpaca
        # credentials aren't configured, rather than guessing.
        def get_current_positions() -> Dict[str, float]:
            if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
                return {}
            from alpaca.trading.client import TradingClient
            from alpaca_clients import AlpacaBrokerClient
            tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"],
                               paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true")
            return AlpacaBrokerClient(tc).get_current_positions(self.total_capital)
        execute_trigger = (
            make_execution_trigger(
                temporal_target=self.temporal_target,
                total_capital=self.total_capital,
                get_current_positions=get_current_positions,
            )
            if self.temporal_target else None
        )

        # Durable checkpointer: a run paused at a freshness-gate or risk
        # escalation must survive a container restart, not just live for
        # this process. Firestore is schema-less — unlike PostgresSaver there's
        # no .setup() migration step; the collection is created on first write.
        # Requires roles/datastore.user on Agent Engine's runtime service
        # account (by default the "Vertex AI Reasoning Engine Service Agent",
        # service-<PROJECT_NUMBER>@gcp-sa-aiplatform-re.iam.gserviceaccount.com)
        # for the Firestore database in firestore_project.
        self.checkpointer = FirestoreSaver(
            project_id=self.firestore_project,
            checkpoints_collection=self.checkpoints_collection,
        )

        # Restricts checkpoint deserialization to known-safe types — matters
        # because a compromised DB shouldn't be able to achieve code
        # execution via a poisoned checkpoint payload. Current LangGraph
        # docs call this out explicitly; set it before graphs start running.
        os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

        self.graph = build_compiled_graph(
            checkpointer=self.checkpointer,
            reason_strategic=reason_strategic,
            reason_tactical=reason_tactical,
            reason_equities=reason_analyst,
            reason_forex=reason_analyst,
            reason_crypto=reason_analyst,
            ingestion_node=ingestion_node,
            execute_trigger=execute_trigger,
        )

    def query(self, *, thread_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """One-shot invocation. `thread_id` ties this call to a persisted
        checkpoint — reusing it resumes a run paused at an interrupt (e.g.
        the freshness_gate or risk human-approval nodes).

        NOTE: if execute_trigger ended up async (temporal_target is set —
        see set_up()), the graph's execute node is itself async, and a sync
        .invoke() will raise once execution reaches that node. Use
        async_query() below instead whenever temporal_target is configured."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(_coerce_firm_state(state), config=config)

    async def async_query(self, *, thread_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Vertex AI Agent Engine's documented convention for async agent
        methods: define one named async_query, and the platform's SDK
        exposes it as agent.async_query(...) to callers. This is the path
        that actually supports an async execute node — .ainvoke() awaits
        async nodes directly and runs sync nodes normally, so this works
        whether or not execute_trigger is configured. Use this method
        (not query()) once temporal_target is set in __init__."""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.ainvoke(_coerce_firm_state(state), config=config)

    def resume(self, *, thread_id: str, human_response: str) -> Dict[str, Any]:
        """Resume a run that's paused at an interrupt() call — e.g. the
        person answering a freshness-gate escalation with
        're_affirm:energy' / 'override' / 'revise:...'."""
        from langgraph.types import Command
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(Command(resume=human_response), config=config)

    def stream_query(self, *, thread_id: str, state: Dict[str, Any]):
        """Streaming variant — yields each node's output as the graph runs,
        useful for a live dashboard showing 'strategist thinking... risk
        checking... awaiting approval' in real time. Default stream_mode
        ("updates") yields {node_name: partial_state} per completed node —
        NOT the accumulated final state, so a caller driving a progress UI
        off this still needs get_final_state() below once the generator is
        exhausted."""
        config = {"configurable": {"thread_id": thread_id}}
        for chunk in self.graph.stream(_coerce_firm_state(state), config=config):
            yield chunk

    def get_final_state(self, *, thread_id: str) -> Dict[str, Any]:
        """Reads back the accumulated final state for a thread_id from the
        checkpointer (Firestore in production) WITHOUT re-running the graph
        — pairs with stream_query() above: drain that generator for live
        progress, then call this once to get the same complete result
        query() would have returned for an equivalent invoke()."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.get_state(config).values

    def register_operations(self) -> Dict[str, List[str]]:
        """REQUIRED as soon as any custom method beyond query/stream_query
        needs to be remotely callable: Vertex AI Agent Engine only
        auto-registers query() (standard mode) and stream_query() (stream
        mode) for free — confirmed by reading _reasoning_engines.py's
        _get_registered_operations/_register_api_methods_or_actions, which
        default to exactly those two UNLESS this method exists, in which
        case its return value becomes the SOLE source of truth (so query
        must be re-listed here or it silently stops being callable).
        get_final_state pairs with stream_query for a live-progress UI."""
        return {
            "": ["query", "get_final_state"],
            "stream": ["stream_query"],
        }


# Shared between build_and_deploy() (first-time create) and
# update_deployed_agent() (push a code/dependency change to the existing
# live resource) so the two never drift apart.
_REQUIREMENTS = [
    "google-cloud-aiplatform[reasoningengine,agent-engines]",
    "langgraph",
    "langgraph-checkpoint-firestore",   # Firestore-backed checkpointer (community package,
                                        # not an official LangChain/Google release — see
                                        # set_up()'s comment above the FirestoreSaver() call)
    "google-cloud-firestore",
    "google-cloud-secret-manager",      # fetches Alpaca creds — see _resolve_secret()
    "google-genai",        # Gemini on Vertex — the default provider (llm.py)
    "anthropic[vertex]",   # [vertex] extra needed for AnthropicVertex; drop it if
                          # use_direct_anthropic_api=True and you only need the base package
    "alpaca-py",           # equities + crypto broker, market data, (not forex)
    "yfinance",            # real per-instrument fundamentals — see fundamentals.py
    "scipy",               # mean-variance weight optimization — see portfolio_optimizer.py
    "pyyaml",
    # Cloud Trace export for set_up()'s tracer setup (enable_tracing=True) —
    # harmless to install even when tracing is left off, since the import
    # only happens inside that `if self.enable_tracing:` block.
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-gcp-trace",
]
_EXTRA_PACKAGES = [
    "universe.py",            # shared candidate universe — see set_up()'s import comment
    "fundamentals.py",        # real quality_score from yfinance fundamentals, cached
    "vertex_deploy.py",       # InvestmentFirmAgent itself lives here — must be bundled or
                             # the container can't unpickle the agent ("No module named
                             # 'vertex_deploy'") when it boots
    "mandate_allocator.py",
    "portfolio_optimizer.py", # real mean-variance weight optimization, called from
                             # mandate_allocator._allocate_within_bucket
    "strategist.py",
    "macro_reasoning.py",     # MacroSnapshot + heuristic reasoning (strategist's fallback path)
    "analyst_reasoning.py",   # InstrumentSnapshot + heuristic scoring (analysts' fallback path)
    "claude_reasoning.py",    # LLM-backed reasoning for both strategist and analysts
    "llm.py",                 # provider switch: Gemini (default) behind the Anthropic call shape
    "data_ingestion.py",      # turns market/news data into snapshots
    "alpaca_clients.py",      # real broker/market-data clients (equities+crypto; no forex yet)
    "execution_workflow.py",  # order-delta + saga logic; the actual Temporal worker deploys
                             # SEPARATELY (Cloud Run/GKE), this is bundled for the pure-logic
                             # parts + as a reference for building that separate worker
    "firm_graph.py",          # build_compiled_graph() lives here
    "promotion_gate.py",      # offline champion/challenger — not on this query path,
                             # but bundled so a scheduled promotion job can import it too
]


def build_and_deploy(
    project: str,
    location: str,
    staging_bucket: str,
    display_name: str = "ai-investment-firm",   # swap in whatever name you land on
    alpaca_api_key_secret: Optional[str] = None,
    alpaca_secret_key_secret: Optional[str] = None,
    enable_tracing: bool = True,
) -> "Any":
    """Deploys InvestmentFirmAgent to Vertex AI Agent Engine.

    Deployment takes a few minutes: it builds a container, installs
    `requirements`, and stands up an HTTP server on Google's managed
    infra. The returned object is a REMOTE handle — querying it goes
    over the network to the deployed instance, not your local process.

    alpaca_api_key_secret / alpaca_secret_key_secret: Secret Manager resource
    names, e.g. "projects/P/secrets/alpaca-api-key/versions/latest" — NOT raw
    key material. Left as None, Alpaca stays disabled on the deployed agent
    (same as before). The runtime service account (by default the Reasoning
    Engine Service Agent) needs roles/secretmanager.secretAccessor on those
    two secrets — this function doesn't grant that, it assumes it's already
    done.

    enable_tracing: see InvestmentFirmAgent.set_up()'s Cloud Trace setup.
    Requires telemetry.googleapis.com + cloudtrace.googleapis.com enabled on
    `project`, and roles/cloudtrace.agent on the runtime service account —
    this function doesn't grant either, it assumes they're already done.
    Creating a brand-new resource this way is a ONE-TIME operation per
    resource; to push a change (like turning tracing on) to the agent
    web_ui.py is already pointed at, use update_deployed_agent() below
    instead — it preserves the resource name.
    """
    import vertexai
    from vertexai.preview import reasoning_engines  # or `agent_engines` — check current docs

    vertexai.init(project=project, location=location, staging_bucket=staging_bucket)

    remote_agent = reasoning_engines.ReasoningEngine.create(
        InvestmentFirmAgent(
            project=project, location=location,
            alpaca_api_key_secret=alpaca_api_key_secret,
            alpaca_secret_key_secret=alpaca_secret_key_secret,
            enable_tracing=enable_tracing,
        ),
        requirements=_REQUIREMENTS,
        extra_packages=_EXTRA_PACKAGES,
        display_name=display_name,
        description="Multi-agent investment firm: research, strategist "
                    "(Claude-backed, two-horizon), risk gate, execution — "
                    "orchestrated via LangGraph",
    )
    return remote_agent


def update_deployed_agent(
    resource_name: str,
    project: str,
    location: str,
    staging_bucket: str,
    alpaca_api_key_secret: Optional[str] = None,
    alpaca_secret_key_secret: Optional[str] = None,
    enable_tracing: bool = True,
) -> "Any":
    """Pushes the CURRENT InvestmentFirmAgent code + _REQUIREMENTS/
    _EXTRA_PACKAGES to the agent already deployed at `resource_name`, in
    place — unlike build_and_deploy()'s .create(), this keeps the same
    resource name/ID, so web_ui.py's MERIDIAN_RESOURCE_NAME never needs to
    change and the live web app keeps working against the same agent
    through the update (a brief redeploy of that one resource, not a new
    one). resource_name is the full path, e.g.
    "projects/P/locations/L/reasoningEngines/12345".

    Uses vertexai.agent_engines (not vertexai.preview.reasoning_engines) —
    the newer module, needed here because it's the one whose .update() this
    was verified against locally (see this file's own module docstring
    about the import path moving between the two)."""
    import vertexai
    from vertexai import agent_engines

    vertexai.init(project=project, location=location, staging_bucket=staging_bucket)

    return agent_engines.update(
        resource_name,
        agent_engine=InvestmentFirmAgent(
            project=project, location=location,
            alpaca_api_key_secret=alpaca_api_key_secret,
            alpaca_secret_key_secret=alpaca_secret_key_secret,
            enable_tracing=enable_tracing,
        ),
        requirements=_REQUIREMENTS,
        extra_packages=_EXTRA_PACKAGES,
    )


# ----------------------------------------------------------------------------
# What monitoring you get "for free" vs. what you still need to add
# ----------------------------------------------------------------------------
# FREE (Agent Engine manages this):
#   - Request/response logging, latency, error rates, autoscaling
#   - Checkpoint persistence for interrupted runs (now backed by Firestore,
#     not MemorySaver — see set_up())
#   - IAM-based access control on who can query/resume a run
#
# YOU STILL NEED:
#   - Agent-reasoning-level tracing (why did the strategist pick this tilt?)
#     — Agent Engine's built-in observability is infra-level, not reasoning-
#     level. Run LangSmith alongside it (set LANGCHAIN_TRACING_V2=true in
#     set_up()) if you want the same per-node rationale trace described
#     earlier — the two are complementary, not exclusive.
#   - roles/datastore.user granted to Agent Engine's runtime service account
#     on the Firestore database in firestore_project — this file assumes the
#     database + IAM binding already exist, it doesn't create either.
#   - Real MarketDataClient/NewsClient implementations and your actual
#     UniverseConfig (which symbols/themes this firm trades) — Alpaca now
#     covers market data for equities+crypto (see alpaca_clients.py);
#     AlpacaNewsClient's exact import path is still unverified (flagged in
#     its docstring), so news_client stays None until that's confirmed.
#     UniverseConfig itself is still empty — fill in your actual universe.
#   - FOREX has no broker/data vendor wired in at all — Alpaca doesn't
#     offer it. A second vendor (OANDA is the common pairing) is needed
#     for forex_snapshots and any forex order to actually work.
#   - Set ALPACA_API_KEY/ALPACA_SECRET_KEY (Secret Manager, not hardcoded)
#     on both this container AND the separate Temporal worker process —
#     both get_current_positions() here and get_broker_client() in
#     execution_workflow.py read them from the environment.
#   - A separately-deployed Temporal worker (Cloud Run/GKE — NOT hosted by
#     Agent Engine) running OrderExecutionWorkflow. Setting temporal_target
#     activates execute_trigger automatically; the worker on the other end
#     still needs to exist and be running.
#   - Callers must use async_query(), not query(), whenever temporal_target
#     is set — a sync .invoke() cannot await the resulting async execute node.


if __name__ == "__main__":
    print(__doc__)

    print("=" * 70)
    print("This file has no standalone-testable pure logic left — switching")
    print("the checkpointer to Firestore removed the pg_conn_string/")
    print("pg_secret_name branching that used to be exercised here (Firestore")
    print("auth rides on ADC, nothing to resolve). Everything below needs")
    print("`pip install google-cloud-aiplatform[reasoningengine,agent-engines]")
    print("langgraph langgraph-checkpoint-firestore google-cloud-firestore")
    print("anthropic[vertex]` and real GCP credentials — call, once you have")
    print("those (plus a Firestore Native-mode database and")
    print("roles/datastore.user for Agent Engine's runtime service account):")
    print("=" * 70)
    print()
    print('    build_and_deploy(project="my-project", location="us-central1",')
    print('                     staging_bucket="gs://my-bucket")')
