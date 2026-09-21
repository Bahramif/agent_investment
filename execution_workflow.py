"""
execution_workflow.py
=======================
The execution/settlement layer — deliberately separate from the LangGraph
thinking layer, per this project's original design: LangGraph produces a
target book (mandate_allocator.Book); everything from "turn that into
orders" onward needs a different kind of guarantee — exactly-once
execution, survives a crash mid-order, retries transient broker failures,
and reconciles what actually happened against what was intended.

WHY TEMPORAL, NOT GOOGLE WORKFLOWS: this saga (submit -> await fill ->
retry on transient failure -> reconcile -> halt-or-continue on repeated
failure) needs first-class support for per-step retry policies, timers, and
mid-saga compensation logic. Google Workflows suits simpler, mostly-linear
API orchestration well; it doesn't have Temporal's equivalent of a durable,
replayable workflow function with this level of per-step control built in.
If you want to stay entirely inside GCP without an external service,
Workflows remains a lighter (less powerful) alternative — mentioned, not
built here.

WHERE THIS RUNS: a Temporal worker is its OWN separate service — typically
on Cloud Run or GKE, with its own deployment — not something Vertex AI
Agent Engine hosts alongside the LangGraph agent. firm_graph.py's
execute_node hands off to this layer and returns immediately; it does not
wait for orders to fill.

HONEST LIMITATION, same shape as elsewhere in this project: everything
that's PURE LOGIC — compute_order_deltas() and the saga's retry/halt
decisions in execute_orders_saga() — is fully tested below, no temporalio
needed. Only the @workflow.defn / @activity.defn code at the bottom touches
the real Temporal SDK and is UNTESTED in this sandbox (no network to `pip
install temporalio`, no Temporal server to run against). It's written to
the current documented SDK shape (verified via Temporal's docs); confirm
against those docs before relying on it, since the API can move.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Optional, Protocol

from mandate_allocator import Book


# -----------------------------------------------------------------------------
# Orders — the output of comparing the target book to current holdings.
# -----------------------------------------------------------------------------
class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    symbol: str
    side: Side
    notional: float          # dollar amount to trade — sizing into shares/lots is
                            # an execution-algo detail, left to the broker layer
    qty: Optional[float] = None   # WHOLE-SHARE order instead of notional/fractional —
                                  # set by web_ui.py's Accept & Execute flow, which
                                  # rounds to whole units per user request. When set,
                                  # alpaca_clients.py's _make_order_request uses qty
                                  # instead of notional; notional is still carried for
                                  # display/sizing math either way.
    client_order_id: str = ""   # stable across Temporal activity retries — see
                                # compute_order_deltas's run_id param. A random
                                # ID here would let a retried activity submit a
                                # SECOND real order at the broker instead of
                                # safely re-checking the first one's status.
    # BRACKET (long entry with protective exits attached at the broker) — set
    # BOTH or neither. Used only by the day-trading capability
    # (daytrade_broker.py); every existing caller leaves these None and gets
    # the plain market order it always did. Requires qty (whole shares).
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None


def compute_order_deltas(
    current_positions: Dict[str, float],   # symbol -> current weight of total capital
    target_weights: Dict[str, float],      # symbol -> target weight (from mandate_allocator.Book)
    total_capital: float,
    min_trade_notional: float = 50.0,      # skip trades too small to be worth the friction
    run_id: str = "",                      # pass workflow.info().workflow_id from inside a
                                           # Temporal workflow — same run_id in, same
                                           # client_order_id out, every time this is
                                           # recomputed (workflow code re-executes on
                                           # replay, so this MUST be deterministic; a
                                           # fresh random ID here would break idempotency)
) -> List[Order]:
    """Pure function: the target book minus current holdings, in dollars.
    Deterministic, so it's safe to call directly inside a Temporal workflow
    body (workflow code must be deterministic — this qualifies; the broker
    calls below do not, which is exactly why THOSE go through Activities)."""
    orders: List[Order] = []
    symbols = set(current_positions) | set(target_weights)
    for symbol in sorted(symbols):   # sorted: deterministic order, matters for replay
        current = current_positions.get(symbol, 0.0)
        target = target_weights.get(symbol, 0.0)
        delta_notional = (target - current) * total_capital
        if abs(delta_notional) < min_trade_notional:
            continue
        side = Side.BUY if delta_notional > 0 else Side.SELL
        client_order_id = f"{run_id}-{symbol}-{side.value}" if run_id else ""
        orders.append(Order(symbol=symbol, side=side, notional=abs(delta_notional),
                            client_order_id=client_order_id))
    return orders


# -----------------------------------------------------------------------------
# Broker contract — bring your own client (Alpaca, IBKR, etc.) satisfying this.
# -----------------------------------------------------------------------------
class OrderResult(str, Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"      # a business decision (insufficient funds, bad
                               # symbol) — NOT something to blindly retry
    TIMEOUT = "timeout"         # transient — safe to retry


@dataclass
class Fill:
    order: Order
    result: OrderResult
    filled_notional: float
    reason: str = ""
    # ACTUAL shares that filled — may be LESS than order.qty for a genuine
    # partial fill (result == PARTIAL). None when unknown (e.g. TIMEOUT, or
    # a BrokerClient implementation that predates this field) — callers
    # that record holdings should fall back to order.qty only in that case,
    # never when a real filled_qty is available, or a genuine partial fill
    # misrecords MORE shares owned than were actually bought.
    filled_qty: Optional[float] = None


class BrokerClient(Protocol):
    def submit_and_await(self, order: Order) -> Fill: ...
    def cancel(self, order: Order) -> None: ...


# -----------------------------------------------------------------------------
# The saga's decision logic — pure, and the part actually worth testing
# rigorously. This mirrors exactly what the Temporal workflow at the bottom
# does, so this function is both a spec for the workflow AND a fully-tested
# stand-in for it in environments without temporalio installed.
# -----------------------------------------------------------------------------
@dataclass
class ExecutionReport:
    filled: List[Fill] = field(default_factory=list)
    rejected: List[Fill] = field(default_factory=list)
    cancelled: List[Order] = field(default_factory=list)
    halted: bool = False
    halt_reason: str = ""


def execute_orders_saga(
    orders: List[Order],
    broker: BrokerClient,
    max_retries_per_order: int = 3,
    halt_threshold_fraction: float = 0.25,
) -> ExecutionReport:
    """Submit each order, retrying transient failures up to
    max_retries_per_order times. If REJECTED/TIMEOUT orders accumulate past
    halt_threshold_fraction of total order notional, STOP submitting further
    orders and cancel what's outstanding — mirrors the risk gate's
    "propose vs. decide" ethos elsewhere in this project: a run of failures
    is itself a signal something's wrong, not just noise to retry through
    forever. Below the threshold, a rejected order doesn't block the rest
    of the book — it's logged and the saga continues."""
    report = ExecutionReport()
    total_notional = sum(o.notional for o in orders) or 1.0
    failed_notional = 0.0

    for order in orders:
        if report.halted:
            # Every order after halt goes through exactly HERE, exactly
            # once — nothing pre-cancels orders at the moment the
            # threshold is crossed, which is what caused a duplication
            # bug in an earlier version of this function (an order got
            # appended to `cancelled` both proactively and again here).
            report.cancelled.append(order)
            broker.cancel(order)
            continue

        fill = None
        for _attempt in range(max_retries_per_order):
            fill = broker.submit_and_await(order)
            if fill.result in (OrderResult.FILLED, OrderResult.PARTIAL):
                break
            if fill.result == OrderResult.REJECTED:
                break   # business rejection — retrying won't help, don't waste attempts
            # TIMEOUT: loop again (transient) until attempts run out

        if fill.result in (OrderResult.FILLED, OrderResult.PARTIAL):
            report.filled.append(fill)
            continue

        report.rejected.append(fill)
        failed_notional += order.notional
        if failed_notional / total_notional >= halt_threshold_fraction:
            report.halted = True
            report.halt_reason = (
                f"failed notional ${failed_notional:,.0f} reached "
                f"{failed_notional / total_notional:.0%} of ${total_notional:,.0f} total "
                f"(threshold {halt_threshold_fraction:.0%}) — halting remaining orders"
            )
            # Note: we do NOT touch orders[i+1:] here. Halting just flips
            # the flag; every order still to come in this loop hits the
            # `if report.halted` branch above on its own turn.

    return report


# -----------------------------------------------------------------------------
# Fake broker for testing — configurable per-symbol behavior, no network.
# -----------------------------------------------------------------------------
class FakeBrokerClient:
    """behaviors[symbol] = list of OrderResults returned on successive calls
    to that symbol (repeats the last entry if called more times than given —
    e.g. ["timeout", "timeout", "filled"] fails twice, then succeeds, which
    is exactly the retry-then-recover case worth testing)."""
    def __init__(self, behaviors: Dict[str, List[OrderResult]]):
        self.behaviors = behaviors
        self.call_counts: Dict[str, int] = {}
        self.cancelled: List[str] = []

    def submit_and_await(self, order: Order) -> Fill:
        seq = self.behaviors.get(order.symbol, [OrderResult.FILLED])
        n = self.call_counts.get(order.symbol, 0)
        result = seq[min(n, len(seq) - 1)]
        self.call_counts[order.symbol] = n + 1
        filled_notional = (
            order.notional if result == OrderResult.FILLED
            else order.notional * 0.5 if result == OrderResult.PARTIAL
            else 0.0
        )
        return Fill(order, result, filled_notional,
                    reason="" if result == OrderResult.FILLED else f"simulated {result.value}")

    def cancel(self, order: Order) -> None:
        self.cancelled.append(order.symbol)


# -----------------------------------------------------------------------------
# Temporal availability check + the Book -> workflow-ID adapter. Placed here,
# before __main__, deliberately: make_execution_trigger() and
# ExecutionWorkflowInput are plain data/logic with NO temporalio dependency,
# so they (and their tests below) work regardless of whether temporalio is
# installed. Only the @workflow.defn/@activity.defn code further down needs
# the real package — see the guard there.
# -----------------------------------------------------------------------------
try:
    from datetime import timedelta
    from temporalio import activity, workflow
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ActivityError, ApplicationError
    from temporalio.client import Client
    _TEMPORAL_AVAILABLE = True
except ImportError:
    _TEMPORAL_AVAILABLE = False


def get_broker_client() -> BrokerClient:
    """Constructs the real broker client for the Temporal worker process.

    Alpaca covers equities, crypto, and options in one account (see
    alpaca_clients.py) — but NOT forex. This mandate includes forex, which
    stays unhandled by this function until a second broker (e.g. OANDA) is
    wired in; an order for a forex pair reaching here will fail loudly via
    the ImportError/KeyError below rather than silently going nowhere.

    Reads ALPACA_API_KEY / ALPACA_SECRET_KEY from the environment (set them
    via Secret Manager in the worker's deployment config, same pattern as
    vertex_deploy.py's pg_secret_name — never hardcode them here)."""
    import os
    from alpaca.trading.client import TradingClient
    from alpaca_clients import AlpacaBrokerClient
    trading_client = TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"],
        paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true",
    )
    return AlpacaBrokerClient(trading_client)


def write_execution_report_to_ledger(report: dict) -> None:
    """Persist the execution report to your data layer (Postgres, BigQuery,
    etc.) so it shows up in the same audit trail as every other decision
    this firm makes. Integration point, not implemented here."""
    raise NotImplementedError("wire up your ledger/DB write here")


@dataclass
class ExecutionWorkflowInput:
    """Plain data, no Temporal dependency — deliberately defined outside the
    _TEMPORAL_AVAILABLE guard further down, unlike the @workflow.defn/
    @activity.defn code, so make_execution_trigger() and its tests don't
    need temporalio installed at all."""
    current_positions: Dict[str, float]
    target_weights: Dict[str, float]
    total_capital: float
    min_trade_notional: float = 50.0
    halt_threshold_fraction: float = 0.25


def make_execution_trigger(
    temporal_target: str,
    total_capital: float,
    get_current_positions: Callable[[], Dict[str, float]],
    halt_threshold_fraction: float = 0.25,
    min_trade_notional: float = 50.0,
    _trigger_execution_fn: Optional[Callable[[ExecutionWorkflowInput, str], Awaitable[str]]] = None,
) -> Callable[[Book], Awaitable[str]]:
    """The adapter firm_graph.py's execute_trigger actually needs: Book ->
    workflow ID. Bridges the gap between what LangGraph produces (a target
    book — weights only) and what the workflow needs (target weights AND
    current positions AND total capital) by calling get_current_positions()
    fresh on every trigger — current holdings change every cycle, unlike
    temporal_target/total_capital, which are static config bound once here.

    _trigger_execution_fn exists so this adapter is testable without
    temporalio: pass a fake async function in tests; production code leaves
    it as None and the real trigger_execution (further down, behind the
    _TEMPORAL_AVAILABLE guard) is resolved at call time.

    Returns an ASYNC callable. firm_graph.make_execute_node() detects this
    (asyncio.iscoroutinefunction) and produces an async graph node
    automatically — see its docstring."""
    async def _trigger(book: Book) -> str:
        fn = _trigger_execution_fn
        if fn is None:
            if not _TEMPORAL_AVAILABLE:
                raise RuntimeError(
                    "temporalio is not installed, so there is no real trigger_execution "
                    "to fall back to. Pass _trigger_execution_fn explicitly for testing, "
                    "or install temporalio for real use."
                )
            fn = trigger_execution
        wf_input = ExecutionWorkflowInput(
            current_positions=get_current_positions(),
            target_weights=dict(book.weights),
            total_capital=total_capital,
            min_trade_notional=min_trade_notional,
            halt_threshold_fraction=halt_threshold_fraction,
        )
        return await fn(wf_input, temporal_target)
    return _trigger


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: compute_order_deltas — buy, sell, and skip-small-trade cases")
    print("=" * 70)
    current = {"NVDA": 0.05, "CAT": 0.10}
    target = {"NVDA": 0.15, "CAT": 0.10, "XOM": 0.0002}   # NVDA: buy; CAT: unchanged; XOM: $20 delta, below the $50 floor
    orders = compute_order_deltas(current, target, total_capital=100_000, min_trade_notional=50)
    print(f"  {[(o.symbol, o.side.value, o.notional) for o in orders]}")
    assert len(orders) == 1 and orders[0].symbol == "NVDA" and orders[0].side == Side.BUY
    assert abs(orders[0].notional - 10_000) < 1e-6
    print("✓ only NVDA trades; CAT is already at target; XOM's delta is below the trade-size floor")

    print("\n" + "=" * 70)
    print("TEST 1b: client_order_id is deterministic given the same run_id —")
    print("required for idempotent resubmission across Temporal activity retries")
    print("=" * 70)
    orders_a = compute_order_deltas(current, target, 100_000, run_id="wf-abc123")
    orders_b = compute_order_deltas(current, target, 100_000, run_id="wf-abc123")
    print(f"  run 1: {orders_a[0].client_order_id}")
    print(f"  run 2: {orders_b[0].client_order_id}")
    assert orders_a[0].client_order_id == orders_b[0].client_order_id != ""
    print("✓ same run_id -> identical client_order_id — a retried activity re-checks")
    print("  the SAME broker order instead of submitting a second real one")

    print("\n" + "=" * 70)
    print("TEST 2: happy path — every order fills immediately")
    print("=" * 70)
    orders2 = [Order("NVDA", Side.BUY, 10_000), Order("CAT", Side.SELL, 5_000)]
    broker_ok = FakeBrokerClient({})   # default behavior: everything fills
    report2 = execute_orders_saga(orders2, broker_ok)
    print(f"  filled={len(report2.filled)} rejected={len(report2.rejected)} halted={report2.halted}")
    assert len(report2.filled) == 2 and not report2.halted
    print("✓ clean run, nothing to retry or halt")

    print("\n" + "=" * 70)
    print("TEST 3: transient failure recovers within the retry budget")
    print("=" * 70)
    broker_flaky = FakeBrokerClient({"NVDA": [OrderResult.TIMEOUT, OrderResult.TIMEOUT, OrderResult.FILLED]})
    report3 = execute_orders_saga([Order("NVDA", Side.BUY, 10_000)], broker_flaky, max_retries_per_order=3)
    print(f"  NVDA attempts made: {broker_flaky.call_counts['NVDA']}, final result: filled={len(report3.filled)}")
    assert len(report3.filled) == 1 and not report3.halted
    print("✓ two timeouts, then a fill within the retry budget — order succeeds")

    print("\n" + "=" * 70)
    print("TEST 4: a small rejection doesn't halt the rest of the book")
    print("=" * 70)
    orders4 = [Order("SMALLCO", Side.BUY, 1_000), Order("NVDA", Side.BUY, 20_000)]
    broker_small_reject = FakeBrokerClient({"SMALLCO": [OrderResult.REJECTED]})
    report4 = execute_orders_saga(orders4, broker_small_reject, halt_threshold_fraction=0.25)
    print(f"  filled={len(report4.filled)} rejected={len(report4.rejected)} halted={report4.halted}")
    assert len(report4.rejected) == 1 and len(report4.filled) == 1 and not report4.halted
    print("✓ SMALLCO's rejection is well under the 25% halt threshold — NVDA still executes")

    print("\n" + "=" * 70)
    print("TEST 5: a large rejection crosses the halt threshold and cancels the rest")
    print("=" * 70)
    orders5 = [Order("BIGCO", Side.BUY, 30_000), Order("NVDA", Side.BUY, 10_000), Order("CAT", Side.SELL, 10_000)]
    broker_big_reject = FakeBrokerClient({"BIGCO": [OrderResult.REJECTED]})
    report5 = execute_orders_saga(orders5, broker_big_reject, halt_threshold_fraction=0.25)
    print(f"  filled={len(report5.filled)} rejected={len(report5.rejected)} "
          f"cancelled={[o.symbol for o in report5.cancelled]} halted={report5.halted}")
    print(f"  halt_reason: {report5.halt_reason}")
    assert report5.halted and len(report5.cancelled) == 2
    assert set(broker_big_reject.cancelled) == {"NVDA", "CAT"}
    print("✓ BIGCO's rejection alone is 60% of total notional (well past the 25% threshold) —")
    print("  remaining orders are cancelled via broker.cancel(), not silently dropped")

    print("\n" + "=" * 70)
    print("TEST 6: make_execution_trigger() — the Book -> workflow-ID adapter")
    print("(fully testable without temporalio: _trigger_execution_fn is a fake)")
    print("=" * 70)
    import asyncio

    call_log = []

    async def fake_trigger_execution(wf_input: ExecutionWorkflowInput, target: str) -> str:
        call_log.append((wf_input, target))
        return "wf-fake-123"

    def fake_get_positions() -> Dict[str, float]:
        return {"NVDA": 0.05, "CASH": 0.95}

    trigger = make_execution_trigger(
        temporal_target="fake-temporal:7233",
        total_capital=100_000.0,
        get_current_positions=fake_get_positions,
        _trigger_execution_fn=fake_trigger_execution,
    )
    book = Book(weights={"NVDA": 0.15, "CASH": 0.85})
    workflow_id = asyncio.run(trigger(book))
    print(f"  returned workflow_id: {workflow_id}")
    assert workflow_id == "wf-fake-123" and len(call_log) == 1
    wf_input, target = call_log[0]
    assert wf_input.current_positions == {"NVDA": 0.05, "CASH": 0.95}
    assert wf_input.target_weights == {"NVDA": 0.15, "CASH": 0.85}
    assert wf_input.total_capital == 100_000.0 and target == "fake-temporal:7233"
    print("✓ current positions fetched fresh, book converted to target_weights correctly,")
    print("  static config (total_capital, temporal_target) passed through unchanged")

    print()
    print("=" * 70)
    print("Everything above is genuinely tested. Below this line, the real")
    print("Temporal workflow/activity code needs `pip install temporalio` and")
    print("a running Temporal server/Cloud account — untestable in this sandbox.")
    print("=" * 70)


# =============================================================================
# Real Temporal SDK code below this point — UNTESTED here (no `temporalio`
# package, no Temporal server — see the limitation note at the top of this
# file). Guarded by an availability check, same pattern as the langgraph-
# dependent code elsewhere in this project: this is REAL code, syntax-
# checked, that activates if temporalio is actually installed; it does not
# run in this sandbox. Written to the current documented SDK shape; verify
# against Temporal's docs before relying on it, since the API can move.
# =============================================================================
if _TEMPORAL_AVAILABLE:

    @activity.defn
    async def submit_and_await_activity(order_dict: dict) -> dict:
        """Real broker call. A TRANSIENT failure (network blip, rate limit)
        should raise a plain exception so Temporal's RetryPolicy retries the
        ACTIVITY CALL itself. A definitive business REJECTION (insufficient
        funds, invalid symbol) must raise ApplicationError(non_retryable=True)
        so Temporal does NOT blindly retry submitting an order that's
        already been rejected — REJECTED and TIMEOUT are handled differently
        for exactly this reason in execute_orders_saga() above."""
        order = Order(symbol=order_dict["symbol"], side=Side(order_dict["side"]),
                     notional=order_dict["notional"])
        broker = get_broker_client()
        fill = broker.submit_and_await(order)   # may itself raise on transient errors
        if fill.result == OrderResult.REJECTED:
            raise ApplicationError(f"order rejected: {fill.reason}", non_retryable=True)
        return {"symbol": order.symbol, "result": fill.result.value,
               "filled_notional": fill.filled_notional}

    @activity.defn
    async def cancel_activity(order_dict: dict) -> None:
        broker = get_broker_client()
        broker.cancel(Order(symbol=order_dict["symbol"], side=Side(order_dict["side"]),
                            notional=order_dict["notional"]))

    @activity.defn
    async def reconcile_activity(report: dict) -> None:
        """Compare expected vs. actual positions after the saga, write the
        result to the ledger. Left as a stub — see write_execution_report_to_ledger."""
        write_execution_report_to_ledger(report)

    @workflow.defn
    class OrderExecutionWorkflow:
        @workflow.run
        async def run(self, input: "ExecutionWorkflowInput") -> dict:
            # Deterministic — safe to call directly in workflow code, no Activity needed.
            orders = compute_order_deltas(
                input.current_positions, input.target_weights,
                input.total_capital, input.min_trade_notional,
            )
            total_notional = sum(o.notional for o in orders) or 1.0
            failed_notional = 0.0
            filled, rejected, cancelled = [], [], []

            for i, order in enumerate(orders):
                if failed_notional / total_notional >= input.halt_threshold_fraction:
                    cancelled.append(order)
                    continue
                try:
                    fill = await workflow.execute_activity(
                        submit_and_await_activity,
                        asdict(order),
                        start_to_close_timeout=timedelta(seconds=30),
                        retry_policy=RetryPolicy(maximum_attempts=3, backoff_coefficient=2.0),
                    )
                    filled.append(fill)
                except ActivityError as e:
                    rejected.append({"symbol": order.symbol, "reason": str(e.cause)})
                    failed_notional += order.notional
                    if failed_notional / total_notional >= input.halt_threshold_fraction:
                        for remaining in orders[i + 1:]:
                            await workflow.execute_activity(
                                cancel_activity, asdict(remaining),
                                start_to_close_timeout=timedelta(seconds=10),
                            )
                            cancelled.append(remaining)

            report = {"filled": filled, "rejected": rejected,
                     "cancelled": [asdict(o) for o in cancelled]}
            await workflow.execute_activity(
                reconcile_activity, report, start_to_close_timeout=timedelta(seconds=30),
            )
            return report

    async def trigger_execution(input: "ExecutionWorkflowInput", temporal_target: str) -> str:
        """The handoff point from firm_graph.py's execute_node: START the
        workflow and return its ID immediately — do NOT await completion
        here. Orders can take seconds to hours to fill; the LangGraph
        request path must never block on that."""
        import uuid
        client = await Client.connect(temporal_target)
        handle = await client.start_workflow(
            OrderExecutionWorkflow.run, input,
            id=f"execution-{uuid.uuid4()}",
            task_queue="investment-firm-execution",
        )
        return handle.id
