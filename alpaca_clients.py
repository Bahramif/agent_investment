"""
alpaca_clients.py
==================
Concrete implementations of the Protocols left open in execution_workflow.py
(BrokerClient) and data_ingestion.py (MarketDataClient, NewsClient), backed
by Alpaca — one account, one set of API keys, covering equities, crypto,
and options, with free paper trading for safe end-to-end testing before any
real capital is at risk.

WHAT ALPACA DOES NOT COVER: forex. This mandate includes forex, and Alpaca
has no forex product — a second broker (OANDA is the common pairing with
Alpaca in practice) is needed for that asset class. Not built in this pass;
forex_snapshots/get_broker_client() for forex orders stay unwired until
that's added. Said plainly so nothing here pretends forex is covered.

IDEMPOTENCY, the part most worth getting right: Temporal's RetryPolicy will
re-run submit_and_await_activity on a transient failure — including cases
where the FIRST attempt's order actually reached Alpaca but the response
was lost (a timeout on the way back, not on the way in). Resubmitting
blindly would double-buy. AlpacaBrokerClient uses order.client_order_id
(deterministic per Order — see execution_workflow.compute_order_deltas's
run_id parameter) and checks for an existing order under that ID before
ever submitting a new one.

HONEST LIMITATION, same shape as elsewhere in this project: alpaca-py isn't
installed in this sandbox (no network to pip install it), so the real API
calls are UNTESTED here. What IS tested: the status-mapping logic (Alpaca's
order-status strings -> execution_workflow.OrderResult) and the full
submit/poll/idempotent-resubmit flow, run against FakeAlpacaTradingClient —
a stand-in shaped like the real SDK's response objects, same pattern as
every other Fake*Client in this project. Only the literal network call is
unverified; the logic around it is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from execution_workflow import Order, Fill, OrderResult, Side

# Guarded, module-level — same pattern as _TEMPORAL_AVAILABLE in
# execution_workflow.py. When alpaca-py IS installed, the real request
# classes are used (production correctness). When it's not (this sandbox),
# _make_order_request below falls back to a plain SimpleNamespace with the
# same fields, so the SUBMISSION/LOOKUP LOGIC around it — idempotency,
# error handling, status polling — is still fully exercisable against
# FakeAlpacaTradingClient. Production always takes the real-class branch;
# the fallback exists purely so this file's own tests can run without the
# package present.
try:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


def _make_order_request(order: Order):
    # Alpaca's MarketOrderRequest wants EXACTLY ONE of qty/notional — never
    # both. order.qty (whole shares) takes priority when set; falls back to
    # notional (fractional/dollar-based) otherwise, same as before.
    sizing = {"qty": order.qty} if order.qty else {"notional": round(order.notional, 2)}

    # Bracket: a market BUY with a take-profit limit and a stop-loss attached
    # at the broker (day trading — see daytrade_broker.py). Only when BOTH
    # prices are set, so every pre-existing caller is untouched. Whole shares
    # only (fractional/notional orders can't carry a bracket), and BUY only —
    # this project is long-only, and a SELL bracket would be a short.
    if order.stop_price is not None and order.take_profit_price is not None:
        if not order.qty or order.qty != int(order.qty):
            raise ValueError("bracket orders need a whole-share qty (not notional/fractional)")
        if order.side != Side.BUY:
            raise ValueError("bracket orders are long-entry only (BUY); shorting is not supported")
        if not order.stop_price < order.take_profit_price:
            raise ValueError(f"bracket needs stop < target, got stop={order.stop_price} target={order.take_profit_price}")
        stop, target = round(order.stop_price, 2), round(order.take_profit_price, 2)
        if _ALPACA_AVAILABLE:
            from alpaca.trading.enums import OrderClass
            from alpaca.trading.requests import StopLossRequest, TakeProfitRequest
            return MarketOrderRequest(
                symbol=order.symbol, qty=int(order.qty), side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=target),
                stop_loss=StopLossRequest(stop_price=stop),
                client_order_id=order.client_order_id or None,
            )
        return SimpleNamespace(
            symbol=order.symbol, qty=int(order.qty), side="buy", time_in_force="day", order_class="bracket",
            take_profit=SimpleNamespace(limit_price=target), stop_loss=SimpleNamespace(stop_price=stop),
            client_order_id=order.client_order_id or None,
        )

    if _ALPACA_AVAILABLE:
        return MarketOrderRequest(
            symbol=order.symbol,
            side=OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=order.client_order_id or None,
            **sizing,
        )
    return SimpleNamespace(
        symbol=order.symbol,
        side=order.side.value, time_in_force="day",
        client_order_id=order.client_order_id or None,
        **sizing,
    )


# -----------------------------------------------------------------------------
# Status mapping — pure, and the part most worth testing carefully. Alpaca's
# order.status is a string; execution_workflow deliberately distinguishes
# REJECTED (a business decision — don't retry) from TIMEOUT (transient —
# safe to retry), so getting this mapping right matters.
# -----------------------------------------------------------------------------
_TERMINAL_FILLED = {"filled"}
_TERMINAL_PARTIAL = {"partially_filled"}
_TERMINAL_REJECTED = {"rejected", "canceled", "expired", "suspended"}
# Anything else (new, accepted, pending_new, accepted_for_bidding, calculated,
# pending_cancel, ...) is still in flight — treated as TIMEOUT if we give up
# polling before it resolves, so Temporal's RetryPolicy retries the ACTIVITY
# (which will find the same order via client_order_id, not double-submit it).


def map_alpaca_status(status: str) -> OrderResult:
    if status in _TERMINAL_FILLED:
        return OrderResult.FILLED
    if status in _TERMINAL_PARTIAL:
        return OrderResult.PARTIAL
    if status in _TERMINAL_REJECTED:
        return OrderResult.REJECTED
    return OrderResult.TIMEOUT


def _filled_notional(alpaca_order: Any) -> float:
    qty = getattr(alpaca_order, "filled_qty", None)
    price = getattr(alpaca_order, "filled_avg_price", None)
    if qty is None or price is None:
        return 0.0
    try:
        return float(qty) * float(price)
    except (TypeError, ValueError):
        return 0.0


def _filled_qty(alpaca_order: Any) -> Optional[float]:
    qty = getattr(alpaca_order, "filled_qty", None)
    if qty is None:
        return None
    try:
        return float(qty)
    except (TypeError, ValueError):
        return None


def apply_default_timeout(client: Any, timeout: float = 30.0) -> Any:
    """CRITICAL, confirmed by reading alpaca-py's own source
    (RESTClient._request builds `opts` with headers/params/allow_redirects
    but NEVER a 'timeout' key, then calls self._session.request(method, url,
    **opts)) — every alpaca-py call has NO timeout at all by default.
    requests defaults timeout=None, meaning a connection Alpaca's gateway
    accepts but never answers on hangs FOREVER: no exception, no log line,
    nothing to retry — confirmed live, this is what actually produced a
    request that never returned, distinct from (and worse than) Alpaca's own
    {"code":50410000,"message":"request timed out"} gateway error, which at
    least comes back as something retry_on_rate_limit can act on. Mounting
    this adapter on the client's underlying requests.Session gives every
    call through it a hard ceiling. Applied automatically by every wrapper
    class below (__init__) so no call site has to remember it."""
    session = getattr(client, "_session", None)
    if session is not None:
        from requests.adapters import HTTPAdapter

        class _TimeoutAdapter(HTTPAdapter):
            def send(self, request, **kwargs):
                kwargs.setdefault("timeout", timeout)
                return super().send(request, **kwargs)

        adapter = _TimeoutAdapter()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    return client


def _is_crypto_symbol(symbol: str) -> bool:
    """Alpaca crypto symbols are pair-formatted, e.g. 'BTC/USD'; equities
    are bare tickers, e.g. 'AAPL'. Options would need a separate contract-
    symbol scheme entirely — not handled here, equities+crypto only."""
    return "/" in symbol


# -----------------------------------------------------------------------------
# Broker client — satisfies execution_workflow.BrokerClient.
# The underlying `trading_client` is INJECTED, not constructed inside this
# class, specifically so this whole class is testable against a fake one —
# see FakeAlpacaTradingClient below and the tests in __main__.
# -----------------------------------------------------------------------------
class AlpacaBrokerClient:
    def __init__(self, trading_client: Any, poll_seconds: float = 8.0, poll_interval: float = 1.0):
        self._client = apply_default_timeout(trading_client)
        self._poll_seconds = poll_seconds
        self._poll_interval = poll_interval

    def submit_and_await(self, order: Order) -> Fill:
        alpaca_order = self._find_existing(order.client_order_id) if order.client_order_id else None
        if alpaca_order is None:
            alpaca_order = self._submit(order)

        deadline = time.monotonic() + self._poll_seconds
        while time.monotonic() < deadline:
            status = map_alpaca_status(alpaca_order.status)
            # BUG FIX, confirmed against real Alpaca paper-account order
            # history: "partially_filled" is NOT a resting state for a
            # market order — it MEANS "still actively filling," and the
            # remaining shares typically land within the next second or
            # two. Treating it as terminal (the original behavior) meant
            # polling once, catching an order mid-fill (e.g. a 1303-share
            # order caught at ~571 shares filled), and recording THAT
            # partial notional divided by the FULL intended share count as
            # the entry price — confirmed live: a real launch recorded an
            # entry price ~44% of the eventual real average fill price this
            # way. Only REJECTED/FILLED are accepted as terminal here now;
            # a PARTIAL status keeps polling like an unresolved order,
            # right up to the deadline.
            if status in (OrderResult.FILLED, OrderResult.REJECTED):
                notional = _filled_notional(alpaca_order) if status == OrderResult.FILLED else 0.0
                if status == OrderResult.FILLED and notional <= 0.0:
                    # RACE CONDITION, also confirmed against real Alpaca
                    # paper-account order history: `status` can flip to
                    # "filled" on an order snapshot a beat before
                    # `filled_qty`/`filled_avg_price` land on that SAME
                    # snapshot. Re-fetch once, after a short beat, before
                    # accepting a zero notional on an allegedly-filled order.
                    time.sleep(1.0)
                    try:
                        alpaca_order = retry_on_rate_limit(self._client.get_order_by_id, alpaca_order.id,
                                                           max_attempts=2, base_delay=1.0)
                        notional = _filled_notional(alpaca_order)
                    except Exception:
                        pass
                filled_qty = _filled_qty(alpaca_order) if status == OrderResult.FILLED else None
                return Fill(order, status, notional, filled_qty=filled_qty,
                           reason="" if status != OrderResult.REJECTED
                                 else getattr(alpaca_order, "status", "rejected"))
            time.sleep(self._poll_interval)
            try:
                alpaca_order = retry_on_rate_limit(
                    self._client.get_order_by_id, alpaca_order.id,
                    max_attempts=2, base_delay=1.0)
            except Exception:
                # A transient lookup failure mid-poll shouldn't blow up the
                # whole submit_and_await call — treat this tick as "still
                # unresolved" and keep polling until the deadline.
                continue

        # Deadline reached without a FILLED/REJECTED resolution. If the
        # order is sitting at "partially_filled" right now, that's a real
        # (if incomplete) result worth reporting as PARTIAL with whatever
        # has genuinely filled so far — NOT the same as never having
        # started (TIMEOUT below), and web_ui.py's callers already handle
        # PARTIAL by recording exactly what filled, same as FILLED.
        final_status = map_alpaca_status(alpaca_order.status)
        if final_status == OrderResult.PARTIAL:
            return Fill(order, OrderResult.PARTIAL, _filled_notional(alpaca_order),
                       filled_qty=_filled_qty(alpaca_order),
                       reason="poll window elapsed while still partially filling")

        # Still not resolved after the poll window — TIMEOUT, not REJECTED.
        # Temporal's RetryPolicy will retry the activity; because
        # client_order_id is stable, that retry calls _find_existing() and
        # picks up THIS SAME order rather than submitting a second one.
        return Fill(order, OrderResult.TIMEOUT, 0.0, reason="poll window elapsed, still in flight")

    def cancel(self, order: Order) -> None:
        alpaca_order = self._find_existing(order.client_order_id) if order.client_order_id else None
        if alpaca_order is not None:
            self._client.cancel_order_by_id(alpaca_order.id)

    def get_current_positions(self, total_capital: float) -> Dict[str, float]:
        """symbol -> weight of total_capital, matching what
        execution_workflow.ExecutionWorkflowInput.current_positions expects."""
        if total_capital <= 0:
            return {}
        positions = self._client.get_all_positions()
        return {p.symbol: float(p.market_value) / total_capital for p in positions}

    def _submit(self, order: Order) -> Any:
        request = _make_order_request(order)
        try:
            # Confirmed live: Alpaca's gateway occasionally returns
            # {"code":50410000,"message":"request timed out"} under load —
            # a transient failure on Alpaca's side, not a rejection of the
            # order itself. Retried like a rate limit; client_order_id
            # means a retry that actually landed the first time is caught
            # by _find_existing below rather than double-submitted.
            return retry_on_rate_limit(self._client.submit_order, request,
                                        max_attempts=3, base_delay=1.5)
        except Exception:
            # A duplicate client_order_id submission is Alpaca's own
            # rejection of a resubmit — that IS the idempotency check
            # working; treat it as "go find the existing order" rather
            # than a real failure. Exact exception type/message to match
            # here depends on alpaca-py's current error shape — verify
            # against current docs before relying on this branch.
            existing = self._find_existing(order.client_order_id)
            if existing is not None:
                return existing
            raise

    def _find_existing(self, client_order_id: str) -> Optional[Any]:
        """BUG FIX: GetOrdersRequest has no client_order_id field at all
        (confirmed against the real alpaca-py request model) — the ORIGINAL
        version of this filtered get_orders() by a field that doesn't
        exist, meaning the entire idempotency safety net (never double-
        submitting a retried order) silently never worked. The real,
        dedicated lookup is get_order_by_client_id(), which raises APIError
        (confirmed live: code 40410000 "order not found for ...") rather
        than returning an empty list when there's no match."""
        if not client_order_id:
            return None
        try:
            return self._client.get_order_by_client_id(client_order_id)
        except Exception:
            return None


def _is_transient_alpaca_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "too many requests" in msg or "429" in msg
        # Confirmed live: {"code":50410000,"message":"request timed out"} —
        # Alpaca's own gateway timing out processing the request, not a
        # rejection of the order/query itself.
        or "50410000" in msg or "request timed out" in msg or "timeout" in msg
    )


def retry_on_rate_limit(fn, *args, max_attempts: int = 4, base_delay: float = 1.0, **kwargs):
    """Alpaca's real rate limit ({"message": "too many requests."}) and its
    gateway timeout ({"code":50410000,"message":"request timed out"}) are
    both transient conditions, not reasons to fail an entire ingestion cycle
    or order submission — confirmed live for both. Retries with exponential
    backoff (1.5s, 3s, ...) on exactly those; anything else (bad symbol,
    auth failure, a genuine order rejection, etc.) raises immediately since
    retrying it would just fail identically."""
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_transient_alpaca_error(e) or attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


# -----------------------------------------------------------------------------
# Market data client — satisfies data_ingestion.MarketDataClient.
# Routes to Alpaca's stock or crypto historical-data client based on symbol
# shape, since Alpaca uses separate endpoints/clients for each.
# -----------------------------------------------------------------------------
class AlpacaMarketDataClient:
    def __init__(self, stock_client: Any, crypto_client: Any):
        self._stock_client = apply_default_timeout(stock_client)
        self._crypto_client = apply_default_timeout(crypto_client)

    def get_price_history(self, symbol: str, days: int) -> List[float]:
        from datetime import datetime, timedelta, timezone
        start = datetime.now(timezone.utc) - timedelta(days=days)
        if _ALPACA_AVAILABLE:
            from alpaca.data.timeframe import TimeFrame
            timeframe = TimeFrame.Day
        else:
            timeframe = "1Day"   # stand-in; FakeAlpacaDataClient below ignores this value

        if _is_crypto_symbol(symbol):
            if _ALPACA_AVAILABLE:
                from alpaca.data.requests import CryptoBarsRequest
                request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start)
            else:
                request = SimpleNamespace(symbol_or_symbols=symbol, timeframe=timeframe, start=start)
            bars = retry_on_rate_limit(self._crypto_client.get_crypto_bars, request)
        else:
            if _ALPACA_AVAILABLE:
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.enums import Adjustment
                # BUG FIX, confirmed live against real CRWD data: without
                # adjustment=ALL, alpaca-py returns RAW bars — a stock split
                # partway through the requested window shows up as a fake
                # ~75% single-day "crash" (CRWD: $772.74 -> $193.98, a
                # 3.98x ratio, on its real split date) with NO adjustment
                # applied on either side of it. Every momentum/volatility
                # number computed from raw bars across a split boundary is
                # silently wrong, not just the chart.
                request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe,
                                          start=start, adjustment=Adjustment.ALL)
            else:
                request = SimpleNamespace(symbol_or_symbols=symbol, timeframe=timeframe, start=start)
            bars = retry_on_rate_limit(self._stock_client.get_stock_bars, request)
        # Response shape is a BarSet keyed by symbol; each bar has .close —
        # verify the exact attribute path against current alpaca-py docs.
        return [bar.close for bar in bars[symbol]]


class AlpacaNewsClient:
    """Wraps Alpaca's NewsClient (alpaca.data.historical.news.NewsClient) —
    construct it the same way as the stock/crypto historical data clients
    (NewsClient(api_key, secret_key)) and inject it here.

    VERIFIED against repeated real live calls (alpaca-py 0.44.0): get_news()
    returns a NewsSet, not a directly-iterable list of news objects — the
    original version of this method iterated the NewsSet itself, which
    yields (field_name, value) tuples (pydantic BaseModel's default
    __iter__), not news items; `item.headline` would have raised
    immediately on first real use. The actual articles are at
    `.data["news"]`. Item TYPE was inconsistent across live calls in
    testing — sometimes plain dicts, sometimes News objects with attribute
    access, across otherwise-identical requests — so both are handled
    defensively rather than betting on one."""
    def __init__(self, news_client: Any):
        self._client = apply_default_timeout(news_client)

    def get_recent_headlines(self, symbol_or_theme: str, days: int) -> List[str]:
        from datetime import datetime, timedelta, timezone
        from alpaca.data.requests import NewsRequest
        start = datetime.now(timezone.utc) - timedelta(days=days)
        request = NewsRequest(symbols=symbol_or_theme, start=start)
        news = retry_on_rate_limit(self._client.get_news, request)
        articles = news.data.get("news", [])
        return [a["headline"] if isinstance(a, dict) else a.headline for a in articles]


# -----------------------------------------------------------------------------
# Fakes for testing — shaped like the real SDK's objects, no network needed.
# -----------------------------------------------------------------------------
@dataclass
class _FakeAlpacaOrder:
    id: str
    client_order_id: str
    status: str
    filled_qty: Optional[str] = None
    filled_avg_price: Optional[str] = None


class FakeAlpacaTradingClient:
    """behaviors[client_order_id] = list of statuses returned on successive
    get_order_by_id calls (simulates an order resolving after N polls)."""
    def __init__(self, behaviors: Dict[str, List[str]]):
        self.behaviors = behaviors
        self.poll_counts: Dict[str, int] = {}
        self.submitted: List[str] = []
        self.cancelled: List[str] = []
        self._orders: Dict[str, _FakeAlpacaOrder] = {}

    def submit_order(self, request: Any) -> _FakeAlpacaOrder:
        coid = request.client_order_id
        if coid in self._orders:
            raise RuntimeError(f"duplicate client_order_id: {coid}")   # mimics Alpaca's real rejection
        self.submitted.append(coid)
        status = self.behaviors.get(coid, ["filled"])[0]
        order = _FakeAlpacaOrder(id=f"id-{coid}", client_order_id=coid, status=status,
                                 filled_qty="10" if status in ("filled", "partially_filled") else None,
                                 filled_avg_price="150.00" if status in ("filled", "partially_filled") else None)
        self._orders[coid] = order
        self.poll_counts[coid] = 1
        return order

    def get_order_by_id(self, order_id: str) -> _FakeAlpacaOrder:
        coid = order_id.replace("id-", "")
        seq = self.behaviors.get(coid, ["filled"])
        n = self.poll_counts.get(coid, 0)
        status = seq[min(n, len(seq) - 1)]
        self.poll_counts[coid] = n + 1
        order = self._orders[coid]
        order.status = status
        if status in ("filled", "partially_filled"):
            order.filled_qty, order.filled_avg_price = "10", "150.00"
        return order

    def get_order_by_client_id(self, client_id: str) -> _FakeAlpacaOrder:
        # Matches the real API's contract (confirmed live): raises rather
        # than returning empty/None when there's no match.
        if client_id not in self._orders:
            raise RuntimeError(f"order not found for {client_id}")
        return self._orders[client_id]

    def cancel_order_by_id(self, order_id: str) -> None:
        coid = order_id.replace("id-", "")
        self.cancelled.append(coid)

    def get_all_positions(self):
        @dataclass
        class _Pos:
            symbol: str
            market_value: str
        return [_Pos("NVDA", "5000.00"), _Pos("BTC/USD", "3000.00")]


class FakeAlpacaDataClient:
    """Stands in for StockHistoricalDataClient / CryptoHistoricalDataClient.
    bars_by_symbol[symbol] = list of closing prices, oldest first."""
    def __init__(self, bars_by_symbol: Dict[str, List[float]], method_name: str):
        self.bars_by_symbol = bars_by_symbol
        self._method_name = method_name   # "get_stock_bars" or "get_crypto_bars"
        setattr(self, method_name, self._get_bars)

    def _get_bars(self, request: Any) -> Dict[str, List[Any]]:
        symbol = request.symbol_or_symbols

        @dataclass
        class _Bar:
            close: float
        return {symbol: [_Bar(p) for p in self.bars_by_symbol.get(symbol, [])]}


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: status mapping — the part most worth getting right")
    print("=" * 70)
    assert map_alpaca_status("filled") == OrderResult.FILLED
    assert map_alpaca_status("partially_filled") == OrderResult.PARTIAL
    assert map_alpaca_status("rejected") == OrderResult.REJECTED
    assert map_alpaca_status("canceled") == OrderResult.REJECTED
    assert map_alpaca_status("new") == OrderResult.TIMEOUT
    assert map_alpaca_status("pending_new") == OrderResult.TIMEOUT
    print("✓ filled/partial/rejected/canceled/in-flight all map correctly")

    print("\n" + "=" * 70)
    print("TEST 2: happy path — order fills on the first poll")
    print("=" * 70)
    fake = FakeAlpacaTradingClient({"wf1-NVDA-buy": ["filled"]})
    broker = AlpacaBrokerClient(fake, poll_seconds=5, poll_interval=0.01)
    order = Order("NVDA", Side.BUY, 5000.0, client_order_id="wf1-NVDA-buy")
    fill = broker.submit_and_await(order)
    print(f"  result={fill.result.value} filled_notional={fill.filled_notional}")
    assert fill.result == OrderResult.FILLED and fill.filled_notional == 1500.0
    print("✓ submitted once, resolved immediately")

    print("\n" + "=" * 70)
    print("TEST 3: order resolves after a couple of polls (still in flight, then fills)")
    print("=" * 70)
    fake2 = FakeAlpacaTradingClient({"wf1-CAT-sell": ["new", "new", "filled"]})
    broker2 = AlpacaBrokerClient(fake2, poll_seconds=5, poll_interval=0.01)
    fill2 = broker2.submit_and_await(Order("CAT", Side.SELL, 2000.0, client_order_id="wf1-CAT-sell"))
    print(f"  result={fill2.result.value} after {fake2.poll_counts['wf1-CAT-sell']} polls")
    assert fill2.result == OrderResult.FILLED
    print("✓ polling loop correctly waits out the in-flight period")

    print("\n" + "=" * 70)
    print("TEST 4: THE IMPORTANT ONE — retried activity does NOT double-submit")
    print("=" * 70)
    fake3 = FakeAlpacaTradingClient({"wf1-NVDA-buy": ["filled"]})
    broker3 = AlpacaBrokerClient(fake3, poll_seconds=5, poll_interval=0.01)
    order3 = Order("NVDA", Side.BUY, 5000.0, client_order_id="wf1-NVDA-buy")
    fill_a = broker3.submit_and_await(order3)      # first "attempt"
    fill_b = broker3.submit_and_await(order3)      # simulated Temporal retry of the SAME activity
    print(f"  submissions to the broker: {len(fake3.submitted)} (should be 1, not 2)")
    assert len(fake3.submitted) == 1
    assert fill_a.result == fill_b.result == OrderResult.FILLED
    print("✓ the second call found the existing order via client_order_id instead of")
    print("  placing a real duplicate trade — this is the whole point of the idempotency key")

    print("\n" + "=" * 70)
    print("TEST 5: timeout after the poll window — reported as TIMEOUT, not REJECTED")
    print("=" * 70)
    fake4 = FakeAlpacaTradingClient({"wf1-XOM-buy": ["new", "new", "new", "new", "new"]})
    broker4 = AlpacaBrokerClient(fake4, poll_seconds=0.05, poll_interval=0.02)
    fill4 = broker4.submit_and_await(Order("XOM", Side.BUY, 1000.0, client_order_id="wf1-XOM-buy"))
    print(f"  result={fill4.result.value}: {fill4.reason}")
    assert fill4.result == OrderResult.TIMEOUT
    print("✓ still in flight when the poll window ran out -> TIMEOUT (retryable),")
    print("  never mislabeled as REJECTED (which execute_orders_saga would NOT retry)")

    print("\n" + "=" * 70)
    print("TEST 6: get_current_positions — converts $ market value to weight-of-capital")
    print("=" * 70)
    positions = broker.get_current_positions(total_capital=100_000.0)
    print(f"  {positions}")
    assert positions == {"NVDA": 0.05, "BTC/USD": 0.03}
    print("✓ matches execution_workflow.ExecutionWorkflowInput.current_positions's expected shape")

    print("\n" + "=" * 70)
    print("TEST 7: AlpacaMarketDataClient — routes equities vs. crypto to the")
    print("correct underlying client based on symbol shape")
    print("=" * 70)
    fake_stock = FakeAlpacaDataClient({"NVDA": [100.0, 105.0, 112.0]}, "get_stock_bars")
    fake_crypto = FakeAlpacaDataClient({"BTC/USD": [60000.0, 61000.0, 63000.0]}, "get_crypto_bars")
    market_data = AlpacaMarketDataClient(fake_stock, fake_crypto)

    nvda_prices = market_data.get_price_history("NVDA", days=21)
    btc_prices = market_data.get_price_history("BTC/USD", days=21)
    print(f"  NVDA (equity) -> {nvda_prices}")
    print(f"  BTC/USD (crypto, has '/') -> {btc_prices}")
    assert nvda_prices == [100.0, 105.0, 112.0]
    assert btc_prices == [60000.0, 61000.0, 63000.0]
    print("✓ 'NVDA' routed to the stock client, 'BTC/USD' (contains '/') routed to the")
    print("  crypto client — exactly the symbol-shape convention Alpaca itself uses")

    print()
    print("=" * 70)
    print("Everything above is genuinely tested against a fake shaped like Alpaca's")
    print("real SDK. The literal network calls (submit_order, get_order_by_id, etc.")
    print("against Alpaca's servers) are untested here — no network access, no")
    print("`pip install alpaca-py`. Verify request/response shapes against current")
    print("alpaca-py docs before going live, especially AlpacaNewsClient's import path.")
    print("=" * 70)
