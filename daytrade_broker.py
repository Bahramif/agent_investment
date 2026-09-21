"""
daytrade_broker.py
===================
Alpaca plumbing for the day-trading capability. It EXTENDS the existing
AlpacaBrokerClient (alpaca_clients.py) instead of replacing it, so day
trading inherits everything that class already got right the hard way:
the no-timeout adapter, transient-error retry, the deterministic
client_order_id idempotency check, and the partial-fill / filled-status-race
polling. What this module adds is what a BRACKET-based intraday loop needs.

WHY A SEPARATE ALPACA ACCOUNT. Every Meridian portfolio is a virtual ledger
over ONE shared paper account (see portfolio_store.py). A day trader in that
account would (1) leak intraday holdings into web_ui's _current_positions(),
skewing portfolio rebalance deltas, and (2) collide with Alpaca's wash-trade
protection, which rejects an order that could match an opposite-side order in
the same account — e.g. a portfolio SELL of NVDA against a day-trade BUY.
Alpaca allows up to 3 paper accounts per user, each with its own keys. So the
day-trading capability uses its own keys (secrets `alpaca-daytrade-api-key` /
`alpaca-daytrade-secret-key`), and refuses to start if they equal the main
account's (assert_dedicated_account) — a mis-pasted secret must fail loudly at
9:00am, not quietly mix ledgers. `paper=True` is hardcoded: there is no
live-trading switch in this capability, on purpose.

WHY LEGS ARE CANCELLED BEFORE A FLATTEN SELL. A bracket's stop and target
legs are resting SELL orders that HOLD the position's shares. Selling those
shares at the flatten deadline while the legs are live is rejected as
"insufficient qty available". flatten_position() therefore cancels the legs,
waits for the shares to be released, re-checks that the position still exists
(a leg may have filled in the meantime — that is a normal exit, not an error),
and only then sells. fake_broker tests below enforce the same rule the real
broker does, so the ordering can't silently regress.

HONEST LIMITATION, same shape as alpaca_clients.py: request/response shapes
were checked against the installed alpaca-py (0.44) models, but no call is
made to Alpaca from this sandbox. The full submit/lookup/cancel/flatten
logic is exercised against FakeDayTradeTradingClient, which reproduces the
real behaviours that matter (legs only appear on a NESTED lookup; held shares
can't be sold; cancelling a non-cancelable order raises).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from alpaca_clients import AlpacaBrokerClient, _ALPACA_AVAILABLE, retry_on_rate_limit
from daytrade_engine import EntryIntent, FillEvent
from execution_workflow import Fill, Order, OrderResult, Side

DAYTRADE_KEY_SECRET = "alpaca-daytrade-api-key"
DAYTRADE_SECRET_SECRET = "alpaca-daytrade-secret-key"
MAIN_KEY_SECRET = "alpaca-api-key"
MAIN_SECRET_SECRET = "alpaca-secret-key"

_OPEN_STATES = {"new", "accepted", "pending_new", "held", "partially_filled", "accepted_for_bidding",
                "pending_replace", "calculated"}


def _v(x: Any) -> str:
    """alpaca-py fields are str-enums; the fake uses plain strings. Normalize."""
    return str(getattr(x, "value", x)).lower()


# -----------------------------------------------------------------------------
# Credentials and the dedicated-account guard
# -----------------------------------------------------------------------------
def _secret(name: str, project: str) -> str:
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    return client.access_secret_version(
        name=f"projects/{project}/secrets/{name}/versions/latest").payload.data.decode("utf-8")


def load_credentials(kind: str, project: Optional[str] = None) -> Tuple[str, str]:
    """kind: "daytrade" or "main". Environment first (local runs, tests), then
    Secret Manager — the same two sources web_ui.py's own loader uses."""
    if kind == "daytrade":
        env = ("ALPACA_DAYTRADE_API_KEY", "ALPACA_DAYTRADE_SECRET_KEY")
        names = (DAYTRADE_KEY_SECRET, DAYTRADE_SECRET_SECRET)
    elif kind == "main":
        env = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
        names = (MAIN_KEY_SECRET, MAIN_SECRET_SECRET)
    else:
        raise ValueError(f"unknown credential kind {kind!r}")
    if os.environ.get(env[0]) and os.environ.get(env[1]):
        return os.environ[env[0]], os.environ[env[1]]
    project = project or os.environ.get("MERIDIAN_PROJECT", "meridian-508414")
    return _secret(names[0], project), _secret(names[1], project)


def assert_dedicated_account(daytrade_key: str, main_key: str) -> None:
    if not daytrade_key:
        raise RuntimeError("day-trading Alpaca API key is empty")
    if daytrade_key == main_key:
        raise RuntimeError(
            "the day-trading API key is IDENTICAL to the main portfolio account's key. Day trading must use its "
            "own Alpaca paper account (a second paper account, with its own keys) so intraday orders can't mix "
            "with portfolio positions or trigger wash-trade rejections. Create one at app.alpaca.markets and "
            f"store its keys as secrets '{DAYTRADE_KEY_SECRET}' / '{DAYTRADE_SECRET_SECRET}'.")


def make_daytrade_trading_client(project: Optional[str] = None) -> Any:
    from alpaca.trading.client import TradingClient
    dt_key, dt_secret = load_credentials("daytrade", project)
    if not os.environ.get("DAYTRADE_SKIP_ACCOUNT_CHECK"):
        try:
            main_key, _ = load_credentials("main", project)
        except Exception as e:
            raise RuntimeError(f"cannot read the main account's key to verify the day-trading account is a "
                               f"different one ({e}). Fix access, or set DAYTRADE_SKIP_ACCOUNT_CHECK=1 to "
                               f"override deliberately.") from e
        assert_dedicated_account(dt_key, main_key)
    return TradingClient(dt_key, dt_secret, paper=True)      # PAPER ONLY, hardcoded


# -----------------------------------------------------------------------------
# Snapshots — plain data extracted from SDK objects
# -----------------------------------------------------------------------------
@dataclass
class LegSnapshot:
    id: str
    kind: str                       # "stop" | "target"
    status: str
    qty: float
    filled_qty: float
    filled_avg_price: Optional[float]
    filled_at: Optional[datetime]
    price: Optional[float] = None   # the leg's own level: target's limit price / stop's stop price

    @property
    def is_open(self) -> bool:
        return self.status in _OPEN_STATES


@dataclass
class BracketSnapshot:
    id: str
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]
    legs: List[LegSnapshot] = field(default_factory=list)


@dataclass
class PositionSnapshot:
    symbol: str
    qty: float
    qty_available: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float


@dataclass
class FlattenResult:
    status: str                     # "sold" | "already_flat" | "pending" | "rejected"
    fill: Optional[FillEvent] = None
    detail: str = ""


def _f(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _snapshot(o: Any) -> BracketSnapshot:
    legs = []
    for leg in (getattr(o, "legs", None) or []):
        kind = "target" if _v(getattr(leg, "order_type", getattr(leg, "type", ""))) == "limit" else "stop"
        level = _f(getattr(leg, "limit_price", None)) if kind == "target" else _f(getattr(leg, "stop_price", None))
        legs.append(LegSnapshot(str(leg.id), kind, _v(leg.status), _f(leg.qty) or 0.0,
                                _f(leg.filled_qty) or 0.0, _f(leg.filled_avg_price),
                                getattr(leg, "filled_at", None), level))
    return BracketSnapshot(str(o.id), _v(o.status), _f(o.filled_qty) or 0.0, _f(o.filled_avg_price), legs)


# -----------------------------------------------------------------------------
# The broker
# -----------------------------------------------------------------------------
class AlpacaDayTradeBroker(AlpacaBrokerClient):
    def __init__(self, trading_client: Any, dedicated: bool = False, poll_seconds: float = 8.0,
                 poll_interval: float = 1.0, release_wait: float = 5.0,
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        super().__init__(trading_client, poll_seconds=poll_seconds, poll_interval=poll_interval)
        self._dedicated = dedicated
        self._release_wait = release_wait
        self._sleep = sleep
        self._now = now

    # ---- entries
    def place_entry(self, intent: EntryIntent) -> Fill:
        """Market BUY with stop + target attached at the broker. Idempotent on
        intent.client_order_id (one per symbol per day): calling this again
        after a crash finds the existing order instead of buying twice."""
        order = Order(intent.symbol, Side.BUY, intent.qty * intent.entry_ref, qty=intent.qty,
                      client_order_id=intent.client_order_id, stop_price=intent.stop,
                      take_profit_price=intent.target)
        return self.submit_and_await(order)

    def lookup_entry(self, client_order_id: str) -> Optional[BracketSnapshot]:
        """Parent order plus its legs. `get_order_by_client_id` does not
        nest legs, so the legs come from a second, nested lookup by id."""
        parent = self._find_existing(client_order_id)
        if parent is None:
            return None
        try:
            if _ALPACA_AVAILABLE:
                from alpaca.trading.requests import GetOrderByIdRequest
                flt = GetOrderByIdRequest(nested=True)
            else:
                flt = SimpleNamespace(nested=True)
            full = retry_on_rate_limit(self._client.get_order_by_id, parent.id, flt, max_attempts=3, base_delay=1.0)
        except Exception:
            full = parent
        return _snapshot(full)

    def cancel_entry_remainder(self, client_order_id: str) -> None:
        """A partially-filled parent keeps trying to buy the rest; cancel that
        remainder so the bracket protects exactly what was actually bought."""
        snap = self.lookup_entry(client_order_id)
        if snap and snap.status in _OPEN_STATES:
            try:
                self._client.cancel_order_by_id(snap.id)
            except Exception:
                pass

    def exit_fill_from_legs(self, snap: Optional[BracketSnapshot], symbol: str) -> Optional[FillEvent]:
        """A leg that has FILLED means the bracket closed the position on its
        own — the normal stop/target exit."""
        if snap is None:
            return None
        for leg in snap.legs:
            if leg.status == "filled" and leg.filled_avg_price:
                return FillEvent(symbol, "sell", leg.filled_qty or leg.qty, leg.filled_avg_price,
                                 leg.filled_at or self._now(), reason=leg.kind)
        return None

    # ---- exits
    def flatten_position(self, symbol: str, qty: int, entry_client_order_id: str, day_tag: str,
                         attempt: int = 0, reason: str = "eod_flatten") -> FlattenResult:
        # 1) free the shares: cancel any live leg (cancelling one cancels its OCO sibling; do both anyway)
        snap = self.lookup_entry(entry_client_order_id)
        for leg in (snap.legs if snap else []):
            if leg.is_open:
                try:
                    self._client.cancel_order_by_id(leg.id)
                except Exception:
                    pass            # already terminal — the re-check below settles what really happened
        # 2) wait for the shares to be released; re-check the position
        pos = self._wait_available(symbol)
        if pos is None:
            snap = self.lookup_entry(entry_client_order_id)
            return FlattenResult("already_flat", self.exit_fill_from_legs(snap, symbol),
                                 "position already closed (a bracket leg filled first)")
        sell_qty = int(min(qty, pos.qty_available if pos.qty_available else pos.qty))
        if sell_qty < 1:
            return FlattenResult("pending", None, f"{pos.qty} held but 0 available — legs still releasing")
        # 3) sell exactly what we own
        coid = f"dt-{day_tag}-{symbol}-flat{attempt}"[:48]
        fill = self.submit_and_await(Order(symbol, Side.SELL, sell_qty * max(pos.current_price, 0.0),
                                           qty=sell_qty, client_order_id=coid))
        if fill.result in (OrderResult.FILLED, OrderResult.PARTIAL) and fill.filled_qty:
            price = fill.filled_notional / fill.filled_qty
            return FlattenResult("sold", FillEvent(symbol, "sell", fill.filled_qty, price, self._now(), reason=reason),
                                 "" if fill.result == OrderResult.FILLED else "partial fill — will retry the rest")
        if fill.result == OrderResult.REJECTED:
            return FlattenResult("rejected", None, fill.reason)
        return FlattenResult("pending", None, fill.reason or "still in flight")

    def _wait_available(self, symbol: str) -> Optional[PositionSnapshot]:
        deadline = time.monotonic() + self._release_wait
        while True:
            pos = self.position(symbol)
            if pos is None or pos.qty_available >= pos.qty or time.monotonic() >= deadline:
                return pos
            self._sleep(0.25)

    # ---- account / positions
    def position(self, symbol: str) -> Optional[PositionSnapshot]:
        try:
            p = self._client.get_open_position(symbol)
        except Exception:
            return None                      # the API raises when there is no position
        return self._pos(p)

    def positions(self) -> Dict[str, PositionSnapshot]:
        return {p.symbol: self._pos(p) for p in self._client.get_all_positions()}

    @staticmethod
    def _pos(p: Any) -> PositionSnapshot:
        qty = _f(p.qty) or 0.0
        avail = _f(getattr(p, "qty_available", None))
        return PositionSnapshot(p.symbol, qty, qty if avail is None else avail, _f(p.avg_entry_price) or 0.0,
                                _f(getattr(p, "current_price", None)) or 0.0, _f(getattr(p, "unrealized_pl", None)) or 0.0)

    def account_snapshot(self) -> Dict[str, Any]:
        a = self._client.get_account()
        return {"equity": _f(a.equity) or 0.0, "cash": _f(a.cash) or 0.0,
                "buying_power": _f(a.buying_power) or 0.0, "status": _v(a.status),
                "trading_blocked": bool(getattr(a, "trading_blocked", False))}

    def emergency_flatten(self) -> None:
        """Cancel everything and close everything. Safe ONLY because this
        account is dedicated to day trading; on a shared account this would
        liquidate portfolio holdings, so it refuses."""
        if not self._dedicated:
            raise RuntimeError("emergency_flatten is only allowed on a verified dedicated day-trading account")
        self._client.cancel_orders()
        self._client.close_all_positions(cancel_orders=True)


# -----------------------------------------------------------------------------
# Fake trading client — reproduces the real behaviours that matter.
# -----------------------------------------------------------------------------
@dataclass
class _FakeOrder:
    id: str
    client_order_id: str
    symbol: str
    side: str
    qty: str
    status: str
    order_type: str = "market"
    filled_qty: Optional[str] = None
    filled_avg_price: Optional[str] = None
    filled_at: Optional[datetime] = None
    limit_price: Optional[str] = None
    stop_price: Optional[str] = None
    order_class: str = "bracket"
    legs: Optional[list] = None
    parent_id: Optional[str] = None


class FakeDayTradeTradingClient:
    def __init__(self, price_fn: Callable[[str], float], equity: float = 25_000.0,
                 reject_symbols: Tuple[str, ...] = ()):
        self.price_fn, self.equity, self.reject_symbols = price_fn, equity, set(reject_symbols)
        self.orders: Dict[str, _FakeOrder] = {}
        self.by_coid: Dict[str, _FakeOrder] = {}
        self.pos: Dict[str, Dict[str, float]] = {}
        self.submitted: List[str] = []
        self.emergency_called = False
        self._n = 0

    # -- helpers
    def _new_id(self) -> str:
        self._n += 1
        return f"ord-{self._n}"

    def _held(self, sym: str) -> float:
        """Shares reserved by resting bracket legs. The stop and target are
        OCO — only one can ever execute — so each bracket group holds its
        shares ONCE, not once per leg."""
        per_group: Dict[str, float] = {}
        for o in self.orders.values():
            if o.symbol == sym and o.parent_id and o.status in _OPEN_STATES:
                per_group[o.parent_id] = max(per_group.get(o.parent_id, 0.0), float(o.qty))
        return sum(per_group.values())

    # -- API surface used by AlpacaDayTradeBroker / AlpacaBrokerClient
    def submit_order(self, req: Any) -> _FakeOrder:
        coid = req.client_order_id
        if coid in self.by_coid:
            raise RuntimeError(f"client_order_id must be unique: {coid}")   # Alpaca's real rejection
        sym, side, qty = req.symbol, _v(req.side), int(req.qty)
        self.submitted.append(coid)
        o = _FakeOrder(self._new_id(), coid, sym, side, str(qty), "new")
        self.orders[o.id] = self.by_coid[coid] = o
        if sym in self.reject_symbols:
            o.status = "rejected"
            return o
        px = self.price_fn(sym)
        if side == "sell":
            p = self.pos.get(sym)
            available = (p["qty"] if p else 0.0) - self._held(sym)
            if qty > available:
                del self.orders[o.id], self.by_coid[coid]
                raise RuntimeError(f"insufficient qty available for order (requested: {qty}, available: {available})")
            p["qty"] -= qty
            if p["qty"] <= 0:
                del self.pos[sym]
            o.status, o.filled_qty, o.filled_avg_price, o.filled_at = "filled", str(qty), f"{px:.4f}", datetime.now(timezone.utc)
            o.order_class = "simple"
            return o
        p = self.pos.setdefault(sym, {"qty": 0.0, "avg": px})
        p["avg"] = (p["avg"] * p["qty"] + px * qty) / (p["qty"] + qty)
        p["qty"] += qty
        o.status, o.filled_qty, o.filled_avg_price, o.filled_at = "filled", str(qty), f"{px:.4f}", datetime.now(timezone.utc)
        if getattr(req, "order_class", None) is not None and _v(req.order_class) == "bracket":
            tp = _FakeOrder(self._new_id(), f"{coid}-tp", sym, "sell", str(qty), "held", "limit",
                            limit_price=str(req.take_profit.limit_price), parent_id=o.id)
            sl = _FakeOrder(self._new_id(), f"{coid}-sl", sym, "sell", str(qty), "held", "stop",
                            stop_price=str(req.stop_loss.stop_price), parent_id=o.id)
            for leg in (tp, sl):
                self.orders[leg.id] = leg
            o.legs = [tp, sl]
        else:
            o.order_class = "simple"
        return o

    def _view(self, o: _FakeOrder, nested: bool) -> _FakeOrder:
        return o if nested or not o.legs else _FakeOrder(**{**o.__dict__, "legs": None})

    def get_order_by_client_id(self, client_id: str) -> _FakeOrder:
        if client_id not in self.by_coid:
            raise RuntimeError(f"order not found for {client_id}")
        return self._view(self.by_coid[client_id], nested=False)     # legs NOT included, like the real endpoint

    def get_order_by_id(self, order_id: str, filter: Any = None) -> _FakeOrder:
        return self._view(self.orders[order_id], nested=bool(getattr(filter, "nested", False)))

    def cancel_order_by_id(self, order_id: str) -> None:
        o = self.orders[order_id]
        if o.status not in _OPEN_STATES:
            raise RuntimeError(f"order {order_id} is not cancelable (status {o.status})")
        o.status = "canceled"
        for sib in self.orders.values():
            if sib.parent_id and sib.parent_id == o.parent_id and sib.status in _OPEN_STATES:
                sib.status = "canceled"          # OCO: the sibling goes with it

    def get_open_position(self, symbol: str) -> SimpleNamespace:
        if symbol not in self.pos:
            raise RuntimeError("position does not exist")
        return self._pos_obj(symbol)

    def _pos_obj(self, symbol: str) -> SimpleNamespace:
        p = self.pos[symbol]
        px = self.price_fn(symbol)
        return SimpleNamespace(symbol=symbol, qty=str(p["qty"]), qty_available=str(p["qty"] - self._held(symbol)),
                               avg_entry_price=str(p["avg"]), current_price=str(px),
                               unrealized_pl=str((px - p["avg"]) * p["qty"]))

    def get_all_positions(self) -> List[SimpleNamespace]:
        return [self._pos_obj(s) for s in self.pos]

    def get_account(self) -> SimpleNamespace:
        return SimpleNamespace(equity=str(self.equity), cash=str(self.equity), buying_power=str(self.equity * 4),
                               status="ACTIVE", trading_blocked=False)

    def cancel_orders(self) -> list:
        self.emergency_called = True
        for o in self.orders.values():
            if o.status in _OPEN_STATES:
                o.status = "canceled"
        return []

    def close_all_positions(self, cancel_orders: Optional[bool] = None) -> list:
        self.pos.clear()
        return []

    # -- test driver: markets move, legs fill
    def tick(self, symbol: str, high: float, low: float, ts: datetime) -> None:
        for leg in [o for o in self.orders.values() if o.symbol == symbol and o.parent_id and o.status == "held"]:
            hit = (leg.order_type == "limit" and high >= float(leg.limit_price)) or \
                  (leg.order_type == "stop" and low <= float(leg.stop_price))
            if not hit or leg.status != "held":
                continue
            px = float(leg.limit_price if leg.order_type == "limit" else leg.stop_price)
            leg.status, leg.filled_qty, leg.filled_avg_price, leg.filled_at = "filled", leg.qty, f"{px:.4f}", ts
            for sib in self.orders.values():
                if sib.parent_id == leg.parent_id and sib is not leg and sib.status in _OPEN_STATES:
                    sib.status = "canceled"
            p = self.pos.get(symbol)
            if p:
                p["qty"] -= float(leg.qty)
                if p["qty"] <= 0:
                    del self.pos[symbol]


if __name__ == "__main__":
    from datetime import date
    prices = {"AAA": 100.6, "BBB": 50.0}
    ts0 = datetime(2026, 3, 4, 14, 48, tzinfo=timezone.utc)

    def make(**kw):
        fake = FakeDayTradeTradingClient(lambda s: prices[s], **kw)
        return fake, AlpacaDayTradeBroker(fake, dedicated=True, poll_seconds=0.5, poll_interval=0.01,
                                          release_wait=0.2, sleep=lambda s: None)

    def intent(sym="AAA", qty=49):
        return EntryIntent(sym, qty, 100.6, 100.0, 101.8, "orb_long", "t", f"dt-20260304-{sym}-buy", ts0)

    print("TEST 1: bracket entry fills, legs carry the intended stop and target")
    fake, br = make()
    fill = br.place_entry(intent())
    assert fill.result == OrderResult.FILLED and fill.filled_qty == 49
    snap = br.lookup_entry("dt-20260304-AAA-buy")
    kinds = {l.kind: l for l in snap.legs}
    assert set(kinds) == {"stop", "target"} and all(l.is_open for l in snap.legs)
    assert kinds["target"].price == 101.8 and kinds["stop"].price == 100.0, "levels must be readable for crash recovery"
    assert float(fake.orders[kinds["target"].id].limit_price) == 101.8
    assert float(fake.orders[kinds["stop"].id].stop_price) == 100.0
    print(f"✓ 49 sh filled; stop leg @100.0 and target leg @101.8 live at the broker")

    print("TEST 2: retrying the same entry (crash + restart) never buys twice")
    again = br.place_entry(intent())
    assert len(fake.submitted) == 1 and again.result == OrderResult.FILLED
    assert fake.pos["AAA"]["qty"] == 49
    print("✓ 1 order at the broker, position still 49")

    print("TEST 3: legs are invisible on the client-id lookup but found via the nested lookup")
    raw = fake.get_order_by_client_id("dt-20260304-AAA-buy")
    assert raw.legs is None and len(br.lookup_entry("dt-20260304-AAA-buy").legs) == 2
    print("✓ lookup_entry does the second, nested fetch")

    print("TEST 4: a leg filling is detected as a stop/target EXIT with the real fill price")
    fake.tick("AAA", high=101.9, low=100.5, ts=ts0)
    ex = br.exit_fill_from_legs(br.lookup_entry("dt-20260304-AAA-buy"), "AAA")
    assert ex and ex.reason == "target" and ex.price == 101.8 and ex.qty == 49 and ex.side == "sell"
    assert "AAA" not in fake.pos
    print(f"✓ target exit {ex.qty} @ {ex.price}")

    print("TEST 5: flatten with live legs — cancels legs FIRST, then sells (selling held shares is rejected)")
    fake, br = make()
    br.place_entry(intent())
    try:
        fake.submit_order(SimpleNamespace(client_order_id="naive", symbol="AAA", side="sell", qty=49))
        raise AssertionError("fake must reject selling shares held by resting legs")
    except RuntimeError as e:
        assert "insufficient qty available" in str(e)
    res = br.flatten_position("AAA", 49, "dt-20260304-AAA-buy", "20260304")
    assert res.status == "sold" and res.fill.qty == 49 and res.fill.price == 100.6 and res.fill.reason == "eod_flatten"
    assert "AAA" not in fake.pos
    assert all(o.status in ("canceled", "filled") for o in fake.orders.values() if o.parent_id), "legs must be cancelled"
    print("✓ legs cancelled, position sold once at 100.6")

    print("TEST 6: flatten when the stop already filled -> 'already_flat' with the stop's fill, no second sell")
    fake, br = make()
    br.place_entry(intent())
    fake.tick("AAA", high=100.4, low=99.5, ts=ts0)              # stop @100.0 hit
    n_before = len(fake.submitted)
    res = br.flatten_position("AAA", 49, "dt-20260304-AAA-buy", "20260304")
    assert res.status == "already_flat" and res.fill.reason == "stop" and res.fill.price == 100.0
    assert len(fake.submitted) == n_before, "must NOT sell a position that no longer exists"
    print("✓ exit attributed to the stop leg; nothing sold twice")

    print("TEST 7: a rejected entry surfaces as REJECTED, with no position")
    fake, br = make(reject_symbols=("BBB",))
    f = br.place_entry(intent("BBB", 10))
    assert f.result == OrderResult.REJECTED and "BBB" not in fake.pos
    print("✓ REJECTED")

    print("TEST 8: a bracket for a fractional qty is refused before it reaches the broker")
    try:
        br.place_entry(EntryIntent("AAA", 49.5, 100.6, 100.0, 101.8, "x", "t", "c-frac", ts0))
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "whole-share" in str(e)
    print("✓ ValueError: whole-share qty required")

    print("TEST 9: positions and account snapshots")
    fake, br = make()
    br.place_entry(intent())
    ps = br.positions()
    assert ps["AAA"].qty == 49 and ps["AAA"].qty_available == 0, "legs hold the shares"
    acct = br.account_snapshot()
    assert acct["equity"] == 25_000.0 and acct["status"] == "active" and not acct["trading_blocked"]
    print("✓ 49 held, 0 available (legs hold them); account active")

    print("TEST 10: emergency_flatten works only on a verified dedicated account")
    fake, br = make()
    br.place_entry(intent())
    br.emergency_flatten()
    assert fake.emergency_called and not fake.pos
    shared = AlpacaDayTradeBroker(FakeDayTradeTradingClient(lambda s: 1.0), dedicated=False)
    try:
        shared.emergency_flatten()
        raise AssertionError("must refuse on a non-dedicated account")
    except RuntimeError as e:
        assert "dedicated" in str(e)
    print("✓ refuses unless dedicated=True (would liquidate portfolio holdings otherwise)")

    print("TEST 11: the dedicated-account guard")
    assert_dedicated_account("PKDAYTRADE", "PKMAIN")
    for a, b in (("PKSAME", "PKSAME"), ("", "PKMAIN")):
        try:
            assert_dedicated_account(a, b)
            raise AssertionError("should have raised")
        except RuntimeError:
            pass
    saved = {k: os.environ.get(k) for k in ("ALPACA_DAYTRADE_API_KEY", "ALPACA_DAYTRADE_SECRET_KEY",
                                            "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "DAYTRADE_SKIP_ACCOUNT_CHECK")}
    try:
        os.environ.update(ALPACA_DAYTRADE_API_KEY="PKX", ALPACA_DAYTRADE_SECRET_KEY="s1",
                          ALPACA_API_KEY="PKX", ALPACA_SECRET_KEY="s2")
        os.environ.pop("DAYTRADE_SKIP_ACCOUNT_CHECK", None)
        try:
            make_daytrade_trading_client()
            raise AssertionError("same keys must be refused")
        except RuntimeError as e:
            assert "IDENTICAL" in str(e)
        os.environ["ALPACA_API_KEY"] = "PKMAIN"
        client = make_daytrade_trading_client()
        assert client is not None
        print("✓ same keys -> refused at startup; distinct keys -> a paper TradingClient")
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    print("\nAll daytrade_broker tests passed.")
