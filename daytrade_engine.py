"""
daytrade_engine.py
===================
The decision core of the day-trading capability. ONE class, driven by TWO
different callers:

    daytrade_backtest.py  -> feeds historical bars, fills via a simulator
    daytrade_runner.py    -> feeds live polled bars, fills via Alpaca

Because both run this exact object, a backtest exercises the same signal,
risk and flatten logic that trades live — the only thing that differs is
where fills come from. That parity is the point of the design.

WHAT IT DOES NOT DO: it never touches a broker, a clock, or the network.
It consumes bars and fills; it emits INTENTS (enter / exit / cancel). The
caller executes them and reports back with apply_fill()/apply_reject().
Position state changes ONLY through reported fills — never on the strength
of an intent — so a rejected or partial order can't leave the engine
believing in a position that doesn't exist.

Long-only. Every entry is a bracket (stop + target live at the broker), so a
crash of this process cannot leave a position unprotected. The engine adds
the two exits a broker can't: the end-of-day flatten and the daily-loss halt.

Playbook (optional) is the LLM's only input, and it can only RESTRICT: fewer
strategies, fewer names, risk-off, or a smaller risk multiplier. It is
clamped here, so a malformed or overreaching playbook can't widen anything.

LEARNING HOOKS. The playbook also carries the ACTIVE learned lessons
(daytrade_lessons.py): deterministic, restrict-only rules checked at signal
time (skip, or shrink size — never below learning.min_combined_risk_mult).
And every first signal per symbol is journaled (daytrade_journal.py) with its
entry snapshot and a stand-alone virtual outcome — including the ones the
playbook, a lesson or a capacity veto stopped — so the learning loop can see
what each restriction actually cost or saved. Journaling never changes a
decision: the signals it evaluates for filtered names are computed and
recorded, not traded.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from daytrade_config import DayTradingConfig
from daytrade_journal import SignalTracker, entry_features
from daytrade_lessons import LessonSet
from daytrade_risk import RiskState, check_entry, daily_loss_breached, daily_pnl
from daytrade_strategies import STRATEGIES, SymbolContext, first_signal, update_vwaps
from intraday_data import MARKET_CLOSE, Bar, effective_session, in_regular_session, to_et


@dataclass(frozen=True)
class EntryIntent:
    symbol: str
    qty: int
    entry_ref: float
    stop: float
    target: float
    strategy: str
    reason: str
    client_order_id: str
    decision_ts: datetime


@dataclass(frozen=True)
class ExitIntent:
    symbol: str
    qty: int
    reason: str            # "eod_flatten" | "halt_flatten"


@dataclass(frozen=True)
class CancelIntent:
    symbol: str            # an entry still in flight when flatten/halt hits
    reason: str


Intent = Union[EntryIntent, ExitIntent, CancelIntent]


@dataclass
class FillEvent:
    symbol: str
    side: str              # "buy" | "sell"
    qty: float
    price: float
    ts: datetime
    reason: str = ""       # sells: "stop" | "target" | "eod_flatten" | "halt_flatten" | ...
    fees: float = 0.0


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    stop: float
    target: float
    strategy: str
    entry_ts: datetime
    client_order_id: str
    initial_qty: float = 0.0
    entry_ref: float = 0.0                 # the signal's reference price (fill - ref = entry slippage)
    snapshot: Dict[str, Any] = field(default_factory=dict)
    hi: float = 0.0                        # price extremes while held -> MAE / MFE
    lo: float = 0.0

    def __post_init__(self):
        if not self.initial_qty:
            self.initial_qty = self.qty
        if not self.hi:
            self.hi = self.lo = self.entry_price


@dataclass
class Trade:
    symbol: str
    strategy: str
    qty: float
    entry_price: float
    exit_price: float
    entry_ts: datetime
    exit_ts: datetime
    pnl: float             # NET of fees
    fees: float
    reason: str
    r_multiple: float
    snapshot: Dict[str, Any] = field(default_factory=dict)   # entry features (daytrade_journal.FEATURES)
    mae_r: float = 0.0                     # worst price while held, in R (<= 0)
    mfe_r: float = 0.0                     # best price while held, in R (>= 0)
    entry_ref: float = 0.0


@dataclass
class Playbook:
    """What the pre-market LLM step may set. Every field can only tighten."""
    risk_on: bool = True
    enabled_strategies: Optional[List[str]] = None    # subset of cfg.strategies, or None = all
    watchlist: Optional[List[str]] = None             # symbols to trade, or None = all fed
    risk_multiplier: float = 1.0                      # clamped to [0, 1]
    rationale: str = ""
    lessons: Optional[List[Dict[str, Any]]] = None    # ACTIVE learned lessons (restrict-only), as stored dicts


# Capacity vetoes whose signal is still a real setup worth a counterfactual.
# Window / halt / malformed-stop vetoes are not: those setups could never trade.
_GHOSTABLE_VETOES = ("at max_positions", "at max_trades_per_day", "position size rounds to 0")


class DayTradingEngine:
    EXIT_RETRY = timedelta(seconds=60)

    def __init__(self, cfg: DayTradingConfig, day: date, prev_close: Optional[Dict[str, float]] = None,
                 playbook: Optional[Playbook] = None, close_time: time = MARKET_CLOSE,
                 max_signal_age: timedelta = timedelta(seconds=150), market: Optional[Dict[str, float]] = None):
        self.cfg, self.day, self.close_time = cfg, day, close_time
        self.prev_close = prev_close or {}
        self.last_entry, self.flatten = effective_session(cfg.session, close_time)
        self.sleeve = cfg.sleeve_capital
        self.max_signal_age = max_signal_age
        self.ctx: Dict[str, SymbolContext] = {}
        self._cum: Dict[str, List[float]] = {}
        self.positions: Dict[str, Position] = {}
        self.pending: Dict[str, EntryIntent] = {}
        self._exit_sent: Dict[str, datetime] = {}
        self.state = RiskState()
        self.trades: List[Trade] = []
        self.decisions: List[Dict[str, Any]] = []
        self.market = dict(market or {})                   # daily benchmark read for entry snapshots
        self.journal = SignalTracker(day, cfg.costs.slippage_bps, self.flatten, cfg.learning.post_exit_minutes)
        self._ghosted: Set[str] = set()
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self.set_playbook(playbook)

    # ------------------------------------------------------------------ playbook
    def set_playbook(self, playbook: Optional[Playbook]) -> None:
        pb = playbook or Playbook()
        allowed = [s for s in self.cfg.strategies if pb.enabled_strategies is None or s in pb.enabled_strategies]
        self.enabled = [s for s in allowed if s in STRATEGIES]
        self.watchlist: Optional[Set[str]] = None if pb.watchlist is None else {s.upper() for s in pb.watchlist}
        self.risk_on = bool(pb.risk_on)
        self.risk_multiplier = min(1.0, max(0.0, float(pb.risk_multiplier)))
        self.all_strategies = [s for s in self.cfg.strategies if s in STRATEGIES]
        self.lessons = LessonSet.from_dicts(pb.lessons, self.cfg.learning.min_combined_risk_mult)
        self.playbook = pb

    # ------------------------------------------------------------------ helpers
    def _log(self, ts: datetime, kind: str, symbol: str = "", **detail: Any) -> None:
        self.decisions.append({"ts": ts.isoformat(), "kind": kind, "symbol": symbol, **detail})
        if len(self.decisions) > 500:
            del self.decisions[:100]

    def _mark(self) -> None:
        unreal = 0.0
        for sym, pos in self.positions.items():
            c = self.ctx.get(sym)
            if c and c.bars:
                unreal += (c.bars[-1].close - pos.entry_price) * pos.qty
        self.state.unrealized_pnl = unreal

    def _snapshot(self, sig: Any) -> Dict[str, Any]:
        return entry_features(self.ctx[sig.symbol], sig, self.ctx, self.market, self.risk_multiplier)

    def _ghost(self, sig: Any, ts: datetime, status: str, snap: Optional[Dict[str, Any]] = None,
               reasons: Sequence[str] = (), lesson_ids: Sequence[str] = ()) -> None:
        """Journal a signal the engine did NOT take — once per symbol per day."""
        if sig.symbol in self._ghosted:
            return
        self._ghosted.add(sig.symbol)
        self.journal.track(sig, snap if snap is not None else self._snapshot(sig), status, ts, reasons, lesson_ids)

    def journal_records(self) -> List[Dict[str, Any]]:
        """Every journaled signal with its virtual outcome, real outcome merged
        into the taken ones. Call after the day's last fill."""
        self.journal.finalize()
        return self.journal.records(self.trades)

    @staticmethod
    def entry_order_id(day: date, symbol: str) -> str:
        """One entry per symbol per day -> the ID needs no counter, so a
        restart that lost its counters still produces the SAME ID and the
        broker's idempotency check finds the existing order (see
        alpaca_clients.AlpacaBrokerClient._find_existing)."""
        return f"dt-{day:%Y%m%d}-{symbol}-buy"[:48]

    # ------------------------------------------------------------------ main step
    def on_bars(self, bars: List[Bar], now: datetime) -> List[Intent]:
        intents: List[Intent] = []
        entries_allowed = True

        # 1) mark-to-market and halts run on the CLOCK, not on bar arrival
        ordered = sorted(bars, key=lambda b: (b.ts, b.symbol))
        fresh: List[Bar] = []
        for b in ordered:
            if not in_regular_session(b.ts, self.close_time):
                continue
            sym = b.symbol
            ctx = self.ctx.get(sym)
            if ctx is None:
                ctx = self.ctx[sym] = SymbolContext(sym, self.day, self.prev_close.get(sym), [])
                self._cum[sym] = [0.0, 0.0]
            if ctx.bars and b.ts <= ctx.bars[-1].ts:
                continue                                   # duplicate / out-of-order
            ctx.bars.append(b)
            update_vwaps(ctx.vwaps, self._cum[sym], b)
            fresh.append(b)
            self.journal.on_bar(b)
            pos = self.positions.get(sym)
            if pos is not None and b.end > pos.entry_ts:
                pos.hi, pos.lo = max(pos.hi, b.high), min(pos.lo, b.low)
        self._mark()

        if not self.state.halted and daily_loss_breached(self.state, self.sleeve, self.cfg.limits):
            self.state.halted = True
            self.state.halt_reason = (f"daily loss {daily_pnl(self.state):+,.0f} reached "
                                      f"-{self.cfg.limits.daily_loss_halt:.0%} of sleeve")
            self._log(now, "halt", reason=self.state.halt_reason)

        # 2) exits: end-of-day flatten and halt flatten
        flatten_now = to_et(now).time() >= self.flatten
        halt_flatten = self.state.halted and self.cfg.limits.flatten_on_halt
        if flatten_now or halt_flatten:
            entries_allowed = False
            why = "eod_flatten" if flatten_now else "halt_flatten"
            for sym in sorted(self.pending):
                intents.append(CancelIntent(sym, why))
            for sym, pos in sorted(self.positions.items()):
                last = self._exit_sent.get(sym)
                if last is None or now - last >= self.EXIT_RETRY:
                    self._exit_sent[sym] = now
                    intents.append(ExitIntent(sym, int(pos.qty), why))
                    self._log(now, "exit", sym, qty=pos.qty, reason=why)

        # 3) entries — evaluate every NEW bar in time order. Signals for names the
        #    playbook filtered out are still computed, but only to be JOURNALED.
        if entries_allowed and not self.state.halted:
            base_limits = self.cfg.limits
            if self.risk_multiplier < 1.0:
                base_limits = dataclasses.replace(base_limits,
                                                  risk_per_trade=base_limits.risk_per_trade * self.risk_multiplier)
            fewer = len(self.enabled) < len(self.all_strategies)
            for b in fresh:
                sym = b.symbol
                if sym in self.positions or sym in self.pending or sym in self.state.symbols_traded:
                    continue
                if now - b.end > self.max_signal_age:
                    continue                               # stale: a replayed bar must never trade
                filtered = ("risk_off" if not self.risk_on else
                            "off_watchlist" if self.watchlist is not None and sym.upper() not in self.watchlist
                            else None)
                sig = first_signal(self.ctx[sym], self.enabled, self.cfg.params)
                if sig is None:
                    if fewer and sym not in self._ghosted:
                        ghost = first_signal(self.ctx[sym], self.all_strategies, self.cfg.params)
                        if ghost is not None:
                            self._ghost(ghost, b.end, filtered or "strategy_disabled")
                    continue
                if filtered:
                    self._ghost(sig, b.end, filtered)
                    continue
                snap = self._snapshot(sig)
                skip, mult, lesson_ids = self.lessons.evaluate(snap) if self.lessons else (False, 1.0, [])
                if skip:
                    self._log(b.end, "lesson_skip", sym, strategy=sig.strategy, lessons=lesson_ids)
                    self._ghost(sig, b.end, "filtered_lesson", snap, lesson_ids=lesson_ids)
                    continue
                limits = base_limits
                if mult < 1.0:
                    limits = dataclasses.replace(limits, risk_per_trade=limits.risk_per_trade * mult)
                qty, why = check_entry(sig, b.end, self.state, self.sleeve, limits, self.cfg.session,
                                       self.last_entry, self.cfg.universe.min_price)
                if qty < 1:
                    self._log(b.end, "veto", sym, strategy=sig.strategy, reasons=why)
                    if why and all(r.startswith(_GHOSTABLE_VETOES) for r in why):
                        self._ghost(sig, b.end, "vetoed", snap, reasons=why, lesson_ids=lesson_ids)
                    continue
                stop, target = round(sig.stop, 2), round(sig.target, 2)
                intent = EntryIntent(sym, qty, round(sig.entry, 2), stop, target, sig.strategy, sig.reason,
                                     self.entry_order_id(self.day, sym), b.end)
                self.pending[sym] = intent
                self.state.entries_today += 1
                self.state.open_or_pending += 1
                self.state.committed_notional += qty * intent.entry_ref
                self.state.symbols_traded.add(sym)
                intents.append(intent)
                self._snapshots[sym] = snap
                self.journal.track(sig, snap, "taken", b.end, lesson_ids=lesson_ids, lesson_mult=mult)
                self._log(b.end, "entry", sym, strategy=sig.strategy, qty=qty, entry=intent.entry_ref,
                          stop=stop, target=target, reason=sig.reason,
                          **({"lessons": lesson_ids, "lesson_mult": mult} if lesson_ids else {}))
        return intents

    # ------------------------------------------------------------------ feedback
    def apply_fill(self, fill: FillEvent) -> None:
        sym = fill.symbol
        if fill.side == "buy":
            intent = self.pending.pop(sym, None)
            if intent is None:
                self._log(fill.ts, "warn", sym, msg="buy fill with no pending entry — ignored")
                return
            self.state.committed_notional += fill.qty * fill.price - intent.qty * intent.entry_ref
            self.positions[sym] = Position(sym, fill.qty, fill.price, intent.stop, intent.target,
                                           intent.strategy, fill.ts, intent.client_order_id,
                                           entry_ref=intent.entry_ref, snapshot=self._snapshots.pop(sym, {}))
            self._log(fill.ts, "filled_buy", sym, qty=fill.qty, price=fill.price)
        elif fill.side == "sell":
            pos = self.positions.get(sym)
            if pos is None:
                self._log(fill.ts, "warn", sym, msg="sell fill with no open position — ignored")
                return
            qty = min(fill.qty, pos.qty)
            pnl = (fill.price - pos.entry_price) * qty - fill.fees
            risk_ps = max(pos.entry_price - pos.stop, 1e-9)
            risk = risk_ps * qty
            self.trades.append(Trade(sym, pos.strategy, qty, pos.entry_price, fill.price, pos.entry_ts,
                                     fill.ts, pnl, fill.fees, fill.reason, pnl / risk, snapshot=dict(pos.snapshot),
                                     mae_r=(min(pos.lo, fill.price) - pos.entry_price) / risk_ps,
                                     mfe_r=(max(pos.hi, fill.price) - pos.entry_price) / risk_ps,
                                     entry_ref=pos.entry_ref))
            self.journal.watch_exit(sym, fill.ts, fill.price, risk_ps)
            self.state.realized_pnl += pnl
            self.state.committed_notional -= qty * pos.entry_price
            pos.qty -= qty
            self._log(fill.ts, "filled_sell", sym, qty=qty, price=fill.price, reason=fill.reason, pnl=round(pnl, 2))
            if pos.qty <= 1e-9:
                del self.positions[sym]
                self._exit_sent.pop(sym, None)
                self.state.open_or_pending -= 1
            self._mark()
        else:
            raise ValueError(f"unknown side {fill.side!r}")

    def apply_reject(self, symbol: str, reason: str = "", ts: Optional[datetime] = None) -> None:
        """An entry that never became a position (rejected, cancelled, or
        timed out with zero fill). Frees its slot and cash. The name stays in
        symbols_traded and entries_today keeps counting it, deliberately: a
        broker that keeps rejecting must hit max_trades_per_day, not loop."""
        intent = self.pending.pop(symbol, None)
        if intent is None:
            return
        self.state.open_or_pending -= 1
        self.state.committed_notional -= intent.qty * intent.entry_ref
        self._log(ts or intent.decision_ts, "entry_failed", symbol, reason=reason)

    def adopt_position(self, symbol: str, qty: float, entry_price: float, stop: float, target: float,
                       strategy: str, entry_ts: datetime, client_order_id: str = "") -> None:
        """Crash recovery: a position that already exists at the broker."""
        if symbol in self.positions:
            return
        self.positions[symbol] = Position(symbol, qty, entry_price, stop, target, strategy, entry_ts,
                                          client_order_id or self.entry_order_id(self.day, symbol))
        self.pending.pop(symbol, None)
        if symbol not in self.state.symbols_traded:
            self.state.symbols_traded.add(symbol)
            self.state.entries_today += 1
        self.state.open_or_pending += 1
        self.state.committed_notional += qty * entry_price
        self._log(entry_ts, "adopted", symbol, qty=qty, entry=entry_price)

    def restore_progress(self, entries_today: int, symbols_traded: Set[str], realized_pnl: float,
                         trades: Optional[List[Trade]] = None) -> None:
        """Crash recovery: what today's earlier run had already done."""
        self.state.entries_today = max(self.state.entries_today, entries_today)
        self.state.symbols_traded |= set(symbols_traded)
        self.state.realized_pnl = realized_pnl
        if trades:
            self.trades = list(trades)

    def force_halt(self, reason: str, now: datetime) -> None:
        """External kill switch (Firestore flag, operator). Same effect as a
        loss halt: no entries, and — per config — flatten."""
        if not self.state.halted:
            self.state.halted, self.state.halt_reason = True, reason
            self._log(now, "halt", reason=reason)


if __name__ == "__main__":
    from datetime import timezone
    from daytrade_config import config_from_dict
    from daytrade_strategies import _flat_open, _path
    from intraday_data import et_datetime

    day = date(2026, 3, 4)
    cfg = config_from_dict({"sleeve_capital": 25_000})
    UTC = timezone.utc

    def breakout_rows(sym_px=100.0):
        return _flat_open(15, sym_px) + [(sym_px, sym_px + .2, sym_px - .1, sym_px + .1, 1000)] * 2 + \
               [(sym_px + .15, sym_px + .7, sym_px + .1, sym_px + .6, 4000)]

    def bars_for(sym, rows):
        return [dataclasses.replace(b, symbol=sym) for b in _path(day, rows)]

    def run(engine, bars, until=None):
        """feed minute by minute like the real callers do"""
        out = []
        for b in bars:
            now = b.end
            out += engine.on_bars([b], now)
        return out

    print("TEST 1: breakout -> exactly one sized EntryIntent with a deterministic order id")
    eng = DayTradingEngine(cfg, day)
    intents = run(eng, bars_for("AAA", breakout_rows()))
    entries = [i for i in intents if isinstance(i, EntryIntent)]
    assert len(entries) == 1
    e = entries[0]
    import math
    assert e.qty == math.floor(25_000 * 0.20 / e.entry_ref) == 49, "20%-of-sleeve per-name cap binds at $100.60"
    assert e.symbol == "AAA" and e.stop < e.entry_ref < e.target
    assert e.client_order_id == "dt-20260304-AAA-buy"
    assert eng.state.entries_today == 1 and eng.state.open_or_pending == 1
    assert abs(eng.state.committed_notional - e.qty * e.entry_ref) < 1e-6
    print(f"✓ {e.qty} sh @ {e.entry_ref} stop {e.stop} target {e.target}; id={e.client_order_id}")

    print("TEST 2: position exists ONLY after a fill; a reject frees slot and cash")
    assert "AAA" not in eng.positions
    eng.apply_fill(FillEvent("AAA", "buy", 50, 100.62, et_datetime(day, time(9, 47)).astimezone(UTC)))
    assert eng.positions["AAA"].qty == 50 and not eng.pending
    eng2 = DayTradingEngine(cfg, day)
    run(eng2, bars_for("AAA", breakout_rows()))
    eng2.apply_reject("AAA", "insufficient buying power")
    assert eng2.state.open_or_pending == 0 and abs(eng2.state.committed_notional) < 1e-9
    assert eng2.state.entries_today == 1 and "AAA" in eng2.state.symbols_traded
    print("✓ fill creates the position; reject restores cash/slots but still counts against trade limits")

    print("TEST 3: no re-entry into a name already traded today")
    more = bars_for("AAA", _flat_open(15) + [(100, 100.2, 99.9, 100.1, 1000)] * 2 +
                    [(100.15, 100.7, 100.1, 100.6, 4000)] + [(100.6, 100.7, 100.5, 100.55, 900)] * 5 +
                    [(100.6, 101.5, 100.55, 101.4, 6000)])
    eng3 = DayTradingEngine(cfg, day)
    got = [i for i in run(eng3, more) if isinstance(i, EntryIntent)]
    assert len(got) == 1
    print("✓ one entry per name per day")

    print("TEST 4: sell fill realizes NET pnl (fees), R-multiple and frees the slot")
    eng.apply_fill(FillEvent("AAA", "sell", 50, e.target, et_datetime(day, time(10, 30)).astimezone(UTC),
                             reason="target", fees=0.25))
    t = eng.trades[0]
    assert abs(t.pnl - ((e.target - 100.62) * 50 - 0.25)) < 1e-9
    assert t.r_multiple > 1.5 and not eng.positions and eng.state.open_or_pending == 0
    assert abs(eng.state.realized_pnl - t.pnl) < 1e-9 and abs(eng.state.committed_notional) < 1e-6
    print(f"✓ pnl ${t.pnl:.2f} net, {t.r_multiple:.2f}R, slot and cash freed")

    print("TEST 5: at flatten time open positions get an ExitIntent, pending entries a CancelIntent")
    eng5 = DayTradingEngine(cfg, day)
    run(eng5, bars_for("AAA", breakout_rows()))
    eng5.apply_fill(FillEvent("AAA", "buy", 50, 100.6, et_datetime(day, time(9, 47)).astimezone(UTC)))
    flat_ts = et_datetime(day, time(15, 45)).astimezone(UTC)
    out = eng5.on_bars([], flat_ts)
    assert [type(i).__name__ for i in out] == ["ExitIntent"] and out[0].qty == 50 and out[0].reason == "eod_flatten"
    assert eng5.on_bars([], flat_ts + timedelta(seconds=20)) == [], "must not spam exits every tick"
    assert len(eng5.on_bars([], flat_ts + timedelta(seconds=61))) == 1, "but retries after 60s if still open"
    eng5b = DayTradingEngine(cfg, day)
    run(eng5b, bars_for("AAA", breakout_rows()))                # entry pending, never filled
    outb = eng5b.on_bars([], flat_ts)
    assert [type(i).__name__ for i in outb] == ["CancelIntent"]
    print("✓ exit once, retry after 60s; in-flight entry is cancelled")

    print("TEST 6: no entries after last_entry even on a perfect setup")
    late = _flat_open(15) + [(100, 100.2, 99.9, 100.1, 1000)] * 350
    late_bars = bars_for("AAA", late)
    spike = dataclasses.replace(late_bars[-1], close=101.5, high=101.6, volume=90_000)
    eng6 = DayTradingEngine(cfg, day)
    run(eng6, late_bars[:-1])
    assert eng6.on_bars([spike], spike.end) == [] , "09:30 + 365 min = 15:35 is past last_entry"
    vetoes = [d for d in eng6.decisions if d["kind"] == "veto"]
    assert vetoes and any("entry window" in r for r in vetoes[-1]["reasons"]), \
        f"the signal must have FIRED and been vetoed for the entry window specifically, got {vetoes}"
    print("✓ 15:35 breakout fired a real signal and was vetoed for 'outside the entry window'")

    print("TEST 7: daily-loss halt flips the switch, then flattens")
    eng7 = DayTradingEngine(cfg, day)
    run(eng7, bars_for("AAA", breakout_rows()))
    eng7.apply_fill(FillEvent("AAA", "buy", 50, 100.6, et_datetime(day, time(9, 47)).astimezone(UTC)))
    eng7.state.realized_pnl = -700.0                           # -2.8% realized
    crash = Bar("AAA", et_datetime(day, time(9, 48)).astimezone(UTC), 100.6, 100.6, 84.0, 84.0, 9000)  # -$830 unrealized
    out7 = eng7.on_bars([crash], crash.end)
    assert eng7.state.halted and any(isinstance(i, ExitIntent) and i.reason == "halt_flatten" for i in out7)
    assert "daily loss" in eng7.state.halt_reason
    print(f"✓ {eng7.state.halt_reason}")

    print("TEST 8: stale bars never trade (a restart replaying the morning must not enter)")
    eng8 = DayTradingEngine(cfg, day)
    rows8 = bars_for("AAA", breakout_rows())
    got8 = eng8.on_bars(rows8, rows8[-1].end + timedelta(minutes=30))
    assert [i for i in got8 if isinstance(i, EntryIntent)] == []
    assert len(eng8.ctx["AAA"].bars) == len(rows8), "but the bars still build context (vwap, ATR, range)"
    fresh8 = DayTradingEngine(cfg, day)
    assert [i for i in fresh8.on_bars(rows8, rows8[-1].end + timedelta(seconds=10)) if isinstance(i, EntryIntent)]
    print("✓ 30-min-old signal ignored but bars kept; same signal 10s old trades")

    print("TEST 9: playbook can only restrict — risk-off, watchlist, strategy list, clamped multiplier")
    for pb, expect in [(Playbook(risk_on=False), 0), (Playbook(watchlist=["ZZZ"]), 0),
                       (Playbook(enabled_strategies=["gap_and_go_long"]), 0), (Playbook(), 1)]:
        e9 = DayTradingEngine(cfg, day, playbook=pb)
        n = len([i for i in run(e9, bars_for("AAA", breakout_rows())) if isinstance(i, EntryIntent)])
        assert n == expect, (pb, n)
    half = DayTradingEngine(cfg, day, playbook=Playbook(risk_multiplier=0.5))
    boosted = DayTradingEngine(cfg, day, playbook=Playbook(risk_multiplier=5.0))
    assert boosted.risk_multiplier == 1.0 and half.risk_multiplier == 0.5
    wide = [(100, 100.2, 97.0, 100, 1000)] + [(100, 100.2, 99.8, 100, 1000)] * 14 + [(100.3, 101.0, 100.2, 100.9, 6000)]
    q_full = [i for i in run(DayTradingEngine(cfg, day), bars_for("AAA", wide)) if isinstance(i, EntryIntent)]
    q_half = [i for i in run(half, bars_for("AAA", wide)) if isinstance(i, EntryIntent)]
    assert q_half[0].qty <= q_full[0].qty
    print("✓ risk-off / off-list / wrong-strategy -> no entry; multiplier clamped to <= 1.0 and shrinks size")

    print("TEST 10: crash recovery — adopt + restore, and no duplicate entry")
    e10 = DayTradingEngine(cfg, day)
    e10.restore_progress(entries_today=3, symbols_traded={"AAA", "BBB"}, realized_pnl=-120.0)
    e10.adopt_position("BBB", 30, 50.0, 49.0, 52.0, "orb_long", et_datetime(day, time(10, 0)).astimezone(UTC))
    assert e10.state.entries_today == 3 and e10.state.open_or_pending == 1
    assert abs(e10.state.committed_notional - 1500.0) < 1e-9
    assert [i for i in run(e10, bars_for("AAA", breakout_rows())) if isinstance(i, EntryIntent)] == []
    print("✓ AAA was already traded before the crash -> not re-entered; BBB adopted with its bracket levels")

    print("TEST 11: garbage in — duplicates, out-of-order and after-hours bars are ignored")
    e11 = DayTradingEngine(cfg, day)
    bs = bars_for("AAA", breakout_rows())
    after_hours = dataclasses.replace(bs[0], ts=et_datetime(day, time(17, 0)).astimezone(UTC))
    e11.on_bars(bs + bs[:3] + [after_hours], bs[-1].end)
    assert len(e11.ctx["AAA"].bars) == len(bs)
    print("✓ context has exactly the unique in-session bars")

    print("TEST 12: learned lessons — skip journals a ghost, risk_mult shrinks size, neither widens anything")
    skip_orb = {"id": "Lskip", "when": {"strategy": {"==": "orb_long"}}, "action": {"type": "skip"}}
    e12 = DayTradingEngine(cfg, day, playbook=Playbook(lessons=[skip_orb]))
    assert [i for i in run(e12, bars_for("AAA", wide)) if isinstance(i, EntryIntent)] == []
    recs = e12.journal_records()
    assert [(r["status"], r["lesson_ids"]) for r in recs] == [("filtered_lesson", ["Lskip"])]
    assert any(d["kind"] == "lesson_skip" for d in e12.decisions)
    shrink = {"id": "Lhalf", "when": {"rel_volume": {">": 1.0}}, "action": {"type": "risk_mult", "value": 0.5}}
    risk_bound = config_from_dict({"sleeve_capital": 25_000, "limits": {"max_position_pct": 1.0}})   # risk, not cap, binds
    q_base = [i for i in run(DayTradingEngine(risk_bound, day), bars_for("AAA", wide)) if isinstance(i, EntryIntent)]
    e12b = DayTradingEngine(risk_bound, day, playbook=Playbook(lessons=[shrink]))
    q_lesson = [i for i in run(e12b, bars_for("AAA", wide)) if isinstance(i, EntryIntent)]
    assert q_lesson and q_lesson[0].qty < q_base[0].qty, (q_lesson[0].qty, q_base[0].qty)
    widen = {"id": "Lbad", "when": {"rel_volume": {">": 1.0}}, "action": {"type": "risk_mult", "value": 3.0}}
    e12c = DayTradingEngine(risk_bound, day, playbook=Playbook(lessons=[widen, {"garbage": 1}]))
    q_widen = [i for i in run(e12c, bars_for("AAA", wide)) if isinstance(i, EntryIntent)]
    assert len(e12c.lessons) == 0 and q_widen[0].qty == q_base[0].qty, "an invalid/widening lesson is dropped"
    print(f"✓ skip -> ghost; 0.5x lesson -> {q_lesson[0].qty} sh vs {q_base[0].qty}; a 3x 'lesson' is ignored")

    print("TEST 13: playbook-filtered signals are journaled with outcomes but never traded")
    for pb, status in [(Playbook(watchlist=["ZZZ"]), "off_watchlist"), (Playbook(risk_on=False), "risk_off"),
                       (Playbook(enabled_strategies=["gap_and_go_long"]), "strategy_disabled")]:
        e13 = DayTradingEngine(cfg, day, playbook=pb)
        rows13 = breakout_rows() + [(100.7, 101.9, 100.6, 101.0, 3000)]
        assert [i for i in run(e13, bars_for("AAA", rows13)) if isinstance(i, EntryIntent)] == []
        recs = e13.journal_records()
        assert [r["status"] for r in recs] == [status], (status, recs)
        assert recs[0]["exit_virtual"] == "target" and recs[0]["r_virtual"] > 1.5
        assert recs[0]["snapshot"]["strategy"] == "orb_long" and recs[0]["snapshot"]["minutes_since_open"] == 17
    print("✓ off-watchlist / risk-off / disabled-strategy ghosts each carry a snapshot and a +R virtual outcome")

    print("TEST 14: a taken trade carries its snapshot, MAE/MFE and slippage into the journal")
    e14 = DayTradingEngine(cfg, day, market={"spy_prev_change": -0.01})
    got = [i for i in run(e14, bars_for("AAA", breakout_rows())) if isinstance(i, EntryIntent)]
    fill_ts = et_datetime(day, time(9, 48)).astimezone(UTC)
    e14.apply_fill(FillEvent("AAA", "buy", got[0].qty, 100.65, fill_ts))
    dip = [Bar("AAA", fill_ts, 100.65, 100.9, 100.3, 100.5, 3000),
           Bar("AAA", fill_ts + timedelta(minutes=1), 100.5, 101.2, 100.4, 101.1, 3000)]
    for b in dip:
        e14.on_bars([b], b.end)
    e14.apply_fill(FillEvent("AAA", "sell", got[0].qty, 101.0, fill_ts + timedelta(minutes=2), reason="eod_flatten"))
    t = e14.trades[0]
    risk_ps = 100.65 - got[0].stop
    assert abs(t.mae_r - (100.3 - 100.65) / risk_ps) < 1e-9 and abs(t.mfe_r - (101.2 - 100.65) / risk_ps) < 1e-9
    assert t.snapshot["spy_prev_change"] == -0.01 and t.entry_ref == got[0].entry_ref
    rec = [r for r in e14.journal_records() if r["status"] == "taken"][0]
    assert rec["r_real"] == round(t.r_multiple, 6) and rec["slippage_bps"] > 0 and rec["exit_reason"] == "eod_flatten"
    print(f"✓ MAE {t.mae_r:+.2f}R, MFE {t.mfe_r:+.2f}R, slippage {rec['slippage_bps']:.1f}bps, snapshot kept")

    print("TEST 15: capacity vetoes are journaled once; window vetoes are not (never tradable)")
    e15 = DayTradingEngine(config_from_dict({"sleeve_capital": 25_000, "limits": {"max_positions": 1}}), day)
    both = sorted(bars_for("AAA", breakout_rows()) + bars_for("BBB", breakout_rows(50.0)), key=lambda b: (b.ts, b.symbol))
    for b in both:
        e15.on_bars([b], b.end)
    recs = e15.journal_records()
    assert sorted((r["symbol"], r["status"]) for r in recs) == [("AAA", "taken"), ("BBB", "vetoed")], recs
    assert any("max_positions" in x for x in [r for r in recs if r["status"] == "vetoed"][0]["reasons"])
    assert [r for r in eng6.journal_records()] == [], "the 15:35 window veto is not a counterfactual"
    print("✓ BBB blocked by max_positions -> one 'vetoed' ghost; late-day window veto -> nothing")
    print("\nAll daytrade_engine tests passed.")
