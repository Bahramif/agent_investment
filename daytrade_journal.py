"""
daytrade_journal.py
====================
Phase 0 of the failure-learning loop: the record a failure can be learned
FROM. The engine already knows what it did (Trade); this module adds what
the market looked like when it decided, and what happened to the setups it
did NOT take.

Three things per signal, all pure / no I/O, owned by DayTradingEngine so the
backtest and the live runner produce the identical journal:

  1. ENTRY SNAPSHOT (`entry_features`) — a fixed vocabulary of numbers known
     at decision time (gap, relative volume, distance from VWAP, tape, ...).
     FEATURES is also the ONLY vocabulary a learned lesson may use
     (daytrade_lessons.py), so every rule is checkable against history.

  2. VIRTUAL OUTCOME (`SignalTracker`) — every first signal per symbol per day
     is resolved as if it had been taken on its own: fill at the next bar's
     open plus slippage, stop before target when a bar hits both, flat at the
     flatten time. Same pessimism as daytrade_backtest.SimBroker. Because it
     ignores slots, vetoes and the playbook, `r_virtual` is comparable across
     every record — taken, vetoed, filtered by the playbook, filtered by a
     lesson — which is what makes counterfactuals ("what did that filter
     cost?") and unbiased lesson validation possible. Regulatory fees are
     not charged on virtual outcomes (they are ~0.01R); slippage is.

  3. POST-EXIT WATCH — after a real exit, the best price over the next N
     minutes. A stop followed by a run to +1R is "stop noise", not "wrong
     trade", and those need opposite fixes.

Records are plain dicts (Firestore-ready). Risk unit for virtual R is the
SIGNAL's entry-to-stop distance, so a gap through the stop at the fill bar
reads worse than -1R — as it would in reality.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

from intraday_data import Bar, minutes_since_open, to_et

MAX_ENTRY_DELAY = timedelta(minutes=3)       # same as daytrade_backtest: no fill at a stale price

# Feature vocabulary: kind + the bucket edges used by attribution stats and the
# deterministic lesson miner. Adding a feature here makes it usable in lessons.
FEATURES: Dict[str, Dict[str, Any]] = {
    "strategy":           {"kind": "category", "desc": "which setup fired"},
    "minutes_since_open": {"kind": "number", "bins": [15, 30, 60, 120, 240], "desc": "minutes after 09:30 ET"},
    "gap_pct":            {"kind": "number", "bins": [-0.02, 0.0, 0.02, 0.04, 0.08], "desc": "open / prev close - 1"},
    "day_change_pct":     {"kind": "number", "bins": [-0.01, 0.0, 0.01, 0.02, 0.04], "desc": "entry / today's open - 1"},
    "rel_volume":         {"kind": "number", "bins": [1.5, 2.0, 3.0, 5.0], "desc": "signal bar volume / earlier bars' mean"},
    "atr_pct":            {"kind": "number", "bins": [0.001, 0.002, 0.004], "desc": "1-min ATR / entry"},
    "vwap_dist_pct":      {"kind": "number", "bins": [0.0, 0.0025, 0.005, 0.01], "desc": "entry / VWAP - 1"},
    "or_range_pct":       {"kind": "number", "bins": [0.005, 0.01, 0.02, 0.04], "desc": "15-min opening range width / entry"},
    "stop_pct":           {"kind": "number", "bins": [0.002, 0.004, 0.008, 0.015], "desc": "(entry - stop) / entry"},
    "tape_change_pct":    {"kind": "number", "bins": [-0.01, -0.003, 0.003, 0.01],
                           "desc": "median open->now move of every name the engine watches"},
    "spy_prev_change":    {"kind": "number", "bins": [-0.01, -0.003, 0.003, 0.01], "desc": "SPY yesterday close/close"},
    "spy_mom_5d":         {"kind": "number", "bins": [-0.02, 0.0, 0.02], "desc": "SPY 5-day change"},
}

# Record statuses. Everything but "taken" is a ghost: a signal the engine saw but did not trade.
STATUSES = ("taken", "vetoed", "off_watchlist", "strategy_disabled", "risk_off", "filtered_lesson")


def _r6(x: Optional[float]) -> Optional[float]:
    return None if x is None or not math.isfinite(x) else round(x, 6)


def entry_features(ctx: Any, sig: Any, contexts: Dict[str, Any], market: Optional[Dict[str, float]] = None,
                   risk_multiplier: float = 1.0) -> Dict[str, Any]:
    """Snapshot at decision time. `ctx` is the signal's SymbolContext,
    `contexts` every symbol the engine is watching (for the tape read)."""
    market = market or {}
    last = ctx.bars[-1]
    entry = sig.entry
    m = minutes_since_open(last.ts)
    atr = ctx.atr()
    rv = None
    if len(ctx.bars) >= 4:
        base = statistics.fmean(b.volume for b in ctx.bars[:-1])
        rv = last.volume / base if base > 0 else None
    or_range = None
    if m >= 15:
        opening = ctx.opening_bars(15)
        if len(opening) >= 3:
            or_range = (max(b.high for b in opening) - min(b.low for b in opening)) / entry
    moves = [c.bars[-1].close / c.bars[0].open - 1.0 for c in contexts.values() if c.bars and c.bars[0].open > 0]
    return {
        "strategy": sig.strategy,
        "minutes_since_open": m,
        "gap_pct": _r6(ctx.day_open / ctx.prev_close - 1.0) if ctx.prev_close else None,
        "day_change_pct": _r6(entry / ctx.day_open - 1.0) if ctx.day_open > 0 else None,
        "rel_volume": _r6(rv),
        "atr_pct": _r6(atr / entry) if atr else None,
        "vwap_dist_pct": _r6(entry / ctx.vwap - 1.0) if ctx.vwaps and ctx.vwap > 0 else None,
        "or_range_pct": _r6(or_range),
        "stop_pct": _r6((entry - sig.stop) / entry),
        "tape_change_pct": _r6(statistics.median(moves)) if moves else None,
        "spy_prev_change": _r6(market.get("spy_prev_change")),
        "spy_mom_5d": _r6(market.get("spy_mom_5d")),
        "risk_multiplier": _r6(risk_multiplier),
    }


@dataclass
class _Virtual:
    rec: Dict[str, Any]
    decision_ts: datetime
    risk: float
    stop: float
    target: float
    fill: Optional[float] = None
    hi: float = 0.0
    lo: float = 0.0
    done: bool = False


@dataclass
class _ExitWatch:
    exit_ts: datetime
    exit_price: float
    risk: float
    until: datetime
    best: float


class SignalTracker:
    def __init__(self, day: date, slippage_bps: float, flatten: time, post_exit_minutes: int = 30):
        self.day, self.flatten = day, flatten
        self.slip = slippage_bps / 1e4
        self.post_exit = timedelta(minutes=post_exit_minutes)
        self._open: Dict[str, List[_Virtual]] = {}
        self._all: List[_Virtual] = []
        self._exits: Dict[str, _ExitWatch] = {}
        self._last_close: Dict[str, float] = {}
        self.partial = False                         # set by the runner after a mid-day restart

    # ------------------------------------------------------------------ inputs
    def track(self, sig: Any, snapshot: Dict[str, Any], status: str, decision_ts: datetime,
              reasons: Sequence[str] = (), lesson_ids: Sequence[str] = (), lesson_mult: float = 1.0) -> Dict[str, Any]:
        if status not in STATUSES:
            raise ValueError(f"unknown journal status {status!r}")
        rec = {"day": self.day.isoformat(), "symbol": sig.symbol, "strategy": sig.strategy, "status": status,
               "ts": decision_ts.isoformat(), "reasons": list(reasons), "lesson_ids": list(lesson_ids),
               "lesson_mult": _r6(lesson_mult), "snapshot": dict(snapshot),
               "entry": _r6(sig.entry), "stop": _r6(sig.stop), "target": _r6(sig.target),
               "r_virtual": None, "exit_virtual": None, "mae_r_virtual": None, "mfe_r_virtual": None}
        v = _Virtual(rec, decision_ts, max(sig.entry - sig.stop, 1e-9), sig.stop, sig.target)
        self._open.setdefault(sig.symbol, []).append(v)
        self._all.append(v)
        return rec

    def watch_exit(self, symbol: str, exit_ts: datetime, exit_price: float, risk_per_share: float) -> None:
        self._exits[symbol] = _ExitWatch(exit_ts, exit_price, max(risk_per_share, 1e-9),
                                         exit_ts + self.post_exit, exit_price)

    def _close(self, v: _Virtual, px: float, reason: str) -> None:
        v.done = True
        v.rec["exit_virtual"] = reason
        v.rec["r_virtual"] = _r6((px - v.fill) / v.risk)
        v.rec["mae_r_virtual"] = _r6((min(v.lo, px) - v.fill) / v.risk)
        v.rec["mfe_r_virtual"] = _r6((max(v.hi, px) - v.fill) / v.risk)

    def on_bar(self, b: Bar) -> None:
        sym = b.symbol
        self._last_close[sym] = b.close
        w = self._exits.get(sym)
        if w is not None and w.exit_ts < b.ts <= w.until:
            w.best = max(w.best, b.high)
        for v in self._open.get(sym, ()):
            if v.done or b.ts < v.decision_ts:
                continue
            if v.fill is None:
                if b.ts - v.decision_ts > MAX_ENTRY_DELAY:
                    v.done = True
                    v.rec["exit_virtual"] = "no_fill"
                    continue
                v.fill = b.open * (1 + self.slip)
                v.hi, v.lo = b.high, b.low
            else:
                if to_et(b.ts).time() >= self.flatten:
                    self._close(v, b.open * (1 - self.slip), "eod_flatten")
                    continue
                v.hi, v.lo = max(v.hi, b.high), min(v.lo, b.low)
            # same order and pessimism as SimBroker: gap-through first, stop before target
            if b.open <= v.stop:
                self._close(v, b.open * (1 - self.slip), "stop")
            elif b.open >= v.target:
                self._close(v, b.open, "target")
            elif b.low <= v.stop:
                self._close(v, v.stop * (1 - self.slip), "stop")
            elif b.high >= v.target + 0.01:
                self._close(v, v.target, "target")

    def finalize(self) -> None:
        """End of data: anything still open is closed at its last print."""
        for v in self._all:
            if v.done:
                continue
            if v.fill is None:
                v.done = True
                v.rec["exit_virtual"] = "no_fill"
                continue
            px = self._last_close.get(v.rec["symbol"], v.fill) * (1 - self.slip)
            self._close(v, px, "eod_forced")

    # ------------------------------------------------------------------ output
    def records(self, trades: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        """All records, with the REAL outcome merged into each taken one.
        Partial exits of one position are combined (qty-weighted R)."""
        by_sym: Dict[str, List[Any]] = {}
        for t in trades:
            by_sym.setdefault(t.symbol, []).append(t)
        out = []
        for v in self._all:
            rec = dict(v.rec)
            rec["partial_day"] = self.partial
            if rec["status"] == "taken":
                ts = by_sym.get(rec["symbol"], [])
                if ts:
                    qty = sum(t.qty for t in ts) or 1.0
                    first, last = ts[0], ts[-1]
                    w = self._exits.get(rec["symbol"])
                    rec.update({
                        "r_real": _r6(sum(t.r_multiple * t.qty for t in ts) / qty),
                        "pnl": _r6(sum(t.pnl for t in ts)),
                        "exit_reason": last.reason,
                        "mae_r": _r6(min(t.mae_r for t in ts)),
                        "mfe_r": _r6(max(t.mfe_r for t in ts)),
                        "slippage_bps": _r6((first.entry_price / first.entry_ref - 1.0) * 1e4) if first.entry_ref else None,
                        "mfe_after_exit_r": _r6((w.best - w.exit_price) / w.risk) if w else None,
                    })
                else:
                    rec.update({"r_real": None, "exit_reason": "not_filled"})
            out.append(rec)
        return out


if __name__ == "__main__":
    import dataclasses
    from types import SimpleNamespace as NS
    from daytrade_strategies import _ctx, _flat_open, _path

    day = date(2026, 3, 4)

    def sig(sym="TST", entry=100.6, stop=100.0, target=101.8, strategy="orb_long"):
        return NS(symbol=sym, strategy=strategy, entry=entry, stop=stop, target=target)

    def bars(rows, sym="TST"):
        return [dataclasses.replace(b, symbol=sym) for b in _path(day, rows)]

    print("TEST 1: entry snapshot uses only decision-time data and the fixed vocabulary")
    rows = _flat_open(15) + [(100.0, 100.2, 99.9, 100.1, 1000)] * 2 + [(100.15, 100.7, 100.1, 100.6, 4000)]
    ctx = _ctx(bars(rows), prev_close=97.0)
    other = _ctx(bars(_flat_open(18, 50.0), "OTH"))
    snap = entry_features(ctx, sig(), {"TST": ctx, "OTH": other}, {"spy_prev_change": -0.012, "spy_mom_5d": 0.01}, 0.5)
    assert set(snap) - {"risk_multiplier"} == set(FEATURES), set(snap) ^ set(FEATURES)
    assert snap["minutes_since_open"] == 17 and abs(snap["gap_pct"] - (100 / 97 - 1)) < 1e-4
    assert snap["rel_volume"] > 3 and snap["or_range_pct"] is not None and snap["spy_prev_change"] == -0.012
    assert abs(snap["stop_pct"] - 0.6 / 100.6) < 1e-6 and snap["risk_multiplier"] == 0.5
    no_prev = entry_features(_ctx(bars(rows)), sig(), {}, None)
    assert no_prev["gap_pct"] is None and no_prev["spy_prev_change"] is None and no_prev["tape_change_pct"] is None
    print(f"✓ {len(snap)} fields; gap {snap['gap_pct']:+.2%}, rv {snap['rel_volume']:.1f}x; missing inputs -> None")

    def tracker():
        return SignalTracker(day, slippage_bps=0.0, flatten=time(15, 45), post_exit_minutes=30)

    def feed(tr, rows, sym="TST"):
        for b in bars(rows, sym):
            tr.on_bar(b)

    signal_end = _path(day, [QUIET := (100.6, 100.7, 100.5, 100.6, 900)] * 18)[17].end   # decision at 09:48

    print("TEST 2: virtual bracket — next-bar fill, target only when traded THROUGH")
    tr = tracker()
    tr.track(sig(), {}, "vetoed", signal_end, reasons=["at max_positions (5)"])
    feed(tr, [QUIET] * 18 + [(100.6, 101.2, 100.5, 101.0, 900), (101.0, 101.81, 100.9, 101.5, 900),
                             (101.5, 101.9, 101.4, 101.8, 900)])
    r = tr.records()[0]
    assert r["exit_virtual"] == "target" and abs(r["r_virtual"] - 2.0) < 1e-9, r
    assert r["mfe_r_virtual"] >= 2.0 and r["mae_r_virtual"] <= 0
    print(f"✓ filled 100.60 at 09:48 open, 101.81 only touched, 101.90 fills target -> {r['r_virtual']:+.2f}R")

    print("TEST 3: stop first when one bar spans both; a gap through the stop reads worse than -1R")
    tr = tracker()
    tr.track(sig(), {}, "off_watchlist", signal_end)
    feed(tr, [QUIET] * 18 + [(100.6, 102.0, 99.9, 101.0, 900)])
    assert tr.records()[0]["exit_virtual"] == "stop" and abs(tr.records()[0]["r_virtual"] + 1.0) < 1e-9
    tr = tracker()
    tr.track(sig(), {}, "filtered_lesson", signal_end, lesson_ids=["L1"])
    feed(tr, [QUIET] * 18 + [(100.6, 100.7, 100.5, 100.6, 900), (99.4, 99.5, 99.0, 99.2, 900)])
    g = tr.records()[0]
    assert g["exit_virtual"] == "stop" and g["r_virtual"] < -1.9 and g["lesson_ids"] == ["L1"]
    print(f"✓ both-hit bar -> stop (-1.00R); gap to 99.40 -> {g['r_virtual']:+.2f}R")

    print("TEST 4: no bar within 3 minutes -> no_fill (no R); open at the end -> closed at last print")
    tr = tracker()
    tr.track(sig(), {}, "vetoed", signal_end)
    late = _path(day, [QUIET] * 30)[25]
    tr.on_bar(dataclasses.replace(late, symbol="TST"))
    assert tr.records()[0]["exit_virtual"] == "no_fill" and tr.records()[0]["r_virtual"] is None
    tr = tracker()
    tr.track(sig(), {}, "vetoed", signal_end)
    feed(tr, [QUIET] * 18 + [(100.6, 100.9, 100.5, 100.9, 900)])
    tr.finalize()
    assert tr.records()[0]["exit_virtual"] == "eod_forced" and abs(tr.records()[0]["r_virtual"] - 0.5) < 1e-9
    print("✓ stale -> no_fill; unresolved -> eod_forced at +0.50R")

    print("TEST 5: flatten time closes virtual positions at that bar's open, with slippage")
    tr = SignalTracker(day, slippage_bps=10.0, flatten=time(9, 50))
    tr.track(sig(), {}, "vetoed", signal_end)
    feed(tr, [QUIET] * 18 + [(100.6, 100.8, 100.5, 100.7, 900), (100.8, 100.9, 100.7, 100.8, 900),
                             (100.8, 100.9, 100.7, 100.8, 900)])
    r = tr.records()[0]
    fill = 100.6 * 1.001
    assert r["exit_virtual"] == "eod_flatten" and abs(r["r_virtual"] - ((100.8 * 0.999 - fill) / 0.6)) < 1e-6
    print(f"✓ flat at 09:50 open, both sides slipped -> {r['r_virtual']:+.3f}R")

    print("TEST 6: taken record merges the real trade(s) and the post-exit watch")
    tr = tracker()
    tr.track(sig(), {"strategy": "orb_long"}, "taken", signal_end)
    exit_bar = _path(day, [QUIET] * 20)[19]
    tr.watch_exit("TST", exit_bar.ts, 100.0, 0.6)
    feed(tr, [QUIET] * 18 + [(100.6, 100.7, 100.0, 100.1, 900), (100.1, 100.2, 99.95, 100.0, 900),
                             (100.0, 100.5, 99.9, 100.4, 900), (100.4, 100.9, 100.3, 100.8, 900)])
    t = NS(symbol="TST", qty=40, entry_price=100.62, entry_ref=100.6, pnl=-24.8, r_multiple=-1.03, reason="stop",
           mae_r=-1.05, mfe_r=0.13)
    r = tr.records([t])[0]
    assert r["r_real"] == -1.03 and r["exit_reason"] == "stop" and abs(r["slippage_bps"] - 1.988) < 0.01
    assert abs(r["mfe_after_exit_r"] - 1.5) < 1e-6, r["mfe_after_exit_r"]
    missing = tracker()
    missing.track(sig(), {}, "taken", signal_end)
    assert missing.records([])[0]["exit_reason"] == "not_filled"
    print(f"✓ real -1.03R, {r['slippage_bps']:.1f}bps slippage; price ran +{r['mfe_after_exit_r']:.1f}R after the stop")

    print("TEST 7: unknown status is a bug, not a silent record")
    try:
        tracker().track(sig(), {}, "maybe", signal_end)
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("✓ ok")
    print("\nAll daytrade_journal tests passed.")
