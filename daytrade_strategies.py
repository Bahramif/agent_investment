"""
daytrade_strategies.py
=======================
Long-only intraday signal generators. Each strategy is a PURE function:
(SymbolContext, params) -> Optional[Signal]. No I/O, no clock, no broker, no
LLM — so the exact same code produces signals in the backtest and in the
live runner, which is what makes backtest numbers mean something.

These are deliberately boring, textbook setups. The point of this module is
NOT to claim any of them has edge — most retail intraday rules don't once
costs are charged — but to give daytrade_backtest.py something concrete to
falsify. A strategy earns a live slot only by surviving the promotion gate.

A Signal carries a reference entry (the signal bar's close), a stop and a
target. Stops are the TIGHTER of a structural level (opening-range low, VWAP
pullback low, ...) and an ATR-based distance, so a wide structure can never
produce an oversized risk. Sizing/vetoes happen in daytrade_risk.py, not here.

Volume is only ever used RELATIVELY (this bar vs. earlier bars the same day)
because minute bars come from IEX, a small slice of consolidated volume — see
intraday_data.py.

`random_long` is the NULL model, not a strategy: entries at random times with
the same stop/target machinery. Its net result is what a strategy has to beat
to claim its signal adds anything beyond market drift + exit mechanics.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional

from intraday_data import Bar, minutes_since_open, session_date


@dataclass
class SymbolContext:
    symbol: str
    day: date
    prev_close: Optional[float]
    bars: List[Bar]                      # today's regular-session bars so far, oldest first
    vwaps: List[float] = field(default_factory=list)   # cumulative VWAP AFTER each bar
    cache: Dict[str, object] = field(default_factory=dict)   # per-symbol-day memo (opening ranges never change)

    def opening_bars(self, minutes: int) -> List[Bar]:
        """Bars in the first `minutes` of the session. Memoized ONCE the window
        has closed: after that no new bar can fall inside it, and recomputing
        it per bar made backtests quadratic in bars-per-day."""
        key = f"open{minutes}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        got = [b for b in self.bars if minutes_since_open(b.ts) < minutes]
        if self.bars and minutes_since_open(self.bars[-1].ts) >= minutes:
            self.cache[key] = got
        return got

    @property
    def day_open(self) -> float:
        return self.bars[0].open

    @property
    def vwap(self) -> float:
        return self.vwaps[-1]

    def atr(self, n: int = 14, min_n: int = 6) -> Optional[float]:
        """Mean true range of the last n one-minute bars, or of however many
        exist if that's at least min_n (each needs a prior close, so that's
        min_n+1 bars). The floor matters: gap-and-go's window opens at minute
        5, and demanding a full 14 would silently kill every signal before
        minute 15. None below the floor — too few bars to size a stop."""
        n = min(n, len(self.bars) - 1)
        if n < min_n:
            return None
        trs = []
        for prev, cur in zip(self.bars[-n - 1:-1], self.bars[-n:]):
            trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
        return sum(trs) / len(trs)


@dataclass(frozen=True)
class Signal:
    symbol: str
    strategy: str
    entry: float          # reference price: the signal bar's close
    stop: float
    target: float
    reason: str


def update_vwaps(vwaps: List[float], cum: List[float], bar: Bar) -> None:
    """Incremental cumulative VWAP. `cum` is [sum(price*vol), sum(vol)] carried
    by the caller so this is O(1) per bar."""
    cum[0] += bar.typical * bar.volume
    cum[1] += bar.volume
    vwaps.append(cum[0] / cum[1] if cum[1] > 0 else bar.close)


DEFAULTS: Dict[str, Dict[str, float]] = {
    "orb_long": {"or_minutes": 15, "min_or_bars": 3, "vol_mult": 1.5, "rr": 2.0, "stop_atr_mult": 1.5},
    "vwap_pullback_long": {"min_minutes": 30, "lookback": 12, "above_frac": 0.7, "tol": 0.001,
                           "min_up_on_day": 0.003, "rr": 1.5, "stop_atr_mult": 1.5},
    "gap_and_go_long": {"min_gap": 0.02, "hold_from_min": 5, "hold_to_min": 30, "vol_mult": 1.5,
                        "rr": 2.0, "stop_atr_mult": 1.5},
    "random_long": {"p_per_bar": 0.004, "from_min": 10, "to_min": 300, "seed": 0, "rr": 2.0, "stop_atr_mult": 1.5},
}


def _p(strategy: str, params: Optional[Dict[str, float]], name: str) -> float:
    if params and name in params:
        return params[name]
    return DEFAULTS[strategy][name]


def _finish(ctx: SymbolContext, strategy: str, structural_stop: float,
            params: Optional[Dict[str, float]], reason: str) -> Optional[Signal]:
    """Shared tail: stop = tighter of structure and ATR, target from R-multiple."""
    last = ctx.bars[-1]
    entry = last.close
    atr = ctx.atr()
    if atr is None or atr <= 0:
        return None
    stop = max(structural_stop, entry - _p(strategy, params, "stop_atr_mult") * atr)
    if not stop < entry:
        return None
    target = entry + _p(strategy, params, "rr") * (entry - stop)
    return Signal(ctx.symbol, strategy, entry, stop, target, reason)


def _rel_volume(ctx: SymbolContext) -> Optional[float]:
    if len(ctx.bars) < 4:
        return None
    baseline = statistics.fmean(b.volume for b in ctx.bars[:-1])
    return ctx.bars[-1].volume / baseline if baseline > 0 else None


def orb_long(ctx: SymbolContext, params: Optional[Dict[str, float]] = None) -> Optional[Signal]:
    """Opening-range breakout: first N minutes define a range; a FRESH close
    above its high, on above-average relative volume and above VWAP."""
    n = int(_p("orb_long", params, "or_minutes"))
    last = ctx.bars[-1]
    if minutes_since_open(last.ts) < n:
        return None
    opening = ctx.opening_bars(n)
    if len(opening) < int(_p("orb_long", params, "min_or_bars")):
        return None                                   # too thin (IEX skips quiet minutes) to define a range
    or_high, or_low = max(b.high for b in opening), min(b.low for b in opening)
    if len(ctx.bars) < 2 or not (last.close > or_high and ctx.bars[-2].close <= or_high):
        return None
    if not (last.close > ctx.vwap and last.close > last.open):
        return None
    rv = _rel_volume(ctx)
    if rv is None or rv < _p("orb_long", params, "vol_mult"):
        return None
    return _finish(ctx, "orb_long", or_low, params,
                   f"closed {last.close:.2f} above {n}m range high {or_high:.2f} on {rv:.1f}x volume")


def vwap_pullback_long(ctx: SymbolContext, params: Optional[Dict[str, float]] = None) -> Optional[Signal]:
    """Trend day pullback: stock is up on the day and has lived above VWAP; a
    dip touches VWAP and the current bar reclaims it with a bullish close."""
    last = ctx.bars[-1]
    if minutes_since_open(last.ts) < int(_p("vwap_pullback_long", params, "min_minutes")) or len(ctx.bars) < 20:
        return None
    if last.close < ctx.day_open * (1 + _p("vwap_pullback_long", params, "min_up_on_day")):
        return None
    look = int(_p("vwap_pullback_long", params, "lookback"))
    window = list(zip(ctx.bars[-look - 2:-2], ctx.vwaps[-look - 2:-2]))
    if len(window) < look:
        return None
    above = sum(1 for b, v in window if b.close > v) / len(window)
    if above < _p("vwap_pullback_long", params, "above_frac"):
        return None
    tol = _p("vwap_pullback_long", params, "tol")
    if not ctx.bars[-2].low <= ctx.vwaps[-2] * (1 + tol):
        return None
    if not (last.close > ctx.vwap and last.close > last.open):
        return None
    return _finish(ctx, "vwap_pullback_long", min(b.low for b in ctx.bars[-3:]), params,
                   f"pullback to VWAP {ctx.vwaps[-2]:.2f} reclaimed after {above:.0%} of bars above it")


def gap_and_go_long(ctx: SymbolContext, params: Optional[Dict[str, float]] = None) -> Optional[Signal]:
    """Gap-up continuation: opens >= min_gap above yesterday's close; after the
    first 5 minutes, a fresh close above that 5-minute high, above VWAP."""
    if not ctx.prev_close:
        return None
    last = ctx.bars[-1]
    m = minutes_since_open(last.ts)
    if not (_p("gap_and_go_long", params, "hold_from_min") <= m < _p("gap_and_go_long", params, "hold_to_min")):
        return None
    gap = ctx.day_open / ctx.prev_close - 1.0
    if gap < _p("gap_and_go_long", params, "min_gap"):
        return None
    first = ctx.opening_bars(int(_p("gap_and_go_long", params, "hold_from_min")))
    if len(first) < 2:
        return None
    hi5, lo5 = max(b.high for b in first), min(b.low for b in first)
    if len(ctx.bars) < 2 or not (last.close > hi5 and ctx.bars[-2].close <= hi5):
        return None
    if not (last.close > ctx.vwap and last.close > last.open):
        return None
    rv = _rel_volume(ctx)
    if rv is None or rv < _p("gap_and_go_long", params, "vol_mult"):
        return None
    return _finish(ctx, "gap_and_go_long", lo5, params,
                   f"gap {gap:+.1%}, closed {last.close:.2f} above 5m high {hi5:.2f} on {rv:.1f}x volume")


def random_long(ctx: SymbolContext, params: Optional[Dict[str, float]] = None) -> Optional[Signal]:
    """NULL MODEL — see module docstring. Deterministic per (seed, symbol, day,
    bar): a fresh Random seeded from that string, so results never depend on
    call order or PYTHONHASHSEED."""
    last = ctx.bars[-1]
    m = minutes_since_open(last.ts)
    if not (_p("random_long", params, "from_min") <= m < _p("random_long", params, "to_min")):
        return None
    rng = random.Random(f"{int(_p('random_long', params, 'seed'))}-{ctx.symbol}-{ctx.day.isoformat()}-{m}")
    if rng.random() >= _p("random_long", params, "p_per_bar"):
        return None
    return _finish(ctx, "random_long", 0.0, params, "null-model random entry")


STRATEGIES: Dict[str, Callable[[SymbolContext, Optional[Dict[str, float]]], Optional[Signal]]] = {
    "orb_long": orb_long,
    "vwap_pullback_long": vwap_pullback_long,
    "gap_and_go_long": gap_and_go_long,
    "random_long": random_long,
}


def first_signal(ctx: SymbolContext, enabled: List[str],
                 params: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[Signal]:
    """First enabled strategy (in list order) that fires on this bar."""
    for name in enabled:
        sig = STRATEGIES[name](ctx, (params or {}).get(name))
        if sig is not None:
            return sig
    return None


# -----------------------------------------------------------------------------
# Tests — hand-built bar paths, one per condition, so a failure names its cause.
# -----------------------------------------------------------------------------
def _path(day: date, rows) -> List[Bar]:
    """rows: (open, high, low, close, volume) starting at 09:30 ET, contiguous."""
    from intraday_data import trading_minutes
    mins = trading_minutes(day)
    return [Bar("TST", mins[i], *r) for i, r in enumerate(rows)]


def _ctx(bars: List[Bar], prev_close: Optional[float] = None) -> SymbolContext:
    ctx = SymbolContext("TST", session_date(bars[0].ts), prev_close, [])
    cum = [0.0, 0.0]
    for b in bars:
        ctx.bars.append(b)
        update_vwaps(ctx.vwaps, cum, b)
    return ctx


def _flat_open(n: int = 15, px: float = 100.0):
    return [(px, px + 0.2, px - 0.2, px, 1000)] * n


if __name__ == "__main__":
    day = date(2026, 3, 4)

    print("TEST 1: ORB fires on a fresh, high-volume close above the opening range")
    rows = _flat_open() + [(100.0, 100.2, 99.9, 100.1, 1000)] * 2 + [(100.15, 100.7, 100.1, 100.6, 4000)]
    sig = orb_long(_ctx(_path(day, rows)))
    assert sig and sig.strategy == "orb_long" and sig.entry == 100.6
    assert sig.stop < sig.entry < sig.target
    risk, reward = sig.entry - sig.stop, sig.target - sig.entry
    assert abs(reward / risk - 2.0) < 1e-9
    assert sig.stop >= 99.8, "stop must never be looser than the opening-range low"
    print(f"✓ entry {sig.entry} stop {sig.stop:.2f} target {sig.target:.2f} (2R): {sig.reason}")

    print("TEST 2: ORB stays quiet when any condition is missing")
    weak_vol = _flat_open() + [(100.0, 100.2, 99.9, 100.1, 1000)] * 2 + [(100.15, 100.7, 100.1, 100.6, 1100)]
    assert orb_long(_ctx(_path(day, weak_vol))) is None
    stale = _flat_open() + [(100.3, 100.7, 100.2, 100.6, 4000), (100.6, 100.8, 100.5, 100.7, 4000)]
    assert orb_long(_ctx(_path(day, stale))) is None, "second bar above the range is not a FRESH breakout"
    inside = _flat_open() + [(100.0, 100.2, 99.9, 100.1, 4000)]
    assert orb_long(_ctx(_path(day, inside))) is None
    too_early = _flat_open(10) + [(100.15, 100.9, 100.1, 100.8, 4000)]
    assert orb_long(_ctx(_path(day, too_early))) is None, "range not yet complete before minute 15"
    thin_or = [b for i, b in enumerate(_path(day, _flat_open(15) + [(100.15, 100.9, 100.1, 100.8, 4000)]))
               if i in (0, 1, 15)]
    assert orb_long(_ctx(thin_or)) is None, "only 2 bars in the opening range — too thin to define one"
    print("✓ low volume / stale breakout / inside range / too early / thin range -> no signal")

    print("TEST 3: a range made wide by ONE opening flush cannot produce an oversized stop (ATR cap)")
    # bar 0 flushes to 97 then recovers; every other bar is tight -> OR low is 97 but ATR is small
    flush = [(100, 100.2, 97.0, 100, 1000)] + [(100, 100.2, 99.8, 100, 1000)] * 14 + \
            [(100.3, 101.0, 100.2, 100.9, 6000)]
    sw = orb_long(_ctx(_path(day, flush)))
    assert sw is not None
    assert sw.stop > 97.0 + 1.0, f"stop {sw.stop:.2f} must be far tighter than the 97.00 structural low"
    assert abs((sw.entry - sw.stop) - 1.5 * _ctx(_path(day, flush)).atr()) < 1e-9, "stop is exactly 1.5 ATR away"
    print(f"✓ structural stop 97.00 rejected in favour of ATR stop {sw.stop:.2f}")

    print("TEST 3b: when every bar is wide, ATR is wide too — structure (the tighter one) wins")
    wide = [(100, 103, 97, 100, 1000)] * 15 + [(100.5, 104.6, 100.4, 104.5, 9000)]
    sw2 = orb_long(_ctx(_path(day, wide)))
    assert sw2 is not None and sw2.stop == 97.0
    print("✓ stop = OR low 97.00, never looser than structure")

    print("TEST 4: gap-and-go needs the gap, the window, and a fresh 5m-high break")
    g = [(103.0, 103.4, 102.8, 103.0, 1500)] * 5 + [(103.0, 103.3, 102.9, 103.1, 1000)] * 3 + \
        [(103.2, 104.2, 103.1, 104.0, 5000)]
    sg = gap_and_go_long(_ctx(_path(day, g), prev_close=100.0))
    assert sg and sg.strategy == "gap_and_go_long" and "gap +3.0%" in sg.reason
    assert gap_and_go_long(_ctx(_path(day, g), prev_close=102.5)) is None, "0.5% gap is below the 2% minimum"
    assert gap_and_go_long(_ctx(_path(day, g), prev_close=None)) is None
    print(f"✓ {sg.reason}")

    print("TEST 5: VWAP pullback: uptrend, dip to VWAP, bullish reclaim")
    up = [(100 + i * 0.03, 100.1 + i * 0.03, 99.95 + i * 0.03, 100.05 + i * 0.03, 1000) for i in range(38)]
    px = up[-1][3]
    # bar 38 dips to VWAP (well below price) and closes red; bar 39 reclaims with a bullish close
    dip = (px, px + 0.02, px - 0.9, px - 0.3, 1000)
    reclaim = (px - 0.3, px + 0.4, px - 0.35, px + 0.3, 1600)
    ctx = _ctx(_path(day, up + [dip, reclaim]))
    sv = vwap_pullback_long(ctx)
    assert sv and sv.strategy == "vwap_pullback_long" and sv.stop < sv.entry < sv.target
    flat_day = _ctx(_path(day, [(100, 100.1, 99.9, 100, 1000)] * 40))
    assert vwap_pullback_long(flat_day) is None, "not up on the day -> no signal"
    print(f"✓ {sv.reason}")

    print("TEST 6: null model is deterministic per seed and differs across seeds")
    bars = _path(day, [(100, 100.2, 99.8, 100, 1000)] * 300)
    def hits(seed):
        out = []
        c = SymbolContext("TST", day, None, [])
        cum = [0.0, 0.0]
        for b in bars:
            c.bars.append(b); update_vwaps(c.vwaps, cum, b)
            if random_long(c, {"seed": seed, "p_per_bar": 0.05}):
                out.append(len(c.bars))
        return out
    assert hits(1) == hits(1) and hits(1) != hits(2) and 0 < len(hits(1)) < 60
    print(f"✓ seed 1 fires at {len(hits(1))} bars, reproducibly")

    print("TEST 7: first_signal respects list order and only evaluates enabled strategies")
    ctx7 = _ctx(_path(day, rows))
    assert first_signal(ctx7, ["gap_and_go_long", "orb_long"]).strategy == "orb_long"
    assert first_signal(ctx7, ["gap_and_go_long"]) is None
    print("✓ ok")
    print("\nAll daytrade_strategies tests passed.")
