"""
intraday_data.py
=================
Minute-bar data for the day-trading capability: the Bar type, Eastern-time
session helpers, an Alpaca-backed historical/live client, and a synthetic
bar generator for tests.

WHY POLLING, NOT A WEBSOCKET, for live bars: the decision cadence is one
1-minute bar, so a REST poll every ~15s adds no meaningful latency, and it
is self-healing — each poll re-requests everything after the last bar seen,
so a dropped connection, a Cloud Run restart or a laggy bar can never leave
a permanent hole. Alpaca's free plan also allows only ONE websocket
connection, which a crash-and-restart could leave dangling. If sub-minute
decisions are ever needed, a StockDataStream-backed BarFeed can implement
the same `poll()` contract without touching the engine.

DATA FEED, stated plainly: minute bars come from IEX (the only live feed on
Alpaca's free plan), which is a single exchange — a small slice of
consolidated volume. Consequences baked into this design: (1) strategies
only ever use RELATIVE volume (this bar vs. earlier bars the same day),
never absolute volume thresholds; (2) backtest and live use the SAME feed,
so their numbers are comparable; (3) the universe liquidity filter uses
consolidated daily volume from daily bars, a different field entirely.
IEX also skips minutes with no trades, so a symbol can legitimately have no
bar for a minute — nothing here assumes bars are contiguous.

HONEST LIMITATION, same shape as alpaca_clients.py: the alpaca-py request
construction below matches the installed SDK's models (checked), but the
literal network calls are untested here — no network / credentials in the
sandbox. Everything else (session math, feed filtering, generator) is
tested against fakes shaped like the SDK's BarSet.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from zoneinfo import ZoneInfo

from daytrade_config import SessionRules

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
BAR = timedelta(minutes=1)


# -----------------------------------------------------------------------------
# Bar + session math. All datetimes are timezone-aware; `ts` is the bar's START.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Bar:
    symbol: str
    ts: datetime          # start of the 1-minute interval, tz-aware
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None   # the feed's own bar VWAP when provided

    @property
    def end(self) -> datetime:
        return self.ts + BAR

    @property
    def typical(self) -> float:
        return self.vwap if self.vwap else (self.high + self.low + self.close) / 3.0


def to_et(ts: datetime) -> datetime:
    return ts.astimezone(ET)


def session_date(ts: datetime) -> date:
    return to_et(ts).date()


def et_datetime(day: date, t: time) -> datetime:
    return datetime.combine(day, t, tzinfo=ET)


def in_regular_session(ts: datetime, close: time = MARKET_CLOSE) -> bool:
    t = to_et(ts).time()
    return MARKET_OPEN <= t < close


def trading_minutes(day: date, close: time = MARKET_CLOSE) -> List[datetime]:
    out, cur, end = [], et_datetime(day, MARKET_OPEN), et_datetime(day, close)
    while cur < end:
        out.append(cur.astimezone(UTC))
        cur += BAR
    return out


def minutes_since_open(ts: datetime) -> int:
    et = to_et(ts)
    return int((et - et_datetime(et.date(), MARKET_OPEN)).total_seconds() // 60)


def effective_session(rules: SessionRules, close: time = MARKET_CLOSE) -> Tuple[time, time]:
    """(last_entry, flatten) for a session that closes at `close`. On a
    normal day this is just the configured values; on an early-close day
    (1:00pm) the configured 3:30/3:45 would fall AFTER the close, so they're
    pulled back to 30/15 minutes before it. Never later than configured."""
    close_dt = datetime.combine(date(2000, 1, 3), close)
    last_entry = min(datetime.combine(date(2000, 1, 3), rules.last_entry_et), close_dt - timedelta(minutes=30))
    flatten = min(datetime.combine(date(2000, 1, 3), rules.flatten_et), close_dt - timedelta(minutes=15))
    return last_entry.time(), flatten.time()


def group_by_day(bars: Sequence[Bar]) -> Dict[date, List[Bar]]:
    """Regular-session bars only, per Eastern trading day, sorted by time."""
    out: Dict[date, List[Bar]] = {}
    for b in sorted(bars, key=lambda x: x.ts):
        if in_regular_session(b.ts):
            out.setdefault(session_date(b.ts), []).append(b)
    return out


# -----------------------------------------------------------------------------
# Contracts — what the engine/runner need; Alpaca is one implementation.
# -----------------------------------------------------------------------------
class IntradayDataClient(Protocol):
    def get_minute_bars(self, symbols: Sequence[str], start: datetime, end: datetime) -> Dict[str, List[Bar]]: ...
    def get_daily_bars(self, symbols: Sequence[str], start: date, end: date) -> Dict[str, List[Bar]]: ...


class MarketClock(Protocol):
    def now(self) -> datetime: ...
    def is_open(self) -> bool: ...
    def session_close_et(self) -> Optional[time]: ...   # today's close, None if not a trading day


class BarFeed(Protocol):
    def poll(self, since: Dict[str, Optional[datetime]], now: datetime) -> List[Bar]: ...


def _bar_from_alpaca(symbol: str, b: Any) -> Bar:
    ts = getattr(b, "timestamp", None) or getattr(b, "ts")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    vwap = getattr(b, "vwap", None)
    return Bar(symbol=symbol, ts=ts.astimezone(UTC), open=float(b.open), high=float(b.high),
               low=float(b.low), close=float(b.close), volume=float(b.volume),
               vwap=float(vwap) if vwap else None)


def _barset_to_dict(barset: Any, symbols: Sequence[str]) -> Dict[str, List[Bar]]:
    """alpaca-py's BarSet keeps per-symbol lists under `.data`; a symbol
    with no bars is simply absent, and indexing it raises KeyError."""
    data = getattr(barset, "data", None)
    if data is None:
        data = barset
    out: Dict[str, List[Bar]] = {}
    for sym in symbols:
        raw = data.get(sym) if hasattr(data, "get") else None
        out[sym] = sorted((_bar_from_alpaca(sym, b) for b in (raw or [])), key=lambda x: x.ts)
    return out


class AlpacaIntradayDataClient:
    """StockHistoricalDataClient wrapper. Timeout adapter + transient-error
    retry come from alpaca_clients — the SAME protections every other
    Alpaca call in this project already has (see apply_default_timeout's
    docstring for why a hung request with no timeout is the real hazard)."""

    def __init__(self, stock_client: Any, feed: str = "iex"):
        from alpaca_clients import apply_default_timeout
        self._client = apply_default_timeout(stock_client)
        self._feed = feed

    def get_minute_bars(self, symbols: Sequence[str], start: datetime, end: datetime) -> Dict[str, List[Bar]]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca_clients import retry_on_rate_limit
        req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame.Minute,
                               start=start, end=end, adjustment=Adjustment.ALL,
                               feed=DataFeed(self._feed))
        return _barset_to_dict(retry_on_rate_limit(self._client.get_stock_bars, req), symbols)

    def get_daily_bars(self, symbols: Sequence[str], start: date, end: date) -> Dict[str, List[Bar]]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment
        from alpaca_clients import retry_on_rate_limit
        req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame.Day,
                               start=datetime.combine(start, time(0), tzinfo=UTC),
                               end=datetime.combine(end, time(23, 59), tzinfo=UTC),
                               adjustment=Adjustment.ALL)
        return _barset_to_dict(retry_on_rate_limit(self._client.get_stock_bars, req), symbols)


class PollingBarFeed:
    """Live bars by polling. `poll()` returns only COMPLETE bars that are
    strictly newer than what the caller has already seen for that symbol,
    so it is safe to call as often as you like and safe to call again after
    a restart (pass since={} to rebuild the whole day)."""

    def __init__(self, client: IntradayDataClient, symbols: Sequence[str], settle_seconds: int = 2):
        self._client = client
        self._symbols = list(symbols)
        self._settle = timedelta(seconds=settle_seconds)

    def poll(self, since: Dict[str, Optional[datetime]], now: datetime) -> List[Bar]:
        day = session_date(now)
        open_dt = et_datetime(day, MARKET_OPEN).astimezone(UTC)
        seen = [t for t in since.values() if t]
        start = min(seen) if len(seen) == len(self._symbols) and seen else open_dt
        fetched = self._client.get_minute_bars(self._symbols, start, now)
        out: List[Bar] = []
        for sym, bars in fetched.items():
            last = since.get(sym)
            for b in bars:
                if not in_regular_session(b.ts) or session_date(b.ts) != day:
                    continue
                if b.end > now - self._settle:        # still forming — never trade on a partial bar
                    continue
                if last is not None and b.ts <= last:
                    continue
                out.append(b)
        return sorted(out, key=lambda b: (b.ts, b.symbol))


class AlpacaClock:
    """Wraps TradingClient.get_clock()/get_calendar() — the source of truth
    for holidays and early closes, which no hardcoded calendar here tries
    to replicate."""

    def __init__(self, trading_client: Any):
        from alpaca_clients import apply_default_timeout
        self._tc = apply_default_timeout(trading_client)

    def now(self) -> datetime:
        return self._tc.get_clock().timestamp.astimezone(UTC)

    def is_open(self) -> bool:
        return bool(self._tc.get_clock().is_open)

    def session_close_et(self) -> Optional[time]:
        clock = self._tc.get_clock()
        nxt = clock.next_close
        if nxt is None:
            return None
        nxt_et = nxt.astimezone(ET)
        return nxt_et.time() if nxt_et.date() == clock.timestamp.astimezone(ET).date() else None


# -----------------------------------------------------------------------------
# Test doubles + a synthetic-day generator (deterministic).
# -----------------------------------------------------------------------------
def synthetic_day(symbol: str, day: date, seed: int, start_price: float = 100.0,
                  drift_bps_per_min: float = 0.0, vol_bps_per_min: float = 8.0,
                  gap: float = 0.0, base_volume: float = 20_000.0,
                  close: time = MARKET_CLOSE, prev_close: Optional[float] = None) -> List[Bar]:
    """Geometric random walk, 1-minute bars, U-shaped volume (heavy at the
    open and close) with volume spikes on large moves. `gap` opens the day
    that fraction above `start_price`. Deterministic per (symbol, day, seed)."""
    rng = random.Random(f"{symbol}-{day.isoformat()}-{seed}")
    minutes = trading_minutes(day, close)
    n = len(minutes)
    price = start_price * (1.0 + gap)
    bars: List[Bar] = []
    for i, ts in enumerate(minutes):
        ret = rng.gauss(drift_bps_per_min / 1e4, vol_bps_per_min / 1e4)
        o = price
        c = max(0.5, o * (1.0 + ret))
        wiggle = abs(rng.gauss(0, vol_bps_per_min / 1e4)) * o
        h = max(o, c) + wiggle * 0.5
        l = min(o, c) - wiggle * 0.5
        u_shape = 1.0 + 2.0 * math.exp(-i / 20.0) + 1.2 * math.exp(-(n - i) / 25.0)
        spike = 1.0 + 8.0 * abs(ret) / (vol_bps_per_min / 1e4)
        vol = max(100.0, base_volume * u_shape * spike * rng.uniform(0.6, 1.4) / 3.0)
        bars.append(Bar(symbol, ts, round(o, 4), round(h, 4), round(max(l, 0.01), 4), round(c, 4), round(vol)))
        price = c
    return bars


class FakeIntradayDataClient:
    """bars[symbol] = list[Bar] across any number of days (minute bars);
    daily[symbol] = list[Bar] daily bars."""

    def __init__(self, bars: Dict[str, List[Bar]], daily: Optional[Dict[str, List[Bar]]] = None):
        self.bars, self.daily = bars, daily or {}
        self.minute_calls: List[Tuple[Tuple[str, ...], datetime, datetime]] = []

    def get_minute_bars(self, symbols, start, end):
        self.minute_calls.append((tuple(symbols), start, end))
        return {s: [b for b in self.bars.get(s, []) if start <= b.ts <= end] for s in symbols}

    def get_daily_bars(self, symbols, start, end):
        return {s: [b for b in self.daily.get(s, []) if start <= to_et(b.ts).date() <= end] for s in symbols}


class FakeClock:
    def __init__(self, now: datetime, close: Optional[time] = MARKET_CLOSE, is_open: bool = True):
        self._now, self._close, self._open = now, close, is_open

    def now(self) -> datetime:
        return self._now

    def advance(self, **kw) -> None:
        self._now += timedelta(**kw)

    def is_open(self) -> bool:
        return self._open

    def session_close_et(self) -> Optional[time]:
        return self._close


if __name__ == "__main__":
    print("TEST 1: session math is correct across the DST boundary")
    winter = datetime(2026, 1, 15, 14, 30, tzinfo=UTC)     # EST: 09:30 ET
    summer = datetime(2026, 7, 15, 13, 30, tzinfo=UTC)     # EDT: 09:30 ET
    assert minutes_since_open(winter) == 0 and minutes_since_open(summer) == 0
    assert in_regular_session(winter) and in_regular_session(summer)
    assert not in_regular_session(winter - timedelta(minutes=1))
    assert not in_regular_session(datetime(2026, 1, 15, 21, 0, tzinfo=UTC))   # 16:00 ET — session over
    assert in_regular_session(datetime(2026, 1, 15, 20, 59, tzinfo=UTC))      # 15:59 ET — last bar
    print("✓ 09:30 ET is 14:30Z in winter and 13:30Z in summer; 15:59 in, 16:00 out")

    print("TEST 2: trading_minutes has 390 bars; early close has 210")
    d = date(2026, 3, 4)
    assert len(trading_minutes(d)) == 390
    assert len(trading_minutes(d, close=time(13, 0))) == 210
    print("✓ 390 normal, 210 early-close")

    print("TEST 3: effective_session pulls entries/flatten back on early-close days")
    rules = SessionRules()
    assert effective_session(rules) == (time(15, 30), time(15, 45))
    assert effective_session(rules, close=time(13, 0)) == (time(12, 30), time(12, 45))
    print("✓ normal (15:30, 15:45); early close (12:30, 12:45)")

    print("TEST 4: synthetic_day is deterministic, well-formed, regular-session only")
    a = synthetic_day("AAA", d, seed=1)
    b = synthetic_day("AAA", d, seed=1)
    c = synthetic_day("AAA", d, seed=2)
    assert a == b and a != c and len(a) == 390
    assert all(x.low <= min(x.open, x.close) and x.high >= max(x.open, x.close) and x.volume > 0 for x in a)
    assert all(in_regular_session(x.ts) for x in a)
    assert a[0].volume > a[195].volume            # U-shaped volume: heavy at the open
    print("✓ deterministic per seed; OHLC consistent; opening volume > midday")

    print("TEST 5: PollingBarFeed returns only COMPLETE, NEW, in-session bars")
    day_bars = synthetic_day("AAA", d, seed=1)
    client = FakeIntradayDataClient({"AAA": day_bars})
    feed = PollingBarFeed(client, ["AAA"])
    now = day_bars[10].ts + timedelta(minutes=1, seconds=1)   # bar 10 ended 1s ago (not yet settled)
    first = feed.poll({}, now)
    assert [x.ts for x in first] == [x.ts for x in day_bars[:10]], "bar 10 is <2s old — must be withheld"
    later = feed.poll({"AAA": first[-1].ts}, now + timedelta(seconds=30))
    assert [x.ts for x in later] == [day_bars[10].ts]
    assert feed.poll({"AAA": later[-1].ts}, now + timedelta(seconds=30)) == []
    print("✓ partial bar withheld, then delivered once; re-poll returns nothing (no duplicates)")

    print("TEST 6: after a restart (since={}) the feed rebuilds the whole day, ignoring prior days / after-hours")
    prev = synthetic_day("AAA", date(2026, 3, 3), seed=9)
    ah = Bar("AAA", et_datetime(d, time(8, 0)).astimezone(UTC), 1, 1, 1, 1, 1)
    client2 = FakeIntradayDataClient({"AAA": prev + [ah] + day_bars})
    rebuilt = PollingBarFeed(client2, ["AAA"]).poll({}, day_bars[-1].ts + timedelta(minutes=2))
    assert len(rebuilt) == 390 and all(session_date(x.ts) == d for x in rebuilt)
    print("✓ 390 bars: yesterday and the 08:00 pre-market print excluded")

    print("TEST 7: _barset_to_dict tolerates a symbol with NO bars (IEX skips quiet symbols)")
    from types import SimpleNamespace as NS
    raw = NS(data={"AAA": [NS(timestamp=day_bars[0].ts, open=1, high=2, low=.5, close=1.5, volume=10, vwap=None)]})
    out = _barset_to_dict(raw, ["AAA", "ZZZ"])
    assert len(out["AAA"]) == 1 and out["ZZZ"] == [] and out["AAA"][0].vwap is None
    print("✓ missing symbol -> [] not KeyError; SDK-shaped objects convert")

    print("TEST 8: request construction matches the installed alpaca-py models")
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment, DataFeed
        req = StockBarsRequest(symbol_or_symbols=["AAA", "BBB"], timeframe=TimeFrame.Minute,
                               start=day_bars[0].ts, end=day_bars[-1].ts,
                               adjustment=Adjustment.ALL, feed=DataFeed("iex"))
        assert req.feed == DataFeed.IEX
        print("✓ StockBarsRequest(minute, adjustment=ALL, feed=IEX) validates")
    except ImportError:
        print("- skipped (alpaca-py not installed)")
    print("\nAll intraday_data tests passed.")
