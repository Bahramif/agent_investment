"""
simulate.py
===========
Walk-forward historical simulation: "if I'd followed this system's rules
starting on DATE with $X, what would I have gotten?"

It runs the SAME rules as live portfolios: snapshots from
data_ingestion.snapshot_from_prices, scores from analyst_reasoning.reason_desk,
selection, sizing and limits from mandate_allocator.allocate, sector tilts from
the real strategist, and stops, re-entry blocks, targeted exits and the
no-small-trades rule from rulebook.py. Change a rule and this changes with it.

STATED PLAINLY, what it can't replay, so nothing overstates itself:
  - THE LLM. Its read and news verdicts aren't available historically, so its
    weight is shared across the other factors and news is assumed clean. That
    makes this a test of the deterministic rules, not of the LLM.
  - THE EARNINGS BLOCK and the news watch's news-driven sales, for the same reason.
  - POINT-IN-TIME FUNDAMENTALS. Quality, EPS and industry are today's values
    applied to every date: a mild lookahead.
  - SURVIVORSHIP. The stock list is today's: companies that failed or were removed
    are missing, which flatters results.
  - TAXES. Trading costs are charged (rulebook cost_bps per side); taxes aren't.
Prices are only ever used up to the date of each decision. None of this is a
promise of what would really have happened.
"""

from __future__ import annotations

import bisect
import dataclasses
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import rulebook
from analyst_reasoning import reason_desk
from data_ingestion import build_macro_from_snapshots, snapshot_from_prices
from macro_reasoning import reason_strategic_view, reason_tactical_view
from mandate_allocator import allocate
from strategist import ThemeStrategy, TiltCaps, combine

LOOKBACK_CALENDAR_DAYS = 400


def rulebook_lookback_days() -> int:
    """How much history to fetch before the first decision."""
    return LOOKBACK_CALENDAR_DAYS


@dataclass
class SimulationResult:
    dates: List[str]
    strategy_values: List[float]
    benchmark_values: List[float]          # equal-weight buy-and-hold, no rebalancing
    rebalance_dates: List[str]
    rebalance_weights: List[Dict[str, float]]   # what was targeted at each rebalance
    rebalance_actions: List[Dict[str, Tuple[str, float]]]   # symbol -> (BUY/SELL, $) at that rebalance
    rebalance_tilts: List[Dict[str, float]]     # sector -> net_tilt applied at that rebalance
    final_value: float
    benchmark_final_value: float
    total_return: float
    benchmark_return: float
    spy_values: List[float] = field(default_factory=list)
    spy_final_value: float = 0.0
    spy_return: float = 0.0
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Optional[float]] = field(default_factory=dict)   # rulebook.performance_stats
    total_costs: float = 0.0
    annual_turnover: float = 0.0
    n_trades: int = 0
    n_stops: int = 0


def fetch_price_series(
    symbols: List[str], start: str, end: str, api_key: str, secret_key: str,
    timeframe: str = "day",
) -> Dict[str, List[Tuple[str, float]]]:
    """symbol -> [(timestamp_str, close), ...] ascending. Fetches once for the
    whole window. timeframe: "day" ("%Y-%m-%d" stamps) or "hour" ("%Y-%m-%d %H:%M"),
    which web_ui.py's portfolio_detail() uses for its chart.

    BUG FIXES carried here, each confirmed live: a client timeout and retry on
    rate limits; adjustment=ALL (a real split otherwise looks like a crash); `end`
    is exclusive at midnight UTC, so +1 day makes it inclusive; and `end` is capped
    at 20 minutes ago, because the free data plan rejects recent SIP data."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment
    from alpaca_clients import apply_default_timeout, retry_on_rate_limit

    client = apply_default_timeout(StockHistoricalDataClient(api_key, secret_key))
    start_dt = datetime.fromisoformat(start)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    if timeframe == "day":
        end_dt += timedelta(days=1)
    now_safe = datetime.now(timezone.utc) - timedelta(minutes=20)
    if end_dt > now_safe:
        end_dt = now_safe
    tf = TimeFrame.Day if timeframe == "day" else TimeFrame(1, TimeFrameUnit.Hour)
    request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf,
                               start=start_dt, end=end_dt, adjustment=Adjustment.ALL)
    bars = retry_on_rate_limit(client.get_stock_bars, request, max_attempts=5, base_delay=2.0)
    fmt = "%Y-%m-%d" if timeframe == "day" else "%Y-%m-%d %H:%M"
    out: Dict[str, List[Tuple[str, float]]] = {}
    for sym in symbols:
        series = bars[sym] if sym in bars.data else []
        out[sym] = [(b.timestamp.strftime(fmt), float(b.close)) for b in series]
    return out


class _Series:
    """One symbol's closes, addressable by date without lookahead."""
    def __init__(self, rows: List[Tuple[str, float]]):
        self.dates = [d for d, _ in rows]
        self.prices = [p for _, p in rows]

    def upto(self, day: str) -> List[float]:
        return self.prices[:bisect.bisect_right(self.dates, day)]

    def on(self, day: str) -> Optional[float]:
        i = bisect.bisect_right(self.dates, day)
        return self.prices[i - 1] if i else None


def _fundamentals(symbols: List[str], fallback: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Today's fundamentals, once per symbol (see the module docstring's caveat)."""
    from concurrent.futures import ThreadPoolExecutor
    from fundamentals import get_factor_inputs, get_quality_score

    def one(s: str) -> Dict[str, Any]:
        return {**get_factor_inputs(s), "quality": get_quality_score(s, fallback=fallback.get(s, 0.5))}
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(symbols, pool.map(one, symbols)))


def run_simulation(
    universe: List[Tuple[str, str, float]],   # (symbol, path, quality_score)
    sectors: List[str],                        # bare sector keys, e.g. ["tech_ai", "energy"]
    price_series: Dict[str, List[Tuple[str, float]]],   # must reach a year BEFORE `start`
    initial_capital: float,
    all_sectors: Optional[List[str]] = None,   # full known sector list, to explicitly lock out the rest
    rebalance_every_days: int = 21,            # in trading days
    max_per_name: float = 0.30,
    top_n: int = 12,
    start: Optional[str] = None,               # first day traded; default: as soon as a year of history exists
    end: Optional[str] = None,
    cfg: Optional[rulebook.RulebookConfig] = None,
    fundamentals: Optional[Dict[str, Dict[str, Any]]] = None,   # {symbol: {quality, eps, industry, avg_volume}}; default: fetched
    scorer: Optional[Callable[[List[Any]], List[Any]]] = None,  # replace the score (the backtest's null model)
) -> SimulationResult:
    cfg = cfg or rulebook.default_config()
    sector_by_symbol = {sym: rulebook.sector_of(path) for sym, path, _ in universe if rulebook.sector_of(path) in sectors}
    series = {s: _Series(price_series[s]) for s in sector_by_symbol if price_series.get(s)}
    warnings: List[str] = []
    missing = set(sector_by_symbol) - set(series)
    if missing:
        warnings.append(f"Skipped (no price history): {', '.join(sorted(missing))}")
    if not series:
        raise ValueError(f"No usable price history for sector(s) {sectors} in the given date range.")
    spy = _Series(price_series["SPY"]) if price_series.get("SPY") else None
    if spy is None:
        warnings.append("SPY price history unavailable: beta unknown and the S&P 500 comparison omitted.")

    if fundamentals is None:
        fundamentals = _fundamentals(sorted(series), {s: q for s, _, q in universe})
    all_dates = sorted({d for s in series.values() for d in s.dates} | set(spy.dates if spy else []))
    first_ready = all_dates[min(cfg.min_history_bars, len(all_dates) - 1)]
    begin = max(start or first_ready, all_dates[0])
    if start and start < first_ready:
        warnings.append(f"Under a year of history before {first_ready}: until then few stocks are eligible.")
    dates = [d for d in all_dates if begin <= d and (not end or d <= end)]
    if len(dates) < 2:
        raise ValueError("No trading days in the given date range.")

    excluded = [s for s in (all_sectors or sectors) if s not in sectors]
    caps = TiltCaps()
    shares: Dict[str, float] = {}
    entries: Dict[str, Tuple[float, str]] = {}       # symbol -> (entry price, entry date)
    sold_on: Dict[str, str] = {}
    cash = float(initial_capital)
    values: List[float] = []
    rebalance_dates: List[str] = []
    rebalance_weights: List[Dict[str, float]] = []
    rebalance_tilts: List[Dict[str, float]] = []
    rebalance_actions: List[Dict[str, Tuple[str, float]]] = []
    total_costs, traded, n_trades, n_stops = 0.0, 0.0, 0, 0
    bps = cfg.cost_bps / 10_000.0

    def value_on(day: str) -> float:
        return cash + sum(n * (series[s].on(day) or 0.0) for s, n in shares.items())

    def blocked_on(day: str) -> Set[str]:
        d = date.fromisoformat(day)
        return {s for s, on in sold_on.items() if s not in shares
                and (d - date.fromisoformat(on)).days < cfg.reentry_block_days}

    def snapshots_on(day: str) -> List[Any]:
        benchmark = spy.upto(day) if spy else None
        snaps = []
        for s, ser in series.items():
            prices = ser.upto(day)
            if len(prices) < 2:
                continue
            f = fundamentals.get(s) or {}
            snaps.append(snapshot_from_prices(
                s, f"equities.{sector_by_symbol[s]}", prices, f.get("quality", 0.5), eps=f.get("eps"),
                industry=f.get("industry") or "", benchmark=benchmark, avg_volume=f.get("avg_volume"),
                news_status="clean"))
        return snaps

    def book_on(day: str, n: int, held: Set[str], blocked: Set[str]) -> Tuple[Any, Dict[str, float], Dict[str, float]]:
        snaps = snapshots_on(day)
        signals = scorer(snaps) if scorer else reason_desk(snaps, cfg)
        tilts = {}
        if len(sectors) > 1:
            for sec in sectors:
                peers = [x for x in snaps if rulebook.sector_of(x.path) == sec]
                if peers:
                    macro = build_macro_from_snapshots(sec, peers)
                    tilts[sec] = combine(ThemeStrategy(sec, reason_strategic_view(macro), reason_tactical_view(macro)), caps)
        mandate: Dict[str, Any] = {
            "total": 1.0,
            "default_limits": {"max_per_name": max_per_name, "min_names": 1, "top_n": n, "as_of": day,
                               "held": sorted(held), "blocked": sorted(blocked), "rulebook": rulebook.to_dict(cfg)},
            "buckets": {"equities": {"set_by": "user", "weight": 1.0}},
        }
        if excluded:
            mandate["buckets"]["equities"]["children"] = {s: {"set_by": "user", "weight": 0.0} for s in excluded}
        book = allocate(mandate, signals, theme_tilts=tilts or None)
        return book, {n_: w for n_, w in book.weights.items() if n_ != rulebook.CASH}, {k: t.net_tilt for k, t in tilts.items()}

    def trade(day: str, sym: str, new_shares: float) -> Tuple[str, float]:
        nonlocal cash, total_costs, traded, n_trades
        px = series[sym].on(day) or 0.0
        cur = shares.get(sym, 0.0)
        notional = abs(new_shares - cur) * px
        cost = notional * bps
        cash -= (new_shares - cur) * px + cost
        total_costs += cost
        traded += notional
        n_trades += 1
        if new_shares > cur:
            old_entry = entries.get(sym)
            if old_entry and cur > 0:
                old_cost = cur * old_entry[0]
                entries[sym] = ((old_cost + notional) / new_shares,
                                rulebook.blended_entry_date(old_entry[1], old_cost, day, notional))
            else:
                entries[sym] = (px, day)
        if new_shares <= 1e-9:
            shares.pop(sym, None)
            entries.pop(sym, None)
            sold_on[sym] = day
        else:
            shares[sym] = new_shares
        return ("BUY" if new_shares > cur else "SELL"), notional

    def stops_on(day: str) -> List[str]:
        out = []
        for sym in sorted(shares):
            entry_px, entry_day = entries[sym]
            px = series[sym].on(day)
            peers = [ser for s, ser in series.items() if s != sym and sector_by_symbol[s] == sector_by_symbol[sym]]
            rets = []
            for ser in peers:
                b, now = ser.on(entry_day), ser.on(day)
                if b and now:
                    rets.append(now / b - 1.0)
            if not rets or not px:
                continue
            vol = rulebook.annual_window_vol(series[sym].upto(day)[:-1])   # to yesterday: a crash can't move its own stop
            if rulebook.stop_breached(px, entry_px, statistics.fmean(rets), vol, cfg):
                out.append(sym)
        return out

    for i, day in enumerate(dates):
        if i % rebalance_every_days == 0:
            book, weights, tilts = book_on(day, top_n, set(shares), blocked_on(day))
            total = value_on(day)
            actions: Dict[str, Tuple[str, float]] = {}
            orders = []
            for sym in set(shares) | set(weights):
                px = series[sym].on(day)
                if not px:
                    continue
                tgt = total * weights.get(sym, 0.0) / px
                cur = shares.get(sym, 0.0)
                if abs(tgt - cur) < 1e-9 or rulebook.below_min_trade((tgt - cur) * px, total, tgt, cfg):
                    continue
                orders.append((tgt > cur, sym, tgt))
            for _, sym, tgt in sorted(orders):              # sells first
                actions[sym] = trade(day, sym, tgt)
            rebalance_dates.append(day)
            rebalance_weights.append(dict(weights))
            rebalance_tilts.append(tilts)
            rebalance_actions.append(actions)
        else:
            hit = stops_on(day)
            if hit:                                         # targeted: only these are sold, slots filled
                n_stops += len(hit)
                total = value_on(day)
                current = {s: n * (series[s].on(day) or 0.0) / total for s, n in shares.items()}
                held_after = set(shares) - set(hit)
                book, weights, _ = book_on(day, top_n, held_after, blocked_on(day) | set(hit))
                fresh = sorted((s for s in weights if s not in shares), key=lambda s: -weights[s])
                info = book.notes
                industry = {s: (fundamentals.get(s) or {}).get("industry") or "" for s in set(current) | set(fresh)}
                target = rulebook.targeted_weights(current, hit, fresh, industry,
                                                   {s: (info.get(s) or {}).get("beta") for s in fresh}, cfg)
                actions = {}
                for sym in hit:
                    actions[sym] = trade(day, sym, 0.0)
                for sym in target:
                    if sym not in held_after and series[sym].on(day):
                        actions[sym] = trade(day, sym, total * target[sym] / series[sym].on(day))
                rebalance_dates.append(day)
                rebalance_weights.append(dict(target))
                rebalance_tilts.append({})
                rebalance_actions.append(actions)
        values.append(value_on(day))

    # equal-weight buy-and-hold of whatever could be bought on day one
    first = dates[0]
    tradable = [s for s, ser in series.items() if ser.on(first)]
    bench_shares = {s: initial_capital / len(tradable) / series[s].on(first) for s in tradable}
    benchmark_values = [sum(n * (series[s].on(d) or 0.0) for s, n in bench_shares.items()) for d in dates]
    spy_values: List[float] = []
    if spy and spy.on(first):
        spy_values = [initial_capital * (spy.on(d) or spy.on(first)) / spy.on(first) for d in dates]

    final = values[-1]
    years = max(len(dates) - 1, 1) / rulebook.TRADING_DAYS_PER_YEAR
    avg_value = statistics.fmean(values)
    return SimulationResult(
        dates=dates, strategy_values=values, benchmark_values=benchmark_values,
        rebalance_dates=rebalance_dates, rebalance_weights=rebalance_weights,
        rebalance_actions=rebalance_actions, rebalance_tilts=rebalance_tilts,
        final_value=final, benchmark_final_value=benchmark_values[-1],
        total_return=final / initial_capital - 1.0 if initial_capital else 0.0,
        benchmark_return=benchmark_values[-1] / initial_capital - 1.0 if initial_capital else 0.0,
        spy_values=spy_values, spy_final_value=spy_values[-1] if spy_values else 0.0,
        spy_return=(spy_values[-1] / initial_capital - 1.0) if spy_values and initial_capital else 0.0,
        warnings=warnings, stats=rulebook.performance_stats(values), total_costs=total_costs,
        annual_turnover=(traded / 2.0 / avg_value / years) if avg_value > 0 else 0.0,
        n_trades=n_trades, n_stops=n_stops,
    )
