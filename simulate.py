"""
simulate.py
===========
Walk-forward historical simulation: "if I'd followed this system's
recommendations starting on DATE with $X, what would I have gotten?"

DESIGN CHOICES, stated plainly rather than left implicit:
  - Uses the HEURISTIC analyst scorer (analyst_reasoning.reason_desk), not
    the Claude-backed one. Calling a real LLM at every rebalance point over
    a multi-month window would be slow and (at scale) costly, and this
    project's Claude Vertex quota is currently 0 anyway (see vertex_deploy.py's
    notes). The simulation exercises the exact same mandate_allocator
    water-fill/risk logic the live agent uses — only the SCORING step
    differs from what a quota-enabled live deployment would do.
  - NO LOOKAHEAD: at each rebalance point, momentum is computed ONLY from
    price history up to and including that date. Prices used to mark the
    portfolio to market BETWEEN rebalances are real, later, historical
    closes — that's what a backtest is; it isn't a lookahead violation.
  - No historical news feed is used — news_sentiment is held at 0.0 for
    every simulated period. Building a historical sentiment feed is out of
    scope here; this measurably weakens the scorer relative to the live
    version, which does have real news sentiment.
  - quality_score uses fundamentals.get_quality_score() — the SAME real
    yfinance-derived fundamentals (margins, ROE, leverage, revenue growth)
    the live path uses, not universe.py's flat placeholder. HONEST
    LIMITATION: this is TODAY's fundamentals applied to every historical
    rebalance date, not true point-in-time history (a free source doesn't
    give historical fundamentals) — a mild lookahead on this one input,
    reasonable since quality-factor fundamentals move slowly, but worth
    knowing rather than assuming this is a fully rigorous backtest input.
  - Transaction costs, slippage, bid/ask spread, and taxes are NOT modeled.
    Real returns following this system live would be lower than shown here.
  - Survivorship / universe drift isn't modeled — the same fixed ticker list
    (today's UniverseConfig) is used for the whole backtest window,
    regardless of what would actually have been known/tradable at each
    historical date.
  - Rebalancing only happens on rebalance dates; between them, the
    portfolio just rides real subsequent price moves (buy-and-hold within
    each period) — it is not continuously re-weighted.
None of this is a promise of what would really have happened. It's the
best a same-logic-as-the-live-system walk-forward test can show without a
historical sentiment feed or live LLM calls at every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from analyst_reasoning import InstrumentSnapshot, reason_desk
from mandate_allocator import allocate
from fundamentals import get_quality_score


@dataclass
class SimulationResult:
    dates: List[str]
    strategy_values: List[float]
    benchmark_values: List[float]          # equal-weight buy-and-hold, no rebalancing
    rebalance_dates: List[str]
    rebalance_weights: List[Dict[str, float]]   # what was held after each rebalance
    rebalance_actions: List[Dict[str, Tuple[str, float]]]   # symbol -> (BUY/SELL, $) vs the PREVIOUS rebalance
    rebalance_tilts: List[Dict[str, float]]     # sector -> net_tilt applied at that rebalance
    final_value: float
    benchmark_final_value: float
    total_return: float
    benchmark_return: float
    # SPY (SPDR S&P 500 ETF Trust) buy-and-hold over the same window — the
    # standard real, tradable proxy for "the S&P 500 index" (Alpaca deals
    # in tradable securities, not the raw ^GSPC index itself). Empty/0.0
    # when SPY price history isn't available for this window.
    spy_values: List[float] = field(default_factory=list)
    spy_final_value: float = 0.0
    spy_return: float = 0.0
    warnings: List[str] = field(default_factory=list)


def fetch_price_series(
    symbols: List[str], start: str, end: str, api_key: str, secret_key: str,
    timeframe: str = "day",
) -> Dict[str, List[Tuple[str, float]]]:
    """symbol -> [(timestamp_str, close), ...] ascending. Fetches once for
    the whole window — callers slice locally per rebalance rather than
    making one API call per period.

    timeframe: "day" (default — every existing caller, including this
    module's own multi-month/year backtests, wants one bar per calendar
    day) or "hour" — web_ui.py's portfolio_detail() always uses "hour" so
    the chart reflects the portfolio's actual hour-by-hour path regardless
    of its age. Day bars are timestamped "%Y-%m-%d"
    (unchanged format existing callers rely on for string date comparison);
    hour bars are timestamped "%Y-%m-%d %H:%M" so same-day bars don't
    collide under that same string-comparison logic."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment
    from alpaca_clients import apply_default_timeout, retry_on_rate_limit

    # BUG FIX: this constructed its own client with no timeout (a hung
    # connection would block forever — see alpaca_clients.apply_default_timeout's
    # docstring for the confirmed-live root cause) and no retry on Alpaca's
    # real, frequently-observed "too many requests" error. Same two fixes
    # already applied everywhere else Alpaca is called from this app.
    #
    # adjustment=Adjustment.ALL is ESPECIALLY important here: a backtest
    # spans months/years, making it far more likely to cross a real stock
    # split than a live 21-day momentum window — confirmed live with CRWD's
    # real split (~4:1), which without this shows up as a fake ~75%
    # single-day "crash" that would poison every rebalance's momentum and
    # produce a nonsensical BUY/SELL signal right at the split date.
    client = apply_default_timeout(StockHistoricalDataClient(api_key, secret_key))
    # BUG FIX, confirmed live: `end` is an EXCLUSIVE midnight-UTC boundary —
    # requesting start=end="2026-09-15" (a caller's natural "through today
    # inclusive") returned ZERO bars for every symbol, not "today's bar
    # plus whatever came before". Any caller whose window ends on "today"
    # (portfolio_detail's day-one portfolios, simulate's default end-date)
    # silently lost the most recent day/days. +1 day makes `end` inclusive
    # of the calendar date callers actually pass, matching what "start to
    # end" means to every caller of this function.
    #
    # SECOND BUG FIX, also confirmed live: naively using end-of-day+1 for
    # "today" pushes `end` into the future relative to wall-clock now,
    # which this account's free/IEX-tier data plan flatly REJECTS —
    # {"message":"subscription does not permit querying recent SIP data"}
    # — confirmed the real threshold is between 10 and 16 minutes before
    # now (Alpaca's standard ~15-minute real-time embargo on this tier).
    # Capping `end` at (now - 20 minutes) keeps every request comfortably
    # on the allowed side of that embargo while still returning today's
    # bar once the trading day has enough data for one.
    start_dt = datetime.fromisoformat(start)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    if timeframe == "day":
        end_dt += timedelta(days=1)   # inclusive of the calendar date passed, per the bug fix above
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


def _momentum_at(series: List[Tuple[str, float]], idx: int, lookback: int) -> float:
    """Trailing return ending at series[idx], using only series[:idx+1] —
    the no-lookahead guarantee."""
    start_idx = max(0, idx - lookback)
    if idx <= start_idx or series[start_idx][1] == 0:
        return 0.0
    return (series[idx][1] - series[start_idx][1]) / series[start_idx][1]


def _price_on_or_before(price_series: Dict[str, List[Tuple[str, float]]], sym: str, date: str) -> Optional[float]:
    for d, p in reversed(price_series.get(sym, [])):
        if d <= date:
            return p
    return None


def _compound(
    all_dates: List[str], rebalance_dates: List[str], rebalance_weights: List[Dict[str, float]],
    price_series: Dict[str, List[Tuple[str, float]]], initial_capital: float,
) -> List[float]:
    """Portfolio value on every date in all_dates. base_value is the value
    AS OF the most recent rebalance; between rebalances, the value just
    rides the weighted price ratio since that rebalance's anchor prices."""
    values: List[float] = []
    base_value = initial_capital
    weights: Dict[str, float] = {}
    anchor_prices: Dict[str, float] = {}
    rb_idx = 0

    def ratio_since_anchor(date: str) -> float:
        return sum(
            w * ((_price_on_or_before(price_series, s, date) or anchor_prices[s]) / anchor_prices[s])
            for s, w in weights.items() if anchor_prices.get(s, 0) > 0
        )

    for date in all_dates:
        if rb_idx < len(rebalance_dates) and date == rebalance_dates[rb_idx]:
            if weights and anchor_prices:
                base_value *= ratio_since_anchor(date)   # crystallize gains under the OLD weights
            weights = rebalance_weights[rb_idx]
            anchor_prices = {s: _price_on_or_before(price_series, s, date) or 0.0 for s in weights}
            rb_idx += 1
        values.append(base_value * ratio_since_anchor(date) if weights and anchor_prices else base_value)
    return values


def _sector_tilts(
    sectors: List[str], momentum_by_symbol: Dict[str, float], symbols_by_sector: Dict[str, List[str]],
    scale: float = 2.0, cap: float = 0.40,
) -> Dict[str, "object"]:
    """A simulation-only stand-in for the live strategist's multi-horizon
    HorizonView/decay machinery (which is designed around real calendar
    time and carried-forward memory, not a historical replay loop). Each
    sector's net_tilt is just its constituents' average momentum, scaled
    and clipped — simpler than the live strategist, but it feeds the SAME
    mandate_allocator.allocate()/strategist.theme_budgets() code the live
    agent uses, so the ALLOCATION math itself isn't a simulation-only path."""
    from strategist import ThemeTilt
    tilts = {}
    for s in sectors:
        syms = symbols_by_sector.get(s, [])
        if not syms:
            continue
        avg_mom = sum(momentum_by_symbol.get(sym, 0.0) for sym in syms) / len(syms)
        net = max(-cap, min(cap, avg_mom * scale))
        tilts[s] = ThemeTilt(theme=s, strategic_tilt=net, tactical_tilt=0.0, net_tilt=net)
    return tilts


def run_simulation(
    universe: List[Tuple[str, str, float]],   # (symbol, path, quality_score)
    sectors: List[str],                        # bare sector keys, e.g. ["tech_ai", "energy"]
    price_series: Dict[str, List[Tuple[str, float]]],
    initial_capital: float,
    all_sectors: Optional[List[str]] = None,   # full known sector list, to explicitly lock out the rest
    rebalance_every_days: int = 21,
    max_per_name: float = 0.50,
    top_n: Optional[int] = 12,   # keep only the top-N convictions per theme before water-filling —
                                # see mandate_allocator._top_n() and web_ui.py's TOP_N_PER_SECTOR
    momentum_lookback_days: int = 21,
) -> SimulationResult:
    symbols_by_sector: Dict[str, List[str]] = {}
    quality: Dict[str, float] = {}
    for sym, path, q in universe:
        s = path.split(".")[-1]
        if s in sectors:
            symbols_by_sector.setdefault(s, []).append(sym)
            quality[sym] = q
    symbols = [sym for syms in symbols_by_sector.values() for sym in syms]

    warnings: List[str] = []
    usable = [s for s in symbols if len(price_series.get(s, [])) > momentum_lookback_days]
    missing = set(symbols) - set(usable)
    if missing:
        warnings.append(f"Skipped (insufficient price history): {', '.join(sorted(missing))}")
    symbols = usable
    for s in list(symbols_by_sector):
        symbols_by_sector[s] = [sym for sym in symbols_by_sector[s] if sym in usable]
    if not symbols:
        raise ValueError(f"No usable price history for sector(s) {sectors} in the given date range.")

    # Quality is CONSTANT across every rebalance date in this run (see the
    # module docstring's honest caveat: today's fundamentals, not true
    # point-in-time history) — fetched ONCE per symbol here, in parallel,
    # rather than repeatedly inside the per-date loop below. A cold cache
    # (fundamentals.py's) doing this sequentially across a 50+ name sector
    # would otherwise add real, avoidable seconds to the very first
    # rebalance's computation.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        quality_scores = dict(zip(
            symbols,
            pool.map(lambda s: get_quality_score(s, fallback=quality.get(s, 0.5)), symbols),
        ))

    all_dates = sorted(set(d for s in symbols for d, _ in price_series[s]))

    excluded = [s for s in (all_sectors or sectors) if s not in sectors]
    mandate: Dict[str, Any] = {
        "total": 1.0,
        "default_limits": {"max_per_name": max_per_name, "min_names": 1, "top_n": top_n},
        "buckets": {"equities": {"set_by": "user", "weight": 1.0}},
    }
    if excluded:
        mandate["buckets"]["equities"]["children"] = {s: {"set_by": "user", "weight": 0.0} for s in excluded}

    rebalance_dates: List[str] = []
    rebalance_weights: List[Dict[str, float]] = []
    rebalance_tilts: List[Dict[str, float]] = []
    for i, date in enumerate(all_dates):
        if i % rebalance_every_days != 0:
            continue
        snapshots = []
        momentum_by_symbol = {}
        for s in symbols:
            series = price_series[s]
            idx = max((j for j, (d, _) in enumerate(series) if d <= date), default=None)
            if idx is None:
                continue
            mom = _momentum_at(series, idx, momentum_lookback_days)
            momentum_by_symbol[s] = mom
            sector = next(sec for sec, syms in symbols_by_sector.items() if s in syms)
            snapshots.append(InstrumentSnapshot(
                name=s, path=f"equities.{sector}", momentum_1m=mom,
                news_sentiment=0.0,   # no historical news feed — see module docstring
                # Real fundamentals, precomputed once above (constant across
                # every rebalance date in this run — see module docstring).
                quality_score=quality_scores.get(s, 0.5),
            ))
        if not snapshots:
            continue
        signals = reason_desk(snapshots)
        tilts = _sector_tilts(sectors, momentum_by_symbol, symbols_by_sector) if len(sectors) > 1 else {}
        book = allocate(mandate, signals, theme_tilts=tilts or None)
        if book.weights:
            rebalance_dates.append(date)
            rebalance_weights.append(dict(book.weights))
            rebalance_tilts.append({s: t.net_tilt for s, t in tilts.items()})

    if not rebalance_dates:
        raise ValueError("No rebalance point produced a valid allocation in this window.")

    strategy_values = _compound(all_dates, rebalance_dates, rebalance_weights, price_series, initial_capital)
    benchmark_weights = {s: 1.0 / len(symbols) for s in symbols}
    benchmark_values = _compound(all_dates, [all_dates[0]], [benchmark_weights], price_series, initial_capital)
    rebalance_actions = _rebalance_actions(rebalance_dates, rebalance_weights, strategy_values, all_dates)

    # SPY buy-and-hold baseline — real, tradable S&P 500 proxy (see
    # SimulationResult's field docstring). Only computed if the caller
    # actually fetched SPY into price_series; a single anchor point at the
    # FIRST date this run covers (all_dates[0]), same buy-and-hold pattern
    # as the equal-weight benchmark above.
    spy_series = price_series.get("SPY", [])
    spy_values: List[float] = []
    if len(spy_series) > 1 and _price_on_or_before(price_series, "SPY", all_dates[0]):
        spy_values = _compound(all_dates, [all_dates[0]], [{"SPY": 1.0}], price_series, initial_capital)
    else:
        warnings.append("SPY price history unavailable for this window — S&P 500 comparison omitted.")
    spy_final = spy_values[-1] if spy_values else 0.0

    final_value = strategy_values[-1]
    benchmark_final = benchmark_values[-1]
    return SimulationResult(
        dates=all_dates,
        strategy_values=strategy_values,
        benchmark_values=benchmark_values,
        rebalance_dates=rebalance_dates,
        rebalance_weights=rebalance_weights,
        rebalance_actions=rebalance_actions,
        rebalance_tilts=rebalance_tilts,
        final_value=final_value,
        benchmark_final_value=benchmark_final,
        total_return=(final_value / initial_capital - 1.0) if initial_capital else 0.0,
        benchmark_return=(benchmark_final / initial_capital - 1.0) if initial_capital else 0.0,
        spy_values=spy_values,
        spy_final_value=spy_final,
        spy_return=(spy_final / initial_capital - 1.0) if (initial_capital and spy_values) else 0.0,
        warnings=warnings,
    )


def _rebalance_actions(
    rebalance_dates: List[str], rebalance_weights: List[Dict[str, float]],
    strategy_values: List[float], all_dates: List[str], min_trade_notional: float = 1.0,
) -> List[Dict[str, Tuple[str, float]]]:
    """Per rebalance, what changed vs the PREVIOUS rebalance's holdings —
    this is the actual 'monitor and decide when to sell' signal: a name
    dropping out or shrinking shows as a SELL, a new or growing one as BUY."""
    from execution_workflow import compute_order_deltas
    date_index = {d: i for i, d in enumerate(all_dates)}
    actions: List[Dict[str, Tuple[str, float]]] = []
    prev_weights: Dict[str, float] = {}
    for date, weights in zip(rebalance_dates, rebalance_weights):
        capital_at_rebalance = strategy_values[date_index[date]]
        orders = compute_order_deltas(prev_weights, weights, capital_at_rebalance,
                                      min_trade_notional=min_trade_notional)
        actions.append({o.symbol: (o.side.value.upper(), o.notional) for o in orders})
        prev_weights = weights
    return actions
