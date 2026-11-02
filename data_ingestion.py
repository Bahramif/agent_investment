"""
data_ingestion.py
==================
Turns raw market/news data into the InstrumentSnapshot / MacroSnapshot
objects the analysts and strategist reason over. This is the upstream piece
flagged as missing earlier — those objects were hand-supplied state in the
demos; this is what actually produces them.

SCOPE: this module computes what is MECHANICALLY derivable from price history
and fundamentals. Headlines are passed through as text; judging them is the LLM's
job (claude_reasoning.py), so without a model a stock's news is "unknown", never
guessed from keywords. snapshot_from_prices() is the ONE place a snapshot is
built from prices: the live path and the backtest (simulate.py) both use it, so
what is tested is what runs.

MarketDataClient / NewsClient are Protocols — bring your own client
(Polygon.io, Alpaca Market Data, a news API, etc.) satisfying the same
shape. FakeMarketDataClient / FakeNewsClient stand in for testing here, same
pattern as claude_reasoning.py's FakeAnthropicClient.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import rulebook
from analyst_reasoning import InstrumentSnapshot
from macro_reasoning import MacroSnapshot

BENCHMARK = "SPY"


# -----------------------------------------------------------------------------
# Client contracts — bring your own implementation satisfying these.
# -----------------------------------------------------------------------------
class MarketDataClient(Protocol):
    def get_price_history(self, symbol: str, days: int) -> List[float]:
        """Daily closing prices, oldest first, most recent last."""
        ...


class NewsClient(Protocol):
    def get_recent_headlines(self, symbol_or_theme: str, days: int) -> List[str]:
        """Recent headline TEXT — sentiment is scored from this below, not
        returned pre-scored, so the scoring logic lives in one place."""
        ...


# -----------------------------------------------------------------------------
# Pure computations — fully deterministic, fully testable.
# -----------------------------------------------------------------------------
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_momentum(prices: List[float]) -> float:
    """Simple trailing return over the given price history."""
    if len(prices) < 2 or prices[0] == 0:
        return 0.0
    return (prices[-1] - prices[0]) / prices[0]


def compute_returns(prices: List[float]) -> List[float]:
    """Daily returns over the given price history — the raw series both
    compute_volatility (a per-instrument summary stat) and
    mandate_allocator's real covariance-based optimizer (a CROSS-instrument
    view, needing the actual series, not just its stdev) are built from."""
    if len(prices) < 2:
        return []
    return [(prices[i] - prices[i - 1]) / prices[i - 1]
           for i in range(1, len(prices)) if prices[i - 1] != 0]


def compute_volatility(prices: List[float]) -> float:
    """Stdev of daily returns over the given price history."""
    returns = compute_returns(prices)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return variance ** 0.5


def compute_valuation_percentile(prices: List[float]) -> float:
    """Where the CURRENT price sits within its own trailing range — a crude
    but real proxy for 'cheap vs. rich relative to its own history,' with no
    dependence on external fundamentals data."""
    if not prices:
        return 0.5
    lo, hi = min(prices), max(prices)
    if hi == lo:
        return 0.5
    return _clip((prices[-1] - lo) / (hi - lo), 0.0, 1.0)


# -----------------------------------------------------------------------------
# Snapshot builders
# -----------------------------------------------------------------------------
HISTORY_DAYS = 400        # calendar days fetched per instrument: a full year of trading days plus a margin
RECENT_MONTH_BARS = 21    # the latest month's return, for display
SECTOR_HEADLINE_SAMPLE = 12


def compute_drawdown(prices: List[float]) -> float:
    """Latest price vs. the highest price in the window, <= 0."""
    if not prices or max(prices) <= 0:
        return 0.0
    return min(0.0, prices[-1] / max(prices) - 1.0)


def snapshot_from_prices(symbol: str, path: str, prices: List[float], quality_score: float, eps: Optional[float] = None,
                         industry: str = "", benchmark: Optional[List[float]] = None,
                         earnings_date: Optional[str] = None, avg_volume: Optional[float] = None,
                         headlines: Optional[List[str]] = None, news_status: str = "unknown",
                         analyst: Optional[Tuple[float, str, int]] = None) -> InstrumentSnapshot:
    """Prices are daily closes up to the moment of the decision, oldest first."""
    last = prices[-1] if prices else None
    earnings_yield = (eps / last) if eps is not None and last else None
    expected, rating, count = analyst if analyst else (None, None, 0)
    return InstrumentSnapshot(
        name=symbol, path=path,
        momentum_1m=compute_momentum(prices[-RECENT_MONTH_BARS - 1:]),
        news_sentiment=0.0,
        quality_score=quality_score,
        volatility=rulebook.annual_window_vol(prices) or 0.0,
        analyst_expected_return=expected, analyst_rating=rating, analyst_count=count,
        drawdown_6m=rulebook.drawdown_from_high(prices),
        pullback_5d=rulebook.trailing_return(prices, 5),
        momentum_12_1=rulebook.momentum_12_1(prices),
        earnings_yield=earnings_yield,
        industry=industry,
        beta=rulebook.beta_to(prices, benchmark) if benchmark else None,
        earnings_date=earnings_date,
        history_bars=len(prices),
        avg_dollar_volume=(avg_volume * last) if avg_volume and last else None,
        range_position=compute_valuation_percentile(prices[-rulebook.TRADING_DAYS_PER_YEAR:]),
        headlines=[str(h)[:200] for h in (headlines or [])[:5]],
        news_status=news_status,
    )


def build_instrument_snapshot(
    symbol: str, path: str, quality_score: float,
    market_client: MarketDataClient, news_client: NewsClient,
    benchmark: Optional[List[float]] = None,
) -> InstrumentSnapshot:
    """One price request covers everything: a year of factors plus the latest
    month for display. `quality_score` is the fallback when there are no
    fundamentals for this name."""
    from fundamentals import get_quality_score, get_analyst_forecast, get_factor_inputs
    prices = market_client.get_price_history(symbol, HISTORY_DAYS)
    headlines = news_client.get_recent_headlines(symbol, days=7)
    inputs = get_factor_inputs(symbol)
    return snapshot_from_prices(
        symbol, path, prices, get_quality_score(symbol, fallback=quality_score), eps=inputs["eps"],
        industry=inputs["industry"], benchmark=benchmark, earnings_date=inputs["earnings_date"],
        avg_volume=inputs["avg_volume"], headlines=headlines,
        analyst=get_analyst_forecast(symbol, current_price=prices[-1] if prices else None))


def build_macro_from_snapshots(theme: str, snapshots: Sequence[InstrumentSnapshot]) -> MacroSnapshot:
    """A sector's read from ALL its stocks: the median, so one outlier can't
    speak for the sector. Re-uses the data already fetched for the stocks."""
    headlines: List[str] = []
    for s in sorted(snapshots, key=lambda x: -len(x.headlines)):
        if s.headlines:
            headlines.append(f"{s.name}: {s.headlines[0]}")
    return MacroSnapshot(
        theme=theme,
        momentum_1m=statistics.median(s.momentum_1m for s in snapshots) if snapshots else 0.0,
        news_sentiment=0.0,
        valuation_percentile=statistics.median(s.range_position for s in snapshots) if snapshots else 0.5,
        headlines=headlines[:SECTOR_HEADLINE_SAMPLE],
    )


def build_macro_snapshot(
    theme: str, constituents: Sequence[str], secular_score: float,
    market_client: MarketDataClient, news_client: NewsClient,
    lookback_days: int = 21,
) -> MacroSnapshot:
    """A theme with an explicit constituent list. Headlines are fetched per
    constituent: a news API knows tickers, not a theme name like "tech_ai"."""
    momenta, valuations, headlines = [], [], []
    for symbol in constituents:
        prices = market_client.get_price_history(symbol, lookback_days)
        momenta.append(compute_momentum(prices))
        valuations.append(compute_valuation_percentile(prices))
        headlines += [f"{symbol}: {h}" for h in news_client.get_recent_headlines(symbol, days=7)[:1]]
    return MacroSnapshot(
        theme=theme,
        momentum_1m=sum(momenta) / len(momenta) if momenta else 0.0,
        news_sentiment=0.0,
        valuation_percentile=sum(valuations) / len(valuations) if valuations else 0.5,
        secular_score=secular_score,
        headlines=headlines[:SECTOR_HEADLINE_SAMPLE],
    )


# -----------------------------------------------------------------------------
# LangGraph node factory — bakes in the universe + clients at graph-build
# time, same pattern as claude_reasoning.make_claude_reasoners() /
# firm_graph.make_analyst_node().
# -----------------------------------------------------------------------------
@dataclass
class UniverseConfig:
    """What to fetch data for, and how to tag it. Static config, bound once
    at graph-build time — not per-cycle state, since it rarely changes."""
    equities: List[Tuple[str, str, float]]       # (symbol, path, quality_score)
    forex: List[Tuple[str, str, float]]
    crypto: List[Tuple[str, str, float]]
    themes: List[Tuple[str, List[str], float]]   # (theme, constituent symbols, secular_score);
                                                 # no constituents listed = all the stocks in that sector


def make_data_ingestion_node(
    universe: UniverseConfig,
    market_client: MarketDataClient,
    news_client: NewsClient,
    max_workers: int = 16,
):
    """Each build_instrument_snapshot/build_macro_snapshot call is TWO
    sequential network round trips (price history + news) against a single
    symbol/theme. With a small hand-picked universe (a handful of names)
    that's fine done one at a time; once the universe grows toward
    real-sized sector coverage (dozens to ~100 names — see
    vertex_deploy.py's UniverseConfig), doing this serially turns into
    minutes of wall-clock time per query() call. These are all independent,
    network-bound I/O calls, so a thread pool (not multiprocessing — the
    GIL isn't the bottleneck here, waiting on HTTP responses is) gives a
    near-linear speedup without changing what gets fetched."""
    def _node(state: dict) -> dict:
        from concurrent.futures import ThreadPoolExecutor

        benchmark = None
        try:
            benchmark = market_client.get_price_history(BENCHMARK, HISTORY_DAYS)
        except Exception as e:                       # beta is one input, not a prerequisite
            print(f"[data_ingestion] {BENCHMARK} history unavailable ({e}): beta unknown this run")

        def _equity(args):
            sym, path, q = args
            return build_instrument_snapshot(sym, path, q, market_client, news_client, benchmark)

        def _macro(args):
            theme, constituents, sec = args
            if constituents:
                return build_macro_snapshot(theme, constituents, sec, market_client, news_client)
            return build_macro_from_snapshots(theme, [s for s in equities if rulebook.sector_of(s.path) == theme])

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            equities = list(pool.map(_equity, universe.equities))
            forex = list(pool.map(_equity, universe.forex))
            crypto = list(pool.map(_equity, universe.crypto))
            macro = list(pool.map(_macro, universe.themes))

        return {
            "equities_snapshots": equities, "forex_snapshots": forex,
            "crypto_snapshots": crypto, "macro_snapshots": macro,
        }
    return _node


# -----------------------------------------------------------------------------
# Fakes for testing — no network access needed.
# -----------------------------------------------------------------------------
class FakeMarketDataClient:
    def __init__(self, price_series: Dict[str, List[float]]):
        self.price_series = price_series

    def get_price_history(self, symbol: str, days: int) -> List[float]:
        return self.price_series.get(symbol, [100.0] * days)


class FakeNewsClient:
    def __init__(self, headlines: Dict[str, List[str]]):
        self.headlines = headlines

    def get_recent_headlines(self, symbol_or_theme: str, days: int) -> List[str]:
        return self.headlines.get(symbol_or_theme, [])


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: pure computations")
    print("=" * 70)
    assert compute_momentum([100, 110]) == 0.10
    assert compute_momentum([100]) == 0.0            # too little history
    assert compute_momentum([0, 50]) == 0.0           # avoid div-by-zero
    assert compute_valuation_percentile([90, 110, 100]) == 0.5    # current price sits at the midpoint of its own range
    assert compute_valuation_percentile([90, 90, 90]) == 0.5      # flat range, no signal either way
    assert compute_valuation_percentile([90, 95, 100]) == 1.0     # current price is the high of its range
    print("✓ momentum and valuation-percentile computations correct")

    print("\n" + "=" * 70)
    print("TEST 2: snapshot builders with fake clients")
    print("=" * 70)
    market = FakeMarketDataClient({
        "NVDA": [100, 105, 110, 118],
        "CAT": [100, 99, 97, 95],
        "XOM": [80, 82, 81, 85],
        "FSLR": [50, 52, 54, 53],
    })
    news = FakeNewsClient({
        "NVDA": ["NVDA beats on record AI demand, stock surges"],
        "CAT": ["CAT misses on weak industrial demand"],
    })

    nvda_snap = build_instrument_snapshot("NVDA", "equities.tech_ai", quality_score=0.7,
                                         market_client=market, news_client=news)
    print(f"  NVDA: momentum={nvda_snap.momentum_1m:+.2%} "
          f"quality={nvda_snap.quality_score:+.2f} (REAL fundamentals via yfinance, not the 0.7 fallback)")
    assert nvda_snap.momentum_1m > 0 and nvda_snap.news_status == "unknown", "headlines are the LLM's to judge"
    assert nvda_snap.headlines == ["NVDA beats on record AI demand, stock surges"]
    assert 0.0 <= nvda_snap.quality_score <= 1.0
    # A fake ticker yfinance can't resolve -> falls back to the supplied value untouched.
    fake_snap = build_instrument_snapshot("ZZZZFAKETICKER", "equities.tech_ai", quality_score=0.42,
                                          market_client=market, news_client=news)
    print(f"  unresolvable ticker: quality={fake_snap.quality_score:+.2f} (falls back to supplied 0.42)")
    assert fake_snap.quality_score == 0.42
    print("✓ instrument snapshot combines derived fields with REAL fundamentals, falling back cleanly")

    energy_snap = build_macro_snapshot("energy", ["XOM", "FSLR"], secular_score=0.2,
                                       market_client=market, news_client=news)
    print(f"  energy theme: momentum={energy_snap.momentum_1m:+.2%} "
          f"(avg of XOM/FSLR) secular_score={energy_snap.secular_score:+.2f} (supplied unchanged)")
    assert energy_snap.secular_score == 0.2
    print("✓ macro snapshot correctly averages constituents and passes through secular_score")

    print("\n" + "=" * 70)
    print("TEST 3: make_data_ingestion_node() — the actual LangGraph node")
    print("=" * 70)
    universe = UniverseConfig(
        equities=[("NVDA", "equities.tech_ai", 0.7), ("CAT", "equities.industrials", 0.3)],
        forex=[], crypto=[],
        themes=[("energy", ["XOM", "FSLR"], 0.2)],
    )
    ingestion_node = make_data_ingestion_node(universe, market, news)
    result = ingestion_node({})
    print(f"  keys produced: {sorted(result.keys())}")
    assert set(result.keys()) == {"equities_snapshots", "forex_snapshots", "crypto_snapshots", "macro_snapshots"}
    assert len(result["equities_snapshots"]) == 2 and len(result["macro_snapshots"]) == 1
    print("✓ node produces exactly the state keys firm_graph.py's analyst/strategist nodes expect")

    print("\n" + "=" * 70)
    print("TEST 4: INTEGRATION — ingested snapshots feed straight into the")
    print("real analyst-scoring logic from analyst_reasoning.py")
    print("=" * 70)
    from analyst_reasoning import reason_desk
    signals = reason_desk(result["equities_snapshots"])
    print(f"  {[(s.name, round(s.score, 3)) for s in signals]}")
    assert len(signals) == 2
    print("✓ ingestion output is directly consumable by the existing analyst scorer —")
    print("  no adapter needed between this module and firm_graph.py")

    print("TEST: ONE ~13-month fetch feeds every factor; the snapshot needs nothing else from prices")
    import fundamentals as _f
    _f.get_quality_score = lambda sym, fallback=0.5: 0.8
    _f.get_analyst_forecast = lambda sym, current_price=None: (0.2, "Buy", 15)
    _f.get_factor_inputs = lambda sym: {"eps": 10.0, "industry": "Semiconductors", "earnings_date": "2026-11-20",
                                        "avg_volume": 2_000_000}
    asked = []
    hist = [100 + i * 0.5 for i in range(240)] + [220 - i for i in range(40)]
    class LongFake(FakeMarketDataClient):
        def get_price_history(self, symbol, days):
            asked.append(days)
            return hist
    snap = build_instrument_snapshot("DIP", "equities.tech_ai", 0.5, LongFake({}), FakeNewsClient({"DIP": ["x"]}), benchmark=hist)
    assert asked == [HISTORY_DAYS], "one request"
    assert abs(snap.momentum_1m - (hist[-1] / hist[-22] - 1)) < 1e-12, "the latest month, for display"
    assert abs(snap.momentum_12_1 - (hist[-22] / hist[-253] - 1)) < 1e-12, "12 months ago to 1 month ago"
    assert snap.history_bars == 280 and abs(snap.beta - 1.0) < 1e-9 and snap.industry == "Semiconductors"
    assert abs(snap.earnings_yield - 10.0 / hist[-1]) < 1e-12 and abs(snap.avg_dollar_volume - 2e6 * hist[-1]) < 1e-6
    assert snap.volatility > 0 and snap.drawdown_6m < 0 and snap.analyst_expected_return == 0.2
    print(f"✓ 12-1 momentum {snap.momentum_12_1:+.1%}, latest month {snap.momentum_1m:+.1%}, beta {snap.beta:.2f}")

    print("TEST: a sector is read from ALL its stocks, and the news API is never asked about a theme name")
    asked_news = []
    class SpyNews(FakeNewsClient):
        def get_recent_headlines(self, symbol_or_theme, days):
            asked_news.append(symbol_or_theme)
            return [f"{symbol_or_theme} headline"]
    uni = UniverseConfig(equities=[("AAA", "equities.energy", 0.5), ("BBB", "equities.energy", 0.5), ("CCC", "equities.tech_ai", 0.5)],
                         forex=[], crypto=[], themes=[("energy", [], 0.0), ("tech_ai", [], 0.0)])
    out = make_data_ingestion_node(uni, LongFake({}), SpyNews({}))({})
    macros = {m.theme: m for m in out["macro_snapshots"]}
    assert "energy" not in asked_news and "tech_ai" not in asked_news, asked_news
    assert sorted(macros["energy"].headlines) == ["AAA: AAA headline", "BBB: BBB headline"]
    assert macros["energy"].secular_score == 0.0
    print("✓ medians across the sector's stocks; headlines by ticker; no hand-set tailwind")
