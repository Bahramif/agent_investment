"""
data_ingestion.py
==================
Turns raw market/news data into the InstrumentSnapshot / MacroSnapshot
objects the analysts and strategist reason over. This is the upstream piece
flagged as missing earlier — those objects were hand-supplied state in the
demos; this is what actually produces them.

SCOPE, STATED HONESTLY: this module computes what's MECHANICALLY derivable
from price history and headlines — momentum, a valuation percentile, and
aggregate news sentiment. `quality_score` is now real too (fundamentals.py,
via yfinance — profit margins, ROE, leverage, revenue growth), cached
per-symbol so it doesn't add yfinance latency to every request. UniverseConfig's
per-symbol quality value survives only as the FALLBACK when yfinance has
nothing usable for a name. `secular_score` (per-theme structural thesis)
is still supplied config — a structural/narrative judgment about a THEME
(e.g. "AI capex supercycle"), not a per-company number a data feed
produces, so it stays hand-set.

MarketDataClient / NewsClient are Protocols — bring your own client
(Polygon.io, Alpaca Market Data, a news API, etc.) satisfying the same
shape. FakeMarketDataClient / FakeNewsClient stand in for testing here, same
pattern as claude_reasoning.py's FakeAnthropicClient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from analyst_reasoning import InstrumentSnapshot
from macro_reasoning import MacroSnapshot


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
    """Stdev of daily returns over the given price history — a simple,
    transparent risk proxy. Per-instrument only; see returns_window on
    InstrumentSnapshot/Signal for the real per-instrument return SERIES
    that lets mandate_allocator build an actual cross-instrument covariance
    matrix instead of treating instruments as if they never move together."""
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


# A tiny, transparent sentiment scorer — explicitly NOT a real NLP model. In
# production, swap this for a real sentiment classifier or an LLM call; the
# interface (headlines in, score in [-1, 1] out) is what actually matters.
_POSITIVE_WORDS = {"beats", "surges", "record", "growth", "upgrade", "rally", "strong", "wins", "soars"}
_NEGATIVE_WORDS = {"misses", "plunges", "recall", "downgrade", "lawsuit", "weak", "cuts", "probe", "slumps"}


def score_headline_sentiment(headlines: List[str]) -> float:
    if not headlines:
        return 0.0
    scores = []
    for h in headlines:
        words = set(h.lower().split())
        scores.append(len(words & _POSITIVE_WORDS) - len(words & _NEGATIVE_WORDS))
    avg = sum(scores) / len(scores)
    return _clip(avg / 2.0, -1.0, 1.0)   # crude normalization — swap for something real


# -----------------------------------------------------------------------------
# Snapshot builders
# -----------------------------------------------------------------------------
def build_instrument_snapshot(
    symbol: str, path: str, quality_score: float,
    market_client: MarketDataClient, news_client: NewsClient,
    lookback_days: int = 21,
) -> InstrumentSnapshot:
    from fundamentals import get_quality_score, get_analyst_forecast
    prices = market_client.get_price_history(symbol, lookback_days)
    headlines = news_client.get_recent_headlines(symbol, days=7)
    # None (not 0.0/"") when there's no real analyst coverage — the scorer
    # distinguishes "genuinely no coverage" from "a real 0% expected return"
    # and falls back to its own heuristic only in the former case.
    forecast = get_analyst_forecast(symbol, current_price=prices[-1] if prices else None)
    analyst_expected_return, analyst_rating, analyst_count = forecast if forecast else (None, None, 0)
    return InstrumentSnapshot(
        name=symbol, path=path,
        momentum_1m=compute_momentum(prices),
        news_sentiment=score_headline_sentiment(headlines),
        # `quality_score` param is now a FALLBACK ONLY, used when yfinance
        # has nothing usable for this symbol — real value comes from
        # fundamentals.py (profit margins, ROE, leverage, revenue growth).
        quality_score=get_quality_score(symbol, fallback=quality_score),
        volatility=compute_volatility(prices),
        analyst_expected_return=analyst_expected_return,
        analyst_rating=analyst_rating,
        analyst_count=analyst_count,
        returns_window=compute_returns(prices),
    )


def build_macro_snapshot(
    theme: str, constituents: Sequence[str], secular_score: float,
    market_client: MarketDataClient, news_client: NewsClient,
    lookback_days: int = 21,
) -> MacroSnapshot:
    momenta, valuations = [], []
    for symbol in constituents:
        prices = market_client.get_price_history(symbol, lookback_days)
        momenta.append(compute_momentum(prices))
        valuations.append(compute_valuation_percentile(prices))
    headlines = news_client.get_recent_headlines(theme, days=7)
    return MacroSnapshot(
        theme=theme,
        momentum_1m=sum(momenta) / len(momenta) if momenta else 0.0,
        news_sentiment=score_headline_sentiment(headlines),
        valuation_percentile=sum(valuations) / len(valuations) if valuations else 0.5,
        secular_score=secular_score,   # supplied, not derived — see module docstring
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
    themes: List[Tuple[str, List[str], float]]   # (theme, constituent symbols, secular_score)


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

        def _equity(args):
            sym, path, q = args
            return build_instrument_snapshot(sym, path, q, market_client, news_client)

        def _macro(args):
            theme, constituents, sec = args
            return build_macro_snapshot(theme, constituents, sec, market_client, news_client)

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

    pos = score_headline_sentiment(["Company beats estimates, stock surges on record growth"])
    neg = score_headline_sentiment(["Company misses estimates as lawsuit probe weighs on shares"])
    neutral = score_headline_sentiment([])
    print(f"  positive headline -> {pos:+.2f}, negative -> {neg:+.2f}, none -> {neutral:+.2f}")
    assert pos > 0 and neg < 0 and neutral == 0.0
    print("✓ sentiment scorer distinguishes positive/negative/absent headlines")

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
    print(f"  NVDA: momentum={nvda_snap.momentum_1m:+.2%} sentiment={nvda_snap.news_sentiment:+.2f} "
          f"quality={nvda_snap.quality_score:+.2f} (REAL fundamentals via yfinance, not the 0.7 fallback)")
    assert nvda_snap.momentum_1m > 0 and nvda_snap.news_sentiment > 0
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
