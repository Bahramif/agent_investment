"""
fundamentals.py
================
Real per-instrument data from Yahoo Finance (via yfinance), covering two
things this app previously faked or self-derived:

  1. "Quality" scoring from actual company fundamentals — REPLACES the flat,
     hardcoded 0.5 quality_score every equity in universe.py previously
     carried. Confirmed live: every single one of the 223 names had the
     literal identical value 0.5, contributing ZERO discriminating power to
     conviction scoring despite nominally carrying 20% of the formula's
     weight.

  2. REAL analyst forecasts (target price, consensus rating, analyst count)
     — REPLACES the self-referential "expected_return"/"rating" the
     heuristic and Claude scorers previously invented from the SAME four
     numbers already driving conviction (i.e. not independent information
     at all). This module's numbers are genuine third-party Wall Street
     analyst data, confirmed live against real tickers (targetMeanPrice,
     recommendationKey, numberOfAnalystOpinions are all real yfinance
     fields, not something this app computes).

WHY CACHED, NOT FETCHED LIVE ON EVERY REQUEST: a single yfinance lookup
takes ~0.25-0.5s (measured live) — across a 223-symbol universe, even at
the existing 3-way ingestion concurrency, that's 25+ seconds ADDED to every
recommendation request, which would directly reintroduce the timeout/
latency problem already fixed once this session (see alpaca_clients.py's
apply_default_timeout). Both quality fundamentals AND analyst estimates
move on an earnings/coverage-update cadence, not tick by tick — so a
same-process in-memory cache with a generous TTL is the correct engineering
trade, not a shortcut. Both scores are derived from ONE cached yfinance
`get_info()` call per symbol (not two), so adding analyst forecasts costs
nothing extra beyond what quality scoring already paid.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional, Tuple

_INFO_CACHE: Dict[str, Tuple[dict, float]] = {}   # symbol -> (info dict, fetched_at monotonic)
_TTL_SECONDS = 20 * 3600   # fundamentals/estimates move on a coverage cadence, not intraday


def _get_info(symbol: str) -> Optional[dict]:
    """The one cached yfinance call every function in this module shares.
    Returns None (never raises) if the lookup fails outright."""
    cached = _INFO_CACHE.get(symbol)
    if cached is not None and (time.monotonic() - cached[1]) < _TTL_SECONDS:
        return cached[0]
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).get_info()
    except Exception as e:
        print(f"[fundamentals] info lookup failed for {symbol}: {e}")
        return None
    _INFO_CACHE[symbol] = (info, time.monotonic())
    return info


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_from_info(info: dict) -> Optional[float]:
    """Pure function: a yfinance Ticker.get_info() dict -> a 0-1 quality
    score, or None if too little fundamental data is present to say
    anything (caller falls back to universe.py's static default in that
    case, rather than silently scoring on partial nonsense).

    Weights favor profitability/returns — the core of a "quality" factor in
    the Fama-French/AQR sense — over growth or leverage, and deliberately
    exclude valuation (P/E etc.): mixing valuation in would make this a
    VALUE score, not a quality score, and would double up with momentum/
    price data that already drives the rest of the conviction formula."""
    parts = []   # (weight, normalized_0_1_value)
    margins = info.get("profitMargins")
    if margins is not None:
        parts.append((0.35, _clip01(margins / 0.30)))            # 30%+ net margin -> full credit
    roe = info.get("returnOnEquity")
    if roe is not None:
        parts.append((0.35, _clip01(roe / 0.30)))                # 30%+ ROE -> full credit
    dte = info.get("debtToEquity")
    if dte is not None:
        parts.append((0.15, _clip01(1.0 - dte / 200.0)))         # 200%+ debt/equity -> zero credit
    growth = info.get("revenueGrowth")
    if growth is not None:
        parts.append((0.15, _clip01((growth + 0.05) / 0.25)))    # -5%..+20% mapped to 0..1

    if not parts:
        return None
    total_weight = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / total_weight


def factor_inputs_from_info(info: Optional[dict], today: Optional[date] = None) -> Dict[str, Any]:
    """Pure function: the fields of a yfinance info dict the rulebook reads.
    Anything missing comes back None/"" and is treated as unknown, not as bad."""
    info = info or {}
    today = today or date.today()
    eps = info.get("forwardEps")
    if eps is None:
        eps = info.get("trailingEps")
    earnings = None
    stamp = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
    if stamp:
        try:
            d = datetime.fromtimestamp(float(stamp), tz=timezone.utc).date()
            earnings = d.isoformat() if d >= today else None
        except (TypeError, ValueError, OverflowError, OSError):
            earnings = None
    return {"eps": float(eps) if eps is not None else None, "industry": str(info.get("industry") or ""),
            "earnings_date": earnings,
            "avg_volume": info.get("averageVolume") or info.get("averageDailyVolume3Month")}


def get_factor_inputs(symbol: str) -> Dict[str, Any]:
    """Cached, resilient lookup of factor_inputs_from_info. Never raises."""
    return factor_inputs_from_info(_get_info(symbol))


def get_quality_score(symbol: str, fallback: float = 0.5) -> float:
    """Cached, resilient lookup — real fundamentals when yfinance has them,
    the caller's fallback (universe.py's static per-symbol value) when it
    doesn't or the call fails outright. Never raises."""
    info = _get_info(symbol)
    if info is None:
        return fallback
    score = score_from_info(info)
    return fallback if score is None else score


def forecast_from_info(info: dict, current_price: Optional[float]) -> Optional[Tuple[float, str, int]]:
    """Pure function: a yfinance info dict + a current price -> (expected_return,
    rating, num_analysts), or None if there's no real analyst coverage to
    report (small-cap/thinly-covered names, or a missing current_price).

    expected_return is the REAL Wall Street analyst consensus target price's
    implied forward return — (targetMeanPrice - current_price) / current_price
    — not derived from momentum/sentiment/quality/volatility at all, unlike
    the heuristic/Claude "forecast" this replaces. rating is mapped from
    yfinance's own recommendationKey (the analysts' actual consensus call),
    not re-labeled conviction. num_analysts (numberOfAnalystOpinions) is
    surfaced so callers/UI can show confidence — 2 analysts covering a name
    is a much weaker signal than 40."""
    target = info.get("targetMeanPrice")
    current_price = current_price or info.get("currentPrice")
    num_analysts = info.get("numberOfAnalystOpinions") or 0
    if not target or not current_price or num_analysts < 1:
        return None
    expected_return = max(-0.75, min(0.75, (target - current_price) / current_price))
    key = (info.get("recommendationKey") or "").lower()
    if key in ("strong_buy", "buy"):
        rating = "Buy"
    elif key in ("sell", "strong_sell", "underperform"):
        rating = "Sell"
    else:
        rating = "Hold"   # covers "hold", "none", "", and anything unrecognized
    return expected_return, rating, int(num_analysts)


def get_analyst_forecast(
    symbol: str, current_price: Optional[float] = None,
) -> Optional[Tuple[float, str, int]]:
    """Cached, resilient lookup — see forecast_from_info's docstring for
    what these numbers actually mean. Returns None (caller falls back to
    its own heuristic/Claude-generated estimate) when yfinance has no
    analyst coverage for this symbol or the lookup fails. Never raises."""
    info = _get_info(symbol)
    if info is None:
        return None
    return forecast_from_info(info, current_price)


if __name__ == "__main__":
    print("=" * 70)
    print("TEST: score_from_info is a pure function of the numbers given —")
    print("no network needed to verify the scoring logic itself.")
    print("=" * 70)
    high_quality = {"profitMargins": 0.35, "returnOnEquity": 0.40, "debtToEquity": 20.0, "revenueGrowth": 0.15}
    low_quality = {"profitMargins": 0.02, "returnOnEquity": 0.03, "debtToEquity": 250.0, "revenueGrowth": -0.10}
    s_high, s_low = score_from_info(high_quality), score_from_info(low_quality)
    print(f"  high-margin, high-ROE, low-debt name -> {s_high:.2f}")
    print(f"  thin-margin, high-debt, shrinking name -> {s_low:.2f}")
    assert s_high > 0.8, "a genuinely strong balance sheet/margin profile should score high"
    assert s_low < 0.2, "a genuinely weak one should score low"
    assert score_from_info({}) is None, "no usable fields at all -> None, not a fake middle score"
    print("✓ high-quality profile scores high, low-quality scores low, empty info -> None (fallback)")

    print()
    print("=" * 70)
    print("TEST: forecast_from_info is a pure function too")
    print("=" * 70)
    covered = {"targetMeanPrice": 150.0, "recommendationKey": "buy", "numberOfAnalystOpinions": 30}
    f = forecast_from_info(covered, current_price=100.0)
    print(f"  $100 current, $150 target, 'buy' -> {f}")
    assert f is not None and abs(f[0] - 0.5) < 1e-9 and f[1] == "Buy" and f[2] == 30
    assert forecast_from_info({}, current_price=100.0) is None, "no target/coverage -> None, not a fake number"
    assert forecast_from_info(covered, current_price=None) is None, "no current price to compare against -> None"
    print("✓ real target price -> real implied return, no coverage -> None (fallback), consistent with quality")

    print()
    print("TEST: factor_inputs_from_info reads EPS, industry, next earnings and volume; missing fields stay unknown")
    day = date(2026, 11, 1)
    soon = int(datetime(2026, 11, 3, 21, tzinfo=timezone.utc).timestamp())
    past = int(datetime(2026, 8, 3, 21, tzinfo=timezone.utc).timestamp())
    got = factor_inputs_from_info({"forwardEps": 5.0, "trailingEps": 4.0, "industry": "Semiconductors",
                                   "earningsTimestampStart": soon, "averageVolume": 1000}, today=day)
    assert got == {"eps": 5.0, "industry": "Semiconductors", "earnings_date": "2026-11-03", "avg_volume": 1000}, got
    assert factor_inputs_from_info({"trailingEps": 4.0}, today=day)["eps"] == 4.0
    assert factor_inputs_from_info({"earningsTimestamp": past}, today=day)["earnings_date"] is None, "a past report is not upcoming"
    assert factor_inputs_from_info(None) == {"eps": None, "industry": "", "earnings_date": None, "avg_volume": None}
    print("✓ ok")

    print()
    print("=" * 70)
    print("LIVE TEST (needs network): a few real symbols")
    print("=" * 70)
    try:
        for sym in ["AAPL", "NVDA", "XOM"]:
            q = get_quality_score(sym)
            fc = get_analyst_forecast(sym)
            print(f"  {sym}: quality={q:.3f}  analyst_forecast={fc}")
        print("  (second call should be instant — served from the shared info cache)")
        t0 = time.monotonic()
        get_analyst_forecast("AAPL")
        print(f"  AAPL cached lookup took {time.monotonic()-t0:.4f}s")
    except Exception as e:
        print(f"  (skipped — no network in this environment: {e})")
