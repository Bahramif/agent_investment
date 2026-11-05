"""
rulebook_backtest.py
====================
The evidence behind the numbers in mandate.yaml's `rulebook:` block. Runs the real
rules (simulate.run_simulation) over several separate market periods, and asks:

  1. Do they beat what they should beat? Random picks with the SAME mechanics (the
     null model), equal-weighting the same stocks, and SPY.
  2. Are the settings robust? Each rule is varied one at a time. A setting should
     hold up across periods, not win one of them. Change a default only on that
     evidence, never to fit one incident.

READ THE RESULTS WITH THESE LIMITS IN MIND:
  - Survivorship: the stock list is today's, so every line is flattered. That's why
    the comparison that matters is against the null model and equal weight, which
    share the same bias, not the absolute return.
  - Lookahead in fundamentals: quality, EPS and industry are today's values applied
    to the past. A stock whose earnings later grew looks "cheap" back then. The
    "prices only" line removes quality and value, so it's the cleanest test.
  - The LLM and news verdicts can't be replayed: this tests the deterministic rules.

Prices come from Yahoo Finance so the run needs no broker keys.

Run:  python rulebook_backtest.py --out backtest_report.md [--cache-dir DIR] [--workers 8]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import rulebook
from analyst_reasoning import reason_desk
from simulate import run_simulation
from universe import EQUITIES, SECTORS

PERIODS: List[Tuple[str, str, str]] = [
    ("2017-18", "2017-01-03", "2018-12-31"),
    ("2019-20", "2019-01-02", "2020-12-31"),
    ("2021-22", "2021-01-04", "2022-12-30"),
    ("2023-24", "2023-01-03", "2024-12-31"),
    ("2025-26", "2025-01-02", "2026-10-30"),
]
ALL_SECTORS = list(SECTORS)
TOP_N, MAX_PER_NAME, CAPITAL = 12, 1 / 6, 100_000.0
BASE = rulebook.default_config()


def _vary(**kw) -> rulebook.RulebookConfig:
    return dataclasses.replace(BASE, **kw)


VARIANTS: Dict[str, rulebook.RulebookConfig] = {
    "default": BASE,
    "prices only (no quality/value)": _vary(weight_quality=0.0, weight_value=0.0),
    "no stop": _vary(stop_sigma_mult=99.0),
    "stop 1.5 sigma": _vary(stop_sigma_mult=1.5),
    "stop 2.0 sigma": _vary(stop_sigma_mult=2.0),
    "no keep band (1x)": _vary(keep_rank_multiple=1.0),
    "keep band 1.5x": _vary(keep_rank_multiple=1.5),
    "keep band 3x": _vary(keep_rank_multiple=3.0),
    "no pullback credit": _vary(pullback_weight=0.0),
    "no re-entry block": _vary(reentry_block_days=0),
    "no sub-industry limit": _vary(max_industry_weight=1.0),
    "no beta limit": _vary(max_portfolio_beta=99.0),
}
PRICES_ONLY = dict(weight_quality=0.0, weight_value=0.0)
# The clean test: the same questions with the lookahead-prone factors removed.
CLEAN_VARIANTS: Dict[str, rulebook.RulebookConfig] = {
    "default": _vary(**PRICES_ONLY),
    "no stop": _vary(stop_sigma_mult=99.0, **PRICES_ONLY),
    "stop 1.5 sigma": _vary(stop_sigma_mult=1.5, **PRICES_ONLY),
    "stop 2.0 sigma": _vary(stop_sigma_mult=2.0, **PRICES_ONLY),
    "no keep band (1x)": _vary(keep_rank_multiple=1.0, **PRICES_ONLY),
    "keep band 1.5x": _vary(keep_rank_multiple=1.5, **PRICES_ONLY),
    "keep band 3x": _vary(keep_rank_multiple=3.0, **PRICES_ONLY),
    "no pullback credit": _vary(pullback_weight=0.0, **PRICES_ONLY),
    "no re-entry block": _vary(reentry_block_days=0, **PRICES_ONLY),
}
VARIANTS.update({f"clean: {k}": v for k, v in CLEAN_VARIANTS.items()})
NULL_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)


def _job_key(job: Tuple[str, str, Optional[int], List[str], int, float]) -> str:
    """A finished run is reused only while its exact config and setup are unchanged."""
    name, period, seed, sectors, n, cap = job
    cfg = rulebook.to_dict(VARIANTS.get(name, BASE))
    return json.dumps([name, period, seed, sectors, n, round(cap, 6), cfg], sort_keys=True)


def load_prices(cache_dir: str) -> Dict[str, List[Tuple[str, float]]]:
    path = os.path.join(cache_dir, "prices.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    import yfinance as yf
    symbols = sorted({s for s, _, _ in EQUITIES} | {"SPY"})
    data = yf.download(symbols, start="2015-06-01", auto_adjust=True, progress=False, threads=True)["Close"]
    out = {}
    for s in symbols:
        col = data[s].dropna()
        out[s] = [(d.strftime("%Y-%m-%d"), float(v)) for d, v in col.items()]
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def load_fundamentals(cache_dir: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(cache_dir, "fundamentals.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    from concurrent.futures import ThreadPoolExecutor
    from fundamentals import get_factor_inputs, get_quality_score
    symbols = sorted(s for s, _, _ in EQUITIES)

    def one(s: str) -> Dict[str, Any]:
        return {**get_factor_inputs(s), "quality": get_quality_score(s, fallback=0.5)}
    with ThreadPoolExecutor(max_workers=6) as pool:
        out = dict(zip(symbols, pool.map(one, symbols)))
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


def random_scorer(seed: int, cfg: rulebook.RulebookConfig):
    """The null model: the same eligibility, limits, stops and costs, with the
    score replaced by a coin toss at each decision."""
    rng = random.Random(seed)

    def score(snaps: List[Any]) -> List[Any]:
        signals = reason_desk(snaps, cfg)
        for s in sorted(signals, key=lambda x: x.name):
            s.score = rng.random()
        return signals
    return score


_PRICES: Dict[str, Any] = {}
_FUND: Dict[str, Any] = {}


def _init(prices: Dict[str, Any], fund: Dict[str, Any]) -> None:
    global _PRICES, _FUND
    _PRICES, _FUND = prices, fund


def _run(job: Tuple[str, str, Optional[int], List[str], int, float]) -> Dict[str, Any]:
    name, period, seed, sectors, n, cap = job
    label, start, end = next(p for p in PERIODS if p[0] == period)
    cfg = VARIANTS.get(name, BASE)
    # only prices known by `end` are passed, and each symbol only exists once it has data
    series = {s: [r for r in rows if r[0] <= end] for s, rows in _PRICES.items()}
    res = run_simulation(list(EQUITIES), sectors, series, CAPITAL, all_sectors=ALL_SECTORS, top_n=n,
                         max_per_name=cap, start=start, end=end, cfg=cfg, fundamentals=_FUND,
                         scorer=random_scorer(seed, cfg) if seed is not None else None)
    bench = rulebook.performance_stats(res.benchmark_values)
    spy = rulebook.performance_stats(res.spy_values)
    return {"key": _job_key(job), "name": name, "period": period, "seed": seed, "sectors": sectors, "n": n, **res.stats,
            "turnover": res.annual_turnover, "stops": res.n_stops, "costs": res.total_costs,
            "ew_return": bench["annual_return"], "ew_sharpe": bench["sharpe"], "ew_dd": bench["max_drawdown"],
            "spy_return": spy["annual_return"], "spy_sharpe": spy["sharpe"], "spy_dd": spy["max_drawdown"],
            "warnings": res.warnings[:1]}


def _sector_job(job: Tuple[str, float, Optional[float], bool]) -> Dict[str, Any]:
    """One run of the sector-move study. `sigmas` None = no sector rule."""
    period, cap, sigmas, clean = job
    label, start, end = next(p for p in PERIODS if p[0] == period)
    cfg = dataclasses.replace(BASE, **(PRICES_ONLY if clean else {}), sector_move_sigmas=sigmas or BASE.sector_move_sigmas)
    series = {s: [r for r in rows if r[0] <= end] for s, rows in _PRICES.items()}
    res = run_simulation(list(EQUITIES), ALL_SECTORS, series, CAPITAL, all_sectors=ALL_SECTORS, top_n=TOP_N, max_per_name=cap,
                         start=start, end=end, cfg=cfg, fundamentals=_FUND, sector_rerank=sigmas is not None)
    return {"period": period, "sigmas": sigmas, "clean": clean, **res.stats, "turnover": res.annual_turnover,
            "reranks": res.n_sector_rerank, "costs": res.total_costs}


def sector_study(cache_dir: str, workers: int, out: str) -> None:
    """Does re-ranking on unusual sector moves help? This fires on EVERY unusual move
    up to the limit: the model can't be replayed historically, and live it would
    filter these. So it shows the mechanical rule, not the model's judgement."""
    prices, fund = load_prices(cache_dir), load_fundamentals(cache_dir)
    choices = (None, 2.0, 3.0)
    jobs = [(p[0], MAX_PER_NAME, s, clean) for clean in (True, False) for s in choices for p in PERIODS]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(prices, fund)) as pool:
        rows = list(pool.map(_sector_job, jobs, chunksize=1))
    periods = [p[0] for p in PERIODS]
    L = ["", "## Re-ranking on unusual sector moves (mechanical, all five sectors, 12 stocks)", "",
         "Fires on every unusual sector move, up to 2 per 30 days. The model's filter can't be replayed, so this isn't a "
         "test of its judgement. Annualised return / Sharpe.", "",
         "| Rule | " + " | ".join(periods) + " | Avg return | Avg Sharpe | Avg max DD | Avg turnover | Re-ranks per period |",
         "|---|" + "---|" * (len(periods) + 5)]
    for clean in (True, False):
        for s in choices:
            got = {r["period"]: r for r in rows if r["clean"] == clean and r["sigmas"] == s}
            name = f"{'clean: ' if clean else ''}" + (f"re-rank at {s:.0f} sigma" if s else "no sector rule")
            L.append(f"| {name} | " + " | ".join(f"{_pct(got[p]['annual_return'])} / {_num(got[p]['sharpe'])}" for p in periods)
                     + f" | {_pct(statistics.fmean(got[p]['annual_return'] for p in periods))}"
                     + f" | {_num(statistics.fmean(got[p]['sharpe'] for p in periods))}"
                     + f" | {_pct(statistics.fmean(got[p]['max_drawdown'] for p in periods))}"
                     + f" | {statistics.fmean(got[p]['turnover'] for p in periods):.1f}x"
                     + f" | {statistics.fmean(got[p]['reranks'] for p in periods):.0f} |")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# Ways to limit dip buying's trading that don't depend on when the portfolio last recalibrated.
DIP_LIMITS: Dict[str, Dict[str, float]] = {
    "slots": {"dip_replace_margin": 1.0},
    "anytime": {"dip_replace_margin": 0.0},
    "margin": {"dip_replace_margin": 0.20},
    "strict": {"dip_min_credit": 0.8, "dip_min_quality_pct": 0.75},
}


def _dip_job(job: Tuple[str, float, str, bool]) -> Dict[str, Any]:
    period, cap, mode, clean = job
    label, start, end = next(p for p in PERIODS if p[0] == period)
    cfg = dataclasses.replace(BASE, **(PRICES_ONLY if clean else {}), pullback_weight=0.0 if mode == "off" else BASE.pullback_weight,
                              **DIP_LIMITS.get(mode, {}))
    series = {s: [r for r in rows if r[0] <= end] for s, rows in _PRICES.items()}
    res = run_simulation(list(EQUITIES), ALL_SECTORS, series, CAPITAL, all_sectors=ALL_SECTORS, top_n=TOP_N, max_per_name=cap,
                         start=start, end=end, cfg=cfg, fundamentals=_FUND, dip_buys=mode == "daily" or mode in DIP_LIMITS)
    return {"period": period, "mode": mode, "clean": clean, **res.stats, "turnover": res.annual_turnover,
            "dips": res.n_dip_buys, "costs": res.total_costs}


def dip_study(cache_dir: str, workers: int, out: str) -> None:
    """Does buying dips help? 'Clean' takes quality and value out of the SCORE, but
    the dip rule still needs today's quality to say what's valuable. News is assumed
    clean: a fall with real bad news behind it would be bought here, and live the model
    would stop it. Both flatter these results."""
    prices, fund = load_prices(cache_dir), load_fundamentals(cache_dir)
    modes = (("off", "no dip rule"), ("score", "pullback credit only"),
             ("slots", "daily, open slots only"), ("anytime", "daily, replace if any better"),
             ("margin", "daily, replace only if 0.20 better (the default)"),
             ("strict", "daily, stricter dip (credit 0.8, top-quarter quality)"))
    jobs = [(p[0], MAX_PER_NAME, m, clean) for clean in (True, False) for m, _ in modes for p in PERIODS]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(prices, fund)) as pool:
        rows = list(pool.map(_dip_job, jobs, chunksize=1))
    periods = [p[0] for p in PERIODS]
    L = ["", "## Buying dips (all five sectors, 12 stocks)", "",
         "Annualised return / Sharpe. News is assumed clean and `clean` lines still use today's quality to define a "
         "valuable stock, so these results are flattered.", "",
         "| Rule | " + " | ".join(periods) + " | Avg return | Avg Sharpe | Avg max DD | Avg turnover | Dip buys per period |",
         "|---|" + "---|" * (len(periods) + 5)]
    for clean in (True, False):
        for m, label in modes:
            got = {r["period"]: r for r in rows if r["clean"] == clean and r["mode"] == m}
            L.append(f"| {'clean: ' if clean else ''}{label} | " + " | ".join(f"{_pct(got[p]['annual_return'])} / {_num(got[p]['sharpe'])}" for p in periods)
                     + f" | {_pct(statistics.fmean(got[p]['annual_return'] for p in periods))}"
                     + f" | {_num(statistics.fmean(got[p]['sharpe'] for p in periods))}"
                     + f" | {_pct(statistics.fmean(got[p]['max_drawdown'] for p in periods))}"
                     + f" | {statistics.fmean(got[p]['turnover'] for p in periods):.1f}x"
                     + f" | {statistics.fmean(got[p]['dips'] for p in periods):.0f} |")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:+.1%}"


def _num(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="backtest_report.md")
    ap.add_argument("--cache-dir", default=".backtest_cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sector-study", action="store_true", help="only the study of re-ranking on sector moves")
    ap.add_argument("--dip-study", action="store_true", help="only the study of buying dips")
    a = ap.parse_args()
    os.makedirs(a.cache_dir, exist_ok=True)
    if a.dip_study:
        dip_study(a.cache_dir, a.workers, a.out)
        return
    if a.sector_study:
        sector_study(a.cache_dir, a.workers, a.out)
        return
    prices = load_prices(a.cache_dir)
    fund = load_fundamentals(a.cache_dir)
    print(f"{len(prices)} price series, {len(fund)} fundamentals", flush=True)

    jobs = [(name, p[0], None, ALL_SECTORS, TOP_N, MAX_PER_NAME) for name in VARIANTS for p in PERIODS]
    jobs += [("default", p[0], seed, ALL_SECTORS, TOP_N, MAX_PER_NAME) for seed in NULL_SEEDS for p in PERIODS]
    jobs += [("default", p[0], None, ["tech_ai"], 5, 0.30) for p in PERIODS]
    jobs += [("default", p[0], seed, ["tech_ai"], 5, 0.30) for seed in NULL_SEEDS for p in PERIODS]
    results_path = os.path.join(a.cache_dir, "results.json")
    done: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(results_path):
        with open(results_path) as fh:
            done = {r["key"]: r for r in json.load(fh) if r.get("key")}
    todo = [j for j in jobs if _job_key(j) not in done]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already done", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(prices, fund)) as pool:
        for i, row in enumerate(pool.map(_run, todo, chunksize=1)):
            done[row["key"]] = row
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(todo)}", flush=True)
                with open(results_path, "w") as fh:
                    json.dump(list(done.values()), fh)
    rows = [done[_job_key(j)] for j in jobs]
    with open(results_path, "w") as fh:
        json.dump(rows, fh)

    periods = [p[0] for p in PERIODS]
    wide = [r for r in rows if r["sectors"] == ALL_SECTORS]
    tech = [r for r in rows if r["sectors"] == ["tech_ai"]]

    def pick(pool, name, period, seed=None):
        return next(r for r in pool if r["name"] == name and r["period"] == period and r["seed"] == seed)

    def null(pool, period, key):
        vals = [r[key] for r in pool if r["seed"] is not None and r["period"] == period and r[key] is not None]
        return statistics.median(vals) if vals else None

    L = ["# Rulebook backtest", "",
         "Real rules, real prices, net of costs. **Every absolute return here is flattered by survivorship and "
         "fundamentals lookahead**: compare within a row. The LLM isn't included. See `rulebook_backtest.py`.", ""]

    for title, pool in (("All five sectors, 12 stocks", wide), ("Tech only, 5 stocks", tech)):
        L += [f"## {title}: rules vs. baselines", "",
              "| Period | Rules ann. return | Random picks (median of 8) | Equal weight | SPY | Rules Sharpe | Random Sharpe | EW Sharpe | SPY Sharpe | Rules max DD | Random max DD | SPY max DD | Turnover | Stops |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in periods:
            r = pick(pool, "default", p)
            L.append(f"| {p} | {_pct(r['annual_return'])} | {_pct(null(pool, p, 'annual_return'))} | {_pct(r['ew_return'])} | "
                     f"{_pct(r['spy_return'])} | {_num(r['sharpe'])} | {_num(null(pool, p, 'sharpe'))} | {_num(r['ew_sharpe'])} | "
                     f"{_num(r['spy_sharpe'])} | {_pct(r['max_drawdown'])} | {_pct(null(pool, p, 'max_drawdown'))} | {_pct(r['spy_dd'])} | "
                     f"{r['turnover']:.1f}x | {r['stops']} |")
        L.append("")

    clean_base = "clean: default"
    L += ["## The clean test: prices only (no quality or value), all five sectors, 12 stocks", "",
          "This removes the fundamentals lookahead, so it is the fairest read. Survivorship still flatters every "
          "column equally.", "",
          "| Period | Rules ann. return | Random picks (median of 8) | Equal weight | SPY | Rules Sharpe | Random Sharpe | EW Sharpe | SPY Sharpe |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in periods:
        r = pick(wide, clean_base, p)
        L.append(f"| {p} | {_pct(r['annual_return'])} | {_pct(null(wide, p, 'annual_return'))} | {_pct(r['ew_return'])} | {_pct(r['spy_return'])} | "
                 f"{_num(r['sharpe'])} | {_num(null(wide, p, 'sharpe'))} | {_num(r['ew_sharpe'])} | {_num(r['spy_sharpe'])} |")
    L.append("")

    L += ["## Each rule varied, all five sectors, 12 stocks", "",
          "Annualised return / Sharpe per period; the last columns are simple averages over the periods. "
          "Lines marked `clean:` have no quality or value factor, so read those first.", "",
          "| Variant | " + " | ".join(periods) + " | Avg return | Avg Sharpe | Avg max DD | Avg turnover | Avg stops |",
          "|---|" + "---|" * (len(periods) + 5)]
    for name in VARIANTS:
        cells, rets, sharpes, dds, turns, stops = [], [], [], [], [], []
        for p in periods:
            r = pick(wide, name, p)
            cells.append(f"{_pct(r['annual_return'])} / {_num(r['sharpe'])}")
            rets.append(r["annual_return"] or 0.0); sharpes.append(r["sharpe"] or 0.0); dds.append(r["max_drawdown"] or 0.0)
            turns.append(r["turnover"]); stops.append(r["stops"])
        L.append(f"| {name} | " + " | ".join(cells) + f" | {_pct(statistics.fmean(rets))} | {_num(statistics.fmean(sharpes))} | "
                 f"{_pct(statistics.fmean(dds))} | {statistics.fmean(turns):.1f}x | {statistics.fmean(stops):.0f} |")
    L.append("")
    with open(a.out, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
