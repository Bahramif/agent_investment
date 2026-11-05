"""
test_rulebook.py
=================
Proves the three promises in rulebook.py: GENERIC (no rule names a stock; limits
scale with each stock's own volatility), ALIGNED (no rule can undo another) and
SMOOTH (no sudden cutoffs). Includes a walk through the real simulation on made-up
prices, and a no-alpha Monte Carlo of how often the sell rule fires by chance.

Needs no network: fundamentals are passed in.

Run:  python test_rulebook.py
"""

from __future__ import annotations

import ast
import os
import random
import re
from datetime import date, timedelta
from types import SimpleNamespace as NS

import numpy as np
import yaml

import rulebook as rb
from analyst_reasoning import InstrumentSnapshot, reason_desk
from simulate import run_simulation
from universe import EQUITIES, PROVIDER_SECTORS, SECTORS, THEMES

CFG = rb.RulebookConfig()
TODAY = date(2026, 11, 1)


def ok(msg): print(f"✓ {msg}")


def cand(name, score, **kw):
    return NS(name=name, score=score, industry=kw.get("industry", ""), eligible=kw.get("eligible", True),
              earnings_date=kw.get("earnings_date"))


def code_without_tests(path: str) -> str:
    src = open(path).read()
    return src.split('if __name__ == "__main__":')[0]


def string_literals(path: str):
    tree = ast.parse(code_without_tests(path))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False):
            docs.add(ast.get_docstring(node, clean=False))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in docs]


if __name__ == "__main__":
    print("GENERIC 1: no rule module mentions a stock from the universe")
    tickers = {s for s, _, _ in EQUITIES if len(s) >= 3}
    for path in ("rulebook.py", "analyst_reasoning.py", "mandate_allocator.py", "proactive_recal.py", "rec_evaluator.py",
                 "claude_reasoning.py", "data_ingestion.py", "strategist.py", "macro_reasoning.py"):
        for text in string_literals(path):
            hits = set(re.findall(r"\b[A-Z]{3,5}\b", text)) & tickers - {"CEO", "CFO", "SEC", "AI", "LLM"}
            hits -= {"ALL", "KEY", "BEN", "FAST", "NOW", "CASH", "IT", "GE", "CAT"}
            assert not hits, (path, hits, text[:80])
    ok("checked every string in the rule modules")

    print("GENERIC 2: universe follows one mapping; no hand-set sector score; recent IPOs wait like everyone else")
    assert all(rb.sector_of(p) in SECTORS and rb.sector_of(p) in PROVIDER_SECTORS.values() for _, p, _ in EQUITIES)
    assert all(not members and secular == 0.0 for _, members, secular in THEMES) and {t for t, _, _ in THEMES} == set(SECTORS)
    assert "SPCX" in {s for s, _, _ in EQUITIES}, "listed..."
    assert not rb.eligibility(100, 5e9, "clean", CFG)[0] and rb.eligibility(252, 5e9, "clean", CFG)[0], "...eligible only after a year"
    assert not rb.eligibility(500, 1e6, "clean", CFG)[0] and rb.eligibility(None, None, "unknown", CFG)[0]
    assert not rb.eligibility(500, 5e9, "thesis_break", CFG)[0]
    ok("same test for every stock; unknown data isn't held against a stock, a known shortfall is")

    print("GENERIC 3: the stop scales with each stock's own volatility")
    calm, wild = 0.25 / 252 ** 0.5, 0.40 / 252 ** 0.5
    assert abs(rb.stop_threshold(wild, CFG) - 0.346) < 0.01 and abs(rb.stop_threshold(calm, CFG) - 0.217) < 0.01
    assert rb.stop_breached(100 * (1 - 0.26), 100, 0.0, calm, CFG) and not rb.stop_breached(100 * (1 - 0.26), 100, 0.0, wild, CFG)
    ok(f"limit {rb.stop_threshold(calm, CFG):.1%} for a 25%-vol stock, {rb.stop_threshold(wild, CFG):.1%} for a 40%-vol one")

    print("ALIGNED 1: the stop only sells a LOSER that lags its OWN sector")
    assert rb.stop_breached(60, 100, 0.0, 0.02, CFG), "down 40%, sector flat"
    assert not rb.stop_breached(60, 100, -0.40, 0.02, CFG), "down with its sector"
    assert not rb.stop_breached(110, 100, 0.60, 0.02, CFG), "above cost: a winner that lags isn't a loser"
    assert not rb.stop_breached(80, 100, None, 0.02, CFG) and not rb.stop_breached(80, 100, 0.0, None, CFG), "can't judge: no stop"
    ok("ok")

    print("ALIGNED 2: a pullback only earns credit in an uptrend; the factors can't pull opposite ways")
    vol = 0.02
    sig_m = rb.monthly_sigma(vol)
    good = rb.pullback_credit(0.30, -1.8 * sig_m, 0.0, vol, "clean")
    assert good > 0.9, good
    assert rb.pullback_credit(-0.05, -1.8 * sig_m, 0.0, vol, "clean") == 0.0, "trend down: no credit"
    assert rb.pullback_credit(None, -1.8 * sig_m, 0.0, vol, "clean") == 0.0, "no history: no credit"
    assert rb.pullback_credit(0.30, -5 * sig_m, 0.0, vol, "clean") == 0.0, "a crash isn't a discount"
    assert rb.pullback_credit(0.30, -0.1 * sig_m, 0.0, vol, "clean") == 0.0, "not a pullback yet"
    assert rb.pullback_credit(0.30, -1.8 * sig_m, -0.15, vol, "clean") == 0.0, "still falling hard"
    for news in ("unknown", "negative", "thesis_break"):
        assert rb.pullback_credit(0.30, -1.8 * sig_m, 0.0, vol, news) == 0.0, news
    ok("trend, depth, steadiness and clean news are all required")

    print("ALIGNED 3: over random books, selection never buys what another rule forbids")
    rng = random.Random(7)
    for _ in range(2000):
        names = [f"S{i}" for i in range(rng.randint(5, 40))]
        n = rng.randint(1, 12)
        cands = [cand(s, rng.random(), eligible=rng.random() > 0.1, industry=rng.choice(["", "a", "b", "c", "d"]),
                      earnings_date=(TODAY + timedelta(days=rng.randint(0, 9))).isoformat() if rng.random() < 0.2 else None)
                 for s in names]
        held = set(rng.sample(names, min(n, len(names))))
        blocked = set(rng.sample(names, rng.randint(0, 3))) - held
        out = rb.select_names(cands, n, held, blocked, CFG, TODAY)
        by = {c.name: c for c in cands}
        assert len(out) <= n and len(out) == len(set(out))
        for s in out:
            assert s not in blocked and by[s].eligible
            if s not in held:
                assert not rb.earnings_imminent(by[s].earnings_date, TODAY, CFG), "a new buy reporting imminently"
    ok("2,000 random books: no blocked, ineligible or about-to-report new buy")

    print("ALIGNED 4: the keep band keeps a slipping holding; a held stock outside it goes")
    cands = [cand(f"S{i:02d}", 1.0 - i * 0.01) for i in range(30)]
    assert "S07" in rb.select_names(cands, 5, {"S07"}, (), CFG, TODAY), "8th of 30, top 2x5: stays"
    assert "S15" not in rb.select_names(cands, 5, {"S15"}, (), CFG, TODAY), "16th: outside the band"
    assert rb.select_names(cands, 5, {f"S{i:02d}" for i in range(5, 10)}, (), CFG, TODAY) == [f"S{i:02d}" for i in range(5, 10)], "no churn"
    ok("ok")

    print("ALIGNED 5: the sub-industry limit: selection spreads first; weights always obey it")
    cands = [cand(f"A{i}", 1.0 - i * 0.01, industry="semis") for i in range(8)] + [cand(f"B{i}", 0.5 - i * 0.01, industry=f"i{i}") for i in range(8)]
    out = rb.select_names(cands, 5, (), (), CFG, TODAY)
    assert sum(1 for s in out if s.startswith("A")) == 2 and len(out) == 5, out
    for _ in range(2000):
        n = rng.randint(1, 15)
        w = {f"S{i}": rng.random() for i in range(n)}
        t = sum(w.values())
        w = {k: v / t for k, v in w.items()}
        ind = {k: rng.choice(["a", "b", "c", ""]) for k in w}
        beta = {k: rng.choice([0.5, 1.0, 1.5, 2.5, None]) for k in w}
        cap = rng.choice([0.1, 0.2, 0.3, 1.0])
        lim, _ = rb.apply_portfolio_limits(w, ind, beta, cap, CFG)
        assert not rb.limit_breaches(lim, ind, beta, CFG), (lim, ind, beta)
        assert sum(lim.values()) <= 1.0 + 1e-9 and min(lim.values(), default=0) >= 0
    ok("2,000 random books: the risk check's own test always passes after the limits; the rest is cash")

    print("ALIGNED 6: anything sold is blocked for 30 days, whatever sold it; a targeted exit touches only that stock")
    p = {"holdings": {"KEEP": {"shares": 1}}, "pending_order_ids": {},
         "rebalance_history": [{"date": "2026-10-20", "trigger": "scheduled", "orders": [{"symbol": "OLD", "side": "SELL"}, {"symbol": "KEEP", "side": "SELL"}]},
                               {"date": "2026-10-25", "trigger": "manual", "orders": [{"symbol": "RJCT", "side": "SELL", "result": "REJECTED"}]},
                               {"date": "2026-08-01", "trigger": "proactive", "orders": [{"symbol": "AGED", "side": "SELL"}]},
                               {"date": "2026-10-28", "trigger": "proactive", "orders": [], "excluded": ["NOTHELD"]}]}
    blocked = rb.reentry_blocked(p, TODAY, CFG)
    assert blocked == {"OLD": "2026-11-19", "NOTHELD": "2026-11-27"}, blocked
    t = rb.targeted_weights({"A": 0.3, "B": 0.2, "C": 0.1}, ["B"], ["A", "B", "D", "E"])
    assert t == {"A": 0.3, "C": 0.1, "D": 0.2}, t
    assert rb.targeted_weights({"A": 0.5, "B": 0.5}, ["B"], []) == {"A": 0.5}, "no replacement: the slot stays cash"
    ok("ok")

    print("GENERIC 4: an unusual sector move is judged in that sector's own volatility, up or down, today or over a week")
    assert rb.unusual_sector_move(-0.03, None, 0.01, CFG) and rb.unusual_sector_move(0.03, None, 0.01, CFG)
    assert not rb.unusual_sector_move(-0.03, None, 0.02, CFG), "ordinary for a wilder sector"
    assert rb.unusual_sector_move(-0.001, -0.05, 0.01, CFG), "a slide over 5 days counts"
    assert not rb.unusual_sector_move(None, None, 0.01, CFG) and not rb.unusual_sector_move(-0.5, -0.5, None, CFG), "can't judge: no signal"
    ok("ok")

    print("SMOOTH: no cliffs: a tiny change in any input moves the pullback credit by a tiny amount")
    for _ in range(20000):
        mom, dd, p5, v = rng.uniform(-0.2, 0.6), rng.uniform(-0.6, 0), rng.uniform(-0.2, 0.1), rng.uniform(0.01, 0.05)
        c0 = rb.pullback_credit(mom, dd, p5, v, "clean")
        for args in ((mom + 1e-3, dd, p5, v), (mom, dd - 1e-3, p5, v), (mom, dd, p5 - 1e-3, v)):
            assert abs(rb.pullback_credit(*args, "clean") - c0) < 0.08, (args, c0)
    ok("worst single-step change stays small over 20,000 random inputs (the old dip score jumped 0.70 between -3.9% and -4.1%)")

    print("DIPS: only a valuable stock on a real dip; every other rule still applies")
    def dip_snap(name, quality=0.9, **kw):
        base = dict(momentum_1m=-0.05, news_sentiment=0.0, quality_score=quality, volatility=0.02, momentum_12_1=0.30,
                    earnings_yield=0.05, drawdown_6m=-1.8 * rb.monthly_sigma(0.02), news_status="clean")
        return InstrumentSnapshot(name, "equities.tech_ai", **{**base, **kw})
    desk = [dip_snap("GOOD"), dip_snap("JUNK", quality=0.1), dip_snap("LOSS", earnings_yield=-0.02), dip_snap("KNIFE", pullback_5d=-0.15),
            dip_snap("CRASH", drawdown_6m=-5 * rb.monthly_sigma(0.02)), dip_snap("DOWN", momentum_12_1=-0.2), dip_snap("NOTYET", drawdown_6m=-0.01),
            dip_snap("YOUNG", history_bars=50), dip_snap("SOON", earnings_date="2026-11-02"), dip_snap("HELD"), dip_snap("SOLD"),
            dip_snap("MID", quality=0.5)]
    scores = rb.composite_scores(desk, None, CFG)
    got = rb.dip_candidates(desk, scores, {"HELD"}, {"SOLD"}, TODAY, CFG)
    assert "GOOD" in got and not set(got) & {"JUNK", "LOSS", "KNIFE", "CRASH", "DOWN", "NOTYET", "YOUNG", "SOON", "HELD", "SOLD"}, got
    assert rb.composite_scores([dip_snap("D")], None, CFG)["D"][1] > 0.5 and CFG.pullback_weight > 0
    ok(f"only {got} qualify")

    print("DIPS: how it's funded: an open slot with cash, else the lowest scorer if the pick beats it; limits cut it back")
    cur = {"A": 0.3, "B": 0.2}
    assert rb.dip_trade("N", 0.9, cur, {"A": 0.5, "B": 0.2}, 3, 0.5, 0.3, {}, {}, CFG) == (None, 1 / 3 if 1 / 3 < 0.3 else 0.3)
    assert rb.dip_trade("N", 0.9, cur, {"A": 0.5, "B": 0.2}, 2, 0.5, 0.3, {}, {}, CFG) == ("B", 0.2), "a full book replaces its lowest scorer"
    assert rb.dip_trade("N", 0.1, cur, {"A": 0.5, "B": 0.2}, 2, 0.5, 0.3, {}, {}, CFG) is None, "only if the pick scores higher"
    never = rb.RulebookConfig(dip_replace_margin=1.0)
    assert rb.dip_trade("N", 0.9, cur, {"A": 0.5, "B": 0.2}, 2, 0.5, 0.3, {}, {}, never) is None, "1 = never replace"
    assert rb.dip_trade("N", 0.9, cur, {"A": 0.5, "B": 0.2}, 3, 0.5, 0.3, {}, {}, never)[0] is None, "an open slot still works"
    assert rb.dip_trade("N", 0.9, cur, {"A": 0.5, "B": 0.2}, 3, 0.0, 0.3, {}, {}, CFG) == ("B", 0.2), "an open slot with no cash"
    assert rb.dip_trade("N", 0.9, {"A": 0.35, "B": 0.2}, {"A": 0.5, "B": 0.2}, 2, 0.5, 0.3, {"A": "x", "N": "x"}, {}, CFG) is None, "the sub-industry is full"
    ok("ok")

    print("SCORE: ranks inside each sector; a sector's scale doesn't decide across sectors; the LLM is one input")
    def snap(name, sector, mom, q, ey):
        return InstrumentSnapshot(name, f"equities.{sector}", momentum_1m=0.0, news_sentiment=0.0, quality_score=q,
                                  volatility=0.02, momentum_12_1=mom, earnings_yield=ey)
    desk = [snap("T1", "tech_ai", 1.2, 0.9, 0.04), snap("T2", "tech_ai", 0.8, 0.8, 0.03),
            snap("E1", "energy", 0.05, 0.3, 0.09), snap("E2", "energy", 0.01, 0.2, 0.08)]
    s = {x.name: x.score for x in reason_desk(desk, CFG)}
    assert s["E1"] == s["T1"] and s["E2"] == s["T2"], ("the best of each sector scores alike", s)
    with_llm = rb.composite_scores(desk, {"T1": 0.0, "T2": 1.0, "E1": 0.5, "E2": 0.5}, CFG)
    assert with_llm["T1"][0] < s["T1"] and with_llm["T2"][0] > s["T2"] and with_llm["T1"][0] > with_llm["T2"][0], \
        "the model shifts a score but can't overturn every other factor at once"
    missing = rb.percentile_ranks({"A": 1.0, "B": 2.0, "C": None, "D": 3.0})
    assert missing == {"A": 0.0, "B": 0.5, "C": 0.5, "D": 1.0}, missing
    ok("ok")

    print("CONFIG: mandate.yaml's rulebook block loads, and is complete; typos are errors")
    raw = yaml.safe_load(open("mandate.yaml"))
    assert set(raw["rulebook"]) == set(rb.RulebookConfig.__dataclass_fields__), "every rule is stated in one place"
    assert rb.config_from_dict(raw["rulebook"]) == rb.default_config() and set(raw["buckets"]) == {"equities"}
    for bad in ({"stop_sigma": 1.0}, {"max_industry_weight": 1.5}, {"keep_rank_multiple": 0.5}, {"weight_momentum": -1}):
        try:
            rb.config_from_dict(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    ok("ok")

    print("PACKAGING: both deployments ship everything the rules import, and the config")
    def imports_of(module, seen):
        if module in seen or not os.path.exists(module + ".py"):
            return seen
        seen.add(module)
        for node in ast.walk(ast.parse(code_without_tests(module + ".py"))):    # what runs, not self-tests
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                imports_of(name.split(".")[0], seen)
        return seen
    docker = set(re.findall(r"[\w]+\.(?:py|yaml)", open("Dockerfile").read()))
    assert not {m + ".py" for m in imports_of("web_ui", set())} - docker, "Cloud Run image is missing a module"
    assert "mandate.yaml" in docker
    agent = set(re.findall(r'"([\w]+\.(?:py|yaml))"', open("vertex_deploy.py").read().split("_EXTRA_PACKAGES = [")[1].split("]")[0]))
    assert not {m + ".py" for m in imports_of("vertex_deploy", set())} - agent - {"web_ui.py"}, "agent is missing a module"
    assert "mandate.yaml" in agent
    ok("Dockerfile and _EXTRA_PACKAGES are complete")

    print("SIMULATION: the real rules on made-up prices")
    gen = np.random.default_rng(3)
    start_day = date(2023, 1, 2)
    days = []
    d = start_day
    while len(days) < 650:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)
    market = gen.normal(0.0004, 0.01, len(days))
    syms = [f"S{i:02d}" for i in range(24)]

    def walk(ret):
        return [(day, float(p)) for day, p in zip(days, 100 * np.cumprod(1 + ret))]
    series = {"SPY": walk(market)}
    for i, s_ in enumerate(syms):
        series[s_] = walk(market + gen.normal(0.0002 * (i % 3 - 1), 0.018, len(days)))
    uni = [(s_, f"equities.{'tech_ai' if i % 2 else 'energy'}", 0.5) for i, s_ in enumerate(syms)]
    fund = {s_: {"quality": 0.2 + 0.03 * i, "eps": 5.0, "industry": f"ind{i % 6}", "avg_volume": 5e6} for i, s_ in enumerate(syms)}
    kw = dict(universe=uni, sectors=["tech_ai", "energy"], initial_capital=100_000.0, all_sectors=list(SECTORS),
              top_n=6, max_per_name=0.30, start=days[260], fundamentals=fund)
    res = run_simulation(price_series=series, **kw)
    assert res.n_trades > 0 and res.total_costs > 0 and len(res.dates) == len(days) - 260
    free = run_simulation(price_series=series, cfg=rb.RulebookConfig(cost_bps=0.0), **kw)
    assert free.final_value > res.final_value, "costs are charged"
    ok(f"{res.n_trades} trades, {res.n_stops} stops, {res.annual_turnover:.1f}x turnover, ${res.total_costs:,.0f} costs")

    print("SIMULATION: no lookahead: changing later prices can't change earlier decisions")
    cut = 400
    shocked = {s_: [(day, p * (3.0 if i >= cut else 1.0)) for i, (day, p) in enumerate(rows)] for s_, rows in series.items()}
    other = run_simulation(price_series=shocked, **kw)
    before = cut - 260
    assert res.strategy_values[:before] == other.strategy_values[:before]
    assert [w for dt, w in zip(res.rebalance_dates, res.rebalance_weights) if dt < days[cut]] == \
           [w for dt, w in zip(other.rebalance_dates, other.rebalance_weights) if dt < days[cut]]
    ok("identical up to the day the future changes")

    print("SIMULATION: nothing sold comes back within the block; limits hold at every rebalance")
    holdings: dict = {}
    sold: dict = {}
    for dt, actions in zip(res.rebalance_dates, res.rebalance_actions):
        for sym, (side, _) in actions.items():
            if side == "BUY" and sym in sold and sym not in holdings:
                assert (date.fromisoformat(dt) - date.fromisoformat(sold[sym])).days >= CFG.reentry_block_days, (sym, sold[sym], dt)
            if side == "BUY":
                holdings[sym] = True
        target = res.rebalance_weights[res.rebalance_dates.index(dt)]
        for sym in list(holdings):
            if sym not in target:
                holdings.pop(sym)
                sold[sym] = dt
    scheduled = [w for w, t in zip(res.rebalance_weights, res.rebalance_tilts) if t]   # a targeted exit leaves holdings alone
    for w in scheduled:
        ind = {s_: fund[s_]["industry"] for s_ in w}
        assert not rb.limit_breaches(w, ind, {}, CFG), w
    ok(f"checked {len(res.rebalance_dates)} rebalances, {len(scheduled)} of them scheduled")

    print("ALIGNED 7: a replacement bought on a targeted exit obeys the limits; the rest stays cash")
    ind = {"A": "x", "B": "y", "N": "x", "M": "z"}
    t = rb.targeted_weights({"A": 0.30, "B": 0.30, "C": 0.40}, ["C"], ["N"], ind, {}, CFG)
    assert abs(t["N"] - 0.05) < 1e-9, t
    t = rb.targeted_weights({"A": 0.30, "B": 0.30, "C": 0.40}, ["C"], ["M"], ind, {"A": 1.5, "B": 1.5, "M": 3.0}, CFG)
    assert abs(t["M"] - (1.5 - 0.9) / 3.0) < 1e-9 and t["A"] == 0.30, t
    assert "N" not in rb.targeted_weights({"A": 0.36, "C": 0.2}, ["C"], ["N"], ind, {}, CFG), "already over: no new buy in it"
    ok("ok")

    print("SIMULATION: daily dip buys run, cost money, are off by default and don't look ahead")
    dipped = run_simulation(price_series=series, dip_buys=True, **kw)
    assert res.n_dip_buys == 0 and dipped.n_trades >= res.n_trades
    other = run_simulation(price_series=shocked, dip_buys=True, **kw)
    assert dipped.strategy_values[:before] == other.strategy_values[:before], "identical until the future changes"
    ok(f"{dipped.n_dip_buys} dip buys, {dipped.n_trades} trades vs {res.n_trades}")

    print("SIMULATION: re-ranking on sector moves stays within the early-trade limit and costs show up")
    moved = run_simulation(price_series=series, sector_rerank=True, **kw)
    assert moved.n_sector_rerank > 0 and moved.n_trades > res.n_trades, (moved.n_sector_rerank, moved.n_trades, res.n_trades)
    assert moved.n_sector_rerank <= (len(res.dates) / 21 + 1) * 2 * 1.5
    assert res.n_sector_rerank == 0, "off by default"
    ok(f"{moved.n_sector_rerank} re-ranks, {moved.n_trades} trades vs {res.n_trades}")

    print("BY CHANCE: with no alpha at all, how often does each sell rule fire on one stock in a month?")
    paths, length = 20000, 21
    for label, vol, rho in (("40% vol", 0.025, 0.5), ("25% vol", 0.016, 0.35)):
        m = gen.standard_normal((paths, length))
        stock = vol * (rho ** 0.5 * m + (1 - rho) ** 0.5 * gen.standard_normal((paths, length)))
        sector = vol * rho ** 0.5 * m                                   # a large sector: its stocks' shared part
        own = np.cumprod(1 + stock, axis=1) - 1
        sec = np.cumprod(1 + sector, axis=1) - 1
        old = ((own <= -0.05) | (stock <= -0.05)).any(axis=1).mean()
        new = ((own < 0) & (own - sec <= -rb.stop_threshold(vol, CFG))).any(axis=1).mean()
        print(f"  {label}: old flat 5% rule {old:.0%} of stocks per month; new stop {new:.1%}")
        assert new < 0.10 < old, (new, old)
    ok("the stop fires on chance moves about an order of magnitude less often")
    print("\nAll rulebook tests passed.")
