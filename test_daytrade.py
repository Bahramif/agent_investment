"""
test_daytrade.py
=================
End-to-end tests for the day-trading capability: whole simulated trading
days driven through daytrade_runner.run_session() — the REAL feed, engine,
risk layer, broker wrapper, store and shadow simulator — against a fake
clock and a fake Alpaca trading client that reproduces the behaviours that
matter (bracket legs, held shares, idempotency, nested lookups).

Run:  python test_daytrade.py
Per-module unit tests live in each module's own __main__ block, per this
project's convention; this file is for what only shows up when they're wired
together (restarts, kill switch, flatten deadlines, failure paths).
"""

from __future__ import annotations

import dataclasses
from bisect import bisect_right
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, Dict, List, Optional

from daytrade_broker import AlpacaDayTradeBroker, FakeDayTradeTradingClient
from daytrade_config import config_from_dict
from daytrade_engine import Playbook
from daytrade_runner import DayPlan, RunnerDeps, run_session
from daytrade_store import InMemoryDayTradeStore
from daytrade_strategies import _flat_open, _path
from intraday_data import (BAR, Bar, FakeClock, FakeIntradayDataClient, PollingBarFeed, et_datetime, to_et)

UTC = timezone.utc
DAY = date(2026, 3, 4)                              # EST: 09:00 ET == 14:00Z
NINE = et_datetime(DAY, time(9, 0)).astimezone(UTC)
SYMS = ["AAA", "BBB", "CCC"]


def at(h: int, m: int) -> datetime:
    return et_datetime(DAY, time(h, m)).astimezone(UTC)


# ---------------------------------------------------------------- market data
QUIET = (100.6, 100.7, 100.5, 100.6, 900)


def full_day(sym: str, rows: list, pad=QUIET) -> List[Bar]:
    rows = list(rows) + [pad] * (390 - len(rows))
    return [dataclasses.replace(b, symbol=sym) for b in _path(DAY, rows)]


def orb_rows(after: list) -> list:
    """ORB breakout whose signal bar is index 17 (09:47-09:48), then `after`."""
    return _flat_open(15) + [(100.0, 100.2, 99.9, 100.1, 1000)] * 2 + \
           [(100.15, 100.7, 100.1, 100.6, 4000)] + after


TARGET_HIT = (100.7, 101.9, 100.6, 101.0, 3000)         # high clears the 101.80 target by > 1 cent
SLOW_UP = [(100.62, 100.8, 100.5, 100.7, 3000)] * 5      # drifts, no exit


def quiet_symbol(sym: str, px: float = 50.0) -> List[Bar]:
    return full_day(sym, [], pad=(px, px + 0.05, px - 0.05, px, 800))


# ---------------------------------------------------------------- the world
class World:
    def __init__(self, bars: Dict[str, List[Bar]], start: datetime = NINE, close: Optional[time] = time(16, 0),
                 playbook: Optional[Playbook] = None, cfg_overrides: Optional[dict] = None,
                 reject: tuple = (), orphan: Optional[dict] = None):
        self.bars = bars
        self.clock = FakeClock(start, close=close)
        self.store = InMemoryDayTradeStore()
        self._ts = {s: [b.ts for b in bs] for s, bs in bars.items()}
        self.fake = FakeDayTradeTradingClient(self._price, equity=100_000.0, reject_symbols=reject)
        self.fake._ticked = {}
        if orphan:
            self.fake.pos[orphan["symbol"]] = {"qty": orphan["qty"], "avg": orphan["avg"]}
        self.broker = AlpacaDayTradeBroker(self.fake, dedicated=True, poll_seconds=0.3, poll_interval=0.001,
                                           release_wait=0.2, sleep=lambda s: None, now=self.clock.now)
        self.cfg = config_from_dict({"sleeve_capital": 25_000, **(cfg_overrides or {})})
        self.playbook = playbook or Playbook(watchlist=list(SYMS))
        self.sleep_hooks: List[Callable[[datetime], None]] = []
        self.feed_failures = 0
        self.plan_calls = 0

    def _price(self, sym: str) -> float:
        ts = self._ts.get(sym)
        if not ts:
            return {"ZZZ": 20.0}.get(sym, 100.0)
        i = bisect_right(ts, self.clock.now() - BAR) - 1
        return self.bars[sym][max(i, 0)].close

    def plan_day(self, day: date) -> DayPlan:
        self.plan_calls += 1
        prev = {s: 100.0 for s in SYMS}
        return DayPlan(self.playbook, Playbook(watchlist=list(SYMS)), prev, list(SYMS))

    def make_feed(self, symbols: List[str]):
        inner = PollingBarFeed(FakeIntradayDataClient(self.bars), symbols)
        world = self

        class TickingFeed:
            def poll(self, since, now):
                if world.feed_failures > 0:
                    world.feed_failures -= 1
                    raise ConnectionError("simulated data outage")
                got = inner.poll(since, now)
                for b in got:                       # markets move at the broker only for bars not yet seen
                    if b.ts > world.fake._ticked.get(b.symbol, datetime.min.replace(tzinfo=UTC)):
                        world.fake._ticked[b.symbol] = b.ts
                        world.fake.tick(b.symbol, b.high, b.low, b.ts)
                return got
        return TickingFeed()

    def _sleep(self, seconds: float) -> None:
        self.clock.advance(seconds=seconds)
        for hook in list(self.sleep_hooks):
            hook(self.clock.now())

    def deps(self) -> RunnerDeps:
        return RunnerDeps(cfg=self.cfg, store=self.store, broker=self.broker, clock=self.clock,
                          plan_day=self.plan_day, make_feed=self.make_feed, sleep=self._sleep, poll_seconds=15.0)

    def session(self) -> dict:
        return self.store.get_session(DAY) or {}


class Crash(Exception):
    pass


def ok(msg: str) -> None:
    print(f"✓ {msg}")


def standard_bars(after_breakout: list) -> Dict[str, List[Bar]]:
    return {"AAA": full_day("AAA", orb_rows(after_breakout)), "BBB": quiet_symbol("BBB"), "CCC": quiet_symbol("CCC", 75.0)}


if __name__ == "__main__":
    print("SCENARIO A: breakout -> bracket entry -> target leg fills at the broker -> trade recorded")
    w = World(standard_bars(SLOW_UP + [TARGET_HIT]))
    res = run_session(w.deps())
    s = w.session()
    assert res["status"] == "completed" and s["status"] == "completed"
    assert len(s["trades"]) == 1, s["trades"]
    t = s["trades"][0]
    assert t["symbol"] == "AAA" and t["strategy"] == "orb_long" and t["reason"] == "target"
    assert t["exit_price"] == 101.8 and t["qty"] == 49 and t["entry_price"] == 100.6
    assert abs(t["pnl"] - ((101.8 - 100.6) * 49 - t["fees"])) < 1e-6 and t["pnl"] > 50
    assert len(w.fake.submitted) == 1 and not w.fake.pos, "one buy, nothing sold manually, account flat"
    assert s["flat_verified"] and s["realized_pnl"] == round(t["pnl"], 4)
    assert s["shadow"]["entries"] == 1 and s["shadow"]["pnl"] > 0 and s["shadow"]["partial"] is False
    ok(f"1 trade: 49 AAA 100.60 -> 101.80 (target), net ${t['pnl']:.2f}; shadow also traded and won; account flat")

    print("SCENARIO B: nothing hits; the engine flattens at 15:45 — legs cancelled first, then sold once")
    w = World(standard_bars(SLOW_UP))
    res = run_session(w.deps())
    s = w.session()
    t = s["trades"][0]
    assert t["reason"] == "eod_flatten" and not w.fake.pos and s["flat_verified"]
    exit_et = to_et(datetime.fromisoformat(t["exit_ts"])).time()
    assert time(15, 44) <= exit_et <= time(15, 50), exit_et
    assert all(o.status in ("canceled", "filled") for o in w.fake.orders.values() if o.parent_id)
    sells = [c for c in w.fake.submitted if "flat" in c]
    assert len(sells) == 1, sells
    ok(f"closed at {exit_et:%H:%M} ET as eod_flatten via ONE market sell; both legs cancelled")

    print("SCENARIO C: kill switch flipped mid-day -> position flattened, later breakout NOT traded")
    bbb = full_day("BBB", [(50, 50.2, 49.9, 50, 1000)] * 15 + [(50, 50.2, 49.9, 50.1, 1000)] * 2 +
                   [(50.05, 50.7, 50.0, 50.6, 4000)] + [(50.6, 50.7, 50.5, 50.6, 900)] * 50, pad=(50.6, 50.7, 50.5, 50.6, 900))
    # move BBB's breakout to ~10:25 by front-padding quiet bars
    bbb_late = full_day("BBB", [(50, 50.2, 49.9, 50, 1000)] * 40 + [(50, 50.2, 49.9, 50, 1000)] * 55 +
                        [(50.05, 50.9, 50.0, 50.8, 9000)], pad=(50.8, 50.9, 50.7, 50.8, 900))
    bars = {"AAA": full_day("AAA", orb_rows(SLOW_UP)), "BBB": bbb_late, "CCC": quiet_symbol("CCC", 75.0)}
    w = World(bars)
    fired = {"v": False}
    def halt_at_1005(now):
        if not fired["v"] and now >= at(10, 5):
            fired["v"] = True
            w.store.set_halted(True, "operator says stop", by="test")
    w.sleep_hooks.append(halt_at_1005)
    res = run_session(w.deps())
    s = w.session()
    assert s["status"] == "halted" and s["halted"] and "kill switch" in s["halt_reason"]
    assert [t["symbol"] for t in s["trades"]] == ["AAA"] and s["trades"][0]["reason"] == "halt_flatten"
    assert "BBB" not in " ".join(w.fake.submitted), "no entry after the kill switch"
    assert not w.fake.pos
    ok("AAA closed as halt_flatten at ~10:05; BBB's 10:2x breakout was never entered; status=halted")

    print("SCENARIO D: CRASH mid-position, then restart — adopts the position, no duplicate order, exits normally")
    w = World(standard_bars(SLOW_UP + [(100.7, 100.9, 100.6, 100.8, 900)] * 40 + [TARGET_HIT]))
    def crash_at_1010(now):
        if now >= at(10, 10):
            raise Crash()
    w.sleep_hooks.append(crash_at_1010)
    try:
        run_session(w.deps())
        raise AssertionError("expected the simulated crash")
    except Crash:
        pass
    mid = w.session()
    assert mid["status"] == "running" and len(mid["open_positions"]) == 1 and len(w.fake.submitted) == 1
    assert "AAA" in w.fake.pos, "position is held at the broker while the process is dead"
    w.sleep_hooks.clear()
    res = run_session(w.deps())                                   # <- the Cloud Run Job retry
    s = w.session()
    assert res["status"] == "completed" and s["status"] == "completed"
    assert len(w.fake.submitted) == 1, f"restart must NOT re-enter: {w.fake.submitted}"
    assert len(s["trades"]) == 1 and s["trades"][0]["reason"] == "target" and s["trades"][0]["strategy"] == "recovered"
    assert s["entries_today"] == 1 and not w.fake.pos and s["flat_verified"]
    assert s["shadow"]["partial"] is True, "shadow state is not restored after a crash — must be flagged"
    assert any(e["kind"] == "adopted" for e in s["events"])
    ok("restart adopted AAA with its real bracket levels, saw the target fill, still exactly ONE buy order")

    print("SCENARIO E: market holiday -> skipped cleanly, nothing touched")
    w = World(standard_bars(SLOW_UP), close=None)
    res = run_session(w.deps())
    assert res["status"] == "skipped" and w.session()["status"] == "skipped" and not w.fake.submitted and w.plan_calls == 0
    ok("skipped before even planning the day")

    print("SCENARIO F: kill switch already ON at start -> no trading at all")
    w = World(standard_bars(SLOW_UP))
    w.store.set_halted(True, "left on overnight", by="test")
    res = run_session(w.deps())
    assert res["status"] == "halted" and not w.fake.submitted and "kill switch" in w.session()["halt_reason"]
    ok("halted at start, zero orders")

    print("SCENARIO G: risk-off playbook -> no trading, no orders")
    w = World(standard_bars(SLOW_UP + [TARGET_HIT]), playbook=Playbook(risk_on=False, watchlist=[], rationale="risk-off tape"))
    res = run_session(w.deps())
    assert res["status"] == "completed" and res["reason"] == "risk-off" and not w.fake.submitted
    assert w.session()["playbook"]["risk_on"] is False
    ok("the playbook's restriction is honored even on a textbook breakout")

    print("SCENARIO H: an ORPHAN position (no bracket legs) at start is flattened immediately")
    w = World(standard_bars([]), orphan={"symbol": "ZZZ", "qty": 10.0, "avg": 20.0})
    res = run_session(w.deps())
    assert "ZZZ" not in w.fake.pos
    assert any(e["kind"] == "orphan_position" for e in w.session()["events"])
    ok("unprotected position closed before any trading began")

    print("SCENARIO I: broker REJECTS the entry -> no position, slot freed, the day still completes")
    w = World(standard_bars(SLOW_UP + [TARGET_HIT]), reject=("AAA",))
    res = run_session(w.deps())
    s = w.session()
    assert res["status"] == "completed" and s["trades"] == [] and not w.fake.pos
    assert s["entries_today"] == 1 and any(e.get("kind") == "entry_failed" for e in s["events"])
    ok("rejection recorded as entry_failed; nothing held; counts toward the trade cap")

    print("SCENARIO J: the data feed drops out for a while -> runner keeps going, then trades normally")
    w = World(standard_bars(SLOW_UP + [TARGET_HIT]))
    w.feed_failures = 3
    res = run_session(w.deps())
    s = w.session()
    assert res["status"] == "completed" and len(s["trades"]) == 1
    ok("3 poll failures logged as data_error; session unaffected (bars are rebuilt by the next poll)")

    print("SCENARIO K: sells keep FAILING -> deadline reached -> emergency flatten of the dedicated account")
    w = World(standard_bars(SLOW_UP))
    real_submit = w.fake.submit_order
    def refuse_sells(req):
        if str(getattr(req.side, "value", req.side)).lower() == "sell":
            raise RuntimeError("broker down for sells")
        return real_submit(req)
    w.fake.submit_order = refuse_sells
    res = run_session(w.deps())
    s = w.session()
    assert res["status"] == "error" and s["status"] == "error" and s["flat_verified"] is False
    assert w.fake.emergency_called and not w.fake.pos
    assert any(e["kind"] == "not_flat" for e in s["events"])
    ok("could not sell -> status=error, EMERGENCY FLATTEN ran, account ended flat")

    print("SCENARIO L: a finished day is not re-run by a retried job")
    w = World(standard_bars(SLOW_UP + [TARGET_HIT]))
    run_session(w.deps())
    n_orders = len(w.fake.submitted)
    again = run_session(w.deps())
    assert again["reason"] == "already finished" and len(w.fake.submitted) == n_orders
    ok("second run returns 'already finished', places nothing")

    print("SCENARIO M: early close (13:00) pulls flatten forward to 12:45")
    half_bars = {}
    for sym, rows in (("AAA", orb_rows(SLOW_UP)), ("BBB", None), ("CCC", None)):
        if rows is None:
            half_bars[sym] = [b for b in quiet_symbol(sym) if to_et(b.ts).time() < time(13, 0)]
        else:
            half_bars[sym] = [b for b in full_day(sym, rows) if to_et(b.ts).time() < time(13, 0)]
    w = World(half_bars, close=time(13, 0))
    res = run_session(w.deps())
    t = w.session()["trades"][0]
    assert t["reason"] == "eod_flatten" and to_et(datetime.fromisoformat(t["exit_ts"])).time() <= time(12, 50)
    assert not w.fake.pos
    ok(f"flattened at {to_et(datetime.fromisoformat(t['exit_ts'])):%H:%M} ET on a 13:00 close")

    print("SCENARIO N: learning — the day's journal is saved with real outcomes, THEN the post-close hook runs")
    w = World(standard_bars(SLOW_UP + [TARGET_HIT]))
    seen = []
    deps = w.deps()
    deps.post_close = lambda day: seen.append((day, len(w.store.get_journal(day))))
    run_session(deps)
    j = w.store.get_journal(DAY)
    taken = [r for r in j if r["status"] == "taken"]
    assert len(taken) == 1 and taken[0]["symbol"] == "AAA" and taken[0]["exit_reason"] == "target"
    assert taken[0]["r_real"] > 1.5 and taken[0]["snapshot"]["strategy"] == "orb_long"
    assert taken[0]["exit_virtual"] == "target", "the virtual outcome agrees with what the broker did"
    assert seen == [(DAY, len(j))], "post_close runs once, after the journal exists"
    assert w.session()["trades"][0]["snapshot"]["minutes_since_open"] == 17
    ok(f"journal: {len(j)} record(s), AAA taken {taken[0]['r_real']:+.2f}R real / "
       f"{taken[0]['r_virtual']:+.2f}R virtual; hook ran after the save")

    print("SCENARIO O: an ACTIVE lesson on the live playbook skips the setup; the shadow still takes it")
    from daytrade_learning import run_post_close
    lesson = {"id": "Lorb", "status": "active", "activated": "2026-03-01", "when": {"strategy": {"==": "orb_long"}},
              "action": {"type": "skip"}}
    w = World(standard_bars(SLOW_UP + [TARGET_HIT]), playbook=Playbook(watchlist=list(SYMS), lessons=[lesson]))
    w.store.save_lesson(lesson)
    deps = w.deps()
    deps.post_close = lambda day: run_post_close(w.store, w.cfg, day, use_llm=False)
    res = run_session(deps)
    s = w.session()
    assert res["status"] == "completed" and not w.fake.submitted and s["trades"] == []
    assert s["shadow"]["entries"] == 1 and s["shadow"]["pnl"] > 0, "the no-playbook baseline is untouched by lessons"
    ghost = [r for r in w.store.get_journal(DAY) if r["status"] == "filtered_lesson"]
    assert len(ghost) == 1 and ghost[0]["lesson_ids"] == ["Lorb"] and ghost[0]["r_virtual"] > 1.5
    stored = w.store.list_lessons()[0]
    assert stored["live"]["n"] == 1 and stored["live"]["sum_r"] > 1.5, "the lesson's live record shows what it cost"
    assert s["attribution"]["filters"]["filtered_lesson"]["verdict"] == "cost"
    ok(f"no order; the skipped breakout would have made {ghost[0]['r_virtual']:+.2f}R — charged to lesson Lorb")

    print("SCENARIO P: a failing learning hook cannot change how the day ended")
    w = World(standard_bars(SLOW_UP + [TARGET_HIT]))
    deps = w.deps()
    def boom(day):
        raise RuntimeError("learning exploded")
    deps.post_close = boom
    res = run_session(deps)
    assert res["status"] == "completed" and w.session()["status"] == "completed" and len(w.session()["trades"]) == 1
    ok("hook raised; session still completed with its trade recorded")

    print("\nAll day-trading integration scenarios passed.")
