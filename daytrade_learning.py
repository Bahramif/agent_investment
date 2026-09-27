"""
daytrade_learning.py
=====================
The failure-learning loop, end to end. The runner calls run_post_close()
after the flatten; everything else is the CLI and the helpers the planner and
the UI use. Each step is in its own module with its own tests:

    journal (engine)  ->  attribution  ->  reflection  ->  validation  ->  lessons
    daytrade_journal     _attribution     _reflection     _lessons         (store)

post-close, in order:
  1. tag today's journal (why each trade lost) and store the day's attribution
     on the session document;
  2. fold today into every ACTIVE lesson's live record, and retire the ones
     whose stopped trades turned out to be net winners (or that expired);
  3. reflect: Claude (or the miner) proposes new lessons from today + history;
  4. count every NEW proposal as a trial, then validate each one out of
     sample against the whole journal (backtest + live);
  5. validated lessons wait for activation — by a human (CLI / UI), or
     automatically when learning.auto_activate is on — capped at
     learning.max_active_lessons.

FAILURE ISOLATION. run_post_close never raises into the runner, and nothing
here runs before or during the session. The worst a failure can cause is
"nothing learned today".

GOVERNANCE. Lessons only restrict (skip / shrink), are clamped on the way in
and again when the engine loads them, and never touch daytrade_risk limits.
Activation is the one human-gated step by default. Automatic retirement is
allowed because it only returns the engine to its configured baseline.

CLI
  python daytrade_learning.py list
  python daytrade_learning.py post-close --day 2026-03-04 [--no-llm]
  python daytrade_learning.py bootstrap [--no-llm]      # propose + validate from history (e.g. a saved backtest)
  python daytrade_learning.py activate L1a2b3c4d5 --by you
  python daytrade_learning.py retire   L1a2b3c4d5 --by you --why "..."
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from daytrade_attribution import summarize_day, tag_all
from daytrade_config import DayTradingConfig, load_day_trading_config
from daytrade_lessons import Lesson, lesson_from_dict, review, update_live, validate
from daytrade_reflection import reflect
from daytrade_store import DayTradeStore

# A rejected rule proposed again is re-judged on newer data at most this often.
# Re-testing the same hypothesis daily until it passes is "peeking" — a form of
# multiple testing the trial counter would not see.
REVALIDATE_AFTER_DAYS = 20


def _log(msg: str) -> None:
    print(f"[daytrade_learning] {msg}", flush=True)


def load_lessons(store: DayTradeStore) -> List[Lesson]:
    return [lesson_from_dict(d) for d in store.list_lessons()]


def active_lesson_dicts(store: DayTradeStore) -> List[Dict[str, Any]]:
    """What the planner attaches to the live playbook."""
    return [l.to_dict() for l in load_lessons(store) if l.status == "active"]


def memory_lines(store: DayTradeStore, days: int = 5) -> List[str]:
    """Short cross-day memory for the pre-market playbook prompt: recent
    attribution and the rules already applied deterministically."""
    lines: List[str] = []
    for s in store.list_sessions(limit=days):
        a = s.get("attribution")
        if not a:
            continue
        t = a.get("taken") or {}
        tags = ", ".join(f"{k} x{v['n']}" for k, v in sorted((a.get("by_tag") or {}).items()) if k != "win")
        filt = ", ".join(f"{k} {v['verdict']} {abs(v['sum_r']):.1f}R" for k, v in sorted((a.get("filters") or {}).items()))
        lines.append(f"{s.get('date')}: {t.get('n', 0)} trades, {t.get('sum_r', 0.0):+.1f}R"
                     + (f"; failures: {tags}" if tags else "") + (f"; filters: {filt}" if filt else ""))
    active = [l for l in load_lessons(store) if l.status == "active"]
    if active:
        lines.append("Learned rules ALREADY enforced by the engine (don't double-count them): "
                     + "; ".join(l.describe() for l in active))
    return lines


def _validate_and_store(store: DayTradeStore, cfg: DayTradingConfig, lesson: Lesson, history: List[Dict[str, Any]],
                        n_trials: int, day: date, n_active: int) -> Lesson:
    v = validate(lesson, history, cfg.learning, n_trials)
    lesson.validation = {**v.to_dict(), "day": day.isoformat()}
    if v.passed:
        lesson.set_status("validated", day.isoformat(), f"passed on {v.n} out-of-sample setups "
                          f"(mean {v.mean_r:+.2f}R, z {v.z:.2f}>={v.z_required:.2f})")
        if cfg.learning.auto_activate and n_active < cfg.learning.max_active_lessons:
            lesson.set_status("active", day.isoformat(), "auto_activate", by="auto")
    else:
        lesson.set_status("rejected", day.isoformat(), "; ".join(v.reasons))
    store.save_lesson(lesson.to_dict())
    return lesson


def propose_and_validate(store: DayTradeStore, cfg: DayTradingConfig, day: date, today: List[Dict[str, Any]],
                         client: Any = None, use_llm: bool = True) -> Dict[str, Any]:
    lessons = load_lessons(store)
    by_id = {l.id: l for l in lessons}
    history = store.list_journal()
    res = reflect(day, today, history, lessons, cfg, store.get_trials(), client=client, use_llm=use_llm)
    new = [p for p in res.proposals if p.id not in by_id]
    def _stale(l: Lesson) -> bool:
        last = (l.validation or {}).get("day")
        return not last or (day - date.fromisoformat(last)).days >= REVALIDATE_AFTER_DAYS
    retry = [by_id[p.id] for p in res.proposals
             if p.id in by_id and by_id[p.id].status == "rejected" and _stale(by_id[p.id])]
    n_trials = store.bump_trials(len(new))                 # counted BEFORE validation, pass or fail
    n_active = sum(1 for l in lessons if l.status == "active")
    out = {"source": res.source, "summary": res.summary, "error": res.error, "n_trials": n_trials,
           "malformed": [why for _, why in res.rejected], "proposed": [], "validated": [], "rejected": []}
    for p in retry:                                        # same hypothesis again: re-judged on newer data, no new trial
        p.origin_days = sorted(set(p.origin_days) | {day.isoformat()})
    for p in new + retry:
        _validate_and_store(store, cfg, p, history, n_trials, day, n_active)
        n_active += p.status == "active"
        out["proposed"].append(p.id)
        out["validated" if p.status in ("validated", "active") else "rejected"].append(
            {"id": p.id, "rule": p.describe(), "status": p.status, "why": p.history[-1]["why"]})
    for f in res.flags:
        l = by_id.get(f["id"])
        if l is not None and l.status == "active":
            l.history.append({"day": day.isoformat(), "from": l.status, "to": l.status,
                              "why": f"review requested by {res.source}: {f['reason']}", "by": res.source})
            store.save_lesson(l.to_dict())
    return out


def run_post_close(store: DayTradeStore, cfg: DayTradingConfig, day: date, client: Any = None,
                   use_llm: bool = True) -> Dict[str, Any]:
    """Never raises."""
    report: Dict[str, Any] = {"day": day.isoformat()}
    try:
        today = tag_all(store.get_journal(day, "live"), cfg.costs.slippage_bps)
        if today:
            store.save_journal(day, today, "live")
        report["attribution"] = summarize_day(today)
        store.save_session(day, {"attribution": report["attribution"]})

        retired = []
        for l in load_lessons(store):
            if l.status != "active":
                continue
            update_live(l, today)
            why = review(l, day, cfg.learning)
            if why:
                l.set_status("retired", day.isoformat(), why, by="auto")
                retired.append({"id": l.id, "rule": l.describe(), "why": why})
            store.save_lesson(l.to_dict())
        report["retired"] = retired

        if today:
            report["learning"] = propose_and_validate(store, cfg, day, today, client, use_llm)
        else:
            report["learning"] = {"source": "none", "summary": "no journaled signals today — nothing to reflect on"}
        store.save_session(day, {"learning": {k: v for k, v in report["learning"].items()}, "retired_lessons": retired})
    except Exception as e:                                  # learning must never break the trading job
        report["error"] = f"{type(e).__name__}: {e}"
        _log(f"post-close learning failed: {report['error']}")
    return report


def activate_lesson(store: DayTradeStore, cfg: DayTradingConfig, lesson_id: str, by: str, day: date) -> str:
    lessons = load_lessons(store)
    l = next((x for x in lessons if x.id == lesson_id), None)
    if l is None:
        return f"no lesson {lesson_id}"
    if l.status != "validated":
        return f"{lesson_id} is {l.status}; only a validated lesson can be activated"
    if sum(1 for x in lessons if x.status == "active") >= cfg.learning.max_active_lessons:
        return f"already {cfg.learning.max_active_lessons} active lessons (learning.max_active_lessons) — retire one first"
    l.set_status("active", day.isoformat(), "activated manually", by=by or "unknown")
    store.save_lesson(l.to_dict())
    return ""


def retire_lesson(store: DayTradeStore, lesson_id: str, by: str, why: str, day: date) -> str:
    l = next((x for x in load_lessons(store) if x.id == lesson_id), None)
    if l is None:
        return f"no lesson {lesson_id}"
    if l.status not in ("active", "validated"):
        return f"{lesson_id} is {l.status}; nothing to retire"
    l.set_status("retired", day.isoformat(), why or "retired manually", by=by or "unknown")
    store.save_lesson(l.to_dict())
    return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Meridian day-trading failure-learning loop")
    ap.add_argument("--config", default="mandate.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    pc = sub.add_parser("post-close")
    pc.add_argument("--day", required=True)
    pc.add_argument("--no-llm", action="store_true")
    bs = sub.add_parser("bootstrap")
    bs.add_argument("--no-llm", action="store_true")
    for name in ("activate", "retire"):
        p = sub.add_parser(name)
        p.add_argument("lesson_id")
        p.add_argument("--by", required=True)
        if name == "retire":
            p.add_argument("--why", default="")
    a = ap.parse_args(argv)

    from daytrade_store import default_store
    cfg, store, today = load_day_trading_config(a.config), default_store(), date.today()
    if a.cmd == "list":
        for l in load_lessons(store):
            v, live = l.validation or {}, l.live or {}
            print(f"{l.id}  {l.status:<9} {l.describe()}")
            print(f"    oos: n={v.get('n', '-')} mean={v.get('mean_r', '-')} z={v.get('z', '-')}/{v.get('z_required', '-')}"
                  f"   live: n={live.get('n', 0)} sum={live.get('sum_r', 0.0):+.2f}R   src={l.source}")
        print(f"trials so far: {store.get_trials()}")
        return 0
    if a.cmd == "post-close":
        print(run_post_close(store, cfg, date.fromisoformat(a.day), use_llm=not a.no_llm))
        return 0
    if a.cmd == "bootstrap":
        print(propose_and_validate(store, cfg, today, [], use_llm=not a.no_llm))
        return 0
    err = (activate_lesson(store, cfg, a.lesson_id, a.by, today) if a.cmd == "activate"
           else retire_lesson(store, a.lesson_id, a.by, a.why, today))
    print(err or "ok")
    return 1 if err else 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main(sys.argv[1:]))
    import random
    from datetime import timedelta
    from types import SimpleNamespace as NS
    from daytrade_config import config_from_dict
    from daytrade_store import InMemoryDayTradeStore

    day = date(2026, 3, 20)

    def backtest_days(store, n_days=80, seed=0):
        rng = random.Random(seed)
        for d in range(n_days):
            dd = day - timedelta(days=n_days - d)
            recs = []
            for i in range(6):
                gap = rng.choice([0.01, 0.03, 0.05, 0.06])
                strat = rng.choice(["orb_long", "vwap_pullback_long"])
                r = rng.gauss(-0.6 if strat == "orb_long" and gap > 0.04 else 0.1, 1.0)
                recs.append({"day": dd.isoformat(), "symbol": f"S{i}", "status": "taken", "strategy": strat,
                             "r_virtual": r, "r_real": r, "exit_reason": "stop" if r < 0 else "target",
                             "snapshot": {"strategy": strat, "gap_pct": gap}})
            store.save_journal(dd, recs, "backtest")

    today_recs = [{"day": day.isoformat(), "symbol": "AAA", "strategy": "orb_long", "status": "taken",
                   "r_real": -1.0, "r_virtual": -1.0, "exit_reason": "stop", "exit_virtual": "stop", "mfe_r": 0.1,
                   "snapshot": {"strategy": "orb_long", "gap_pct": 0.055}},
                  {"day": day.isoformat(), "symbol": "BBB", "strategy": "orb_long", "status": "off_watchlist",
                   "r_virtual": -0.8, "exit_virtual": "stop", "snapshot": {"strategy": "orb_long", "gap_pct": 0.05}}]
    GOOD = {"when": {"strategy": {"==": "orb_long"}, "gap_pct": {">=": 0.04}}, "action": {"type": "skip"},
            "hypothesis": "large-gap ORBs fade"}
    NOISE = {"when": {"strategy": {"==": "vwap_pullback_long"}, "gap_pct": {"<": 0.02}}, "action": {"type": "skip"},
             "hypothesis": "anecdote"}

    class FakeClient:
        def __init__(self, lessons):
            self.lessons, self.messages = lessons, self
        def create(self, **kw):
            return NS(content=[NS(type="tool_use", name="propose_lessons",
                                  input={"summary": "gaps", "lessons": self.lessons})])

    print("TEST 1: post-close tags the day, proposes, counts trials, validates — and does NOT auto-activate")
    store, cfg = InMemoryDayTradeStore(), config_from_dict({})
    backtest_days(store)
    store.save_journal(day, today_recs, "live")
    rep = run_post_close(store, cfg, day, client=FakeClient([GOOD, NOISE]))
    assert "error" not in rep, rep
    assert store.get_journal(day)[0]["tags"] == ["false_breakout"]
    assert store.get_session(day)["attribution"]["taken"]["n"] == 1
    assert rep["learning"]["n_trials"] == 2 and store.get_trials() == 2
    status = {l["id"]: l["status"] for l in store.list_lessons()}
    assert sorted(status.values()) == ["rejected", "validated"], status
    assert active_lesson_dicts(store) == [], "validated is not active without a human"
    print(f"✓ {rep['learning']['validated'][0]['rule']} validated; the anecdote rejected; 2 trials counted")

    print("TEST 2: proposing the same rules again adds no trials; activation is gated and capped")
    run_post_close(store, cfg, day, client=FakeClient([GOOD]))
    assert store.get_trials() == 2
    bad_before = next(l for l in store.list_lessons() if l["status"] == "rejected")
    for later in (day + timedelta(days=1), day + timedelta(days=REVALIDATE_AFTER_DAYS)):
        store.save_journal(later, [{**r, "day": later.isoformat()} for r in today_recs], "live")
    rep = run_post_close(store, cfg, day + timedelta(days=1), client=FakeClient([NOISE]))
    assert rep["learning"]["proposed"] == [], "a rejected rule is not re-tested the next day (no peeking)"
    rep = run_post_close(store, cfg, day + timedelta(days=REVALIDATE_AFTER_DAYS), client=FakeClient([NOISE]))
    assert rep["learning"]["proposed"] == [bad_before["id"]] and store.get_trials() == 2
    good_id = next(l["id"] for l in store.list_lessons() if l["status"] == "validated")
    bad_id = next(l["id"] for l in store.list_lessons() if l["status"] == "rejected")
    assert "only a validated" in activate_lesson(store, cfg, bad_id, "me", day)
    capped = config_from_dict({"learning": {"max_active_lessons": 0}})
    assert "already 0 active" in activate_lesson(store, capped, good_id, "me", day)
    assert activate_lesson(store, cfg, good_id, "me", day) == ""
    act = active_lesson_dicts(store)
    assert [a["id"] for a in act] == [good_id] and act[0]["history"][-1]["by"] == "me"
    print("✓ rejected can't be activated; cap enforced; audit trail records who")

    print("TEST 3: live tracking retires a lesson whose stopped trades turn out to be winners")
    for i in range(12):
        d = day + timedelta(days=i + 1)
        store.save_journal(d, [{"day": d.isoformat(), "symbol": "CCC", "strategy": "orb_long",
                                "status": "filtered_lesson", "lesson_ids": [good_id], "r_virtual": 1.2,
                                "exit_virtual": "target", "snapshot": {"strategy": "orb_long", "gap_pct": 0.05}}], "live")
        rep = run_post_close(store, cfg, d, use_llm=False)
    lesson = next(l for l in store.list_lessons() if l["id"] == good_id)
    assert lesson["status"] == "retired" and "net winners" in lesson["history"][-1]["why"], lesson["history"][-1]
    assert active_lesson_dicts(store) == []
    print(f"✓ retired: {lesson['history'][-1]['why']}")

    print("TEST 4: memory lines for the playbook prompt")
    lines = memory_lines(store)
    assert lines and "trades" in lines[0]
    print(f"✓ e.g. {lines[0]}")

    print("TEST 5: auto_activate honors the cap; a broken store never raises out of post-close")
    s2, auto = InMemoryDayTradeStore(), config_from_dict({"learning": {"auto_activate": True}})
    backtest_days(s2)
    s2.save_journal(day, today_recs, "live")
    run_post_close(s2, auto, day, client=FakeClient([GOOD]))
    assert [l["status"] for l in s2.list_lessons()] == ["active"]

    class Broken(InMemoryDayTradeStore):
        def get_journal(self, *a, **k):
            raise RuntimeError("firestore down")
    rep = run_post_close(Broken(), cfg, day)
    assert "firestore down" in rep["error"]
    print("✓ ok")

    print("TEST 6: no signals today -> no reflection, but TTL review still runs")
    s3 = InMemoryDayTradeStore()
    rep = run_post_close(s3, cfg, day, client=FakeClient([GOOD]))
    assert rep["learning"]["source"] == "none" and s3.get_trials() == 0
    print("✓ ok")
    print("\nAll daytrade_learning tests passed.")
