"""
daytrade_reflection.py
=======================
Phase 2 of the failure-learning loop: after the close, Claude reads the
day's ATTRIBUTED journal plus the longer-run statistics and proposes
LESSONS — testable, restrict-only rules in daytrade_lessons' closed
language. It is a hypothesis generator, not a judge:

  * Output is a forced `propose_lessons` tool call. Every proposal goes
    through daytrade_lessons.parse_lesson (the clamp): unknown features,
    widening actions and bloated rules are dropped, not repaired.
  * Nothing proposed here changes trading. A proposal must then pass
    daytrade_lessons.validate on history it was not derived from (today and
    the recent days shown in the prompt are its origin_days, excluded), and
    only then waits for activation.
  * Claude may FLAG an active lesson for review; it cannot retire one.
    Retirement comes from the lesson's own live numbers or from a human.
  * Every proposal counts as a trial for the multiple-testing correction,
    whether or not it survives — the prompt says so, to discourage spraying.

The prompt carries only numbers, tags, strategy names and tickers — no news
text — so there is no untrusted free text to inject through. Tickers come
from the engine's own records.

Without a model (`use_llm=False`, or any failure) the deterministic miner
in daytrade_lessons is the proposer, so learning never depends on the LLM
being up — and the result records which path produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Sequence, Tuple

from daytrade_attribution import bucket_stats, summarize_day, tag_totals
from daytrade_config import DayTradingConfig
from daytrade_journal import FEATURES
from daytrade_lessons import OPS, Lesson, LessonError, mine_candidates, parse_lesson
from daytrade_playbook import PLAYBOOK_MODEL

RECENT_DAYS = 10            # live days shown in detail — and therefore excluded when validating proposals

PROPOSE_LESSONS_TOOL = {
    "name": "propose_lessons",
    "description": ("Propose at most a few restrict-only lessons for the day-trading engine, as rules over the "
                    "entry-snapshot features. Each will be backtested out of sample before it can be used; "
                    "each proposal also raises the statistical bar for all future ones."),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "3-5 sentences: what went wrong/right today and why"},
            "lessons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "when": {"type": "object",
                                 "description": "feature -> {operator: value}; at most 3 features"},
                        "action": {"type": "object",
                                   "description": "{'type': 'skip'} or {'type': 'risk_mult', 'value': 0.25..0.99}"},
                        "hypothesis": {"type": "string", "description": "why this should hold, citing the stats"},
                    },
                    "required": ["when", "action", "hypothesis"],
                },
            },
            "flag_for_review": {
                "type": "array",
                "items": {"type": "object",
                          "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
                          "required": ["id", "reason"]},
                "description": "active lessons whose evidence looks wrong — a human will look",
            },
        },
        "required": ["summary", "lessons"],
    },
}


@dataclass
class ReflectionResult:
    proposals: List[Lesson] = field(default_factory=list)
    rejected: List[Tuple[Dict[str, Any], str]] = field(default_factory=list)   # (raw, why) — malformed proposals
    flags: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""
    source: str = "none"                  # "claude" | "miner" | "none"
    error: str = ""


def _fmt_r(x: Any) -> str:
    return "-" if x is None else f"{float(x):+.2f}R"


def build_reflection_prompt(day: date, today: List[Dict[str, Any]], history: List[Dict[str, Any]],
                            lessons: Sequence[Lesson], cfg: DayTradingConfig, n_trials: int) -> str:
    lc = cfg.learning
    s = summarize_day(today)
    lines = [
        f"Date: {day.isoformat()} (after the close). You are reviewing a LONG-ONLY intraday engine "
        f"(opening-range breakouts, VWAP pullbacks, gap-and-go), flat every day.",
        "Your job: find REPEATABLE failure patterns and express each as a restrict-only rule. One bad trade is "
        "not a pattern. A stop-out on a valid setup is the cost of doing business, not a failure.",
        f"Every lesson you propose will be tested on OTHER days and needs >= {lc.min_matched_trades} matching "
        f"setups over >= {lc.min_distinct_days} days with a significant loss. {n_trials} lessons have been "
        "proposed so far; each new one raises the significance bar for all of them. Propose 0 if nothing in "
        f"the statistics below is convincing (max {lc.max_proposals_per_day}).",
        "",
        "TODAY — signals the engine saw (R = real for taken trades, virtual stand-alone outcome otherwise):",
        "  symbol | strategy | status | R | exit | MFE | tags | gap | relvol | vwapdist | tape | min",
    ]
    for r in today:
        sn = r.get("snapshot") or {}
        real = r.get("r_real") if r.get("status") == "taken" else r.get("r_virtual")
        lines.append(
            f"  {r.get('symbol')} | {r.get('strategy')} | {r.get('status')} | {_fmt_r(real)} | "
            f"{r.get('exit_reason') or r.get('exit_virtual')} | {_fmt_r(r.get('mfe_r', r.get('mfe_r_virtual')))} | "
            f"{','.join(r.get('tags') or []) or '-'} | {sn.get('gap_pct')} | {sn.get('rel_volume')} | "
            f"{sn.get('vwap_dist_pct')} | {sn.get('tape_change_pct')} | {sn.get('minutes_since_open')}")
    lines += ["", f"Today taken: {s['taken']}", f"Today by tag: {s['by_tag']}",
              f"Filters today (ghost outcomes; 'saved' = they blocked losers): {s['filters']}", ""]

    tt = tag_totals(history)
    lines.append(f"HISTORY ({len({r.get('day') for r in history})} days, backtest + live) — failure tags on taken "
                 f"trades: {tt}")
    stats = bucket_stats(history, min_n=max(10, lc.min_matched_trades // 3))
    lines.append("Worst strategy x feature buckets (virtual R, all statuses):")
    lines += [f"  {b['strategy']} | {b['feature']} {b['bucket']} | n={b['n']} | mean {b['mean_r']:+.2f}R | "
              f"win {b['win_rate']:.0%}" for b in stats[:15]]
    base = [b for b in stats if b["feature"] == "*"]
    lines.append("Baselines: " + "; ".join(f"{b['strategy']} n={b['n']} mean {b['mean_r']:+.2f}R" for b in base))
    lines.append("")

    if lessons:
        lines.append("EXISTING LESSONS (do not re-propose; flag an active one only if the numbers argue against it):")
        for l in lessons:
            live = l.live or {}
            lines.append(f"  [{l.id}] {l.status}: {l.describe()} | live: n={live.get('n', 0)} "
                         f"sum {live.get('sum_r', 0.0):+.2f}R of stopped trades")
        lines.append("")

    lines.append("RULE LANGUAGE — 'when' uses ONLY these features:")
    for f, spec in FEATURES.items():
        extra = f" (bucket edges {spec['bins']})" if spec.get("bins") else ""
        lines.append(f"  {f}: {spec['desc']}{extra}")
    lines += [f"Operators: {', '.join(OPS)} (strategy takes '==' or 'in'). Up to 2 operators per feature for a range.",
              f"Actions: {{'type': 'skip'}} or {{'type': 'risk_mult', 'value': v}} with "
              f"{lc.min_combined_risk_mult} <= v < 1. Nothing can increase size or add trades.",
              "Example: {\"when\": {\"strategy\": {\"==\": \"orb_long\"}, \"gap_pct\": {\">=\": 0.04}}, "
              "\"action\": {\"type\": \"skip\"}, \"hypothesis\": \"...\"}",
              "", "Call the propose_lessons tool."]
    return "\n".join(lines)


def call_claude(client: Any, prompt: str, model: str = PLAYBOOK_MODEL) -> Dict[str, Any]:
    response = client.messages.create(
        model=model, max_tokens=2000, tools=[PROPOSE_LESSONS_TOOL],
        tool_choice={"type": "tool", "name": "propose_lessons"},
        messages=[{"role": "user", "content": prompt}])
    for block in response.content:
        if block.type == "tool_use" and block.name == "propose_lessons":
            return dict(block.input)
    raise ValueError("Claude did not return the expected propose_lessons tool call")


def _parse_all(raws: Sequence[Any], cfg: DayTradingConfig, day: date, source: str, origin_days: List[str],
               res: ReflectionResult) -> None:
    for raw in list(raws)[: cfg.learning.max_proposals_per_day]:
        try:
            origin = sorted(set(origin_days) | set((raw or {}).get("origin_days") or []))
            res.proposals.append(parse_lesson(raw, cfg.learning.min_combined_risk_mult, source=source,
                                              created=day.isoformat(), origin_days=origin))
        except (LessonError, TypeError, AttributeError) as e:
            res.rejected.append((raw if isinstance(raw, dict) else {"raw": str(raw)[:200]}, str(e)))


def reflect(day: date, today: List[Dict[str, Any]], history: List[Dict[str, Any]], lessons: Sequence[Lesson],
            cfg: DayTradingConfig, n_trials: int, client: Any = None, use_llm: bool = True) -> ReflectionResult:
    """Never raises. Falls back to the deterministic miner."""
    res = ReflectionResult()
    live_days = sorted({r["day"] for r in history if r.get("source") == "live" and r.get("day")})
    origin = sorted(set(live_days[-RECENT_DAYS:]) | {day.isoformat()})
    if use_llm:
        try:
            if client is None:
                from claude_reasoning import _client
                client = _client()
            raw = call_claude(client, build_reflection_prompt(day, today, history, lessons, cfg, n_trials))
            res.summary = str(raw.get("summary", ""))[:2000]
            lessons_raw = raw.get("lessons") or []
            if not isinstance(lessons_raw, list):
                raise ValueError("lessons must be a list")
            _parse_all(lessons_raw, cfg, day, "claude", origin, res)
            for f in (raw.get("flag_for_review") or [])[:10]:
                if isinstance(f, dict) and f.get("id"):
                    res.flags.append({"id": str(f["id"])[:40], "reason": str(f.get("reason", ""))[:500]})
            res.source = "claude"
            return res
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"[:300]
            res.proposals, res.rejected, res.flags = [], [], []
    mined = mine_candidates(history, cfg.learning, max_n=cfg.learning.max_proposals_per_day)
    _parse_all(mined, cfg, day, "miner", [], res)
    res.source = "miner"
    res.summary = res.summary or f"deterministic miner proposed {len(res.proposals)} rule(s)"
    return res


if __name__ == "__main__":
    import random
    from types import SimpleNamespace as NS
    from daytrade_config import config_from_dict

    cfg = config_from_dict({"learning": {"max_proposals_per_day": 2}})
    day = date(2026, 3, 20)

    def hist(n_days=60, seed=0):
        rng, out = random.Random(seed), []
        for d in range(n_days):
            dd = (day - timedelta(days=n_days - d)).isoformat()
            for _ in range(6):
                gap = rng.choice([0.01, 0.03, 0.05, 0.06])
                strat = rng.choice(["orb_long", "vwap_pullback_long"])
                r = rng.gauss(-0.6 if strat == "orb_long" and gap > 0.04 else 0.1, 1.0)
                out.append({"day": dd, "source": "backtest" if d < 50 else "live", "status": "taken",
                            "strategy": strat, "r_virtual": r, "r_real": r,
                            "snapshot": {"strategy": strat, "gap_pct": gap}})
        return out

    history = hist()
    today = [{"day": day.isoformat(), "symbol": "AAA", "strategy": "orb_long", "status": "taken", "r_real": -1.0,
              "r_virtual": -1.0, "exit_reason": "stop", "mfe_r": 0.1, "tags": ["false_breakout"],
              "snapshot": {"strategy": "orb_long", "gap_pct": 0.055}},
             {"day": day.isoformat(), "symbol": "BBB", "strategy": "orb_long", "status": "off_watchlist",
              "r_virtual": 2.0, "exit_virtual": "target", "snapshot": {"strategy": "orb_long", "gap_pct": 0.01}}]

    class FakeClient:
        def __init__(self, tool_input=None, error=None):
            self.tool_input, self.error, self.calls = tool_input, error, []
            self.messages = self
        def create(self, **kw):
            self.calls.append(kw)
            if self.error:
                raise self.error
            return NS(content=[NS(type="tool_use", name="propose_lessons", input=self.tool_input)])

    print("TEST 1: prompt carries today's records, tags, bucket stats, existing lessons and the rule language")
    existing = parse_lesson({"when": {"strategy": {"==": "gap_and_go_long"}}, "action": {"type": "skip"}}, 0.25)
    p = build_reflection_prompt(day, today, history, [existing], cfg, n_trials=7)
    for needle in ("AAA", "false_breakout", "gap_pct", "Worst strategy x feature", existing.id, "7 lessons",
                   "restrict-only", "Nothing can increase size"):
        assert needle in p, needle
    print(f"✓ {len(p.splitlines())} lines")

    print("TEST 2: Claude's proposals are clamped — good kept, widening/unknown dropped, count capped")
    fc = FakeClient({"summary": "Large-gap ORBs keep failing.",
                     "lessons": [{"when": {"strategy": {"==": "orb_long"}, "gap_pct": {">=": 0.04}},
                                  "action": {"type": "skip"}, "hypothesis": "gap fades"},
                                 {"when": {"gap_pct": {">": 0.0}}, "action": {"type": "risk_mult", "value": 1.5},
                                  "hypothesis": "size up!"},
                                 {"when": {"gap_pct": {">": 0.0}}, "action": {"type": "skip"}, "hypothesis": "3rd"}],
                     "flag_for_review": [{"id": existing.id, "reason": "looks stale"}, "junk"]})
    res = reflect(day, today, history, [existing], cfg, 7, client=fc)
    assert res.source == "claude" and len(res.proposals) == 1 and len(res.rejected) == 1, (res.proposals, res.rejected)
    L = res.proposals[0]
    assert L.source == "claude" and L.created == day.isoformat() and day.isoformat() in L.origin_days
    assert len(L.origin_days) == 11, "today + the 10 most recent live days shown in the prompt are in-sample"
    assert res.flags == [{"id": existing.id, "reason": "looks stale"}]
    call = fc.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "propose_lessons"} and call["model"] == PLAYBOOK_MODEL
    print(f"✓ kept: {L.describe()}; rejected: {res.rejected[0][1][:50]}...; 3rd proposal over the cap ignored")

    print("TEST 3: model failure / garbage -> deterministic miner, never raises")
    for bad in (FakeClient(error=TimeoutError("slow")), FakeClient({"summary": "x", "lessons": "not a list"})):
        r = reflect(day, today, history, [], cfg, 7, client=bad)
        assert r.source == "miner" and r.error and r.proposals, r
        assert all(p.source == "miner" and p.origin_days for p in r.proposals)
    r = reflect(day, today, history, [], cfg, 7, use_llm=False)
    assert r.source == "miner" and not r.error
    print(f"✓ miner proposed: {r.proposals[0].describe()}")
    print("\nAll daytrade_reflection tests passed.")
