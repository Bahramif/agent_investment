"""
daytrade_lessons.py
====================
Learned rules ("lessons") for the day-trading engine: what they may say, how
they are judged, and how they are applied. Pure stdlib, no I/O.

WHAT A LESSON IS. A condition over the entry snapshot (daytrade_journal.
FEATURES — nothing else) plus a RESTRICT-ONLY action:

    {"when":   {"strategy": {"==": "orb_long"}, "gap_pct": {">": 0.04}},
     "action": {"type": "skip"}}                         # or {"type": "risk_mult", "value": 0.5}

parse_lesson() is the clamp, same philosophy as daytrade_playbook.
clamp_playbook: unknown features/operators are rejected, at most three
conditions, a risk_mult >= 1 is a no-op and rejected, one below the floor is
raised to it. Whoever proposes a lesson — Claude, the miner, a human — the
worst a lesson can do is make the engine trade LESS.

HOW A LESSON IS JUDGED (validate). On journal records it was NOT derived
from (its origin_days are excluded), using `r_virtual` — each signal's
stand-alone outcome, unaffected by slots, vetoes or the playbook:
  * enough evidence: >= min_matched_trades matches over >= min_distinct_days;
  * the matched setups lose: mean R < 0 with a z-score above BOTH the plain
    significance bar and the expected maximum z of `n_trials` pure-noise
    rules (the Bailey/Lopez de Prado deflation idea applied to rule search —
    every lesson ever proposed is a trial, so each new one faces a higher bar);
  * it holds up over time: negative mean R in a majority of walk-forward folds;
  * it doesn't throw away too much: the share of all winning R it removes is
    below max_opportunity_cost.
A lesson that passes is "validated"; it becomes "active" only by a human
(or when learning.auto_activate is on) and only while under max_active_lessons.

HOW A LESSON STAYS HONEST (update_live / review). Every signal an active
lesson filters is journaled with its virtual outcome, so its live record is
exactly "the R of the trades it stopped". Net-positive over enough of them,
or no evidence by the TTL, and it is retired — back to the config baseline,
which is never looser than the config itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from daytrade_config import KNOWN_STRATEGIES, LearningConfig
from daytrade_journal import FEATURES

OPS = ("<", "<=", ">", ">=", "==", "in")
MAX_CONDITIONS = 3
LESSON_STATUSES = ("proposed", "validated", "rejected", "active", "retired")
_EULER_GAMMA = 0.5772156649015329


class LessonError(ValueError):
    pass


@dataclass
class Lesson:
    id: str
    when: Dict[str, Dict[str, Any]]
    action: Dict[str, Any]
    hypothesis: str = ""
    status: str = "proposed"
    source: str = ""                              # "claude" | "miner" | "manual"
    created: str = ""                             # ISO date
    origin_days: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    live: Dict[str, Any] = field(default_factory=lambda: {"n": 0, "sum_r": 0.0, "wins": 0, "days": []})
    activated: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)   # audit trail of status changes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        conds = []
        for feat, cond in self.when.items():
            for op, val in cond.items():
                conds.append(f"{feat} {op} {val}")
        act = "skip" if self.action["type"] == "skip" else f"size x{self.action['value']:g}"
        return f"IF {' AND '.join(conds)} THEN {act}"

    def set_status(self, status: str, day: str, why: str, by: str = "system") -> None:
        if status not in LESSON_STATUSES:
            raise LessonError(f"unknown status {status!r}")
        self.history.append({"day": day, "from": self.status, "to": status, "why": why[:500], "by": by})
        self.status = status
        if status == "active":
            self.activated = day


# -----------------------------------------------------------------------------
# Parsing = clamping
# -----------------------------------------------------------------------------
def _num(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise LessonError(f"{where}: needs a finite number, got {v!r}")
    return float(v)


def _canonical_id(when: Dict[str, Any], action: Dict[str, Any]) -> str:
    blob = json.dumps({"when": when, "action": action}, sort_keys=True)
    return "L" + hashlib.sha1(blob.encode()).hexdigest()[:10]


def parse_lesson(raw: Dict[str, Any], min_mult: float, **meta: Any) -> Lesson:
    """Validate and normalize a proposed lesson. Raises LessonError. The id
    is a hash of the rule itself, so the same rule proposed twice is one lesson."""
    if not isinstance(raw, dict):
        raise LessonError("lesson must be an object")
    when_raw = raw.get("when")
    if not isinstance(when_raw, dict) or not when_raw:
        raise LessonError("'when' must be a non-empty object")
    if len(when_raw) > MAX_CONDITIONS:
        raise LessonError(f"at most {MAX_CONDITIONS} conditions (narrow rules over many features are curve-fits)")
    when: Dict[str, Dict[str, Any]] = {}
    for feat, cond in when_raw.items():
        if feat not in FEATURES:
            raise LessonError(f"unknown feature {feat!r} (allowed: {sorted(FEATURES)})")
        if not isinstance(cond, dict) or not cond or len(cond) > 2:
            raise LessonError(f"{feat}: condition must be an object with 1-2 operators")
        out: Dict[str, Any] = {}
        for op, val in cond.items():
            if op not in OPS:
                raise LessonError(f"{feat}: unknown operator {op!r} (allowed: {OPS})")
            if FEATURES[feat]["kind"] == "category":
                if op == "==":
                    vals = [val]
                elif op == "in" and isinstance(val, list) and val:
                    vals = list(val)
                else:
                    raise LessonError(f"{feat}: categorical features take '==' or 'in' [list]")
                bad = [v for v in vals if v not in KNOWN_STRATEGIES]
                if bad:
                    raise LessonError(f"{feat}: unknown value(s) {bad}")
                out[op] = vals[0] if op == "==" else sorted(set(vals))
            else:
                if op == "in":
                    raise LessonError(f"{feat}: numeric features take comparison operators")
                out[op] = round(_num(val, feat), 6)
        when[feat] = out

    act = raw.get("action")
    if not isinstance(act, dict) or act.get("type") not in ("skip", "risk_mult"):
        raise LessonError("action must be {'type': 'skip'} or {'type': 'risk_mult', 'value': <1}")
    if act["type"] == "skip":
        action: Dict[str, Any] = {"type": "skip"}
    else:
        v = _num(act.get("value"), "action.value")
        if v >= 1.0:
            raise LessonError("risk_mult must be < 1 — a lesson can only shrink size")
        action = {"type": "risk_mult", "value": round(max(min_mult, v), 4)}

    lesson = Lesson(id=_canonical_id(when, action), when=when, action=action,
                    hypothesis=str(raw.get("hypothesis", ""))[:500])
    for k, v in meta.items():
        setattr(lesson, k, v)
    return lesson


def lesson_from_dict(d: Dict[str, Any]) -> Lesson:
    known = set(Lesson.__dataclass_fields__)
    return Lesson(**{k: v for k, v in d.items() if k in known})


# -----------------------------------------------------------------------------
# Matching and application
# -----------------------------------------------------------------------------
def _cmp(op: str, x: Any, v: Any) -> bool:
    if op == "==":
        return x == v
    if op == "in":
        return x in v
    if x is None or isinstance(x, str):
        return False                              # a missing feature never matches (conservative: no restriction)
    return {"<": x < v, "<=": x <= v, ">": x > v, ">=": x >= v}[op]


def matches(lesson: Lesson, snapshot: Dict[str, Any]) -> bool:
    return all(_cmp(op, snapshot.get(feat), val) for feat, cond in lesson.when.items() for op, val in cond.items())


class LessonSet:
    """What the engine consults at signal time. Built from stored dicts;
    anything that no longer parses is dropped (it can only have restricted)."""

    def __init__(self, lessons: Sequence[Lesson], min_mult: float = 0.25):
        self.lessons = list(lessons)
        self.min_mult = min_mult

    @classmethod
    def from_dicts(cls, dicts: Optional[Iterable[Dict[str, Any]]], min_mult: float = 0.25) -> "LessonSet":
        out = []
        for d in dicts or ():
            try:
                parsed = parse_lesson({"when": d.get("when"), "action": d.get("action")}, min_mult)
                parsed.id = d.get("id", parsed.id)
                out.append(parsed)
            except (LessonError, AttributeError, TypeError):
                continue
        return cls(out, min_mult)

    def __len__(self) -> int:
        return len(self.lessons)

    def evaluate(self, snapshot: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """(skip, size multiplier, ids of lessons that matched). Skip wins;
        multipliers compound but never below the floor."""
        skip, mult, ids = False, 1.0, []
        for l in self.lessons:
            if matches(l, snapshot):
                ids.append(l.id)
                if l.action["type"] == "skip":
                    skip = True
                else:
                    mult *= float(l.action["value"])
        return skip, max(self.min_mult, min(1.0, mult)), ids


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def expected_max_z(n_trials: int) -> float:
    """Expected maximum of n_trials independent standard normals — the z a
    pure-noise rule search reaches by luck (the SR0 term of the Deflated Sharpe)."""
    if n_trials <= 1:
        return 0.0
    nd = NormalDist()
    return (1 - _EULER_GAMMA) * nd.inv_cdf(1 - 1 / n_trials) + _EULER_GAMMA * nd.inv_cdf(1 - 1 / (n_trials * math.e))


def usable(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("r_virtual") is not None and isinstance(r.get("snapshot"), dict)]


@dataclass
class ValidationResult:
    passed: bool
    n: int = 0
    days: int = 0
    mean_r: float = 0.0
    z: float = 0.0
    z_required: float = 0.0
    fold_wins: int = 0
    folds: int = 0
    opportunity_cost: float = 0.0
    r_saved_per_day: float = 0.0
    n_trials: int = 1
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def validate(lesson: Lesson, records: Iterable[Dict[str, Any]], cfg: LearningConfig, n_trials: int,
             folds: int = 4) -> ValidationResult:
    excluded = set(lesson.origin_days)
    pool = [r for r in usable(records) if r.get("day") not in excluded]
    matched = [r for r in pool if matches(lesson, r["snapshot"])]
    rs = [float(r["r_virtual"]) for r in matched]
    days = sorted({r["day"] for r in matched})
    res = ValidationResult(passed=False, n=len(rs), days=len(days), n_trials=max(1, n_trials))
    scale = 1.0 if lesson.action["type"] == "skip" else 1.0 - float(lesson.action["value"])
    if len(rs) < cfg.min_matched_trades:
        res.reasons.append(f"only {len(rs)} matching trades out of sample (need {cfg.min_matched_trades})")
    if len(days) < cfg.min_distinct_days:
        res.reasons.append(f"matches span {len(days)} days (need {cfg.min_distinct_days})")
    if len(rs) < 2:
        res.reasons.append("not enough data to test")
        return res

    res.mean_r = statistics.fmean(rs)
    sd = statistics.stdev(rs)
    res.z = (-res.mean_r / (sd / math.sqrt(len(rs)))) if sd > 0 else (math.inf if res.mean_r < 0 else 0.0)
    res.z_required = max(NormalDist().inv_cdf(1 - cfg.max_p_value), expected_max_z(res.n_trials))
    if res.mean_r >= 0:
        res.reasons.append(f"matched setups made money (mean {res.mean_r:+.2f}R) — skipping them would hurt")
    elif res.z < res.z_required:
        res.reasons.append(f"loss not significant: z {res.z:.2f} < {res.z_required:.2f} required after "
                           f"{res.n_trials} trials")

    # walk-forward: chronological day folds; a fold counts if it has >= 3 matches
    if days:
        size = max(1, math.ceil(len(days) / folds))
        for i in range(0, len(days), size):
            chunk = set(days[i:i + size])
            fr = [float(r["r_virtual"]) for r in matched if r["day"] in chunk]
            if len(fr) >= 3:
                res.folds += 1
                res.fold_wins += statistics.fmean(fr) < 0
    if res.folds == 0 or res.fold_wins / res.folds < cfg.min_walkforward_win_rate:
        res.reasons.append(f"inconsistent over time: losing in {res.fold_wins}/{res.folds} folds")

    all_pos = sum(max(0.0, float(r["r_virtual"])) for r in pool)
    lost_pos = sum(max(0.0, x) for x in rs) * scale
    res.opportunity_cost = lost_pos / all_pos if all_pos > 0 else 0.0
    if res.opportunity_cost > cfg.max_opportunity_cost:
        res.reasons.append(f"throws away {res.opportunity_cost:.0%} of all winning R "
                           f"(max {cfg.max_opportunity_cost:.0%})")
    pool_days = len({r["day"] for r in pool}) or 1
    res.r_saved_per_day = -sum(rs) * scale / pool_days
    res.passed = not res.reasons
    return res


# -----------------------------------------------------------------------------
# Live tracking and retirement
# -----------------------------------------------------------------------------
def update_live(lesson: Lesson, day_records: Iterable[Dict[str, Any]]) -> int:
    """Fold one day's journal into the lesson's live record. Counts the
    virtual R of every signal this lesson acted on (skipped or shrunk).
    Idempotent per day. Returns how many records were added."""
    day_records = list(day_records)
    if not day_records:
        return 0
    day = day_records[0].get("day")
    live = lesson.live
    if day in live.get("days", []):
        return 0
    hits = [r for r in usable(day_records) if lesson.id in (r.get("lesson_ids") or [])]
    for r in hits:
        live["n"] = live.get("n", 0) + 1
        live["sum_r"] = round(live.get("sum_r", 0.0) + float(r["r_virtual"]), 6)
        live["wins"] = live.get("wins", 0) + (float(r["r_virtual"]) > 0)
    live.setdefault("days", []).append(day)
    live["days"] = live["days"][-120:]
    return len(hits)


def review(lesson: Lesson, today: date, cfg: LearningConfig, min_live: int = 10) -> Optional[str]:
    """Reason to retire an ACTIVE lesson, or None."""
    if lesson.status != "active":
        return None
    n, s = lesson.live.get("n", 0), lesson.live.get("sum_r", 0.0)
    if n >= min_live and s > 0:
        return f"the {n} trades it stopped live were net winners ({s:+.2f}R)"
    if lesson.activated:
        age = (today - date.fromisoformat(lesson.activated)).days
        if age > cfg.lesson_ttl_days and n < 3:
            return f"expired: {age} days active with only {n} live matches"
    return None


# -----------------------------------------------------------------------------
# Deterministic miner — a hypothesis source that needs no LLM
# -----------------------------------------------------------------------------
def _bucket_conditions(feat: str) -> List[Dict[str, Any]]:
    edges = FEATURES[feat]["bins"]
    out: List[Dict[str, Any]] = [{"<": edges[0]}]
    out += [{">=": lo, "<": hi} for lo, hi in zip(edges, edges[1:])]
    out.append({">=": edges[-1]})
    return out


def mine_candidates(records: Iterable[Dict[str, Any]], cfg: LearningConfig, max_n: int = 3,
                    mine_frac: float = 0.5, min_loss_r: float = 0.25) -> List[Dict[str, Any]]:
    """Scan (strategy x one feature bucket) on the OLDEST `mine_frac` of days
    and return the worst as raw lesson proposals whose origin_days are those
    mining days — so validate() judges them only on the later days."""
    pool = usable(records)
    days = sorted({r["day"] for r in pool})
    if len(days) < 2:
        return []
    mine_days = set(days[: max(1, int(len(days) * mine_frac))])
    mine = [r for r in pool if r["day"] in mine_days]
    need = max(5, cfg.min_matched_trades // 2)
    scored = []
    for strat in sorted({r["snapshot"].get("strategy") for r in mine if r["snapshot"].get("strategy")}):
        for feat, spec in FEATURES.items():
            if spec["kind"] != "number":
                continue
            for cond in _bucket_conditions(feat):
                probe = Lesson("probe", {"strategy": {"==": strat}, feat: cond}, {"type": "skip"})
                rs = [float(r["r_virtual"]) for r in mine if matches(probe, r["snapshot"])]
                if len(rs) < need:
                    continue
                m = statistics.fmean(rs)
                sd = statistics.stdev(rs)
                if m > -min_loss_r or sd <= 0:
                    continue
                scored.append((-m / (sd / math.sqrt(len(rs))), strat, feat, cond, m, len(rs)))
    scored.sort(key=lambda x: -x[0])
    out = []
    for z, strat, feat, cond, m, n in scored[:max_n]:
        out.append({"when": {"strategy": {"==": strat}, feat: cond}, "action": {"type": "skip"},
                    "hypothesis": f"{strat} with {feat} in {cond} averaged {m:+.2f}R over {n} setups "
                                  f"(mining days only, z={z:.1f})",
                    "origin_days": sorted(mine_days)})
    return out


if __name__ == "__main__":
    import random
    from daytrade_config import LearningConfig as LC

    cfg = LC(min_matched_trades=30, min_distinct_days=10)

    print("TEST 1: parse = clamp — unknown features/ops/values, widening actions and bloat are rejected")
    good = parse_lesson({"when": {"strategy": {"==": "orb_long"}, "gap_pct": {">": 0.04}},
                         "action": {"type": "skip"}, "hypothesis": "big gaps fade"}, 0.25)
    assert good.id.startswith("L") and good.describe() == "IF strategy == orb_long AND gap_pct > 0.04 THEN skip"
    same = parse_lesson({"when": {"gap_pct": {">": 0.04}, "strategy": {"==": "orb_long"}}, "action": {"type": "skip"}}, 0.25)
    assert same.id == good.id, "same rule, different key order -> same id (dedupe)"
    for bad in ({"when": {"ceo_mood": {">": 1}}, "action": {"type": "skip"}},
                {"when": {"gap_pct": {"~": 1}}, "action": {"type": "skip"}},
                {"when": {"strategy": {"==": "yolo_short"}}, "action": {"type": "skip"}},
                {"when": {"gap_pct": {">": "0.04; DROP TABLE"}}, "action": {"type": "skip"}},
                {"when": {"gap_pct": {">": float("nan")}}, "action": {"type": "skip"}},
                {"when": {"gap_pct": {">": 0.01}}, "action": {"type": "risk_mult", "value": 2.0}},
                {"when": {"gap_pct": {">": 0.01}}, "action": {"type": "buy_more"}},
                {"when": {}, "action": {"type": "skip"}},
                {"when": {f: {">": 0} for f in ["gap_pct", "rel_volume", "atr_pct", "stop_pct"]}, "action": {"type": "skip"}}):
        try:
            parse_lesson(bad, 0.25)
            raise AssertionError(f"should have rejected {bad}")
        except LessonError:
            pass
    tiny = parse_lesson({"when": {"gap_pct": {">": 0.01}}, "action": {"type": "risk_mult", "value": 0.01}}, 0.25)
    assert tiny.action["value"] == 0.25
    print("✓ 9 hostile/malformed lessons rejected; risk_mult 0.01 raised to the 0.25 floor; ids dedupe")

    print("TEST 2: matching — ranges, categories, missing features never match")
    snap = {"strategy": "orb_long", "gap_pct": 0.05, "rel_volume": None}
    assert matches(good, snap)
    assert not matches(good, {**snap, "gap_pct": 0.03}) and not matches(good, {**snap, "strategy": "gap_and_go_long"})
    rng_l = parse_lesson({"when": {"rel_volume": {">=": 2.0, "<": 3.0}}, "action": {"type": "skip"}}, 0.25)
    assert not matches(rng_l, snap), "None never matches a numeric condition"
    assert matches(rng_l, {"rel_volume": 2.5}) and not matches(rng_l, {"rel_volume": 3.0})
    cat = parse_lesson({"when": {"strategy": {"in": ["orb_long", "gap_and_go_long"]}}, "action": {"type": "skip"}}, 0.25)
    assert matches(cat, snap) and not matches(cat, {"strategy": "vwap_pullback_long"})
    print("✓ ok")

    print("TEST 3: LessonSet — skip wins, multipliers compound down to the floor, garbage dicts dropped")
    half = parse_lesson({"when": {"gap_pct": {">": 0.02}}, "action": {"type": "risk_mult", "value": 0.5}}, 0.25)
    third = parse_lesson({"when": {"strategy": {"==": "orb_long"}}, "action": {"type": "risk_mult", "value": 0.3}}, 0.25)
    ls = LessonSet.from_dicts([half.to_dict(), third.to_dict(), {"when": "junk"}, {"action": None}], 0.25)
    assert len(ls) == 2
    skip, mult, ids = ls.evaluate(snap)
    assert not skip and mult == 0.25 and ids == [half.id, third.id], (skip, mult, ids)
    assert LessonSet.from_dicts([good.to_dict(), half.to_dict()]).evaluate(snap)[0] is True
    assert LessonSet([]).evaluate(snap) == (False, 1.0, [])
    print("✓ 0.5 x 0.3 = 0.15 -> floored at 0.25; skip overrides; empty set is a no-op")

    def synth(n_days=60, per_day=6, seed=0, bad_mean=-0.6):
        """orb_long with gap > 4% loses; everything else is a fair coin at +0.1R."""
        rng = random.Random(seed)
        recs = []
        for d in range(n_days):
            day = date(2026, 1, 1).toordinal() + d
            for _ in range(per_day):
                gap = rng.choice([0.01, 0.03, 0.05, 0.06])
                strat = rng.choice(["orb_long", "vwap_pullback_long"])
                losing = strat == "orb_long" and gap > 0.04
                r = rng.gauss(bad_mean if losing else 0.1, 1.0)
                recs.append({"day": date.fromordinal(day).isoformat(), "status": "taken", "r_virtual": r,
                             "snapshot": {"strategy": strat, "gap_pct": gap, "rel_volume": rng.uniform(1, 4)}})
        return recs

    print("TEST 4: validation passes a real effect out of sample and reports what it saves")
    recs = synth()
    res = validate(good, recs, cfg, n_trials=5)
    assert res.passed, res.reasons
    assert res.n >= 30 and res.mean_r < 0 and res.z >= res.z_required and res.r_saved_per_day > 0
    print(f"✓ n={res.n} over {res.days} days, mean {res.mean_r:+.2f}R, z {res.z:.1f} >= {res.z_required:.2f}, "
          f"folds {res.fold_wins}/{res.folds}, saves {res.r_saved_per_day:+.2f}R/day")

    print("TEST 5: validation rejects noise, anecdotes, winners, and in-sample evidence")
    noise = validate(good, synth(bad_mean=0.1), cfg, n_trials=5)
    assert not noise.passed and any("made money" in r or "not significant" in r for r in noise.reasons)
    few = validate(good, synth(n_days=4), cfg, n_trials=1)
    assert not few.passed and any("days" in r for r in few.reasons)
    winners = parse_lesson({"when": {"strategy": {"==": "vwap_pullback_long"}}, "action": {"type": "skip"}}, 0.25)
    assert not validate(winners, recs, cfg, 1).passed
    all_days = sorted({r["day"] for r in recs})
    insample = Lesson(good.id, good.when, good.action, origin_days=all_days)
    assert validate(insample, recs, cfg, 1).n == 0, "origin days must be excluded"
    print(f"✓ noise (z {noise.z:.2f}), 4-day anecdote, profitable setup and in-sample-only evidence all rejected")

    print("TEST 6: the multiple-testing bar rises with the number of trials")
    assert expected_max_z(1) == 0 and expected_max_z(10) < expected_max_z(100) < expected_max_z(1000)
    marginal = synth(bad_mean=-0.25, seed=3)
    r_few, r_many = validate(good, marginal, cfg, 2), validate(good, marginal, cfg, 5000)
    assert r_many.z_required > r_few.z_required
    print(f"✓ required z: {r_few.z_required:.2f} after 2 trials -> {r_many.z_required:.2f} after 5000")

    print("TEST 7: opportunity cost — a rule that removes most winners is rejected even if it 'helps'")
    broad = parse_lesson({"when": {"rel_volume": {">": 0.0}}, "action": {"type": "skip"}}, 0.25)
    rb = validate(broad, recs, cfg, 1)
    assert rb.opportunity_cost > 0.9 and not rb.passed
    print(f"✓ skip-everything throws away {rb.opportunity_cost:.0%} of winning R -> rejected")

    print("TEST 8: live tracking is idempotent per day; review retires net-positive and stale lessons")
    L = lesson_from_dict(good.to_dict())
    L.set_status("active", "2026-03-01", "validated", by="test")
    day_recs = [{"day": "2026-03-02", "r_virtual": 1.5, "lesson_ids": [L.id], "snapshot": {}},
                {"day": "2026-03-02", "r_virtual": -1.0, "lesson_ids": ["other"], "snapshot": {}}]
    assert update_live(L, day_recs) == 1 and update_live(L, day_recs) == 0
    assert L.live["n"] == 1 and L.live["sum_r"] == 1.5
    for i in range(10):
        update_live(L, [{"day": f"2026-04-{i + 1:02d}", "r_virtual": 0.4, "lesson_ids": [L.id], "snapshot": {}}])
    assert "net winners" in review(L, date(2026, 4, 20), cfg)
    stale = lesson_from_dict(good.to_dict())
    stale.set_status("active", "2026-01-01", "validated")
    assert review(stale, date(2026, 2, 1), cfg) is None
    assert "expired" in review(stale, date(2026, 3, 15), cfg)
    assert review(lesson_from_dict(good.to_dict()), date(2026, 9, 1), cfg) is None, "only active lessons retire"
    assert [h["to"] for h in L.history] == ["active"]
    print("✓ ok")

    print("TEST 9: the miner finds the planted effect on OLD days and leaves NEW days for validation")
    props = mine_candidates(recs, cfg, max_n=3)
    assert props, "should find something"
    top = parse_lesson(props[0], 0.25, origin_days=props[0]["origin_days"], source="miner")
    assert top.when["strategy"] == {"==": "orb_long"} and "gap_pct" in top.when, top.describe()
    v = validate(top, recs, cfg, n_trials=len(FEATURES) * 6 * 2)
    assert v.n > 0 and max(top.origin_days) < min(r["day"] for r in recs if r["day"] not in top.origin_days)
    print(f"✓ mined: {top.describe()} -> out-of-sample {v.mean_r:+.2f}R n={v.n} passed={v.passed}")
    print("\nAll daytrade_lessons tests passed.")
