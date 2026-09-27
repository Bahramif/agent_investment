"""
daytrade_attribution.py
========================
Phase 1 of the failure-learning loop: deterministic, rule-based answers to
"WHY did this trade lose?" — and the aggregate tables the reflection step
reads. No LLM, no I/O, every threshold visible below.

A losing trade is not automatically a failure. A valid setup that hits its
stop is the expected cost of the strategy; the tags here separate that
("valid_loss") from patterns with a fix:

  stop_noise       stopped out, then price ran >= +1R within the post-exit
                   window -> the stop sat inside normal noise
  false_breakout   a breakout setup (ORB / gap-and-go) that never got going
                   (MFE < 0.3R) and was stopped -> the breakout itself failed
  weak_tape        a loss taken while the watched universe was falling
                   (tape <= -0.5% open->now) -> fighting the market
  extended_entry   entered > 1% above VWAP -> chasing
  slippage_drag    entry slippage > 2x the cost model's assumption -> execution,
                   not signal (and a sim-vs-live divergence to watch)
  eod_giveback     was >= +1R at some point, closed at the flatten below +0.5R
  valid_loss       none of the above
  (winners get "win", plus any of the descriptive tags that apply)

Ghost records (signals NOT taken) are summarized by the filter that stopped
them, using their virtual outcome: a filter whose ghosts were net losers
SAVED money; net winners means it COST money. That is the per-restriction
ablation the session-level shadow total can't give.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from daytrade_journal import FEATURES

BREAKOUT_STRATEGIES = ("orb_long", "gap_and_go_long")
TAGS = ("stop_noise", "false_breakout", "weak_tape", "extended_entry", "slippage_drag", "eod_giveback",
        "valid_loss", "win")


def _f(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def outcome_r(rec: Dict[str, Any]) -> Optional[float]:
    """Real R for taken trades, virtual R for everything else."""
    if rec.get("status") == "taken" and _f(rec.get("r_real")) is not None:
        return float(rec["r_real"])
    return _f(rec.get("r_virtual"))


def tag_record(rec: Dict[str, Any], model_slippage_bps: float = 3.0) -> List[str]:
    r = outcome_r(rec)
    if r is None:
        return []
    snap = rec.get("snapshot") or {}
    taken = rec.get("status") == "taken" and _f(rec.get("r_real")) is not None
    exit_reason = rec.get("exit_reason") if taken else rec.get("exit_virtual")
    mfe = _f(rec.get("mfe_r")) if taken else _f(rec.get("mfe_r_virtual"))
    tags: List[str] = []
    after = _f(rec.get("mfe_after_exit_r"))
    if exit_reason == "stop" and after is not None and after >= 1.0:
        tags.append("stop_noise")
    if exit_reason == "stop" and rec.get("strategy") in BREAKOUT_STRATEGIES and mfe is not None and mfe < 0.3:
        tags.append("false_breakout")
    tape = _f(snap.get("tape_change_pct"))
    if r < 0 and tape is not None and tape <= -0.005:
        tags.append("weak_tape")
    vd = _f(snap.get("vwap_dist_pct"))
    if vd is not None and vd > 0.01:
        tags.append("extended_entry")
    slip = _f(rec.get("slippage_bps"))
    if taken and slip is not None and slip > 2 * model_slippage_bps:
        tags.append("slippage_drag")
    if exit_reason in ("eod_flatten", "eod_forced") and mfe is not None and mfe >= 1.0 and r < 0.5:
        tags.append("eod_giveback")
    if r > 0:
        tags.append("win")
    elif not tags:
        tags.append("valid_loss")
    return tags


def tag_all(records: Iterable[Dict[str, Any]], model_slippage_bps: float = 3.0) -> List[Dict[str, Any]]:
    return [{**r, "tags": tag_record(r, model_slippage_bps)} for r in records]


def _acc(d: Dict[str, Any], r: float) -> None:
    d["n"] = d.get("n", 0) + 1
    d["sum_r"] = round(d.get("sum_r", 0.0) + r, 4)
    d["wins"] = d.get("wins", 0) + (r > 0)


def summarize_day(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """One day's attribution, stored on the session document."""
    taken: Dict[str, Any] = {}
    by_tag: Dict[str, Dict[str, Any]] = defaultdict(dict)
    by_strategy: Dict[str, Dict[str, Any]] = defaultdict(dict)
    filters: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for rec in records:
        r = outcome_r(rec)
        if r is None:
            continue
        if rec.get("status") == "taken":
            _acc(taken, r)
            _acc(by_strategy[rec.get("strategy", "?")], r)
            for t in rec.get("tags") or tag_record(rec):
                _acc(by_tag[t], r)
        else:
            _acc(filters[rec["status"]], r)
    for f in filters.values():
        f["verdict"] = "saved" if f["sum_r"] < 0 else "cost" if f["sum_r"] > 0 else "neutral"
    return {"taken": taken, "by_tag": dict(by_tag), "by_strategy": dict(by_strategy), "filters": dict(filters)}


def bucket_label(feature: str, value: Any) -> Optional[str]:
    spec = FEATURES.get(feature)
    if spec is None or value is None:
        return None
    if spec["kind"] == "category":
        return str(value)
    v = _f(value)
    if v is None:
        return None
    edges = spec["bins"]
    if v < edges[0]:
        return f"<{edges[0]:g}"
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v < hi:
            return f"[{lo:g},{hi:g})"
    return f">={edges[-1]:g}"


def bucket_stats(records: Iterable[Dict[str, Any]], min_n: int = 10) -> List[Dict[str, Any]]:
    """strategy x (feature bucket) -> n, mean R (virtual — comparable across
    statuses), win rate. Sorted worst mean R first."""
    cells: Dict[tuple, List[float]] = defaultdict(list)
    for rec in records:
        r = _f(rec.get("r_virtual"))
        snap = rec.get("snapshot") or {}
        if r is None or not snap:
            continue
        strat = snap.get("strategy") or rec.get("strategy")
        cells[(strat, "*", "all")].append(r)
        for feat, spec in FEATURES.items():
            if spec["kind"] != "number":
                continue
            lab = bucket_label(feat, snap.get(feat))
            if lab is not None:
                cells[(strat, feat, lab)].append(r)
    out = []
    for (strat, feat, lab), rs in cells.items():
        if len(rs) < min_n:
            continue
        out.append({"strategy": strat, "feature": feat, "bucket": lab, "n": len(rs),
                    "mean_r": round(statistics.fmean(rs), 3),
                    "win_rate": round(sum(1 for x in rs if x > 0) / len(rs), 3)})
    out.sort(key=lambda d: (d["mean_r"], -d["n"]))
    return out


def tag_totals(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for rec in records:
        if rec.get("status") != "taken":
            continue
        r = outcome_r(rec)
        if r is None:
            continue
        for t in rec.get("tags") or tag_record(rec):
            _acc(out[t], r)
    return dict(out)


if __name__ == "__main__":
    def rec(status="taken", r_real=None, r_virtual=None, strategy="orb_long", exit_reason="stop", **kw):
        base = {"status": status, "strategy": strategy, "r_real": r_real, "r_virtual": r_virtual,
                "exit_reason": exit_reason, "exit_virtual": kw.pop("exit_virtual", exit_reason),
                "snapshot": {"strategy": strategy, **kw.pop("snapshot", {})}}
        base.update(kw)
        return base

    print("TEST 1: each failure pattern gets its tag; an ordinary stop-out is a valid loss")
    cases = [
        (rec(r_real=-1.0, mfe_r=0.5, mfe_after_exit_r=1.4, strategy="vwap_pullback_long"), ["stop_noise"]),
        (rec(r_real=-1.0, mfe_r=0.1), ["false_breakout"]),
        (rec(r_real=-0.6, exit_reason="eod_flatten", mfe_r=0.2, snapshot={"tape_change_pct": -0.009}), ["weak_tape"]),
        (rec(r_real=-1.0, mfe_r=0.6, snapshot={"vwap_dist_pct": 0.015}), ["extended_entry"]),
        (rec(r_real=-1.1, mfe_r=0.6, slippage_bps=9.0), ["slippage_drag"]),
        (rec(r_real=0.2, exit_reason="eod_flatten", mfe_r=1.3), ["eod_giveback", "win"]),
        (rec(r_real=-1.0, mfe_r=0.6, strategy="vwap_pullback_long"), ["valid_loss"]),
        (rec(r_real=2.0, exit_reason="target", mfe_r=2.0), ["win"]),
        (rec(r_real=None, exit_reason="not_filled"), []),
    ]
    for r, want in cases:
        got = tag_record(r)
        assert got == want, (want, got, r)
    print(f"✓ {len(cases)} cases")

    print("TEST 2: ghosts are tagged from their VIRTUAL outcome")
    g = rec(status="off_watchlist", r_virtual=-1.0, exit_virtual="stop", mfe_r_virtual=0.05, exit_reason=None)
    assert tag_record(g) == ["false_breakout"]
    print("✓ ok")

    print("TEST 3: day summary separates what was traded from what each filter saved or cost")
    day = tag_all([
        rec(r_real=2.0, exit_reason="target", mfe_r=2.0),
        rec(r_real=-1.0, mfe_r=0.1),
        rec(status="off_watchlist", r_virtual=-1.0), rec(status="off_watchlist", r_virtual=-0.5),
        rec(status="filtered_lesson", r_virtual=1.8, lesson_ids=["L1"]),
        rec(status="vetoed", r_virtual=None),
    ])
    s = summarize_day(day)
    assert s["taken"] == {"n": 2, "sum_r": 1.0, "wins": 1}
    assert s["filters"]["off_watchlist"]["verdict"] == "saved" and s["filters"]["off_watchlist"]["sum_r"] == -1.5
    assert s["filters"]["filtered_lesson"]["verdict"] == "cost"
    assert "vetoed" not in s["filters"], "no-outcome ghosts are not counted"
    assert s["by_tag"]["false_breakout"]["n"] == 1 and s["by_strategy"]["orb_long"]["n"] == 2
    print("✓ watchlist filter saved 1.5R; lesson cost 1.8R today; tags/strategies rolled up")

    print("TEST 4: bucket labels and stats (worst first, min-n respected)")
    assert bucket_label("gap_pct", 0.05) == "[0.04,0.08)" and bucket_label("gap_pct", -0.5) == "<-0.02"
    assert bucket_label("gap_pct", 0.2) == ">=0.08" and bucket_label("gap_pct", None) is None
    assert bucket_label("strategy", "orb_long") == "orb_long" and bucket_label("nope", 1) is None
    many = [rec(r_virtual=-0.8, snapshot={"gap_pct": 0.05}) for _ in range(12)] + \
           [rec(r_virtual=0.5, snapshot={"gap_pct": 0.01}) for _ in range(12)] + \
           [rec(r_virtual=0.5, snapshot={"gap_pct": -0.05}) for _ in range(3)]
    st = bucket_stats(many, min_n=10)
    assert st[0]["feature"] == "gap_pct" and st[0]["bucket"] == "[0.04,0.08)" and st[0]["mean_r"] == -0.8
    assert not any(x["bucket"] == "<-0.02" for x in st), "3 samples < min_n"
    assert any(x["feature"] == "*" and x["n"] == 27 for x in st)
    print(f"✓ worst cell: {st[0]['strategy']} {st[0]['feature']} {st[0]['bucket']} -> {st[0]['mean_r']:+.2f}R")
    print("\nAll daytrade_attribution tests passed.")
