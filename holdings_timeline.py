"""
holdings_timeline.py
=====================
Turns a portfolio's launch holdings + recalibration history into what the
Portfolio page shows: what it STARTED with, how its holdings changed at each
recalibration, and a short, scannable summary per recalibration (type, what
changed, why, the reviewer's verdict) with the long text kept for an
expandable detail.

Holdings after each recalibration come from the event's own `holdings_after`
(recorded since this module exists); for older events they are reconstructed
by applying that event's orders to the previous holdings — exact for filled
orders, an approximation when an order filled later or partially.

Pure functions, no I/O.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

TYPE_LABEL = {"scheduled": "📅 Scheduled", "proactive": "📰 Mid-day", "manual": "👆 Manual"}
EXECUTED = {"FILLED", "PARTIAL", "TIMEOUT", ""}     # TIMEOUT = accepted, filled later (reconciled into the ledger)


def first_sentence(text: Any, limit: int = 140) -> str:
    """The first sentence (or clause) of `text`, at most `limit` chars."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return ""
    m = re.match(r"(.+?[.!?])(\s|$)", s)
    out = m.group(1) if m else s
    if len(out) > limit:
        out = out[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:—-") + "…"
    return out


def _shares(h: Any) -> float:
    return float((h or {}).get("shares") or 0) if isinstance(h, dict) else float(h or 0)


def apply_orders(before: Dict[str, float], orders: List[Dict[str, Any]]) -> Dict[str, float]:
    after = dict(before)
    for o in orders or []:
        if str(o.get("result", "")).upper() not in EXECUTED:
            continue
        sym, qty = o.get("symbol"), float(o.get("qty") or 0)
        side = str(o.get("side", "")).upper()
        after[sym] = after.get(sym, 0.0) + (qty if side == "BUY" else -qty if side == "SELL" else 0.0)
        if after[sym] <= 1e-9:
            after.pop(sym, None)
    return after


def summarize_event(ev: Dict[str, Any], before: Dict[str, float], after: Dict[str, float]) -> Dict[str, Any]:
    exited = sorted(s for s in before if s not in after)
    entered = sorted(s for s in after if s not in before)
    resized = sorted(s for s in after if s in before and abs(after[s] - before[s]) > 1e-9)
    review = ev.get("review") or {}
    orders = review.get("orders") or {}
    objections = [(s, first_sentence(o.get("reason"), 110)) for s, o in sorted(orders.items())
                  if o.get("verdict") == "object"]
    concerns = [(s, first_sentence(o.get("reason"), 110)) for s, o in sorted(orders.items())
                if o.get("verdict") == "concern"]
    blocked = [(s, first_sentence(why, 110)) for s, why in sorted((review.get("vetoed") or {}).items())]
    trigger = ev.get("trigger") or ""
    reason = str(ev.get("reason") or "")
    return {
        "date": ev.get("date"), "at": ev.get("at"),
        "type": TYPE_LABEL.get(trigger, "Not recorded"), "trigger": trigger,
        "exited": exited, "entered": entered, "resized": resized,
        "holds": sorted(after), "n_holds": len(after),
        "no_change": not (exited or entered or resized),
        "errors": len(ev.get("errors") or []),
        "why_short": first_sentence(reason) if reason else ("Regular schedule" if trigger == "scheduled"
                                                           else "Recalibrate Now clicked" if trigger == "manual"
                                                           else ""),
        "why_full": reason if len(reason) > len(first_sentence(reason)) else "",
        "grade": review.get("grade"),
        "review_short": first_sentence(review.get("summary"), 120),
        "review_full": str(review.get("summary") or "") if len(str(review.get("summary") or "")) > 120 else "",
        "blocked": blocked, "objections": objections, "concerns": concerns,
        "value_before": ev.get("portfolio_value_before"),
        "orders": ev.get("orders") or [],
    }


def build_timeline(p: Dict[str, Any]) -> Dict[str, Any]:
    """{"initial": [...], "events": [newest first], "summary": {...}}"""
    initial = {s: _shares(h) for s, h in (p.get("initial_holdings") or {}).items() if _shares(h) > 0}
    known_start = bool(initial)
    holdings = dict(initial)
    events = []
    exited_on: Dict[str, str] = {}
    for ev in p.get("rebalance_history") or []:
        before = dict(holdings)
        if ev.get("holdings_after") is not None:
            after = {s: _shares(h) for s, h in ev["holdings_after"].items() if _shares(h) > 0}
        else:
            after = apply_orders(before, ev.get("orders") or [])
        e = summarize_event(ev, before, after) if (known_start or ev.get("holdings_after") is not None) \
            else summarize_event(ev, {}, {})
        if not known_start and ev.get("holdings_after") is None:
            # no launch record: show the orders, not a (wrong) reconstructed holdings diff
            e.update(exited=[], entered=[], resized=[], holds=[], n_holds=None, no_change=not ev.get("orders"),
                     unknown_holdings=True)
        for s in e["exited"]:
            exited_on.setdefault(s, ev.get("date"))
        for s in e["entered"]:
            exited_on.pop(s, None)
        events.append(e)
        holdings = after
    current = {s: _shares(h) for s, h in (p.get("holdings") or {}).items() if _shares(h) > 0}
    launch_rows = []
    for s, h in sorted((p.get("initial_holdings") or {}).items()):
        launch_rows.append({"symbol": s, "shares": _shares(h),
                            "entry_price": (h or {}).get("entry_price") if isinstance(h, dict) else None,
                            "now_shares": current.get(s), "sold_on": None if s in current else exited_on.get(s)})
    counts: Dict[str, int] = {}
    for e in events:
        counts[e["trigger"] or "unrecorded"] = counts.get(e["trigger"] or "unrecorded", 0) + 1
    return {"initial": launch_rows, "events": events[::-1],
            "summary": {"n": len(events), "by_type": counts, "started": sorted(initial), "now": sorted(current),
                        "kept": sorted(set(initial) & set(current))}}


if __name__ == "__main__":
    print("TEST 1: first sentences are short and clean")
    assert first_sentence("Rule — the portfolio is down 11.5% since its last recalibration. Rotating now.") == \
        "Rule — the portfolio is down 11.5% since its last recalibration."
    long = "word " * 80
    fs = first_sentence(long, 50)
    assert len(fs) <= 50 and fs.endswith("…") and first_sentence("") == "" and first_sentence(None) == ""
    print("✓ ok")

    p = {"initial_holdings": {"AMAT": {"shares": 3, "entry_price": 200.0}, "KLAC": {"shares": 8, "entry_price": 90.0},
                              "KEYS": {"shares": 4, "entry_price": 150.0}},
         "holdings": {"KLAC": {"shares": 4}, "NVDA": {"shares": 10}, "GOOGL": {"shares": 6}},
         "rebalance_history": [
             {"date": "2026-10-06", "trigger": "manual", "orders": [
                 {"symbol": "KEYS", "side": "SELL", "qty": 4, "result": "FILLED"},
                 {"symbol": "NVDA", "side": "BUY", "qty": 10, "result": "FILLED"},
                 {"symbol": "XYZ", "side": "BUY", "qty": 9, "result": "REJECTED"}],
              "portfolio_value_before": 10024.0},
             {"date": "2026-10-16", "at": "2026-10-16T16:31:00+00:00", "trigger": "proactive",
              "reason": "Rule — the portfolio is down 11.5% since its last recalibration (2026-10-06) while its own "
                        "sector pool is -4.2%. (Model's own view: concentrated semicap sell-off.)",
              "orders": [{"symbol": "AMAT", "side": "SELL", "qty": 3, "result": "FILLED"},
                         {"symbol": "KLAC", "side": "SELL", "qty": 4, "result": "FILLED"},
                         {"symbol": "GOOGL", "side": "BUY", "qty": 6, "result": "FILLED"}],
              "holdings_after": {"KLAC": {"shares": 4}, "NVDA": {"shares": 10}, "GOOGL": {"shares": 6}},
              "review": {"grade": "B", "summary": "Sound rotation out of a lagging sub-industry. Watch NVDA weight; "
                                                  "otherwise balanced and well diversified across software and internet.",
                         "orders": {"GOOGL": {"verdict": "agree", "reason": "Quality."},
                                    "AMAT": {"verdict": "concern", "reason": "May be selling near a bottom. Risky."},
                                    "KLAC": {"verdict": "object", "reason": "Halving after a crash locks in losses."}},
                         "vetoed": {"ZZZ": "red-flag news: \"ZZZ halted\""}},
              "portfolio_value_before": 9001.0}]}

    print("TEST 2: timeline — launch rows, per-event changes, reconstructed and recorded holdings")
    t = build_timeline(p)
    assert [r["symbol"] for r in t["initial"]] == ["AMAT", "KEYS", "KLAC"]
    rows = {r["symbol"]: r for r in t["initial"]}
    assert rows["KEYS"]["sold_on"] == "2026-10-06" and rows["AMAT"]["sold_on"] == "2026-10-16"
    assert rows["KLAC"]["now_shares"] == 4 and rows["KLAC"]["sold_on"] is None and rows["AMAT"]["entry_price"] == 200.0
    new, old = t["events"]                                   # newest first
    assert old["exited"] == ["KEYS"] and old["entered"] == ["NVDA"] and old["holds"] == ["AMAT", "KLAC", "NVDA"]
    assert "XYZ" not in old["entered"], "rejected orders don't change holdings"
    assert new["exited"] == ["AMAT"] and new["entered"] == ["GOOGL"] and new["resized"] == ["KLAC"]
    assert new["n_holds"] == 3 and new["type"] == "📰 Mid-day"
    print(f"✓ Oct 6: -KEYS +NVDA; Oct 16: -AMAT +GOOGL ⇅KLAC -> holds {new['holds']}")

    print("TEST 3: short summaries with the long text kept for an expandable detail")
    assert new["why_short"] == "Rule — the portfolio is down 11.5% since its last recalibration (2026-10-06) while its own sector pool is -4.2%."
    assert "Model's own view" in new["why_full"]
    assert old["why_short"] == "Recalibrate Now clicked" and old["why_full"] == ""
    assert new["grade"] == "B" and new["review_short"] == "Sound rotation out of a lagging sub-industry."
    assert new["review_full"].startswith("Sound rotation") and new["objections"] == [("KLAC", "Halving after a crash locks in losses.")]
    assert new["concerns"] == [("AMAT", "May be selling near a bottom.")] and new["blocked"][0][0] == "ZZZ"
    print("✓ why / reviewer cut to one sentence; full text, objections, concerns and blocks kept separately")

    print("TEST 4: summary counts; a portfolio with no launch record doesn't fake a holdings diff")
    assert t["summary"]["n"] == 2 and t["summary"]["by_type"] == {"manual": 1, "proactive": 1}
    assert t["summary"]["started"] == ["AMAT", "KEYS", "KLAC"] and t["summary"]["kept"] == ["KLAC"]
    bare = build_timeline({"holdings": {"A": {"shares": 1}}, "rebalance_history": p["rebalance_history"][:1]})
    assert bare["initial"] == [] and bare["events"][0].get("unknown_holdings") and bare["events"][0]["entered"] == []
    print("✓ ok")
    print("\nAll holdings_timeline tests passed.")
