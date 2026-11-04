"""
test_portfolio_reconcile.py
============================
Tests for portfolio_store.resolve_pending / reconcile_portfolio — the fix for
the Portfolio page showing PRE-recalibration holdings when a recalibration's
orders were still in flight (e.g. placed after hours) at the moment the ledger
was written. See the block comment above resolve_pending in portfolio_store.py.

Run:  python test_portfolio_reconcile.py
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict

import portfolio_store as ps
from portfolio_store import resolve_pending, reconcile_portfolio


def doc(**over) -> Dict[str, Any]:
    d = {"status": "open", "cash_remaining": 1_000.0,
         "holdings": {"AAA": {"shares": 100, "entry_price": 10.0, "cost": 1000.0},
                      "BBB": {"shares": 50, "entry_price": 20.0, "cost": 1000.0}},
         "pending_order_ids": {}}
    d.update(over)
    return d


def fact(coid="c1", side="buy", status="filled", qty=0, px=0.0):
    return {"client_order_id": coid, "side": side, "status": status, "filled_qty": qty, "filled_avg_price": px}


def value(d: Dict[str, Any], marks: Dict[str, float]) -> float:
    return d["cash_remaining"] + sum(h["shares"] * marks[s] for s, h in d["holdings"].items())


# ---------------------------------------------------------------- Firestore double
class _Snap:
    def __init__(self, data, update_time): self._d, self.update_time = data, update_time
    @property
    def exists(self): return self._d is not None
    def to_dict(self): return copy.deepcopy(self._d)


class _Opt:
    def __init__(self, last_update_time): self.last_update_time = last_update_time


class _Ref:
    def __init__(self, client, pid): self.c, self.pid = client, pid
    def get(self):
        d = self.c.docs.get(self.pid)
        snap = _Snap(d, self.c.versions.get(self.pid)) if d is not None else _Snap(None, None)
        snap.id = self.pid                       # the real DocumentSnapshot exposes .id; get_portfolio() reads it
        return snap
    def set(self, fields):
        self.c.docs[self.pid] = copy.deepcopy(fields)
        self.c.versions[self.pid] = self.c.versions.get(self.pid, 0) + 1
    def update(self, fields, option=None):
        if option is not None and option.last_update_time != self.c.versions.get(self.pid):
            raise RuntimeError("FailedPrecondition: document was modified")
        self.c.docs[self.pid].update(copy.deepcopy(fields))
        self.c.versions[self.pid] += 1


class FakeClient:
    def __init__(self, docs):
        self.docs = {k: copy.deepcopy(v) for k, v in docs.items()}
        self.versions = {k: 1 for k in docs}
    def collection(self, name):
        c = self
        other = "" if name == "meridian_portfolios" else f"{name}/"   # other collections don't mix with portfolios
        class _C:
            def document(self_inner, pid): return _Ref(c, f"{other}{pid}")
        return _C()
    def write_option(self, last_update_time): return _Opt(last_update_time)


def install(docs):
    fake = FakeClient(docs)
    ps._client = fake
    return fake


if __name__ == "__main__":
    print("TEST 1: a pending BUY that later filled is added to holdings and paid for out of cash")
    d = doc(pending_order_ids={"CCC": "c1"}, cash_remaining=2_000.0)
    up = resolve_pending(d, {"CCC": fact("c1", "buy", "filled", 30, 25.0)})
    entered = up["holdings"]["CCC"].pop("entry_date")
    assert up["holdings"]["CCC"] == {"shares": 30, "entry_price": 25.0, "cost": 750.0} and entered == up["log"][0]["at"][:10]
    assert up["cash_remaining"] == 1_250.0 and up["pending_order_ids"] == {}
    assert up["log"][0]["side"] == "BUY" and up["log"][0]["filled_qty"] == 30
    print("✓ CCC 30 @ 25.00 added; cash 2000 -> 1250; nothing pending")

    print("TEST 2: THE REPORTED CASE — recalibration sold part of AAA and it filled after the ledger was written")
    d = doc(pending_order_ids={"AAA": "r1"})
    up = resolve_pending(d, {"AAA": fact("r1", "sell", "filled", 40, 12.0)})
    assert up["holdings"]["AAA"]["shares"] == 60 and up["holdings"]["AAA"]["entry_price"] == 10.0
    assert type(up["holdings"]["AAA"]["shares"]) is int, "whole shares must stay ints (else the page shows '60.0')"
    assert type(resolve_pending(doc(pending_order_ids={"CCC": "c"}), {"CCC": fact("c", "buy", "filled", 30.0, 25.0)})
                ["holdings"]["CCC"]["shares"]) is int
    assert up["cash_remaining"] == 1_000.0 + 480.0 and up["holdings"]["BBB"] == d["holdings"]["BBB"]
    print("✓ AAA 100 -> 60 (cost basis unchanged at 10.00); proceeds $480 returned to cash")

    print("TEST 3: a full SELL removes the position entirely")
    up = resolve_pending(doc(pending_order_ids={"AAA": "r1"}), {"AAA": fact("r1", "sell", "filled", 100, 11.0)})
    assert "AAA" not in up["holdings"] and up["cash_remaining"] == 2_100.0
    print("✓ AAA gone; cash +1100")

    print("TEST 4: a BUY into an existing position blends the cost basis")
    up = resolve_pending(doc(pending_order_ids={"AAA": "b1"}, cash_remaining=5_000.0),
                         {"AAA": fact("b1", "buy", "filled", 100, 14.0)})
    h = up["holdings"]["AAA"]
    assert h["shares"] == 200 and abs(h["entry_price"] - 12.0) < 1e-9 and abs(h["cost"] - 2400.0) < 1e-9
    assert up["cash_remaining"] == 3_600.0
    print("✓ 100@10 + 100@14 -> 200@12.00")

    print("TEST 5: orders still working (accepted / new / partially_filled) are NOT applied and stay pending")
    for status in ("accepted", "new", "pending_new", "partially_filled", "held"):
        assert resolve_pending(doc(pending_order_ids={"AAA": "r1"}),
                               {"AAA": fact("r1", "sell", status, 10, 12.0)}) is None, status
    print("✓ a half-filled order is not counted until it reaches a terminal state")

    print("TEST 6: canceled/expired with NO fill just clears the pending flag")
    up = resolve_pending(doc(pending_order_ids={"AAA": "r1"}), {"AAA": fact("r1", "sell", "expired", 0, 0.0)})
    assert up["holdings"] == doc()["holdings"] and up["cash_remaining"] == 1_000.0 and up["pending_order_ids"] == {}
    print("✓ ledger untouched, order no longer pending (a DAY order that expired unfilled)")

    print("TEST 7: canceled AFTER a partial fill — the shares that did trade are still applied")
    up = resolve_pending(doc(pending_order_ids={"AAA": "r1"}), {"AAA": fact("r1", "sell", "canceled", 25, 12.0)})
    assert up["holdings"]["AAA"]["shares"] == 75 and up["cash_remaining"] == 1_300.0
    print("✓ 25 of 100 sold before the cancel -> 75 held")

    print("TEST 8: IDEMPOTENT — a fact for an order that is no longer pending (already applied) is ignored")
    already = doc(pending_order_ids={})
    assert resolve_pending(already, {"AAA": fact("r1", "sell", "filled", 40, 12.0)}) is None
    superseded = doc(pending_order_ids={"AAA": "r2"})
    assert resolve_pending(superseded, {"AAA": fact("r1", "sell", "filled", 40, 12.0)}) is None
    print("✓ same fill can't be applied twice; a stale order id can't touch a newer pending order")

    print("TEST 9: a SELL can never take the ledger negative or invent shares")
    up = resolve_pending(doc(pending_order_ids={"AAA": "r1"}), {"AAA": fact("r1", "sell", "filled", 500, 12.0)})
    assert "AAA" not in up["holdings"] and up["cash_remaining"] == 1_000.0 + 100 * 12.0
    up = resolve_pending(doc(pending_order_ids={"ZZZ": "r1"}), {"ZZZ": fact("r1", "sell", "filled", 5, 9.0)})
    assert "ZZZ" not in up["holdings"] and up["cash_remaining"] == 1_000.0 and "note" in up["log"][0]
    print("✓ oversell clamped to the 100 held; selling an unheld name changes nothing and is flagged")

    print("TEST 10: mixed batch — the finished order applies, the working one stays pending")
    d = doc(pending_order_ids={"AAA": "r1", "BBB": "r2"})
    up = resolve_pending(d, {"AAA": fact("r1", "sell", "filled", 40, 12.0), "BBB": fact("r2", "buy", "accepted", 0, 0)})
    assert up["holdings"]["AAA"]["shares"] == 60 and up["pending_order_ids"] == {"BBB": "r2"}
    print("✓ AAA applied; BBB still pending")

    print("TEST 11: the input document is never mutated (pure)")
    d = doc(pending_order_ids={"AAA": "r1"}); before = copy.deepcopy(d)
    resolve_pending(d, {"AAA": fact("r1", "sell", "filled", 40, 12.0)})
    assert d == before
    print("✓ ok")

    print("TEST 12: VALUE IS CONSERVED — at the fill price, applying a fill never creates or destroys money")
    for side, sym in (("buy", "CCC"), ("sell", "AAA")):
        d = doc(pending_order_ids={sym: "x"}, cash_remaining=5_000.0)
        marks = {"AAA": 12.0, "BBB": 20.0, "CCC": 12.0}
        pre = value(d, marks)                      # ledger BEFORE the fill: in-flight buy funds still sit in cash,
        up = resolve_pending(d, {sym: fact("x", side, "filled", 30, 12.0)})   # in-flight sell shares still in holdings
        post = value({**d, **up}, marks)
        assert abs(pre - post) < 1e-6, (side, pre, post)
    print("✓ total value identical before/after for both a BUY and a SELL filled at the mark")

    print("\n--- reconcile_portfolio against a Firestore-shaped double (with update-time preconditions) ---")

    def broker(answers):
        calls = []
        def fetch(coid):
            calls.append(coid)
            a = answers[coid]
            if isinstance(a, Exception):
                raise a
            return a
        fetch.calls = calls
        return fetch

    print("TEST 13: end to end — ledger read, broker asked, holdings updated, audit log written")
    fake = install({"p1": doc(pending_order_ids={"AAA": "r1"})})
    applied = reconcile_portfolio("p1", broker({"r1": {"side": "sell", "status": "filled", "filled_qty": 40, "filled_avg_price": 12.0}}))
    saved = fake.docs["p1"]
    assert applied and saved["holdings"]["AAA"]["shares"] == 60 and saved["pending_order_ids"] == {}
    assert saved["cash_remaining"] == 1_480.0 and len(saved["reconciliation_log"]) == 1
    print("✓ AAA 100 -> 60 persisted; reconciliation_log has 1 entry")

    print("TEST 14: running it again is a no-op and does not even call the broker")
    f2 = broker({})
    assert reconcile_portfolio("p1", f2) is None and f2.calls == [] and fake.docs["p1"]["holdings"]["AAA"]["shares"] == 60
    print("✓ nothing pending -> no broker calls, no change")

    print("TEST 15: a broker lookup that FAILS leaves the order pending (never guessed)")
    fake = install({"p2": doc(pending_order_ids={"AAA": "r1", "BBB": "r2"})})
    out = reconcile_portfolio("p2", broker({"r1": TimeoutError("boom"),
                                            "r2": {"side": "buy", "status": "filled", "filled_qty": 5, "filled_avg_price": 20.0}}))
    s = fake.docs["p2"]
    assert s["pending_order_ids"] == {"AAA": "r1"} and s["holdings"]["BBB"]["shares"] == 55
    print("✓ AAA stays pending (lookup failed); BBB's fill still applied")

    print("TEST 16: two requests race — the loser's write is REJECTED, so the fill is applied exactly once")
    fake = install({"p3": doc(pending_order_ids={"AAA": "r1"})})
    def racing_fetch(coid):
        # while request A is talking to the broker, request B finishes the same reconciliation first
        reconcile_portfolio("p3", broker({"r1": {"side": "sell", "status": "filled", "filled_qty": 40, "filled_avg_price": 12.0}}))
        return {"side": "sell", "status": "filled", "filled_qty": 40, "filled_avg_price": 12.0}
    out = reconcile_portfolio("p3", racing_fetch)
    s = fake.docs["p3"]
    assert out is None, "the slower request must NOT write"
    assert s["holdings"]["AAA"]["shares"] == 60 and s["cash_remaining"] == 1_480.0, "40 shares sold ONCE, not twice"
    assert len(s["reconciliation_log"]) == 1
    print("✓ shares 100 -> 60 (not 20), cash +480 (not +960), one log entry")

    print("TEST 17: closed portfolios and missing ones are ignored")
    fake = install({"p4": doc(status="closed", pending_order_ids={"AAA": "r1"})})
    f4 = broker({})
    assert reconcile_portfolio("p4", f4) is None and reconcile_portfolio("nope", f4) is None and f4.calls == []
    print("✓ ok")

    print("\nAll portfolio reconciliation tests passed.")
