"""
daytrade_store.py
==================
Persistence for the day-trading capability, in Firestore — the same backend
and project as portfolio_store.py, but in its OWN collections, so nothing
here can touch or be mistaken for a portfolio ledger:

  meridian_daytrade_sessions/{YYYY-MM-DD}   one document per trading day
  meridian_daytrade_control/state           the kill switch

The session document is what makes the runner restartable: after a crash,
the new process reads today's document to learn which names were already
traded, how many entries were sent and what was realized — then rebuilds
positions from the BROKER (the source of truth for what is actually held).
Nothing about an open position is trusted from this store alone.

The kill switch is a single Firestore flag the runner re-reads every cycle.
Asymmetry by design (enforced in daytrade_ui.py): anyone with page access can
HALT — halting can only reduce risk — but RESUMING requires the shared token,
the same pattern web_ui.py uses for its order-placing endpoint.

Two implementations of one tiny interface: FirestoreDayTradeStore (real) and
InMemoryDayTradeStore (tests, local dry runs).
"""

from __future__ import annotations

import copy
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from daytrade_engine import Trade

_SESSIONS = "meridian_daytrade_sessions"
_CONTROL = "meridian_daytrade_control"
MAX_EVENTS = 200


def _now() -> datetime:
    return datetime.now(timezone.utc)


def trade_to_dict(t: Trade) -> Dict[str, Any]:
    return {"symbol": t.symbol, "strategy": t.strategy, "qty": t.qty, "entry_price": t.entry_price,
            "exit_price": t.exit_price, "entry_ts": t.entry_ts.isoformat(), "exit_ts": t.exit_ts.isoformat(),
            "pnl": round(t.pnl, 4), "fees": round(t.fees, 4), "reason": t.reason,
            "r_multiple": round(t.r_multiple, 4)}


def trade_from_dict(d: Dict[str, Any]) -> Trade:
    return Trade(d["symbol"], d["strategy"], d["qty"], d["entry_price"], d["exit_price"],
                 datetime.fromisoformat(d["entry_ts"]), datetime.fromisoformat(d["exit_ts"]),
                 d["pnl"], d.get("fees", 0.0), d.get("reason", ""), d.get("r_multiple", 0.0))


class DayTradeStore(Protocol):
    def get_control(self) -> Dict[str, Any]: ...
    def set_halted(self, halted: bool, reason: str = "", by: str = "") -> Dict[str, Any]: ...
    def get_session(self, day: date) -> Optional[Dict[str, Any]]: ...
    def save_session(self, day: date, patch: Dict[str, Any]) -> None: ...
    def append_events(self, day: date, events: List[Dict[str, Any]]) -> None: ...
    def list_sessions(self, limit: int = 30) -> List[Dict[str, Any]]: ...


def _default_control() -> Dict[str, Any]:
    return {"halted": False, "reason": "", "updated_by": "", "updated_at": None}


class InMemoryDayTradeStore:
    def __init__(self) -> None:
        self._control = _default_control()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_control(self) -> Dict[str, Any]:
        return dict(self._control)

    def set_halted(self, halted: bool, reason: str = "", by: str = "") -> Dict[str, Any]:
        self._control = {"halted": bool(halted), "reason": reason, "updated_by": by,
                         "updated_at": _now().isoformat()}
        return dict(self._control)

    def get_session(self, day: date) -> Optional[Dict[str, Any]]:
        doc = self._sessions.get(day.isoformat())
        return copy.deepcopy(doc) if doc else None

    def save_session(self, day: date, patch: Dict[str, Any]) -> None:
        doc = self._sessions.setdefault(day.isoformat(), {"date": day.isoformat()})
        doc.update(copy.deepcopy(patch))
        doc["updated_at"] = _now().isoformat()

    def append_events(self, day: date, events: List[Dict[str, Any]]) -> None:
        doc = self._sessions.setdefault(day.isoformat(), {"date": day.isoformat()})
        doc["events"] = (doc.get("events", []) + copy.deepcopy(events))[-MAX_EVENTS:]

    def list_sessions(self, limit: int = 30) -> List[Dict[str, Any]]:
        return [copy.deepcopy(self._sessions[k]) for k in sorted(self._sessions, reverse=True)[:limit]]


class FirestoreDayTradeStore:
    def __init__(self, client: Any = None, project: Optional[str] = None):
        if client is None:
            import os
            from google.cloud import firestore
            client = firestore.Client(project=project or os.environ.get("MERIDIAN_PROJECT", "meridian-508414"))
        self._db = client

    def _ctl(self):
        return self._db.collection(_CONTROL).document("state")

    def _sess(self, day: date):
        return self._db.collection(_SESSIONS).document(day.isoformat())

    def get_control(self) -> Dict[str, Any]:
        snap = self._ctl().get()
        return {**_default_control(), **(snap.to_dict() or {})} if snap.exists else _default_control()

    def set_halted(self, halted: bool, reason: str = "", by: str = "") -> Dict[str, Any]:
        doc = {"halted": bool(halted), "reason": reason, "updated_by": by, "updated_at": _now().isoformat()}
        self._ctl().set(doc)
        return doc

    def get_session(self, day: date) -> Optional[Dict[str, Any]]:
        snap = self._sess(day).get()
        return snap.to_dict() if snap.exists else None

    def save_session(self, day: date, patch: Dict[str, Any]) -> None:
        self._sess(day).set({"date": day.isoformat(), **patch, "updated_at": _now().isoformat()}, merge=True)

    def append_events(self, day: date, events: List[Dict[str, Any]]) -> None:
        cur = (self.get_session(day) or {}).get("events", [])
        self._sess(day).set({"date": day.isoformat(), "events": (cur + events)[-MAX_EVENTS:]}, merge=True)

    def list_sessions(self, limit: int = 30) -> List[Dict[str, Any]]:
        from google.cloud import firestore
        q = self._db.collection(_SESSIONS).order_by("date", direction=firestore.Query.DESCENDING).limit(limit)
        return [s.to_dict() for s in q.stream()]


def default_store() -> DayTradeStore:
    return FirestoreDayTradeStore()


def summarize_sessions(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-day roll-up for the dashboard and for the Phase-5 paper
    evaluation: total/avg P&L, win rate, and — when the runner recorded them —
    the no-playbook SHADOW numbers, which are the LLM ablation data."""
    days = [s for s in sessions if s.get("status") in ("completed", "halted")]
    trades = [t for s in days for t in s.get("trades", [])]
    wins = [t for t in trades if t["pnl"] > 0]
    with_shadow = [s for s in days if s.get("shadow") is not None]
    return {
        "days": len(days),
        "trades": len(trades),
        "net_pnl": sum(s.get("realized_pnl", 0.0) for s in days),
        "win_rate": (len(wins) / len(trades)) if trades else None,
        "avg_r": (sum(t.get("r_multiple", 0.0) for t in trades) / len(trades)) if trades else None,
        "halted_days": sum(1 for s in days if s.get("halted")),
        "shadow_days": len(with_shadow),
        "shadow_net_pnl": sum(s["shadow"].get("pnl", 0.0) for s in with_shadow),
        "live_net_pnl_on_shadow_days": sum(s.get("realized_pnl", 0.0) for s in with_shadow),
    }


# ---------------------------------------------------------------------------
# A tiny Firestore double, shaped like the client surface used above, so the
# Firestore store's own calls (set/merge, get, order_by/limit/stream) run.
# ---------------------------------------------------------------------------
class _Snap:
    def __init__(self, data): self._d = data
    @property
    def exists(self): return self._d is not None
    def to_dict(self): return copy.deepcopy(self._d)


class _Doc:
    def __init__(self, store, path): self._s, self._p = store, path
    def get(self): return _Snap(self._s.get(self._p))
    def set(self, data, merge=False):
        cur = self._s.get(self._p) if merge else None
        self._s[self._p] = {**(cur or {}), **copy.deepcopy(data)}


class _Query:
    def __init__(self, rows): self._rows = rows
    def order_by(self, field, direction=None):
        from google.cloud import firestore      # compare against the SDK's own constant, not a guess of it
        return _Query(sorted(self._rows, key=lambda r: r.get(field, ""),
                             reverse=(direction == firestore.Query.DESCENDING)))
    def limit(self, n): return _Query(self._rows[:n])
    def stream(self): return [_Snap(r) for r in self._rows]


class _Coll(_Query):
    def __init__(self, store, name):
        self._store, self._name = store, name
        super().__init__([v for k, v in store.items() if k.startswith(name + "/")])
    def document(self, doc_id): return _Doc(self._store, f"{self._name}/{doc_id}")


class FakeFirestoreClient:
    def __init__(self): self._data: Dict[str, Dict[str, Any]] = {}
    def collection(self, name): return _Coll(self._data, name)


if __name__ == "__main__":
    from datetime import timedelta
    import daytrade_store as me

    day = date(2026, 3, 4)
    t0 = datetime(2026, 3, 4, 14, 48, tzinfo=timezone.utc)
    trade = Trade("AAA", "orb_long", 49, 100.62, 101.8, t0, t0 + timedelta(minutes=40), 57.5, 0.2, "target", 1.9)

    def suite(name, store):
        print(f"--- {name} ---")
        print("TEST 1: kill switch defaults to OFF, and halt/resume round-trips with who/why")
        assert store.get_control()["halted"] is False
        c = store.set_halted(True, "manual", by="ui")
        got = store.get_control()
        assert got["halted"] and got["reason"] == "manual" and got["updated_by"] == "ui" and got["updated_at"]
        store.set_halted(False, "resume", by="token")
        assert store.get_control()["halted"] is False
        print("✓ default off; halt -> resume")

        print("TEST 2: session saves MERGE — a later patch never wipes earlier fields")
        assert store.get_session(day) is None
        store.save_session(day, {"status": "running", "sleeve": 25_000.0, "watchlist": ["AAA", "BBB"]})
        store.save_session(day, {"realized_pnl": 57.3, "trades": [trade_to_dict(trade)]})
        s = store.get_session(day)
        assert s["status"] == "running" and s["watchlist"] == ["AAA", "BBB"] and s["realized_pnl"] == 57.3
        assert trade_from_dict(s["trades"][0]) == trade
        print("✓ status/watchlist survive the second save; Trade round-trips exactly")

        print("TEST 3: events append and are capped")
        store.append_events(day, [{"kind": "entry", "n": i} for i in range(150)])
        store.append_events(day, [{"kind": "entry", "n": i} for i in range(150, 300)])
        ev = store.get_session(day)["events"]
        assert len(ev) == MAX_EVENTS and ev[-1]["n"] == 299 and ev[0]["n"] == 100
        print(f"✓ capped at {MAX_EVENTS}, newest kept")

        print("TEST 4: list_sessions is newest-first and honors the limit")
        for i in (1, 2, 3):
            store.save_session(day - timedelta(days=i), {"status": "completed", "realized_pnl": float(i)})
        got = store.list_sessions(limit=3)
        assert [g["date"] for g in got] == ["2026-03-04", "2026-03-03", "2026-03-02"]
        print("✓ 03-04, 03-03, 03-02")

    suite("InMemoryDayTradeStore", InMemoryDayTradeStore())
    fake = FakeFirestoreClient()
    suite("FirestoreDayTradeStore (against a Firestore-shaped double)", FirestoreDayTradeStore(client=fake))

    print("--- summarize_sessions ---")
    sessions = [
        {"status": "completed", "realized_pnl": 100.0, "trades": [{"pnl": 60.0, "r_multiple": 1.5}, {"pnl": 40.0, "r_multiple": 1.0}],
         "shadow": {"pnl": 30.0}},
        {"status": "halted", "halted": True, "realized_pnl": -750.0, "trades": [{"pnl": -750.0, "r_multiple": -2.0}]},
        {"status": "running", "realized_pnl": 999.0, "trades": [{"pnl": 999.0, "r_multiple": 9.0}]},   # in-progress: excluded
        {"status": "skipped"},
    ]
    r = summarize_sessions(sessions)
    assert r["days"] == 2 and r["trades"] == 3 and r["net_pnl"] == -650.0 and r["halted_days"] == 1
    assert abs(r["win_rate"] - 2 / 3) < 1e-9 and r["shadow_days"] == 1 and r["shadow_net_pnl"] == 30.0
    assert r["live_net_pnl_on_shadow_days"] == 100.0
    print("✓ in-progress and skipped days excluded; win rate, halted days and shadow ablation roll up")
    print("\nAll daytrade_store tests passed.")
