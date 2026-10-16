"""
portfolio_store.py
===================
Tracks multiple independently-named "portfolios" so a user can run several
strategies (different sectors, different max-names settings, different
capital) side by side and compare their performance over time.

HONEST LIMITATION, stated plainly: Alpaca's PAPER trading product is ONE
account per set of API keys — there is no way to create several separate
paper brokerage accounts from this app (that would require Alpaca's Broker
API, a heavier onboarding-oriented product not wired in here). Real orders
placed for every portfolio therefore land in the SAME underlying Alpaca
paper account and net into ONE combined position per symbol there.

What this module gives you instead: an independent LEDGER per portfolio —
symbol, shares, entry price, capital, timestamps — recorded in Firestore at
launch time. Each portfolio's performance is computed by marking ITS OWN
recorded shares to live prices, entirely independently of what any other
portfolio (or the raw combined Alpaca account) is doing. That's what makes
"Portfolio A vs Portfolio B" a fair, apples-to-apples comparison even though
the real fills all land in one shared brokerage account under the hood.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

PROJECT = os.environ.get("MERIDIAN_PROJECT", "meridian-508414")
_COLLECTION = "meridian_portfolios"

_client = None


def _get_client():
    global _client
    if _client is None:
        from google.cloud import firestore
        _client = firestore.Client(project=PROJECT)
    return _client


def create_portfolio(name: str, sectors: List[str], capital: float, top_n: int,
                     holdings: Dict[str, Dict[str, float]], cash_remaining: float,
                     errors: Optional[List[Dict[str, str]]] = None,
                     pending_order_ids: Optional[Dict[str, str]] = None,
                     rebalance_every_days: int = 0,
                     estimated_1mo_return: Optional[float] = None,
                     news_watch: bool = False, n_stocks: Optional[int] = None) -> str:
    """holdings: {symbol: {"shares": int, "entry_price": float, "cost": float}}
    for whatever actually filled. pending_order_ids: {symbol: client_order_id}
    for orders accepted by Alpaca but not yet resolved (e.g. placed after
    market hours) — close_portfolio() cancels these explicitly rather than
    leaving them to fill into an untracked position later.

    rebalance_every_days: 0 means "manual only" — no automatic recalibration
    is ever scheduled for this portfolio. Any positive value sets
    next_rebalance_date to that many days out; the daily due-check (see
    list_due_portfolios) picks it up automatically once that date arrives.

    estimated_1mo_return: the SAME "naive expected 1-month portfolio
    return" already shown on the recommendation page before you clicked
    Launch (see web_ui.enrich() — trailing 1-month momentum extrapolated
    forward, weighted by allocation; not a forecast, just a momentum
    read at the moment of launch). Logged here, once, at creation time so
    the Portfolio tab can later show what was naively projected vs. what
    actually happened — None if it wasn't available for some reason,
    never recomputed after the fact.

    news_watch: the owner's switch for the mid-day news watch
    (proactive_recal.py) — off unless they turned it on at launch; flip it
    later with set_news_watch().
    Returns the new portfolio_id."""
    portfolio_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    doc = {
        "name": name or f"Portfolio {portfolio_id[:6]}",
        "sectors": sectors,
        "capital": capital,
        "top_n": top_n,
        "holdings": holdings,
        "cash_remaining": cash_remaining,
        "errors": errors or [],
        "pending_order_ids": pending_order_ids or {},
        "status": "open",
        "created_at": now.isoformat(),
        "closed_at": None,
        "rebalance_every_days": rebalance_every_days,
        "next_rebalance_date": (now + timedelta(days=rebalance_every_days)).date().isoformat()
                               if rebalance_every_days > 0 else None,
        "rebalance_history": [],
        "estimated_1mo_return": estimated_1mo_return,
        "news_watch": bool(news_watch),
        # how many stocks it STARTED with — every recalibration re-selects exactly this many
        "n_stocks": int(n_stocks) if n_stocks else len(holdings),
    }
    _get_client().collection(_COLLECTION).document(portfolio_id).set(doc)
    return portfolio_id


def list_portfolios() -> List[Dict[str, Any]]:
    out = []
    for snap in _get_client().collection(_COLLECTION).stream():
        row = snap.to_dict()
        row["id"] = snap.id
        out.append(row)
    out.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return out


def get_portfolio(portfolio_id: str) -> Optional[Dict[str, Any]]:
    snap = _get_client().collection(_COLLECTION).document(portfolio_id).get()
    if not snap.exists:
        return None
    row = snap.to_dict()
    row["id"] = snap.id
    return row


def close_portfolio(portfolio_id: str, exit_holdings: Dict[str, Dict[str, float]],
                    proceeds: float) -> None:
    """Marks a portfolio closed after its holdings have been sold. Keeps the
    original entry-price ledger intact (for the final return-since-inception
    figure) and records what it actually sold for."""
    _get_client().collection(_COLLECTION).document(portfolio_id).update({
        "status": "closed",
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "exit_holdings": exit_holdings,
        "exit_proceeds": proceeds,
        "next_rebalance_date": None,   # a closed portfolio is never due again
    })


def list_due_portfolios(as_of: Optional[str] = None) -> List[Dict[str, Any]]:
    """Open portfolios whose next_rebalance_date has arrived (<= as_of, an
    ISO date string; defaults to today UTC) — what the daily scheduled
    recalibration check (see web_ui.py's /portfolios/recalibrate_due) acts
    on. A firestore range query needs an index-friendly field, so this
    fetches all open portfolios and filters in Python — fine at the scale
    a single user's tracked-portfolio list actually reaches."""
    as_of = as_of or datetime.now(timezone.utc).date().isoformat()
    return [
        p for p in list_portfolios()
        if p.get("status") == "open" and p.get("next_rebalance_date")
        and p["next_rebalance_date"] <= as_of
    ]


def record_rebalance(portfolio_id: str, new_holdings: Dict[str, Dict[str, float]],
                     new_cash_remaining: float, pending_order_ids: Dict[str, str],
                     event: Dict[str, Any], next_rebalance_date: Optional[str]) -> None:
    """Applies the result of a real recalibration: replaces the holdings
    ledger with what the portfolio holds AFTER the rebalance trades,
    replaces pending_order_ids with whatever THIS pass left unresolved
    (every previously-pending symbol is re-examined every recalibration, so
    a full replace — not a merge — is correct), and appends `event` (the
    orders placed, old vs new weights, timestamp) to rebalance_history for
    a visible audit trail on the detail page."""
    doc_ref = _get_client().collection(_COLLECTION).document(portfolio_id)
    snap = doc_ref.get()
    history = (snap.to_dict() or {}).get("rebalance_history", []) if snap.exists else []
    history.append(event)
    doc_ref.update({
        "holdings": new_holdings,
        "cash_remaining": new_cash_remaining,
        "pending_order_ids": pending_order_ids,
        "rebalance_history": history,
        "next_rebalance_date": next_rebalance_date,
    })


def set_news_watch(portfolio_id: str, enabled: bool) -> bool:
    """Turns the mid-day news watch on/off for one OPEN portfolio. Returns
    False if there is no such open portfolio."""
    doc_ref = _get_client().collection(_COLLECTION).document(portfolio_id)
    snap = doc_ref.get()
    if not snap.exists or (snap.to_dict() or {}).get("status") != "open":
        return False
    doc_ref.update({"news_watch": bool(enabled),
                    "news_watch_changed_at": datetime.now(timezone.utc).isoformat()})
    return True


def record_proactive_check(portfolio_id: str, entry: Dict[str, Any], keep: int = 30) -> None:
    """Appends one mid-day news-watch result (proactive_recal.py) — acted on
    or not — to the portfolio's `proactive_checks`, newest last, capped at
    `keep`. A check that led to a recalibration ALSO shows up in
    rebalance_history (trigger="proactive"); this log is what explains the
    days it decided to wait."""
    doc_ref = _get_client().collection(_COLLECTION).document(portfolio_id)
    snap = doc_ref.get()
    if not snap.exists:
        return
    checks = (snap.to_dict() or {}).get("proactive_checks", []) + [entry]
    doc_ref.update({"proactive_checks": checks[-keep:]})


# =============================================================================
# Reconciling orders that were still IN FLIGHT when the ledger was written.
#
# BUG THIS FIXES: launch() and recalibrate_portfolio() poll an order for only a
# few seconds. A market order placed outside trading hours (which is exactly
# when the scheduled daily recalibration tends to run) is accepted but not
# filled inside that window, so it is recorded in `pending_order_ids` and the
# holdings ledger is left as it was BEFORE the trade. Alpaca then fills it at
# the next open — and nothing ever read the outcome back. The Portfolio page
# kept showing the pre-recalibration holdings indefinitely, and the NEXT
# recalibration sized its trades off that stale ledger (its order IDs are
# date-stamped, so it would happily place a second, duplicate rebalance).
#
# reconcile_portfolio() asks the broker how each pending order ended, and folds
# every TERMINAL one into holdings and cash, exactly once.
# =============================================================================
_TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day", "suspended", "replaced"}


def _shares(x: float):
    """Alpaca reports quantities as floats (60.0). This app's ledger and its
    templates deal in whole shares, so a whole quantity is stored as an int —
    otherwise the Holdings table would show '60.0'."""
    return int(round(x)) if abs(x - round(x)) < 1e-9 else x


def resolve_pending(doc: Dict[str, Any], facts: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """PURE. `doc` is the portfolio document; `facts[symbol]` is what the
    broker reports for that symbol's pending order:
        {"client_order_id", "side": "buy"|"sell", "status",
         "filled_qty", "filled_avg_price"}
    Returns the fields to write ({holdings, cash_remaining, pending_order_ids,
    log}) or None if nothing was resolved.

    Rules:
      * only TERMINAL orders resolve; a still-working order stays pending;
      * a fact whose client_order_id no longer matches the doc's pending entry
        is skipped — another request already applied it (this is what makes a
        double page-load harmless);
      * whatever DID fill (even on a canceled/expired order) is applied, so a
        partial fill is never lost;
      * BUY blends into an existing cost basis, SELL leaves the per-share cost
        basis unchanged and never sells more than is held;
      * cash moves by the filled notional: BUY spends it, SELL returns it (the
        ledger reserved in-flight buy funds inside cash, and kept in-flight
        sells inside holdings — see launch()/recalibrate_portfolio()).
    """
    pending = dict(doc.get("pending_order_ids") or {})
    holdings = {s: dict(h) for s, h in (doc.get("holdings") or {}).items()}
    cash = float(doc.get("cash_remaining") or 0.0)
    log: List[Dict[str, Any]] = []

    for sym, f in facts.items():
        if pending.get(sym) != f.get("client_order_id"):
            continue                                   # already applied elsewhere, or superseded
        status = str(f.get("status", "")).lower()
        if status not in _TERMINAL_STATUSES:
            continue                                   # still working — leave it pending
        qty = float(f.get("filled_qty") or 0.0)
        avg = float(f.get("filled_avg_price") or 0.0)
        side = str(f.get("side", "")).lower()
        del pending[sym]
        entry = {"symbol": sym, "side": side.upper(), "status": status, "filled_qty": qty, "avg_price": avg,
                 "client_order_id": f.get("client_order_id"),
                 "at": datetime.now(timezone.utc).isoformat()}
        if qty > 0 and avg > 0:
            notional = qty * avg
            cur = holdings.get(sym)
            if side == "buy":
                if cur:
                    total = cur["shares"] + qty
                    price = (cur["shares"] * cur["entry_price"] + notional) / total
                else:
                    total, price = qty, avg
                holdings[sym] = {"shares": _shares(total), "entry_price": price, "cost": total * price}
                cash = max(0.0, cash - notional)
            elif side == "sell":
                held = cur["shares"] if cur else 0.0
                sold = min(qty, held)
                if cur and sold > 0:
                    remaining = held - sold
                    if remaining <= 1e-9:
                        del holdings[sym]
                    else:
                        holdings[sym] = {"shares": _shares(remaining), "entry_price": cur["entry_price"],
                                         "cost": remaining * cur["entry_price"]}
                    cash += sold * avg
                    entry["filled_qty"] = sold
                else:
                    entry["note"] = "sell filled but the ledger held no such shares — ledger left unchanged"
        log.append(entry)

    if not log:
        return None
    return {"holdings": holdings, "cash_remaining": cash, "pending_order_ids": pending, "log": log}


def reconcile_portfolio(portfolio_id: str,
                        fetch_order: "Callable[[str], Optional[Dict[str, Any]]]") -> Optional[List[Dict[str, Any]]]:
    """Ask the broker (via `fetch_order(client_order_id)`, which returns the
    facts dict described in resolve_pending, or None/raises when it can't be
    determined) how this portfolio's pending orders ended, and apply the
    terminal ones. Returns the log of what was applied, or None if nothing
    changed. A lookup that fails leaves that order pending — never guessed.

    Write is OPTIMISTIC: it only lands if the document hasn't changed since it
    was read, so two simultaneous page loads can't both apply the same fill.
    The loser simply skips; the next load sees the winner's result."""
    ref = _get_client().collection(_COLLECTION).document(portfolio_id)
    snap = ref.get()
    if not snap.exists:
        return None
    doc = snap.to_dict()
    if doc.get("status") != "open" or not doc.get("pending_order_ids"):
        return None

    facts: Dict[str, Dict[str, Any]] = {}
    for sym, coid in doc["pending_order_ids"].items():
        try:
            f = fetch_order(coid)
        except Exception as e:
            print(f"[portfolio_store] could not look up pending order {coid}: {e} — leaving it pending")
            continue
        if f:
            facts[sym] = {**f, "client_order_id": coid}
    upd = resolve_pending(doc, facts)
    if upd is None:
        return None

    log = (doc.get("reconciliation_log") or []) + upd["log"]
    fields = {"holdings": upd["holdings"], "cash_remaining": upd["cash_remaining"],
              "pending_order_ids": upd["pending_order_ids"], "reconciliation_log": log[-100:]}
    try:
        ref.update(fields, option=_get_client().write_option(last_update_time=snap.update_time))
    except Exception as e:
        # Precondition failed => someone else changed the doc between our read and write (very likely
        # the same reconciliation from a concurrent request). Do nothing; the next read is correct.
        print(f"[portfolio_store] reconcile skipped for {portfolio_id}: {type(e).__name__} (concurrent update)")
        return None
    return upd["log"]
