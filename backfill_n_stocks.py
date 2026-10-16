"""
backfill_n_stocks.py
=====================
One-time fix for portfolios launched before `n_stocks` existed.

Each portfolio must keep the number of stocks it was LAUNCHED with
(recalibrate_portfolio now re-selects exactly `n_stocks` names). New launches
record it; older ones can't be trusted for `top_n`, because a launch-form bug
saved every portfolio with the default 12 whatever was chosen on the
recommendation page. The truthful record is Alpaca's: the launch placed one
BUY per stock, client_order_id "webui-<batch>-<SYMBOL>-BUY", all with one
batch id, submitted moments before the portfolio document was created.

For every portfolio without `n_stocks`, this finds the launch batch CREATED
(accepted by Alpaca) in the 15 minutes before the portfolio's created_at (the latest such batch) and counts the
distinct symbols that batch actually bought (filled, partially filled, or
still open — not rejected/canceled with nothing filled).

    python backfill_n_stocks.py            # dry run: prints what it would write
    python backfill_n_stocks.py --apply    # writes n_stocks (+ n_stocks_source) to Firestore
"""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

LAUNCH_ID = re.compile(r"^webui-([0-9a-f]{10})-([A-Z.]+)-BUY$")
WINDOW = timedelta(minutes=15)


def launch_count(orders: List[Any], created_at: datetime) -> Optional[Dict[str, Any]]:
    """PURE. orders: objects with client_order_id, created_at (aware — when
    Alpaca ACCEPTED the order, i.e. the moment of the launch; submitted_at is
    later for after-hours launches, when the order is routed at the next
    session), status, filled_qty. Returns {"n", "batch", "symbols"} or None."""
    batches: Dict[str, Dict[str, Any]] = {}
    for o in orders:
        m = LAUNCH_ID.match(str(getattr(o, "client_order_id", "") or ""))
        if not m:
            continue
        sub = getattr(o, "created_at", None)
        if sub is None or not (created_at - WINDOW <= sub <= created_at + timedelta(minutes=1)):
            continue
        status = str(getattr(getattr(o, "status", ""), "value", getattr(o, "status", ""))).lower()
        filled = float(getattr(o, "filled_qty", 0) or 0)
        if status in ("rejected", "canceled", "expired") and filled <= 0:
            continue
        b = batches.setdefault(m.group(1), {"latest": sub, "symbols": set()})
        b["symbols"].add(m.group(2))
        b["latest"] = max(b["latest"], sub)
    if not batches:
        return None
    batch, info = max(batches.items(), key=lambda kv: kv[1]["latest"])
    return {"n": len(info["symbols"]), "batch": batch, "symbols": sorted(info["symbols"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    import portfolio_store as ps
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    sec = lambda n: subprocess.run(["gcloud", "secrets", "versions", "access", "latest", f"--secret={n}"],
                                   capture_output=True, text=True, check=True).stdout.strip()
    tc = TradingClient(sec("alpaca-api-key"), sec("alpaca-secret-key"), paper=True)
    for p in ps.list_portfolios():
        if p.get("n_stocks"):
            print(f"{p['name'][:30]:30} already has n_stocks={p['n_stocks']}")
            continue
        created = datetime.fromisoformat(p["created_at"])
        # Alpaca filters this query on SUBMISSION time, which for an after-hours launch is the next
        # session — so ask for a day after creation and match on created_at in launch_count().
        orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=created - WINDOW,
                                                until=created + timedelta(hours=24), limit=500))
        found = launch_count(orders, created)
        if not found:
            print(f"{p['name'][:30]:30} ({p.get('status')}) — no launch orders found; leaving it unset "
                  f"(recalibration falls back to the current number of holdings: {len(p.get('holdings') or {})})")
            continue
        print(f"{p['name'][:30]:30} ({p.get('status')}) launched with {found['n']} stocks {found['symbols']} "
              f"[batch {found['batch']}; saved top_n was {p.get('top_n')}]")
        if a.apply:
            ps._get_client().collection(ps._COLLECTION).document(p["id"]).update(
                {"n_stocks": found["n"], "n_stocks_source": f"alpaca launch batch {found['batch']}"})
    print("\n(dry run — nothing written; re-run with --apply)" if not a.apply else "\nwritten.")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        raise SystemExit(main(sys.argv[1:]))
    from datetime import timezone
    from types import SimpleNamespace as NS
    t0 = datetime(2026, 9, 16, 15, 0, tzinfo=timezone.utc)
    O = lambda coid, mins, status="filled", qty=5: NS(client_order_id=coid, created_at=t0 + timedelta(minutes=mins),
                                                       submitted_at=t0 + timedelta(hours=7), status=status, filled_qty=qty)
    orders = [O("webui-aaaaaaaaaa-NVDA-BUY", -2), O("webui-aaaaaaaaaa-MSFT-BUY", -2), O("webui-aaaaaaaaaa-AMD-BUY", -2),
              O("webui-aaaaaaaaaa-INTC-BUY", -2, "rejected", 0),                       # never bought
              O("webui-bbbbbbbbbb-XOM-BUY", -40),                                      # an older launch
              O("webui-rc-p1-2026-09-16-NVDA-BUY", -1), O("webui-close-p1-AMD-SELL", -1)]
    r = launch_count(orders, t0)
    assert r == {"n": 3, "batch": "aaaaaaaaaa", "symbols": ["AMD", "MSFT", "NVDA"]}, r
    assert launch_count(orders, t0 + timedelta(hours=2)) is None
    print("✓ counts the launch batch created just before the portfolio (even if routed hours later); "
          "ignores rejected, older launches, rc/close orders")
