"""
test_portfolio_page_reconcile.py
=================================
Drives the REAL web_ui Flask app to prove the Portfolio page shows the LATEST
holdings after a recalibration whose orders were still in flight when the
ledger was written — and that recalibrate/close act on the corrected ledger.

Only the outside world is faked: GCP at import time (Vertex, Secret Manager),
Firestore, and Alpaca (trading client, prices, historical bars). Everything in
web_ui.py and portfolio_store.py is the real code.

Run:  python test_portfolio_page_reconcile.py
"""

from __future__ import annotations

import copy
import sys
import types
from types import SimpleNamespace as NS
from unittest import mock


# ---- stub only what web_ui touches at import time ---------------------------------------
class _Payload: data = b"stub"
class _Resp: payload = _Payload()
class _SM:
    def access_secret_version(self, name): return _Resp()
_m = types.ModuleType("google.cloud.secretmanager"); _m.SecretManagerServiceClient = _SM
sys.modules["google.cloud.secretmanager"] = _m
import google.cloud
google.cloud.secretmanager = _m
for _n in ("vertexai", "vertexai.preview", "vertexai.preview.reasoning_engines"):
    sys.modules[_n] = mock.MagicMock()

import web_ui                     # noqa: E402  (real module, real routes)
import portfolio_store as ps      # noqa: E402
import alpaca.trading.client as alpaca_trading_client   # noqa: E402
import simulate                   # noqa: E402
from test_portfolio_reconcile import FakeClient, _Snap   # noqa: E402  (Firestore double with update-time preconditions)


class _StreamClient(FakeClient):
    """FakeClient + collection().stream() for list_portfolios()."""
    def collection(self, name):
        c = self
        base = super().collection(name)

        class _C:
            def document(self_inner, pid): return base.document(pid)
            def stream(self_inner):
                out = []
                for pid, d in c.docs.items():
                    if "/" in pid:                       # a document of another collection
                        continue
                    s = _Snap(d, c.versions[pid]); s.id = pid
                    out.append(s)
                return out
        return _C()


def base_doc(**over):
    d = {"name": "Test PF", "sectors": ["tech_ai"], "capital": 10_000.0, "top_n": 5, "status": "open",
         "created_at": "2026-09-10T15:00:00+00:00", "closed_at": None, "cash_remaining": 1_000.0,
         "holdings": {"AAA": {"shares": 100, "entry_price": 10.0, "cost": 1000.0},
                      "BBB": {"shares": 50, "entry_price": 20.0, "cost": 1000.0}},
         "pending_order_ids": {"AAA": "webui-rc-p1-2026-09-18-AAA-sell", "CCC": "webui-rc-p1-2026-09-18-CCC-buy"},
         "rebalance_history": [], "rebalance_every_days": 21, "next_rebalance_date": "2026-10-09",
         "estimated_1mo_return": None, "errors": []}
    d.update(over)
    return d


class FakeTradingClient:
    """Alpaca as it looks AFTER the market opened: last night's recalibration orders have filled."""
    outage = False
    submitted = []
    _orders = {
        # rebalance sold 40 of AAA and bought 30 of CCC — both filled at the open, after the ledger was written
        "webui-rc-p1-2026-09-18-AAA-sell": NS(id="o1", side="sell", status="filled", filled_qty="40", filled_avg_price="12.00"),
        "webui-rc-p1-2026-09-18-CCC-buy": NS(id="o2", side="buy", status="filled", filled_qty="30", filled_avg_price="25.00"),
    }
    lookups = 0

    def __init__(self, *a, **k): pass

    def get_order_by_client_id(self, coid):
        FakeTradingClient.lookups += 1
        if FakeTradingClient.outage:
            raise ConnectionError("alpaca unreachable")
        if coid in FakeTradingClient._orders:
            return FakeTradingClient._orders[coid]
        raise RuntimeError('{"code":40410000,"message":"order not found"}')

    def submit_order(self, req):
        FakeTradingClient.submitted.append((req.symbol, str(getattr(req.side, "value", req.side)), req.qty))
        return NS(id=f"s{len(FakeTradingClient.submitted)}", client_order_id=req.client_order_id, status="filled",
                  filled_qty=str(req.qty), filled_avg_price="12.00")

    def get_order_by_id(self, oid, *a, **k):
        raise AssertionError("fill was already terminal on submit")

    def cancel_order_by_id(self, oid): pass
    def get_all_positions(self): return []


def install(doc):
    fake = _StreamClient({"p1": doc})
    ps._client = fake
    FakeTradingClient.outage, FakeTradingClient.submitted, FakeTradingClient.lookups = False, [], 0
    web_ui._RECONCILE_LAST.clear()
    web_ui._market_open = lambda: True
    return fake


def ok(msg): print(f"✓ {msg}")


# ---- P0.3 helpers: sign in as the admin and POST to destructive routes ----
# The stubbed Secret Manager returns b"stub" for EVERY secret, so the admin
# token (the recalibrate-token secret) is "stub" in tests.
ADMIN_TOKEN = "stub"


def login(c):
    """Sign the test client in at /login with the admin token."""
    r = c.post("/login", data={"admin_token": ADMIN_TOKEN})
    assert r.status_code == 302, r.status_code
    return r


def admin_post(c, url, data=None, **kw):
    """POST to a destructive route as a signed-in admin, with a valid
    per-session CSRF token. Logs in fresh each call (cheap) so the CSRF
    token is always the current one."""
    login(c)
    with c.session_transaction() as sess:
        tok = sess.get("csrf_token")
    assert tok, "login must issue a CSRF token"
    body = dict(data or {})
    body.setdefault("csrf_token", tok)
    return c.post(url, data=body, **kw)


def recalibrate_now(c, pid="p1"):
    """The manual recalibration's two-step flow (P0.3): first submission
    renders the preview (no orders), second submission with confirm=1
    actually trades. Returns the confirming submission's response."""
    r = admin_post(c, f"/portfolios/{pid}/recalibrate")
    assert r.status_code == 200 and "Preview: recalibrate" in r.get_data(as_text=True), r.status_code
    return admin_post(c, f"/portfolios/{pid}/recalibrate", data={"confirm": "1"})


if __name__ == "__main__":
    alpaca_trading_client.TradingClient = FakeTradingClient
    web_ui._live_prices = lambda syms: {s: 12.0 for s in syms}
    simulate.fetch_price_series = lambda *a, **k: {}              # no network for the chart
    c = web_ui.app.test_client()

    print("TEST 1: THE REPORTED BUG — the page shows the recalibrated holdings, not the stale pre-recalibration ledger")
    fake = install(base_doc())
    html = c.get("/portfolios/p1").get_data(as_text=True)
    doc = fake.docs["p1"]
    assert doc["holdings"]["AAA"]["shares"] == 60, doc["holdings"]
    assert doc["holdings"]["CCC"]["shares"] == 30 and doc["holdings"]["BBB"]["shares"] == 50
    assert doc["pending_order_ids"] == {} and doc["cash_remaining"] == 1_000.0 + 480.0 - 750.0
    holdings_html = html.split("<h2>Holdings</h2>")[1].split("Holdings total")[0]
    assert "/stock/CCC" in holdings_html, "the newly bought CCC must appear in the table"
    import re
    rows = {m[0]: m[1] for m in re.findall(r'/stock/(\w+)">\w+</a></td>\s*<td>(\d+)</td>', holdings_html)}
    assert rows == {"AAA": "60", "BBB": "50", "CCC": "30"}, rows
    assert "still pending fill" not in html
    ok(f"table shows AAA 60 (was 100), BBB 50, CCC 30 (new); pending cleared; cash ${doc['cash_remaining']:,.0f}")

    print("TEST 2: reloading the page does not re-apply the fills or hit Alpaca again")
    n = FakeTradingClient.lookups
    c.get("/portfolios/p1")
    assert FakeTradingClient.lookups == n and fake.docs["p1"]["holdings"]["AAA"]["shares"] == 60
    assert len(fake.docs["p1"]["reconciliation_log"]) == 2
    ok("no pending orders -> no broker calls; AAA still 60 (not 20)")

    print("TEST 3: an Alpaca outage degrades gracefully — the page still loads and shows the ledger as it was")
    fake = install(base_doc())
    FakeTradingClient.outage = True
    r = c.get("/portfolios/p1")
    assert r.status_code == 200 and fake.docs["p1"]["holdings"]["AAA"]["shares"] == 100
    assert fake.docs["p1"]["pending_order_ids"], "orders whose outcome is unknown stay pending"
    FakeTradingClient.outage = False
    c.get("/portfolios/p1")
    assert fake.docs["p1"]["holdings"]["AAA"]["shares"] == 60, "next load after the outage catches up"
    ok("200 during the outage with the old ledger; the next load reconciles")

    print("TEST 4: an order still WORKING at the broker is left pending and not applied")
    fake = install(base_doc())
    FakeTradingClient._orders["webui-rc-p1-2026-09-18-AAA-sell"] = NS(id="o1", side="sell", status="accepted",
                                                                     filled_qty="0", filled_avg_price=None)
    c.get("/portfolios/p1")
    d = fake.docs["p1"]
    assert d["holdings"]["AAA"]["shares"] == 100 and "AAA" in d["pending_order_ids"]
    assert d["holdings"]["CCC"]["shares"] == 30 and "CCC" not in d["pending_order_ids"], "the finished one still applies"
    FakeTradingClient._orders["webui-rc-p1-2026-09-18-AAA-sell"] = NS(id="o1", side="sell", status="filled",
                                                                     filled_qty="40", filled_avg_price="12.00")
    ok("AAA (accepted, unfilled) untouched; CCC (filled) applied in the same pass")

    print("TEST 5: the dropdown / list path reconciles too, and is throttled")
    fake = install(base_doc())
    web_ui._all_portfolios_enriched()
    assert fake.docs["p1"]["holdings"]["AAA"]["shares"] == 60
    n = FakeTradingClient.lookups
    web_ui._all_portfolios_enriched()
    assert FakeTradingClient.lookups == n
    ok("list view sees the reconciled holdings; second call within 30s makes no broker calls")

    print("TEST 6: RECALIBRATE sizes its trades off the CORRECTED ledger (no duplicate rebalance)")
    fake = install(base_doc())
    seen = {}
    class Stop(Exception): pass
    def fake_live_prices(syms):
        seen.setdefault("symbols", sorted(syms)); return {s: 12.0 for s in syms}
    web_ui._live_prices = fake_live_prices
    def fake_recommend(*a, **k):
        seen["doc_at_recommend"] = copy.deepcopy(fake.docs["p1"]); raise Stop()
    orig_recommend, web_ui.recommend = web_ui.recommend, fake_recommend
    try:
        web_ui.recalibrate_portfolio("p1")
    except Stop:
        pass
    finally:
        web_ui.recommend = orig_recommend
        web_ui._live_prices = lambda syms: {s: 12.0 for s in syms}
    assert seen["symbols"] == ["AAA", "BBB", "CCC"], f"must price the reconciled holdings incl. new CCC, got {seen['symbols']}"
    assert seen["doc_at_recommend"]["holdings"]["AAA"]["shares"] == 60
    ok("recalibration started from AAA 60 / BBB 50 / CCC 30 — not from the stale AAA 100 / BBB 50")

    print("TEST 7: CLOSE sells the shares actually held (60), not the stale ledger's 100")
    fake = install(base_doc())
    # P0.3 — liquidation is two server-side steps: the first submission only
    # renders the confirm page and must trade nothing...
    r = admin_post(c, "/portfolios/p1/close")
    assert r.status_code == 200 and "Type the portfolio name" in r.get_data(as_text=True)
    assert not FakeTradingClient.submitted and fake.docs["p1"]["status"] == "open"
    # ...only the second submission, typing the exact portfolio name, sells.
    r = admin_post(c, "/portfolios/p1/close", data={"confirm_name": "Test PF"})
    sells = {sym: qty for sym, side, qty in FakeTradingClient.submitted if side == "sell"}
    assert sells == {"AAA": 60, "BBB": 50, "CCC": 30}, sells
    assert fake.docs["p1"]["status"] == "closed"
    ok(f"sell orders placed: {sells}")

    print("\nAll Portfolio-page reconciliation checks passed.")
