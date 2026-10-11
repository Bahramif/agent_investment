"""
test_whatsapp.py
=================
Drives the REAL web_ui Flask app to prove WhatsApp notifications are wired
end to end: settings need the token and a valid international number, the
test button sends through the Cloud API shape, every recalibration (manual,
scheduled, mid-day) notifies a portfolio that opted in, news-watch flags
notify, dry runs and opted-out portfolios don't, and a WhatsApp failure never
blocks a recalibration. Each attempt is logged and shown on the page.

Only the outside world is faked (same harness as test_proactive_recal.py):
GCP at import, Firestore, Alpaca, recommend(), headlines, Gemini — and the
HTTP call to graph.facebook.com.

Run:  python test_whatsapp.py
"""

from __future__ import annotations

import json

import test_proactive_recal as base                 # stubs GCP, imports the real web_ui, fake world helpers
from test_portfolio_page_reconcile import FakeTradingClient, install

web_ui = base.web_ui
import whatsapp_notify                              # noqa: E402

TOKEN = base.TOKEN


def ok(msg): print(f"✓ {msg}")


class FakeGraph:
    """Stands in for POST graph.facebook.com/{v}/{phone_id}/messages."""
    def __init__(self):
        self.sent, self.fail = [], None

    def __call__(self, url, headers, body):
        if self.fail:
            return 400, json.dumps({"error": {"message": self.fail, "code": 131026}}).encode()
        self.sent.append((url, headers, json.loads(body)))
        return 200, json.dumps({"messages": [{"id": f"wamid.{len(self.sent)}"}]}).encode()


def setup(graph, creds=("TKN", "PNID")):
    whatsapp_notify._urllib_post = graph                       # notify()'s default http
    whatsapp_notify.notify.__defaults__ = (graph, False)
    web_ui._WA_CREDS, web_ui._WA_CHECKED = creds, 0.0


if __name__ == "__main__":
    import alpaca.trading.client as alpaca_trading_client
    import time
    alpaca_trading_client.TradingClient = FakeTradingClient
    web_ui._live_prices = lambda syms: {s: 12.0 for s in syms}
    base.harness.simulate.fetch_price_series = lambda *a, **k: {}
    c = web_ui.app.test_client()

    print("SCENARIO A: settings need the token and a real international number")
    fake = install(base.doc())
    g = FakeGraph(); setup(g)
    r = c.post("/portfolios/p1/whatsapp", data={"enabled": "1", "number": "+1 416 555 0100"})
    assert "token" in r.headers["Location"] and "whatsapp_notify" not in fake.docs["p1"]
    r = c.post("/portfolios/p1/whatsapp", data={"enabled": "1", "number": "416-555-0100", "token": "stub"})
    assert "Not+saved" in r.headers["Location"] or "Not%20saved" in r.headers["Location"]
    r = c.post("/portfolios/p1/whatsapp", data={"enabled": "1", "token": "stub"})
    assert "number" in r.headers["Location"] and "whatsapp_notify" not in fake.docs["p1"], "can't enable with no number"
    c.post("/portfolios/p1/whatsapp", data={"enabled": "1", "number": "+1 (416) 555-0100", "token": "stub"})
    assert fake.docs["p1"]["whatsapp_notify"] is True and fake.docs["p1"]["whatsapp_number"] == "14165550100"
    html = c.get("/portfolios/p1").get_data(as_text=True)
    assert "📱 WhatsApp notifications" in html and ">ON<" in html.split("📱 WhatsApp notifications")[1] and "+14…100" in html
    assert "14165550100" not in html, "the full number is never rendered"
    ok("no token -> refused; bad number / no number -> not saved; valid -> ON, shown masked")

    print("SCENARIO B: the test button sends the approved template through the Cloud API")
    r = c.post("/portfolios/p1/whatsapp/test", data={"token": "stub"})
    url, headers, body = g.sent[-1]
    assert url == "https://graph.facebook.com/v26.0/PNID/messages" and headers["Authorization"] == "Bearer TKN"
    assert body["to"] == "14165550100" and body["template"]["name"] == "meridian_portfolio_update"
    assert "Test+sent" in r.headers["Location"] or "Test%20sent" in r.headers["Location"]
    assert c.post("/portfolios/p1/whatsapp/test", data={}).status_code == 302 and len(g.sent) == 1, "no token, no send"
    ok("POST v26.0/PNID/messages, template meridian_portfolio_update, to the stored number")

    print("SCENARIO C: a mid-day news-watch recalibration notifies with the trades and Gemini's reason")
    st = base.world(news=base.NEWS)
    n = len(g.sent)
    c.post("/portfolios/proactive_check", headers=TOKEN)
    assert len(g.sent) == n + 1
    p1, p2, p3 = [x["text"] for x in g.sent[-1][2]["template"]["components"][0]["parameters"]]
    assert p1 == "Test PF" and p2.startswith("📰 Mid-day news-watch recalibration:") and "SEC accounting probe" in p3
    assert "Sold" in p3 or "Bought" in p3
    ok(f"{p2} | {p3[:80]}…")

    print("SCENARIO D: same-day re-run -> FLAG notification (Gemini wanted to trade; rule stopped it)")
    web_ui._headlines_many = lambda syms, days: {"BBB": ["BBB recalls its flagship product"]}
    n = len(g.sent)
    c.post("/portfolios/proactive_check", headers=TOKEN)
    p2 = g.sent[-1][2]["template"]["components"][0]["parameters"][1]["text"]
    assert len(g.sent) == n + 1 and "NOT traded" in p2
    ok(p2)

    print("SCENARIO E: dry runs don't notify; opted-out portfolios don't; manual + scheduled do")
    fake = install(base.doc(whatsapp_notify=True, whatsapp_number="14165550100"))
    base.world(news=base.NEWS)
    n = len(g.sent)
    c.post("/portfolios/proactive_check?dry_run=1", headers=TOKEN)
    assert len(g.sent) == n, "dry run: flagged, no message"
    c.post("/portfolios/p1/recalibrate")
    assert g.sent[-1][2]["template"]["components"][0]["parameters"][1]["text"].startswith("👆 Manual recalibration")
    fake = install(base.doc(whatsapp_notify=True, whatsapp_number="14165550100",
                            next_rebalance_date=base.TODAY.isoformat()))
    base.world()
    c.post("/portfolios/recalibrate_due", headers=TOKEN)
    assert g.sent[-1][2]["template"]["components"][0]["parameters"][1]["text"].startswith("📅 Scheduled recalibration")
    fake = install(base.doc(whatsapp_notify=False, whatsapp_number="14165550100"))
    base.world()
    n = len(g.sent)
    c.post("/portfolios/p1/recalibrate")
    assert len(g.sent) == n and not fake.docs["p1"].get("notifications"), "opted out: nothing sent, nothing logged"
    ok("dry run silent; 👆 manual and 📅 scheduled both notify; OFF portfolio silent")

    print("SCENARIO F: WhatsApp failing never blocks the recalibration; the failure is logged and shown")
    fake = install(base.doc(whatsapp_notify=True, whatsapp_number="14165550100"))
    base.world()
    g.fail = "Message failed to send because the recipient is not a WhatsApp user"
    c.post("/portfolios/p1/recalibrate")
    d = fake.docs["p1"]
    assert d["rebalance_history"][-1]["trigger"] == "manual", "the recalibration itself completed"
    log = d["notifications"][-1]
    assert log["sent"] is False and "131026" in log["error"] and log["to"] == "+14…100"
    html = c.get("/portfolios/p1").get_data(as_text=True)
    assert "⚠️" in html and "not a WhatsApp user" in html
    g.fail = None
    setup(g, creds=None)
    web_ui._WA_CHECKED = time.monotonic()               # "secrets missing, checked just now"
    c.post("/portfolios/p1/recalibrate")
    assert "credentials" in fake.docs["p1"]["notifications"][-1]["error"]
    ok("recipient error and missing credentials both logged; recalibrations unaffected")
    print("\nAll WhatsApp notification scenarios passed.")
