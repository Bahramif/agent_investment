"""
test_proactive_recal.py
========================
Drives the REAL web_ui Flask app through /portfolios/proactive_check — the
endpoint the mid-day Cloud Scheduler job hits — to prove the news watch is
wired end to end: pending fills are reconciled first, the token is enforced,
a material-news verdict runs a REAL recalibration (orders placed, history
tagged "proactive" with the reason), quiet days and dry runs never trade, and
every check is logged and rendered on the portfolio page.

Only the outside world is faked (same harness as
test_portfolio_page_reconcile.py): GCP at import, Firestore, Alpaca, the
deployed agent's recommend(), headlines, and Claude.

Run:  python test_proactive_recal.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import test_portfolio_page_reconcile as harness      # stubs GCP, imports the real web_ui
from test_portfolio_page_reconcile import FakeTradingClient, base_doc, install

web_ui = harness.web_ui
import alpaca.trading.client as alpaca_trading_client   # noqa: E402
import anthropic                                         # noqa: E402

TODAY = datetime.now(timezone.utc).date()
TOKEN = {"X-Recalibrate-Token": "stub"}                  # what the stubbed Secret Manager returns


def doc(**over):
    """A portfolio launched 20 days ago, next scheduled recalibration 12 days
    out, with its owner's news-watch switch ON."""
    fields = {"created_at": (TODAY - timedelta(days=20)).isoformat() + "T15:00:00+00:00",
              "next_rebalance_date": (TODAY + timedelta(days=12)).isoformat(), "news_watch": True}
    return base_doc(**{**fields, **over})


class FakeClaude:
    def __init__(self, verdict):
        self.verdict, self.calls, self.messages, self.client_kwargs = verdict, [], self, None
    def bind(self, kwargs):
        self.client_kwargs = kwargs          # how web_ui constructed the Vertex client
        return self
    def create(self, **kw):
        self.calls.append(kw)
        return NS(stop_reason="tool_use", content=[NS(type="tool_use", name="decide_recalibration", input=self.verdict)])


YES = {"recalibrate": True, "confidence": 0.92, "urgency": "high", "key_symbols": ["AAA"],
       "reason": "AAA disclosed an SEC accounting probe and its CFO resigned — the thesis for the largest holding changed."}
NEWS = {"AAA": ["AAA discloses SEC probe into revenue recognition", "AAA CFO resigns"]}


def world(news=None, verdict=YES, market_open=True):
    state = {"recommend_calls": 0, "claude": FakeClaude(verdict)}
    web_ui._intraday_quotes = lambda syms: {s: {"price": 12.0, "prev_close": 12.2} for s in syms}
    web_ui._headlines = lambda s, days: (news or {}).get(s, [])
    web_ui._market_open = lambda: market_open
    anthropic.AnthropicVertex = lambda **kw: state["claude"].bind(kw)

    def fake_recommend(sectors, top_n):
        state["recommend_calls"] += 1
        return {"final_weights": {"BBB": 0.5, "CCC": 0.4}, "status": "ok"}
    web_ui.recommend = fake_recommend
    return state


def ok(msg): print(f"✓ {msg}")


if __name__ == "__main__":
    alpaca_trading_client.TradingClient = FakeTradingClient
    web_ui._live_prices = lambda syms: {s: 12.0 for s in syms}
    harness.simulate.fetch_price_series = lambda *a, **k: {}
    c = web_ui.app.test_client()

    print("SCENARIO A: the endpoint is token-protected — it can place real orders")
    fake = install(doc())
    st = world(news=NEWS)
    assert c.post("/portfolios/proactive_check").status_code == 403
    assert c.post("/portfolios/proactive_check", headers={"X-Recalibrate-Token": "wrong"}).status_code == 403
    assert not FakeTradingClient.submitted and st["recommend_calls"] == 0 and not st["claude"].calls
    ok("403 without / with a wrong token; nothing assessed, nothing traded")

    print("SCENARIO B: material news -> in-flight fills reconciled, then a REAL proactive recalibration")
    fake = install(doc())
    st = world(news=NEWS)
    r = c.post("/portfolios/proactive_check", headers=TOKEN).get_json()
    d = fake.docs["p1"]
    assert r["results"][0]["action"] == "recalibrated" and r["results"][0]["source"] == "claude", r
    assert st["recommend_calls"] == 1 and FakeTradingClient.submitted, "the real recalibration ran and placed orders"
    ev = d["rebalance_history"][-1]
    assert ev["trigger"] == "proactive" and "SEC accounting probe" in ev["reason"]
    assert d["next_rebalance_date"] == (TODAY + timedelta(days=21)).isoformat(), "the cadence restarts from today"
    assert d["proactive_checks"][-1]["action"] == "recalibrated" and d["proactive_checks"][-1]["recalibration"]["orders"] > 0
    assert st["claude"].client_kwargs == {"project_id": web_ui.PROJECT, "region": "global"}, st["claude"].client_kwargs
    prompt = st["claude"].calls[0]["messages"][0]["content"]
    assert "AAA discloses SEC probe" in prompt and "AAA: " in prompt and "CCC: " in prompt, \
        "the model saw the RECONCILED holdings (CCC was bought by the in-flight order)"
    ok(f"orders {FakeTradingClient.submitted}; history tagged proactive; next scheduled date pushed to {d['next_rebalance_date']}")

    print("SCENARIO C: running again the same day does nothing — guardrails, no model call, no orders")
    n_orders, n_calls = len(FakeTradingClient.submitted), len(st["claude"].calls)
    r = c.post("/portfolios/proactive_check", headers=TOKEN).get_json()
    assert r["results"][0]["action"] == "skipped" and "0 day(s) ago" in r["results"][0]["reason"]
    assert len(FakeTradingClient.submitted) == n_orders and len(st["claude"].calls) == n_calls
    ok(r["results"][0]["reason"])

    print("SCENARIO D: dry run and a closed market FLAG instead of trading")
    for query, kw, needle in (("?dry_run=1", {}, "dry run"), ("", {"market_open": False}, "market closed")):
        fake = install(doc())
        st = world(news=NEWS, **kw)
        r = c.post("/portfolios/proactive_check" + query, headers=TOKEN).get_json()
        assert r["results"][0]["action"] == "flagged" and needle in r["results"][0]["reason"], r
        assert st["recommend_calls"] == 0 and not [x for x in FakeTradingClient.submitted]
        assert fake.docs["p1"]["proactive_checks"][-1]["action"] == "flagged"
    ok("dry_run=1 and market-closed both flagged, zero orders, still logged")

    print("SCENARIO E: a quiet day -> hold, no model call, logged")
    # on target (AAA 1200 / BBB 600 / cash 1000 at $12) and nothing in flight: no drift flag either
    fake = install(doc(pending_order_ids={}, rebalance_history=[
        {"date": (TODAY - timedelta(days=10)).isoformat(), "target_weights": {"AAA": 0.43, "BBB": 0.21}}]))
    st = world(news={})
    r = c.post("/portfolios/proactive_check", headers=TOKEN).get_json()
    assert r["results"][0]["action"] == "hold" and not st["claude"].calls and st["recommend_calls"] == 0
    ok(r["results"][0]["reason"])

    print("SCENARIO F: the model says wait -> hold with its reason")
    fake = install(doc())
    st = world(news=NEWS, verdict={"recalibrate": False, "confidence": 0.8, "urgency": "low",
                                   "reason": "Probe is preliminary; next scheduled rebalance is close enough."})
    r = c.post("/portfolios/proactive_check", headers=TOKEN).get_json()
    assert r["results"][0]["action"] == "hold" and "preliminary" in r["results"][0]["reason"]
    assert not FakeTradingClient.submitted
    ok("held")

    print("SCENARIO G: the page shows the news watch and tags each recalibration with its trigger")
    fake = install(doc())
    st = world(news=NEWS)
    c.post("/portfolios/proactive_check", headers=TOKEN)
    html = c.get("/portfolios/p1").get_data(as_text=True)
    assert "Mid-day news watch" in html and "recalibrated" in html and "SEC accounting probe" in html
    history = html.split("<h2>Recalibration history</h2>")[1].split("</table>")[0]
    assert "📰 Mid-day news watch" in history and " ET</div>" in history, "type label + New York time on the row"
    ok("news-watch table rendered; history row labelled '📰 Mid-day news watch' with its ET time")

    print("SCENARIO H: the scheduled due-check and the manual button are tagged too")
    fake = install(doc(next_rebalance_date=TODAY.isoformat()))
    world()
    c.post("/portfolios/recalibrate_due", headers=TOKEN)
    assert fake.docs["p1"]["rebalance_history"][-1]["trigger"] == "scheduled"
    fake = install(doc())
    world()
    c.post("/portfolios/p1/recalibrate")
    assert fake.docs["p1"]["rebalance_history"][-1]["trigger"] == "manual"
    fake = install(doc(rebalance_history=[
        {"date": "2026-09-01", "orders": [], "portfolio_value_before": 1.0},                              # pre-label
        {"date": "2026-09-08", "at": "2026-09-08T21:00:00+00:00", "trigger": "scheduled", "orders": [],
         "portfolio_value_before": 1.0},
        {"date": "2026-09-10", "at": "2026-09-10T16:31:00+00:00", "trigger": "proactive", "reason": "probe",
         "orders": [], "portfolio_value_before": 1.0}]))
    world()
    history = c.get("/portfolios/p1").get_data(as_text=True).split("<h2>Recalibration history</h2>")[1].split("</table>")[0]
    for needle in ("📅 Scheduled", "17:00 ET", "📰 Mid-day news watch", "12:31 ET", "Not recorded", "before labels"):
        assert needle in history, needle
    assert history.index("2026-09-10") < history.index("2026-09-08") < history.index("2026-09-01"), "newest first"
    ok("scheduled / manual")

    print("SCENARIO I: the owner's switch — OFF is never checked; the page toggles it")
    fake = install(doc(news_watch=False))
    st = world(news=NEWS)
    r = c.post("/portfolios/proactive_check", headers=TOKEN).get_json()
    assert r["checked"] == 0 and not st["claude"].calls and not FakeTradingClient.submitted
    assert "proactive_checks" not in fake.docs["p1"], "an OFF portfolio leaves no trace"
    html = c.get("/portfolios/p1").get_data(as_text=True)
    assert "Mid-day news watch:" in html and ">OFF<" in html and "Turn on" in html
    assert c.post("/portfolios/p1/news_watch", data={"enabled": "1"}).status_code == 302
    assert fake.docs["p1"]["news_watch"] is True and fake.docs["p1"]["news_watch_changed_at"]
    assert ">ON<" in c.get("/portfolios/p1").get_data(as_text=True)
    r = c.post("/portfolios/proactive_check", headers=TOKEN).get_json()
    assert r["checked"] == 1 and r["results"][0]["action"] == "recalibrated"
    c.post("/portfolios/p1/news_watch", data={"enabled": "0"})
    assert fake.docs["p1"]["news_watch"] is False
    ok("off -> not checked, no log; Turn on -> next run acts; Turn off -> back off")

    print("SCENARIO J: the launch form's checkbox sets the switch on the new portfolio")
    import portfolio_store as ps
    seen = {}
    real_create = ps.create_portfolio
    ps.create_portfolio = lambda **kw: seen.update(kw) or "new1"
    web_ui._LAST_CONTEXT.clear()
    web_ui._LAST_CONTEXT.update({"rows": [{"name": "AAA", "price": 12.0, "shares": 10, "action": "BUY",
                                           "weight": 1.0, "action_notional": 120.0}],
                                 "expected_return": 0.01, "batch_id": "b1"})
    install(doc())
    for form, want in (({"news_watch": "1"}, True), ({}, False)):
        seen.clear()
        c.post("/execute", data={"sectors": "tech_ai", "capital": "1000", "top_n": "5", "batch_id": "b1",
                                 "rebalance_cadence": "21", **form})
        assert seen.get("news_watch") is want, (form, seen.get("news_watch"))
    ps.create_portfolio = real_create
    ok("checked -> news_watch=True, unchecked -> False")

    print("\nAll proactive-recalibration integration scenarios passed.")
