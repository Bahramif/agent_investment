"""
test_proactive_recal.py
========================
Drives the REAL web_ui Flask app through /portfolios/proactive_check — the
endpoint the mid-day Cloud Scheduler job hits — to prove the news watch is
wired end to end: pending fills are reconciled first, the token is enforced,
a material-news verdict runs a REAL recalibration (orders placed, history
tagged "proactive" with the reason, only the named stock sold), quiet days and dry runs never trade, and
every check is logged and rendered on the portfolio page.

Only the outside world is faked (same harness as
test_portfolio_page_reconcile.py): GCP at import, Firestore, Alpaca, the
deployed agent's recommend(), headlines, and Claude.

Run:  python test_proactive_recal.py
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import test_portfolio_page_reconcile as harness      # stubs GCP, imports the real web_ui
from test_portfolio_page_reconcile import FakeTradingClient, base_doc, install

web_ui = harness.web_ui
import alpaca.trading.client as alpaca_trading_client   # noqa: E402
import os                                                # noqa: E402
import anthropic                                         # noqa: E402
import llm                                               # noqa: E402

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
        return NS(stop_reason="tool_use", model="fake-model",
                  content=[NS(type="tool_use", name="decide_recalibration", input=self.verdict)])


YES = {"recalibrate": True, "confidence": 0.92, "urgency": "high", "key_symbols": ["AAA"], "exit_symbols": ["AAA"],
       "reason": "AAA disclosed an SEC accounting probe and its CFO resigned — the thesis for the largest holding changed."}
NEWS = {"AAA": ["AAA discloses SEC probe into revenue recognition", "AAA CFO resigns"]}


def world(news=None, verdict=YES, market_open=True):
    state = {"recommend_calls": 0, "claude": FakeClaude(verdict)}
    web_ui._intraday_quotes = lambda syms: {s: {"price": 12.0, "prev_close": 12.2} for s in syms}
    web_ui._headlines = lambda s, days: (news or {}).get(s, [])
    state["batched_news"] = []
    def many(syms, days):
        state["batched_news"].append(sorted(syms))
        return {s: (news or {}).get(s, []) for s in syms}
    web_ui._headlines_many = many
    web_ui._market_open = lambda: market_open
    web_ui._daily_closes = lambda syms, start: {}       # no price history unless a scenario sets it
    web_ui._SCORECARD.update(at=-1e9, value=None)    # the harness's fake Secret Manager returns "stub" for ANY secret; default
    # these worlds to "no anthropic-api-key" (absent, checked just now)
    web_ui._ANTHROPIC_KEY, web_ui._ANTHROPIC_KEY_CHECKED = None, time.monotonic()
    os.environ.pop("MERIDIAN_LLM_PROVIDER", None)                                   # default provider: gemini
    llm.GeminiClaudeShim = lambda **kw: state["claude"].bind(dict(kw, via="gemini"))
    anthropic.AnthropicVertex = lambda **kw: state["claude"].bind(dict(kw, via="vertex"))
    anthropic.Anthropic = lambda **kw: state["claude"].bind(dict(kw, via="anthropic"))

    state["contexts"] = []

    def fake_recommend(sectors, top_n, context=None):
        state["recommend_calls"] += 1
        state["contexts"].append(context or {})
        return {"final_weights": {"BBB": 0.4, "CCC": 0.3, "DDD": 0.2}, "status": "ok"}
    web_ui.recommend = fake_recommend
    return state


def ok(msg): print(f"✓ {msg}")


if __name__ == "__main__":
    alpaca_trading_client.TradingClient = FakeTradingClient
    web_ui._live_prices = lambda syms: {s: 12.0 for s in syms}
    harness.simulate.fetch_price_series = lambda *a, **k: {}
    c = web_ui.app.test_client()
    REAL_RECOMMEND = web_ui.recommend
    import portfolio_store as ps_store

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
    assert st["claude"].client_kwargs == {"project": web_ui.PROJECT, "via": "gemini"}, \
        "default provider -> Gemini on this project"
    assert d["proactive_checks"][-1]["review"]["model"] == "fake-model", "the model that decided is recorded"
    prompt = st["claude"].calls[0]["messages"][0]["content"]
    assert "AAA discloses SEC probe" in prompt and "- AAA: " in prompt and "- CCC: " in prompt, \
        "the model saw the RECONCILED holdings (CCC was bought by the in-flight order)"
    assert {s for s, _, _ in FakeTradingClient.submitted} == {"AAA", "DDD"}, \
        ("targeted: AAA sold, its slot filled, BBB and CCC not touched", FakeTradingClient.submitted)
    assert d["holdings"]["BBB"]["shares"] == 50 and d["holdings"]["CCC"]["shares"] == 30 and "AAA" not in d["holdings"]
    assert st["contexts"][0]["blocked"] == ["AAA"] and "AAA" not in st["contexts"][0]["held"]
    ok(f"orders {FakeTradingClient.submitted}; history tagged proactive; next scheduled date pushed to {d['next_rebalance_date']}")

    print("SCENARIO C: running again the same day still REVIEWS, but a 'recalibrate' verdict is flagged, not traded")
    n_orders, n_calls = len(FakeTradingClient.submitted), len(st["claude"].calls)
    web_ui._headlines_many = lambda syms, days: {"BBB": ["BBB recalls its flagship product"]}   # AAA was sold in B
    st["claude"].verdict = {**YES, "exit_symbols": ["BBB"]}
    r = c.post("/portfolios/proactive_check", headers=TOKEN).get_json()
    assert r["results"][0]["action"] == "flagged" and "already recalibrated today" in r["results"][0]["reason"], r
    assert len(st["claude"].calls) == n_calls + 1, "the model reviewed the news again"
    assert len(FakeTradingClient.submitted) == n_orders, "no second rebalance on the same date"
    ok(r["results"][0]["reason"][-90:])

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
    assert "📰 Mid-day" in history and " ET</div>" in history, "type label + New York time on the row"
    from universe import EQUITIES
    n_tech = sum(1 for _, path, _ in EQUITIES if path.endswith(".tech_ai"))
    assert "other tech_ai stocks" in html and f"+ {n_tech} other" in html, "coverage line: holdings + sector pool"
    assert len(st["batched_news"]) >= 1 and len(st["batched_news"][0]) >= n_tech, \
        "the news watch's batched request covers the holdings AND the whole tech_ai pool"
    assert all(len(x) < n_tech for x in st["batched_news"][1:]), \
        "any later request is the pre-trade reviewer's, for just the stocks being traded"
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
    for needle in ("📅 Scheduled", "17:00 ET", "📰 Mid-day", "12:31 ET", "Not recorded", "probe"):
        assert needle in history, needle
    assert history.index("2026-09-10") < history.index("2026-09-08") < history.index("2026-09-01"), "newest first"
    ok("scheduled / manual")

    print("SCENARIO H2: MERIDIAN_LLM_PROVIDER=anthropic -> Anthropic API with the key, else Claude on Vertex")
    for key, want in (("sk-ant-test", {"api_key": "sk-ant-test", "via": "anthropic"}),
                      (None, {"project_id": web_ui.PROJECT, "region": "global", "via": "vertex"})):
        fake = install(doc())
        st = world(news=NEWS)
        os.environ["MERIDIAN_LLM_PROVIDER"] = "anthropic"
        web_ui._ANTHROPIC_KEY, web_ui._ANTHROPIC_KEY_CHECKED = key, (-1e9 if key else time.monotonic())
        try:
            r = c.post("/portfolios/proactive_check", headers=TOKEN).get_json()
        finally:
            os.environ.pop("MERIDIAN_LLM_PROVIDER", None)
            web_ui._ANTHROPIC_KEY, web_ui._ANTHROPIC_KEY_CHECKED = None, time.monotonic()
        assert r["results"][0]["source"] == "claude" and st["claude"].client_kwargs == want, st["claude"].client_kwargs
    ok("anthropic provider: key -> Anthropic(api_key=...), no key -> AnthropicVertex")

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

    print("SCENARIO K: EXITS are targeted: each slot goes to the best available name, the rest stays as it is")
    fake = install(doc())                            # after reconciling: AAA 60, BBB 50, CCC 30 -> a 3-stock book
    st = world(news=NEWS, verdict={**YES, "exit_symbols": ["AAA", "BBB"]})
    asked = []
    def ranked(sectors, top_n, context=None):
        asked.append((top_n, context))
        return {"final_weights": {"CCC": 0.4, "DDD": 0.35, "EEE": 0.25}, "status": "ok",
                "book_notes": {"DDD": {"conviction": 0.9}, "EEE": {"conviction": 0.8}}}
    web_ui.recommend = ranked
    r = c.post("/portfolios/proactive_check", headers=TOKEN).get_json()
    assert r["results"][0]["action"] == "recalibrated" and "Selling: AAA, BBB." in r["results"][0]["reason"], r
    assert asked == [(3, {"held": ["CCC"], "blocked": ["AAA", "BBB"]})], asked
    d = fake.docs["p1"]
    ev = d["rebalance_history"][-1]
    assert ev["excluded"] == ["AAA", "BBB"] and set(ev["target_weights"]) == {"CCC", "DDD", "EEE"}, ev["target_weights"]
    trades = [(sym, side) for sym, side, _ in FakeTradingClient.submitted]
    assert "CCC" not in {s for s, _ in trades} and d["holdings"]["CCC"]["shares"] == 30, "the third holding isn't touched"
    assert [side for _, side in trades] == ["sell", "sell", "buy", "buy"], ("sells go first", trades)
    assert d["holdings"]["DDD"]["entry_date"] == TODAY.isoformat(), "'since purchase' starts at the purchase"
    ok(f"orders {trades}")

    print("SCENARIO L: what was sold can't come back for 30 days, whatever sold it; a bad agent answer can't undo that")
    d["rebalance_history"][-1]["date"] = (TODAY - timedelta(days=10)).isoformat()   # no longer today
    FakeTradingClient.submitted = []
    seen_ctx = []
    def stale_agent(sectors, top_n, context=None):                      # an agent that ignores the block
        seen_ctx.append(context)
        return {"final_weights": {"AAA": 0.4, "CCC": 0.3, "DDD": 0.3}, "status": "ok"}
    web_ui.recommend = stale_agent
    c.post("/portfolios/p1/recalibrate")
    assert seen_ctx[0]["blocked"] == ["AAA", "BBB"], seen_ctx
    assert "AAA" not in {s for s, side, _ in FakeTradingClient.submitted if side == "buy"}, FakeTradingClient.submitted
    assert "AAA" not in fake.docs["p1"]["holdings"]
    ok("blocked names reach the agent, and are dropped here even if it returns them")

    print("SCENARIO M: nothing trades with the market closed; a due portfolio stays due for the next run")
    fake = install(doc(next_rebalance_date=TODAY.isoformat(), pending_order_ids={}))
    st = world(market_open=False)
    r = c.post("/portfolios/recalibrate_due", headers=TOKEN).get_json()
    assert "market is closed" in r["results"][0]["error"] and not FakeTradingClient.submitted, r
    assert fake.docs["p1"]["next_rebalance_date"] == TODAY.isoformat() and st["recommend_calls"] == 0
    html = c.post("/portfolios/p1/recalibrate").get_data(as_text=True)
    assert "market is closed" in html, "the button says why nothing happened"
    ok("deferred; the manual button shows the reason")

    print("SCENARIO N: a small change doesn't trade; closing a position always does")
    fake = install(doc(pending_order_ids={}, cash_remaining=0.0,
                       holdings={"AAA": {"shares": 500, "entry_price": 10.0, "cost": 5000.0},
                                 "BBB": {"shares": 500, "entry_price": 10.0, "cost": 5000.0},
                                 "ZZZ": {"shares": 10, "entry_price": 10.0, "cost": 100.0}}))
    world()
    web_ui.recommend = lambda sectors, top_n, context=None: {"final_weights": {"AAA": 0.495, "BBB": 0.498}, "status": "ok"}
    c.post("/portfolios/p1/recalibrate")
    assert [(s, side) for s, side, _ in FakeTradingClient.submitted] == [("ZZZ", "sell")], FakeTradingClient.submitted
    ok("0.3% resizes skipped; the 1% position closed in full")

    print("SCENARIO O: CASH from the rulebook's limits is never priced or ordered; strategist views persist")
    fake = install(doc(pending_order_ids={}))
    world()
    state_seen = {}
    def remote_query(thread_id, state):
        state_seen.update(state)
        return {"final_weights": {"BBB": 0.3, "CCC": 0.3, "CASH": 0.4}, "status": "ok",
                "theme_memory": {"tech_ai": {"theme": "tech_ai", "strategic": {"raw": 0.1}, "tactical": {"raw": 0.0}}}}
    web_ui._remote_agent.query = remote_query
    web_ui.recommend = REAL_RECOMMEND
    web_ui._call_with_timeout = lambda fn, *a, timeout=None, **k: fn(*a, **k)
    c.post("/portfolios/p1/recalibrate")
    assert "CASH" not in {s for s, _, _ in FakeTradingClient.submitted}, FakeTradingClient.submitted
    assert "CASH" not in fake.docs["p1"]["rebalance_history"][-1]["target_weights"]
    assert state_seen["theme_memory"] == {} and state_seen["mandate"]["default_limits"]["held"] == ["AAA", "BBB"]
    first_saved = ps_store.get_strategist_memory()
    assert first_saved["theme_memory"]["tech_ai"]["strategic"]["raw"] == 0.1 and first_saved["saved_at"]
    thread, state2 = web_ui._build_recommend_state(["tech_ai"], 3)
    assert state2["theme_memory"]["tech_ai"]["strategic"]["raw"] == 0.1 and 0 <= state2["elapsed_days"] < 1
    ok("cash stays cash; sector views saved and re-read, so a strategic review lasts until it's due")

    print("\nAll proactive-recalibration integration scenarios passed.")
