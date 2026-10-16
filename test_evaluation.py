"""
test_evaluation.py
===================
Drives the REAL web_ui Flask app to prove the evaluation agent is wired end
to end: during a real recalibration the pre-trade reviewer blocks ONLY buys
that fail a hard check (red-flag news, concentration), its advisory AI grade
is stored and shown but never blocks, a broken reviewer never stops a
recalibration, and the 🧪 Evaluation page grades past calls against their
sector.

Only the outside world is faked (same harness as test_proactive_recal.py).

Run:  python test_evaluation.py
"""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace as NS

import test_proactive_recal as base                 # stubs GCP, imports the real web_ui, fake world helpers
from test_portfolio_page_reconcile import FakeTradingClient, install

web_ui = base.web_ui


def ok(msg): print(f"✓ {msg}")


class FakeReviewer:
    def __init__(self, review=None, error=None):
        self.review, self.error, self.calls, self.messages = review, error, [], self

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return NS(stop_reason="tool_use", model="fake-reviewer",
                  content=[NS(type="tool_use", name="review_trades", input=self.review)])


def setup(weights, heads=None, reviewer=None):
    """One recalibration world: recommend() returns `weights`, the reviewer's
    headlines and model are faked."""
    base.world()
    web_ui.recommend = lambda sectors, top_n: {"final_weights": weights, "status": "ok",
                                               "book_notes": {s: {"conviction": 0.7, "rating": "Buy",
                                                                  "momentum_1m": 0.03, "dip_score": 0.0,
                                                                  "rationale": f"{s} looks good"} for s in weights}}
    web_ui._headlines_many = lambda syms, days: {s: v for s, v in (heads or {}).items() if s in syms}
    web_ui._claude_client = lambda: reviewer or FakeReviewer(error=RuntimeError("no model in this test"))


def bought(sym):
    return any(s == sym and side == "buy" for s, side, _ in FakeTradingClient.submitted)


if __name__ == "__main__":
    import alpaca.trading.client as alpaca_trading_client
    alpaca_trading_client.TradingClient = FakeTradingClient
    web_ui._live_prices = lambda syms: {s: 12.0 for s in syms}
    base.harness.simulate.fetch_price_series = lambda *a, **k: {}
    c = web_ui.app.test_client()

    print("SCENARIO A: red-flag news blocks THAT buy only; the rest of the recalibration trades")
    fake = install(base.doc(pending_order_ids={}))
    setup({"BBB": 0.5, "CCC": 0.4}, heads={"CCC": ["CCC under SEC investigation over accounting irregularities"],
                                            "BBB": ["BBB misses estimates, shares slide 6%"]})
    c.post("/portfolios/p1/recalibrate")
    ev = fake.docs["p1"]["rebalance_history"][-1]
    assert not bought("CCC") and bought("BBB"), FakeTradingClient.submitted
    assert any(s == "AAA" and side == "sell" for s, side, _ in FakeTradingClient.submitted), "sells still go out"
    assert list(ev["review"]["vetoed"]) == ["CCC"] and "red-flag" in ev["review"]["vetoed"]["CCC"]
    assert "CCC" not in fake.docs["p1"]["holdings"] and fake.docs["p1"]["cash_remaining"] > 1000, "the money stays cash"
    html = c.get("/portfolios/p1").get_data(as_text=True)
    assert "⛔ blocked BUY CCC" in html and "SEC investigation" in html
    ok(f"orders {FakeTradingClient.submitted}; CCC blocked, cash ${fake.docs['p1']['cash_remaining']:,.0f}")

    print("SCENARIO B: concentration — a >40% buy in a 3-name portfolio is blocked; a 2-name one is not")
    fake = install(base.doc(pending_order_ids={}))
    setup({"BBB": 0.5, "CCC": 0.3, "DDD": 0.2})
    c.post("/portfolios/p1/recalibrate")
    v = fake.docs["p1"]["rebalance_history"][-1]["review"]["vetoed"]
    assert list(v) == ["BBB"] and "concentration" in v["BBB"] and bought("CCC") and bought("DDD")
    fake = install(base.doc(pending_order_ids={}))
    setup({"BBB": 0.6, "CCC": 0.4})
    c.post("/portfolios/p1/recalibrate")
    assert fake.docs["p1"]["rebalance_history"][-1]["review"]["vetoed"] == {} and bought("BBB")
    ok("BBB 50% of a 3-name book blocked; 60% of a 2-name book allowed (concentrated by design)")

    print("SCENARIO C: the AI review is stored and shown, and its 'object' never blocks a trade")
    rv = FakeReviewer({"grade": "C", "summary": "Chases BBB after a run-up.",
                       "orders": [{"symbol": "BBB", "verdict": "object", "reason": "Chasing a 30% run."},
                                  {"symbol": "CCC", "verdict": "agree", "reason": "Sound."},
                                  {"symbol": "AAA", "verdict": "concern", "reason": "Selling a quality name."}]})
    fake = install(base.doc(pending_order_ids={}))
    setup({"BBB": 0.5, "CCC": 0.4}, reviewer=rv)
    c.post("/portfolios/p1/recalibrate")
    review = fake.docs["p1"]["rebalance_history"][-1]["review"]
    assert review["grade"] == "C" and review["model"] == "fake-reviewer" and review["vetoed"] == {}
    assert bought("BBB"), "advisory: the AI's objection does not block"
    prompt = rv.calls[0]["messages"][0]["content"]
    assert "BUY" in prompt and "BBB looks good" in prompt and "Trigger: manual" in prompt
    html = c.get("/portfolios/p1").get_data(as_text=True)
    assert "🧪 Reviewer: <b>C</b>" in html and "✋ BBB: Chasing a 30% run." in html and "⚠️ AAA" in html
    assert "CCC: Sound." not in html, "agreements aren't listed — only concerns and objections"
    signals = fake.docs["p1"]["rebalance_history"][-1]["signals"]
    assert set(signals) == {"BBB", "CCC"} and signals["BBB"]["conviction"] == 0.7, "signals stored for the scorecard"
    ok("grade C + objection logged and rendered; BBB still bought; signals recorded per pick")

    print("SCENARIO D: a broken reviewer never stops a recalibration")
    fake = install(base.doc(pending_order_ids={}))
    setup({"BBB": 0.5, "CCC": 0.4})
    web_ui._headlines_many = lambda syms, days: (_ for _ in ()).throw(TimeoutError("news down"))
    web_ui._daily_closes = lambda syms, start: (_ for _ in ()).throw(RuntimeError("bars down"))
    web_ui._SCORECARD.update(at=-1e9, value=None)
    c.post("/portfolios/p1/recalibrate")
    ev = fake.docs["p1"]["rebalance_history"][-1]
    assert bought("BBB") and bought("CCC") and ev["review"]["vetoed"] == {} and ev["trigger"] == "manual"
    ok("news, prices and model all down -> recalibration traded normally, nothing blocked")

    print("SCENARIO E: the Evaluation page grades past calls against their sector")
    d0 = date(2026, 9, 1)
    days = [d0 + timedelta(days=i) for i in range(40)]
    path = lambda start, step: {d: start * (1 + step) ** i for i, d in enumerate(days)}
    from universe import EQUITIES
    tech = [s for s, p, _ in EQUITIES if p.endswith(".tech_ai")]
    closes = {s: path(100, 0.0) for s in tech}
    closes.update({"NVDA": path(100, 0.01), "INTC": path(100, -0.01), "SPY": path(500, 0.001)})
    web_ui._daily_closes = lambda syms, start: closes
    web_ui._SCORECARD.update(at=-1e9, value=None)
    fake = install(base.doc(pending_order_ids={}, rebalance_history=[
        {"date": "2026-09-02", "trigger": "scheduled", "portfolio_value_before": 1.0,
         "orders": [{"symbol": "NVDA", "side": "BUY", "qty": 5, "result": "FILLED"},
                    {"symbol": "INTC", "side": "SELL", "qty": 5, "result": "FILLED"}],
         "signals": {"NVDA": {"dip_score": 0.6}}},
        {"date": "2026-09-04", "trigger": "proactive", "portfolio_value_before": 1.0,
         "orders": [{"symbol": "INTC", "side": "BUY", "qty": 3, "result": "FILLED"}],
         "signals": {"INTC": {"momentum_1m": 0.05}}}]))
    html = c.get("/evaluation").get_data(as_text=True)
    assert "🧪 How good have the recommendations been?" in html and 'class="active"' in html.split("🧪 Evaluation")[0][-120:]
    for needle in ("By what triggered the recalibration", "scheduled", "proactive", "Buys, by the signal behind them",
                   "dip", "momentum", "BUY NVDA", "SELL INTC", "BUY INTC", "67%"):
        assert needle in html, needle
    graded, agg = web_ui._scorecard()
    assert agg["by_signal"]["dip"][5]["hit_rate"] == 1.0 and agg["by_signal"]["momentum"][5]["hit_rate"] == 0.0
    assert agg["by_trigger"]["proactive"][5]["avg_value_added"] < 0
    ok(f"3 calls graded: hit rate {agg['overall'][5]['hit_rate']:.0%} at 5 days; dip buy hit, momentum buy missed")
    web_ui._daily_closes = lambda syms, start: (_ for _ in ()).throw(RuntimeError("bars down"))
    web_ui._SCORECARD.update(at=-1e9, value=None)
    r = c.get("/evaluation?refresh=1")
    assert r.status_code == 200 and "Could not compute the scorecard" in r.get_data(as_text=True)
    ok("a price outage degrades to a message, not an error page")
    print("\nAll evaluation-agent scenarios passed.")
