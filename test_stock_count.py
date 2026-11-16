"""
test_stock_count.py
====================
Drives the REAL web_ui Flask app to prove a portfolio keeps the number of
stocks it was launched with — for every portfolio:

  * the number chosen on the recommendation page survives the page reload
    into the launch form (it used to fall back to the default 12, so every
    portfolio was saved with top_n=12 whatever was chosen);
  * a launch records n_stocks = how many stocks it actually bought;
  * every recalibration re-selects exactly n_stocks names (older portfolios
    without it keep their current count);
  * the per-stock cap never exceeds what the pre-trade reviewer would block.

Only the outside world is faked (same harness as test_proactive_recal.py).

Run:  python test_stock_count.py
"""

from __future__ import annotations

import re

import test_proactive_recal as base                 # stubs GCP, imports the real web_ui, fake world helpers
from test_portfolio_page_reconcile import FakeTradingClient, install

web_ui = base.web_ui
import portfolio_store as ps                         # noqa: E402


def ok(msg): print(f"✓ {msg}")


if __name__ == "__main__":
    import alpaca.trading.client as alpaca_trading_client
    alpaca_trading_client.TradingClient = FakeTradingClient
    web_ui._live_prices = lambda syms: {s: 12.0 for s in syms}
    base.harness.simulate.fetch_price_series = lambda *a, **k: {}
    c = web_ui.app.test_client()

    print("SCENARIO A: the chosen number survives the reload into the launch form (was: always 12)")
    web_ui._LAST_CONTEXT.clear()
    web_ui._LAST_CONTEXT.update({"rows": [{"name": s, "price": 12.0, "shares": 10, "action": "BUY", "weight": 1 / 3,
                                           "action_notional": 120.0, "sector": "tech_ai"} for s in ("AAA", "BBB", "CCC")],
                                 "selected_sectors": ["tech_ai"], "capital": 10_000.0, "top_n": 3,
                                 "expected_return": 0.01, "batch_id": "b1", "status": "ok", "risk_verdict": "approve",
                                 "groups": [], "tilts": []})
    html = c.get("/").get_data(as_text=True)
    hidden = re.findall(r'<input type="hidden" name="top_n" value="(\d+)">', html)
    assert hidden and set(hidden) == {"3"}, hidden
    ok(f"launch form carries top_n={hidden[0]}")

    print("SCENARIO B: a launch records n_stocks = the number of stocks it actually bought")
    seen = {}
    real_create = ps.create_portfolio
    ps.create_portfolio = lambda **kw: seen.update(kw) or "new1"
    install(base.doc())
    base.world()
    base.admin_post(c, "/execute", data={"sectors": "tech_ai", "capital": "10000", "top_n": "3", "batch_id": "b1",
                                         "rebalance_cadence": "21"})
    ps.create_portfolio = real_create
    assert seen.get("n_stocks") == 3 and seen.get("top_n") == 3, {k: seen.get(k) for k in ("n_stocks", "top_n")}
    ok("3 stocks bought -> n_stocks=3, top_n=3")

    print("SCENARIO C: every recalibration re-selects exactly n_stocks names")
    for doc_over, want, label in (({"n_stocks": 5, "top_n": 12}, 5, "n_stocks recorded (5) beats the bad saved top_n (12)"),
                                  ({"top_n": 12}, 2, "no n_stocks -> keeps its current count (2 holdings)")):
        install(base.doc(pending_order_ids={}, **doc_over))
        base.world()
        asked = []
        web_ui.recommend = lambda sectors, top_n, context=None: asked.append(top_n) or {
            "final_weights": {"BBB": 0.5, "CCC": 0.5}, "status": "ok", "book_notes": {}}
        base.recalibrate_now(c)
        # P0.3 — preview + confirm runs recommend() twice with the same top_n.
        assert asked == [want, want], (label, asked)
        ok(label)
    install(base.doc(pending_order_ids={}, n_stocks=5))
    base.world()
    asked = []
    html = c.get("/portfolios/p1").get_data(as_text=True)
    assert "5 stocks (kept at every recalibration)" in html
    ok("page shows '5 stocks (kept at every recalibration)'")

    print("SCENARIO D: the per-stock cap never exceeds what the reviewer would block")
    caps = {n: web_ui._max_per_name_for(n) for n in range(1, 31)}
    assert caps[1] == 1.0 and caps[2] == 1.0, "1-2 stock books may be concentrated by design"
    assert all(caps[n] <= max(0.30, 1.0 / n) + 1e-12 for n in range(3, 31)), caps
    assert abs(caps[3] - 1 / 3) < 1e-12 and caps[5] == 0.30, "a 5-stock book: at most 30% per stock (was 40%)"
    assert all(caps[n] >= 1.0 / n - 1e-12 for n in range(1, 31)), "the budget can always be fully placed"
    assert abs(caps[12] - 2 / 12) < 1e-12 and caps[30] == 0.10
    ok(f"3 stocks -> 33%, 4-6 -> 30% (was 40%), 12 -> {caps[12]:.1%}, 30 -> 10%")
    print("\nAll stock-count scenarios passed.")
