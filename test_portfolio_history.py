"""
test_portfolio_history.py
==========================
Drives the REAL web_ui Flask app to prove the Portfolio page lets a user
track holdings from launch through every recalibration:

  * "Started with": each launch position, its launch price, and what became
    of it (still held / sold on <date>);
  * a one-line journey summary above the history;
  * the history table: Date | Type | What changed | Why | 🧪 Reviewer |
    Value before — one short line per cell, long text in expandable details;
  * launches record initial_holdings; recalibrations record holdings_after.

Run:  python test_portfolio_history.py
"""

from __future__ import annotations

import re

import test_proactive_recal as base                 # stubs GCP, imports the real web_ui, fake world helpers
from test_portfolio_page_reconcile import FakeTradingClient, install

web_ui = base.web_ui
import portfolio_store as ps                         # noqa: E402


def ok(msg): print(f"✓ {msg}")


def cells(html: str, header: str) -> str:
    return html.split(f"<h2>{header}</h2>")[1].split("</table>")[0]


if __name__ == "__main__":
    import alpaca.trading.client as alpaca_trading_client
    alpaca_trading_client.TradingClient = FakeTradingClient
    web_ui._live_prices = lambda syms: {s: 12.0 for s in syms}
    base.harness.simulate.fetch_price_series = lambda *a, **k: {}
    c = web_ui.app.test_client()

    LONG_WHY = ("Rule — the portfolio is down 11.5% since its last recalibration (2026-10-06) while its own sector "
                "pool is -4.2%. (Model's own view: a concentrated sell-off in semiconductor equipment.)")
    history = [
        {"date": "2026-10-06", "at": "2026-10-06T19:10:00+00:00", "trigger": "manual", "portfolio_value_before": 10024.0,
         "orders": [{"symbol": "KEYS", "side": "SELL", "qty": 4, "result": "FILLED"},
                    {"symbol": "NVDA", "side": "BUY", "qty": 10, "result": "FILLED"}]},
        {"date": "2026-10-16", "at": "2026-10-16T16:31:00+00:00", "trigger": "proactive", "reason": LONG_WHY,
         "portfolio_value_before": 9001.0,
         "orders": [{"symbol": "AMAT", "side": "SELL", "qty": 3, "result": "FILLED"},
                    {"symbol": "GOOGL", "side": "BUY", "qty": 6, "result": "FILLED"}],
         "holdings_after": {"KLAC": {"shares": 8}, "NVDA": {"shares": 10}, "GOOGL": {"shares": 6}},
         "review": {"grade": "B", "summary": "Sound rotation out of a lagging sub-industry. Keep an eye on NVDA, which "
                                             "is now the largest single position after two consecutive additions.",
                    "orders": {"AMAT": {"verdict": "concern", "reason": "May be selling near a bottom."},
                               "GOOGL": {"verdict": "agree", "reason": "Quality."}},
                    "vetoed": {}}}]

    print("SCENARIO A: 'Started with' — every launch position and what became of it")
    fake = install(base.doc(pending_order_ids={}, rebalance_history=history,
                            initial_holdings={"AMAT": {"shares": 3, "entry_price": 200.0},
                                              "KEYS": {"shares": 4, "entry_price": 150.0},
                                              "KLAC": {"shares": 8, "entry_price": 90.0}},
                            holdings={"KLAC": {"shares": 8, "entry_price": 90.0, "cost": 720.0},
                                      "NVDA": {"shares": 10, "entry_price": 100.0, "cost": 1000.0},
                                      "GOOGL": {"shares": 6, "entry_price": 150.0, "cost": 900.0}}))
    base.world()
    html = c.get("/portfolios/p1").get_data(as_text=True)
    started = cells(html, "Started with")
    assert "$200.00" in started and "sold 2026-10-16" in started and "sold 2026-10-06" in started
    assert "✓ held (8 shares)" in started, "KLAC is still held"
    ok("AMAT sold 2026-10-16, KEYS sold 2026-10-06, KLAC still held — with launch prices")

    print("SCENARIO B: the journey summary and the restructured history table")
    hist = html.split("<h2>Recalibration history</h2>")[1].split("</table>")[0]
    assert "2 recalibrations since launch (1 👆 manual, 1 📰 mid-day)" in hist
    assert "Started with AMAT, KEYS, KLAC → now GOOGL, KLAC, NVDA (kept KLAC)" in hist
    heads = re.findall(r"<th>(.*?)</th>", hist)
    assert heads == ["Date", "Type", "What changed", "Why", "🧪 Reviewer", "Value before"], heads
    rows = hist.split("<tr>")[2:]
    newest, oldest = rows[0], rows[1]
    assert "📰 Mid-day" in newest and "12:31 ET" in newest
    assert ">−AMAT 3</span>" in newest and ">+GOOGL 6</span>" in newest, "share counts right on the chips"
    assert "holds 3" in newest
    assert "orders ▸" not in newest, "no dropdown needed any more"
    short_why = "Rule — the portfolio is down 11.5% since its last recalibration (2026-10-06) while its own sector pool is -4.2%."
    assert short_why in newest and "more ▸" in newest and "Model&#39;s own view" in newest, "short line + full text in details"
    assert '<span class="grade B">B</span> Sound rotation out of a lagging sub-industry.' in newest
    assert "0 objections · ⚠️ 1 concern" in newest and "⚠️ AMAT: May be selling near a bottom." in newest
    assert "GOOGL: Quality" not in newest, "agreements aren't listed"
    assert "$9,001" in newest
    assert "👆 Manual" in oldest and ">−KEYS 4</span>" in oldest and ">+NVDA 10</span>" in oldest
    assert "Recalibrate Now clicked" in oldest
    ok("6 short columns; chips for what changed; one-line why/reviewer with the rest behind 'more ▸' / 'details ▸'")

    print("SCENARIO B2: a resized position shows its signed share change with a real minus sign")
    resized_doc = base.doc(pending_order_ids={}, rebalance_history=[
        {"date": "2026-10-16", "trigger": "scheduled", "portfolio_value_before": 1.0,
         "orders": [{"symbol": "AAA", "side": "SELL", "qty": 40, "result": "FILLED"},
                    {"symbol": "BBB", "side": "BUY", "qty": 5, "result": "TIMEOUT"}]}])
    install(resized_doc)
    base.world()
    html = c.get("/portfolios/p1").get_data(as_text=True)
    hist = html.split("<h2>Recalibration history</h2>")[1].split("</table>")[0]
    assert ">⇅AAA −40</span>" in hist and ">⇅BBB +5</span>" in hist and "1 order still filling" in hist, hist[-1500:]
    ok("⇅AAA −40, ⇅BBB +5, '1 order still filling'")

    print("SCENARIO C: no launch record -> no 'Started with', but a real 'what changed' rebuilt from today's holdings")
    install(base.doc(pending_order_ids={}, rebalance_history=history[:1],
                     holdings={"AAA": {"shares": 100, "entry_price": 10.0, "cost": 1000.0},
                               "NVDA": {"shares": 10, "entry_price": 100.0, "cost": 1000.0}}))
    base.world()
    html = c.get("/portfolios/p1").get_data(as_text=True)
    assert "<h2>Started with</h2>" not in html
    hist = html.split("<h2>Recalibration history</h2>")[1].split("</table>")[0]
    assert "−KEYS" in hist and "+NVDA" in hist and "holds 2" in hist
    ok("no 'Started with' section; history row: −KEYS +NVDA, holds 2")

    print("SCENARIO D: launches record initial_holdings; recalibrations record holdings_after")
    seen = {}
    real_create = ps.create_portfolio
    ps.create_portfolio = lambda **kw: seen.update(kw) or "new1"
    web_ui._LAST_CONTEXT.clear()
    web_ui._LAST_CONTEXT.update({"rows": [{"name": s, "price": 12.0, "shares": 10, "action": "BUY", "weight": 0.5,
                                           "action_notional": 120.0, "sector": "tech_ai"} for s in ("AAA", "BBB")],
                                 "selected_sectors": ["tech_ai"], "capital": 1000.0, "top_n": 2,
                                 "expected_return": 0.0, "batch_id": "b2"})
    install(base.doc())
    base.world()
    base.admin_post(c, "/execute", data={"sectors": "tech_ai", "capital": "1000", "top_n": "2", "batch_id": "b2",
                                          "rebalance_cadence": "21"})
    ps.create_portfolio = real_create
    assert set(seen["initial_holdings"]) == {"AAA", "BBB"} and seen["initial_holdings"]["AAA"]["shares"] > 0
    fake = install(base.doc(pending_order_ids={}))
    base.world()
    base.recalibrate_now(c)
    ev = fake.docs["p1"]["rebalance_history"][-1]
    assert set(ev["holdings_after"]) == set(fake.docs["p1"]["holdings"]), (ev["holdings_after"], fake.docs["p1"]["holdings"])
    ok(f"launch stored {sorted(seen['initial_holdings'])}; recalibration stored holdings_after {sorted(ev['holdings_after'])}")
    print("\nAll portfolio-history scenarios passed.")
