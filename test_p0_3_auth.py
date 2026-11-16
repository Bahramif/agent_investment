"""
test_p0_3_auth.py
=================
P0.3 — server-side auth + confirm on destructive actions. Drives the REAL
web_ui Flask app (same harness as test_portfolio_page_reconcile.py) to prove:

  * an unauthenticated POST to close/liquidate, execute, recalibrate, or the
    news-watch toggle returns 403 and changes nothing (no orders, no ledger
    write, no flag flip);
  * a signed-in admin WITHOUT a valid CSRF token is refused the same way;
  * the manual recalibration is preview-then-confirm: the first submission
    renders the plan and trades nothing, only the confirming submission trades;
  * liquidation is two server-side steps: the first submission only renders
    the confirm page, and only a second submission typing the exact portfolio
    name sells anything;
  * the scheduled endpoints (/portfolios/recalibrate_due,
    /portfolios/proactive_check) REFUSE to run (503) when the recalibrate
    token is unset — fail closed, never unprotected;
  * the production portfolio switcher hides test portfolios
    (e.g. "Oct2-actual-test") without deleting them;
  * the day-trade halt button needs the shared token, and refuses when no
    token is configured (fail closed).

Only the outside world is faked (same harness as
test_portfolio_page_reconcile.py): GCP at import, Firestore, Alpaca, the
deployed agent's recommend(), headlines, and Claude.

Run:  python test_p0_3_auth.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import test_proactive_recal as base                 # stubs GCP, imports the real web_ui, fake world helpers
from test_portfolio_page_reconcile import FakeTradingClient, install, login, admin_post, recalibrate_now

web_ui = base.web_ui


def ok(msg): print(f"✓ {msg}")


if __name__ == "__main__":
    import alpaca.trading.client as alpaca_trading_client
    alpaca_trading_client.TradingClient = FakeTradingClient
    web_ui._live_prices = lambda syms: {s: 12.0 for s in syms}
    base.harness.simulate.fetch_price_series = lambda *a, **k: {}
    c = web_ui.app.test_client()

    print("TEST 1: unauthenticated POSTs to every destructive route return 403 and change nothing")
    fake = install(base.doc(pending_order_ids={}))
    base.world()
    news_watch_before = fake.docs["p1"].get("news_watch")
    for url, data in (("/portfolios/p1/close", {}),
                      ("/execute", {"sectors": "tech_ai", "capital": "1000", "top_n": "2", "batch_id": "b1"}),
                      ("/portfolios/p1/recalibrate", {}),
                      ("/portfolios/p1/news_watch", {"enabled": "1"})):
        r = c.post(url, data=data)
        assert r.status_code == 403, (url, r.status_code)
    assert not FakeTradingClient.submitted, "no orders from any unauthenticated POST"
    assert fake.docs["p1"]["status"] == "open" and fake.docs["p1"]["rebalance_history"] == []
    assert fake.docs["p1"].get("news_watch") == news_watch_before, "the toggle must not flip"
    ok("close / execute / recalibrate / news_watch all 403; no orders, no ledger writes, no flag flip")

    print("TEST 2: a signed-in admin WITHOUT a valid CSRF token is refused the same way")
    fake = install(base.doc(pending_order_ids={}))
    base.world()
    login(c)                                        # signed in, but no csrf_token field below
    for url, data in (("/portfolios/p1/close", {}),
                      ("/portfolios/p1/recalibrate", {}),
                      ("/portfolios/p1/news_watch", {"enabled": "1"})):
        assert c.post(url, data=data).status_code == 403, url
    assert c.post("/portfolios/p1/close", data={"csrf_token": "wrong-token"}).status_code == 403
    assert not FakeTradingClient.submitted and fake.docs["p1"]["status"] == "open"
    ok("no token / wrong token -> 403; a stolen session cookie alone is not enough")

    print("TEST 3: login rejects a wrong token, and a signed-in admin can sign out again")
    c2 = web_ui.app.test_client()            # fresh client: no session carried over from TEST 2
    assert c2.post("/login", data={"admin_token": "not-the-token"}).status_code == 200
    assert c2.post("/portfolios/p1/news_watch", data={"enabled": "1"}).status_code == 403
    login(c2)
    assert admin_post(c2, "/portfolios/p1/news_watch", data={"enabled": "1"}).status_code == 302
    assert fake.docs["p1"]["news_watch"] is True
    # sign out (needs the CSRF token, like every other destructive POST)
    with c2.session_transaction() as sess:
        tok = sess["csrf_token"]
    assert c2.post("/logout", data={"csrf_token": tok}).status_code == 302
    assert c2.post("/portfolios/p1/news_watch", data={"enabled": "0"}).status_code == 403
    ok("wrong token can't sign in; sign-out revokes the session")

    print("TEST 4: recalibration is preview-then-confirm — the preview trades nothing")
    fake = install(base.doc(pending_order_ids={}))
    st = base.world()
    r = admin_post(c, "/portfolios/p1/recalibrate")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "Preview: recalibrate" in html, r.status_code
    assert "BBB" in html and "CCC" in html, "the plan is shown before anything trades"
    assert not FakeTradingClient.submitted and fake.docs["p1"]["rebalance_history"] == []
    assert st["recommend_calls"] == 1, "the preview computed the real plan once"
    r = admin_post(c, "/portfolios/p1/recalibrate", data={"confirm": "1"})
    assert r.status_code == 200 and FakeTradingClient.submitted, "only the confirming submission trades"
    assert len(fake.docs["p1"]["rebalance_history"]) == 1
    ok(f"preview showed the plan, traded nothing; confirm placed {FakeTradingClient.submitted}")

    print("TEST 5: liquidation is two server-side steps and needs the exact portfolio name")
    fake = install(base.doc(pending_order_ids={}))
    base.world()
    r = admin_post(c, "/portfolios/p1/close")
    assert r.status_code == 200 and "Type the portfolio name" in r.get_data(as_text=True)
    assert not FakeTradingClient.submitted and fake.docs["p1"]["status"] == "open"
    r = admin_post(c, "/portfolios/p1/close", data={"confirm_name": "Wrong Name"})
    assert r.status_code == 200 and "Type the portfolio name" in r.get_data(as_text=True)
    assert not FakeTradingClient.submitted and fake.docs["p1"]["status"] == "open"
    admin_post(c, "/portfolios/p1/close", data={"confirm_name": "Test PF"})
    assert fake.docs["p1"]["status"] == "closed"
    sells = {sym: qty for sym, side, qty in FakeTradingClient.submitted if side == "sell"}
    assert sells == {"AAA": 100, "BBB": 50}, sells
    ok("confirm page first, wrong name refused, exact name liquidated and closed")

    print("TEST 6: scheduled endpoints FAIL CLOSED when the recalibrate token is unset")
    fake = install(base.doc(pending_order_ids={}, next_rebalance_date=datetime.now(timezone.utc).date().isoformat()))
    base.world()
    import time
    saved = dict(web_ui._SECRET_CACHE)
    web_ui._SECRET_CACHE["recalibrate-token"] = (None, time.monotonic())   # simulates "the secret can't be read"
    try:
        assert web_ui._recalibrate_token() is None
        for url in ("/portfolios/recalibrate_due", "/portfolios/proactive_check"):
            r = c.post(url)                                  # no token at all
            assert r.status_code == 503, (url, r.status_code)
            r = c.post(url, headers={"X-Recalibrate-Token": "stub"})   # even the "right" token
            assert r.status_code == 503, (url, r.status_code)
        assert not FakeTradingClient.submitted and fake.docs["p1"]["rebalance_history"] == []
    finally:
        web_ui._SECRET_CACHE.clear()
        web_ui._SECRET_CACHE.update(saved)
    # ...and with the token configured again, the right header works as before.
    r = c.post("/portfolios/recalibrate_due", headers={"X-Recalibrate-Token": "stub"})
    assert r.status_code == 200 and r.get_json()["checked"] == 1, r.get_data(as_text=True)
    ok("503 with no token and with a token header; nothing traded; restores to working afterwards")

    print("TEST 7: the production switcher hides test portfolios (without deleting them)")
    html = web_ui._portfolio_selector([
        {"id": "p1", "name": "Test PF", "status": "open", "mtm_value": 9000.0},
        {"id": "p2", "name": "Oct2-actual-test", "status": "open", "mtm_value": 7834.0},
        {"id": "p3", "name": "Real Book", "status": "open", "mtm_value": 12000.0,
         "is_test": True},
    ], "/portfolios/p1")
    assert "Test PF" in html and "Oct2-actual-test" not in html and "Real Book" not in html
    ok("Test PF shown; Oct2-actual-test (name) and Real Book (is_test flag) hidden")

    print("TEST 8: read endpoints stay public — viewing needs no sign-in")
    fake = install(base.doc(pending_order_ids={}))
    base.world()
    assert c.get("/portfolios/p1").status_code == 200
    assert c.get("/portfolios").status_code in (200, 302)
    assert c.get("/about").status_code == 200
    ok("GET /portfolios/p1, /portfolios, /about all work signed out")

    print("TEST 9: day-trade halt needs the token, and refuses when no token is configured")
    from flask import Flask
    from daytrade_ui import make_daytrade_blueprint

    class FakeDayStore:
        def __init__(self): self.halted = False
        def set_halted(self, halted, reason="", by=""):
            self.halted = halted
            return {"halted": halted}
        def get_control(self): return {"halted": self.halted}
        def list_sessions(self, n): return []
        def get_session(self, day): return None
        def get_trials(self): return 0
        def list_journal(self): return []

    store = FakeDayStore()
    day_app = Flask(__name__)
    day_app.secret_key = "test-only"
    day_app.register_blueprint(make_daytrade_blueprint(
        "", lambda: store, lambda: {}, lambda: "day-token",
        lambda: __import__("daytrade_config").load_day_trading_config("mandate.yaml")))
    dc = day_app.test_client()
    assert dc.post("/daytrade/halt").status_code == 403
    assert dc.post("/daytrade/halt", data={"token": "wrong"}).status_code == 403
    assert store.halted is False
    assert dc.post("/daytrade/halt", data={"token": "day-token"}).status_code == 302
    assert store.halted is True
    # fail closed: token_fn returns None -> even the correct-looking token is refused
    day_app2 = Flask(__name__)
    day_app2.secret_key = "test-only"
    day_app2.register_blueprint(make_daytrade_blueprint(
        "", lambda: store, lambda: {}, lambda: None,
        lambda: __import__("daytrade_config").load_day_trading_config("mandate.yaml")))
    assert day_app2.test_client().post("/daytrade/halt", data={"token": "anything"}).status_code == 403
    ok("halt 403 without/wrong token, 302 with it; 403 for every caller when no token is configured")

    print("TEST 10: one transient Secret Manager error doesn't lock the scheduled endpoints out")
    import time
    import google.cloud.secretmanager as sm
    real_client = sm.SecretManagerServiceClient

    class Flaky:
        calls = 0
        def access_secret_version(self, name):
            Flaky.calls += 1
            raise ConnectionError("Secret Manager unavailable")
    web_ui._SECRET_CACHE.pop("recalibrate-token", None)
    sm.SecretManagerServiceClient = Flaky
    try:
        assert c.post("/portfolios/recalibrate_due", headers={"X-Recalibrate-Token": "stub"}).status_code == 503, "fail closed while it's down"
        assert c.post("/portfolios/recalibrate_due", headers={"X-Recalibrate-Token": "stub"}).status_code == 503
        assert Flaky.calls == 1, "not hammered: retried only after SECRET_RETRY_S"
    finally:
        sm.SecretManagerServiceClient = real_client
    web_ui._SECRET_CACHE["recalibrate-token"] = (None, time.monotonic() - web_ui.SECRET_RETRY_S - 1)   # the retry is due
    assert c.post("/portfolios/recalibrate_due", headers={"X-Recalibrate-Token": "stub"}).status_code == 200, "recovers by itself"
    ok("503 while unreadable, one read per retry window, back to 200 once it's readable")

    print("TEST 11: the admin token is its own secret, not the scheduler's")
    saved = dict(web_ui._SECRET_CACHE)
    try:
        web_ui._SECRET_CACHE["admin-token"] = (None, time.monotonic())
        web_ui._SECRET_CACHE["recalibrate-token"] = ("scheduler-only", time.monotonic())
        c3 = web_ui.app.test_client()
        r = c3.post("/login", data={"admin_token": "scheduler-only"})
        assert r.status_code == 200 and "No admin token is configured" in r.get_data(as_text=True), "the scheduler token doesn't sign in"
        web_ui._SECRET_CACHE["admin-token"] = ("admin-only", time.monotonic())
        assert c3.post("/login", data={"admin_token": "admin-only"}).status_code == 302
        assert c3.post("/portfolios/recalibrate_due", headers={"X-Recalibrate-Token": "admin-only"}).status_code == 403, \
            "and the admin token doesn't run the scheduled endpoints"
    finally:
        web_ui._SECRET_CACHE.clear()
        web_ui._SECRET_CACHE.update(saved)
    ok("separate credentials for the person and for the scheduler")

    print("TEST 12: failed sign-ins are rate-limited per client and in total; a success clears the client's count")
    web_ui._LOGIN_FAILURES.clear()
    c4 = web_ui.app.test_client()
    attacker = {"X-Forwarded-For": "203.0.113.9, 10.0.0.1"}
    for _ in range(web_ui.LOGIN_MAX_FAILURES):
        assert c4.post("/login", data={"admin_token": "guess"}, headers=attacker).status_code == 200
    r = c4.post("/login", data={"admin_token": "stub"}, headers=attacker)
    assert r.status_code == 429 and "Too many failed" in r.get_data(as_text=True), "even the right token waits once blocked"
    owner = {"X-Forwarded-For": "198.51.100.7"}
    assert c4.post("/login", data={"admin_token": "stub"}, headers=owner).status_code == 302, "another client isn't blocked"
    web_ui._LOGIN_FAILURES.clear()
    for k in range(web_ui.LOGIN_MAX_FAILURES_TOTAL):
        c4.post("/login", data={"admin_token": "guess"}, headers={"X-Forwarded-For": f"192.0.2.{k}"})
    assert c4.post("/login", data={"admin_token": "stub"}, headers=owner).status_code == 429, "spoofed addresses hit the total cap"
    web_ui._LOGIN_FAILURES.clear()
    for _ in range(web_ui.LOGIN_MAX_FAILURES - 1):
        c4.post("/login", data={"admin_token": "guess"}, headers=owner)
    assert c4.post("/login", data={"admin_token": "stub"}, headers=owner).status_code == 302
    assert "198.51.100.7" not in web_ui._LOGIN_FAILURES, "a success resets that client's count"
    web_ui._LOGIN_FAILURES.clear()
    ok("5 misses -> 429 for that client; 30 in total -> 429 for all; success resets")

    print("TEST 13: one session key for every instance, and safe cookie flags")
    assert web_ui.app.secret_key == "stub", "read from the session-secret secret (the test stub), not random per process"
    assert web_ui.app.config["SESSION_COOKIE_HTTPONLY"] and web_ui.app.config["SESSION_COOKIE_SAMESITE"] == "Lax"
    import importlib, os
    saved_cache = dict(web_ui._SECRET_CACHE)
    try:
        os.environ["MERIDIAN_SESSION_SECRET"] = "from-env"
        assert web_ui._session_secret() == "from-env"
        del os.environ["MERIDIAN_SESSION_SECRET"]
        web_ui._SECRET_CACHE["session-secret"] = (None, time.monotonic())
        a, b = web_ui._session_secret(), web_ui._session_secret()
        assert a != b and len(a) == 64, "no secret anywhere: random, and logged as an alert"
    finally:
        os.environ.pop("MERIDIAN_SESSION_SECRET", None)
        web_ui._SECRET_CACHE.clear()
        web_ui._SECRET_CACHE.update(saved_cache)
    other_instance = web_ui.app.test_client()
    login(c)
    with c.session_transaction() as sess:
        tok = sess["csrf_token"]
    cookie = c.get_cookie("session")
    other_instance.set_cookie("session", cookie.value)
    assert other_instance.post("/portfolios/p1/news_watch", data={"enabled": "1", "csrf_token": tok}).status_code == 302, \
        "a cookie from one instance works on another with the same key"
    ok("shared key; HttpOnly + SameSite=Lax; Secure on Cloud Run")

    print("\nAll P0.3 auth checks passed.")
