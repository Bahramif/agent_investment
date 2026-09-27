"""
daytrade_runner.py
===================
The live intraday loop — the Cloud Run Job entrypoint for the day-trading
capability. One process = one trading day. It composes pieces that each have
their own tests:

    plan_day()  ->  DayTradingEngine  ->  AlpacaDayTradeBroker  ->  Firestore
   (scanner +      (signals, risk,        (bracket entries,         (session doc,
    Claude)         flatten, halts)        exits, flatten)           kill switch)

RUNNING IT (deployment is left to you — nothing here touches GCP):
  Build the SAME image as the web app (Dockerfile) and create a Cloud Run Job
  that overrides the command, then trigger it before the open with Scheduler:

    gcloud run jobs create meridian-daytrade --image IMAGE --region us-central1 \\
        --command python --args daytrade_runner.py \\
        --task-timeout 8h --max-retries 2 --service-account SA
    gcloud scheduler jobs create http meridian-daytrade-open --location us-central1 \\
        --schedule "0 9 * * 1-5" --time-zone "America/New_York" \\
        --uri "https://run.googleapis.com/v2/projects/PROJECT/locations/us-central1/jobs/meridian-daytrade:run" \\
        --http-method POST --oauth-service-account-email SA

  max-retries 2 is SAFE because every start reconciles from the broker and the
  session document (see "RESTARTS"), and entry order IDs are deterministic.
  The scheduler timezone handles DST; the runner itself asks Alpaca's clock
  whether today is a trading day and when it closes (early closes included),
  and exits cleanly on holidays.
  Local dry run against paper:  python daytrade_runner.py --no-llm

RESTARTS. A crash mid-day is expected to happen, not to be prevented:
  * bars are rebuilt from the feed (since={}), and STALE signals never trade;
  * names already traded, entries sent and realized P&L come from today's
    session document;
  * open positions come from the BROKER, not from memory: one with live
    bracket legs is adopted with its real stop/target; one WITHOUT live legs
    is unprotected and is flattened immediately;
  * entry order IDs are one-per-symbol-per-day, so a re-sent entry finds the
    original order instead of buying twice.

INVARIANTS the end of run_session enforces: the account is flat (anything left
is emergency-flattened — allowed only because the account is dedicated) and
the session document says how the day ended.

SHADOW ENGINE. When `run_unfiltered_shadow` is on, a second engine with NO
playbook restrictions is fed the same bars and filled by the same simulator
the backtest uses; its P&L is stored beside the live result. That pair is the
LLM ablation: on days with a playbook, did the model's restrictions beat the
plain engine? It is a simulation, not a broker account, and after a mid-day
restart it is marked `partial`.

LEARNING. The live playbook carries the ACTIVE learned lessons (attached by
the planner from the store; the shadow gets none, so the shadow stays the
plain-engine baseline). At wrap-up the engine's journal — every signal it
saw, taken or not, with entry features and outcomes — is saved, then the
post-close learning hook runs (daytrade_learning.run_post_close). Both are
wrapped so that a learning failure can never change how the day ended.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from daytrade_backtest import SimBroker
from daytrade_broker import AlpacaDayTradeBroker
from daytrade_config import DayTradingConfig, load_day_trading_config
from daytrade_engine import (CancelIntent, DayTradingEngine, EntryIntent, ExitIntent, FillEvent, Playbook)
from daytrade_playbook import (build_candidates, deterministic_playbook, make_playbook, market_read,
                               playbook_from_dict, playbook_to_dict)
from daytrade_store import DayTradeStore, trade_from_dict, trade_to_dict
from execution_workflow import OrderResult
from intraday_data import ET, UTC, BarFeed, IntradayDataClient, MarketClock, et_datetime, session_date, to_et

MARKET_OPEN_T = time(9, 30)
PENDING_TIMEOUT = timedelta(seconds=120)      # an in-flight entry not filled by then is cancelled
PERSIST_EVERY = timedelta(seconds=60)


@dataclass
class DayPlan:
    playbook: Playbook                  # what the LIVE engine may do (possibly LLM-restricted)
    shadow_playbook: Playbook           # the unrestricted deterministic baseline
    prev_close: Dict[str, float]
    candidates: List[str]               # symbols worth feeding (playbook + shadow watchlists)
    market: Dict[str, float] = field(default_factory=dict)   # SPY read for entry snapshots


PlanFn = Callable[[date], DayPlan]


@dataclass
class RunnerDeps:
    cfg: DayTradingConfig
    store: DayTradeStore
    broker: AlpacaDayTradeBroker
    clock: MarketClock
    plan_day: PlanFn
    make_feed: Callable[[List[str]], BarFeed]
    sleep: Callable[[float], None] = _time.sleep
    poll_seconds: float = 15.0
    post_close: Optional[Callable[[date], Any]] = None      # learning hook, after the journal is saved


def _log(msg: str) -> None:
    print(f"[daytrade_runner {datetime.now(UTC):%H:%M:%S}Z] {msg}", flush=True)


def _avg_price(fill) -> float:
    return fill.filled_notional / fill.filled_qty if fill.filled_qty else 0.0


class SessionRunner:
    def __init__(self, deps: RunnerDeps):
        self.d = deps
        self.cfg = deps.cfg
        self.store, self.broker, self.clock = deps.store, deps.broker, deps.clock
        self.events: List[Dict[str, Any]] = []
        self._last_persist: Optional[datetime] = None
        self._attempts: Dict[str, int] = {}
        self._dirty = True

    # ------------------------------------------------------------------ events
    def _event(self, now: datetime, kind: str, **kw: Any) -> None:
        self.events.append({"ts": now.isoformat(), "kind": kind, **kw})
        self.events = self.events[-100:]
        self._dirty = True
        _log(f"{kind} {kw}")

    # ------------------------------------------------------------------ persistence
    def _persist(self, now: datetime, status: str, force: bool = False, **extra: Any) -> None:
        if not force and not self._dirty and self._last_persist and now - self._last_persist < PERSIST_EVERY:
            return
        eng = self.engine
        doc: Dict[str, Any] = {
            "status": status,
            "sleeve": eng.sleeve,
            "entries_today": eng.state.entries_today,
            "symbols_traded": sorted(eng.state.symbols_traded),
            "realized_pnl": round(eng.state.realized_pnl, 4),
            "unrealized_pnl": round(eng.state.unrealized_pnl, 4),
            "trades": [trade_to_dict(t) for t in eng.trades],
            "open_positions": [{"symbol": p.symbol, "qty": p.qty, "entry_price": p.entry_price, "stop": p.stop,
                                "target": p.target, "strategy": p.strategy} for p in eng.positions.values()],
            "halted": eng.state.halted, "halt_reason": eng.state.halt_reason,
            "events": (self.events + eng.decisions[-40:])[-120:],
            **extra,
        }
        if getattr(self, "shadow", None) is not None:
            doc["shadow"] = {"pnl": round(self.shadow.state.realized_pnl, 4),
                             "entries": self.shadow.state.entries_today,
                             "trades": [trade_to_dict(t) for t in self.shadow.trades],
                             "partial": self.recovered}
        self.store.save_session(self.day, doc)
        self._last_persist, self._dirty = now, False

    # ------------------------------------------------------------------ recovery
    def _recover(self, now: datetime, existing: Optional[Dict[str, Any]]) -> None:
        eng = self.engine
        if existing:
            eng.restore_progress(existing.get("entries_today", 0), set(existing.get("symbols_traded", [])),
                                 existing.get("realized_pnl", 0.0),
                                 [trade_from_dict(t) for t in existing.get("trades", [])])
        for sym, pos in self.broker.positions().items():
            coid = eng.entry_order_id(self.day, sym)
            snap = self.broker.lookup_entry(coid)
            legs = {l.kind: l for l in (snap.legs if snap else []) if l.is_open}
            if snap and "stop" in legs and "target" in legs and legs["stop"].price and legs["target"].price:
                eng.adopt_position(sym, pos.qty, pos.avg_entry_price, legs["stop"].price, legs["target"].price,
                                   "recovered", now, coid)
                self._event(now, "adopted", symbol=sym, qty=pos.qty, stop=legs["stop"].price, target=legs["target"].price)
            else:
                self._event(now, "orphan_position", symbol=sym, qty=pos.qty,
                            msg="position with no live bracket legs is unprotected — flattening now")
                res = self.broker.flatten_position(sym, int(pos.qty), coid, f"{self.day:%Y%m%d}", attempt=99,
                                                   reason="orphan_flatten")
                self._event(now, "orphan_flatten", symbol=sym, status=res.status, detail=res.detail)
        self.recovered = bool(existing) or bool(self.broker.positions())

    # ------------------------------------------------------------------ live execution
    def _apply_sell(self, fill: FillEvent) -> None:
        fill.fees = self.cfg.costs.sell_fees(fill.qty, fill.price)
        self.engine.apply_fill(fill)

    def _handle_entry(self, intent: EntryIntent, now: datetime) -> None:
        try:
            fill = self.broker.place_entry(intent)
        except Exception as e:
            self.engine.apply_reject(intent.symbol, f"order error: {e}", now)
            self._event(now, "entry_error", symbol=intent.symbol, error=str(e))
            return
        if fill.result == OrderResult.FILLED and fill.filled_qty:
            self.engine.apply_fill(FillEvent(intent.symbol, "buy", fill.filled_qty, _avg_price(fill), now))
        elif fill.result == OrderResult.PARTIAL:
            self.broker.cancel_entry_remainder(intent.client_order_id)      # protect exactly what was bought
            self._reconcile_one(intent.symbol, now, final=True)
        elif fill.result == OrderResult.REJECTED:
            self.engine.apply_reject(intent.symbol, fill.reason or "rejected", now)
        # TIMEOUT: still in flight — _reconcile_pending() resolves it on later ticks

    def _reconcile_one(self, sym: str, now: datetime, final: bool = False) -> None:
        intent = self.engine.pending.get(sym)
        if intent is None:
            return
        snap = self.broker.lookup_entry(intent.client_order_id)
        if snap is None:
            if final or now - intent.decision_ts > PENDING_TIMEOUT:
                self.engine.apply_reject(sym, "no order found at broker", now)
            return
        if snap.filled_qty > 0 and (snap.status == "filled" or final):
            self.engine.apply_fill(FillEvent(sym, "buy", snap.filled_qty, snap.filled_avg_price or intent.entry_ref, now))
        elif snap.status in ("canceled", "rejected", "expired", "done_for_day", "suspended") or final:
            self.engine.apply_reject(sym, f"entry {snap.status}", now)
        elif now - intent.decision_ts > PENDING_TIMEOUT:
            self.broker.cancel_entry_remainder(intent.client_order_id)
            self._event(now, "entry_timeout_cancel", symbol=sym)

    def _reconcile_pending(self, now: datetime) -> None:
        for sym in list(self.engine.pending):
            self._reconcile_one(sym, now)

    def _sync_bracket_exits(self, now: datetime) -> None:
        for sym, pos in list(self.engine.positions.items()):
            fill = self.broker.exit_fill_from_legs(self.broker.lookup_entry(pos.client_order_id), sym)
            if fill is not None:
                self._apply_sell(fill)
                self._event(now, "bracket_exit", symbol=sym, reason=fill.reason, price=fill.price)

    def _handle_exit(self, intent: ExitIntent, now: datetime) -> None:
        pos = self.engine.positions.get(intent.symbol)
        if pos is None:
            return
        attempt = self._attempts.get(intent.symbol, 0)
        self._attempts[intent.symbol] = attempt + 1
        try:
            res = self.broker.flatten_position(intent.symbol, int(pos.qty), pos.client_order_id,
                                               f"{self.day:%Y%m%d}", attempt, intent.reason)
        except Exception as e:
            self._event(now, "flatten_error", symbol=intent.symbol, error=str(e))
            return
        if res.fill is not None:
            self._apply_sell(res.fill)
        elif res.status == "already_flat":
            ctx = self.engine.ctx.get(intent.symbol)
            px = ctx.bars[-1].close if ctx and ctx.bars else pos.entry_price
            self._apply_sell(FillEvent(intent.symbol, "sell", pos.qty, px, now, reason="already_flat_price_unknown"))
            self._event(now, "warn", symbol=intent.symbol, msg="position closed at broker; exit price unknown, "
                                                               "marked at last bar")
        self._event(now, "flatten", symbol=intent.symbol, status=res.status, detail=res.detail)

    def _handle_cancel(self, intent: CancelIntent, now: datetime) -> None:
        pend = self.engine.pending.get(intent.symbol)
        if pend is not None:
            self.broker.cancel_entry_remainder(pend.client_order_id)
            self._reconcile_one(intent.symbol, now, final=True)

    # ------------------------------------------------------------------ the day
    def run(self) -> Dict[str, Any]:
        clock, cfg = self.clock, self.cfg
        now = clock.now()
        self.day = session_date(now)
        self.recovered = False
        self.shadow: Optional[DayTradingEngine] = None
        day_tag = f"{self.day:%Y%m%d}"

        if not cfg.enabled:
            self.store.save_session(self.day, {"status": "skipped", "note": "day_trading.enabled is false"})
            return {"status": "skipped", "reason": "disabled"}
        close = clock.session_close_et()
        if close is None:
            self.store.save_session(self.day, {"status": "skipped", "note": "market closed today"})
            return {"status": "skipped", "reason": "market closed"}

        existing = self.store.get_session(self.day)
        if existing and existing.get("status") in ("completed", "halted") and not existing.get("open_positions"):
            _log("today's session already finished — nothing to do")
            return {"status": existing["status"], "reason": "already finished"}

        acct = self.broker.account_snapshot()
        if acct["trading_blocked"] or acct["status"] != "active":
            self.store.save_session(self.day, {"status": "error", "note": f"account not tradable: {acct}"})
            return {"status": "error", "reason": "account not tradable"}
        sleeve = min(cfg.sleeve_capital, acct["equity"])
        eff = dataclasses.replace(cfg, sleeve_capital=sleeve)

        control = self.store.get_control()
        if control["halted"]:
            self.store.save_session(self.day, {"status": "halted", "halted": True, "halt_reason": f"kill switch: {control['reason']}",
                                               "note": "kill switch was ON at start — no trading"})
            return {"status": "halted", "reason": "kill switch on at start"}

        if existing and existing.get("playbook") and existing.get("plan_symbols"):
            pb = playbook_from_dict(existing["playbook"])
            plan = self.d.plan_day(self.day)                  # prev_close + shadow are cheap to recompute...
            plan = dataclasses.replace(plan, playbook=pb)     # ...but the LIVE playbook must not change mid-day
            symbols = list(existing["plan_symbols"])
        else:
            plan = self.d.plan_day(self.day)
            symbols = sorted(set(plan.candidates) | set(plan.playbook.watchlist or []) |
                             set(plan.shadow_playbook.watchlist or []))
        close_dt = et_datetime(self.day, close).astimezone(UTC)
        self.engine = DayTradingEngine(eff, self.day, plan.prev_close, plan.playbook, close_time=close,
                                       market=plan.market)
        self.sim: Optional[SimBroker] = None
        if cfg.run_unfiltered_shadow:
            self.shadow = DayTradingEngine(eff, self.day, plan.prev_close, plan.shadow_playbook, close_time=close,
                                           market=plan.market)
            self.sim = SimBroker(cfg.costs)

        self._recover(now, existing)
        self.store.save_session(self.day, {"status": "running", "started_at": now.isoformat(), "sleeve": sleeve,
                                           "playbook": playbook_to_dict(plan.playbook), "plan_symbols": symbols,
                                           "watchlist": plan.playbook.watchlist or []})
        self._event(now, "start", sleeve=sleeve, symbols=len(symbols), watchlist=len(plan.playbook.watchlist or []),
                    risk_on=plan.playbook.risk_on, recovered=self.recovered)

        if not plan.playbook.risk_on and not self.engine.positions and not self.recovered:
            self._persist(now, "completed", force=True, note="playbook is risk-off: no trading today")
            return {"status": "completed", "reason": "risk-off", "trades": 0}

        feed = self.d.make_feed(symbols)
        since: Dict[str, Optional[datetime]] = {s: None for s in symbols}
        open_dt = et_datetime(self.day, MARKET_OPEN_T).astimezone(UTC)
        last_close: Dict[str, float] = {}
        status = "completed"
        deadline = close_dt + timedelta(minutes=3)

        while True:
            now = clock.now()
            if now < open_dt:
                self.d.sleep(min(30.0, (open_dt - now).total_seconds()))
                continue
            if now >= deadline:
                break

            # 1) kill switch
            control = self.store.get_control()
            if control["halted"] and not self.engine.state.halted:
                self.engine.force_halt(f"kill switch: {control['reason']}", now)
                self._event(now, "kill_switch", reason=control["reason"])

            # 2) data
            try:
                bars = feed.poll(since, now)
            except Exception as e:
                self._event(now, "data_error", error=str(e))
                bars = []
            for b in bars:
                since[b.symbol] = b.ts
                last_close[b.symbol] = b.close

            # 3) shadow (simulated fills first, like the backtest) then live
            if self.shadow is not None and bars:
                for b in sorted(bars, key=lambda x: (x.ts, x.symbol)):
                    fills, rejects = self.sim.on_bar(b)
                    for f in fills:
                        self.shadow.apply_fill(f)
                    for s, why in rejects:
                        self.shadow.apply_reject(s, why)
            if self.shadow is not None:
                for i in self.shadow.on_bars(bars, now):
                    for s in self.sim.submit(i):
                        self.shadow.apply_reject(s, "cancelled at flatten/halt")

            self._reconcile_pending(now)
            self._sync_bracket_exits(now)
            for intent in self.engine.on_bars(bars, now):
                if isinstance(intent, EntryIntent):
                    self._handle_entry(intent, now)
                elif isinstance(intent, ExitIntent):
                    self._handle_exit(intent, now)
                elif isinstance(intent, CancelIntent):
                    self._handle_cancel(intent, now)

            # 4) done for the day?
            over = to_et(now).time() >= self.engine.flatten or self.engine.state.halted
            if over and not self.engine.positions and not self.engine.pending:
                break
            self._persist(now, "running")
            self.d.sleep(self.d.poll_seconds)

        # ---- wrap-up
        now = clock.now()
        if self.shadow is not None:
            for f in self.sim.force_close_all(last_close, now):
                self.shadow.apply_fill(f)
        leftover = self.broker.positions()
        if leftover:
            self._event(now, "not_flat", symbols=sorted(leftover), msg="EMERGENCY FLATTEN of the dedicated account")
            self.broker.emergency_flatten()
            for sym in list(self.engine.positions):
                ctx = self.engine.ctx.get(sym)
                self._apply_sell(FillEvent(sym, "sell", self.engine.positions[sym].qty,
                                           ctx.bars[-1].close if ctx and ctx.bars else self.engine.positions[sym].entry_price,
                                           now, reason="emergency_flatten"))
            status = "error"
        elif self.engine.state.halted:
            status = "halted"
        self._persist(now, status, force=True, ended_at=now.isoformat(), flat_verified=not leftover)
        self._learn()
        return {"status": status, "trades": len(self.engine.trades), "realized_pnl": self.engine.state.realized_pnl,
                "halted": self.engine.state.halted}


    def _learn(self) -> None:
        """After the day is fully recorded: save the journal, then learn. A
        failure here is logged, never raised — the session is already final."""
        if not self.cfg.learning.enabled:
            return
        try:
            self.engine.journal.partial = self.recovered
            self.store.save_journal(self.day, self.engine.journal_records(), "live")
        except Exception as e:
            _log(f"journal save failed: {type(e).__name__}: {e}")
            return
        if self.d.post_close is not None:
            try:
                self.d.post_close(self.day)
            except Exception as e:
                _log(f"post-close learning failed: {type(e).__name__}: {e}")


def run_session(deps: RunnerDeps) -> Dict[str, Any]:
    return SessionRunner(deps).run()


# -----------------------------------------------------------------------------
# Production wiring
# -----------------------------------------------------------------------------
def make_plan_day(cfg: DayTradingConfig, data: IntradayDataClient, universe: Sequence[str], use_llm: bool = True,
                  news_fn: Optional[Callable[[Sequence[str]], Dict[str, List[str]]]] = None,
                  lessons_fn: Optional[Callable[[], List[Dict[str, Any]]]] = None,
                  memory_fn: Optional[Callable[[], List[str]]] = None) -> PlanFn:
    """lessons_fn / memory_fn come from the learning loop. Either failing
    degrades to "no lessons" / "no memory" — never to a failed plan."""
    from daytrade_playbook import BENCHMARKS, snapshot_market

    def _safe(fn, default):
        if fn is None:
            return default
        try:
            return fn()
        except Exception as e:
            _log(f"learning input unavailable ({type(e).__name__}: {e}) — planning without it")
            return default

    def plan(day: date) -> DayPlan:
        daily = data.get_daily_bars(list(universe) + list(BENCHMARKS), day - timedelta(days=45), day - timedelta(days=1))
        cands = build_candidates(daily, cfg.universe.min_price, cfg.universe.min_avg_volume)
        if news_fn is not None:
            heads = news_fn([c.symbol for c in cands[:20]])
            for c in cands:
                c.headlines = heads.get(c.symbol, [])
        mkt = market_read(daily)
        pb = make_playbook(cands, mkt, day, cfg, use_llm=use_llm, memory=_safe(memory_fn, []))
        if cfg.learning.enabled:
            pb.lessons = _safe(lessons_fn, []) or None     # attached here, never by the model
        shadow_pb = deterministic_playbook(cands, cfg)
        prev = {s: sorted(bs, key=lambda b: b.ts)[-1].close for s, bs in daily.items() if bs}
        top = [c.symbol for c in cands[: cfg.universe.max_watchlist * 2]]
        return DayPlan(pb, shadow_pb, prev, top, snapshot_market(mkt))
    return plan


def build_real_deps(config_path: str, use_llm: bool) -> RunnerDeps:
    from alpaca.data.historical import StockHistoricalDataClient
    from daytrade_broker import load_credentials, make_daytrade_trading_client
    from daytrade_learning import active_lesson_dicts, memory_lines, run_post_close
    from daytrade_store import default_store
    from intraday_data import AlpacaClock, AlpacaIntradayDataClient, PollingBarFeed
    from universe import EQUITIES

    cfg = load_day_trading_config(config_path)
    tc = make_daytrade_trading_client()
    key, secret = load_credentials("daytrade")
    data = AlpacaIntradayDataClient(StockHistoricalDataClient(key, secret))

    def news_fn(symbols: Sequence[str]) -> Dict[str, List[str]]:
        try:
            from alpaca.data.historical.news import NewsClient
            from alpaca_clients import AlpacaNewsClient
            nc = AlpacaNewsClient(NewsClient(key, secret))
        except Exception:
            return {}
        out: Dict[str, List[str]] = {}
        for s in symbols:
            try:
                out[s] = nc.get_recent_headlines(s, 1)[:3]
            except Exception:
                out[s] = []                     # news is optional context; never block the day on it
        return out

    store = default_store()
    return RunnerDeps(cfg=cfg, store=store, broker=AlpacaDayTradeBroker(tc, dedicated=True),
                      clock=AlpacaClock(tc),
                      plan_day=make_plan_day(cfg, data, [e[0] for e in EQUITIES], use_llm, news_fn,
                                             lessons_fn=lambda: active_lesson_dicts(store),
                                             memory_fn=lambda: memory_lines(store)),
                      make_feed=lambda syms: PollingBarFeed(data, syms),
                      post_close=lambda day: run_post_close(store, cfg, day, use_llm=use_llm))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Meridian day-trading runner (one trading day, paper account)")
    ap.add_argument("--config", default="mandate.yaml")
    ap.add_argument("--no-llm", action="store_true", help="use the deterministic scanner playbook (no Claude call)")
    args = ap.parse_args(argv)
    result = run_session(build_real_deps(args.config, use_llm=not args.no_llm))
    _log(f"session result: {result}")
    return 1 if result.get("status") == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
