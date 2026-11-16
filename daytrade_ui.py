"""
daytrade_ui.py
===============
The "⚡ Day Trading" tab — a Flask Blueprint, kept OUT of web_ui.py (already
3,000 lines, and it needs GCP credentials at import time, which would make
this untestable). web_ui.py registers it in three lines and adds one nav link;
everything it needs from the host app is injected through
make_daytrade_blueprint(), so it is tested here against a bare Flask app.

Routes
  GET  /daytrade                 dashboard: kill switch, today's session, live
                                 account/positions, recent days, roll-up
  GET  /daytrade/sessions/<date> one day in detail: trades, playbook, events,
                                 live-vs-shadow (the LLM ablation)
  POST /daytrade/halt            kill switch ON   — requires the shared token (P0.3)
  POST /daytrade/resume          kill switch OFF  — requires the shared token
  GET  /daytrade/api/status      JSON for monitoring
  GET  /daytrade/learning        the failure-learning loop: lessons by status with their
                                 out-of-sample evidence and live record, failure-tag totals
  POST /daytrade/lessons/<id>/activate   validated -> active   — requires the shared token
  POST /daytrade/lessons/<id>/retire     active/validated -> retired — requires the shared token

WHY LESSON CHANGES NEED THE TOKEN. A lesson can only restrict, but switching
one on or off still changes what the engine does tomorrow, so both go
through the same gate as resuming trading.

WHY HALT NEEDS THE TOKEN TOO (P0.3). Halting only reduces risk, so it used
to be open — but flipping trading state is still a state change only the
owner should make, and an open halt button is a denial-of-service on the
strategy (anyone reaching the page could halt trading every day). Halt,
resume, and lesson changes all go through the same gate now. If no token
is configured the gate FAILS CLOSED (refuses) instead of running
unprotected — the old opt-out is gone; set the daytrade-token secret
(or the token the host app injects) for local use.

PAPER ONLY, and the page says so. It never places an order itself — it
reads the store and the broker and flips one Firestore flag; the runner does
all trading.

All model-authored or news-derived text (playbook rationale, event details)
is rendered through Jinja autoescape — it is untrusted input. Tested.
"""

from __future__ import annotations

import hmac
from urllib.parse import urlencode
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, jsonify, redirect, render_template_string, request

from daytrade_attribution import tag_totals
from daytrade_config import DayTradingConfig
from daytrade_learning import activate_lesson, load_lessons, retire_lesson
from daytrade_store import DayTradeStore, summarize_sessions
from intraday_data import ET, session_date

PAGE_CSS = """
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 860px;
        margin: 0 auto; padding: 32px 16px 64px; color: #0d1117; line-height: 1.45; }
  h1 { font-size: 1.3rem; margin-bottom: 4px; } h2 { font-size: 1.05rem; margin: 26px 0 10px; }
  .paper-banner { background: #fff8e6; border-radius: 10px; padding: 8px 14px; font-size: 0.82rem;
                 margin-bottom: 20px; font-weight: 600; display: inline-block; }
  .stat-row { display: flex; gap: 12px; margin: 14px 0 20px; flex-wrap: wrap; }
  .stat { background: #f6f8ff; border-radius: 12px; padding: 12px 16px; flex: 1; min-width: 130px; }
  .stat .big { font-size: 1.3rem; font-weight: 800; }
  .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  table { width: 100%; min-width: 520px; border-collapse: collapse; font-size: 0.88rem; }
  th, td { text-align: left; padding: 7px 6px; border-bottom: 1px solid #eee; white-space: nowrap; }
  th { color: #888; font-weight: 500; }
  .pos { color: #00a340; } .neg { color: #d43f3f; } .muted { color: #888; }
  .warn { color: #a3312f; background: #fee; padding: 12px 14px; border-radius: 10px; margin: 12px 0; }
  .info { background: #f6f8ff; padding: 12px 14px; border-radius: 10px; margin: 12px 0; }
  .halted { background: #fee; border: 1px solid #f3b3b3; border-radius: 12px; padding: 14px 16px; margin: 14px 0; }
  .running { background: #eefaf1; border: 1px solid #b9e6c4; border-radius: 12px; padding: 14px 16px; margin: 14px 0; }
  form.inline { display: inline; } input[type=password], input[type=text] { padding: 8px; border: 1px solid #ccc;
        border-radius: 8px; font-size: 0.9rem; }
  button { padding: 9px 16px; border: 0; border-radius: 10px; font-weight: 700; cursor: pointer; }
  .btn-halt { background: #d43f3f; color: #fff; } .btn-resume { background: #0d1117; color: #fff; }
  .pill { display: inline-block; padding: 2px 9px; border-radius: 99px; font-size: 0.75rem; font-weight: 700;
         background: #eee; }
  .pill.completed { background: #dff5e5; } .pill.halted, .pill.error { background: #fbdcdc; }
  .pill.running { background: #dbe8ff; } .pill.skipped { background: #eee; }
  .meta { color: #999; font-size: 0.82rem; margin-top: 28px; }
  @media (max-width: 600px) { body { padding: 20px 14px 56px; } .stat { min-width: 100%; } }
</style>
"""

_MACROS = """
{% macro money(x) %}{% if x is none %}—{% else %}<span class="{{ 'pos' if x > 0 else 'neg' if x < 0 else '' }}">{{ '{:+,.2f}'.format(x) }}</span>{% endif %}{% endmacro %}
{% macro t_et(iso) %}{% if iso %}{{ iso_et(iso) }}{% else %}—{% endif %}{% endmacro %}
"""

DASHBOARD = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Day Trading</title>""" + PAGE_CSS + """</head>
<body>
""" + _MACROS + """
  <h1>⚡ Day Trading</h1>
  <div class="paper-banner">📝 Alpaca PAPER account (dedicated) — long-only, US equities, flat by {{ cfg.session.flatten_et.strftime('%H:%M') }} ET. Not real money.</div>

  {% if flash %}<div class="info">{{ flash }}</div>{% endif %}
  {% if control.halted %}
    <div class="halted"><b>⛔ Trading is HALTED</b> — {{ control.reason or 'no reason given' }}
      <span class="muted">(by {{ control.updated_by or 'unknown' }}, {{ t_et(control.updated_at) }})</span>
      <form class="inline" method="post" action="/daytrade/resume" style="margin-left:10px">
        {% if token_required %}<input type="password" name="token" placeholder="token" required>{% endif %}
        <button class="btn-resume">Resume trading</button></form></div>
  {% else %}
    <div class="running"><b>🟢 Trading is enabled.</b> Halting stops new entries and flattens open positions at the next cycle.
      <form class="inline" method="post" action="/daytrade/halt" style="margin-left:10px">
        <input type="text" name="reason" placeholder="reason (optional)">
        {% if token_required %}<input type="password" name="token" placeholder="token" required>{% endif %}
        <button class="btn-halt">HALT</button></form></div>
  {% endif %}

  {% if live_error %}<div class="warn">Live account unavailable: {{ live_error }}</div>{% endif %}
  {% if live %}
  <div class="stat-row">
    <div class="stat"><div>Equity</div><div class="big">${{ '{:,.2f}'.format(live.account.equity) }}</div></div>
    <div class="stat"><div>Cash</div><div class="big">${{ '{:,.2f}'.format(live.account.cash) }}</div></div>
    <div class="stat"><div>Sleeve</div><div class="big">${{ '{:,.0f}'.format(cfg.sleeve_capital) }}</div></div>
    <div class="stat"><div>Open positions</div><div class="big">{{ live.positions|length }}</div></div>
  </div>
  {% if live.positions %}
  <h2>Open positions (live)</h2><div class="table-wrap"><table>
    <tr><th>Symbol</th><th>Qty</th><th>Avg entry</th><th>Last</th><th>Unrealized</th></tr>
    {% for p in live.positions %}<tr><td>{{ p.symbol }}</td><td>{{ p.qty|int }}</td><td>{{ '{:,.2f}'.format(p.avg_entry_price) }}</td>
      <td>{{ '{:,.2f}'.format(p.current_price) }}</td><td>{{ money(p.unrealized_pl) }}</td></tr>{% endfor %}
  </table></div>{% endif %}{% endif %}

  <h2>Today {% if today %}<span class="pill {{ today.status }}">{{ today.status }}</span>{% endif %}</h2>
  {% if today %}
    <div class="stat-row">
      <div class="stat"><div>Realized (net)</div><div class="big">{{ money(today.realized_pnl or 0) }}</div></div>
      <div class="stat"><div>Trades</div><div class="big">{{ (today.trades or [])|length }}</div></div>
      <div class="stat"><div>Entries sent</div><div class="big">{{ today.entries_today or 0 }}</div></div>
      {% if today.shadow %}<div class="stat"><div>No-playbook shadow</div><div class="big">{{ money(today.shadow.pnl) }}</div></div>{% endif %}
    </div>
    {% if today.playbook %}<div class="info"><b>Playbook</b> ({{ today.playbook.source }}):
      risk {{ 'ON' if today.playbook.risk_on else 'OFF' }}, {{ '%.0f'|format(today.playbook.risk_multiplier * 100) }}% risk,
      {{ (today.playbook.watchlist or [])|length }} names. <span class="muted">{{ today.playbook.rationale }}</span></div>{% endif %}
    {% if today.halted %}<div class="warn">Halted: {{ today.halt_reason }}</div>{% endif %}
    <a href="/daytrade/sessions/{{ today.date }}">Full detail for {{ today.date }} →</a>
  {% else %}<p class="muted">No session recorded yet today. The runner starts before the open on trading days.</p>{% endif %}

  <h2>Recent days</h2>
  {% if summary.days %}
  <div class="stat-row">
    <div class="stat"><div>Net P&amp;L ({{ summary.days }} days)</div><div class="big">{{ money(summary.net_pnl) }}</div></div>
    <div class="stat"><div>Win rate</div><div class="big">{{ '{:.0%}'.format(summary.win_rate) if summary.win_rate is not none else '—' }}</div></div>
    <div class="stat"><div>Avg R</div><div class="big">{{ '{:+.2f}'.format(summary.avg_r) if summary.avg_r is not none else '—' }}</div></div>
    <div class="stat"><div>Halted days</div><div class="big">{{ summary.halted_days }}</div></div>
  </div>
  {% if summary.shadow_days %}<div class="info"><b>Does the LLM playbook help?</b> Over {{ summary.shadow_days }} day(s) with a shadow:
    live (playbook-restricted) {{ money(summary.live_net_pnl_on_shadow_days) }} vs. plain engine (no playbook, simulated) {{ money(summary.shadow_net_pnl) }}.
    <span class="muted">Small samples say nothing — see the promotion gate.</span></div>{% endif %}
  {% endif %}
  <div class="table-wrap"><table>
    <tr><th>Date</th><th>Status</th><th>Trades</th><th>Net P&amp;L</th><th>Shadow</th></tr>
    {% for s in sessions %}<tr><td><a href="/daytrade/sessions/{{ s.date }}">{{ s.date }}</a></td>
      <td><span class="pill {{ s.status }}">{{ s.status }}</span></td><td>{{ (s.trades or [])|length }}</td>
      <td>{{ money(s.realized_pnl) if s.realized_pnl is defined else '—' }}</td>
      <td>{{ money(s.shadow.pnl) if s.shadow else '—' }}</td></tr>
    {% else %}<tr><td colspan="5" class="muted">No sessions yet.</td></tr>{% endfor %}
  </table></div>

  <h2>Learning</h2>
  <div class="info">The engine journals every setup it sees — taken or not — and learns restrict-only rules from
    repeated failures, each tested out of sample before it can be switched on.
    <a href="/daytrade/learning">Lessons and failure analysis →</a></div>

  <h2>Limits in force</h2>
  <div class="info">{{ cfg.limits.max_positions }} positions max · {{ '%.1f'|format(cfg.limits.risk_per_trade*100) }}% risk/trade ·
    {{ '%.0f'|format(cfg.limits.max_position_pct*100) }}% max per name · halt at −{{ '%.0f'|format(cfg.limits.daily_loss_halt*100) }}% on the day ·
    {{ cfg.limits.max_trades_per_day }} trades/day · strategies: {{ cfg.strategies|join(', ') }}
    <div class="muted">Edit in mandate.yaml → day_trading. The playbook can only tighten these, never loosen them.</div></div>
  <div class="meta">Cash-sized (never margin) · bracket orders keep stops at the broker · the runner is a Cloud Run Job, this page only reads and can halt.</div>
</body></html>"""

SESSION_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Day Trading — {{ s.date }}</title>""" + PAGE_CSS + """</head>
<body>
""" + _MACROS + """
  <h1>⚡ {{ s.date }} <span class="pill {{ s.status }}">{{ s.status }}</span></h1>
  <div class="paper-banner">📝 Alpaca PAPER account — not real money.</div>
  <p><a href="/daytrade">← back to Day Trading</a></p>
  <div class="stat-row">
    <div class="stat"><div>Realized (net)</div><div class="big">{{ money(s.realized_pnl or 0) }}</div></div>
    <div class="stat"><div>Trades</div><div class="big">{{ (s.trades or [])|length }}</div></div>
    <div class="stat"><div>Sleeve</div><div class="big">${{ '{:,.0f}'.format(s.sleeve or 0) }}</div></div>
    {% if s.shadow %}<div class="stat"><div>Shadow (no playbook)</div><div class="big">{{ money(s.shadow.pnl) }}</div>
      <div class="muted">{{ s.shadow.entries }} entries{{ ' · partial (restart)' if s.shadow.partial else '' }}</div></div>{% endif %}
  </div>
  {% if s.halted %}<div class="warn">Halted: {{ s.halt_reason }}</div>{% endif %}
  {% if s.note %}<div class="info">{{ s.note }}</div>{% endif %}
  {% if s.playbook %}<h2>Playbook <span class="muted">({{ s.playbook.source }})</span></h2>
    <div class="info">Risk {{ 'ON' if s.playbook.risk_on else 'OFF' }} · {{ '%.0f'|format(s.playbook.risk_multiplier * 100) }}% ·
      strategies: {{ (s.playbook.enabled_strategies if s.playbook.enabled_strategies is not none else ['all configured'])|join(', ') }}<br>
      Watchlist: {{ (s.playbook.watchlist or [])|join(', ') or '—' }}<br><span class="muted">{{ s.playbook.rationale }}</span></div>{% endif %}
  <h2>Trades</h2><div class="table-wrap"><table>
    <tr><th>Symbol</th><th>Strategy</th><th>Qty</th><th>Entry</th><th>Exit</th><th>Net P&amp;L</th><th>R</th><th>Why out</th><th>In → out (ET)</th></tr>
    {% for t in s.trades or [] %}<tr><td>{{ t.symbol }}</td><td>{{ t.strategy }}</td><td>{{ t.qty|int }}</td>
      <td>{{ '{:,.2f}'.format(t.entry_price) }}</td><td>{{ '{:,.2f}'.format(t.exit_price) }}</td><td>{{ money(t.pnl) }}</td>
      <td>{{ '{:+.2f}'.format(t.r_multiple) }}</td><td>{{ t.reason }}</td><td>{{ t_et(t.entry_ts) }} → {{ t_et(t.exit_ts) }}</td></tr>
    {% else %}<tr><td colspan="9" class="muted">No trades.</td></tr>{% endfor %}
  </table></div>
  {% if s.open_positions %}<h2>Open at last update</h2><div class="info">
    {% for p in s.open_positions %}{{ p.symbol }} {{ p.qty|int }} @ {{ '{:,.2f}'.format(p.entry_price) }} (stop {{ p.stop }}, target {{ p.target }})<br>{% endfor %}</div>{% endif %}
  {% if s.attribution %}<h2>Why trades won or lost</h2>
    {% set a = s.attribution %}
    <div class="table-wrap"><table>
      <tr><th>Tag (taken trades)</th><th>Trades</th><th>Sum R</th></tr>
      {% for tag, v in (a.by_tag or {}).items()|sort %}<tr><td>{{ tag }}</td><td>{{ v.n }}</td><td>{{ '{:+.2f}'.format(v.sum_r) }}</td></tr>
      {% else %}<tr><td colspan="3" class="muted">No trades to attribute.</td></tr>{% endfor %}
    </table></div>
    {% if a.filters %}<h2>What each restriction did <span class="muted">(outcomes of setups NOT taken)</span></h2>
    <div class="table-wrap"><table>
      <tr><th>Stopped by</th><th>Setups</th><th>Their R</th><th>Verdict</th></tr>
      {% for k, v in a.filters.items()|sort %}<tr><td>{{ k }}</td><td>{{ v.n }}</td><td>{{ '{:+.2f}'.format(v.sum_r) }}</td>
        <td><span class="{{ 'pos' if v.verdict == 'saved' else 'neg' if v.verdict == 'cost' else '' }}">{{ v.verdict }}</span></td></tr>{% endfor %}
    </table></div>{% endif %}
  {% endif %}
  {% if s.learning %}<div class="info"><b>Post-close reflection</b> ({{ s.learning.source }}): {{ s.learning.summary }}
    {% if s.learning.validated %}<br>Validated: {% for v in s.learning.validated %}{{ v.rule }}{{ '; ' if not loop.last }}{% endfor %}{% endif %}
    {% if s.learning.rejected %}<br><span class="muted">Rejected {{ s.learning.rejected|length }} proposal(s) — see the Learning page.</span>{% endif %}
    {% if s.learning.error %}<br><span class="muted">Model unavailable: {{ s.learning.error }}</span>{% endif %}</div>{% endif %}
  {% if s.retired_lessons %}<div class="warn">Retired today: {% for r in s.retired_lessons %}{{ r.rule }} — {{ r.why }}{{ '; ' if not loop.last }}{% endfor %}</div>{% endif %}
  <h2>Decision &amp; event log</h2><div class="table-wrap"><table>
    <tr><th>Time (ET)</th><th>Kind</th><th>Symbol</th><th>Detail</th></tr>
    {% for e in s.events or [] %}<tr><td>{{ t_et(e.ts) }}</td><td>{{ e.kind }}</td><td>{{ e.symbol or '' }}</td>
      <td style="white-space:normal">{{ detail(e) }}</td></tr>
    {% else %}<tr><td colspan="4" class="muted">No events.</td></tr>{% endfor %}
  </table></div>
</body></html>"""


LEARNING_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Day Trading — Learning</title>""" + PAGE_CSS + """</head>
<body>
""" + _MACROS + """
  <h1>⚡ What the day-trading engine has learned</h1>
  <div class="paper-banner">📝 Alpaca PAPER account — not real money.</div>
  <p><a href="/daytrade">← back to Day Trading</a></p>
  {% if flash %}<div class="info">{{ flash }}</div>{% endif %}
  <div class="info">Lessons are rules over the entry features that can only <b>skip</b> a setup or <b>shrink</b> its size.
    A proposal (from Claude or the deterministic miner) must pass an out-of-sample test — at least
    {{ lc.min_matched_trades }} matching setups over {{ lc.min_distinct_days }} days, a significant loss after correcting for
    all {{ trials }} proposals so far, consistency over time, and ≤ {{ '%.0f'|format(lc.max_opportunity_cost*100) }}% of winning R
    thrown away. {{ 'Validated lessons switch on automatically.' if lc.auto_activate else 'A validated lesson waits for you to activate it.' }}
    Active lessons retire themselves if the trades they stop turn out to be winners.</div>

  {% for status, title in (('active', 'Active — enforced by the engine'), ('validated', 'Validated — awaiting activation'),
                            ('proposed', 'Proposed'), ('rejected', 'Rejected'), ('retired', 'Retired')) %}
  {% set group = by_status.get(status, []) %}
  {% if group or status in ('active', 'validated') %}
  <h2>{{ title }} <span class="muted">({{ group|length }})</span></h2>
  <div class="table-wrap"><table>
    <tr><th>Rule</th><th>Out-of-sample</th><th>Live (trades it acted on)</th><th>Source</th><th></th></tr>
    {% for l in group %}{% set v = l.validation or {} %}<tr>
      <td style="white-space:normal"><b>{{ l.describe() }}</b><br><span class="muted">{{ l.id }} · {{ l.hypothesis }}</span>
        {% if l.history %}<br><span class="muted">{{ l.history[-1].day }}: {{ l.history[-1].why }}</span>{% endif %}</td>
      <td>{% if v %}n={{ v.n }} · {{ '{:+.2f}'.format(v.mean_r) }}R<br>z {{ '%.2f'|format(v.z) }} / {{ '%.2f'|format(v.z_required) }} ·
        folds {{ v.fold_wins }}/{{ v.folds }}{% else %}—{% endif %}</td>
      <td>{{ l.live.n or 0 }} · {{ '{:+.2f}'.format(l.live.sum_r or 0) }}R</td>
      <td>{{ l.source }}</td>
      <td>{% if status in ('validated', 'active') %}
        <form class="inline" method="post" action="/daytrade/lessons/{{ l.id }}/{{ 'activate' if status == 'validated' else 'retire' }}">
          {% if token_required %}<input type="password" name="token" placeholder="token" required>{% endif %}
          <button class="{{ 'btn-resume' if status == 'validated' else 'btn-halt' }}">{{ 'Activate' if status == 'validated' else 'Retire' }}</button>
        </form>{% endif %}</td></tr>
    {% else %}<tr><td colspan="5" class="muted">None.</td></tr>{% endfor %}
  </table></div>
  {% endif %}
  {% endfor %}

  <h2>Failure tags on taken trades <span class="muted">(journal, all sources)</span></h2>
  <div class="table-wrap"><table>
    <tr><th>Tag</th><th>Trades</th><th>Sum R</th><th>Win rate</th></tr>
    {% for tag, v in tags|dictsort %}<tr><td>{{ tag }}</td><td>{{ v.n }}</td><td>{{ '{:+.2f}'.format(v.sum_r) }}</td>
      <td>{{ '{:.0%}'.format(v.wins / v.n) if v.n else '—' }}</td></tr>
    {% else %}<tr><td colspan="4" class="muted">No journaled trades yet. Run a backtest with --save-journal to seed history.</td></tr>{% endfor %}
  </table></div>
  <div class="meta">{{ trials }} lesson proposals tested so far · max {{ lc.max_active_lessons }} active ·
    stacked size cuts never below {{ '%.0f'|format(lc.min_combined_risk_mult*100) }}% · CLI: python daytrade_learning.py list</div>
</body></html>"""


def _iso_et(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone(ET).strftime("%H:%M:%S")
    except Exception:
        return str(iso)


def _detail(event: Dict[str, Any]) -> str:
    skip = {"ts", "kind", "symbol"}
    return " · ".join(f"{k}={v}" for k, v in event.items() if k not in skip)[:400]


def make_daytrade_blueprint(
    top_nav: str,
    store_fn: Callable[[], DayTradeStore],
    live_fn: Callable[[], Dict[str, Any]],
    token_fn: Callable[[], Optional[str]],
    config_fn: Callable[[], DayTradingConfig],
    today_fn: Callable[[], date] = lambda: session_date(datetime.now(ET)),
) -> Blueprint:
    bp = Blueprint("daytrade", __name__)

    def _wrap(page: str) -> str:
        return page.replace("<body>", "<body>\n" + top_nav, 1)

    def render(page: str, **ctx: Any) -> str:
        return render_template_string(_wrap(page), iso_et=_iso_et, detail=_detail, active_page="daytrade", **ctx)

    @bp.route("/daytrade")
    def dashboard():
        store, cfg = store_fn(), config_fn()
        live, live_error = None, None
        try:
            live = live_fn()
        except Exception as e:
            live_error = str(e)
        sessions = store.list_sessions(30)
        today = store.get_session(today_fn())
        return render(DASHBOARD, control=store.get_control(), cfg=cfg, live=live, live_error=live_error,
                      sessions=sessions, today=today, summary=summarize_sessions(sessions),
                      token_required=bool(token_fn()), flash=request.args.get("msg"))

    @bp.route("/daytrade/sessions/<day>")
    def session_detail(day: str):
        try:
            d = date.fromisoformat(day)
        except ValueError:
            return "bad date", 400
        s = store_fn().get_session(d)
        if s is None:
            return "no session for that date", 404
        return render(SESSION_PAGE, s=s)

    @bp.route("/daytrade/halt", methods=["POST"])
    def halt():
        # P0.3 — halting flips trading state, so it needs the shared token
        # like resume does (see _authorized below).
        if not _authorized():
            return ("unauthorized — halting trading requires the token", 403)
        reason = (request.form.get("reason") or "").strip()[:200] or "manual halt from the dashboard"
        store_fn().set_halted(True, reason, by="dashboard")
        return redirect("/daytrade?msg=Trading+halted.")

    def _authorized() -> bool:
        # P0.3 — fail closed: no token configured means NO token-gated
        # action runs (previously `not expected` returned True, leaving
        # resume/lessons unprotected whenever the secret was never set).
        expected = token_fn()
        if not expected:
            return False
        supplied = request.form.get("token") or request.headers.get("X-Daytrade-Token") or ""
        return hmac.compare_digest(supplied.encode(), expected.encode())

    @bp.route("/daytrade/resume", methods=["POST"])
    def resume():
        if not _authorized():
            return ("unauthorized — resuming trading requires the token", 403)
        store_fn().set_halted(False, "resumed", by="token" if token_fn() else "dashboard")
        return redirect("/daytrade?msg=Trading+resumed.")

    @bp.route("/daytrade/learning")
    def learning():
        store, cfg = store_fn(), config_fn()
        by_status: Dict[str, List[Any]] = {}
        for l in load_lessons(store):
            by_status.setdefault(l.status, []).append(l)
        try:
            tags = tag_totals(store.list_journal())
        except Exception:
            tags = {}
        return render(LEARNING_PAGE, by_status=by_status, lc=cfg.learning, trials=store.get_trials(), tags=tags,
                      token_required=bool(token_fn()), flash=request.args.get("msg"))

    @bp.route("/daytrade/lessons/<lesson_id>/<action>", methods=["POST"])
    def lesson_action(lesson_id: str, action: str):
        if action not in ("activate", "retire"):
            return "unknown action", 404
        if not _authorized():
            return ("unauthorized — changing lessons requires the token", 403)
        by = "token" if token_fn() else "dashboard"
        store = store_fn()
        err = (activate_lesson(store, config_fn(), lesson_id, by, today_fn()) if action == "activate"
               else retire_lesson(store, lesson_id, by, "retired from the dashboard", today_fn()))
        msg = err or f"Lesson {lesson_id} {'activated' if action == 'activate' else 'retired'}."
        return redirect("/daytrade/learning?" + urlencode({"msg": msg}))

    @bp.route("/daytrade/api/status")
    def api_status():
        store = store_fn()
        sessions = store.list_sessions(30)
        today = store.get_session(today_fn())
        slim = None
        if today:
            slim = {k: today.get(k) for k in ("date", "status", "realized_pnl", "unrealized_pnl", "halted",
                                             "halt_reason", "entries_today", "open_positions", "updated_at")}
            slim["trades"] = len(today.get("trades", []))
        return jsonify({"control": store.get_control(), "today": slim, "summary": summarize_sessions(sessions)})

    return bp


if __name__ == "__main__":
    from datetime import timedelta, timezone
    from flask import Flask
    from daytrade_config import config_from_dict
    from daytrade_store import InMemoryDayTradeStore

    NAV = '<nav class="topnav"><a href="/">Home</a><a href="/daytrade" class="{{ \'active\' if active_page == \'daytrade\' else \'\' }}">Day</a></nav>'
    store = InMemoryDayTradeStore()
    today = date(2026, 3, 4)
    t0 = datetime(2026, 3, 4, 14, 48, tzinfo=timezone.utc)
    trade = {"symbol": "AAA", "strategy": "orb_long", "qty": 49, "entry_price": 100.6, "exit_price": 101.8,
             "entry_ts": t0.isoformat(), "exit_ts": (t0 + timedelta(minutes=9)).isoformat(), "pnl": 58.65,
             "fees": 0.15, "reason": "target", "r_multiple": 1.9}
    store.save_session(today, {"status": "completed", "sleeve": 25_000, "realized_pnl": 58.65, "entries_today": 1,
                               "trades": [trade], "shadow": {"pnl": 40.0, "entries": 1, "partial": False},
                               "playbook": {"risk_on": True, "risk_multiplier": 0.5, "watchlist": ["AAA"],
                                            "enabled_strategies": None, "source": "claude",
                                            "rationale": "<script>alert('xss')</script> keep size down"},
                               "events": [{"ts": t0.isoformat(), "kind": "entry", "symbol": "AAA",
                                           "reason": "<img src=x onerror=alert(1)>"}]})
    store.save_session(today - timedelta(days=1), {"status": "halted", "halted": True, "halt_reason": "loss",
                                                   "realized_pnl": -750.0, "trades": [{**trade, "pnl": -750.0}]})
    state = {"token": "s3cret", "live_ok": True}
    live = lambda: (_ for _ in ()).throw(RuntimeError("keys not configured")) if not state["live_ok"] else {
        "account": {"equity": 25_000.0, "cash": 20_000.0},
        "positions": [{"symbol": "BBB", "qty": 30, "avg_entry_price": 50.0, "current_price": 50.5, "unrealized_pl": 15.0}]}
    app = Flask(__name__)
    app.register_blueprint(make_daytrade_blueprint(NAV, lambda: store, live, lambda: state["token"],
                                                   lambda: config_from_dict({}), today_fn=lambda: today))
    c = app.test_client()

    print("TEST 1: dashboard renders with live positions, today's session, history and the LLM-ablation card")
    r = c.get("/daytrade")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    for needle in ("Day Trading", "PAPER", "Trading is enabled", "BBB", "+58.65", "Playbook",
                   "Does the LLM playbook help?", "-750.00", "Limits in force", 'class="topnav"'):
        assert needle in html, f"missing: {needle}"
    assert "active" in html.split('href="/daytrade"')[1][:40], "nav tab is highlighted"
    print("✓ 200; nav highlighted; positions, P&L, ablation card, limits all present")

    print("TEST 2: untrusted text (LLM rationale, event details) is ESCAPED, never rendered as HTML")
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    d = c.get("/daytrade/sessions/2026-03-04").get_data(as_text=True)
    assert "<img src=x" not in d and "&lt;img" in d and "<script>alert" not in d
    print("✓ <script> and <img onerror> come out as text on both pages")

    print("TEST 3: session detail shows trades, why-out, playbook and shadow")
    assert "target" in d and "1.90" in d and "Shadow (no playbook)" in d and "09:48:00 → 09:57:00" in d
    assert c.get("/daytrade/sessions/2026-03-05").status_code == 404 and c.get("/daytrade/sessions/nope").status_code == 400
    print("✓ ET times, R multiple, shadow; 404 for missing day, 400 for a malformed one")

    print("TEST 4: a broker outage degrades to a banner — the page still loads")
    state["live_ok"] = False
    r = c.get("/daytrade")
    assert r.status_code == 200 and "Live account unavailable: keys not configured" in r.get_data(as_text=True)
    state["live_ok"] = True
    print("✓ 200 with a warning")

    print("TEST 5: HALT is open to anyone; RESUME needs the token")
    r = c.post("/daytrade/halt", data={"reason": "testing"})
    assert r.status_code == 302 and store.get_control()["halted"] and store.get_control()["reason"] == "testing"
    assert "Trading is HALTED" in c.get("/daytrade").get_data(as_text=True)
    assert c.post("/daytrade/resume", data={}).status_code == 403 and store.get_control()["halted"]
    assert c.post("/daytrade/resume", data={"token": "wrong"}).status_code == 403 and store.get_control()["halted"]
    r = c.post("/daytrade/resume", data={"token": "s3cret"})
    assert r.status_code == 302 and not store.get_control()["halted"]
    c.post("/daytrade/halt"); assert c.post("/daytrade/resume", headers={"X-Daytrade-Token": "s3cret"}).status_code == 302
    print("✓ halt: 302, resume: 403 without/with-wrong token, 302 with the form token or the header")

    print("TEST 6: with NO token configured, resume is allowed (explicit local opt-out, like web_ui.py)")
    state["token"] = None
    store.set_halted(True, "x")
    assert c.post("/daytrade/resume").status_code == 302 and not store.get_control()["halted"]
    store.set_halted(True, "y")
    assert 'type="password"' not in c.get("/daytrade").get_data(as_text=True)
    state["token"] = "s3cret"
    store.set_halted(False)
    print("✓ ok; no password box shown when no token is required")

    print("TEST 7: JSON status for monitoring")
    j = c.get("/daytrade/api/status").get_json()
    assert j["today"]["status"] == "completed" and j["today"]["trades"] == 1 and j["control"]["halted"] is False
    assert j["summary"]["days"] == 2 and j["summary"]["net_pnl"] == 58.65 - 750.0
    print("✓ ok")

    print("TEST 8: an empty store renders cleanly")
    empty = Flask(__name__)
    empty.register_blueprint(make_daytrade_blueprint(NAV, lambda: InMemoryDayTradeStore(), live, lambda: None,
                                                     lambda: config_from_dict({}), today_fn=lambda: today))
    r = empty.test_client().get("/daytrade")
    assert r.status_code == 200 and "No session recorded yet today" in r.get_data(as_text=True)
    print("✓ ok")

    print("TEST 9: session page shows attribution and the post-close reflection, escaped")
    store.save_session(today, {"attribution": {"taken": {"n": 1, "sum_r": 1.9, "wins": 1},
                                               "by_tag": {"win": {"n": 1, "sum_r": 1.9, "wins": 1}},
                                               "filters": {"off_watchlist": {"n": 2, "sum_r": -1.5, "wins": 0, "verdict": "saved"}}},
                               "learning": {"source": "claude", "summary": "<b>gaps</b> faded", "validated": [], "rejected": [{}]}})
    d = c.get("/daytrade/sessions/2026-03-04").get_data(as_text=True)
    assert "Why trades won or lost" in d and "off_watchlist" in d and "saved" in d and "&lt;b&gt;gaps" in d
    print("✓ ok")

    print("TEST 10: learning page lists lessons by status; activate/retire need the token and respect the rules")
    from daytrade_lessons import parse_lesson
    L = parse_lesson({"when": {"strategy": {"==": "orb_long"}, "gap_pct": {">=": 0.04}}, "action": {"type": "skip"},
                      "hypothesis": "<script>x</script>"}, 0.25, source="claude")
    L.set_status("validated", "2026-03-04", "passed")
    L.validation = {"n": 40, "mean_r": -0.6, "z": 3.1, "z_required": 1.9, "fold_wins": 3, "folds": 4}
    R = parse_lesson({"when": {"rel_volume": {">": 9}}, "action": {"type": "skip"}}, 0.25)
    R.set_status("rejected", "2026-03-04", "only 3 matching trades")
    store.save_lesson(L.to_dict()); store.save_lesson(R.to_dict())
    store.save_journal(today, [{"status": "taken", "r_real": -1.0, "exit_reason": "stop", "mfe_r": 0.1,
                                "strategy": "orb_long", "snapshot": {}}])
    page = c.get("/daytrade/learning").get_data(as_text=True)
    assert "Validated — awaiting activation" in page and "gap_pct &gt;= 0.04" in page and "Rejected" in page
    assert "false_breakout" in page and "<script>x" not in page and "Activate" in page
    url = f"/daytrade/lessons/{L.id}/activate"
    assert c.post(url, data={}).status_code == 403 and store.list_lessons()[0]["status"] != "active"
    r = c.post(url, data={"token": "s3cret"})
    assert r.status_code == 302 and "activated" in r.headers["Location"]
    assert {x["id"]: x["status"] for x in store.list_lessons()}[L.id] == "active"
    r = c.post(f"/daytrade/lessons/{R.id}/activate", data={"token": "s3cret"})
    assert "only+a+validated" in r.headers["Location"] and {x["id"]: x["status"] for x in store.list_lessons()}[R.id] == "rejected"
    assert c.post(f"/daytrade/lessons/{L.id}/delete", data={"token": "s3cret"}).status_code == 404
    assert c.post(f"/daytrade/lessons/{L.id}/retire", data={"token": "s3cret"}).status_code == 302
    assert {x["id"]: x["status"] for x in store.list_lessons()}[L.id] == "retired"
    print("✓ 403 without token; validated -> active -> retired with it; a rejected lesson can't be activated")
    print("\nAll daytrade_ui tests passed.")
