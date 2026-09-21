"""
daytrade_risk.py
=================
The deterministic, non-overridable risk layer for day trading — same
philosophy as firm_graph._deterministic_risk_check: no LLM, no learning, no
"but the setup looks great". A proposal (Signal) goes in; a sized, approved
entry or a list of vetoes comes out.

Nothing an LLM produces can widen any limit here. The pre-market playbook
(daytrade_playbook.py) may only make trading MORE restrictive (fewer names,
risk-off day); it never reaches this module's limits.

Sizing is CASH-based: quantity is capped by cash the sleeve actually has,
never by margin buying power, so the capability can't leverage itself even
though the account could. That also keeps it clear of Alpaca's intraday-
margin-deficit machinery entirely.

Pure functions + a plain RiskState dataclass — no I/O, fully testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import List, Set, Tuple

from daytrade_config import RiskLimits, SessionRules
from daytrade_strategies import Signal
from intraday_data import minutes_since_open, to_et


@dataclass
class RiskState:
    realized_pnl: float = 0.0          # net of fees, closed trades today
    unrealized_pnl: float = 0.0        # marked to the latest bar
    entries_today: int = 0             # entry ORDERS emitted (filled or not) — a rejected-and-retried loop must still count
    open_or_pending: int = 0           # positions held + entries in flight
    committed_notional: float = 0.0    # cash tied up in open positions + pending entries
    symbols_traded: Set[str] = field(default_factory=set)
    halted: bool = False
    halt_reason: str = ""


def daily_pnl(state: RiskState) -> float:
    return state.realized_pnl + state.unrealized_pnl


def daily_loss_breached(state: RiskState, sleeve: float, limits: RiskLimits) -> bool:
    return daily_pnl(state) <= -limits.daily_loss_halt * sleeve


def size_position(entry: float, stop: float, sleeve: float, cash_available: float,
                  limits: RiskLimits) -> int:
    """Whole shares (bracket orders and fractional shares don't mix at the
    broker, and the rest of Meridian is already whole-share). Smallest of:
    risk budget, per-name cap, and cash on hand."""
    per_share_risk = entry - stop
    if per_share_risk <= 0 or entry <= 0:
        return 0
    by_risk = math.floor(sleeve * limits.risk_per_trade / per_share_risk)
    by_cap = math.floor(sleeve * limits.max_position_pct / entry)
    by_cash = math.floor(max(cash_available, 0.0) / entry)
    return max(0, min(by_risk, by_cap, by_cash))


def entry_window_open(ts: datetime, session: SessionRules, last_entry: time) -> bool:
    """`ts` is the moment of DECISION (signal bar's end). Entries begin after
    the opening-auction noise and stop at last_entry (already adjusted for
    early-close days by intraday_data.effective_session)."""
    return minutes_since_open(ts) >= session.no_entry_first_minutes and to_et(ts).time() < last_entry


def check_entry(signal: Signal, decision_ts: datetime, state: RiskState, sleeve: float,
                limits: RiskLimits, session: SessionRules, last_entry: time,
                min_price: float = 0.0) -> Tuple[int, List[str]]:
    """Returns (qty, reasons). qty == 0 means VETOED and `reasons` says why —
    every veto is collected, not just the first, so the audit log shows the
    whole picture."""
    reasons: List[str] = []
    if state.halted:
        reasons.append(f"trading halted: {state.halt_reason or 'halted'}")
    if signal.symbol in state.symbols_traded:
        reasons.append(f"{signal.symbol} already traded today (one entry per name per day)")
    if not entry_window_open(decision_ts, session, last_entry):
        reasons.append("outside the entry window")
    if state.open_or_pending >= limits.max_positions:
        reasons.append(f"at max_positions ({limits.max_positions})")
    if state.entries_today >= limits.max_trades_per_day:
        reasons.append(f"at max_trades_per_day ({limits.max_trades_per_day})")
    if daily_loss_breached(state, sleeve, limits):
        reasons.append("daily loss limit reached")
    if signal.entry < min_price:
        reasons.append(f"price {signal.entry:.2f} below min_price {min_price:.2f}")
    if not signal.stop < signal.entry < signal.target:
        reasons.append("malformed signal: need stop < entry < target")
    else:
        # Alpaca rejects bracket legs within a cent of the base price.
        if round(signal.entry - signal.stop, 2) < 0.01 or round(signal.target - signal.entry, 2) < 0.01:
            reasons.append("stop/target within $0.01 of entry — broker would reject the bracket")
        if (signal.entry - signal.stop) / signal.entry < limits.min_stop_pct:
            reasons.append(f"stop only {(signal.entry - signal.stop) / signal.entry:.2%} away "
                           f"(min {limits.min_stop_pct:.2%}) — noise, not risk control")
    if reasons:
        return 0, reasons

    cash = sleeve + state.realized_pnl - state.committed_notional
    qty = size_position(signal.entry, signal.stop, sleeve, cash, limits)
    if qty < 1:
        return 0, ["position size rounds to 0 shares (risk budget, per-name cap, or cash too small)"]
    return qty, []


if __name__ == "__main__":
    from datetime import date, timedelta, timezone
    from intraday_data import et_datetime

    limits, session = RiskLimits(), SessionRules()
    last_entry = time(15, 30)
    day = date(2026, 3, 4)
    at = lambda h, m: et_datetime(day, time(h, m)).astimezone(timezone.utc)
    sig = Signal("AAA", "orb_long", entry=100.0, stop=99.0, target=102.0, reason="t")
    SLEEVE = 25_000.0

    print("TEST 1: sizing = min(risk budget, per-name cap, cash)")
    # risk: 25000*1% = $250 / $1 = 250 sh; cap: 25000*20%/100 = 50 sh; cash 25000 -> 250 sh
    assert size_position(100, 99, SLEEVE, SLEEVE, limits) == 50, "per-name cap should bind"
    assert size_position(100, 90, SLEEVE, SLEEVE, limits) == 25, "wide stop -> risk budget binds (250/10)"
    assert size_position(100, 99, SLEEVE, 1_234, limits) == 12, "cash binds"
    assert size_position(100, 100, SLEEVE, SLEEVE, limits) == 0 and size_position(100, 101, SLEEVE, SLEEVE, limits) == 0
    print("✓ 50 (cap), 25 (risk), 12 (cash), 0 (no risk distance)")

    print("TEST 2: a clean signal inside the window is approved and sized")
    qty, why = check_entry(sig, at(10, 0), RiskState(), SLEEVE, limits, session, last_entry)
    assert qty == 50 and not why
    print("✓ 50 shares")

    print("TEST 3: each veto fires, and ALL applicable vetoes are reported")
    bad = RiskState(halted=True, halt_reason="kill switch", open_or_pending=5, entries_today=20,
                    realized_pnl=-800.0, symbols_traded={"AAA"})
    qty, why = check_entry(sig, at(15, 40), bad, SLEEVE, limits, session, last_entry)
    joined = " | ".join(why)
    assert qty == 0
    for needle in ("halted", "already traded", "entry window", "max_positions", "max_trades_per_day", "daily loss"):
        assert needle in joined, f"missing veto: {needle}  ({joined})"
    print(f"✓ {len(why)} vetoes reported together")

    print("TEST 4: entry window edges")
    assert check_entry(sig, at(9, 34), RiskState(), SLEEVE, limits, session, last_entry)[0] == 0    # 4 min in
    assert check_entry(sig, at(9, 35), RiskState(), SLEEVE, limits, session, last_entry)[0] > 0     # 5 min in
    assert check_entry(sig, at(15, 29), RiskState(), SLEEVE, limits, session, last_entry)[0] > 0
    assert check_entry(sig, at(15, 30), RiskState(), SLEEVE, limits, session, last_entry)[0] == 0   # last_entry is exclusive
    print("✓ 09:34 no, 09:35 yes, 15:29 yes, 15:30 no")

    print("TEST 5: daily-loss halt uses realized + unrealized, at exactly -3% of sleeve")
    assert not daily_loss_breached(RiskState(realized_pnl=-500, unrealized_pnl=-249), SLEEVE, limits)
    assert daily_loss_breached(RiskState(realized_pnl=-500, unrealized_pnl=-250), SLEEVE, limits)
    assert daily_loss_breached(RiskState(realized_pnl=100, unrealized_pnl=-851), SLEEVE, limits)
    print("✓ -$749 ok, -$750 halts (unrealized counts)")

    print("TEST 6: malformed / untradeable signals are vetoed")
    for s, needle in [(Signal("A", "x", 100, 101, 102, ""), "malformed"),
                      (Signal("A", "x", 100, 99.996, 102, ""), "$0.01"),
                      (Signal("A", "x", 100, 99.9, 102, ""), "noise")]:
        q, w = check_entry(s, at(10, 0), RiskState(), SLEEVE, limits, session, last_entry)
        assert q == 0 and any(needle in x for x in w), (needle, w)
    q, w = check_entry(Signal("A", "x", 3.0, 2.9, 3.3, ""), at(10, 0), RiskState(), SLEEVE, limits, session,
                       last_entry, min_price=5.0)
    assert q == 0 and any("min_price" in x for x in w)
    print("✓ inverted, sub-cent, sub-0.2% stops, and sub-$5 names all vetoed")

    print("TEST 7: cash is what's LEFT — committed notional shrinks the next position, never leverages")
    st = RiskState(committed_notional=24_900.0)
    q, w = check_entry(sig, at(10, 0), st, SLEEVE, limits, session, last_entry)
    assert q == 1, q   # only $100 of cash remains -> 1 share
    st2 = RiskState(committed_notional=25_000.0)
    q2, w2 = check_entry(sig, at(10, 0), st2, SLEEVE, limits, session, last_entry)
    assert q2 == 0 and "rounds to 0" in w2[0]
    print("✓ $100 left -> 1 share; $0 left -> vetoed")
    print("\nAll daytrade_risk tests passed.")
