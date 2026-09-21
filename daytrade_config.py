"""
daytrade_config.py
===================
Typed configuration for the day-trading capability, loaded from the
`day_trading:` block of mandate.yaml — the SAME file that already holds the
user's investment mandate, so there's one place a user goes to set risk
limits for everything Meridian does.

Deliberately NOT part of mandate.yaml's `buckets:` tree. Buckets are
fractions of the TOTAL portfolio and are consumed by mandate_allocator.py
(which only ever reads `total`, `default_limits` and `buckets`, so an extra
top-level key is invisible to it — verified by a test below). The day-trading
sleeve is a fixed dollar amount living in its OWN Alpaca paper account (see
daytrade_broker.py for why), not a slice of the weight-based book.

LONG-ONLY, by explicit scope: `long_only: false` raises here instead of
being silently ignored — shorting needs borrow/locate handling, different
risk math, and different order plumbing, none of which exist yet, and a
config flag that appears to enable it but does nothing would be worse than
a loud error.

`load_day_trading_config` needs PyYAML; `config_from_dict` is stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any, Dict, List, Optional

KNOWN_STRATEGIES = ("orb_long", "vwap_pullback_long", "gap_and_go_long")


def _parse_hhmm(value: Any, name: str) -> time:
    try:
        h, m = str(value).split(":")
        return time(int(h), int(m))
    except Exception:
        raise ValueError(f"day_trading.session.{name} must look like 'HH:MM' (Eastern), got {value!r}")


@dataclass
class SessionRules:
    no_entry_first_minutes: int = 5      # opening auction noise — no NEW entries yet
    last_entry_et: time = time(15, 30)   # no new entries after this
    flatten_et: time = time(15, 45)      # everything closed by this, no exceptions


@dataclass
class RiskLimits:
    max_positions: int = 5
    risk_per_trade: float = 0.01         # fraction of sleeve risked (entry -> stop) per trade
    max_position_pct: float = 0.20       # max notional in ONE name, as a fraction of sleeve
    daily_loss_halt: float = 0.03        # realized+unrealized loss (fraction of sleeve) that halts the day
    max_trades_per_day: int = 20
    min_stop_pct: float = 0.002          # reject stops tighter than 0.2% — that's noise, not risk control
    flatten_on_halt: bool = True         # a daily-loss halt closes open positions too, not just blocks entries


@dataclass
class CostModel:
    """Every backtest number is NET of these. Regulatory fee defaults below
    are deliberately on the conservative (higher) side — verify current
    rates before trusting cost-sensitive conclusions:
      * sec_fee_per_million: SEC Section 31, charged on SELLS. The SEC has
        revised this rate several times, including downward; 27.80 is an
        older, higher figure that errs conservative.
      * taf_per_share / taf_cap: FINRA Trading Activity Fee on SELLS.
    Alpaca charges no commission on US equities, so slippage — not fees —
    is the cost that actually matters intraday, and it's the one a paper
    account does NOT charge you realistically (see daytrade_backtest.py)."""
    slippage_bps: float = 3.0            # adverse, per side, applied to market-style fills
    sec_fee_per_million: float = 27.80
    taf_per_share: float = 0.000166
    taf_cap: float = 8.30
    participation_cap: float = 0.10      # never fill more than 10% of a bar's volume

    def sell_fees(self, qty: float, price: float) -> float:
        notional = qty * price
        return (notional / 1_000_000.0) * self.sec_fee_per_million + min(qty * self.taf_per_share, self.taf_cap)


@dataclass
class UniverseRules:
    min_price: float = 5.0
    min_avg_volume: float = 1_000_000    # SIP-consolidated daily volume (from daily bars)
    max_watchlist: int = 30


@dataclass
class DayTradingConfig:
    enabled: bool = True
    sleeve_capital: float = 25_000.0
    long_only: bool = True
    universe: UniverseRules = field(default_factory=UniverseRules)
    limits: RiskLimits = field(default_factory=RiskLimits)
    session: SessionRules = field(default_factory=SessionRules)
    costs: CostModel = field(default_factory=CostModel)
    strategies: List[str] = field(default_factory=lambda: list(KNOWN_STRATEGIES))
    params: Dict[str, Dict[str, float]] = field(default_factory=dict)   # per-strategy overrides
    run_unfiltered_shadow: bool = True   # runner also simulates the no-playbook engine, for the LLM ablation


def _fill(dc_cls, raw: Optional[Dict[str, Any]], name: str):
    raw = dict(raw or {})
    known = set(dc_cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"day_trading.{name}: unknown key(s) {sorted(unknown)} (valid: {sorted(known)})")
    return dc_cls(**raw)


def config_from_dict(raw: Optional[Dict[str, Any]]) -> DayTradingConfig:
    raw = dict(raw or {})
    session_raw = dict(raw.pop("session", None) or {})
    for k in ("last_entry_et", "flatten_et"):
        if k in session_raw:
            session_raw[k] = _parse_hhmm(session_raw[k], k)

    strategies = raw.pop("strategies", None)
    params = raw.pop("params", None) or {}
    universe = _fill(UniverseRules, raw.pop("universe", None), "universe")
    limits = _fill(RiskLimits, raw.pop("limits", None), "limits")
    session = _fill(SessionRules, session_raw, "session")
    costs = _fill(CostModel, raw.pop("costs", None), "costs")
    raw.pop("account", None)   # documented in mandate.yaml; consumed by daytrade_broker, not here

    top_known = {"enabled", "sleeve_capital", "long_only", "run_unfiltered_shadow"}
    unknown = set(raw) - top_known
    if unknown:
        raise ValueError(f"day_trading: unknown key(s) {sorted(unknown)}")
    cfg = DayTradingConfig(universe=universe, limits=limits, session=session, costs=costs, **raw)
    if strategies is not None:
        cfg.strategies = list(strategies)
    cfg.params = {k: dict(v) for k, v in params.items()}

    if not cfg.long_only:
        raise ValueError("day_trading.long_only must be true — shorting is not implemented "
                         "(it needs borrow/locate handling and different risk math)")
    bad = [s for s in cfg.strategies if s not in KNOWN_STRATEGIES]
    if bad:
        raise ValueError(f"day_trading.strategies: unknown {bad} (known: {list(KNOWN_STRATEGIES)})")
    if cfg.sleeve_capital <= 0:
        raise ValueError("day_trading.sleeve_capital must be positive")
    if not 0 < cfg.limits.risk_per_trade <= 0.05:
        raise ValueError("day_trading.limits.risk_per_trade must be in (0, 0.05]")
    if not 0 < cfg.limits.max_position_pct <= 1.0:
        raise ValueError("day_trading.limits.max_position_pct must be in (0, 1]")
    if not 0 < cfg.limits.daily_loss_halt <= 0.2:
        raise ValueError("day_trading.limits.daily_loss_halt must be in (0, 0.2]")
    if not cfg.session.last_entry_et < cfg.session.flatten_et:
        raise ValueError("day_trading.session.last_entry_et must be before flatten_et")
    return cfg


def load_day_trading_config(path: str = "mandate.yaml") -> DayTradingConfig:
    import yaml
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return config_from_dict(data.get("day_trading"))


if __name__ == "__main__":
    print("TEST 1: defaults are valid, long-only, cash-sized")
    d = config_from_dict(None)
    assert d.long_only and d.limits.max_positions == 5 and d.session.flatten_et == time(15, 45)
    print("✓ empty config -> safe defaults")

    print("TEST 2: shorting is refused loudly, not silently ignored")
    try:
        config_from_dict({"long_only": False})
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "long_only" in str(e)
    print("✓ long_only: false raises")

    print("TEST 3: typos in config keys are errors, not silent no-ops")
    for bad in ({"limits": {"max_positon": 3}}, {"strategies": ["nope"]}, {"sleve_capital": 5}):
        try:
            config_from_dict(bad)
            raise AssertionError(f"should have raised for {bad}")
        except ValueError:
            pass
    print("✓ unknown limit key / strategy / top-level key all rejected")

    print("TEST 4: HH:MM parsing and ordering")
    c = config_from_dict({"session": {"last_entry_et": "15:00", "flatten_et": "15:40"}})
    assert c.session.last_entry_et == time(15, 0)
    try:
        config_from_dict({"session": {"last_entry_et": "15:50", "flatten_et": "15:40"}})
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("✓ parsed; last_entry after flatten rejected")

    print("TEST 5: real mandate.yaml loads, and the allocator ignores the new block")
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "mandate.yaml")
    cfg = load_day_trading_config(path)
    assert cfg.sleeve_capital > 0 and cfg.long_only
    from mandate_allocator import load_mandate
    m = load_mandate(path)
    assert "day_trading" in m and set(m["buckets"]) == {"equities", "forex", "crypto", "cash"}
    print(f"✓ sleeve=${cfg.sleeve_capital:,.0f}, strategies={cfg.strategies}; buckets untouched")

    print("TEST 6: sell-side fees")
    fee = CostModel().sell_fees(100, 50.0)
    assert 0 < fee < 1.0
    assert CostModel().sell_fees(10_000_000, 1.0) >= 8.30   # TAF capped, SEC part still scales
    print(f"✓ 100sh@$50 sell fees = ${fee:.4f}")
    print("\nAll daytrade_config tests passed.")
