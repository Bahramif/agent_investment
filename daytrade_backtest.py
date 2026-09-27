"""
daytrade_backtest.py
=====================
Backtest harness for the day-trading capability, and the bridge into the
EXISTING champion/challenger machinery (promotion_gate.py +
promotion_criteria.yaml). This is the go/no-go instrument: if no strategy
survives here, no live infrastructure is worth running.

It drives the SAME DayTradingEngine that trades live (daytrade_engine.py),
fed minute by minute, with fills coming from SimBroker below instead of
Alpaca. Same signals, same risk vetoes, same flatten logic.

FILL MODEL — deliberately pessimistic, because optimism is how intraday
backtests lie:
  * A signal fires on bar t's CLOSE; the entry fills at bar t+1's OPEN plus
    slippage. No filling at the signal price.
  * Stops are stop-MARKET: fill at the stop minus slippage, or at the open
    (worse) if the bar gaps through it.
  * Targets are limits and only fill if price trades THROUGH by a cent, not
    merely touches — a touched limit usually isn't filled.
  * If a bar hits both stop and target, the STOP is assumed to have hit first.
  * Fills are capped at participation_cap of the bar's volume (IEX volume, so
    this is conservative); an entry that can't fill at least 1 share is a
    capacity reject, and those are counted and gated on.
  * Sells pay SEC/FINRA regulatory fees (daytrade_config.CostModel).
  * An entry with no bar for that symbol within 3 minutes is cancelled — an
    illiquid name must not fill at a stale price.
Alpaca PAPER trading fills more generously than this. Backtest P&L below
what paper shows is the expected, healthy direction.

THE CHAMPION is a NULL MODEL, not a previous strategy: random entries with
the same stop/target/risk/flatten machinery, calibrated to the SAME number of
trades. A real cash-flat champion can't work — promotion_criteria.yaml's
relative drawdown/CVaR gates compare against the champion's own (zero)
values. The null model answers the question that matters: does the SIGNAL
add anything beyond market drift and exit mechanics?

THE GATE WILL FAIL `ops.shadow_vs_sim` UNTIL PAPER RESULTS EXIST, on purpose:
build_challenger_report() leaves shadow_return as NaN unless you pass real
paper-trading results (Phase 5), and the gate's comparison against NaN is
False. A strategy must not be promotable on backtest evidence alone.

LEARNING JOURNAL. Every simulated day also produces the engine's journal
(daytrade_journal.py): each signal's entry features plus a stand-alone
virtual outcome. That is the history the failure-learning loop validates
lessons against — thousands of labelled setups instead of a few per paper
day. `run --save-journal` stores it (source "backtest"); `--journal-out
FILE.jsonl` writes it locally. `--lessons active` replays the currently
active lessons, to see what they would have done to the whole backtest.

HONEST LIMITATIONS: walk-forward "folds" here are fixed-parameter time
splits (there is nothing being re-fit), parameter robustness is a joint
+/-pct perturbation of all parameters (4 corners), not a full grid, and the
Deflated Sharpe uses the single-strategy sampling variance as V[SR]. Each is
noted where it is computed.
"""

from __future__ import annotations

import dataclasses
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from daytrade_config import CostModel, DayTradingConfig, config_from_dict
from daytrade_engine import (CancelIntent, DayTradingEngine, EntryIntent, ExitIntent, FillEvent, Playbook,
                             Trade)
from daytrade_strategies import DEFAULTS
from intraday_data import (MARKET_CLOSE, Bar, group_by_day, to_et, trading_minutes)
from promotion_gate import ChallengerReport, PromotionVerdict, StrategyMetrics, evaluate_promotion

TRADING_DAYS = 252
MAX_ENTRY_DELAY = timedelta(minutes=3)


# -----------------------------------------------------------------------------
# Simulated broker
# -----------------------------------------------------------------------------
class SimBroker:
    def __init__(self, cost: CostModel):
        self.cost = cost
        self.pending_entries: Dict[str, EntryIntent] = {}
        self.pending_exits: Dict[str, ExitIntent] = {}
        self.open: Dict[str, Dict[str, float]] = {}
        self.entries_attempted = 0
        self.capacity_rejects = 0
        self.max_participation = 0.0

    @property
    def _slip(self) -> float:
        return self.cost.slippage_bps / 1e4

    def submit(self, intent: Any) -> List[str]:
        """Returns symbols whose in-flight ENTRY was cancelled (for the caller
        to report back to the engine as rejects)."""
        if isinstance(intent, EntryIntent):
            self.pending_entries[intent.symbol] = intent
            self.entries_attempted += 1
        elif isinstance(intent, ExitIntent):
            self.pending_exits[intent.symbol] = intent
        elif isinstance(intent, CancelIntent):
            if self.pending_entries.pop(intent.symbol, None) is not None:
                return [intent.symbol]
        return []

    def _sell(self, sym: str, qty: float, price: float, ts: datetime, reason: str) -> FillEvent:
        self.open.pop(sym, None)
        return FillEvent(sym, "sell", qty, price, ts, reason, fees=self.cost.sell_fees(qty, price))

    def on_bar(self, bar: Bar) -> Tuple[List[FillEvent], List[Tuple[str, str]]]:
        fills: List[FillEvent] = []
        rejects: List[Tuple[str, str]] = []
        sym, slip = bar.symbol, self._slip

        exit_intent = self.pending_exits.pop(sym, None)
        if exit_intent is not None and sym in self.open:
            fills.append(self._sell(sym, self.open[sym]["qty"], bar.open * (1 - slip), bar.ts, exit_intent.reason))
            return fills, rejects

        entry = self.pending_entries.pop(sym, None)
        if entry is not None:
            if bar.ts - entry.decision_ts > MAX_ENTRY_DELAY:
                rejects.append((sym, "no bar within 3 minutes of the signal — cancelled, not filled at a stale price"))
            else:
                qty = min(entry.qty, math.floor(self.cost.participation_cap * bar.volume))
                if qty < 1:
                    self.capacity_rejects += 1
                    rejects.append((sym, f"capacity: bar volume {bar.volume:.0f} too thin"))
                else:
                    px = bar.open * (1 + slip)
                    self.max_participation = max(self.max_participation, qty / max(bar.volume, 1.0))
                    fills.append(FillEvent(sym, "buy", qty, px, bar.ts))
                    self.open[sym] = {"qty": float(qty), "stop": entry.stop, "target": entry.target}

        pos = self.open.get(sym)
        if pos is not None:
            stop, target = pos["stop"], pos["target"]
            if bar.open <= stop:
                fills.append(self._sell(sym, pos["qty"], bar.open * (1 - slip), bar.ts, "stop"))
            elif bar.open >= target:
                fills.append(self._sell(sym, pos["qty"], bar.open, bar.ts, "target"))
            elif bar.low <= stop:
                fills.append(self._sell(sym, pos["qty"], stop * (1 - slip), bar.ts, "stop"))
            elif bar.high >= target + 0.01:
                fills.append(self._sell(sym, pos["qty"], target, bar.ts, "target"))
        return fills, rejects

    def force_close_all(self, last_close: Dict[str, float], ts: datetime) -> List[FillEvent]:
        """Safety net for a name that printed no bar after the flatten order."""
        out = []
        for sym in list(self.open):
            px = last_close.get(sym, self.open[sym]["stop"]) * (1 - self._slip)
            out.append(self._sell(sym, self.open[sym]["qty"], px, ts, "eod_forced"))
        return out


# -----------------------------------------------------------------------------
# One day, and many days
# -----------------------------------------------------------------------------
@dataclass
class DayResult:
    day: date
    trades: List[Trade]
    pnl: float
    halted: bool
    halt_reason: str
    entries: int
    capacity_rejects: int
    max_participation: float
    universe_return: float          # median open->close return across symbols: the regime label's source
    journal: List[Dict[str, Any]] = field(default_factory=list)   # learning journal (daytrade_journal)


@dataclass
class BacktestResult:
    days: List[DayResult]
    sleeve: float
    label: str = ""

    @property
    def daily_returns(self) -> List[float]:
        return [d.pnl / self.sleeve for d in self.days]

    @property
    def trades(self) -> List[Trade]:
        return [t for d in self.days for t in d.trades]


def _close_time_for(bars_by_symbol: Dict[str, List[Bar]]) -> time:
    """Real early closes (day after Thanksgiving, Christmas Eve, ...) end at
    exactly 13:00 ET. Inferred from the data rather than a hardcoded calendar,
    and ONLY when the last bar ends within a few minutes of 13:00. Anything
    else — a normal day whose final minutes had no IEX prints, or a day whose
    data was simply truncated — is treated as a full session. A looser rule
    would silently mistake truncated data for an early close and disable
    entries for the whole day, which looks exactly like "the strategy found no
    signals"."""
    last_end = max((b.end for bars in bars_by_symbol.values() for b in bars), default=None)
    if last_end is not None:
        t = to_et(last_end).time()
        if time(12, 55) <= t <= time(13, 5):
            return time(13, 0)
    return MARKET_CLOSE


def run_day(cfg: DayTradingConfig, day: date, bars_by_symbol: Dict[str, List[Bar]],
            prev_close: Optional[Dict[str, float]] = None, playbook: Optional[Playbook] = None,
            cost: Optional[CostModel] = None, market: Optional[Dict[str, float]] = None) -> DayResult:
    close = _close_time_for(bars_by_symbol)
    engine = DayTradingEngine(cfg, day, prev_close or {}, playbook, close_time=close, market=market)
    sim = SimBroker(cost or cfg.costs)
    by_ts: Dict[datetime, List[Bar]] = defaultdict(list)
    for bars in bars_by_symbol.values():
        for b in bars:
            by_ts[b.ts].append(b)
    last_close: Dict[str, float] = {}

    def settle(fills, rejects):
        for f in fills:
            engine.apply_fill(f)
        for sym, why in rejects:
            engine.apply_reject(sym, why)

    minutes = trading_minutes(day, close)
    for ts in minutes:
        bars = by_ts.get(ts, [])
        for b in sorted(bars, key=lambda x: x.symbol):
            settle(*sim.on_bar(b))                       # fills from LAST minute's decisions first
            last_close[b.symbol] = b.close
        for intent in engine.on_bars(bars, ts + timedelta(minutes=1)):
            for sym in sim.submit(intent):
                engine.apply_reject(sym, "cancelled at flatten/halt")

    end = minutes[-1] + timedelta(minutes=1)
    for sym in list(sim.pending_entries):
        engine.apply_reject(sym, "session ended before fill")
    sim.pending_entries.clear()
    for f in sim.force_close_all(last_close, end):
        engine.apply_fill(f)

    opens = {s: bs[0].open for s, bs in bars_by_symbol.items() if bs}
    closes = {s: bs[-1].close for s, bs in bars_by_symbol.items() if bs}
    rets = [closes[s] / opens[s] - 1 for s in opens if opens[s] > 0]
    return DayResult(day, list(engine.trades), sum(t.pnl for t in engine.trades), engine.state.halted,
                     engine.state.halt_reason, engine.state.entries_today, sim.capacity_rejects,
                     sim.max_participation, statistics.median(rets) if rets else 0.0, engine.journal_records())


def _prev_closes(by_day: Dict[str, Dict[date, List[Bar]]], all_days: List[date]) -> Dict[date, Dict[str, float]]:
    out: Dict[date, Dict[str, float]] = {}
    for sym, days in by_day.items():
        ordered = sorted(days)
        for i, d in enumerate(ordered):
            if i == 0:
                continue
            prev = ordered[i - 1]
            if (d - prev).days <= 5:
                out.setdefault(d, {})[sym] = days[prev][-1].close
    return out


def market_by_day(spy_bars: Sequence[Bar]) -> Dict[date, Dict[str, float]]:
    """Per-day SPY read from minute bars, known before that day's open —
    the same two numbers the live planner feeds the engine."""
    closes = [(d, bs[-1].close) for d, bs in sorted(group_by_day(spy_bars).items()) if bs]
    out: Dict[date, Dict[str, float]] = {}
    for i in range(1, len(closes)):
        d = closes[i][0]
        m = {"spy_prev_change": closes[i - 1][1] / closes[i - 2][1] - 1.0} if i >= 2 else {}
        if i >= 6:
            m["spy_mom_5d"] = closes[i - 1][1] / closes[i - 6][1] - 1.0
        out[d] = m
    return out


def run_backtest(data: Dict[str, List[Bar]], cfg: DayTradingConfig, cost: Optional[CostModel] = None,
                 playbooks: Optional[Dict[date, Playbook]] = None, days: Optional[Sequence[date]] = None,
                 label: str = "", markets: Optional[Dict[date, Dict[str, float]]] = None,
                 lessons: Optional[List[Dict[str, Any]]] = None) -> BacktestResult:
    """`lessons` apply to every day that has no explicit playbook."""
    by_day = {sym: group_by_day(bars) for sym, bars in data.items()}
    all_days = sorted({d for dd in by_day.values() for d in dd})
    if days is not None:
        wanted = set(days)
        all_days = [d for d in all_days if d in wanted]
    prev = _prev_closes(by_day, all_days)
    results = []
    for d in all_days:
        todays = {s: dd[d] for s, dd in by_day.items() if d in dd}
        if not todays:
            continue
        pb = (playbooks or {}).get(d) or (Playbook(lessons=lessons) if lessons else None)
        results.append(run_day(cfg, d, todays, prev.get(d), pb, cost, (markets or {}).get(d)))
    return BacktestResult(results, cfg.sleeve_capital, label)


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def _mean(x: Sequence[float]) -> float:
    return sum(x) / len(x) if x else 0.0


def _std(x: Sequence[float]) -> float:
    return statistics.stdev(x) if len(x) > 1 else 0.0


def sharpe(daily: Sequence[float]) -> float:
    s = _std(daily)
    return _mean(daily) / s * math.sqrt(TRADING_DAYS) if s > 0 else 0.0


def max_drawdown(daily: Sequence[float]) -> float:
    """Peak-to-trough of the cumulative return, as a fraction of the sleeve
    (fixed base: the sleeve is not compounded day to day)."""
    peak = cum = worst = 0.0
    for r in daily:
        cum += r
        peak = max(peak, cum)
        worst = max(worst, peak - cum)
    return worst


def cvar(daily: Sequence[float], level: float = 0.95) -> float:
    if not daily:
        return 0.0
    k = max(1, math.ceil((1 - level) * len(daily)))
    return max(0.0, -_mean(sorted(daily)[:k]))


def strategy_metrics(daily: Sequence[float]) -> StrategyMetrics:
    return StrategyMetrics(net_sharpe=sharpe(daily), net_return=_mean(daily) * TRADING_DAYS,
                           max_drawdown=max_drawdown(daily), cvar_95=cvar(daily),
                           volatility=_std(daily) * math.sqrt(TRADING_DAYS), leverage=1.0)


def deflated_sharpe(daily: Sequence[float], n_trials: int) -> float:
    """Bailey & Lopez de Prado's deflation, reported as ANNUALIZED (SR - SR0),
    so `> 0` means "beat the best Sharpe you'd expect from n_trials pure-noise
    strategies" — the form promotion_gate's `deflated_sharpe > 0` gate wants.
    V[SR] is approximated by the single-strategy sampling variance 1/(T-1)
    instead of the cross-trial variance (which needs every trial's returns)."""
    t = len(daily)
    if t < 10 or _std(daily) == 0:
        return 0.0
    sr = _mean(daily) / _std(daily)
    sr0 = 0.0
    if n_trials > 1:
        nd, g = NormalDist(), 0.5772156649015329
        sr0 = math.sqrt(1.0 / (t - 1)) * ((1 - g) * nd.inv_cdf(1 - 1 / n_trials)
                                           + g * nd.inv_cdf(1 - 1 / (n_trials * math.e)))
    return (sr - sr0) * math.sqrt(TRADING_DAYS)


def bootstrap_p_value(challenger: Sequence[float], champion: Sequence[float], n: int = 2000, seed: int = 0) -> float:
    """One-sided paired bootstrap on daily return differences: P(mean diff <= 0)."""
    diffs = [a - b for a, b in zip(challenger, champion)]
    if len(diffs) < 5:
        return 1.0
    rng = random.Random(seed)
    m = len(diffs)
    not_better = sum(1 for _ in range(n) if sum(diffs[rng.randrange(m)] for _ in range(m)) <= 0)
    return not_better / n


def bootstrap_run_returns(daily: Sequence[float], n: int = 200, seed: int = 1) -> List[float]:
    """Annualized return under n resamples of the daily series — the
    'distribution of runs' promotion_gate expects (it was written for
    stochastic agents; a deterministic rule set has only resampling noise)."""
    if not daily:
        return []
    rng, m = random.Random(seed), len(daily)
    return [sum(daily[rng.randrange(m)] for _ in range(m)) / m * TRADING_DAYS for _ in range(n)]


def fold_returns(daily: Sequence[float], k: int = 5) -> List[float]:
    if len(daily) < k:
        return [_mean(daily) * TRADING_DAYS] if daily else []
    size = len(daily) // k
    return [_mean(daily[i * size:(i + 1) * size if i < k - 1 else len(daily)]) * TRADING_DAYS for i in range(k)]


def regime_returns(result: BacktestResult, thresh: float = 0.003) -> Dict[str, float]:
    """Annualized mean daily return within up / flat / down market days, where
    a day's regime is the universe's median open->close move."""
    buckets: Dict[str, List[float]] = {"up": [], "flat": [], "down": []}
    for d in result.days:
        k = "up" if d.universe_return > thresh else "down" if d.universe_return < -thresh else "flat"
        buckets[k].append(d.pnl / result.sleeve)
    return {k: _mean(v) * TRADING_DAYS for k, v in buckets.items() if v}


# -----------------------------------------------------------------------------
# Null-model champion
# -----------------------------------------------------------------------------
def null_baseline(data: Dict[str, List[Bar]], cfg: DayTradingConfig, target_trades: int,
                  cost: Optional[CostModel] = None, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> BacktestResult:
    """Random entries, same stops/targets/risk/flatten, calibrated so the
    EXPECTED number of entries matches `target_trades`. Averaged across seeds
    day by day, so the champion is a smooth reference rather than one lucky draw."""
    sym_days = sum(len(group_by_day(b)) for b in data.values())
    window = DEFAULTS["random_long"]["to_min"] - DEFAULTS["random_long"]["from_min"]
    per_sym_day = min(0.9, max(1e-6, target_trades / max(sym_days, 1)))
    p = 1.0 - (1.0 - per_sym_day) ** (1.0 / window)
    runs = []
    for seed in seeds:
        c = dataclasses.replace(cfg, strategies=["random_long"],
                                params={"random_long": {"p_per_bar": p, "seed": seed}})
        runs.append(run_backtest(data, c, cost, label=f"null[{seed}]"))
    n_days = min(len(r.days) for r in runs)
    avg_days = []
    for i in range(n_days):
        ds = [r.days[i] for r in runs]
        avg_days.append(dataclasses.replace(ds[0], pnl=_mean([d.pnl for d in ds]), trades=[t for d in ds for t in d.trades],
                                            entries=int(round(_mean([d.entries for d in ds])))))
    return BacktestResult(avg_days, cfg.sleeve_capital, "null-baseline (mean of seeds)")


# -----------------------------------------------------------------------------
# Bridge to promotion_gate
# -----------------------------------------------------------------------------
def perturbation_degradation(rerun: Callable[[Dict[str, Dict[str, float]]], BacktestResult],
                             cfg: DayTradingConfig, base: BacktestResult, pct: float) -> float:
    """Joint +/-pct perturbation of EVERY numeric parameter of the enabled
    strategies (all up, all down, and two alternating patterns = 4 corners).
    Returns the worst fractional drop in net return vs. base, floored at 0.
    Cheaper than a full grid and thus weaker: it can miss a single fragile
    parameter. Bar counts are rounded, so a +/-10% on a 15-minute range is
    a 13-17 minute range."""
    base_ret = _mean(base.daily_returns) * TRADING_DAYS
    names = [(s, k) for s in cfg.strategies for k in DEFAULTS[s] if k not in ("seed",)]
    worst = 0.0
    for pattern in range(4):
        params: Dict[str, Dict[str, float]] = {}
        for i, (s, k) in enumerate(names):
            merged = params.setdefault(s, {})
            cur = (cfg.params.get(s) or {}).get(k, DEFAULTS[s][k])
            sign = (1, -1, 1 if i % 2 == 0 else -1, -1 if i % 2 == 0 else 1)[pattern]
            merged[k] = cur * (1 + sign * pct)
        r = _mean(rerun(params).daily_returns) * TRADING_DAYS
        drop = (base_ret - r) / abs(base_ret) if base_ret > 0 else (1.0 if r < base_ret else 0.0)
        worst = max(worst, drop)
    return max(0.0, worst)


def build_challenger_report(challenger: BacktestResult, champion: BacktestResult, n_trials: int,
                            perturbation: float, shadow_return: Optional[float] = None,
                            seed: int = 0) -> ChallengerReport:
    ch, cp = challenger.daily_returns, champion.daily_returns
    n = min(len(ch), len(cp))
    ch, cp = ch[:n], cp[:n]
    cap_rejects = sum(d.capacity_rejects for d in challenger.days)
    attempted = max(1, sum(d.entries for d in challenger.days))
    sim_return = _mean(ch) * TRADING_DAYS
    return ChallengerReport(
        metrics=strategy_metrics(ch),
        n_trades=len(challenger.trades),
        p_value=bootstrap_p_value(ch, cp, seed=seed),
        deflated_sharpe=deflated_sharpe(ch, n_trials),
        run_returns=bootstrap_run_returns(ch, seed=seed + 1),
        regime_returns=regime_returns(challenger),
        regime_returns_champion=regime_returns(champion),
        walkforward_fold_returns=fold_returns(ch),
        perturbation_degradation=perturbation,
        capacity_ok=cap_rejects / attempted <= 0.05,
        # NaN on purpose until real paper results are supplied — see module docstring.
        shadow_return=float("nan") if shadow_return is None else shadow_return,
        sim_return=sim_return,
        costs_are_net=True,
    )


def per_strategy_table(result: BacktestResult) -> List[Dict[str, Any]]:
    rows: Dict[str, List[Trade]] = defaultdict(list)
    for t in result.trades:
        rows[t.strategy].append(t)
    out = []
    for name, ts in sorted(rows.items()):
        wins = [t.pnl for t in ts if t.pnl > 0]
        losses = [-t.pnl for t in ts if t.pnl <= 0]
        out.append({"strategy": name, "trades": len(ts), "net_pnl": sum(t.pnl for t in ts),
                    "win_rate": len(wins) / len(ts), "avg_r": _mean([t.r_multiple for t in ts]),
                    "profit_factor": (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else float("inf")})
    return out


def evaluate(challenger: BacktestResult, champion: BacktestResult, criteria: Dict[str, Any],
             n_trials: int = 1, rerun: Optional[Callable[[Dict[str, Dict[str, float]]], BacktestResult]] = None,
             cfg: Optional[DayTradingConfig] = None, perturb_pct: Optional[float] = None,
             shadow_return: Optional[float] = None) -> Tuple[StrategyMetrics, ChallengerReport, PromotionVerdict]:
    pct = perturb_pct if perturb_pct is not None else criteria["robustness_gates"]["param_perturbation_pct"]
    pert = perturbation_degradation(rerun, cfg, challenger, pct) if rerun and cfg else 1.0   # unmeasured => fail closed
    report = build_challenger_report(challenger, champion, n_trials, pert, shadow_return)
    aligned = min(len(challenger.days), len(champion.days))      # same slice build_challenger_report used
    champ_metrics = strategy_metrics(champion.daily_returns[:aligned])
    return champ_metrics, report, evaluate_promotion(champ_metrics, report, criteria)


def format_report(challenger: BacktestResult, verdict: PromotionVerdict, metrics: StrategyMetrics,
                  champion_metrics: StrategyMetrics) -> str:
    lines = [f"days={len(challenger.days)}  trades={len(challenger.trades)}  sleeve=${challenger.sleeve:,.0f}",
             f"net Sharpe {metrics.net_sharpe:+.2f} (null {champion_metrics.net_sharpe:+.2f})   "
             f"ann. return {metrics.net_return:+.1%} (null {champion_metrics.net_return:+.1%})   "
             f"maxDD {metrics.max_drawdown:.1%}   CVaR95 {metrics.cvar_95:.2%}", ""]
    for row in per_strategy_table(challenger):
        pf = "inf" if math.isinf(row["profit_factor"]) else f"{row['profit_factor']:.2f}"
        lines.append(f"  {row['strategy']:<20} trades={row['trades']:<4} net=${row['net_pnl']:>+9,.0f} "
                     f"win={row['win_rate']:.0%} avgR={row['avg_r']:+.2f} PF={pf}")
    lines += ["", f"VERDICT: {verdict.decision.value.upper()}", "  " + verdict.rationale()]
    for g in verdict.gates:
        lines.append(f"   [{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.detail}")
    return "\n".join(lines)


def go_no_go(verdict: PromotionVerdict) -> Tuple[bool, str]:
    """The single decision this harness exists to make. `ops.shadow_vs_sim`
    cannot pass before paper trading has produced results, so it is the ONE
    gate allowed to be outstanding: everything else passing means "worth
    paper-trading", anything else failing means "stop here"."""
    blocking = [g for g in verdict.failures if g.name != "ops.shadow_vs_sim"]
    if blocking:
        return False, ("NO-GO — do not build or run anything live. Failing: " +
                       "; ".join(f"{g.name} ({g.detail})" for g in blocking))
    return True, ("PROCEED TO PAPER TRADING — every backtest gate passed; only ops.shadow_vs_sim is outstanding, "
                  "which needs real paper results (Phase 5) before any promotion.")


def _fetch_minute_bars(symbols: Sequence[str], start: date, end: date) -> Dict[str, List[Bar]]:
    from alpaca.data.historical import StockHistoricalDataClient
    from daytrade_broker import load_credentials
    from intraday_data import AlpacaIntradayDataClient, UTC
    try:
        key, secret = load_credentials("daytrade")
    except Exception:
        key, secret = load_credentials("main")
    client = AlpacaIntradayDataClient(StockHistoricalDataClient(key, secret))
    out: Dict[str, List[Bar]] = {}
    lo = datetime.combine(start, time(0), tzinfo=UTC)
    hi = datetime.combine(end, time(23, 59), tzinfo=UTC)
    for i in range(0, len(symbols), 10):
        chunk = list(symbols[i:i + 10])
        print(f"  fetching {chunk[0]}..{chunk[-1]} ({i + len(chunk)}/{len(symbols)})", flush=True)
        out.update(client.get_minute_bars(chunk, lo, hi))
    return out


def save_journal(result: BacktestResult, path: str = "", to_store: bool = False, store: Any = None) -> int:
    """Write the learning journal to a .jsonl file and/or the store. Returns records written."""
    n = sum(len(d.journal) for d in result.days)
    if path:
        import json
        with open(path, "w") as fh:
            for d in result.days:
                for r in d.journal:
                    fh.write(json.dumps({**r, "source": "backtest"}) + "\n")
        print(f"  wrote {n} journal records to {path}")
    if to_store:
        if store is None:
            from daytrade_store import default_store
            store = default_store()
        for d in result.days:
            store.save_journal(d.day, d.journal, "backtest")
        print(f"  stored {n} journal records ({len(result.days)} days) for lesson validation")
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python daytrade_backtest.py run",
        description="Backtest the day-trading strategies on real IEX minute bars and decide GO / NO-GO.")
    ap.add_argument("--symbols", required=True, help="comma-separated tickers, or 'universe:N' for the first N of universe.py")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--config", default="mandate.yaml")
    ap.add_argument("--criteria", default="promotion_criteria.yaml")
    ap.add_argument("--trials", type=int, default=1,
                    help="HOW MANY strategy/parameter variants you have tried in total — be honest, the Deflated "
                         "Sharpe punishes hiding failed attempts")
    ap.add_argument("--no-perturb", action="store_true", help="skip the (slow) parameter-robustness reruns; the gate then FAILS closed")
    ap.add_argument("--save-journal", action="store_true",
                    help="store the learning journal in Firestore (source 'backtest') for lesson validation")
    ap.add_argument("--journal-out", default="", help="also write the learning journal to this .jsonl file")
    ap.add_argument("--lessons", choices=("none", "active"), default="none",
                    help="'active' replays the currently active learned lessons (from the store) on every day")
    a = ap.parse_args(argv)

    from daytrade_config import load_day_trading_config
    from promotion_gate import load_criteria
    cfg = load_day_trading_config(a.config)
    criteria = load_criteria(a.criteria)
    if a.symbols.startswith("universe:"):
        from universe import EQUITIES
        symbols = [e[0] for e in EQUITIES][: int(a.symbols.split(":")[1])]
    else:
        symbols = [x.strip().upper() for x in a.symbols.split(",") if x.strip()]
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)

    print(f"Fetching minute bars for {len(symbols)} symbols, {start} -> {end} (IEX feed)...")
    data = _fetch_minute_bars(symbols, start, end)
    got = sum(len(v) for v in data.values())
    if not got:
        print("No bars returned — check credentials, symbols and dates.")
        return 2
    markets: Dict[date, Dict[str, float]] = {}
    try:
        spy = _fetch_minute_bars(["SPY"], start - timedelta(days=10), end).get("SPY", [])
        markets = market_by_day(spy)
    except Exception as e:
        print(f"  (no SPY context for entry snapshots: {e})")
    lessons = None
    if a.lessons == "active":
        from daytrade_learning import active_lesson_dicts
        from daytrade_store import default_store
        lessons = active_lesson_dicts(default_store())
        print(f"Replaying {len(lessons)} active lesson(s).")
    print(f"{got:,} bars. Running strategies...")
    result = run_backtest(data, cfg, label="strategies", markets=markets, lessons=lessons)
    save_journal(result, a.journal_out, a.save_journal)
    print(f"{len(result.trades)} trades over {len(result.days)} days. Building null-model champion...")
    champion = null_baseline(data, cfg, target_trades=max(len(result.trades), 10))
    rerun = None if a.no_perturb else (lambda p: run_backtest(data, dataclasses.replace(cfg, params=p)))
    cm, report, verdict = evaluate(result, champion, criteria, n_trials=a.trials, rerun=rerun, cfg=cfg)
    print("\n" + format_report(result, verdict, report.metrics, cm))
    go, msg = go_no_go(verdict)
    print("\n" + ("GO: " if go else "") + msg)
    return 0 if go else 1


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "run":
        raise SystemExit(main(_sys.argv[2:]))
    import math as _m
    import zlib
    from intraday_data import et_datetime, synthetic_day
    from daytrade_strategies import _flat_open, _path
    UTC = __import__("datetime").timezone.utc
    cost0 = CostModel(slippage_bps=0.0, sec_fee_per_million=0.0, taf_per_share=0.0)
    cfg = config_from_dict({"sleeve_capital": 25_000})
    day = date(2026, 3, 4)

    def sym_bars(sym, rows):
        return [dataclasses.replace(b, symbol=sym) for b in _path(day, rows)]

    def entry_only_setup(after):
        """ORB breakout at minute 17 (signal on that bar's close), then `after` rows."""
        base = _flat_open(15) + [(100.0, 100.2, 99.9, 100.1, 1000)] * 2 + [(100.15, 100.7, 100.1, 100.6, 4000)]
        return base + list(after)

    # signal bar closes 100.6 -> engine: stop 100.0, target 101.8, qty 49; fill at NEXT open
    print("TEST 1: entry fills at the NEXT bar's open (+slippage), never at the signal price")
    cost_s = CostModel(slippage_bps=10.0, sec_fee_per_million=0.0, taf_per_share=0.0)
    rows = entry_only_setup([(100.62, 100.8, 100.5, 100.7, 3000)] * 3)
    res = run_day(cfg, day, {"AAA": sym_bars("AAA", rows)}, cost=cost_s)
    assert res.entries == 1 and len(res.trades) == 1
    assert abs(res.trades[0].entry_price - 100.62 * 1.001) < 1e-9, res.trades[0].entry_price
    assert res.trades[0].entry_price != 100.60, "the signal bar closed at 100.60 — must not fill there"
    print(f"✓ signal closed 100.60; filled {res.trades[0].entry_price:.4f} = next open 100.62 + 10bps slippage")

    print("TEST 1b: early close is detected only at ~13:00; truncated data is NOT an early close")
    from intraday_data import synthetic_day as _sd
    half = {"AAA": _sd("AAA", day, seed=5, close=time(13, 0))}
    assert _close_time_for(half) == time(13, 0)
    assert _close_time_for({"AAA": sym_bars("AAA", entry_only_setup([]))}) == MARKET_CLOSE     # 20 bars, ends 09:50
    assert _close_time_for({"AAA": _sd("AAA", day, seed=5)}) == MARKET_CLOSE
    assert _close_time_for({}) == MARKET_CLOSE
    print("✓ 210-bar day -> 13:00; 20-bar fragment and full day -> 16:00")

    print("TEST 2: target only fills when price trades THROUGH it; touching is not enough")
    touch = entry_only_setup([(100.62, 100.8, 100.5, 100.7, 3000), (100.7, 101.8, 100.6, 101.0, 3000)])
    thru = entry_only_setup([(100.62, 100.8, 100.5, 100.7, 3000), (100.7, 101.85, 100.6, 101.0, 3000)])
    r_touch = run_day(cfg, day, {"AAA": sym_bars("AAA", touch)}, cost=cost0)
    r_thru = run_day(cfg, day, {"AAA": sym_bars("AAA", thru)}, cost=cost0)
    assert r_touch.trades[0].reason != "target", "high == target exactly must NOT count as a fill"
    t = r_thru.trades[0]
    assert t.reason == "target" and t.exit_price == 101.8
    assert abs(t.entry_price - 100.62) < 1e-9, "fill at next bar's OPEN, not the 100.60 signal close"
    assert abs(t.pnl - (101.8 - 100.62) * t.qty) < 1e-9
    print(f"✓ touch -> {r_touch.trades[0].reason}; through -> target @ {t.exit_price}, entry {t.entry_price} (next open)")

    print("TEST 3: stop wins when one bar hits both stop and target (conservative)")
    both = entry_only_setup([(100.62, 100.8, 100.5, 100.7, 3000), (100.7, 102.0, 99.5, 100.9, 3000)])
    rb = run_day(cfg, day, {"AAA": sym_bars("AAA", both)}, cost=cost0)
    assert rb.trades[0].reason == "stop" and rb.trades[0].exit_price == 100.0 and rb.trades[0].pnl < 0
    print("✓ stop @ 100.0 assumed first")

    print("TEST 4: a gap THROUGH the stop fills at the (worse) open, not at the stop")
    gap = entry_only_setup([(100.62, 100.8, 100.5, 100.7, 3000), (99.0, 99.2, 98.8, 99.1, 3000)])
    rg = run_day(cfg, day, {"AAA": sym_bars("AAA", gap)}, cost=cost0)
    assert rg.trades[0].reason == "stop" and rg.trades[0].exit_price == 99.0
    print("✓ exit 99.0 (gap open), well below the 100.0 stop")

    print("TEST 5: slippage and fees make every round trip cost money")
    thru_cost = run_day(cfg, day, {"AAA": sym_bars("AAA", thru)}, cost=CostModel())
    assert thru_cost.trades[0].pnl < r_thru.trades[0].pnl
    assert thru_cost.trades[0].fees > 0 and thru_cost.trades[0].entry_price > 100.62
    print(f"✓ net ${thru_cost.trades[0].pnl:.2f} vs ${r_thru.trades[0].pnl:.2f} frictionless")

    print("TEST 6: end-of-day flatten — nothing is held overnight")
    long_day = entry_only_setup([(100.62, 100.9, 100.4, 100.7, 3000)] * 370)     # goes quiet until close
    rl = run_day(cfg, day, {"AAA": sym_bars("AAA", long_day)}, cost=cost0)
    assert len(rl.trades) == 1 and rl.trades[0].reason == "eod_flatten"
    assert to_et(rl.trades[0].exit_ts).time() <= time(15, 50), rl.trades[0].exit_ts
    print(f"✓ closed at {to_et(rl.trades[0].exit_ts):%H:%M} ET (eod_flatten)")

    print("TEST 7: participation cap -> partial fill; hopeless illiquidity -> capacity reject")
    thin = entry_only_setup([(100.62, 100.8, 100.5, 100.7, 100)] * 3)             # 10% of 100 sh = 10 sh
    rt = run_day(cfg, day, {"AAA": sym_bars("AAA", thin)}, cost=cost0)
    assert rt.trades and rt.trades[0].qty == 10
    dust = entry_only_setup([(100.62, 100.8, 100.5, 100.7, 5)] * 3)               # 10% of 5 sh = 0
    rd = run_day(cfg, day, {"AAA": sym_bars("AAA", dust)}, cost=cost0)
    assert not rd.trades and rd.capacity_rejects == 1
    print("✓ 100-share bar -> 10 filled; 5-share bar -> capacity reject (counted)")

    print("TEST 8: metrics against hand-computed values")
    d = [0.01, -0.02, 0.015, -0.005, 0.02]
    m = strategy_metrics(d)
    assert abs(m.net_return - _mean(d) * 252) < 1e-12
    assert abs(max_drawdown(d) - 0.02) < 1e-12                      # peak .01 -> trough -.01
    assert abs(cvar(d, 0.95) - 0.02) < 1e-12                         # worst 1 of 5
    assert abs(m.net_sharpe - (_mean(d) / statistics.stdev(d)) * _m.sqrt(252)) < 1e-9
    print(f"✓ Sharpe {m.net_sharpe:.2f}, MDD {max_drawdown(d):.1%}, CVaR95 {cvar(d):.1%}")

    print("TEST 9: Deflated Sharpe punishes trying many strategies")
    rng = random.Random(3)
    sample = [rng.gauss(0.0008, 0.01) for _ in range(250)]
    d1, d20, d500 = (deflated_sharpe(sample, n) for n in (1, 20, 500))
    assert d1 > d20 > d500 and deflated_sharpe(sample[:5], 10) == 0.0
    print(f"✓ N=1: {d1:+.2f}   N=20: {d20:+.2f}   N=500: {d500:+.2f}   (tiny samples -> 0.0)")

    print("TEST 10: full pipeline on synthetic RANDOM-WALK data (no edge exists) — invariants hold and the gate rejects")
    syms = [f"S{i:02d}" for i in range(8)]
    days = [date(2026, 3, 2) + timedelta(days=i) for i in range(0, 40) if (date(2026, 3, 2) + timedelta(days=i)).weekday() < 5]
    data = {s: [b for d in days for b in synthetic_day(s, d, seed=11, start_price=50 + 7 * i,
                                                       vol_bps_per_min=9.0, gap=((zlib.crc32(f'{s}{d}'.encode()) % 7) - 3) / 100)]
            for i, s in enumerate(syms)}
    cfg2 = config_from_dict({"sleeve_capital": 25_000})
    result = run_backtest(data, cfg2, label="strategies")
    assert len(result.days) == len(days)
    # invariants that must hold on ANY data
    for dr in result.days:
        assert abs(dr.pnl - sum(t.pnl for t in dr.trades)) < 1e-6
        assert all(to_et(t.exit_ts).time() <= time(16, 0) for t in dr.trades)
        assert all(t.exit_ts >= t.entry_ts for t in dr.trades)
        assert all(to_et(t.entry_ts).date() == to_et(t.exit_ts).date() for t in dr.trades), "nothing held overnight"
    n_tr = len(result.trades)
    champ = null_baseline(data, cfg2, target_trades=max(n_tr, 10))
    print(f"  strategies: {n_tr} trades, ann.return {_mean(result.daily_returns)*252:+.1%};  "
          f"null: {len(champ.trades)} trades, ann.return {_mean(champ.daily_returns)*252:+.1%}")
    criteria = {
        "edge_gates": {"min_sharpe_margin": 0.30, "require_return_ge_champion": True},
        "risk_gates": {"max_drawdown_rel_tolerance": 0.10, "require_cvar_le_champion": True,
                       "max_leverage": 2.0, "leverage_not_above_champion": True},
        "statistical_gates": {"min_trades": 150, "max_p_value": 0.05, "require_positive_deflated_sharpe": True,
                              "distribution": {"require_median_beats_champion": True, "low_percentile": 10,
                                               "low_percentile_floor_vs_champion": 0.90}},
        "robustness_gates": {"regime_worse_tolerance": 0.10, "min_walkforward_win_rate": 0.50,
                             "param_perturbation_pct": 0.10, "max_perturbation_degradation": 0.25},
        "cost_operational_gates": {"costs_already_net": True, "require_capacity_ok": True,
                                   "shadow_vs_sim_tolerance": 0.15},
        "governance": {"auto_promote_requires": {"leverage_unchanged": True, "risk_limits_unchanged": True,
                                                 "capital_rules_unchanged": True}},
    }
    rerun = lambda p: run_backtest(data, dataclasses.replace(cfg2, params=p))
    cm, rep, verdict = evaluate(result, champ, criteria, n_trials=3, rerun=rerun, cfg=cfg2)
    assert verdict.decision.value == "reject"
    assert any(g.name == "ops.shadow_vs_sim" and not g.passed for g in verdict.gates)
    print(f"✓ REJECTED with {len(verdict.failures)} failing gates, including ops.shadow_vs_sim (no paper data yet)")

    print("TEST 11: the plumbing can pass — an injected genuine edge clears every gate except shadow")
    def fake_result(mean, n=300, seed=1, label=""):
        r = random.Random(seed)
        ds = []
        for i in range(n):
            pnl = r.gauss(mean, 0.006) * 25_000
            tr = [Trade("X", "orb_long", 10, 100, 101, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC), pnl, 0, "t", 1.0)]
            ds.append(DayResult(date(2026, 1, 1) + timedelta(days=i), tr, pnl, False, "", 1, 0, 0.01, r.gauss(0, 0.01)))
        return BacktestResult(ds, 25_000, label)
    good, null = fake_result(0.004, seed=1), fake_result(0.0002, seed=2)
    rep2 = build_challenger_report(good, null, n_trials=3, perturbation=0.05)
    cm2 = strategy_metrics(null.daily_returns)
    v2 = evaluate_promotion(cm2, rep2, criteria)
    failed = {g.name for g in v2.failures}
    print(f"  failing gates: {sorted(failed)}")
    assert failed == {"ops.shadow_vs_sim"} or failed <= {"ops.shadow_vs_sim", "stat.min_trades"}, failed
    rep3 = build_challenger_report(good, null, n_trials=3, perturbation=0.05, shadow_return=rep2.sim_return * 0.95)
    assert evaluate_promotion(cm2, rep3, criteria).decision.value in ("promote_auto", "promote_needs_signoff")
    print("✓ with real-looking shadow results the same report PROMOTES; without, it cannot")

    print("TEST 12: perturbation that is never measured fails closed (1.0 > 25% cap)")
    _, rep4, v4 = evaluate(good, null, criteria, n_trials=3)   # no rerun supplied
    assert rep4.perturbation_degradation == 1.0 and any(g.name == "robust.param_stability" and not g.passed for g in v4.gates)
    print("✓ unmeasured robustness = failed robustness")
    print("TEST 13: go/no-go — shadow may be the ONLY outstanding gate; anything else failing is a NO-GO")
    go, msg = go_no_go(v2)          # TEST 11's strong-edge report: only ops.shadow_vs_sim fails
    assert go and msg.startswith("PROCEED TO PAPER TRADING") and "ops.shadow_vs_sim" in msg
    nogo, msg2 = go_no_go(verdict)  # TEST 10's random-walk report: many gates fail
    assert not nogo and msg2.startswith("NO-GO") and "stat" in msg2 or "edge" in msg2
    assert "ops.shadow_vs_sim" not in msg2.split("Failing:")[1], "shadow is excused, so it must not be listed as blocking"
    print("✓ strong edge -> PROCEED TO PAPER; random walk -> NO-GO, with the blocking gates named")

    print("TEST 14: the learning journal — one record per first signal, virtual outcome tracks the simulated fill")
    journal = [r for d in result.days for r in d.journal]
    taken = [r for r in journal if r["status"] == "taken" and r.get("r_real") is not None]
    assert len(taken) == len(result.trades), (len(taken), len(result.trades))
    same_exit = sum(1 for r in taken if r["exit_virtual"] == r["exit_reason"]) / len(taken)
    same_sign = sum(1 for r in taken if (r["r_virtual"] > 0) == (r["r_real"] > 0)) / len(taken)
    assert same_exit > 0.9 and same_sign > 0.9, (same_exit, same_sign)
    assert all(r["snapshot"]["strategy"] == r["strategy"] for r in journal)
    ghosts = [r for r in journal if r["status"] != "taken"]
    print(f"✓ {len(journal)} records ({len(taken)} taken, {len(ghosts)} ghosts); virtual vs real: "
          f"{same_exit:.0%} same exit, {same_sign:.0%} same sign")

    print("TEST 15: SPY context is known BEFORE each day (no look-ahead); journal saves to store and .jsonl")
    spy = [b for d in days for b in synthetic_day("SPY", d, seed=4, start_price=500)]
    mk = market_by_day(spy)
    closes = {d: [b for b in spy if to_et(b.ts).date() == d][-1].close for d in days}
    d2 = days[2]
    assert abs(mk[d2]["spy_prev_change"] - (closes[days[1]] / closes[days[0]] - 1)) < 1e-12
    assert days[0] not in mk and "spy_mom_5d" not in mk[days[2]] and "spy_mom_5d" in mk[days[6]]
    with_mk = run_backtest(data, cfg2, days=days[:8], markets=mk)
    snaps = [(d.day, r["snapshot"]) for d in with_mk.days for r in d.journal]
    assert all((sn["spy_prev_change"] is not None) == (d >= days[2]) for d, sn in snaps), \
        "known from day 3 on (needs two prior closes), and never invented before that"
    import json, os, tempfile
    from daytrade_store import InMemoryDayTradeStore
    st = InMemoryDayTradeStore()
    path = os.path.join(tempfile.mkdtemp(), "j.jsonl")
    n = save_journal(with_mk, path, to_store=True, store=st)
    assert n == len(open(path).read().splitlines()) == len(st.list_journal(source="backtest")) > 0
    assert json.loads(open(path).readline())["source"] == "backtest"
    print(f"✓ {n} records to both; SPY read from the PRIOR close only")

    print("TEST 16: --lessons replay: a skip-everything lesson removes every trade and journals every skip")
    nuke = [{"id": "Lall", "when": {"minutes_since_open": {">=": 0}}, "action": {"type": "skip"}}]
    none = run_backtest(data, cfg2, days=days[:8], lessons=nuke)
    base = run_backtest(data, cfg2, days=days[:8])
    assert base.trades and not none.trades
    skipped = [r for d in none.days for r in d.journal if r["status"] == "filtered_lesson"]
    assert len(skipped) >= len(base.trades) and all(r["lesson_ids"] == ["Lall"] for r in skipped)
    print(f"✓ {len(base.trades)} trades -> 0; {len(skipped)} skips journaled with their outcomes")
    print("\nAll daytrade_backtest tests passed.")
