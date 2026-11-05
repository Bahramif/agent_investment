"""
rulebook.py
===========
THE rule set for long-term portfolios: one module, one config block
(`rulebook:` in mandate.yaml), used unchanged by the live recommendation, the
mid-day news watch, the pre-trade reviewer and the historical simulation.

Three principles, each enforced by test_rulebook.py:

  GENERIC   no rule names a stock, and no threshold is a flat percentage that
            means something different for a calm stock and a wild one: price
            limits are measured in each stock's OWN volatility.
  ALIGNED   every decision reads the same score and the same risk measure, so
            one rule cannot undo another: a pullback only earns credit while
            the long-term trend (the momentum factor) is intact, a position
            bought on a pullback has the same stop as any other, and anything
            sold for any reason cannot be bought back for reentry_block_days.
  SMOOTH    scores change gradually with their inputs. A sudden cutoff (a dip
            score dropping from 0.70 to 0 between -3.9% and -4.1%) turns noise
            into trades.

THE LIFE OF A POSITION
  1. SCORE      within-sector percentile of 12-1 momentum, quality, value and
                (when available) the LLM's read, plus a pullback credit.
  2. BUY        highest-scoring eligible names, within the sub-industry limit.
                Not eligible: under a year of history, too illiquid, broken
                thesis. Not bought now: re-entry block, earnings imminent.
  3. KEEP       a holding stays while it ranks in the top keep_rank_multiple x N,
                so a name slipping from 11th to 13th does not trade.
  4. SIZE       inverse volatility, capped per name. No return forecast.
  5. LIMIT      sub-industry and portfolio-beta limits; any excess stays cash.
  6. RE-RANK    a sector moving unusually (sector_move_sigmas of its own volatility)
                lets the news watch's sector review re-run steps 1-5 early, within
                the early-trade limit, if the move reflects a real change.
  7. DIP BUY    any day, however recently the portfolio recalibrated: a valuable
                stock on a real pullback (rulebook.dip_candidates) is bought with
                cash for an open slot, or replaces the holding that scores lowest.
  8. SELL       rank falls out of the keep band (at a scheduled recalibration);
                or the STOP: below cost AND behind its sector by
                stop_sigma_mult monthly sigmas since purchase; or news the
                model judges thesis-breaking. News and stops sell only the
                affected stock and fill that slot.

Pure and stdlib only (plus PyYAML to read the config), so it runs the same in
Cloud Run, on Agent Engine and in the backtest.
"""

from __future__ import annotations

import math
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
CASH = "CASH"


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RulebookConfig:
    # score: weights of the within-sector percentiles (re-normalised when the LLM has no read)
    weight_momentum: float = 0.35
    weight_quality: float = 0.25
    weight_value: float = 0.20
    weight_llm: float = 0.20
    pullback_weight: float = 0.10       # share of the score a full pullback credit is worth
    # eligibility
    min_history_bars: int = 252         # about a year of trading days
    min_dollar_volume: float = 10_000_000.0
    # holding
    keep_rank_multiple: float = 2.0
    # selling
    stop_sigma_mult: float = 3.0
    reentry_block_days: int = 30
    # a whole sector moving unusually (in its own daily sigmas) lets the news watch consider a re-rank
    sector_move_sigmas: float = 2.0
    # dip buying: how good the pullback must be, and how good the company (its quality rank in its sector)
    dip_min_credit: float = 0.5
    dip_min_quality_pct: float = 0.5
    dip_replace_margin: float = 0.20    # how much better a pick must score to replace a holding (1 = never replace)
    # buying
    earnings_block_days: int = 3
    # portfolio limits
    max_industry_weight: float = 0.35
    max_portfolio_beta: float = 1.5
    # trading
    min_trade_pct: float = 0.005        # a change under this share of the portfolio does not trade
    cost_bps: float = 10.0              # per side; backtests and the scorecard charge this


def config_from_dict(raw: Optional[Dict[str, Any]]) -> RulebookConfig:
    raw = dict(raw or {})
    unknown = set(raw) - set(RulebookConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"rulebook: unknown key(s) {sorted(unknown)}")
    cfg = RulebookConfig(**raw)
    weights = (cfg.weight_momentum, cfg.weight_quality, cfg.weight_value, cfg.weight_llm)
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("rulebook: score weights must be non-negative and not all zero")
    if not 0.0 <= cfg.pullback_weight < 1.0:
        raise ValueError("rulebook.pullback_weight must be in [0, 1)")
    if cfg.keep_rank_multiple < 1.0:
        raise ValueError("rulebook.keep_rank_multiple must be >= 1")
    if cfg.stop_sigma_mult <= 0 or cfg.max_portfolio_beta <= 0 or cfg.sector_move_sigmas <= 0:
        raise ValueError("rulebook: stop_sigma_mult, sector_move_sigmas and max_portfolio_beta must be positive")
    if not 0.0 < cfg.max_industry_weight <= 1.0:
        raise ValueError("rulebook.max_industry_weight must be in (0, 1]")
    if cfg.dip_replace_margin < 0:
        raise ValueError("rulebook.dip_replace_margin must be non-negative")
    if not 0.0 < cfg.dip_min_credit <= 1.0 or not 0.0 <= cfg.dip_min_quality_pct <= 1.0:
        raise ValueError("rulebook: dip_min_credit must be in (0, 1], dip_min_quality_pct in [0, 1]")
    if cfg.reentry_block_days < 0 or cfg.earnings_block_days < 0 or cfg.min_history_bars < 0:
        raise ValueError("rulebook: day and bar counts must be non-negative")
    if not 0.0 <= cfg.min_trade_pct < 1.0 or cfg.cost_bps < 0 or cfg.min_dollar_volume < 0:
        raise ValueError("rulebook: min_trade_pct in [0, 1); cost_bps and min_dollar_volume non-negative")
    return cfg


def to_dict(cfg: RulebookConfig) -> Dict[str, Any]:
    return asdict(cfg)


@lru_cache(maxsize=1)
def default_config() -> RulebookConfig:
    """The `rulebook:` block of the mandate.yaml beside this file (it is bundled
    with the deployed agent too), else the defaults above. A TYPO in the block
    raises; only a missing file or missing PyYAML falls back."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandate.yaml")
    try:
        import yaml
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    except (ImportError, OSError):
        return RulebookConfig()
    return config_from_dict(data.get("rulebook"))


# -----------------------------------------------------------------------------
# Price measures (shared by live ingestion and the backtest)
# -----------------------------------------------------------------------------
def daily_returns(prices: Sequence[float]) -> List[float]:
    return [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices)) if prices[i - 1]]


def momentum_12_1(prices: Sequence[float]) -> Optional[float]:
    """Return from 12 months ago to 1 month ago: the documented momentum effect
    skips the latest month, because returns over about a month tend to reverse.
    None without a full year of history."""
    if len(prices) < TRADING_DAYS_PER_YEAR + 1:
        return None
    start, end = prices[-TRADING_DAYS_PER_YEAR - 1], prices[-TRADING_DAYS_PER_MONTH - 1]
    return end / start - 1.0 if start else None


def annual_window_vol(prices: Sequence[float]) -> Optional[float]:
    """Daily volatility over up to a year. None under a month of data."""
    r = daily_returns(prices[-TRADING_DAYS_PER_YEAR - 1:])
    return statistics.stdev(r) if len(r) >= TRADING_DAYS_PER_MONTH else None


def beta_to(prices: Sequence[float], benchmark: Sequence[float]) -> Optional[float]:
    """Beta over up to a year, aligned on the most recent bars. None under 3 months."""
    n = min(len(prices), len(benchmark), TRADING_DAYS_PER_YEAR + 1)
    if n < 64:
        return None
    a, b = daily_returns(prices[-n:]), daily_returns(benchmark[-n:])
    k = min(len(a), len(b))
    a, b = a[-k:], b[-k:]
    mb = statistics.fmean(b)
    var = sum((x - mb) ** 2 for x in b)
    if var <= 0:
        return None
    ma = statistics.fmean(a)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / var


def drawdown_from_high(prices: Sequence[float], window: int = 126) -> float:
    """Latest price vs. the high of the last `window` bars, <= 0."""
    recent = prices[-window:]
    if not recent or max(recent) <= 0:
        return 0.0
    return min(0.0, recent[-1] / max(recent) - 1.0)


def trailing_return(prices: Sequence[float], bars: int) -> float:
    if len(prices) <= bars or not prices[-bars - 1]:
        return 0.0
    return prices[-1] / prices[-bars - 1] - 1.0


def monthly_sigma(daily_vol: float) -> float:
    return daily_vol * math.sqrt(TRADING_DAYS_PER_MONTH)


def _smoothstep(lo: float, hi: float, x: float) -> float:
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    t = (x - lo) / (hi - lo)
    return t * t * (3.0 - 2.0 * t)


# -----------------------------------------------------------------------------
# Score
# -----------------------------------------------------------------------------
def pullback_credit(momentum_long: Optional[float], drawdown: float, return_5d: float, daily_vol: Optional[float],
                    news_status: str) -> float:
    """0..1. A pullback only earns credit when ALL of these hold, each fading in
    smoothly rather than switching on:
      * the long-term trend is still up (the same 12-1 momentum the score uses),
        so this can never reward what the momentum factor says to avoid;
      * the pullback is meaningful for THIS stock: about 1-3 monthly sigmas below
        its 6-month high (a bigger fall is a broken trend, not a discount);
      * it has stopped falling hard: its 5-day move is not an unusually bad one;
      * the news is CLEAN. 'unknown' earns nothing: no news read, no credit."""
    if news_status != "clean" or not daily_vol or momentum_long is None:
        return 0.0
    trend = _smoothstep(0.0, 0.10, momentum_long)
    depth_sigmas = -min(drawdown, 0.0) / monthly_sigma(daily_vol)
    depth = _smoothstep(0.5, 1.5, depth_sigmas) * (1.0 - _smoothstep(2.5, 4.0, depth_sigmas))
    steady = _smoothstep(-1.5, -0.25, return_5d / (daily_vol * math.sqrt(5)))
    return round(trend * depth * steady, 6)


def percentile_ranks(values: Dict[str, Optional[float]]) -> Dict[str, float]:
    """{name: 0..1} with ties averaged. Missing values sit in the middle: no
    data is neither a reward nor a penalty."""
    known = sorted((v, k) for k, v in values.items() if v is not None)
    out = {k: 0.5 for k, v in values.items() if v is None}
    n = len(known)
    if n == 1:
        out[known[0][1]] = 0.5
        return out
    i = 0
    while i < n:
        j = i
        while j + 1 < n and known[j + 1][0] == known[i][0]:
            j += 1
        pct = ((i + j) / 2.0) / (n - 1)
        for k in range(i, j + 1):
            out[known[k][1]] = pct
        i = j + 1
    return out


def sector_of(path: str) -> str:
    return path.split(".")[-1]


def composite_scores(snapshots: Sequence[Any], llm_pct: Optional[Dict[str, float]] = None,
                     cfg: Optional[RulebookConfig] = None) -> Dict[str, Tuple[float, float]]:
    """{name: (score 0..1, pullback credit 0..1)}. Peers are the stock's own
    sector, so a score means the same thing in every sector and sectors can be
    ranked against each other. `llm_pct` holds the model's read ALREADY ranked
    among the peers it was compared with; a name it has no read for sits in the
    middle. With no LLM at all (the backtest, or a full outage) its weight is
    shared out over the other factors."""
    cfg = cfg or default_config()
    groups: Dict[str, List[Any]] = {}
    for s in snapshots:
        groups.setdefault(sector_of(s.path), []).append(s)
    use_llm = bool(llm_pct) and cfg.weight_llm > 0
    total = cfg.weight_momentum + cfg.weight_quality + cfg.weight_value + (cfg.weight_llm if use_llm else 0.0)
    out: Dict[str, Tuple[float, float]] = {}
    for peers in groups.values():
        mom = percentile_ranks({s.name: s.momentum_12_1 for s in peers})
        qual = percentile_ranks({s.name: s.quality_score for s in peers})
        val = percentile_ranks({s.name: s.earnings_yield for s in peers})
        for s in peers:
            base = cfg.weight_momentum * mom[s.name] + cfg.weight_quality * qual[s.name] + cfg.weight_value * val[s.name]
            if use_llm:
                base += cfg.weight_llm * float((llm_pct or {}).get(s.name, 0.5))
            credit = (pullback_credit(s.momentum_12_1, s.drawdown_6m, s.pullback_5d, s.volatility or None, s.news_status)
                      if cfg.pullback_weight > 0 else 0.0)
            score = (base / total) * (1.0 - cfg.pullback_weight) + cfg.pullback_weight * credit
            out[s.name] = (round(score, 6), credit)
    return out


def eligibility(history_bars: Optional[int], avg_dollar_volume: Optional[float], news_status: str,
                cfg: Optional[RulebookConfig] = None) -> Tuple[bool, str]:
    """(may this stock be held at all?, why not). The same test for every stock.
    Unknown history or liquidity is not held against it (a data outage must not
    empty the portfolio); a known shortfall is."""
    cfg = cfg or default_config()
    if history_bars is not None and history_bars < cfg.min_history_bars:
        return False, f"only {history_bars} trading days of history (needs {cfg.min_history_bars})"
    if avg_dollar_volume is not None and avg_dollar_volume < cfg.min_dollar_volume:
        return False, f"average dollar volume ${avg_dollar_volume:,.0f} below ${cfg.min_dollar_volume:,.0f}"
    if news_status == "thesis_break":
        return False, "the news review judged the thesis broken"
    return True, ""


def earnings_imminent(earnings_date: Optional[str], today: date, cfg: Optional[RulebookConfig] = None) -> bool:
    cfg = cfg or default_config()
    if not earnings_date:
        return False
    try:
        d = date.fromisoformat(str(earnings_date)[:10])
    except ValueError:
        return False
    return 0 <= (d - today).days <= cfg.earnings_block_days


# -----------------------------------------------------------------------------
# Dip buying
# -----------------------------------------------------------------------------
def dip_candidates(snapshots: Sequence[Any], scores: Dict[str, Tuple[float, float]], held: Iterable[str],
                   blocked: Iterable[str], today: date, cfg: Optional[RulebookConfig] = None) -> List[str]:
    """Stocks worth buying on their dip, best score first. VALUABLE: quality in the
    top half of its own sector, and profitable. A REAL DIP: pullback_credit of at
    least dip_min_credit, which already means the long-term trend is up, the fall is
    meaningful for that stock and it has stopped falling hard. Every other rule still
    applies: eligibility, the re-entry block, no buying just before earnings. Whether
    its news is clean is judged separately, because that takes the model."""
    cfg = cfg or default_config()
    held, blocked = set(held), set(blocked)
    groups: Dict[str, List[Any]] = {}
    for s in snapshots:
        groups.setdefault(sector_of(s.path), []).append(s)
    picks: List[str] = []
    for peers in groups.values():
        quality = percentile_ranks({s.name: s.quality_score for s in peers})
        for s in peers:
            credit = pullback_credit(s.momentum_12_1, s.drawdown_6m, s.pullback_5d, s.volatility or None, s.news_status)
            if (s.name not in held and s.name not in blocked and credit >= cfg.dip_min_credit
                    and quality[s.name] >= cfg.dip_min_quality_pct and (s.earnings_yield or 0.0) > 0.0
                    and eligibility(s.history_bars, s.avg_dollar_volume, s.news_status, cfg)[0]
                    and not earnings_imminent(s.earnings_date, today, cfg)):
                picks.append(s.name)
    return sorted(picks, key=lambda n: (-scores[n][0], n))


def dip_trade(pick: str, pick_score: float, current: Dict[str, float], scores: Dict[str, float], n_stocks: int, cash: float,
              max_per_name: float, industry: Dict[str, str], beta: Dict[str, Optional[float]],
              cfg: Optional[RulebookConfig] = None) -> Optional[Tuple[Optional[str], float]]:
    """How to buy `pick`: (holding to sell or None, target weight), or None if it
    shouldn't be bought. An open slot with the cash for it comes first. Otherwise it
    replaces the lowest-scoring holding, only if the pick scores higher. The portfolio
    limits apply as to any new buy: it is cut back rather than breach one."""
    cfg = cfg or default_config()
    sell: Optional[str] = None
    share = min(max_per_name, 1.0 / max(n_stocks, 1))
    if len(current) < n_stocks and cash >= cfg.min_trade_pct:
        weight = min(share, cash)
    else:
        if not current:
            return None
        sell = min(current, key=lambda s: (scores.get(s, 0.0), s))
        if scores.get(sell, 0.0) + cfg.dip_replace_margin >= pick_score:
            return None
        weight = min(max_per_name, current[sell])
    after = {s: w for s, w in current.items() if s != sell}
    target = targeted_weights({**after, "__slot": weight}, ["__slot"], [pick], industry, beta, cfg)
    weight = target.get(pick, 0.0)
    return (sell, weight) if weight >= cfg.min_trade_pct else None


# -----------------------------------------------------------------------------
# Selection
# -----------------------------------------------------------------------------
def select_names(candidates: Sequence[Any], n: int, held: Iterable[str] = (), blocked: Iterable[str] = (),
                 cfg: Optional[RulebookConfig] = None, today: Optional[date] = None) -> List[str]:
    """Best `n` names, best first. `candidates` have .name, .score and optionally
    .industry, .eligible, .earnings_date.

      * ineligible and blocked names are out, held or not;
      * a holding ranked inside the top keep_rank_multiple x n stays (the keep band);
      * remaining slots go to the best names not already kept, skipping new buys
        reporting earnings imminently and honouring the sub-industry slot limit,
        which only gives way if there aren't enough other names to fill the book."""
    cfg = cfg or default_config()
    n = int(n)
    if n <= 0:
        return []
    today = today or date.today()
    held, blocked = set(held), set(blocked)
    pool = [c for c in candidates if c.name not in blocked and getattr(c, "eligible", True)]
    ranked = sorted(pool, key=lambda c: (-c.score, c.name))
    rank_of = {c.name: i for i, c in enumerate(ranked)}
    band = max(n, math.ceil(n * cfg.keep_rank_multiple))
    max_in_industry = max(1, math.ceil(cfg.max_industry_weight * n))

    chosen: List[Any] = []
    per_industry: Dict[str, int] = {}

    def take(c: Any) -> None:
        chosen.append(c)
        ind = getattr(c, "industry", "") or ""
        if ind:
            per_industry[ind] = per_industry.get(ind, 0) + 1

    def has_room(c: Any) -> bool:
        ind = getattr(c, "industry", "") or ""
        return not ind or per_industry.get(ind, 0) < max_in_industry

    for c in ranked:                                       # holdings in the keep band, best first
        if c.name in held and rank_of[c.name] < band and len(chosen) < n and has_room(c):
            take(c)
    picked = {c.name for c in chosen}
    new_ok = [c for c in ranked if c.name not in picked and c.name not in held
              and not earnings_imminent(getattr(c, "earnings_date", None), today, cfg)]
    for c in new_ok:
        if len(chosen) >= n:
            break
        if has_room(c):
            take(c)
            picked.add(c.name)
    for c in new_ok:                                       # a full book beats the slot limit; weights still obey it
        if len(chosen) >= n:
            break
        if c.name not in picked:
            take(c)
            picked.add(c.name)
    return [c.name for c in sorted(chosen, key=lambda c: rank_of[c.name])]


# -----------------------------------------------------------------------------
# Sizing and portfolio limits
# -----------------------------------------------------------------------------
def inverse_vol_weights(vols: Dict[str, Optional[float]], budget: float, cap: float) -> Dict[str, float]:
    """Budget split in inverse proportion to each name's volatility, capped per
    name with the excess cascading to names with room. A name with unknown
    volatility is treated as a typical one. Over-capacity budget stays unplaced."""
    if budget <= 1e-12 or not vols:
        return {}
    known = [v for v in vols.values() if v and v > 0]
    typical = statistics.median(known) if known else 1.0
    raw = {n: 1.0 / (v if v and v > 0 else typical) for n, v in vols.items()}
    weights = {n: 0.0 for n in raw}
    active, remaining = set(raw), budget
    while active and remaining > 1e-12:
        total = sum(raw[n] for n in active)
        proposed = {n: remaining * raw[n] / total for n in active}
        over = [n for n in active if weights[n] + proposed[n] >= cap - 1e-12]
        if not over:
            for n in active:
                weights[n] += proposed[n]
            break
        for n in over:
            weights[n] = cap
            active.discard(n)
        remaining = budget - sum(weights.values())
    return {n: w for n, w in weights.items() if w > 1e-9}


def apply_portfolio_limits(weights: Dict[str, float], industry: Dict[str, str], beta: Dict[str, Optional[float]],
                           max_per_name: float, cfg: Optional[RulebookConfig] = None) -> Tuple[Dict[str, float], List[str]]:
    """(weights, notes). Names over the sub-industry limit are scaled down and
    the excess offered to names with room; beta above the limit scales the whole
    book down. Anything not placed stays cash (weights may sum below 1)."""
    cfg = cfg or default_config()
    out = {n: w for n, w in weights.items() if n != CASH and w > 0}
    notes: List[str] = []
    capped: set = set()
    for _ in range(max(len(out), 1) + 1):
        totals: Dict[str, float] = {}
        for n, w in out.items():
            ind = industry.get(n) or ""
            if ind:
                totals[ind] = totals.get(ind, 0.0) + w
        over = {i: t for i, t in totals.items() if t > cfg.max_industry_weight + 1e-9}
        if not over:
            break
        excess = 0.0
        for ind, total in over.items():
            scale = cfg.max_industry_weight / total
            for n in out:
                if industry.get(n) == ind:
                    excess += out[n] * (1.0 - scale)
                    out[n] *= scale
            capped.add(ind)
            notes.append(f"{ind} held to {cfg.max_industry_weight:.0%} (was {total:.0%})")
        takers = {n: min(max_per_name - w, cfg.max_industry_weight - totals.get(industry.get(n) or "", 0.0)
                         if industry.get(n) else max_per_name - w)
                  for n, w in out.items() if (industry.get(n) or "") not in capped}
        room = {n: r for n, r in takers.items() if r > 1e-9}
        total_room = sum(room.values())
        if total_room > 1e-9 and excess > 1e-9:
            share = min(1.0, excess / total_room)
            for n, r in room.items():
                out[n] += r * share
    known = [(w, beta[n]) for n, w in out.items() if beta.get(n) is not None]
    portfolio_beta = sum(w * b for w, b in known) + sum(w for n, w in out.items() if beta.get(n) is None)
    if portfolio_beta > cfg.max_portfolio_beta + 1e-9:
        scale = cfg.max_portfolio_beta / portfolio_beta
        out = {n: w * scale for n, w in out.items()}
        notes.append(f"portfolio beta {portfolio_beta:.2f} scaled to {cfg.max_portfolio_beta:.2f}; the rest stays cash")
    return out, notes


def limit_breaches(weights: Dict[str, float], industry: Dict[str, str], beta: Dict[str, Optional[float]],
                   cfg: Optional[RulebookConfig] = None) -> List[str]:
    """The deterministic check risk_check runs on a finished book."""
    cfg = cfg or default_config()
    totals: Dict[str, float] = {}
    for n, w in weights.items():
        ind = industry.get(n) or ""
        if ind and n != CASH:
            totals[ind] = totals.get(ind, 0.0) + w
    reasons = [f"{i} at {t:.1%} exceeds max_industry_weight {cfg.max_industry_weight:.1%}"
               for i, t in sorted(totals.items()) if t > cfg.max_industry_weight + 1e-6]
    b = sum(w * (beta[n] if beta.get(n) is not None else 1.0) for n, w in weights.items() if n != CASH)
    if b > cfg.max_portfolio_beta + 1e-6:
        reasons.append(f"portfolio beta {b:.2f} exceeds max_portfolio_beta {cfg.max_portfolio_beta:.2f}")
    return reasons


# -----------------------------------------------------------------------------
# Selling
# -----------------------------------------------------------------------------
def stop_threshold(daily_vol: float, cfg: Optional[RulebookConfig] = None) -> float:
    cfg = cfg or default_config()
    return cfg.stop_sigma_mult * monthly_sigma(daily_vol)


def stop_breached(price: float, entry_price: float, sector_return: Optional[float], daily_vol: Optional[float],
                  cfg: Optional[RulebookConfig] = None) -> bool:
    """A LOSER, by the same measure for every stock: below what was paid AND
    behind its own sector, since purchase, by stop_sigma_mult monthly sigmas.
    A stock falling with its whole sector is not a stop, and neither is a winner
    that merely lags. Without volatility or sector data the stop cannot judge,
    so it does not fire."""
    if not price or not entry_price or sector_return is None or not daily_vol:
        return False
    own = price / entry_price - 1.0
    return own < 0.0 and own - sector_return <= -stop_threshold(daily_vol, cfg)


def unusual_sector_move(day_move: Optional[float], move_5d: Optional[float], daily_vol: Optional[float],
                        cfg: Optional[RulebookConfig] = None) -> bool:
    """Has a whole sector moved unusually for THAT sector, up or down, today or over
    the last 5 trading days? Measured in its own volatility, so the bar is the same
    idea for a calm sector and a wild one."""
    cfg = cfg or default_config()
    if not daily_vol:
        return False
    today = day_move is not None and abs(day_move) >= cfg.sector_move_sigmas * daily_vol
    week = move_5d is not None and abs(move_5d) >= cfg.sector_move_sigmas * daily_vol * math.sqrt(5)
    return today or week


def reentry_blocked(p: Dict[str, Any], today: date, cfg: Optional[RulebookConfig] = None) -> Dict[str, str]:
    """{symbol: ISO date it may be bought again}, for every stock this portfolio
    sold within reentry_block_days and no longer holds, whatever the reason it
    was sold. Read from the history, so scheduled, news-watch and manual sales
    all count and nothing extra has to be kept in sync."""
    cfg = cfg or default_config()
    held = set(p.get("holdings") or {}) | set(p.get("pending_order_ids") or {})
    out: Dict[str, str] = {}
    for ev in p.get("rebalance_history") or []:
        d = _iso_date(ev.get("date"))
        if d is None or (today - d).days >= cfg.reentry_block_days:
            continue
        until = (d + timedelta(days=cfg.reentry_block_days)).isoformat()
        for o in ev.get("orders") or []:
            sym = o.get("symbol")
            if (str(o.get("side", "")).upper() == "SELL" and sym not in held
                    and str(o.get("result", "FILLED")).upper() in ("FILLED", "PARTIAL")):
                out[sym] = max(until, out.get(sym, ""))
        for sym in ev.get("excluded") or []:           # named by the news review: out even if it wasn't held
            if sym not in held:
                out[sym] = max(until, out.get(sym, ""))
    return out


def targeted_weights(current: Dict[str, float], exits: Iterable[str], replacements: Sequence[str],
                     industry: Optional[Dict[str, str]] = None, beta: Optional[Dict[str, Optional[float]]] = None,
                     cfg: Optional[RulebookConfig] = None) -> Dict[str, float]:
    """News and stops act on the affected stock only: everything else keeps its
    weight, each exit's weight passes to one replacement, best first. A slot with
    no replacement stays cash. A replacement is a new buy, so it obeys the portfolio
    limits: it is cut back (the rest stays cash) rather than pushing a sub-industry
    or the portfolio's beta over. Holdings that drifted over a limit aren't sold
    for that here; the next scheduled recalibration resizes them."""
    cfg = cfg or default_config()
    industry, beta = industry or {}, beta or {}
    exits = set(exits)
    out = {s: w for s, w in current.items() if s not in exits}
    freed = sorted((w for s, w in current.items() if s in exits), reverse=True)
    fresh = [s for s in replacements if s not in out and s not in exits]
    b = sum(w * (1.0 if beta.get(s) is None else float(beta[s])) for s, w in out.items())
    for w, s in zip(freed, fresh):
        ind = industry.get(s) or ""
        if ind:
            used = sum(x for k, x in out.items() if industry.get(k) == ind)
            w = min(w, max(0.0, cfg.max_industry_weight - used))
        stock_beta = 1.0 if beta.get(s) is None else float(beta[s])
        if stock_beta > 0:
            w = min(w, max(0.0, cfg.max_portfolio_beta - b) / stock_beta)
        if w > 1e-9:
            out[s] = w
            b += w * stock_beta
    return out


def below_min_trade(delta_value: float, portfolio_value: float, target_shares: float, cfg: Optional[RulebookConfig] = None) -> bool:
    """A small resize isn't worth its cost. Closing a position always trades."""
    cfg = cfg or default_config()
    return target_shares > 0 and portfolio_value > 0 and abs(delta_value) / portfolio_value < cfg.min_trade_pct


# -----------------------------------------------------------------------------
# Position dates (for "since purchase")
# -----------------------------------------------------------------------------
def _iso_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def blended_entry_date(old_date: Optional[str], old_cost: float, new_date: str, added_cost: float) -> str:
    """Adding to a position blends its entry PRICE, so its entry DATE is blended
    the same way, keeping 'since purchase' comparable with the sector."""
    new = _iso_date(new_date)
    old = _iso_date(old_date)
    if old is None or new is None:
        return new_date
    total = old_cost + added_cost
    if total <= 0:
        return new_date
    return (old + timedelta(days=round((new - old).days * added_cost / total))).isoformat()


def holding_entry_date(p: Dict[str, Any], symbol: str) -> Optional[date]:
    """The holding's recorded entry date; for positions opened before dates were
    recorded, its latest buy in the history, else the launch date."""
    d = _iso_date(((p.get("holdings") or {}).get(symbol) or {}).get("entry_date"))
    if d:
        return d
    for ev in reversed(p.get("rebalance_history") or []):
        for o in ev.get("orders") or []:
            if o.get("symbol") == symbol and str(o.get("side", "")).upper() == "BUY":
                found = _iso_date(ev.get("date"))
                if found:
                    return found
    return _iso_date(p.get("created_at"))


# -----------------------------------------------------------------------------
# Performance arithmetic (backtest and scorecard)
# -----------------------------------------------------------------------------
def performance_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Annualised return and volatility, Sharpe (risk-free 0) and max drawdown of
    a daily value series."""
    vals = [v for v in values if v and v > 0]
    if len(vals) < 2:
        return {"annual_return": None, "annual_vol": None, "sharpe": None, "max_drawdown": None}
    rets = daily_returns(vals)
    annual = (vals[-1] / vals[0]) ** (TRADING_DAYS_PER_YEAR / len(rets)) - 1.0
    vol = statistics.stdev(rets) * math.sqrt(TRADING_DAYS_PER_YEAR) if len(rets) > 1 else 0.0
    peak, worst = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1.0)
    sharpe = statistics.fmean(rets) * TRADING_DAYS_PER_YEAR / vol if vol > 0 else None
    return {"annual_return": annual, "annual_vol": vol, "sharpe": sharpe, "max_drawdown": worst}
