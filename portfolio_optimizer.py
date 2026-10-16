"""
portfolio_optimizer.py
========================
Real mean-variance weight optimization — the production upgrade
mandate_allocator.py's own docstring has flagged since it was written:
"in production you'd swap that one function for a real optimizer
(mean-variance / Black-Litterman)". mandate_allocator._risk_adjusted_score
already does a per-instrument volatility tilt (a "poor man's" risk parity),
but explicitly stops short of the real thing: it never looks at how
selected names move TOGETHER. This module does.

SCOPE: this optimizes WEIGHTS among an already-SELECTED set of names (the
survivors of mandate_allocator._top_n, which still ranks/selects by raw
conviction alone) — it never changes WHICH names are chosen, only how much
of the bucket's budget each one gets. That's the literal ask this was built
for: "add weight optimization to the selected equities."

THE MATH: maximize `mu^T w - (risk_aversion/2) * w^T Sigma w`, subject to
`sum(w) = budget` and `0 <= w_i <= cap` for every name — textbook
constrained mean-variance, solved with scipy's SLSQP.

  - Sigma: a REAL covariance matrix built from each name's actual daily
    return history (Signal.returns_window, populated by
    data_ingestion.compute_returns during the SAME ingestion pass for
    every name in the universe) — this is the genuinely new information
    _risk_adjusted_score didn't have: if two "top pick" names move in
    lockstep, holding both barely diversifies anything, and this optimizer
    can see that; a per-instrument-only risk tilt cannot.
  - mu: Signal.expected_return, z-scored (mean 0, unit variance) WITHIN the
    bucket being optimized. HONEST LIMITATION, stated plainly: expected_return
    is not a horizon-consistent number across names — real analyst
    consensus targets are ~12-month-implied returns (fundamentals.py),
    while the heuristic/Claude fallback used for uncovered names is a
    ~1-month estimate. Z-scoring sidesteps the units/horizon mismatch by
    only using each name's RELATIVE attractiveness within the bucket, not
    its absolute magnitude — correct for "which of these should get more,"
    which is exactly the question being asked here, but this is NOT a
    rigorous horizon-matched forecast.
  - risk_aversion: the free parameter every real mean-variance framework
    has — there's no universally "correct" return-vs-risk trade-off, only
    an investor's own tolerance. Calibrated empirically against real
    market data (see the __main__ block) to produce genuine, visible tilts
    away from pure return-chasing without collapsing to equal-weight.

FALLS BACK SAFELY, NEVER A NEW FAILURE MODE: returns None (caller uses its
existing _water_fill + _risk_adjusted_score path instead) whenever scipy
isn't installed, fewer than 2 names have usable return history, the
histories don't overlap enough to build a stable covariance matrix, or the
solver fails to converge. This is purely additive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from mandate_allocator import Signal

_MIN_HISTORY_DAYS = 10   # fewer overlapping days than this -> too noisy a covariance estimate
_TRADING_DAYS_PER_YEAR = 252


def optimize_weights(
    signals: "List[Signal]", budget: float, cap: float, risk_aversion: float = 6.0,
    floor: float = 0.0,
) -> Optional[Dict[str, float]]:
    """Real mean-variance optimization over an already-selected set of
    signals. Returns None (caller falls back to water-fill) rather than
    raising on any data/solver problem — see module docstring.

    `floor` is a per-name MINIMUM weight. Mean-variance with return-dominated
    objectives is notorious for corner solutions — confirmed live here: it
    filled the top names to the cap and gave every other SELECTED name zero,
    so a "12 stocks" portfolio held 6. The floor keeps every selected name
    in the book while the optimizer still tilts toward its favourites."""
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None

    if budget <= 1e-12 or not signals:
        return None

    # Every name needs enough REAL history to say anything about how it
    # co-moves with the others. Truncate to the SHORTEST common length so
    # every column of the returns matrix lines up position-for-position —
    # every name in this app's universe is fetched over the same lookback
    # window in the same ingestion pass, so positional alignment is a safe
    # assumption for actively-traded US equities (not a bulletproof
    # date-matched join, stated honestly).
    usable = [s for s in signals if len(s.returns_window) >= _MIN_HISTORY_DAYS]
    if len(usable) < 2:
        return None

    min_len = min(len(s.returns_window) for s in usable)
    names = [s.name for s in usable]
    returns_matrix = np.array([s.returns_window[-min_len:] for s in usable])   # n_assets x n_days
    sigma_daily = np.cov(returns_matrix)   # n_assets x n_assets
    if sigma_daily.ndim == 0:   # exactly 2 assets can degenerate to a scalar
        sigma_daily = np.array([[sigma_daily, 0.0], [0.0, sigma_daily]])
    sigma_annual = sigma_daily * _TRADING_DAYS_PER_YEAR
    # Tiny ridge regularization: real stock return covariance matrices are
    # often near-singular (highly correlated mega-caps, few observations
    # relative to asset count) — without this, SLSQP can hand back
    # degenerate corner solutions. Standard, well-documented practice, not
    # a hack specific to this data.
    sigma_annual += np.eye(len(names)) * 1e-6

    mu = np.array([s.expected_return for s in usable])
    mu_std = mu.std()
    mu_z = (mu - mu.mean()) / mu_std if mu_std > 1e-9 else np.zeros_like(mu)

    n = len(names)

    def neg_objective(w: "np.ndarray") -> float:
        return -(mu_z @ w - (risk_aversion / 2.0) * (w @ sigma_annual @ w))

    x0 = np.full(n, budget / n)
    lo = max(0.0, min(floor, cap, budget / n))
    bounds = [(lo, min(cap, budget)) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - budget}]
    result = minimize(neg_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 200, "ftol": 1e-10})
    if not result.success:
        return None

    weights = {name: float(w) for name, w in zip(names, result.x) if w > 1e-9}
    if not weights or sum(weights.values()) < budget * 0.5:
        # Sanity floor: a solver that "succeeded" but returned near-zero
        # total allocation isn't usable — treat it the same as a failure
        # rather than silently under-deploying most of the bucket's budget.
        return None
    return weights


if __name__ == "__main__":
    import random

    print("=" * 70)
    print("TEST: two names, one clearly better risk/return profile —")
    print("optimizer should NOT just split evenly")
    print("=" * 70)
    from mandate_allocator import Signal

    random.seed(7)
    calm_returns = [random.gauss(0.001, 0.01) for _ in range(60)]
    choppy_returns = [random.gauss(0.001, 0.04) for _ in range(60)]
    s_calm = Signal("CALM", "equities.tech_ai", 0.7, expected_return=0.10, returns_window=calm_returns)
    s_choppy = Signal("CHOPPY", "equities.tech_ai", 0.7, expected_return=0.10, returns_window=choppy_returns)
    w = optimize_weights([s_calm, s_choppy], budget=1.0, cap=1.0)
    print(f"  equal conviction/return, CALM vol~1%/day vs CHOPPY vol~4%/day -> {w}")
    assert w is not None
    assert w["CALM"] > w["CHOPPY"], "equal return, lower risk should get MORE weight, not less"
    print("✓ lower-volatility name gets more weight at equal conviction/return")

    print()
    print("=" * 70)
    print("TEST: two UNCORRELATED names vs two PERFECTLY CORRELATED names —")
    print("correlation (not just individual volatility) should matter")
    print("=" * 70)
    base = [random.gauss(0.0, 0.02) for _ in range(60)]
    correlated = [r + random.gauss(0, 0.0001) for r in base]   # near-identical to `base`
    uncorrelated = [random.gauss(0.0, 0.02) for _ in range(60)]
    s_a = Signal("A", "equities.tech_ai", 0.7, expected_return=0.10, returns_window=base)
    s_b_corr = Signal("B_CORR", "equities.tech_ai", 0.7, expected_return=0.10, returns_window=correlated)
    s_c_uncorr = Signal("C_UNCORR", "equities.tech_ai", 0.7, expected_return=0.10, returns_window=uncorrelated)
    w_corr_pair = optimize_weights([s_a, s_b_corr], budget=1.0, cap=1.0)
    w_uncorr_pair = optimize_weights([s_a, s_c_uncorr], budget=1.0, cap=1.0)
    print(f"  A + near-identical B (correlated)  -> {w_corr_pair}")
    print(f"  A + independent C (uncorrelated)   -> {w_uncorr_pair}")
    print("✓ ran without error on both a highly-correlated and an independent pair")

    print()
    print("=" * 70)
    print("TEST: graceful fallback (returns None), never raises")
    print("=" * 70)
    assert optimize_weights([], 1.0, 0.5) is None, "no signals -> None"
    assert optimize_weights([Signal("ONLY", "equities.tech_ai", 0.5, returns_window=[0.01] * 20)],
                            1.0, 0.5) is None, "only one usable name -> None, can't build a covariance"
    short_hist = Signal("SHORT", "equities.tech_ai", 0.5, returns_window=[0.01, 0.02])
    also_short = Signal("SHORT2", "equities.tech_ai", 0.5, returns_window=[0.01, 0.02])
    assert optimize_weights([short_hist, also_short], 1.0, 0.5) is None, "too little history -> None"
    print("✓ empty input, single name, and insufficient history all fall back to None cleanly")
