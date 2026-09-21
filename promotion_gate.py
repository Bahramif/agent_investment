"""
promotion_gate.py
=================
Champion / challenger promotion gate.

The scorecard lives in `promotion_criteria.yaml`; this module reads it and turns
each challenger into a REJECT / PROMOTE (auto) / PROMOTE (needs sign-off) verdict,
with a per-gate breakdown you can drop straight into the audit log.

Design principle (same as everywhere else in the firm): the learning loop only
*proposes* a challenger. This gate *decides*. Every hard gate is a veto — one
failure and the challenger is rejected, full stop. No weighted average lets a
great return paper over a risk failure.

Core evaluation (`evaluate_promotion`) is pure stdlib and takes a plain dict of
criteria, so it runs and unit-tests with no third-party deps. Only `load_criteria`
needs PyYAML (`pip install pyyaml`).

Wiring into the LangGraph flow is shown at the bottom (`sim_validation_gate`).
Note this is the *sim-validation* gate for strategy changes — distinct from the
live pre-trade `risk_check` node, which never changes based on learning.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


# ----------------------------------------------------------------------------- 
# Data shapes
# -----------------------------------------------------------------------------
@dataclass
class StrategyMetrics:
    """Net-of-cost, out-of-sample metrics for a champion or challenger."""
    net_sharpe: float
    net_return: float          # annualized, as a fraction (0.14 = 14%)
    max_drawdown: float        # positive fraction (0.18 = 18% peak-to-trough)
    cvar_95: float             # positive fraction, 95% tail loss
    volatility: float
    leverage: float


@dataclass
class ChallengerReport:
    """Everything the sim harness produced for one challenger."""
    metrics: StrategyMetrics
    n_trades: int
    p_value: float                       # vs champion, paired/bootstrap test
    deflated_sharpe: float               # corrected for number of challengers tried
    run_returns: List[float]             # net return per stochastic run
    regime_returns: Dict[str, float]     # challenger net return by regime
    regime_returns_champion: Dict[str, float]
    walkforward_fold_returns: List[float]
    perturbation_degradation: float      # frac perf drop under +/- param perturbation
    capacity_ok: bool
    shadow_return: float
    sim_return: float
    costs_are_net: bool
    # governance flags — did the change touch anything that needs a human?
    leverage_changed: bool = False
    risk_limits_changed: bool = False
    capital_rules_changed: bool = False
    leverage_override: bool = False      # explicitly permits higher leverage


class Decision(str, Enum):
    REJECT = "reject"
    PROMOTE_AUTO = "promote_auto"
    PROMOTE_SIGNOFF = "promote_needs_signoff"


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass
class PromotionVerdict:
    decision: Decision
    rank_score: float
    gates: List[GateResult] = field(default_factory=list)

    @property
    def failures(self) -> List[GateResult]:
        return [g for g in self.gates if not g.passed]

    def rationale(self) -> str:
        """One-line summary for the audit log."""
        if self.decision is Decision.REJECT:
            reasons = "; ".join(f"{g.name}: {g.detail}" for g in self.failures)
            return f"REJECTED ({len(self.failures)} gate(s) failed) — {reasons}"
        tag = "auto-promote" if self.decision is Decision.PROMOTE_AUTO else "promote pending sign-off"
        return f"ALL GATES PASSED, {tag} (rank score {self.rank_score:.3f})"


# ----------------------------------------------------------------------------- 
# Evaluation
# -----------------------------------------------------------------------------
def _pct(values: List[float], p: float) -> float:
    """p-th percentile (0-100) via linear interpolation, stdlib only."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def evaluate_promotion(
    champion: StrategyMetrics,
    challenger: ChallengerReport,
    criteria: Dict[str, Any],
) -> PromotionVerdict:
    """Score a challenger against the champion using the scorecard dict."""
    c, m = challenger, challenger.metrics
    g: List[GateResult] = []

    # --- edge -----------------------------------------------------------------
    eg = criteria["edge_gates"]
    margin = m.net_sharpe - champion.net_sharpe
    g.append(GateResult("edge.sharpe_margin", margin >= eg["min_sharpe_margin"],
                        f"sharpe +{margin:.2f} vs required +{eg['min_sharpe_margin']:.2f}"))
    if eg["require_return_ge_champion"]:
        g.append(GateResult("edge.return", m.net_return >= champion.net_return,
                            f"return {m.net_return:.1%} vs champion {champion.net_return:.1%}"))

    # --- risk -----------------------------------------------------------------
    rg = criteria["risk_gates"]
    dd_cap = champion.max_drawdown * (1 + rg["max_drawdown_rel_tolerance"])
    g.append(GateResult("risk.max_drawdown", m.max_drawdown <= dd_cap,
                        f"MDD {m.max_drawdown:.1%} vs cap {dd_cap:.1%}"))
    if rg["require_cvar_le_champion"]:
        g.append(GateResult("risk.cvar", m.cvar_95 <= champion.cvar_95,
                            f"CVaR {m.cvar_95:.1%} vs champion {champion.cvar_95:.1%}"))
    g.append(GateResult("risk.max_leverage", m.leverage <= rg["max_leverage"],
                        f"leverage {m.leverage:.2f}x vs cap {rg['max_leverage']:.2f}x"))
    if rg["leverage_not_above_champion"] and not c.leverage_override:
        g.append(GateResult("risk.leverage_vs_champion", m.leverage <= champion.leverage,
                            f"leverage {m.leverage:.2f}x vs champion {champion.leverage:.2f}x"))

    # --- statistical validity -------------------------------------------------
    sg = criteria["statistical_gates"]
    g.append(GateResult("stat.min_trades", c.n_trades >= sg["min_trades"],
                        f"{c.n_trades} trades vs min {sg['min_trades']}"))
    g.append(GateResult("stat.significance", c.p_value < sg["max_p_value"],
                        f"p={c.p_value:.3f} vs max {sg['max_p_value']}"))
    if sg["require_positive_deflated_sharpe"]:
        g.append(GateResult("stat.deflated_sharpe", c.deflated_sharpe > 0,
                            f"deflated sharpe {c.deflated_sharpe:.2f}"))
    dist = sg["distribution"]
    if dist["require_median_beats_champion"]:
        med = statistics.median(c.run_returns) if c.run_returns else float("nan")
        g.append(GateResult("stat.median_run", med > champion.net_return,
                            f"median run {med:.1%} vs champion {champion.net_return:.1%}"))
    low = _pct(c.run_returns, dist["low_percentile"])
    floor = dist["low_percentile_floor_vs_champion"] * champion.net_return
    g.append(GateResult("stat.low_percentile", low >= floor,
                        f"p{dist['low_percentile']} run {low:.1%} vs floor {floor:.1%}"))

    # --- robustness -----------------------------------------------------------
    ro = criteria["robustness_gates"]
    worst_regime, worst_gap = None, 0.0
    for regime, ret in c.regime_returns.items():
        champ = c.regime_returns_champion.get(regime, ret)
        threshold = champ * (1 - ro["regime_worse_tolerance"])
        if ret < threshold:
            gap = (champ - ret) / abs(champ) if champ else 1.0
            if gap > worst_gap:
                worst_regime, worst_gap = regime, gap
    g.append(GateResult("robust.regimes", worst_regime is None,
                        "no regime materially worse" if worst_regime is None
                        else f"'{worst_regime}' {worst_gap:.0%} below champion"))
    folds = c.walkforward_fold_returns
    win_rate = sum(1 for r in folds if r > 0) / len(folds) if folds else 0.0
    g.append(GateResult("robust.walkforward", win_rate >= ro["min_walkforward_win_rate"],
                        f"{win_rate:.0%} folds positive vs min {ro['min_walkforward_win_rate']:.0%}"))
    g.append(GateResult("robust.param_stability",
                        c.perturbation_degradation < ro["max_perturbation_degradation"],
                        f"perturbation drop {c.perturbation_degradation:.0%} "
                        f"vs max {ro['max_perturbation_degradation']:.0%}"))

    # --- cost & operational ---------------------------------------------------
    og = criteria["cost_operational_gates"]
    if og["costs_already_net"]:
        g.append(GateResult("ops.costs_net", c.costs_are_net,
                            "metrics net of costs" if c.costs_are_net else "metrics NOT net of costs"))
    if og["require_capacity_ok"]:
        g.append(GateResult("ops.capacity", c.capacity_ok,
                            "capacity ok" if c.capacity_ok else "capacity insufficient"))
    denom = abs(c.sim_return) if c.sim_return else 1.0
    skew = abs(c.shadow_return - c.sim_return) / denom
    g.append(GateResult("ops.shadow_vs_sim", skew <= og["shadow_vs_sim_tolerance"],
                        f"shadow/sim skew {skew:.0%} vs tol {og['shadow_vs_sim_tolerance']:.0%}"))

    # --- decide ---------------------------------------------------------------
    rank_score = c.deflated_sharpe  # ranking metric among eligible challengers
    if any(not gate.passed for gate in g):
        return PromotionVerdict(Decision.REJECT, rank_score, g)

    gov = criteria["governance"]["auto_promote_requires"]
    can_auto = (
        (not c.leverage_changed or not gov["leverage_unchanged"])
        and (not c.risk_limits_changed or not gov["risk_limits_unchanged"])
        and (not c.capital_rules_changed or not gov["capital_rules_unchanged"])
    )
    decision = Decision.PROMOTE_AUTO if can_auto else Decision.PROMOTE_SIGNOFF
    return PromotionVerdict(decision, rank_score, g)


def load_criteria(path: str = "promotion_criteria.yaml") -> Dict[str, Any]:
    """Load the scorecard. Requires PyYAML (`pip install pyyaml`)."""
    import yaml  # local import so evaluate_promotion stays dependency-free
    with open(path) as fh:
        return yaml.safe_load(fh)


# ----------------------------------------------------------------------------- 
# LangGraph wiring — the sim-validation gate node (distinct from live risk_check)
# -----------------------------------------------------------------------------
def sim_validation_gate(state: dict) -> dict:
    """Node: score the challenger, stash the verdict + rationale in state."""
    verdict = evaluate_promotion(
        champion=state["champion_metrics"],
        challenger=state["challenger_report"],
        criteria=state["promotion_criteria"],
    )
    return {"promotion_verdict": verdict, "audit_note": verdict.rationale()}


def promotion_router(state: dict) -> str:
    """Conditional edge: route on the verdict."""
    return state["promotion_verdict"].decision.value

# In the graph:
#   g.add_node("sim_validation", sim_validation_gate)
#   g.add_conditional_edges("sim_validation", promotion_router, {
#       "reject":                 "keep_champion",
#       "promote_auto":           "shadow_mode",     # start the trust ladder
#       "promote_needs_signoff":  "human_signoff",
#   })


if __name__ == "__main__":
    # Inline demo — no YAML needed, uses a dict that mirrors the config file.
    criteria = {
        "edge_gates": {"min_sharpe_margin": 0.30, "require_return_ge_champion": True},
        "risk_gates": {"max_drawdown_rel_tolerance": 0.10, "require_cvar_le_champion": True,
                       "max_leverage": 2.0, "leverage_not_above_champion": True},
        "statistical_gates": {"min_trades": 150, "max_p_value": 0.05,
                              "require_positive_deflated_sharpe": True,
                              "distribution": {"require_median_beats_champion": True,
                                               "low_percentile": 10,
                                               "low_percentile_floor_vs_champion": 0.90}},
        "robustness_gates": {"regime_worse_tolerance": 0.10, "min_walkforward_win_rate": 0.50,
                             "param_perturbation_pct": 0.10, "max_perturbation_degradation": 0.25},
        "cost_operational_gates": {"costs_already_net": True, "require_capacity_ok": True,
                                   "shadow_vs_sim_tolerance": 0.15},
        "ranking": {"metric": "deflated_sharpe"},
        "governance": {"auto_promote_requires": {"leverage_unchanged": True,
                                                 "risk_limits_unchanged": True,
                                                 "capital_rules_unchanged": True}},
    }

    champion = StrategyMetrics(net_sharpe=1.10, net_return=0.14, max_drawdown=0.18,
                               cvar_95=0.06, volatility=0.12, leverage=1.0)

    strong = ChallengerReport(
        metrics=StrategyMetrics(1.55, 0.19, 0.17, 0.055, 0.12, 1.0),
        n_trades=320, p_value=0.012, deflated_sharpe=0.9,
        run_returns=[0.15, 0.17, 0.19, 0.20, 0.22, 0.18, 0.16],
        regime_returns={"bull": 0.24, "bear": 0.08, "high_vol": 0.11},
        regime_returns_champion={"bull": 0.22, "bear": 0.07, "high_vol": 0.10},
        walkforward_fold_returns=[0.03, -0.01, 0.05, 0.04, 0.02],
        perturbation_degradation=0.12, capacity_ok=True,
        shadow_return=0.18, sim_return=0.19, costs_are_net=True,
    )

    risky = ChallengerReport(  # better return, but bought with leverage + deeper DD
        metrics=StrategyMetrics(1.60, 0.26, 0.29, 0.11, 0.20, 1.8),
        n_trades=90, p_value=0.08, deflated_sharpe=-0.2,
        run_returns=[0.05, 0.30, -0.02, 0.40, 0.10],
        regime_returns={"bull": 0.40, "bear": -0.05, "high_vol": 0.02},
        regime_returns_champion={"bull": 0.22, "bear": 0.07, "high_vol": 0.10},
        walkforward_fold_returns=[0.10, -0.08, 0.15, -0.05, -0.03],
        perturbation_degradation=0.40, capacity_ok=True,
        shadow_return=0.12, sim_return=0.26, costs_are_net=True,
        leverage_changed=True,
    )

    for name, ch in [("strong challenger", strong), ("risky challenger", risky)]:
        v = evaluate_promotion(champion, ch, criteria)
        print(f"\n=== {name} ===")
        print(v.rationale())
        for gate in v.gates:
            print(f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name:26s} {gate.detail}")
