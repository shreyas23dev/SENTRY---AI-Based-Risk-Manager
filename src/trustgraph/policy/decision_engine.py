"""
decision_engine.py — TRUSTGRAPH Progressive Risk Decision Policy Engine
=======================================================================

Core deterministic policy layer translating continuous fused risk (R_t)
and multi-source contextual evidence into progressive operational actions:
  - ALLOW    (Low risk)
  - VERIFY   (Moderate risk)
  - THROTTLE (High risk)
  - BLOCK    (Very high risk)
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from trustgraph.policy.config import PolicyAction, RiskBand


@dataclass(frozen=True)
class PolicyThresholds:
    """Ordered threshold boundaries for progressive action selection."""
    tau_verify: float    # Boundary between ALLOW and VERIFY
    tau_throttle: float  # Boundary between VERIFY and THROTTLE
    tau_block: float     # Boundary between THROTTLE and BLOCK

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_verify < self.tau_throttle < self.tau_block <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 <= tau_verify < tau_throttle < tau_block <= 1.0. "
                f"Got tau_verify={self.tau_verify}, tau_throttle={self.tau_throttle}, tau_block={self.tau_block}"
            )

    def to_dict(self) -> Dict[str, float]:
        return {
            "tau_verify": self.tau_verify,
            "tau_throttle": self.tau_throttle,
            "tau_block": self.tau_block,
        }


def assign_action_and_band(R_t: float, thresholds: PolicyThresholds) -> Tuple[PolicyAction, RiskBand]:
    """
    Assign deterministic action and risk band for a single continuous score R_t.
    
    Parameters
    ----------
    R_t : float in [0.0, 1.0]
    thresholds : PolicyThresholds
    
    Returns
    -------
    (PolicyAction, RiskBand)
    """
    r = float(np.clip(R_t, 0.0, 1.0))
    if r < thresholds.tau_verify:
        return PolicyAction.ALLOW, RiskBand.LOW
    elif r < thresholds.tau_throttle:
        return PolicyAction.VERIFY, RiskBand.MODERATE
    elif r < thresholds.tau_block:
        return PolicyAction.THROTTLE, RiskBand.HIGH
    else:
        return PolicyAction.BLOCK, RiskBand.VERY_HIGH


def batch_assign_actions(
    R_t_arr: np.ndarray, thresholds: PolicyThresholds
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized action and risk band assignment for an array of R_t scores.
    
    Returns
    -------
    (actions_array, risk_bands_array) as string ndarrays.
    """
    r = np.clip(np.asarray(R_t_arr, dtype=float), 0.0, 1.0)
    n = len(r)

    actions = np.full(n, PolicyAction.ALLOW.value, dtype=object)
    bands = np.full(n, RiskBand.LOW.value, dtype=object)

    verify_mask = (r >= thresholds.tau_verify) & (r < thresholds.tau_throttle)
    throttle_mask = (r >= thresholds.tau_throttle) & (r < thresholds.tau_block)
    block_mask = (r >= thresholds.tau_block)

    actions[verify_mask] = PolicyAction.VERIFY.value
    bands[verify_mask] = RiskBand.MODERATE.value

    actions[throttle_mask] = PolicyAction.THROTTLE.value
    bands[throttle_mask] = RiskBand.HIGH.value

    actions[block_mask] = PolicyAction.BLOCK.value
    bands[block_mask] = RiskBand.VERY_HIGH.value

    return actions, bands


def generate_explanation(
    A_t: float,
    P_t: float,
    G_t: float,
    R_t: float,
    action: PolicyAction,
    thresholds: PolicyThresholds,
    D_t: float = 0.0,
    V_t: float = 0.0,
    d_t: int = 0,
    v_t: int = 0,
    device_info: Optional[str] = None,
) -> str:
    """
    Generate an auditable, deterministic natural language explanation for an action.
    """
    r_pct = round(R_t * 100, 1)
    a_pct = round(A_t * 100, 1)
    p_pct = round(P_t * 100, 1)
    g_pct = round(G_t * 100, 1)

    reasons = []

    # 1. Primary driver identification
    if A_t >= thresholds.tau_block:
        reasons.append(f"Instantaneous tabular risk is severe (A_t={A_t:.4f}).")
    elif A_t >= thresholds.tau_verify:
        reasons.append(f"Instantaneous tabular risk is elevated (A_t={A_t:.4f}).")
    else:
        reasons.append(f"Instantaneous tabular risk is baseline-normal (A_t={A_t:.4f}).")

    # 2. Contextual temporal contribution
    if P_t >= 0.30:
        reasons.append(f"Entity exhibits persistent longitudinal velocity bursts (P_t={P_t:.4f}).")
    elif P_t > 0.0:
        reasons.append(f"Mild temporal accumulation detected on entity proxy (P_t={P_t:.4f}).")

    # 3. Contextual relational contribution
    if G_t >= 0.50:
        dev_str = f" on device '{device_info}'" if device_info and device_info != "nan" else ""
        reasons.append(f"Cross-entity graph degree/velocity is high (G_t={G_t:.4f}, connected_entities={d_t}{dev_str}).")
    elif G_t > 0.0:
        reasons.append(f"Moderate relational sharing detected (G_t={G_t:.4f}, connected_entities={d_t}).")

    context_summary = " ".join(reasons)

    # 4. Action-specific operational directive
    if action == PolicyAction.ALLOW:
        return f"ALLOW: Combined risk R_t={R_t:.4f} is below verification boundary ({thresholds.tau_verify:.2f}). {context_summary}"
    elif action == PolicyAction.VERIFY:
        return f"VERIFY: Combined risk R_t={R_t:.4f} warrants step-up authentication. {context_summary}"
    elif action == PolicyAction.THROTTLE:
        return f"THROTTLE: Combined risk R_t={R_t:.4f} requires operational restriction / delayed clearing. {context_summary}"
    else: # BLOCK
        return f"BLOCK: Combined risk R_t={R_t:.4f} exceeds critical block threshold ({thresholds.tau_block:.2f}). Transaction rejected. {context_summary}"


def verify_policy_invariants(
    R_t_arr: np.ndarray, actions: np.ndarray, thresholds: PolicyThresholds
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify all mathematical and monotonicity invariants for the progressive policy.
    """
    r = np.asarray(R_t_arr, dtype=float)
    n = len(r)

    # 1. Bounded range [0, 1]
    range_violations = int(np.sum((r < -1e-6) | (r > 1.0 + 1e-6)))

    # 2. Valid actions
    valid_actions = {a.value for a in PolicyAction}
    invalid_actions = int(sum(1 for a in actions if a not in valid_actions))

    # 3. Monotonicity check
    severity_map = {
        PolicyAction.ALLOW.value: 0,
        PolicyAction.VERIFY.value: 1,
        PolicyAction.THROTTLE.value: 2,
        PolicyAction.BLOCK.value: 3,
    }
    ranks = np.array([severity_map[a] for a in actions])

    # Check monotonicity via sort
    sorted_idx = np.argsort(r)
    sorted_ranks = ranks[sorted_idx]
    # Check if ranks are monotonically non-decreasing
    diffs = np.diff(sorted_ranks)
    monotonicity_violations = int(np.sum(diffs < 0))

    # 4. Exact boundary verification
    allow_correct = np.all((r[actions == PolicyAction.ALLOW.value] < thresholds.tau_verify + 1e-9))
    verify_correct = np.all(
        (r[actions == PolicyAction.VERIFY.value] >= thresholds.tau_verify - 1e-9) &
        (r[actions == PolicyAction.VERIFY.value] < thresholds.tau_throttle + 1e-9)
    )
    throttle_correct = np.all(
        (r[actions == PolicyAction.THROTTLE.value] >= thresholds.tau_throttle - 1e-9) &
        (r[actions == PolicyAction.THROTTLE.value] < thresholds.tau_block + 1e-9)
    )
    block_correct = np.all((r[actions == PolicyAction.BLOCK.value] >= thresholds.tau_block - 1e-9))

    boundary_violations = int(not (allow_correct and verify_correct and throttle_correct and block_correct))

    all_passed = (
        range_violations == 0 and
        invalid_actions == 0 and
        monotonicity_violations == 0 and
        boundary_violations == 0
    )

    diagnostics = {
        "total_evaluated": n,
        "range_violations": range_violations,
        "invalid_action_violations": invalid_actions,
        "monotonicity_violations": monotonicity_violations,
        "boundary_violations": boundary_violations,
        "thresholds": thresholds.to_dict(),
        "all_passed": all_passed,
    }
    return all_passed, diagnostics
