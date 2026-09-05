"""
fusion_engine.py — TRUSTGRAPH Phase 3.1 Conditional Risk Fusion Engine
======================================================================

Implements candidate conditional risk fusion rules where contextual signals
(P_t, G_t) provide strictly additive / non-suppressive evidence:

  Candidate F1:
      R_t = clip(A_t + α * P_t + β * G_t, 0.0, 1.0)

  Candidate F2:
      R_t = clip(A_t + β * G_t * (1 - A_t), 0.0, 1.0)

  Candidate F3:
      R_t = clip(A_t + α * P_t * (1 - A_t) + β * G_t * (1 - A_t), 0.0, 1.0)

  Candidate F4:
      R_t = clip(max(A_t, c_P * P_t, c_G * G_t), 0.0, 1.0)

Key Invariants (Enforced & Verified):
  1. Missing Context Invariance: When P_t = 0 and G_t = 0, R_t = A_t.
  2. Non-Suppression (Monotonicity): R_t >= A_t for all transactions.
  3. Boundedness: 0.0 <= R_t <= 1.0 for all transactions.
"""

from typing import Dict, Tuple, Union
import numpy as np


def compute_fusion_f1(
    A_t: np.ndarray,
    P_t: np.ndarray,
    G_t: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> np.ndarray:
    """Candidate F1: Additive Contextual Boost with Clipping."""
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    raw = A_t + alpha * P_t + beta * G_t
    return np.clip(raw, 0.0, 1.0)


def compute_fusion_f2(
    A_t: np.ndarray,
    P_t: np.ndarray,
    G_t: np.ndarray,
    beta: float = 0.5,
    alpha: float = 0.0,  # F2 is relational-only context
) -> np.ndarray:
    """Candidate F2: Residual Relational Boost."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    raw = A_t + beta * G_t * (1.0 - A_t)
    return np.clip(raw, 0.0, 1.0)


def compute_fusion_f3(
    A_t: np.ndarray,
    P_t: np.ndarray,
    G_t: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> np.ndarray:
    """Candidate F3: Residual Joint Temporal + Relational Boost."""
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    raw = A_t + alpha * P_t * (1.0 - A_t) + beta * G_t * (1.0 - A_t)
    return np.clip(raw, 0.0, 1.0)


def compute_fusion_f4(
    A_t: np.ndarray,
    P_t: np.ndarray,
    G_t: np.ndarray,
    cP: float = 1.0,
    cG: float = 1.0,
) -> np.ndarray:
    """Candidate F4: Disjunctive Risk Envelope."""
    if cP < 0 or cG < 0:
        raise ValueError("cP and cG must be non-negative")
    raw = np.maximum.reduce([A_t, cP * P_t, cG * G_t])
    return np.clip(raw, 0.0, 1.0)


FUSION_RULES = {
    "F1": compute_fusion_f1,
    "F2": compute_fusion_f2,
    "F3": compute_fusion_f3,
    "F4": compute_fusion_f4,
}


def apply_fusion_rule(
    rule_name: str,
    A_t: np.ndarray,
    P_t: np.ndarray,
    G_t: np.ndarray,
    params: Dict[str, float],
) -> np.ndarray:
    """Apply named fusion rule with parameter dictionary."""
    if rule_name not in FUSION_RULES:
        raise ValueError(f"Unknown fusion rule: {rule_name}. Choose from {list(FUSION_RULES.keys())}")
    fn = FUSION_RULES[rule_name]
    return fn(A_t, P_t, G_t, **params)


def verify_fusion_invariance(
    A_t: np.ndarray,
    P_t: np.ndarray,
    G_t: np.ndarray,
    R_t: np.ndarray,
    atol: float = 1e-6,
) -> Tuple[bool, Dict[str, Union[bool, float, int]]]:
    """
    Formally verify the critical non-suppression and missing-context invariants.

    Returns
    -------
    passed : bool (True if all invariants hold)
    diagnostics : dict with violation counts and max absolute deviations
    """
    # 1. Missing context invariance: P_t == 0 and G_t == 0 => R_t == A_t
    zero_mask = (P_t == 0.0) & (G_t == 0.0)
    zero_context_count = int(zero_mask.sum())
    if zero_context_count > 0:
        zero_diff = np.abs(R_t[zero_mask] - A_t[zero_mask])
        max_zero_diff = float(np.max(zero_diff))
        zero_invariance_passed = bool(max_zero_diff <= atol)
    else:
        max_zero_diff = 0.0
        zero_invariance_passed = True

    # 2. Non-suppression: R_t >= A_t everywhere
    suppression_diff = A_t - R_t
    violations = int((suppression_diff > atol).sum())
    max_suppression = float(np.max(suppression_diff)) if len(suppression_diff) > 0 else 0.0
    non_suppression_passed = bool(violations == 0)

    # 3. Boundedness: 0 <= R_t <= 1
    bounded_passed = bool(np.all(R_t >= -atol) and np.all(R_t <= 1.0 + atol))

    all_passed = zero_invariance_passed and non_suppression_passed and bounded_passed

    diagnostics = {
        "all_invariants_passed": all_passed,
        "zero_context_invariance_passed": zero_invariance_passed,
        "max_zero_context_deviation": max_zero_diff,
        "zero_context_transactions": zero_context_count,
        "non_suppression_passed": non_suppression_passed,
        "suppression_violations": violations,
        "max_suppression_amount": max(0.0, max_suppression),
        "boundedness_passed": bounded_passed,
        "min_R_t": float(np.min(R_t)),
        "max_R_t": float(np.max(R_t)),
    }
    return all_passed, diagnostics
