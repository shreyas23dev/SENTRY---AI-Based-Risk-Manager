"""
test_fusion.py — Unit Tests for TRUSTGRAPH Phase 3.1 Conditional Risk Fusion
=============================================================================

Tests:
  1. F1, F2, F3, F4 mathematical correctness against known manual calculations
  2. Missing-context invariance: P_t = 0, G_t = 0 => R_t == A_t for all rules
  3. Non-suppression: R_t >= A_t everywhere for all candidate rules
  4. Boundedness: 0.0 <= R_t <= 1.0 everywhere
  5. Negative parameters raise ValueError
  6. verify_fusion_invariance returns True on valid fusion, False on suppressive rule
  7. Coverage-aware metrics accurately partition transactions without missing rows
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.fusion.fusion_engine import (
    compute_fusion_f1,
    compute_fusion_f2,
    compute_fusion_f3,
    compute_fusion_f4,
    apply_fusion_rule,
    verify_fusion_invariance,
)
from trustgraph.fusion.evaluator import compute_system_metrics, compute_coverage_aware_metrics


def test_f1_mathematics():
    A = np.array([0.2, 0.5, 0.8])
    P = np.array([0.0, 0.5, 0.4])
    G = np.array([0.0, 0.2, 0.3])
    # alpha=0.5, beta=0.5
    # i=0: 0.2 + 0 + 0 = 0.2
    # i=1: 0.5 + 0.25 + 0.1 = 0.85
    # i=2: 0.8 + 0.2 + 0.15 = 1.15 -> clip to 1.0
    R = compute_fusion_f1(A, P, G, alpha=0.5, beta=0.5)
    np.testing.assert_allclose(R, [0.2, 0.85, 1.0], atol=1e-6)


def test_f2_mathematics():
    A = np.array([0.2, 0.5, 0.8])
    P = np.array([0.0, 0.5, 0.4])
    G = np.array([0.0, 0.5, 0.5])
    # beta=0.5
    # i=0: 0.2 + 0.5 * 0 * 0.8 = 0.2
    # i=1: 0.5 + 0.5 * 0.5 * 0.5 = 0.5 + 0.125 = 0.625
    # i=2: 0.8 + 0.5 * 0.5 * 0.2 = 0.8 + 0.05 = 0.85
    R = compute_fusion_f2(A, P, G, beta=0.5)
    np.testing.assert_allclose(R, [0.2, 0.625, 0.85], atol=1e-6)


def test_f3_mathematics():
    A = np.array([0.2, 0.5, 0.8])
    P = np.array([0.0, 0.2, 0.5])
    G = np.array([0.0, 0.4, 0.5])
    # alpha=0.5, beta=0.5
    # i=0: 0.2 + 0 = 0.2
    # i=1: 0.5 + (0.5*0.2 + 0.5*0.4)*(1 - 0.5) = 0.5 + 0.3 * 0.5 = 0.65
    # i=2: 0.8 + (0.5*0.5 + 0.5*0.5)*(1 - 0.8) = 0.8 + 0.5 * 0.2 = 0.90
    R = compute_fusion_f3(A, P, G, alpha=0.5, beta=0.5)
    np.testing.assert_allclose(R, [0.2, 0.65, 0.90], atol=1e-6)


def test_f4_mathematics():
    A = np.array([0.3, 0.5, 0.8])
    P = np.array([0.0, 0.7, 0.2])
    G = np.array([0.0, 0.4, 0.9])
    # cP=1.0, cG=1.0
    # i=0: max(0.3, 0, 0) = 0.3
    # i=1: max(0.5, 0.7, 0.4) = 0.7
    # i=2: max(0.8, 0.2, 0.9) = 0.9
    R = compute_fusion_f4(A, P, G, cP=1.0, cG=1.0)
    np.testing.assert_allclose(R, [0.3, 0.7, 0.9], atol=1e-6)


@pytest.mark.parametrize("rule", ["F1", "F2", "F3", "F4"])
def test_missing_context_invariance(rule):
    """When P_t = 0 and G_t = 0, R_t MUST equal A_t exactly."""
    rng = np.random.default_rng(42)
    N = 500
    A = rng.uniform(0.0, 1.0, N)
    P = np.zeros(N)
    G = np.zeros(N)

    params = {"alpha": 0.5, "beta": 0.5} if rule in ["F1", "F3"] else (
             {"beta": 0.5} if rule == "F2" else {"cP": 0.8, "cG": 0.8})

    R = apply_fusion_rule(rule, A, P, G, params)
    np.testing.assert_allclose(R, A, atol=1e-6)


@pytest.mark.parametrize("rule", ["F1", "F2", "F3", "F4"])
def test_non_suppression_invariant(rule):
    """R_t must be >= A_t everywhere for all non-negative parameter settings."""
    rng = np.random.default_rng(42)
    N = 1000
    A = rng.uniform(0.0, 1.0, N)
    P = rng.uniform(0.0, 1.0, N)
    G = rng.uniform(0.0, 1.0, N)

    params = {"alpha": 0.3, "beta": 0.4} if rule in ["F1", "F3"] else (
             {"beta": 0.4} if rule == "F2" else {"cP": 0.9, "cG": 0.9})

    R = apply_fusion_rule(rule, A, P, G, params)
    assert np.all(R >= A - 1e-7), f"Rule {rule} violated R_t >= A_t"
    assert np.all(R <= 1.0 + 1e-7), f"Rule {rule} exceeded 1.0"
    assert np.all(R >= 0.0), f"Rule {rule} below 0.0"

    passed, diag = verify_fusion_invariance(A, P, G, R)
    assert passed, f"Verification failed: {diag}"


def test_invariance_verifier_catches_suppressive_rule():
    """Verify that verify_fusion_invariance correctly flags a suppressive rule (like old B3)."""
    A = np.array([0.8, 0.5])
    P = np.array([0.0, 0.0])
    G = np.array([0.0, 0.0])
    # Suppressive old B3 formula
    R_suppressive = 0.4 * A + 0.3 * P + 0.3 * G  # = [0.32, 0.20] < A
    passed, diag = verify_fusion_invariance(A, P, G, R_suppressive)
    assert not passed
    assert not diag["zero_context_invariance_passed"]
    assert not diag["non_suppression_passed"]
    assert diag["suppression_violations"] == 2


def test_negative_parameters_rejected():
    A = np.array([0.5])
    P = np.array([0.5])
    G = np.array([0.5])
    with pytest.raises(ValueError):
        compute_fusion_f1(A, P, G, alpha=-0.1, beta=0.5)
    with pytest.raises(ValueError):
        compute_fusion_f2(A, P, G, beta=-0.5)
    with pytest.raises(ValueError):
        compute_fusion_f3(A, P, G, alpha=0.5, beta=-0.2)
    with pytest.raises(ValueError):
        compute_fusion_f4(A, P, G, cP=-1.0, cG=1.0)


def test_coverage_aware_metrics_partition():
    N = 100
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, N)
    A_t = rng.uniform(0.0, 1.0, N)
    P_t = np.zeros(N)
    P_t[:20] = rng.uniform(0.1, 1.0, 20)
    G_t = np.zeros(N)
    G_t[15:35] = rng.uniform(0.1, 1.0, 20)

    R_t = compute_fusion_f1(A_t, P_t, G_t, alpha=0.3, beta=0.3)
    b0_pred = (A_t >= 0.5).astype(int)
    b3_pred = (R_t >= 0.5).astype(int)

    cov = compute_coverage_aware_metrics(y_true, A_t, P_t, G_t, R_t, b0_pred, b3_pred)
    assert "overall" in cov
    assert "relational_zero_Gt" in cov
    assert "relational_active_Gt" in cov
    assert "temporal_zero_Pt" in cov
    assert "temporal_active_Pt" in cov
    assert "uncontextualized_zero_both" in cov
    assert "contextualized_any_active" in cov

    # Overall count matches N
    assert cov["overall"]["transaction_count"] == N
    # Zero Gt + Active Gt sums to N
    assert (cov["relational_zero_Gt"]["transaction_count"] +
            cov["relational_active_Gt"]["transaction_count"]) == N
    # Uncontextualized + Contextualized sums to N
    assert (cov["uncontextualized_zero_both"]["transaction_count"] +
            cov["contextualized_any_active"]["transaction_count"]) == N
