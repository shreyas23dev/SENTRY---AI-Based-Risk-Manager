import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.policy.config import PolicyAction, RiskBand
from trustgraph.policy.decision_engine import (
    PolicyThresholds,
    assign_action_and_band,
    batch_assign_actions,
    generate_explanation,
    verify_policy_invariants,
)
from trustgraph.policy.evaluator import evaluate_policy


@pytest.fixture
def sample_thresholds():
    return PolicyThresholds(
        tau_verify=0.55,
        tau_throttle=0.70,
        tau_block=0.85,
    )


def test_threshold_ordering_validation():
    """Verify that invalid threshold orderings raise ValueError."""
    # Valid
    t = PolicyThresholds(tau_verify=0.50, tau_throttle=0.70, tau_block=0.90)
    assert t.tau_verify == 0.50

    # Inverted verify >= throttle
    with pytest.raises(ValueError):
        PolicyThresholds(tau_verify=0.70, tau_throttle=0.60, tau_block=0.90)

    # Inverted throttle >= block
    with pytest.raises(ValueError):
        PolicyThresholds(tau_verify=0.50, tau_throttle=0.90, tau_block=0.80)

    # Equal boundaries
    with pytest.raises(ValueError):
        PolicyThresholds(tau_verify=0.50, tau_throttle=0.50, tau_block=0.80)

    # Out of range [0, 1]
    with pytest.raises(ValueError):
        PolicyThresholds(tau_verify=-0.1, tau_throttle=0.5, tau_block=0.8)

    with pytest.raises(ValueError):
        PolicyThresholds(tau_verify=0.5, tau_throttle=0.8, tau_block=1.1)


def test_action_boundaries_and_exact_transitions(sample_thresholds):
    """Test exact behavior at and around boundary points."""
    # Sub-threshold -> ALLOW
    act, band = assign_action_and_band(0.0, sample_thresholds)
    assert act == PolicyAction.ALLOW
    assert band == RiskBand.LOW

    act, band = assign_action_and_band(0.5499, sample_thresholds)
    assert act == PolicyAction.ALLOW
    assert band == RiskBand.LOW

    # Exact boundary tau_verify -> VERIFY
    act, band = assign_action_and_band(0.55, sample_thresholds)
    assert act == PolicyAction.VERIFY
    assert band == RiskBand.MODERATE

    act, band = assign_action_and_band(0.6999, sample_thresholds)
    assert act == PolicyAction.VERIFY
    assert band == RiskBand.MODERATE

    # Exact boundary tau_throttle -> THROTTLE
    act, band = assign_action_and_band(0.70, sample_thresholds)
    assert act == PolicyAction.THROTTLE
    assert band == RiskBand.HIGH

    act, band = assign_action_and_band(0.8499, sample_thresholds)
    assert act == PolicyAction.THROTTLE
    assert band == RiskBand.HIGH

    # Exact boundary tau_block -> BLOCK
    act, band = assign_action_and_band(0.85, sample_thresholds)
    assert act == PolicyAction.BLOCK
    assert band == RiskBand.VERY_HIGH

    # Extreme value 1.0 -> BLOCK
    act, band = assign_action_and_band(1.0, sample_thresholds)
    assert act == PolicyAction.BLOCK
    assert band == RiskBand.VERY_HIGH


def test_batch_assignment_consistency(sample_thresholds):
    """Ensure batch assignment matches single-instance assignment identically."""
    r_vals = np.linspace(0.0, 1.0, 1001)
    actions, bands = batch_assign_actions(r_vals, sample_thresholds)

    assert len(actions) == 1001
    assert len(bands) == 1001

    for i, r in enumerate(r_vals):
        exp_act, exp_band = assign_action_and_band(float(r), sample_thresholds)
        assert actions[i] == exp_act.value
        assert bands[i] == exp_band.value


def test_monotonicity_invariant(sample_thresholds):
    """Verify that increasing risk score never produces a less severe action."""
    r_vals = np.sort(np.random.uniform(0.0, 1.0, size=5000))
    actions, _ = batch_assign_actions(r_vals, sample_thresholds)

    passed, diag = verify_policy_invariants(r_vals, actions, sample_thresholds)
    assert passed, f"Policy invariance check failed: {diag}"
    assert diag["monotonicity_violations"] == 0
    assert diag["range_violations"] == 0


def test_context_free_transaction_handling(sample_thresholds):
    """Verify that a transaction with zero contextual evidence (P=0, G=0, R=A) behaves correctly."""
    A_t = 0.58
    P_t = 0.0
    G_t = 0.0
    R_t = A_t  # zero-context invariance

    act, band = assign_action_and_band(R_t, sample_thresholds)
    assert act == PolicyAction.VERIFY
    assert band == RiskBand.MODERATE

    exp = generate_explanation(A_t, P_t, G_t, R_t, act, sample_thresholds)
    assert "VERIFY" in exp
    assert "0.5800" in exp
    assert len(exp) > 20


def test_explanation_generation_clarity(sample_thresholds):
    """Test that all 4 actions produce distinct, meaningful audit explanations."""
    # ALLOW
    exp_allow = generate_explanation(0.10, 0.0, 0.0, 0.10, PolicyAction.ALLOW, sample_thresholds)
    assert "ALLOW" in exp_allow

    # VERIFY
    exp_verify = generate_explanation(0.56, 0.0, 0.2, 0.57, PolicyAction.VERIFY, sample_thresholds, d_t=2)
    assert "VERIFY" in exp_verify
    assert "connected_entities=2" in exp_verify

    # THROTTLE
    exp_throttle = generate_explanation(0.30, 0.45, 0.0, 0.75, PolicyAction.THROTTLE, sample_thresholds)
    assert "THROTTLE" in exp_throttle
    assert "longitudinal velocity" in exp_throttle

    # BLOCK
    exp_block = generate_explanation(0.88, 0.0, 0.0, 0.88, PolicyAction.BLOCK, sample_thresholds)
    assert "BLOCK" in exp_block
    assert "severe" in exp_block


def test_evaluator_metrics_integrity(sample_thresholds):
    """Test policy evaluation math and action metrics."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # 2 ALLOW, 2 VERIFY, 2 THROTTLE, 2 BLOCK
    r_vals = np.array([0.1, 0.2, 0.56, 0.60, 0.72, 0.78, 0.88, 0.95])
    actions, _ = batch_assign_actions(r_vals, sample_thresholds)

    eval_res = evaluate_policy(y_true, actions, sample_thresholds)
    assert eval_res["total_transactions"] == 8
    assert eval_res["total_frauds"] == 4
    assert eval_res["actions"]["ALLOW"]["transaction_count"] == 2
    assert eval_res["actions"]["ALLOW"]["fraud_count"] == 0
    assert eval_res["actions"]["BLOCK"]["transaction_count"] == 2
    assert eval_res["actions"]["BLOCK"]["fraud_count"] == 2
    assert eval_res["actions"]["BLOCK"]["fraud_rate"] == 1.0
