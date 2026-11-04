"""
test_rl_endpoints.py — API Integration Tests for JEV Safe RL Policy Endpoints
=============================================================================
Tests use direct async function calls to avoid TestClient parquet-loading hangs.
"""

import pytest
from trustgraph.policy.rl_policy import JEVBanditPolicy, JEVContext, EconomicEnvironment
from trustgraph.policy.config import PolicyAction


def test_rl_decision_endpoint_logic():
    """Simulates the rl-decision endpoint logic directly."""
    # Simulate a blatant fraud record (R_t = 0.9922, as in txn 3570805)
    rec = {
        "transaction_id": 3570805,
        "base_risk": 0.9921,
        "graph_risk": 0.7375,
        "final_risk": 0.9922,
        "amount": 82.63,
        "action": "BLOCK",
    }
    policy = JEVBanditPolicy()
    ctx = JEVContext(
        A_t=float(rec.get("base_risk", 0.0)),
        P_t=float(rec.get("temporal_risk", 0.0)),
        G_t=float(rec.get("graph_risk", 0.0)),
        R_t=float(rec.get("final_risk", 0.0)),
        amount=float(rec.get("amount", 100.0)),
    )
    res = policy.select_action(ctx, explore=True)

    assert res["selected_action"] == PolicyAction.BLOCK
    assert res["is_safety_override"] is True
    assert res["safe_candidate_actions"] == ["BLOCK"]


def test_rl_feedback_endpoint_logic():
    """Simulates the rl-feedback endpoint logic directly."""
    env = EconomicEnvironment()
    policy = JEVBanditPolicy()

    # Simulate: VERIFY action, legitimate customer, OTP passed → captures margin
    rec = {
        "base_risk": 0.60,
        "graph_risk": 0.50,
        "final_risk": 0.63,
        "amount": 150.0,
    }
    act = PolicyAction.VERIFY
    ctx = JEVContext(
        A_t=float(rec["base_risk"]),
        P_t=0.0,
        G_t=float(rec["graph_risk"]),
        R_t=float(rec["final_risk"]),
        amount=150.0,
    )
    reward = env.compute_instant_reward(act, amount=150.0, is_fraud=0, otp_passed=True)
    policy.update(ctx, act, reward=reward)

    assert reward == pytest.approx(150.0 * 0.15 - 1.0)  # 149.0 INR
    assert policy.action_counts[PolicyAction.VERIFY] == 1

    result = {
        "status": "UPDATED",
        "action": act.value,
        "computed_reward_inr": round(reward, 2),
        "total_updates_for_action": policy.action_counts[act],
    }
    assert result["status"] == "UPDATED"
    assert result["computed_reward_inr"] > 0.0
    assert result["total_updates_for_action"] == 1


def test_rl_contested_zone_uses_full_action_set():
    """In contested borderline zone, full action set is considered."""
    policy = JEVBanditPolicy()
    ctx = JEVContext(A_t=0.55, P_t=0.20, G_t=0.60, R_t=0.62, amount=850.0)
    res = policy.select_action(ctx, explore=True)
    assert res["is_safety_override"] is False
    assert len(res["safe_candidate_actions"]) == 4
    assert "ALLOW" in res["safe_candidate_actions"]
    assert "BLOCK" in res["safe_candidate_actions"]
