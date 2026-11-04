"""
test_rl_policy.py — Unit Tests for Joint Expected Value (JEV) Safe RL Policy Engine
===================================================================================
"""

import time
import tempfile
from pathlib import Path
import numpy as np
import pytest

from trustgraph.policy.config import PolicyAction, RiskBand
from trustgraph.policy.rl_policy import (
    EconomicEnvironment,
    JEVContext,
    JEVBanditPolicy,
)


def test_economic_environment_rewards():
    env = EconomicEnvironment(
        gross_margin_rate=0.15,
        chargeback_fee=250.0,
        customer_ltv_churn=1200.0,
    )
    amt = 1000.0

    # 1. ALLOW clean txn -> captures 15% margin
    r_clean = env.compute_instant_reward(PolicyAction.ALLOW, amount=amt, is_fraud=0)
    assert r_clean == pytest.approx(150.0)

    # 2. ALLOW fraudulent txn -> loses amount + chargeback fee
    r_fraud = env.compute_instant_reward(PolicyAction.ALLOW, amount=amt, is_fraud=1)
    assert r_fraud == pytest.approx(-1250.0)

    # 3. BLOCK fraud -> stops loss
    r_block_fraud = env.compute_instant_reward(PolicyAction.BLOCK, amount=amt, is_fraud=1)
    assert r_block_fraud == 0.0

    # 4. BLOCK clean txn (False positive) -> loses margin + customer LTV
    r_block_clean = env.compute_instant_reward(PolicyAction.BLOCK, amount=amt, is_fraud=0)
    assert r_block_clean == pytest.approx(-(150.0 + 1200.0))

    # 5. VERIFY clean txn with OTP passed -> captures margin minus small gateway fee
    r_verify_pass = env.compute_instant_reward(PolicyAction.VERIFY, amount=amt, is_fraud=0, otp_passed=True)
    assert r_verify_pass == pytest.approx(149.0)

    # 6. VERIFY clean txn with customer abandonment -> loses margin
    r_verify_fail = env.compute_instant_reward(PolicyAction.VERIFY, amount=amt, is_fraud=0, otp_passed=False)
    assert r_verify_fail == pytest.approx(-150.0)


def test_safe_rl_boundaries():
    policy = JEVBanditPolicy()

    # Blatant fraud -> Only BLOCK allowed (no random exploration!)
    ctx_blatant = JEVContext(A_t=0.95, P_t=0.40, G_t=0.80, R_t=0.96, amount=500.0)
    res_blatant = policy.select_action(ctx_blatant, explore=True)
    assert res_blatant["selected_action"] == PolicyAction.BLOCK
    assert res_blatant["safe_candidate_actions"] == ["BLOCK"]
    assert res_blatant["is_safety_override"] is True

    # Pure clean customer -> Only ALLOW
    ctx_clean = JEVContext(A_t=0.01, P_t=0.0, G_t=0.0, R_t=0.01, amount=100.0)
    res_clean = policy.select_action(ctx_clean, explore=True)
    assert res_clean["selected_action"] == PolicyAction.ALLOW
    assert res_clean["safe_candidate_actions"] == ["ALLOW"]
    assert res_clean["is_safety_override"] is True

    # Contested borderline zone (R_t = 0.62) -> Full exploration over safe set
    ctx_borderline = JEVContext(A_t=0.55, P_t=0.20, G_t=0.60, R_t=0.62, amount=850.0)
    res_borderline = policy.select_action(ctx_borderline, explore=True)
    assert res_borderline["is_safety_override"] is False
    assert len(res_borderline["safe_candidate_actions"]) == 4


def test_online_update_and_learning():
    policy = JEVBanditPolicy(alpha=0.2)
    ctx = JEVContext(A_t=0.60, P_t=0.25, G_t=0.50, R_t=0.63, amount=1200.0)

    # Initially all actions have 0 updates
    assert policy.action_counts[PolicyAction.VERIFY] == 0

    # Simulate 5 positive VERIFY outcomes (successful 3DS authentication)
    for _ in range(5):
        policy.update(ctx, PolicyAction.VERIFY, reward=180.0)

    assert policy.action_counts[PolicyAction.VERIFY] == 5

    # VERIFY score should now be strongly elevated for similar contexts
    evals = policy.select_action(ctx, explore=False)
    assert evals["action_evaluations"]["VERIFY"]["mean_expected_reward"] > 0.0


def test_sub_millisecond_inference_latency():
    policy = JEVBanditPolicy()
    ctx = JEVContext(A_t=0.45, P_t=0.15, G_t=0.30, R_t=0.48, amount=500.0)

    # Warmup
    for _ in range(10):
        policy.select_action(ctx)

    # Benchmark 1,000 action selections
    t0 = time.perf_counter()
    n_iters = 1000
    for _ in range(n_iters):
        policy.select_action(ctx)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_call_ms = elapsed_ms / n_iters

    # Must be well below 0.5ms (usually ~0.02ms)
    assert per_call_ms < 0.5, f"Expected <0.5ms per call, got {per_call_ms:.4f}ms"


def test_policy_persistence():
    policy = JEVBanditPolicy(alpha=0.4)
    ctx = JEVContext(A_t=0.70, P_t=0.30, G_t=0.65, R_t=0.74, amount=2000.0)
    policy.update(ctx, PolicyAction.BLOCK, reward=0.0)
    policy.update(ctx, PolicyAction.ALLOW, reward=-2250.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_jev_policy.json"
        policy.save(save_path)
        assert save_path.exists()

        new_policy = JEVBanditPolicy()
        new_policy.load(save_path)
        assert new_policy.action_counts[PolicyAction.BLOCK] == 1
        assert new_policy.action_counts[PolicyAction.ALLOW] == 1
        assert new_policy.alpha == pytest.approx(0.4)
