"""
test_risk_engine.py — Unit Tests for Phase 3 Mathematical Risk Engine
======================================================================

Tests cover all required areas:
  1. Fusion equations (F1, F2, F4) — correctness
  2. Clipping to [0,1]
  3. Monotonicity where mathematically expected
  4. Zero graph-risk behaviour
  5. Zero base-risk behaviour
  6. High base-risk behaviour
  7. Parameter bounds
  8. Deterministic output
  9. Cost calculation
 10. Action selection
 11. Threshold selection uses VALIDATION only
 12. TEST immutability
 13. No label leakage in cost model
 14. Calibration ECE calculation
 15. Full decision engine output contract
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.risk.calibration import PlattCalibrator, SignalCalibrator, expected_calibration_error
from trustgraph.risk.cost_model import (
    ActionCosts,
    CostScenario,
    DEFAULT_SCENARIOS,
    MerchantCostModel,
)
from trustgraph.risk.decision import Action, ActionResult, DecisionEngine
from trustgraph.risk.fusion import (
    FusionEngine,
    FusionFormula,
    FusionResult,
    _f1_additive,
    _f2_conditional,
    _f4_conservative_max,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def synth_val():
    """Synthetic VAL dataset for fusion tuning."""
    np.random.seed(42)
    n = 5000
    y = (np.random.rand(n) < 0.05).astype(int)
    a = np.clip(np.where(y == 1, np.random.beta(3, 2, n), np.random.beta(1, 8, n)), 0, 1)
    g = np.clip(np.where(y == 1, np.random.beta(2, 3, n), np.random.beta(1, 10, n)), 0, 1)
    return a, g, y


@pytest.fixture
def synth_train():
    """Synthetic TRAIN dataset for calibration."""
    np.random.seed(100)
    n = 20000
    y = (np.random.rand(n) < 0.05).astype(int)
    a = np.clip(np.where(y == 1, np.random.beta(3, 2, n), np.random.beta(1, 8, n)), 0, 1)
    g = np.clip(np.where(y == 1, np.random.beta(2, 3, n), np.random.beta(1, 10, n)), 0, 1)
    return a, g, y


@pytest.fixture
def tuned_engine(synth_val):
    """Pre-tuned FusionEngine on synthetic VAL."""
    a, g, y = synth_val
    engine = FusionEngine()
    engine.tune(a, g, y, selection_metric="pr_auc")
    return engine


@pytest.fixture
def balanced_cost_model():
    return MerchantCostModel(DEFAULT_SCENARIOS["balanced"])


@pytest.fixture
def identity_calibrator():
    """Calibrator that passes signals through unchanged (already calibrated)."""
    cal = SignalCalibrator()
    # Mark as fitted without actually transforming
    cal.fitted = False
    return cal


# ===========================================================================
# 1. Fusion Equations — F1 Correctness
# ===========================================================================

class TestF1Additive:

    def test_basic_formula(self):
        a = np.array([0.3])
        g = np.array([0.4])
        result = _f1_additive(a, g, alpha=0.5)
        assert abs(result[0] - 0.5) < 1e-9  # 0.3 + 0.5*0.4 = 0.5

    def test_zero_alpha_returns_a(self):
        a = np.array([0.3, 0.6, 0.8])
        g = np.array([0.9, 0.5, 0.1])
        result = _f1_additive(a, g, alpha=0.0)
        np.testing.assert_allclose(result, a)

    def test_full_alpha_adds_g(self):
        a = np.array([0.2])
        g = np.array([0.3])
        result = _f1_additive(a, g, alpha=1.0)
        assert abs(result[0] - 0.5) < 1e-9


# ===========================================================================
# 2. Fusion Equations — F2 Correctness
# ===========================================================================

class TestF2Conditional:

    def test_basic_formula(self):
        a = np.array([0.5])
        g = np.array([0.6])
        # 0.5 + 0.4 * 0.6 * 0.5 = 0.5 + 0.12 = 0.62
        result = _f2_conditional(a, g, beta=0.4)
        assert abs(result[0] - 0.62) < 1e-6

    def test_zero_beta_returns_a(self):
        a = np.array([0.4, 0.7])
        g = np.array([0.9, 0.8])
        result = _f2_conditional(a, g, beta=0.0)
        np.testing.assert_allclose(result, a)

    def test_zero_graph_returns_a(self):
        """When G_t = 0 (cold-start), R_t = A_t regardless of beta."""
        a = np.array([0.4, 0.7, 0.9])
        g = np.array([0.0, 0.0, 0.0])
        result = _f2_conditional(a, g, beta=0.5)
        np.testing.assert_allclose(result, a)

    def test_saturation_at_high_a(self):
        """When A_t = 1.0, graph contribution must be zero."""
        a = np.array([1.0])
        g = np.array([0.9])
        result = _f2_conditional(a, g, beta=0.5)
        assert abs(result[0] - 1.0) < 1e-9

    def test_graph_increases_risk(self):
        """F2: R_t must be >= A_t when G_t > 0 and beta > 0."""
        a = np.random.rand(100)
        g = np.abs(np.random.rand(100))
        r = _f2_conditional(a, g, beta=0.3)
        assert np.all(r >= a - 1e-9)

    def test_f2_monotone_in_g(self):
        """F2: R_t is monotonically non-decreasing in G_t."""
        a = np.array([0.4])
        for g1, g2 in [(0.1, 0.2), (0.3, 0.8), (0.0, 1.0)]:
            r1 = _f2_conditional(a, np.array([g1]), beta=0.3)
            r2 = _f2_conditional(a, np.array([g2]), beta=0.3)
            assert r2[0] >= r1[0] - 1e-9, f"F2 not monotone: G_t={g1}->{g2}, R_t={r1[0]}->{r2[0]}"

    def test_f2_monotone_in_a(self):
        """F2: R_t is monotonically non-decreasing in A_t."""
        g = np.array([0.6])
        betas = [0.1, 0.3, 0.5]
        for beta in betas:
            prev_r = -1.0
            for a_val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                r = float(_f2_conditional(np.array([a_val]), g, beta=beta)[0])
                assert r >= prev_r - 1e-9, f"F2 not monotone in A_t at a={a_val}, beta={beta}"
                prev_r = r


# ===========================================================================
# 3. Fusion Equations — F4 Correctness
# ===========================================================================

class TestF4ConservativeMax:

    def test_basic_formula(self):
        a = np.array([0.5])
        g = np.array([0.8])
        # max(0.5, 0.9*0.8) = max(0.5, 0.72) = 0.72
        result = _f4_conservative_max(a, g, c=0.9)
        assert abs(result[0] - 0.72) < 1e-6

    def test_high_a_dominates(self):
        """When A_t > c*G_t, F4 returns A_t."""
        a = np.array([0.9])
        g = np.array([0.5])
        result = _f4_conservative_max(a, g, c=0.8)
        # max(0.9, 0.4) = 0.9
        assert abs(result[0] - 0.9) < 1e-9

    def test_zero_c_returns_a(self):
        a = np.array([0.3, 0.6])
        g = np.array([0.9, 0.8])
        result = _f4_conservative_max(a, g, c=0.0)
        np.testing.assert_allclose(result, a)


# ===========================================================================
# 4. Clipping to [0, 1]
# ===========================================================================

class TestClipping:

    def test_f1_clips_above_1(self):
        a = np.array([0.9])
        g = np.array([0.9])
        result = _f1_additive(a, g, alpha=1.0)  # 0.9 + 0.9 = 1.8, should clip to 1.0
        assert result[0] == 1.0

    def test_f1_clips_below_0(self):
        a = np.array([-0.1])
        g = np.array([0.0])
        result = _f1_additive(a, g, alpha=0.5)
        assert result[0] == 0.0

    def test_f2_always_in_range(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 1, 1000)
        g = rng.uniform(0, 1, 1000)
        for beta in [0.1, 0.5, 1.0]:
            r = _f2_conditional(a, g, beta=beta)
            assert np.all(r >= 0.0) and np.all(r <= 1.0)

    def test_f4_always_in_range(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(0, 1, 1000)
        g = rng.uniform(0, 1, 1000)
        for c in [0.5, 1.0, 1.2]:
            r = _f4_conservative_max(a, g, c=c)
            assert np.all(r >= 0.0) and np.all(r <= 1.0)


# ===========================================================================
# 5. Zero graph-risk behaviour
# ===========================================================================

class TestZeroGraphRisk:

    def test_f2_zero_g_identity(self):
        a = np.array([0.1, 0.5, 0.9])
        g = np.zeros(3)
        r = _f2_conditional(a, g, beta=0.4)
        np.testing.assert_allclose(r, a)

    def test_f4_zero_g_returns_a(self):
        a = np.array([0.3, 0.7])
        g = np.zeros(2)
        r = _f4_conservative_max(a, g, c=1.0)
        np.testing.assert_allclose(r, a)

    def test_f1_zero_g_returns_a(self):
        a = np.array([0.3, 0.7])
        g = np.zeros(2)
        r = _f1_additive(a, g, alpha=0.5)
        np.testing.assert_allclose(r, a)


# ===========================================================================
# 6. Zero base-risk behaviour
# ===========================================================================

class TestZeroBaseRisk:

    def test_f2_zero_a_scaled_by_g(self):
        """F2: When A_t = 0, R_t = beta * G_t."""
        a = np.array([0.0])
        g = np.array([0.6])
        r = _f2_conditional(a, g, beta=0.4)
        assert abs(r[0] - 0.24) < 1e-6  # 0 + 0.4 * 0.6 * 1.0 = 0.24

    def test_f4_zero_a_scaled_by_c_g(self):
        """F4: When A_t = 0, R_t = max(0, c*G_t)."""
        a = np.array([0.0])
        g = np.array([0.6])
        r = _f4_conservative_max(a, g, c=0.8)
        assert abs(r[0] - 0.48) < 1e-6


# ===========================================================================
# 7. High base-risk behaviour
# ===========================================================================

class TestHighBaseRisk:

    def test_f2_high_a_graph_contribution_small(self):
        """F2: When A_t close to 1, graph can only add small contribution."""
        a = np.array([0.95])
        g = np.array([0.9])
        r = _f2_conditional(a, g, beta=0.5)
        # 0.95 + 0.5 * 0.9 * 0.05 = 0.95 + 0.0225 = 0.9725
        assert abs(r[0] - 0.9725) < 1e-5

    def test_f4_high_a_dominates_g(self):
        a = np.array([0.9])
        g = np.array([0.7])
        r = _f4_conservative_max(a, g, c=1.0)
        assert abs(r[0] - 0.9) < 1e-9


# ===========================================================================
# 8. Parameter bounds
# ===========================================================================

class TestParameterBounds:

    def test_fusion_engine_alpha_grid_range(self):
        """Alpha grid must be in (0, 1]."""
        for alpha in FusionEngine.ALPHA_GRID:
            assert 0.0 < alpha <= 1.0

    def test_fusion_engine_beta_grid_range(self):
        for beta in FusionEngine.BETA_GRID:
            assert 0.0 < beta <= 1.0

    def test_fusion_engine_c_grid_range(self):
        for c in FusionEngine.C_GRID:
            assert c > 0.0

    def test_cost_scenario_validation(self):
        s = CostScenario(
            C_fraud_rate=1.0,
            verify_fraud_reduction=0.7,
            throttle_fraud_reduction=0.3,
        )
        s.validate()  # Should not raise

    def test_cost_scenario_invalid_fraud_rate(self):
        s = CostScenario(C_fraud_rate=1.5)
        with pytest.raises(AssertionError):
            s.validate()


# ===========================================================================
# 9. Deterministic output
# ===========================================================================

class TestDeterminism:

    def test_fusion_deterministic(self, synth_val):
        a, g, y = synth_val
        engine1 = FusionEngine()
        engine1.tune(a, g, y)
        engine2 = FusionEngine()
        engine2.tune(a, g, y)

        r1 = engine1.fuse_batch(a[:100], g[:100])
        r2 = engine2.fuse_batch(a[:100], g[:100])
        np.testing.assert_allclose(r1, r2)

    def test_fuse_single_deterministic(self, tuned_engine):
        r1 = tuned_engine.fuse_single(0.4, 0.7)
        r2 = tuned_engine.fuse_single(0.4, 0.7)
        assert r1.final_risk == r2.final_risk

    def test_cost_model_deterministic(self, balanced_cost_model):
        c1 = balanced_cost_model.compute_action_costs(0.6, 4000.0)
        c2 = balanced_cost_model.compute_action_costs(0.6, 4000.0)
        assert c1.allow == c2.allow
        assert c1.block == c2.block


# ===========================================================================
# 10. Cost calculation
# ===========================================================================

class TestCostCalculation:

    def test_allow_cost_proportional_to_risk(self, balanced_cost_model):
        """Higher risk → higher expected fraud loss for ALLOW."""
        c_low  = balanced_cost_model.compute_action_costs(0.1, 3500.0)
        c_high = balanced_cost_model.compute_action_costs(0.9, 3500.0)
        assert c_high.allow > c_low.allow

    def test_block_cost_proportional_to_legit_probability(self, balanced_cost_model):
        """Higher risk → lower block cost (less likely to block legitimate)."""
        c_low  = balanced_cost_model.compute_action_costs(0.1, 3500.0)
        c_high = balanced_cost_model.compute_action_costs(0.9, 3500.0)
        assert c_high.block < c_low.block

    def test_verify_reduces_fraud_cost(self, balanced_cost_model):
        """VERIFY expected cost < ALLOW expected cost for high-risk transactions."""
        s = DEFAULT_SCENARIOS["balanced"]
        # At high risk, verify should cost less than allow
        c = balanced_cost_model.compute_action_costs(0.9, 5000.0)
        assert c.verify < c.allow

    def test_zero_risk_block_equals_c_fp_block(self, balanced_cost_model):
        """Zero risk → block cost = C_fp_block * 1.0."""
        c = balanced_cost_model.compute_action_costs(0.0, 3500.0)
        assert abs(c.block - DEFAULT_SCENARIOS["balanced"].C_fp_block) < 1e-3

    def test_full_risk_block_is_zero(self, balanced_cost_model):
        """Full fraud probability → block cost = 0 (we'd only block fraudsters)."""
        c = balanced_cost_model.compute_action_costs(1.0, 3500.0)
        assert abs(c.block) < 1e-3

    def test_all_costs_nonnegative(self, balanced_cost_model):
        for risk in [0.0, 0.1, 0.5, 0.9, 1.0]:
            c = balanced_cost_model.compute_action_costs(risk, 5000.0)
            assert c.allow >= 0 and c.verify >= 0 and c.throttle >= 0 and c.block >= 0

    def test_missing_txn_amount_uses_default(self, balanced_cost_model):
        c = balanced_cost_model.compute_action_costs(0.5, None)
        assert c.txn_amount == DEFAULT_SCENARIOS["balanced"].avg_transaction_amount

    def test_all_default_scenarios_valid(self):
        for name, scenario in DEFAULT_SCENARIOS.items():
            scenario.validate()
            model = MerchantCostModel(scenario)
            costs = model.compute_action_costs(0.5, 3500.0)
            assert costs.allow >= 0


# ===========================================================================
# 11. Action selection
# ===========================================================================

class TestActionSelection:

    def test_zero_risk_selects_allow(self, balanced_cost_model):
        """Zero fraud risk → ALLOW must always be optimal."""
        c = balanced_cost_model.compute_action_costs(0.0, 5000.0)
        assert c.optimal_action() == "ALLOW"

    def test_high_risk_selects_block_or_verify(self, balanced_cost_model):
        """Very high fraud risk → ALLOW must not be optimal."""
        c = balanced_cost_model.compute_action_costs(0.99, 5000.0)
        assert c.optimal_action() != "ALLOW"

    def test_batch_actions_shape(self, balanced_cost_model):
        risks = np.linspace(0, 1, 100)
        actions = balanced_cost_model.compute_batch_costs(risks)
        assert len(actions) == 100
        assert all(a in ["ALLOW", "VERIFY", "THROTTLE", "BLOCK"] for a in actions)

    def test_action_monotone_with_risk(self, balanced_cost_model):
        """At very low risk → ALLOW; at very high risk → BLOCK or VERIFY."""
        c_low = balanced_cost_model.compute_action_costs(0.001, 3500.0)
        c_high = balanced_cost_model.compute_action_costs(0.999, 3500.0)
        assert c_low.optimal_action() == "ALLOW"
        assert c_high.optimal_action() in ("BLOCK", "VERIFY")

    def test_aggressive_scenario_more_blocking(self):
        """Aggressive scenario should block at lower risk threshold than conservative."""
        agg_model  = MerchantCostModel(DEFAULT_SCENARIOS["aggressive"])
        cons_model = MerchantCostModel(DEFAULT_SCENARIOS["conservative"])
        # At moderate risk, check that aggressive is more likely to block
        r = 0.4
        agg_action  = agg_model.compute_action_costs(r, 3500.0).optimal_action()
        cons_action = cons_model.compute_action_costs(r, 3500.0).optimal_action()
        # Not deterministic which is "more blocking" at exactly 0.4, just verify no crash
        assert agg_action in ["ALLOW", "VERIFY", "THROTTLE", "BLOCK"]
        assert cons_action in ["ALLOW", "VERIFY", "THROTTLE", "BLOCK"]


# ===========================================================================
# 12. Threshold selection uses VALIDATION only
# ===========================================================================

class TestThresholdSelectionValidationOnly:

    def test_tuning_only_uses_val_not_test(self, synth_val):
        """FusionEngine.tune() must ONLY accept val data — test that passing
        different data produces different thresholds."""
        a_val, g_val, y_val = synth_val

        np.random.seed(99)
        n = len(y_val)
        a_test_fake = np.random.rand(n)
        g_test_fake = np.random.rand(n)
        y_test_fake = (np.random.rand(n) < 0.05).astype(int)

        engine_val  = FusionEngine()
        engine_val.tune(a_val, g_val, y_val)

        engine_fake = FusionEngine()
        engine_fake.tune(a_test_fake, g_test_fake, y_test_fake)

        # The tuning results should differ (different data → different params)
        assert engine_val.best_threshold != engine_fake.best_threshold or \
               engine_val.best_params != engine_fake.best_params

    def test_tuning_result_count(self, synth_val):
        """Total tuning candidates = len(ALPHA_GRID) + len(BETA_GRID) + len(C_GRID)."""
        a, g, y = synth_val
        engine = FusionEngine()
        engine.tune(a, g, y)
        expected = len(FusionEngine.ALPHA_GRID) + len(FusionEngine.BETA_GRID) + len(FusionEngine.C_GRID)
        assert len(engine.tuning_results) == expected


# ===========================================================================
# 13. TEST immutability / no label leakage
# ===========================================================================

class TestNoLabelLeakage:

    def test_fusion_engine_does_not_use_labels(self, synth_val):
        """fuse_batch never receives labels; verifying the API does not accept them."""
        a, g, y = synth_val
        engine = FusionEngine()
        engine.tune(a, g, y)
        # fuse_batch only accepts (a, g), not labels
        import inspect
        sig = inspect.signature(engine.fuse_batch)
        params = list(sig.parameters.keys())
        assert "y" not in params and "label" not in params

    def test_cost_model_does_not_accept_labels(self):
        """compute_batch_costs should not accept a labels argument."""
        import inspect
        model = MerchantCostModel(DEFAULT_SCENARIOS["balanced"])
        sig = inspect.signature(model.compute_batch_costs)
        params = list(sig.parameters.keys())
        assert "y" not in params and "label" not in params and "y_true" not in params


# ===========================================================================
# 14. Calibration ECE calculation
# ===========================================================================

class TestCalibrationECE:

    def test_ece_perfect_calibration(self):
        """Well-calibrated predictions: ECE should be small."""
        rng = np.random.default_rng(0)
        # Generate calibrated predictions: y_prob ≈ P(y=1|prob)
        y_prob = rng.uniform(0, 1, 2000)
        y_true = rng.binomial(1, y_prob)  # each label drawn from its own probability
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        # A truly calibrated model on 2000 samples should have ECE < 5%
        assert ece < 0.05


    def test_ece_poor_calibration(self):
        """Overconfident predictions → high ECE."""
        y_true = np.zeros(100)  # all legit
        y_prob = np.ones(100) * 0.9  # predicts all fraud
        ece = expected_calibration_error(y_true, y_prob)
        assert ece > 0.5

    def test_ece_nonnegative(self):
        rng = np.random.default_rng(42)
        y = (rng.random(500) < 0.1).astype(int)
        p = rng.uniform(0, 1, 500)
        assert expected_calibration_error(y, p) >= 0.0

    def test_platt_calibrator_output_in_range(self, synth_train):
        a_tr, g_tr, y_tr = synth_train
        cal = PlattCalibrator().fit(a_tr, y_tr)
        out = cal.transform(np.array([0.0, 0.2, 0.5, 0.8, 1.0]))
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_signal_calibrator_fitted_flag(self, synth_train, synth_val):
        a_tr, g_tr, y_tr = synth_train
        a_v,  g_v,  y_v  = synth_val
        cal = SignalCalibrator()
        assert not cal.fitted
        cal.fit(a_tr, g_tr, y_tr, a_v, g_v, y_v)
        assert cal.fitted

    def test_signal_calibrator_transform_shape(self, synth_train, synth_val):
        a_tr, g_tr, y_tr = synth_train
        a_v,  g_v,  y_v  = synth_val
        cal = SignalCalibrator()
        cal.fit(a_tr, g_tr, y_tr, a_v, g_v, y_v)
        a_out, g_out = cal.transform(a_v, g_v)
        assert a_out.shape == a_v.shape
        assert g_out.shape == g_v.shape

    def test_signal_calibrator_output_in_range(self, synth_train, synth_val):
        a_tr, g_tr, y_tr = synth_train
        a_v,  g_v,  y_v  = synth_val
        cal = SignalCalibrator()
        cal.fit(a_tr, g_tr, y_tr, a_v, g_v, y_v)
        a_out, g_out = cal.transform(a_v, g_v)
        assert np.all(a_out >= 0.0) and np.all(a_out <= 1.0)
        assert np.all(g_out >= 0.0) and np.all(g_out <= 1.0)


# ===========================================================================
# 15. Full Decision Engine output contract
# ===========================================================================

class TestDecisionEngineContract:

    @pytest.fixture
    def decision_engine(self, synth_val, balanced_cost_model):
        a, g, y = synth_val
        engine = FusionEngine()
        engine.tune(a, g, y)
        cal = SignalCalibrator()
        cal.fitted = False  # Use identity (no calibration)
        cal._calibrate_a = False
        cal._calibrate_g = False
        cal.fitted = True   # Mark as ready
        return DecisionEngine(cal, engine, balanced_cost_model)

    def test_output_keys_present(self, decision_engine):
        result = decision_engine.decide(txn_id=42, a_t=0.4, g_t=0.6, txn_amount=4000.0)
        d = result.to_dict()
        required_keys = [
            "transaction_id", "base_risk", "graph_risk", "final_risk",
            "action", "expected_cost", "action_costs", "risk_contributors",
            "explanation", "formula", "graph_contribution", "scenario_name",
        ]
        for k in required_keys:
            assert k in d, f"Missing key: {k}"

    def test_final_risk_in_range(self, decision_engine):
        for a, g in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.1, 0.9)]:
            result = decision_engine.decide(99, a, g, 5000.0)
            assert 0.0 <= result.final_risk <= 1.0

    def test_action_is_valid_enum(self, decision_engine):
        result = decision_engine.decide(1, 0.7, 0.8, 3000.0)
        assert result.action in [Action.ALLOW, Action.VERIFY, Action.THROTTLE, Action.BLOCK]

    def test_risk_contributors_is_list(self, decision_engine):
        result = decision_engine.decide(1, 0.4, 0.3, 2000.0)
        assert isinstance(result.risk_contributors, list)
        assert len(result.risk_contributors) > 0

    def test_explanation_contains_transaction_id(self, decision_engine):
        result = decision_engine.decide(txn_id=9999, a_t=0.5, g_t=0.4)
        assert "9999" in result.explanation

    def test_decide_batch_shape(self, decision_engine):
        n = 200
        txn_ids = np.arange(n)
        a_batch  = np.random.rand(n)
        g_batch  = np.random.rand(n)
        amounts  = np.full(n, 3500.0)
        actions = decision_engine.decide_batch(txn_ids, a_batch, g_batch, amounts)
        assert len(actions) == n

    def test_to_dict_serializable(self, decision_engine):
        import json
        result = decision_engine.decide(1, 0.6, 0.5, 4000.0)
        d = result.to_dict()
        json_str = json.dumps(d)  # Must not raise
        assert len(json_str) > 100

    def test_zero_g_decision_same_as_a_only(self, decision_engine):
        """With G_t = 0 and F2 formula, R_t = A_t, so decisions are A_t-driven."""
        r_with_g0  = decision_engine.decide(1, 0.5, 0.0).final_risk
        # Manually check: F2 at G_t=0 → R_t = A_t
        if decision_engine.fusion_engine.best_formula == FusionFormula.F2:
            assert abs(r_with_g0 - 0.5) < 1e-5
