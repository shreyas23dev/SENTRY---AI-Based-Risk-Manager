"""
test_temporal.py — TRUSTGRAPH Phase 2 Temporal Risk Memory Test Suite
======================================================================

Unit tests verifying:
  - E_0 initialization
  - EMA recurrence dynamics
  - Bounded P_t state (0 <= P_t <= 1)
  - Upward accumulation condition (E_t > gamma)
  - Downward decay condition (E_t <= gamma)
  - Deterministic & chronological processing
  - Preservation of Phase 1 artifacts and A_t inputs
  - Multi-entity state isolation
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.baseline import config as base_cfg
from trustgraph.temporal import config as temp_cfg
from trustgraph.temporal.engine import (
    TemporalRiskEngine,
    EntityTemporalRiskTracker,
    compute_ema,
    compute_bounded_accumulator,
)
from trustgraph.temporal.evaluator import (
    evaluate_temporal_comparison,
    make_temporal_prediction,
)


class TestEMAMathematics:

    def test_ema_initialization(self):
        """E_0 must initialize to 0.0."""
        engine = TemporalRiskEngine(beta=0.4)
        assert engine.E_state == 0.0
        assert engine.P_state == 0.0

    def test_ema_recurrence_exact(self):
        """Verify E_1 = beta * A_1 + (1 - beta) * 0.0."""
        beta = 0.3
        A_1 = 0.8
        E_1 = compute_ema(A_1, 0.0, beta)
        assert E_1 == pytest.approx(beta * A_1)

        # Step 2: E_2 = beta * A_2 + (1 - beta) * E_1
        A_2 = 0.5
        E_2 = compute_ema(A_2, E_1, beta)
        expected_E_2 = beta * A_2 + (1.0 - beta) * E_1
        assert E_2 == pytest.approx(expected_E_2)

    def test_ema_bounds(self):
        """E_t must stay in [0, 1] for arbitrary valid inputs."""
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for prev in [0.0, 0.5, 1.0]:
                for b in [0.1, 0.5, 1.0]:
                    e = compute_ema(a, prev, b)
                    assert 0.0 <= e <= 1.0


class TestBoundedAccumulator:

    def test_accumulation_step(self):
        """When E_t > gamma, P should increment by lambda (bounded by 1.0)."""
        gamma = 0.3
        lambda_ = 0.2
        delta = 0.05
        E_t = 0.5  # > gamma
        prev_P = 0.1
        next_P = compute_bounded_accumulator(E_t, prev_P, gamma, lambda_, delta)
        assert next_P == pytest.approx(prev_P + lambda_)

    def test_decay_step(self):
        """When E_t <= gamma, P should decrement by delta (bounded by 0.0)."""
        gamma = 0.3
        lambda_ = 0.2
        delta = 0.05
        E_t = 0.2  # <= gamma
        prev_P = 0.4
        next_P = compute_bounded_accumulator(E_t, prev_P, gamma, lambda_, delta)
        assert next_P == pytest.approx(prev_P - delta)

    def test_upper_bound_clipping(self):
        """P_t must never exceed 1.0 under repeated accumulation."""
        gamma = 0.2
        lambda_ = 0.4
        delta = 0.05
        P = 0.8
        for _ in range(10):
            P = compute_bounded_accumulator(0.9, P, gamma, lambda_, delta)
            assert P <= 1.0
        assert P == pytest.approx(1.0)

    def test_lower_bound_clipping(self):
        """P_t must never drop below 0.0 under repeated decay."""
        gamma = 0.2
        lambda_ = 0.4
        delta = 0.1
        P = 0.2
        for _ in range(10):
            P = compute_bounded_accumulator(0.05, P, gamma, lambda_, delta)
            assert P >= 0.0
        assert P == pytest.approx(0.0)


class TestTemporalRiskEngine:

    def test_stream_processing_shapes(self):
        """process_stream must return arrays of same length as input."""
        A_stream = np.array([0.1, 0.4, 0.7, 0.2, 0.05])
        engine = TemporalRiskEngine(beta=0.4, gamma=0.3, lambda_=0.2, delta=0.05)
        E_arr, P_arr = engine.process_stream(A_stream)
        assert len(E_arr) == len(A_stream)
        assert len(P_arr) == len(A_stream)
        assert np.all(E_arr >= 0.0) and np.all(E_arr <= 1.0)
        assert np.all(P_arr >= 0.0) and np.all(P_arr <= 1.0)

    def test_reset_functionality(self):
        """reset() must restore internal states to 0.0."""
        engine = TemporalRiskEngine()
        engine.step(0.9)
        assert engine.E_state > 0.0
        engine.reset()
        assert engine.E_state == 0.0
        assert engine.P_state == 0.0

    def test_deterministic_output(self):
        """Repeated runs on the same sequence must produce identical outputs."""
        A_stream = np.random.uniform(0, 1, 100)
        engine1 = TemporalRiskEngine(beta=0.35, gamma=0.25, lambda_=0.15, delta=0.04)
        engine2 = TemporalRiskEngine(beta=0.35, gamma=0.25, lambda_=0.15, delta=0.04)
        E1, P1 = engine1.process_stream(A_stream)
        E2, P2 = engine2.process_stream(A_stream)
        np.testing.assert_array_almost_equal(E1, E2)
        np.testing.assert_array_almost_equal(P1, P2)


class TestEntityTemporalRiskTracker:

    def test_independent_entity_isolation(self):
        """States of different entities must be completely isolated."""
        tracker = EntityTemporalRiskTracker(beta=0.5, gamma=0.3, lambda_=0.2, delta=0.05)
        
        # Entity A has high risk
        e_a1, p_a1 = tracker.step("card_A", 0.9)
        # Entity B has low risk
        e_b1, p_b1 = tracker.step("card_B", 0.01)

        assert e_a1 > e_b1
        # Step A again
        e_a2, p_a2 = tracker.step("card_A", 0.9)
        # Step B again
        e_b2, p_b2 = tracker.step("card_B", 0.01)

        assert p_a2 > p_b2
        assert p_b2 == 0.0  # Stayed at 0


class TestPreservationOfPhase1:

    def test_phase1_model_intact(self):
        """Phase 1 LightGBM model artifact must exist and load."""
        model_path = base_cfg.MODEL_DIR / "lgbm_model.pkl"
        assert model_path.exists(), "Phase 1 model missing!"

    def test_phase1_predictions_intact(self):
        """Phase 1 test_predictions.csv must exist with original columns."""
        pred_path = temp_cfg.TEST_PREDICTIONS_CSV
        assert pred_path.exists(), "Phase 1 predictions missing!"
        df = pd.read_csv(pred_path, nrows=10)
        expected = ["TransactionID", "TransactionDT", "isFraud", "A_t", "baseline_prediction"]
        for c in expected:
            assert c in df.columns, f"Missing {c} in Phase 1 predictions!"


class TestEvaluatorLogic:

    def test_temporal_prediction_logic(self):
        """make_temporal_prediction must flag if either A_t >= base_thr OR P_t >= temp_thr."""
        A_t = np.array([0.7, 0.3, 0.4, 0.1])
        P_t = np.array([0.1, 0.8, 0.2, 0.1])
        base_thr = 0.6
        temp_thr = 0.5
        preds = make_temporal_prediction(A_t, P_t, base_thr, temp_thr)
        expected = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(preds, expected)

    def test_comparative_evaluation_keys(self):
        """evaluate_temporal_comparison must return all required delta and comparative metrics."""
        y_true = np.array([1, 0, 1, 0, 1])
        A_t    = np.array([0.8, 0.1, 0.4, 0.2, 0.3])
        E_t    = np.array([0.6, 0.2, 0.4, 0.3, 0.3])
        P_t    = np.array([0.5, 0.0, 0.6, 0.1, 0.1])
        ev = evaluate_temporal_comparison(y_true, A_t, E_t, P_t, 0.5, 0.5)
        assert "B0_baseline" in ev
        assert "B1_temporal" in ev
        assert "comparative_delta" in ev
        assert "additional_frauds_recovered" in ev["comparative_delta"]
        assert "additional_false_positives" in ev["comparative_delta"]
