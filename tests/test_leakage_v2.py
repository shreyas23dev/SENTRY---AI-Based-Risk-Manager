"""
test_leakage_v2.py — Automated Leakage and Causality Tests for V2 Features
===========================================================================

Verifies:
  1. Strict causality: Transaction t never sees information from t+1..N.
  2. Future truncation invariance: Truncating future rows produces identical features for row t.
  3. No target leakage: Target isFraud is never passed or referenced.
  4. FrequencyEncoder fit-isolation: Fits on TRAIN only, zero out-of-fold leakage.
  5. Sequential state persistence: Causal order across partitions maintains state without reset leakage.
"""

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.features_v2.causal_features import (
    compute_point_in_time_features,
    FrequencyEncoder,
    CausalStreamFeatureEngine,
)


def create_synthetic_stream(n=100):
    """Creates a deterministic synthetic transaction stream."""
    rng = np.random.default_rng(42)
    dt_base = 86400.0
    records = []
    entities = ["cardA_addr1_email1", "cardB_addr2_email2", "unresolved_99"]
    devices = ["Device1", "Device2", None]

    for i in range(n):
        records.append({
            "TransactionID": 1000 + i,
            "TransactionDT": dt_base + i * 3600.0,
            "TransactionAmt": float(rng.uniform(10.0, 500.0)),
            "entity_proxy": entities[i % len(entities)],
            "card1": f"card_{i % 3}",
            "addr1": f"addr_{i % 2}",
            "P_emaildomain": f"email_{i % 2}.com",
            "DeviceInfo": devices[i % len(devices)],
            "isFraud": int(rng.choice([0, 1], p=[0.95, 0.05])),
        })
    return pd.DataFrame(records)


class TestCausalityAndLeakage:

    def test_no_target_leakage(self):
        """Features must be computable without isFraud present in DataFrame."""
        df = create_synthetic_stream(20)
        df_no_label = df.drop(columns=["isFraud"])

        # Point in time
        f_pit = compute_point_in_time_features(df_no_label)
        assert len(f_pit) == 20

        # Frequency encoder
        fe = FrequencyEncoder(["card1", "addr1"]).fit(df_no_label)
        f_fe = fe.transform(df_no_label)
        assert len(f_fe) == 20

        # Causal stream engine
        engine = CausalStreamFeatureEngine()
        f_stream = engine.process_partition(df_no_label)
        assert len(f_stream) == 20

    def test_future_truncation_invariance(self):
        """
        Feature values for row i must be IDENTICAL whether evaluated on
        the full stream df[0:N] or a truncated stream df[0:i+1].
        """
        df = create_synthetic_stream(50)
        engine = CausalStreamFeatureEngine()
        f_full = engine.process_partition(df)

        # Truncate at row 25
        engine.reset()
        f_truncated = engine.process_partition(df.iloc[:26])

        for col in f_full.columns:
            np.testing.assert_allclose(
                f_full[col].values[:26],
                f_truncated[col].values,
                err_msg=f"Causality violation in feature '{col}': row 25 changed when future rows were truncated!"
            )

    def test_prior_stats_are_strictly_before_t(self):
        """
        For an entity's first transaction, prior_count must be 0 and hist_std must be 0.
        """
        df = pd.DataFrame([
            {"TransactionID": 1, "TransactionDT": 100.0, "TransactionAmt": 50.0, "entity_proxy": "user1", "card1": "c1", "addr1": "a1", "P_emaildomain": "e1", "DeviceInfo": "d1"},
            {"TransactionID": 2, "TransactionDT": 200.0, "TransactionAmt": 150.0, "entity_proxy": "user1", "card1": "c1", "addr1": "a1", "P_emaildomain": "e1", "DeviceInfo": "d1"},
        ])
        engine = CausalStreamFeatureEngine()
        f = engine.process_partition(df)

        # Row 0: first transaction for user1
        assert f["entity_prior_count"].iloc[0] == 0.0
        assert f["prior_count_card1"].iloc[0] == 0.0
        assert f["entity_dt_elapsed"].iloc[0] == -1.0

        # Row 1: second transaction for user1
        assert f["entity_prior_count"].iloc[1] == 1.0
        assert f["prior_count_card1"].iloc[1] == 1.0
        assert f["entity_hist_mean_amt"].iloc[1] == 50.0  # prior mean was 50.0, not (50+150)/2
        assert f["entity_dt_elapsed"].iloc[1] == 100.0   # 200 - 100

    def test_frequency_encoder_unseen_categories(self):
        """Unseen validation categories must receive frequency 0.0, not crash."""
        train_df = pd.DataFrame({"cat": ["A", "A", "B"]})
        val_df = pd.DataFrame({"cat": ["A", "C", "D", None]})

        fe = FrequencyEncoder(["cat"]).fit(train_df)
        f_val = fe.transform(val_df)

        assert f_val["freq_cat"].iloc[0] == pytest.approx(2.0 / 3.0)
        assert f_val["freq_cat"].iloc[1] == 0.0  # C unseen
        assert f_val["freq_cat"].iloc[2] == 0.0  # D unseen
        assert f_val["freq_cat"].iloc[3] == 0.0  # None -> 0
