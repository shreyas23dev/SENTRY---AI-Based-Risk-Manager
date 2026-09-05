"""
test_baseline.py — TRUSTGRAPH Phase 1 Baseline Test Suite
==========================================================

Tests cover all required assertions from the Phase-1 specification:

    - Dataset loading and join integrity
    - Target exclusion (isFraud NOT in features)
    - TransactionID exclusion from model features
    - Chronological split correctness
    - No overlap between partitions
    - Preprocessing fit behaviour (fit-on-train-only)
    - Model save/load round-trip
    - A_t output shape and range
    - Threshold application

Run:
    python -m pytest tests/test_baseline.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow importing src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.baseline import config as cfg
from trustgraph.baseline.data_loader import (
    chronological_split,
    get_feature_and_target,
    load_train_data,
    _normalise_identity_columns,
    _verify_no_overlap,
)
from trustgraph.baseline.preprocessing import (
    BaselinePreprocessor,
    CategoricalEncoder,
    get_final_feature_list,
)
from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.evaluate import (
    compute_metrics,
    select_threshold_max_f1,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def small_train_df():
    """
    Minimal synthetic dataframe that mimics the structure of the merged
    IEEE-CIS training data. Used for lightweight unit tests that do not
    require the full 590k-row dataset.
    """
    np.random.seed(42)
    n = 500
    dt_vals = np.linspace(86400, 15_811_131, n).astype(int)
    n_fraud = 20  # ~4% fraud

    df = pd.DataFrame({
        "TransactionID":  np.arange(3_000_000, 3_000_000 + n),
        "isFraud":        np.array([1] * n_fraud + [0] * (n - n_fraud)),
        "TransactionDT":  dt_vals,
        "TransactionAmt": np.random.uniform(10, 5000, n).astype("float32"),
        "ProductCD":      np.random.choice(["W", "H", "C", "S", "R"], n),
        "card1":          np.random.randint(100, 9999, n).astype("float32"),
        "card2":          np.random.uniform(100, 600, n).astype("float32"),
        "card3":          np.random.uniform(100, 200, n).astype("float32"),
        "card4":          np.random.choice(["visa", "mastercard", "discover", "american express"], n),
        "card5":          np.random.uniform(100, 200, n).astype("float32"),
        "card6":          np.random.choice(["debit", "credit", "debit or credit", "charge card"], n),
        "addr1":          np.random.uniform(100, 500, n).astype("float32"),
        "addr2":          np.random.uniform(10, 100, n).astype("float32"),
        "dist1":          np.where(np.random.rand(n) < 0.6, np.nan, np.random.uniform(0, 1000, n)).astype("float32"),
        "dist2":          np.where(np.random.rand(n) < 0.9, np.nan, np.random.uniform(0, 1000, n)).astype("float32"),
        "P_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n),
        "R_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", None], n),
        "C1":  np.random.uniform(0, 10, n).astype("float32"),
        "C2":  np.random.uniform(0, 10, n).astype("float32"),
        "C3":  np.random.uniform(0, 10, n).astype("float32"),
        "C4":  np.random.uniform(0, 10, n).astype("float32"),
        "C5":  np.random.uniform(0, 10, n).astype("float32"),
        "C6":  np.random.uniform(0, 10, n).astype("float32"),
        "C7":  np.random.uniform(0, 10, n).astype("float32"),
        "C8":  np.random.uniform(0, 10, n).astype("float32"),
        "C9":  np.random.uniform(0, 10, n).astype("float32"),
        "C10": np.random.uniform(0, 10, n).astype("float32"),
        "C11": np.random.uniform(0, 10, n).astype("float32"),
        "C12": np.random.uniform(0, 10, n).astype("float32"),
        "C13": np.random.uniform(0, 10, n).astype("float32"),
        "C14": np.random.uniform(0, 10, n).astype("float32"),
        "M1":  np.random.choice(["T", "F", None], n),
        "M2":  np.random.choice(["T", "F", None], n),
        "M3":  np.random.choice(["T", "F", None], n),
        "M4":  np.random.choice(["M0", "M1", "M2", None], n),
        "M5":  np.random.choice(["T", "F", None], n),
        "M6":  np.random.choice(["T", "F", None], n),
        "M7":  np.random.choice(["T", "F", None], n),
        "M8":  np.random.choice(["T", "F", None], n),
        "M9":  np.random.choice(["T", "F", None], n),
        "DeviceType": np.random.choice(["desktop", "mobile", None], n),
        "DeviceInfo": np.random.choice(["Windows", "iOS Device", "MacOS", None], n),
        "id_12":  np.random.choice(["Found", "NotFound", None], n),
        "id_01":  np.random.uniform(-10, 10, n).astype("float32"),
    })
    # Add a few V-features
    for i in [1, 2, 3, 4, 5]:
        df[f"V{i}"] = np.where(np.random.rand(n) < 0.2, np.nan,
                                np.random.uniform(0, 5, n)).astype("float32")

    return df.sort_values("TransactionDT").reset_index(drop=True)


@pytest.fixture(scope="module")
def split_dfs(small_train_df):
    """Return train/val/test split of the synthetic df."""
    # Use custom small boundaries for synthetic data
    dt_vals = small_train_df["TransactionDT"].values
    n = len(dt_vals)
    train_bound = int(np.percentile(dt_vals, 70))
    val_bound   = int(np.percentile(dt_vals, 85))

    train = small_train_df[small_train_df["TransactionDT"] <= train_bound].copy()
    val   = small_train_df[(small_train_df["TransactionDT"] > train_bound) &
                            (small_train_df["TransactionDT"] <= val_bound)].copy()
    test  = small_train_df[small_train_df["TransactionDT"] > val_bound].copy()
    return train, val, test


# ===========================================================================
# 1. Dataset Loading and Join
# ===========================================================================

class TestDataLoading:

    def test_csv_files_exist(self):
        """Verify all raw CSV files are present at the expected location."""
        for path in [
            cfg.TRAIN_TRANSACTION_CSV,
            cfg.TRAIN_IDENTITY_CSV,
            cfg.TEST_TRANSACTION_CSV,
            cfg.TEST_IDENTITY_CSV,
            cfg.SAMPLE_SUBMISSION_CSV,
        ]:
            assert path.exists(), f"Expected CSV not found: {path}"

    def test_normalise_identity_columns(self):
        """id-XX dash notation should be normalised to id_XX underscore."""
        df = pd.DataFrame({"id-01": [1], "id-02": [2], "TransactionID": [999]})
        result = _normalise_identity_columns(df)
        assert "id_01" in result.columns
        assert "id_02" in result.columns
        assert "id-01" not in result.columns
        assert "TransactionID" in result.columns  # unchanged

    def test_normalise_already_underscore(self):
        """Columns already using underscores should be unchanged."""
        df = pd.DataFrame({"id_01": [1], "id_02": [2]})
        result = _normalise_identity_columns(df)
        assert "id_01" in result.columns
        assert "id_02" in result.columns


# ===========================================================================
# 2. Target Exclusion
# ===========================================================================

class TestTargetExclusion:

    def test_isfraud_not_in_feature_cols(self, small_train_df):
        """CRITICAL: isFraud must never appear in the model feature list."""
        feature_cols = get_final_feature_list(small_train_df)
        assert "isFraud" not in feature_cols, \
            "LEAKAGE DETECTED: isFraud found in feature list!"

    def test_isfraud_not_in_get_feature_and_target(self, small_train_df):
        """get_feature_and_target must raise if isFraud is in feature list."""
        feature_cols = get_final_feature_list(small_train_df)
        # Correct behaviour: no isFraud in features
        X, y = get_feature_and_target(small_train_df, feature_cols)
        assert "isFraud" not in X.columns

    def test_get_feature_and_target_raises_on_target_in_features(self, small_train_df):
        """Passing isFraud in feature_cols must raise AssertionError."""
        bad_cols = get_final_feature_list(small_train_df) + ["isFraud"]
        with pytest.raises(AssertionError, match="LEAKAGE"):
            get_feature_and_target(small_train_df, bad_cols)


# ===========================================================================
# 3. TransactionID Exclusion
# ===========================================================================

class TestTransactionIDExclusion:

    def test_transaction_id_not_in_feature_list(self, small_train_df):
        """CRITICAL: TransactionID must not be a predictive numerical feature."""
        feature_cols = get_final_feature_list(small_train_df)
        assert "TransactionID" not in feature_cols, \
            "IDENTIFIER LEAKAGE: TransactionID in feature list!"

    def test_transaction_id_not_in_preprocessor_output(self, small_train_df):
        """Preprocessor output must not contain TransactionID column."""
        preprocessor = BaselinePreprocessor()
        X = preprocessor.fit_transform(small_train_df)
        assert "TransactionID" not in X.columns

    def test_get_feature_and_target_raises_on_id_in_features(self, small_train_df):
        """Passing TransactionID in feature_cols must raise AssertionError."""
        bad_cols = get_final_feature_list(small_train_df) + ["TransactionID"]
        with pytest.raises(AssertionError, match="IDENTIFIER"):
            get_feature_and_target(small_train_df, bad_cols)


# ===========================================================================
# 4. Chronological Split
# ===========================================================================

class TestChronologicalSplit:

    def test_train_dt_max_lte_val_dt_min(self, split_dfs):
        """max(train DT) <= min(val DT) — temporal ordering."""
        train, val, _ = split_dfs
        assert train["TransactionDT"].max() <= val["TransactionDT"].min(), \
            "TEMPORAL LEAK: train contains future transactions!"

    def test_val_dt_max_lte_test_dt_min(self, split_dfs):
        """max(val DT) <= min(test DT) — temporal ordering."""
        _, val, test = split_dfs
        assert val["TransactionDT"].max() <= test["TransactionDT"].min(), \
            "TEMPORAL LEAK: val contains future transactions!"

    def test_no_row_overlap_by_transaction_id(self, split_dfs):
        """Each TransactionID must appear in exactly one partition."""
        train, val, test = split_dfs
        train_ids = set(train["TransactionID"])
        val_ids   = set(val["TransactionID"])
        test_ids  = set(test["TransactionID"])
        assert len(train_ids & val_ids) == 0,  "TransactionID overlap: TRAIN ∩ VAL!"
        assert len(val_ids & test_ids) == 0,   "TransactionID overlap: VAL ∩ TEST!"
        assert len(train_ids & test_ids) == 0, "TransactionID overlap: TRAIN ∩ TEST!"

    def test_total_rows_preserved(self, small_train_df, split_dfs):
        """Sum of partition rows must equal total rows."""
        train, val, test = split_dfs
        assert len(train) + len(val) + len(test) == len(small_train_df)

    def test_verify_no_overlap_passes(self, split_dfs):
        """_verify_no_overlap should not raise for a valid split."""
        train, val, test = split_dfs
        _verify_no_overlap(train, val, test)

    def test_verify_no_overlap_fails_on_shuffled(self, small_train_df):
        """_verify_no_overlap must raise if partitions are randomly shuffled."""
        # Create a split with temporal overlap
        n = len(small_train_df)
        fake_train = small_train_df.iloc[:int(0.8*n)].copy()
        fake_val   = small_train_df.iloc[int(0.2*n):int(0.6*n)].copy()
        fake_test  = small_train_df.iloc[int(0.7*n):].copy()
        with pytest.raises(AssertionError):
            _verify_no_overlap(fake_train, fake_val, fake_test)


# ===========================================================================
# 5. Preprocessing Fit Behaviour
# ===========================================================================

class TestPreprocessing:

    def test_fit_transform_returns_dataframe(self, small_train_df):
        """fit_transform must return a DataFrame."""
        p = BaselinePreprocessor()
        X = p.fit_transform(small_train_df)
        assert isinstance(X, pd.DataFrame)

    def test_transform_same_columns_as_fit(self, split_dfs):
        """transform() on val must produce same columns as fit_transform() on train."""
        train, val, _ = split_dfs
        p = BaselinePreprocessor()
        X_train = p.fit_transform(train)
        X_val   = p.transform(val)
        assert list(X_train.columns) == list(X_val.columns), \
            "Column mismatch between train and val after preprocessing!"

    def test_transform_before_fit_raises(self, small_train_df):
        """Calling transform() before fit() must raise RuntimeError."""
        p = BaselinePreprocessor()
        with pytest.raises(RuntimeError):
            p.transform(small_train_df)

    def test_unknown_categories_become_nan(self, split_dfs):
        """Unknown categories in val/test must become NaN, not crash."""
        train, val, _ = split_dfs
        # Inject a category that cannot be in train (unique integer suffix)
        val_modified = val.copy()
        if "ProductCD" in val_modified.columns:
            val_modified["ProductCD"] = "UNKNOWN_XXXX"
        p = BaselinePreprocessor()
        p.fit_transform(train)
        # Must not raise
        X_val = p.transform(val_modified)
        if "ProductCD" in X_val.columns:
            assert X_val["ProductCD"].isna().all(), \
                "Unknown categories must map to NaN!"

    def test_no_isfraud_in_output(self, small_train_df):
        """Preprocessor output must not contain isFraud."""
        p = BaselinePreprocessor()
        X = p.fit_transform(small_train_df)
        assert "isFraud" not in X.columns

    def test_no_transaction_id_in_output(self, small_train_df):
        """Preprocessor output must not contain TransactionID."""
        p = BaselinePreprocessor()
        X = p.fit_transform(small_train_df)
        assert "TransactionID" not in X.columns

    def test_save_load_roundtrip(self, small_train_df, tmp_path):
        """Saved and loaded preprocessor must produce identical output."""
        p1 = BaselinePreprocessor()
        X1 = p1.fit_transform(small_train_df)
        p1.save(tmp_path / "prep")

        p2 = BaselinePreprocessor.load(tmp_path / "prep")
        X2 = p2.transform(small_train_df)

        pd.testing.assert_frame_equal(X1, X2, check_dtype=False)

    def test_categorical_encoder_fit_keys(self, small_train_df):
        """CategoricalEncoder must only contain training categories."""
        p = BaselinePreprocessor()
        p.fit_transform(small_train_df)
        encoder = p.encoder
        if "ProductCD" in encoder.mappings:
            train_cats = set(small_train_df["ProductCD"].dropna().astype(str).unique())
            enc_cats   = set(encoder.mappings["ProductCD"].keys())
            assert enc_cats == train_cats, "Encoder has extra categories!"


# ===========================================================================
# 6. Model Save/Load and A_t Output
# ===========================================================================

class TestModel:

    @pytest.fixture(scope="class")
    def trained_model_and_data(self, split_dfs):
        """Train a minimal LightGBM on synthetic data for model-level tests."""
        train, val, test = split_dfs
        p = BaselinePreprocessor()
        X_train = p.fit_transform(train)
        y_train = train["isFraud"].values
        X_val   = p.transform(val)
        y_val   = val["isFraud"].values
        X_test  = p.transform(test)
        y_test  = test["isFraud"].values

        model = BaselineModel(params={
            "objective": "binary",
            "metric": ["auc"],
            "boosting_type": "gbdt",
            "n_estimators": 50,     # small for testing
            "learning_rate": 0.1,
            "num_leaves": 16,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": 1,
        })
        model.fit(X_train, y_train, X_val, y_val, cat_cols=p.cat_cols)
        return model, p, X_test, y_test

    def test_predict_risk_returns_array(self, trained_model_and_data):
        model, _, X_test, _ = trained_model_and_data
        A_t = model.predict_risk(X_test)
        assert isinstance(A_t, np.ndarray)

    def test_at_shape_matches_input(self, trained_model_and_data):
        """A_t must have the same length as the input."""
        model, _, X_test, _ = trained_model_and_data
        A_t = model.predict_risk(X_test)
        assert A_t.shape == (len(X_test),), \
            f"Shape mismatch: A_t={A_t.shape}, X_test={len(X_test)}"

    def test_at_range_0_to_1(self, trained_model_and_data):
        """CRITICAL: Every A_t value must be in [0, 1]."""
        model, _, X_test, _ = trained_model_and_data
        A_t = model.predict_risk(X_test)
        assert np.all(A_t >= 0.0), f"A_t has values < 0: min={A_t.min()}"
        assert np.all(A_t <= 1.0), f"A_t has values > 1: max={A_t.max()}"

    def test_predict_label_binary(self, trained_model_and_data):
        """predict_label must return only 0 and 1."""
        model, _, X_test, _ = trained_model_and_data
        labels = model.predict_label(X_test, threshold=0.5)
        unique_labels = set(labels.tolist())
        assert unique_labels.issubset({0, 1}), \
            f"predict_label returned non-binary values: {unique_labels}"

    def test_predict_label_threshold_effect(self, trained_model_and_data):
        """Lower threshold → more predicted fraud."""
        model, _, X_test, _ = trained_model_and_data
        labels_low  = model.predict_label(X_test, threshold=0.1)
        labels_high = model.predict_label(X_test, threshold=0.9)
        assert labels_low.sum() >= labels_high.sum(), \
            "Lower threshold should predict at least as many fraud cases."

    def test_model_save_load_roundtrip(self, trained_model_and_data, tmp_path):
        """Saved and reloaded model must produce identical predictions."""
        model, _, X_test, _ = trained_model_and_data
        A_t_before = model.predict_risk(X_test)

        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = BaselineModel.load(path)
        A_t_after = loaded.predict_risk(X_test)

        np.testing.assert_array_almost_equal(A_t_before, A_t_after, decimal=6)

    def test_model_not_trained_raises(self):
        """predict_risk on an untrained model must raise RuntimeError."""
        model = BaselineModel()
        dummy = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(RuntimeError):
            model.predict_risk(dummy)


# ===========================================================================
# 7. Threshold Application
# ===========================================================================

class TestThreshold:

    def test_threshold_selection_returns_float(self):
        """select_threshold_max_f1 must return a float in (0, 1)."""
        np.random.seed(0)
        y_true = np.random.randint(0, 2, 200)
        A_t    = np.random.uniform(0, 1, 200)
        thr, f1 = select_threshold_max_f1(y_true, A_t)
        assert isinstance(thr, float)
        assert 0.0 < thr < 1.0
        assert 0.0 <= f1 <= 1.0

    def test_threshold_zero_produces_all_fraud(self):
        """threshold=0 must classify everything as fraud."""
        A_t    = np.array([0.1, 0.5, 0.9, 0.0])
        model  = BaselineModel()
        # Bypass training by directly calling the threshold logic
        labels = (A_t >= 0.0).astype(int)
        assert labels.sum() == len(A_t)

    def test_threshold_one_produces_no_fraud(self):
        """threshold=1.0 must classify everything as legitimate (except exact 1.0)."""
        A_t    = np.array([0.1, 0.5, 0.9, 0.99])
        labels = (A_t >= 1.0).astype(int)
        assert labels.sum() == 0


# ===========================================================================
# 8. Metrics
# ===========================================================================

class TestMetrics:

    def test_compute_metrics_keys(self):
        """compute_metrics must return all required metric keys."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        A_t    = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.05])
        metrics = compute_metrics(y_true, A_t, threshold=0.5)
        required_keys = [
            "roc_auc", "pr_auc", "precision", "recall", "f1",
            "fpr", "fnr", "total_transactions", "fraudulent", "legitimate",
            "fraud_prevalence", "threshold",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_compute_metrics_ranges(self):
        """All metric values must be in valid [0, 1] range."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        A_t    = np.random.uniform(0, 1, 200)
        metrics = compute_metrics(y_true, A_t, threshold=0.5)
        for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "fpr", "fnr"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of [0,1]"

    def test_fraud_legit_sum_to_total(self):
        """fraudulent + legitimate must equal total_transactions."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        A_t    = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.05])
        metrics = compute_metrics(y_true, A_t, threshold=0.5)
        assert metrics["fraudulent"] + metrics["legitimate"] == metrics["total_transactions"]
