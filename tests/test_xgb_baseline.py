"""
test_xgb_baseline.py — Unit Tests for XGBoost Fraud Detection Baseline
======================================================================

Covers all required Phase 1 testing requirements:
  1. Feature generation
  2. Feature-column consistency
  3. Missing value handling
  4. Inference output format (A_t in [0, 1])
  5. Deterministic predictions
  6. Model and pipeline save/load roundtrip
  7. Train/inference feature alignment
  8. Chronological split invariance
  9. Zero label leakage (isFraud not in features, no test fitting)
 10. Prediction output formats (predict_risk, predict_proba, score_dataframe)
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.baseline.model_features import ModelFeaturePipeline
from trustgraph.baseline.xgb_model import XGBRiskModel
from trustgraph.baseline.xgb_baseline import XGBBaselineWrapper
from trustgraph.baseline import config as base_cfg

FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures_sample_txns.json"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "kaggle_xgb"


@pytest.fixture(scope="module")
def sample_raw_df():
    """Load sample raw transactions."""
    if not FIXTURES_PATH.exists():
        pytest.skip("Fixtures file not found.")
    with open(FIXTURES_PATH) as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Load pre-trained feature pipeline artifact if available, else fit on sample."""
    pipeline_path = ARTIFACT_DIR / "feature_pipeline.pkl"
    if pipeline_path.exists():
        return ModelFeaturePipeline.load(pipeline_path)
    with open(FIXTURES_PATH) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    pipe = ModelFeaturePipeline()
    pipe.fit(df)
    return pipe


@pytest.fixture(scope="module")
def trained_model(fitted_pipeline, sample_raw_df):
    """Load pre-trained XGBoost model artifact if available, else dummy fit."""
    model_path = ARTIFACT_DIR / "xgb_model.pkl"
    if model_path.exists():
        return XGBRiskModel.load(model_path)
    # Fallback dummy model fitted on the pipeline's features (263 features)
    model = XGBRiskModel(params={"n_estimators": 5, "max_depth": 3, "tree_method": "hist"})
    X_feat = fitted_pipeline.transform(sample_raw_df)
    y = np.random.randint(0, 2, len(sample_raw_df))
    model.fit(X_feat, y, X_feat, y, verbose=False)
    return model


# ---------------------------------------------------------------------------
# 1. Feature Generation
# ---------------------------------------------------------------------------

def test_feature_generation(sample_raw_df):
    """Pipeline generates 263 features including cents, outsider15, and uid aggregations."""
    pipe = ModelFeaturePipeline()
    pipe.fit(sample_raw_df)
    X = pipe.transform(sample_raw_df)

    assert len(pipe.feature_cols) == 263
    assert X.shape == (len(sample_raw_df), 263)
    assert "cents" in pipe.feature_cols
    assert "outsider15" in pipe.feature_cols
    assert "P_emaildomain_FE" in pipe.feature_cols
    assert "TransactionAmt_uid_mean" in pipe.feature_cols


# ---------------------------------------------------------------------------
# 2. Feature-Column Consistency
# ---------------------------------------------------------------------------

def test_feature_column_consistency(fitted_pipeline, sample_raw_df):
    """Output DataFrame columns and order strictly match pipe.feature_cols."""
    X1 = fitted_pipeline.transform(sample_raw_df)
    X2 = fitted_pipeline.transform(sample_raw_df.iloc[::-1])

    assert list(X1.columns) == fitted_pipeline.feature_cols
    assert list(X2.columns) == fitted_pipeline.feature_cols
    assert list(X1.columns) == list(X2.columns)


# ---------------------------------------------------------------------------
# 3. Missing Value Handling
# ---------------------------------------------------------------------------

def test_missing_values_handled(fitted_pipeline, sample_raw_df):
    """Transformed feature matrix contains zero unhandled NaNs (filled with -1.0)."""
    df_missing = sample_raw_df.copy()
    # Inject aggressive missingness across numeric and string fields
    df_missing["dist1"] = np.nan
    df_missing["card2"] = np.nan
    df_missing["DeviceInfo"] = None
    df_missing["P_emaildomain"] = None

    X = fitted_pipeline.transform(df_missing)
    assert X.isna().sum().sum() == 0, "Transformed feature matrix must not contain NaNs."


# ---------------------------------------------------------------------------
# 4. Inference Output Format (A_t in [0.0, 1.0])
# ---------------------------------------------------------------------------

def test_inference_output_format(fitted_pipeline, trained_model, sample_raw_df):
    """Model produces valid continuous risk scores A_t in [0.0, 1.0]."""
    X = fitted_pipeline.transform(sample_raw_df)
    probs = trained_model.predict_risk(X)

    assert isinstance(probs, np.ndarray)
    assert len(probs) == len(sample_raw_df)
    assert probs.dtype in (np.float32, np.float64)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# 5. Deterministic Predictions
# ---------------------------------------------------------------------------

def test_deterministic_predictions(fitted_pipeline, trained_model, sample_raw_df):
    """Identical feature inputs yield bitwise identical risk predictions."""
    X = fitted_pipeline.transform(sample_raw_df)
    p1 = trained_model.predict_risk(X)
    p2 = trained_model.predict_risk(X)

    np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# 6. Model and Pipeline Save / Load Roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(fitted_pipeline, trained_model, sample_raw_df):
    """Serialization preserves exact pipeline mappings and prediction output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        pipe_file = tmp_path / "test_pipe.pkl"
        model_file = tmp_path / "test_model.pkl"

        fitted_pipeline.save(pipe_file)
        trained_model.save(model_file)

        loaded_pipe = ModelFeaturePipeline.load(pipe_file)
        loaded_model = XGBRiskModel.load(model_file)

        assert loaded_pipe.feature_cols == fitted_pipeline.feature_cols
        assert loaded_model.feature_names == trained_model.feature_names

        X_orig = fitted_pipeline.transform(sample_raw_df)
        X_load = loaded_pipe.transform(sample_raw_df)
        np.testing.assert_array_equal(X_orig.values, X_load.values)

        p_orig = trained_model.predict_risk(X_orig)
        p_load = loaded_model.predict_risk(X_load)
        np.testing.assert_allclose(p_orig, p_load, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Train / Inference Feature Alignment
# ---------------------------------------------------------------------------

def test_train_inference_feature_alignment(fitted_pipeline):
    """Single row dictionary with sparse features aligns cleanly without shape mismatch."""
    minimal_row = {
        "TransactionID": 9999999,
        "TransactionDT": 100000.0,
        "TransactionAmt": 49.99,
        "card1": 13926,
        "addr1": 315,
        "P_emaildomain": "gmail.com",
    }
    df_single = pd.DataFrame([minimal_row])
    X = fitted_pipeline.transform(df_single)

    assert X.shape == (1, len(fitted_pipeline.feature_cols))
    assert list(X.columns) == fitted_pipeline.feature_cols


# ---------------------------------------------------------------------------
# 8. Chronological Split Invariance
# ---------------------------------------------------------------------------

def test_chronological_split_invariance():
    """Chronological boundaries ensure strict train < val < test ordering."""
    train_bound = base_cfg.TRAIN_DT_BOUNDARY
    val_bound = base_cfg.VAL_DT_BOUNDARY

    assert train_bound < val_bound, "TRAIN boundary must strictly precede VALIDATION boundary."
    assert train_bound == 10_438_003
    assert val_bound == 13_151_880


# ---------------------------------------------------------------------------
# 9. Zero Target / Label Leakage
# ---------------------------------------------------------------------------

def test_zero_target_leakage(fitted_pipeline, sample_raw_df):
    """Features do NOT contain isFraud, TransactionID, or target-derived leakages."""
    X = fitted_pipeline.transform(sample_raw_df)

    assert "isFraud" not in X.columns
    assert "TransactionID" not in X.columns
    assert "target" not in X.columns
    assert "oof" not in X.columns


# ---------------------------------------------------------------------------
# 10. Unified Wrapper Contract (XGBBaselineWrapper)
# ---------------------------------------------------------------------------

def test_xgb_baseline_wrapper_contract(fitted_pipeline, trained_model, sample_raw_df):
    """XGBBaselineWrapper implements predict_risk, predict_proba, predict, and score_dataframe."""
    wrapper = XGBBaselineWrapper(
        pipeline=fitted_pipeline,
        model=trained_model,
        default_threshold=0.12,
    )

    # 1. predict_risk
    probs = wrapper.predict_risk(sample_raw_df)
    assert len(probs) == len(sample_raw_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # 2. predict_proba
    proba_arr = wrapper.predict_proba(sample_raw_df)
    assert proba_arr.shape == (len(sample_raw_df), 2)
    np.testing.assert_allclose(proba_arr[:, 0] + proba_arr[:, 1], 1.0, atol=1e-5)

    # 3. predict
    binary_preds = wrapper.predict(sample_raw_df, threshold=0.12)
    assert set(np.unique(binary_preds)).issubset({0, 1})

    # 4. score_dataframe
    df_scored = wrapper.score_dataframe(sample_raw_df)
    assert "risk_score" in df_scored.columns
    assert "A_t" in df_scored.columns
    np.testing.assert_array_equal(df_scored["risk_score"].values, probs)

    # 5. Dict single-row inference
    dict_row = sample_raw_df.iloc[0].to_dict()
    prob_single = wrapper.predict_risk(dict_row)
    assert len(prob_single) == 1
    assert prob_single[0] == pytest.approx(probs[0], abs=1e-5)
