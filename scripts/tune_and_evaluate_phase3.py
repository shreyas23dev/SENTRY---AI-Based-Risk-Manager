"""
tune_and_evaluate_phase3.py
===========================
Phase 3 Mathematical Risk Engine — Full Pipeline

Workflow:
  1. Load A_t (XGBoost) and G_t (Knowledge Graph) for TRAIN, VAL, TEST
  2. Evaluate calibration need (ECE) on TRAIN → VAL
  3. Tune fusion parameters on VALIDATION ONLY
  4. Freeze all parameters
  5. Evaluate all system configurations on TEST (once, final)
  6. Compute business-loss metrics under 3 cost scenarios
  7. Save frozen parameters + results to artifacts/risk/

INVARIANT: TEST labels are NEVER used for parameter selection.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.baseline.data_loader import chronological_split, load_train_data
from trustgraph.baseline.xgb_baseline import XGBBaselineWrapper
from trustgraph.risk.calibration import SignalCalibrator
from trustgraph.risk.cost_model import DEFAULT_SCENARIOS, MerchantCostModel
from trustgraph.risk.decision import DecisionEngine
from trustgraph.risk.fusion import FusionEngine, FusionFormula

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    fraud_total = int(y_true.sum())
    legit_total = n - fraud_total

    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc  = float(average_precision_score(y_true, y_score))
    prec    = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec     = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1      = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr     = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "roc_auc":   round(roc_auc, 6),
        "pr_auc":    round(pr_auc, 6),
        "precision": round(prec, 6),
        "recall":    round(rec, 6),
        "f1":        round(f1, 6),
        "fpr":       round(fpr, 6),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n": int(n),
        "fraud_total": int(fraud_total),
        "legit_total": int(legit_total),
        "fraud_captured_pct": round(100.0 * tp / fraud_total, 4) if fraud_total > 0 else 0.0,
        "legit_affected_pct": round(100.0 * (fp) / legit_total, 4) if legit_total > 0 else 0.0,
        "intervention_rate": round(100.0 * (tp + fp) / n, 4),
        "threshold": round(threshold, 6),
    }


def compute_business_losses(
    y_true: np.ndarray,
    r_t: np.ndarray,
    actions: np.ndarray,
    txn_amounts: np.ndarray,
    scenario,
) -> dict:
    """Calculate total business losses under cost scenario."""
    s = scenario
    total = len(y_true)
    fraud_mask = y_true == 1
    legit_mask = y_true == 0

    allowed_fraud  = fraud_mask & (actions == "ALLOW")
    blocked_legit  = legit_mask & (actions == "BLOCK")
    verify_legit   = legit_mask & (actions == "VERIFY")
    throttle_legit = legit_mask & (actions == "THROTTLE")
    verify_fraud   = fraud_mask & (actions == "VERIFY")
    throttle_fraud = fraud_mask & (actions == "THROTTLE")

    # Fraud losses
    fraud_loss_allow = float(
        np.sum((s.C_fraud_rate * txn_amounts[allowed_fraud]) + s.C_chargeback_fee)
        if allowed_fraud.any() else 0.0
    )
    fraud_loss_verify = float(
        np.sum((1 - s.verify_fraud_reduction) * (s.C_fraud_rate * txn_amounts[verify_fraud] + s.C_chargeback_fee))
        if verify_fraud.any() else 0.0
    )
    fraud_loss_throttle = float(
        np.sum((1 - s.throttle_fraud_reduction) * (s.C_fraud_rate * txn_amounts[throttle_fraud] + s.C_chargeback_fee))
        if throttle_fraud.any() else 0.0
    )
    total_fraud_loss = fraud_loss_allow + fraud_loss_verify + fraud_loss_throttle

    # FP losses
    fp_block    = float(blocked_legit.sum() * s.C_fp_block)
    fp_verify   = float(verify_legit.sum() * s.C_fp_friction_verify)
    fp_throttle = float(throttle_legit.sum() * s.C_fp_friction_throttle)
    total_fp_loss = fp_block + fp_verify + fp_throttle

    # Operational costs
    n_verify   = int((actions == "VERIFY").sum())
    n_throttle = int((actions == "THROTTLE").sum())
    op_cost = n_verify * s.C_verify_fixed + n_throttle * s.C_throttle_fixed

    total_loss = total_fraud_loss + total_fp_loss + op_cost

    action_dist = {a: int((actions == a).sum()) for a in ["ALLOW", "VERIFY", "THROTTLE", "BLOCK"]}

    return {
        "scenario": s.name,
        "fraud_loss_total":    round(total_fraud_loss, 2),
        "fraud_loss_allowed":  round(fraud_loss_allow, 2),
        "fraud_loss_verify":   round(fraud_loss_verify, 2),
        "fraud_loss_throttle": round(fraud_loss_throttle, 2),
        "fp_loss_total":       round(total_fp_loss, 2),
        "fp_loss_block":       round(fp_block, 2),
        "fp_loss_verify":      round(fp_verify, 2),
        "fp_loss_throttle":    round(fp_throttle, 2),
        "operational_cost":    round(op_cost, 2),
        "total_expected_loss": round(total_loss, 2),
        "action_distribution": action_dist,
        "n_transactions":      int(total),
        "frauds_blocked":      int((fraud_mask & (actions == "BLOCK")).sum()),
        "frauds_allowed":      int(allowed_fraud.sum()),
        "legit_allowed":       int((legit_mask & (actions == "ALLOW")).sum()),
        "legit_blocked":       int(blocked_legit.sum()),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    out_dir = Path("artifacts/risk")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load raw data
    # -----------------------------------------------------------------------
    logger.info("Loading dataset...")
    df_raw, _ = load_train_data()
    train_df, val_df, test_df, _ = chronological_split(df_raw)

    y_train = train_df["isFraud"].values.astype(int)
    y_val   = val_df["isFraud"].values.astype(int)
    y_test  = test_df["isFraud"].values.astype(int)

    txn_amt_train = train_df["TransactionAmt"].values.astype(float)
    txn_amt_val   = val_df["TransactionAmt"].values.astype(float)
    txn_amt_test  = test_df["TransactionAmt"].values.astype(float)

    logger.info("TRAIN N=%d frauds=%d (%.3f%%)", len(y_train), y_train.sum(), 100*y_train.mean())
    logger.info("VAL   N=%d frauds=%d (%.3f%%)", len(y_val),   y_val.sum(),   100*y_val.mean())
    logger.info("TEST  N=%d frauds=%d (%.3f%%)", len(y_test),  y_test.sum(),  100*y_test.mean())

    # -----------------------------------------------------------------------
    # 2. Load cached A_t predictions
    # -----------------------------------------------------------------------
    logger.info("Loading A_t predictions...")
    test_preds = pd.read_parquet("artifacts/models/kaggle_xgb/test_predictions_kaggle_xgb.parquet")

    # Load cached features for TRAIN and VAL, then get predictions
    logger.info("Loading XGBoost model for TRAIN/VAL A_t...")
    wrapper = XGBBaselineWrapper.load("artifacts/models/kaggle_xgb")

    X_train_cache = pd.read_parquet("artifacts/models/kaggle_xgb/cache/X_train_kaggle.parquet")
    a_train = wrapper.model.predict_proba(X_train_cache)[:, 1]
    del X_train_cache
    import gc
    gc.collect()

    X_val_cache = pd.read_parquet("artifacts/models/kaggle_xgb/cache/X_val_kaggle.parquet")
    a_val = wrapper.model.predict_proba(X_val_cache)[:, 1]
    del X_val_cache
    gc.collect()

    a_test = test_preds["A_t"].values

    logger.info("A_t loaded. TRAIN mean=%.4f VAL mean=%.4f TEST mean=%.4f",
                a_train.mean(), a_val.mean(), a_test.mean())

    # -----------------------------------------------------------------------
    # 3. Load G_t signals
    # -----------------------------------------------------------------------
    logger.info("Loading G_t predictions (knowledge graph)...")
    test_graph_df = pd.read_parquet("artifacts/graph/test_graph_features.parquet")

    graph_cache = Path("artifacts/graph/train_val_graph_features.parquet")
    if graph_cache.exists():
        logger.info("Loading cached TRAIN/VAL graph features from %s...", graph_cache)
        tv_graph_df = pd.read_parquet(graph_cache)
        train_len = len(train_df)
        g_train = tv_graph_df.iloc[:train_len]["graph_risk"].values
        g_val   = tv_graph_df.iloc[train_len:]["graph_risk"].values
    else:
        logger.info("Computing TRAIN/VAL graph features (streaming)...")
        from trustgraph.graph.builder import GraphPipelineBuilder
        builder = GraphPipelineBuilder()

        # Stream TRAIN first
        logger.info("  Streaming TRAIN (N = %d)...", len(train_df))
        train_results = builder.process_dataframe_stream(train_df, is_train=True, log_interval=100_000)
        g_train = train_results["graph_risk"].values

        # Stream VAL (with is_train=True so labels accumulate causally for calibration and TEST)
        logger.info("  Streaming VAL (N = %d)...", len(val_df))
        val_results = builder.process_dataframe_stream(val_df, is_train=True, log_interval=40_000)
        g_val = val_results["graph_risk"].values

        # Save lightweight combined cache
        combined = pd.concat([
            train_results[["transaction_id", "graph_risk"]],
            val_results[["transaction_id", "graph_risk"]]
        ], ignore_index=True)
        combined.to_parquet(graph_cache, index=False)
        logger.info("  Saved TRAIN+VAL graph cache to %s", graph_cache)

        del train_results, val_results, combined
        gc.collect()

    # Align test G_t to test_df row order
    g_test = test_graph_df["graph_risk"].values

    logger.info("G_t loaded. TRAIN mean=%.4f VAL mean=%.4f TEST mean=%.4f",
                g_train.mean(), g_val.mean(), g_test.mean())

    # -----------------------------------------------------------------------
    # 4. Calibration evaluation
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Calibration evaluation")
    logger.info("=" * 60)
    calibrator = SignalCalibrator()
    calibrator.fit(
        a_train, g_train, y_train,
        a_val,   g_val,   y_val,
        method="platt",
    )
    cal_summary = calibrator.summary()
    logger.info("Calibration summary: %s", json.dumps(cal_summary, indent=2))

    # Apply calibration to get calibrated signals
    a_train_c, g_train_c = calibrator.transform(a_train, g_train)
    a_val_c,   g_val_c   = calibrator.transform(a_val,   g_val)
    a_test_c,  g_test_c  = calibrator.transform(a_test,  g_test)

    # -----------------------------------------------------------------------
    # 5. Fusion tuning on VALIDATION ONLY
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Fusion parameter tuning (VALIDATION ONLY)")
    logger.info("=" * 60)
    fusion_engine = FusionEngine()
    fusion_engine.tune(a_val_c, g_val_c, y_val, selection_metric="pr_auc")
    fusion_summary = fusion_engine.summary()
    logger.info("Fusion tuning complete: %s", json.dumps(fusion_summary, indent=2))

    best_per_formula = fusion_engine.best_per_formula()
    logger.info("\nBest per formula (VAL PR-AUC):")
    for fname, res in best_per_formula.items():
        logger.info("  %s: params=%s | PR-AUC=%.4f | F1=%.4f | τ=%.4f",
                    fname, res.params, res.val_pr_auc, res.val_f1, res.val_threshold)

    # -----------------------------------------------------------------------
    # 6. FREEZE all parameters
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: Freezing parameters")
    logger.info("=" * 60)

    frozen_params = {
        "frozen": True,
        "calibration": cal_summary,
        "fusion": fusion_summary,
        "formula": fusion_engine.best_formula.value,
        "params": fusion_engine.best_params,
        "threshold_val": fusion_engine.best_threshold,
        "cost_scenarios": list(DEFAULT_SCENARIOS.keys()),
        "evaluation_partition": "TEST",
        "n_test": int(len(y_test)),
        "fraud_test": int(y_test.sum()),
        "legit_test": int((y_test == 0).sum()),
    }
    with open(out_dir / "frozen_params.json", "w") as f:
        json.dump(frozen_params, f, indent=2)
    logger.info("Frozen parameters saved.")

    # -----------------------------------------------------------------------
    # 7. TEST EVALUATION — ALL SYSTEM CONFIGS (one-shot, no iteration)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7: Final TEST evaluation (FROZEN — one-shot)")
    logger.info("=" * 60)

    # Use VAL-selected threshold for classification metrics
    tau = fusion_engine.best_threshold

    # For A_t-only baseline, find its own best threshold on VAL
    from sklearn.metrics import precision_recall_curve
    prec_v, rec_v, thr_v = precision_recall_curve(y_val, a_val_c)
    f1_v = np.where((prec_v + rec_v) == 0, 0, 2*prec_v*rec_v/(prec_v + rec_v))
    best_idx_a = int(np.argmax(f1_v))
    if best_idx_a >= len(thr_v): best_idx_a = len(thr_v) - 1
    tau_a = float(thr_v[best_idx_a])

    # For G_t-only baseline
    prec_v, rec_v, thr_v = precision_recall_curve(y_val, g_val_c)
    f1_v = np.where((prec_v + rec_v) == 0, 0, 2*prec_v*rec_v/(prec_v + rec_v))
    best_idx_g = int(np.argmax(f1_v))
    if best_idx_g >= len(thr_v): best_idx_g = len(thr_v) - 1
    tau_g = float(thr_v[best_idx_g])

    # Compute R_t for each formulation on TEST using frozen params
    r_f1 = fusion_engine.apply_formula(a_test_c, g_test_c, FusionFormula.F1,
                                        fusion_engine.best_per_formula()[FusionFormula.F1.value].params)
    r_f2 = fusion_engine.apply_formula(a_test_c, g_test_c, FusionFormula.F2,
                                        fusion_engine.best_per_formula()[FusionFormula.F2.value].params)
    r_f4 = fusion_engine.apply_formula(a_test_c, g_test_c, FusionFormula.F4,
                                        fusion_engine.best_per_formula()[FusionFormula.F4.value].params)
    r_final = fusion_engine.fuse_batch(a_test_c, g_test_c)

    # Individual VAL thresholds per formula for fair comparison
    def get_val_threshold(scores_val, y_val):
        prec, rec, thr = precision_recall_curve(y_val, scores_val)
        f1 = np.where((prec + rec) == 0, 0, 2*prec*rec/(prec + rec))
        idx = min(int(np.argmax(f1)), len(thr) - 1)
        return float(thr[idx])

    tau_f1  = get_val_threshold(fusion_engine.apply_formula(a_val_c, g_val_c, FusionFormula.F1,
                                fusion_engine.best_per_formula()[FusionFormula.F1.value].params), y_val)
    tau_f2  = get_val_threshold(fusion_engine.apply_formula(a_val_c, g_val_c, FusionFormula.F2,
                                fusion_engine.best_per_formula()[FusionFormula.F2.value].params), y_val)
    tau_f4  = get_val_threshold(fusion_engine.apply_formula(a_val_c, g_val_c, FusionFormula.F4,
                                fusion_engine.best_per_formula()[FusionFormula.F4.value].params), y_val)
    tau_fin = fusion_engine.best_threshold

    test_results = {
        "A_t_baseline":   compute_metrics(y_test, a_test_c,  tau_a),
        "G_t_baseline":   compute_metrics(y_test, g_test_c,  tau_g),
        "F1_additive":    compute_metrics(y_test, r_f1,       tau_f1),
        "F2_conditional": compute_metrics(y_test, r_f2,       tau_f2),
        "F4_conservative":compute_metrics(y_test, r_f4,       tau_f4),
        "final_system":   compute_metrics(y_test, r_final,    tau_fin),
    }

    logger.info("\n=== TEST METRICS COMPARISON ===")
    header = f"{'System':<20} {'ROC-AUC':>8} {'PR-AUC':>8} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>7} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}"
    logger.info(header)
    for name, m in test_results.items():
        logger.info(
            f"{name:<20} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['fpr']:>7.4f} "
            f"{m['tp']:>6d} {m['fp']:>6d} {m['fn']:>6d} {m['tn']:>6d}"
        )

    # -----------------------------------------------------------------------
    # 8. Business loss evaluation under 3 cost scenarios
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 8: Business loss evaluation (3 cost scenarios)")
    logger.info("=" * 60)

    business_results = {}
    for scenario_name, scenario in DEFAULT_SCENARIOS.items():
        cost_model = MerchantCostModel(scenario)
        actions = cost_model.compute_batch_costs(r_final, txn_amt_test)
        losses = compute_business_losses(y_test, r_final, actions, txn_amt_test, scenario)
        business_results[scenario_name] = losses
        logger.info(
            "[%s] Total loss: ₹%.0f | Fraud: ₹%.0f | FP: ₹%.0f | Actions: %s",
            scenario_name,
            losses["total_expected_loss"],
            losses["fraud_loss_total"],
            losses["fp_loss_total"],
            losses["action_distribution"],
        )

    # Also compute A_t-only business losses for comparison
    logger.info("\n--- A_t-only comparison (balanced scenario) ---")
    a_actions_val = DEFAULT_SCENARIOS["balanced"]
    cost_model_bal = MerchantCostModel(DEFAULT_SCENARIOS["balanced"])
    a_only_actions = cost_model_bal.compute_batch_costs(a_test_c, txn_amt_test)
    a_only_losses = compute_business_losses(y_test, a_test_c, a_only_actions, txn_amt_test, DEFAULT_SCENARIOS["balanced"])
    logger.info("A_t-only: Total loss ₹%.0f | Fraud ₹%.0f | FP ₹%.0f | Actions: %s",
                a_only_losses["total_expected_loss"],
                a_only_losses["fraud_loss_total"],
                a_only_losses["fp_loss_total"],
                a_only_losses["action_distribution"])

    # -----------------------------------------------------------------------
    # 9. Example transaction decision
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 9: Example transaction decision")
    logger.info("=" * 60)

    calibrator_ready = calibrator
    calibrator_ready.fitted = True

    decision_engine = DecisionEngine(
        calibrator=calibrator_ready,
        fusion_engine=fusion_engine,
        cost_model=MerchantCostModel(DEFAULT_SCENARIOS["balanced"]),
    )

    # Find a high-risk fraud example
    fraud_indices = np.where(y_test == 1)[0]
    high_risk_idx = fraud_indices[np.argmax(r_final[fraud_indices])]
    ex_txn_id = int(test_df.iloc[high_risk_idx]["TransactionID"])
    ex_a = float(a_test[high_risk_idx])
    ex_g = float(g_test[high_risk_idx])
    ex_amt = float(txn_amt_test[high_risk_idx])

    ex_result = decision_engine.decide(ex_txn_id, ex_a, ex_g, ex_amt)
    logger.info("\n=== EXAMPLE TRANSACTION ===")
    logger.info(ex_result.explanation)

    # -----------------------------------------------------------------------
    # 10. Save all results
    # -----------------------------------------------------------------------
    final_output = {
        "metadata": {
            "evaluation_partition": "TEST",
            "n_test": int(len(y_test)),
            "fraud_test": int(y_test.sum()),
            "legit_test": int((y_test == 0).sum()),
            "base_fraud_rate": round(float(y_test.mean()) * 100, 4),
        },
        "calibration": cal_summary,
        "fusion": {
            "selected_formula": fusion_engine.best_formula.value,
            "selected_params": fusion_engine.best_params,
            "selected_threshold_val": fusion_engine.best_threshold,
            "best_per_formula": {
                k: {
                    "params": v.params,
                    "val_pr_auc": v.val_pr_auc,
                    "val_roc_auc": v.val_roc_auc,
                    "val_f1": v.val_f1,
                    "val_threshold": v.val_threshold,
                }
                for k, v in best_per_formula.items()
            },
        },
        "test_metrics": test_results,
        "business_losses": business_results,
        "a_t_only_losses_balanced": a_only_losses,
        "example_decision": ex_result.to_dict(),
    }

    results_path = out_dir / "phase3_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)

    # Save per-transaction R_t for downstream use
    r_df = pd.DataFrame({
        "TransactionID": test_df["TransactionID"].values,
        "isFraud":       y_test,
        "A_t":           a_test,
        "G_t":           g_test,
        "A_t_cal":       a_test_c,
        "G_t_cal":       g_test_c,
        "R_t":           r_final,
        "TransactionAmt": txn_amt_test,
    })
    r_df.to_parquet(out_dir / "test_risk_scores.parquet", index=False)

    logger.info("\nAll results saved to %s", out_dir)
    logger.info("PHASE 3 EVALUATION COMPLETE.")
    return final_output


if __name__ == "__main__":
    results = main()
