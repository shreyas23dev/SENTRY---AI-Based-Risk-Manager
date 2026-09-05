"""
evaluate_fusion.py — TRUSTGRAPH Phase 3.1 Final Held-Out Test Evaluation
==========================================================================

Evaluates the frozen conditional risk fusion rule on the held-out TEST partition.
Generates:
  - 4-way system comparison (B0, B1, B2, B3_new)
  - Detailed coverage-aware analysis (Gt=0 vs Gt>0, Pt=0 vs Pt>0)
  - results/fusion_predictions.csv
  - artifacts/fusion/ JSON results and publication plots
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trustgraph.baseline import config as base_cfg
from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.preprocessing import BaselinePreprocessor
from trustgraph.temporal.entity_tracker import resolve_entity_key, EntityTemporalRiskEngine
from trustgraph.relational.graph_engine import (
    GraphParameters,
    LightweightRelationalGraph,
    process_partition,
)
from trustgraph.fusion.config import (
    FUSION_DIR, FUSION_PLOTS_DIR, RESULTS_DIR,
    BASELINE_THRESHOLD, TEMPORAL_THRESHOLD, RELATIONAL_THRESHOLD, ENTITY_KEY_TYPE,
)
from trustgraph.fusion.fusion_engine import apply_fusion_rule, verify_fusion_invariance
from trustgraph.fusion.evaluator import compute_system_metrics, compute_coverage_aware_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_fusion")

FUSION_DIR.mkdir(parents=True, exist_ok=True)
FUSION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_frozen_parameters() -> Dict[str, Any]:
    params_path = FUSION_DIR / "parameters.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found at {params_path}. Run tune_fusion.py first.")
    with open(params_path) as f:
        return json.load(f)


def prepare_full_data():
    """Load data, compute exact A_t, P_t, G_t across all partitions."""
    logger.info("Loading full dataset...")
    df, _ = load_train_data()
    train_df, val_df, test_df, split_meta = chronological_split(df)
    del df

    # Resolve entity proxy
    logger.info("Resolving entity proxies (%s)...", ENTITY_KEY_TYPE)
    for part, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        part["entity_proxy"] = resolve_entity_key(part, key_type=ENTITY_KEY_TYPE)

    # 1. Point-wise LightGBM Model Inference: A_t
    logger.info("Loading frozen LightGBM model and preprocessor...")
    model = BaselineModel.load(base_cfg.MODEL_DIR / "lgbm_model.pkl")
    preprocessor = BaselinePreprocessor.load(base_cfg.PREPROCESSING_DIR)

    logger.info("Generating A_t scores for all partitions...")
    train_df["A_t"] = model.predict_risk(preprocessor.transform(train_df))
    val_df["A_t"]   = model.predict_risk(preprocessor.transform(val_df))
    test_df["A_t"]  = model.predict_risk(preprocessor.transform(test_df))

    # 2. Entity Temporal Risk: P_t
    frozen_cfg = load_frozen_parameters()["upstream_frozen"]
    temp_cfg = frozen_cfg["temporal_params"]
    logger.info("Generating P_t scores across TRAIN -> VAL -> TEST...")
    temp_engine = EntityTemporalRiskEngine(
        beta=temp_cfg["beta"], gamma=temp_cfg["gamma"],
        lambda_=temp_cfg["lambda"], delta=temp_cfg["delta"],
    )
    for part in [train_df, val_df, test_df]:
        ents = part["entity_proxy"].values
        scores = part["A_t"].values
        P_arr = np.zeros(len(part), dtype=float)
        for i in range(len(part)):
            _, p_val = temp_engine.step(str(ents[i]), float(scores[i]))
            P_arr[i] = p_val
        part["P_t"] = P_arr

    # 3. Persistent Relational Graph: G_t
    rel_cfg = frozen_cfg["relational_params"]
    logger.info("Generating G_t scores across TRAIN -> VAL -> TEST...")
    rel_params = GraphParameters(
        k_attr_max=rel_cfg["k_attr_max"],
        window_sec=rel_cfg["window_sec"],
        d_ref=rel_cfg["d_ref"],
        v_ref=rel_cfg["v_ref"],
        w_D=rel_cfg["w_D"],
        w_V=rel_cfg["w_V"],
        relational_attrs=("DeviceInfo",),
    )
    graph_engine = LightweightRelationalGraph(rel_params)
    graph_engine.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, graph_engine)
    process_partition(val_df, graph_engine)
    test_records = process_partition(test_df, graph_engine)
    test_df["G_t"] = np.array([r.G_t for r in test_records], dtype=float)

    logger.info("Data preparation complete. TEST rows=%d, frauds=%d (%.4f%%)",
                len(test_df), int(test_df["isFraud"].sum()), 100 * test_df["isFraud"].mean())
    return test_df, split_meta


def evaluate_test(test_df: pd.DataFrame, frozen_info: Dict[str, Any]) -> Dict[str, Any]:
    y_test = test_df["isFraud"].values.astype(int)
    A_t = test_df["A_t"].values.astype(float)
    P_t = test_df["P_t"].values.astype(float)
    G_t = test_df["G_t"].values.astype(float)

    rule_name = frozen_info["rule_name"]
    params = frozen_info["params"]
    tau_comb = params["tau_comb"]
    rule_params = {k: v for k, v in params.items() if k != "tau_comb"}

    # Apply selected fusion rule
    R_t = apply_fusion_rule(rule_name, A_t, P_t, G_t, rule_params)
    passed, diag = verify_fusion_invariance(A_t, P_t, G_t, R_t)
    assert passed, f"TEST invariance check FAILED: {diag}"

    # Predictions
    b0_pred = (A_t >= BASELINE_THRESHOLD).astype(int)
    b1_pred = ((A_t >= BASELINE_THRESHOLD) | (P_t >= TEMPORAL_THRESHOLD)).astype(int)
    b2_pred = ((A_t >= BASELINE_THRESHOLD) | (G_t >= RELATIONAL_THRESHOLD)).astype(int)
    b3_pred = (R_t >= tau_comb).astype(int)

    # Old weighted-average B3 (rejected formulation)
    r_old = 0.40 * A_t + 0.30 * P_t + 0.30 * G_t
    b3_old_pred = (r_old >= 0.40).astype(int)

    # Metrics
    b0_m = compute_system_metrics(y_test, b0_pred, A_t)
    b1_m = compute_system_metrics(
        y_test, b1_pred,
        base_tp=b0_m["tp"], base_fp=b0_m["fp"],
        base_recall=b0_m["recall"], base_fpr=b0_m["fpr"],
    )
    b2_m = compute_system_metrics(
        y_test, b2_pred,
        base_tp=b0_m["tp"], base_fp=b0_m["fp"],
        base_recall=b0_m["recall"], base_fpr=b0_m["fpr"],
    )
    b3_m = compute_system_metrics(
        y_test, b3_pred, R_t,
        base_tp=b0_m["tp"], base_fp=b0_m["fp"],
        base_recall=b0_m["recall"], base_fpr=b0_m["fpr"],
    )
    b3_old_m = compute_system_metrics(
        y_test, b3_old_pred, r_old,
        base_tp=b0_m["tp"], base_fp=b0_m["fp"],
        base_recall=b0_m["recall"], base_fpr=b0_m["fpr"],
    )

    # Coverage-aware analysis
    cov_results = compute_coverage_aware_metrics(
        y_test, A_t, P_t, G_t, R_t, b0_pred, b3_pred,
    )

    test_df["R_t"] = R_t
    test_df["baseline_prediction"]   = b0_pred
    test_df["temporal_prediction"]   = b1_pred
    test_df["relational_prediction"] = b2_pred
    test_df["combined_prediction"]   = b3_pred

    return {
        "b0": b0_m,
        "b1": b1_m,
        "b2": b2_m,
        "b3_new": b3_m,
        "b3_old_rejected": b3_old_m,
        "coverage_analysis": cov_results,
        "invariance_diagnostics": diag,
        "test_df": test_df,
    }


def generate_plots(test_df: pd.DataFrame, eval_results: Dict[str, Any]):
    y_true = test_df["isFraud"].values
    A_t = test_df["A_t"].values
    R_t = test_df["R_t"].values
    G_t = test_df["G_t"].values
    P_t = test_df["P_t"].values

    # Plot 1: Baseline A_t vs Combined R_t Distribution by Fraud Label
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(A_t[y_true == 0], bins=50, alpha=0.6, label="Legitimate", color="steelblue", density=True)
    axes[0].hist(A_t[y_true == 1], bins=50, alpha=0.6, label="Fraudulent", color="tomato", density=True)
    axes[0].axvline(BASELINE_THRESHOLD, color="black", linestyle="--", label=f"Threshold ({BASELINE_THRESHOLD:.4f})")
    axes[0].set_title("Baseline Point-wise Score (A_t) on Test")
    axes[0].set_xlabel("A_t")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(R_t[y_true == 0], bins=50, alpha=0.6, label="Legitimate", color="steelblue", density=True)
    axes[1].hist(R_t[y_true == 1], bins=50, alpha=0.6, label="Fraudulent", color="tomato", density=True)
    axes[1].axvline(BASELINE_THRESHOLD, color="black", linestyle="--", label=f"Threshold ({BASELINE_THRESHOLD:.4f})")
    axes[1].set_title("Fused Combined Score (R_t) on Test")
    axes[1].set_xlabel("R_t")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FUSION_PLOTS_DIR / "11_fusion_score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: 4-Way Comparison Bar Chart
    systems = ["B0 (Baseline)", "B1 (+Temporal)", "B2 (+Relational)", "B3_new (Fused)", "B3_old (Rejected)"]
    f1_vals = [
        eval_results["b0"]["f1"],
        eval_results["b1"]["f1"],
        eval_results["b2"]["f1"],
        eval_results["b3_new"]["f1"],
        eval_results["b3_old_rejected"]["f1"],
    ]
    rec_vals = [
        eval_results["b0"]["recall"],
        eval_results["b1"]["recall"],
        eval_results["b2"]["recall"],
        eval_results["b3_new"]["recall"],
        eval_results["b3_old_rejected"]["recall"],
    ]
    prec_vals = [
        eval_results["b0"]["precision"],
        eval_results["b1"]["precision"],
        eval_results["b2"]["precision"],
        eval_results["b3_new"]["precision"],
        eval_results["b3_old_rejected"]["precision"],
    ]

    x = np.arange(len(systems))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, f1_vals, width, label="F1", color="steelblue")
    ax.bar(x, rec_vals, width, label="Recall", color="mediumseagreen")
    ax.bar(x + width, prec_vals, width, label="Precision", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("4-Way System Comparison on Held-Out Test Set")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FUSION_PLOTS_DIR / "12_fusion_4way_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Coverage-Aware Fraud Recovery by Context Slice
    cov = eval_results["coverage_analysis"]
    slice_names = ["overall", "relational_active_Gt", "relational_zero_Gt", "contextualized_any_active", "uncontextualized_zero_both"]
    labels = ["Overall (100%)", "Active G_t (7.6%)", "Zero G_t (92.4%)", "Any Context (8.4%)", "Zero Context (91.6%)"]
    rec_gains = [cov[s]["delta_recall"] * 100 for s in slice_names]
    extra_fps = [cov[s]["extra_false_positives"] for s in slice_names]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    color = "mediumseagreen"
    ax1.set_xlabel("Context Subpopulation")
    ax1.set_ylabel("Recall Gain over B0 (pp)", color=color)
    bars = ax1.bar(np.arange(len(labels)) - 0.15, rec_gains, width=0.3, color=color, label="Recall Gain (pp)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=15)

    ax2 = ax1.twinx()
    color = "tomato"
    ax2.set_ylabel("Additional False Positives", color=color)
    bars2 = ax2.bar(np.arange(len(labels)) + 0.15, extra_fps, width=0.3, color=color, label="Extra FPs")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Coverage-Aware Impact: Recall Gain vs Additional False Positives")
    plt.tight_layout()
    plt.savefig(FUSION_PLOTS_DIR / "13_coverage_aware_impact.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Plots saved to %s", FUSION_PLOTS_DIR)


def main():
    t0 = time.time()
    frozen_info = load_frozen_parameters()
    logger.info("Frozen fusion parameters: %s", frozen_info)

    test_df, split_meta = prepare_full_data()
    eval_res = evaluate_test(test_df, frozen_info)
    elapsed = time.time() - t0

    # Save results/fusion_predictions.csv
    out_cols = [
        "TransactionID", "TransactionDT", "isFraud",
        "A_t", "P_t", "G_t", "R_t",
        "baseline_prediction", "temporal_prediction",
        "relational_prediction", "combined_prediction",
    ]
    pred_path = RESULTS_DIR / "fusion_predictions.csv"
    test_df[out_cols].to_csv(pred_path, index=False)
    logger.info("Fusion predictions saved -> %s (%d rows)", pred_path, len(test_df))

    # Save artifacts/fusion/test_results.json
    test_results_out = {
        "frozen_parameters": frozen_info,
        "split_meta": split_meta,
        "test_metrics": {
            "B0_baseline": eval_res["b0"],
            "B1_temporal": eval_res["b1"],
            "B2_relational": eval_res["b2"],
            "B3_new_fused": eval_res["b3_new"],
            "B3_old_rejected": eval_res["b3_old_rejected"],
        },
        "invariance_diagnostics": eval_res["invariance_diagnostics"],
        "evaluation_elapsed_sec": round(elapsed, 1),
    }
    test_results_path = FUSION_DIR / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump(test_results_out, f, indent=2)
    logger.info("Test results saved -> %s", test_results_path)

    # Save artifacts/fusion/coverage_analysis.json
    cov_path = FUSION_DIR / "coverage_analysis.json"
    with open(cov_path, "w") as f:
        json.dump(eval_res["coverage_analysis"], f, indent=2)
    logger.info("Coverage analysis saved -> %s", cov_path)

    # Save artifacts/fusion/reproducibility.json
    repro = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "phase": "Phase 3.1 Conditional Risk Fusion",
        "selected_rule": frozen_info["rule_name"],
        "formula": frozen_info["formula"],
        "params": frozen_info["params"],
        "data_split": split_meta,
        "random_seed": 42,
        "python_version": sys.version,
    }
    repro_path = FUSION_DIR / "reproducibility.json"
    with open(repro_path, "w") as f:
        json.dump(repro, f, indent=2)
    logger.info("Reproducibility saved -> %s", repro_path)

    # Generate plots
    generate_plots(test_df, eval_res)

    # Final console summary
    logger.info("\n=======================================================")
    logger.info("TRUSTGRAPH PHASE 3.1 — HELD-OUT TEST EVALUATION RESULTS")
    logger.info("=======================================================")
    for s_name, m in [
        ("B0 (Baseline)", eval_res["b0"]),
        ("B1 (Temporal)", eval_res["b1"]),
        ("B2 (Relational)", eval_res["b2"]),
        ("B3_new (Fused)", eval_res["b3_new"]),
        ("B3_old (Rejected)", eval_res["b3_old_rejected"]),
    ]:
        d_frauds = m.get("additional_frauds_recovered", 0)
        d_fps = m.get("additional_false_positives", 0)
        rec_gain = m.get("recall_gain_over_b0", 0.0)
        fpr_chg = m.get("fpr_change_over_b0", 0.0)
        logger.info("  %-18s  F1=%.6f  Prec=%.6f  Rec=%.6f  FPR=%.6f  dFrauds=%+d  dFP=%+d  dRec=%+.4f  dFPR=%+.4f",
                    s_name, m["f1"], m["precision"], m["recall"], m["fpr"],
                    d_frauds, d_fps, rec_gain, fpr_chg)
    logger.info("Total elapsed: %.1f s", elapsed)


if __name__ == "__main__":
    main()
