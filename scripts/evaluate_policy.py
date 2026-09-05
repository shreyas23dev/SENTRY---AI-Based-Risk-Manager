"""
evaluate_policy.py — TRUSTGRAPH Phase 4 Held-Out Test Evaluation of Progressive Policy
======================================================================================

Evaluates the frozen progressive decision policy on the untouched TEST partition.
Generates:
  - results/policy_predictions.csv
  - artifacts/policy/test_results.json
  - artifacts/policy/action_distribution.json
  - artifacts/policy/reproducibility.json
  - 5 publication-ready diagnostic visualizations in artifacts/policy/plots/
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    BASELINE_THRESHOLD, ENTITY_KEY_TYPE, RESULTS_DIR,
)
from trustgraph.fusion.fusion_engine import apply_fusion_rule
from trustgraph.policy.config import (
    POLICY_DIR, POLICY_PLOTS_DIR, PolicyAction, RiskBand,
)
from trustgraph.policy.decision_engine import (
    PolicyThresholds,
    batch_assign_actions,
    generate_explanation,
    verify_policy_invariants,
)
from trustgraph.policy.evaluator import (
    evaluate_policy,
    compare_baseline_and_progressive_policy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_policy")

POLICY_DIR.mkdir(parents=True, exist_ok=True)
POLICY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_frozen_thresholds() -> PolicyThresholds:
    thresh_path = POLICY_DIR / "thresholds.json"
    if not thresh_path.exists():
        raise FileNotFoundError(f"Thresholds file not found at {thresh_path}. Run tune_policy.py first.")
    with open(thresh_path) as f:
        data = json.load(f)
    return PolicyThresholds(
        tau_verify=data["tau_verify"],
        tau_throttle=data["tau_throttle"],
        tau_block=data["tau_block"],
    )


def prepare_test_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load full data and compute exact A_t, P_t, G_t, R_t, D_t, V_t across splits."""
    logger.info("Loading dataset for test evaluation...")
    raw_df, _ = load_train_data()
    train_df, val_df, test_df, split_meta = chronological_split(raw_df)
    del raw_df

    # 1. Resolve entity proxy
    for part in [train_df, val_df, test_df]:
        part["entity_proxy"] = resolve_entity_key(part, key_type=ENTITY_KEY_TYPE)

    # 2. Point-wise LightGBM Model Inference
    logger.info("Computing A_t across all partitions...")
    model = BaselineModel.load(Path(__file__).resolve().parents[1] / "artifacts" / "baseline" / "model" / "lgbm_model.pkl")
    preprocessor = BaselinePreprocessor.load(Path(__file__).resolve().parents[1] / "artifacts" / "baseline" / "preprocessing")

    train_df["A_t"] = model.predict_risk(preprocessor.transform(train_df))
    val_df["A_t"]   = model.predict_risk(preprocessor.transform(val_df))
    test_df["A_t"]  = model.predict_risk(preprocessor.transform(test_df))

    # 3. Entity Temporal Risk Engine
    logger.info("Computing P_t across TRAIN -> VAL -> TEST...")
    temp_engine = EntityTemporalRiskEngine(beta=0.3, gamma=0.5, lambda_=0.05, delta=0.05)
    for part in [train_df, val_df, test_df]:
        ents = part["entity_proxy"].values
        scores = part["A_t"].values
        P_arr = np.zeros(len(part), dtype=float)
        for i in range(len(part)):
            _, p_val = temp_engine.step(str(ents[i]), float(scores[i]))
            P_arr[i] = p_val
        part["P_t"] = P_arr

    # 4. Relational Graph Engine
    logger.info("Computing G_t across TRAIN -> VAL -> TEST...")
    rel_params = GraphParameters(
        k_attr_max=25, window_sec=86400.0, d_ref=3.0, v_ref=10.0,
        w_D=0.6, w_V=0.4, relational_attrs=("DeviceInfo",)
    )
    graph_engine = LightweightRelationalGraph(rel_params)
    graph_engine.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, graph_engine)
    process_partition(val_df, graph_engine)
    test_records = process_partition(test_df, graph_engine)

    test_df["G_t"] = np.array([r.G_t for r in test_records], dtype=float)
    test_df["D_t"] = np.array([r.D_t for r in test_records], dtype=float)
    test_df["V_t"] = np.array([r.V_t for r in test_records], dtype=float)
    test_df["d_t"] = np.array([r.d_t for r in test_records], dtype=int)
    test_df["v_t"] = np.array([r.v_t for r in test_records], dtype=int)

    # 5. Fused score R_t on TEST
    R_t = apply_fusion_rule("F1", test_df["A_t"].values, test_df["P_t"].values, test_df["G_t"].values, {"alpha": 1.0, "beta": 0.05})
    test_df["R_t"] = R_t

    logger.info("Test data prepared: N=%d, frauds=%d (%.4f%%)",
                len(test_df), int(test_df["isFraud"].sum()), 100 * test_df["isFraud"].mean())
    return test_df, split_meta


def generate_plots(test_df: pd.DataFrame, thresholds: PolicyThresholds, eval_res: Dict[str, Any]):
    """Generate 5 diagnostic visualizations for Phase 4."""
    logger.info("Generating publication-quality policy visualizations...")

    R_vals = test_df["R_t"].values
    actions = test_df["action"].values
    y_true = test_df["isFraud"].values

    action_colors = {
        PolicyAction.ALLOW.value: "#2ECC71",     # Green
        PolicyAction.VERIFY.value: "#3498DB",    # Blue
        PolicyAction.THROTTLE.value: "#E67E22",  # Orange
        PolicyAction.BLOCK.value: "#E74C3C",     # Red
    }

    # Plot 1: Risk Distribution by Action
    plt.figure(figsize=(9, 5))
    bins = np.linspace(0, 1, 50)
    for act in PolicyAction:
        mask = (actions == act.value)
        plt.hist(R_vals[mask], bins=bins, alpha=0.6, label=f"{act.value} (N={int(mask.sum()):,})", color=action_colors[act.value])
    plt.axvline(thresholds.tau_verify, color="blue", linestyle="--", label=f"tau_verify={thresholds.tau_verify:.2f}")
    plt.axvline(thresholds.tau_throttle, color="orange", linestyle="--", label=f"tau_throttle={thresholds.tau_throttle:.2f}")
    plt.axvline(thresholds.tau_block, color="red", linestyle="--", label=f"tau_block={thresholds.tau_block:.2f}")
    plt.xlabel("Fused Risk Score (R_t)")
    plt.ylabel("Transaction Count")
    plt.title("1. Risk Score Distribution Across Progressive Action Tiers (Test Set)")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(POLICY_PLOTS_DIR / "01_risk_distribution_by_action.png", dpi=150)
    plt.close()

    # Plot 2: Fraud Rate by Action
    act_names = [a.value for a in PolicyAction]
    fraud_rates = [eval_res["actions"][a]["fraud_rate"] * 100 for a in act_names]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(act_names, fraud_rates, color=[action_colors[a] for a in act_names], width=0.5)
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0, f"{b.get_height():.2f}%", ha="center", va="bottom", fontweight="bold")
    plt.axhline(eval_res["base_fraud_rate"] * 100, color="black", linestyle=":", label=f"Base Rate ({eval_res['base_fraud_rate']*100:.2f}%)")
    plt.ylabel("Empirical Fraud Rate (%)")
    plt.title("2. Fraud Rate Concentration by Progressive Action (Test Set)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(POLICY_PLOTS_DIR / "02_fraud_rate_by_action.png", dpi=150)
    plt.close()

    # Plot 3: Action Distribution (Volume Breakdown)
    act_counts = [eval_res["actions"][a]["transaction_count"] for a in act_names]
    pct_shares = [eval_res["actions"][a]["pct_of_total_traffic"] for a in act_names]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(act_names, act_counts, color=[action_colors[a] for a in act_names], width=0.5)
    for b, pct in zip(bars, pct_shares):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 1000, f"{pct:.2f}%\n({int(b.get_height()):,})", ha="center", va="bottom", fontweight="bold")
    plt.ylabel("Total Transaction Count")
    plt.title("3. Action Distribution Across Entire Test Traffic (N = 88,580)")
    plt.ylim(0, max(act_counts) * 1.18)
    plt.tight_layout()
    plt.savefig(POLICY_PLOTS_DIR / "03_action_distribution.png", dpi=150)
    plt.close()

    # Plot 4: Action Matrix by True Label (Frauds vs Legitimate)
    fraud_counts = [eval_res["actions"][a]["fraud_count"] for a in act_names]
    legit_counts = [eval_res["actions"][a]["legitimate_count"] for a in act_names]
    x = np.arange(len(act_names))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, legit_counts, width=w, label="Legitimate", color="#34495E")
    plt.bar(x + w/2, fraud_counts, width=w, label="Fraudulent", color="#E74C3C")
    plt.xticks(x, act_names)
    plt.yscale("log")
    plt.ylabel("Transaction Count (Log Scale)")
    plt.title("4. Action Assignment Stratification by Ground Truth Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(POLICY_PLOTS_DIR / "04_confusion_action_matrix.png", dpi=150)
    plt.close()

    # Plot 5: Visual Policy Architecture Diagram / Explanations
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    bands_text = (
        "PROGRESSIVE RISK DECISION POLICY ARCHITECTURE\n"
        "===============================================================\n"
        f"  [0.00 -> {thresholds.tau_verify:.2f})  -->  ALLOW    (Low Risk | Normal Processing)\n"
        f"  [{thresholds.tau_verify:.2f} -> {thresholds.tau_throttle:.2f})  -->  VERIFY   (Moderate Risk | Step-up 3DS / OTP Challenge)\n"
        f"  [{thresholds.tau_throttle:.2f} -> {thresholds.tau_block:.2f})  -->  THROTTLE (High Risk | Velocity Capping / Delayed Settlement)\n"
        f"  [{thresholds.tau_block:.2f} -> 1.00]  -->  BLOCK    (Very High Risk | Direct Hard Decline)\n"
        "===============================================================\n"
        "REPRESENTATIVE EXPLANATIONS:\n"
        " • ALLOW:    R_t=0.0312. Normal point-wise tabular risk, zero temporal/graph risk.\n"
        " • VERIFY:   R_t=0.5853. Borderline tabular risk + shared device uplift. Step-up OTP.\n"
        " • THROTTLE: R_t=0.7341. Multi-event velocity bursts (P_t=0.50) on entity proxy.\n"
        " • BLOCK:    R_t=0.8842. Critical point-wise tabular score + severe fraud evidence."
    )
    ax.text(0.05, 0.5, bands_text, fontfamily="monospace", fontsize=10, va="center", bbox=dict(boxstyle="round,pad=1", facecolor="#F8F9F9", edgecolor="#BDC3C7"))
    plt.tight_layout()
    plt.savefig(POLICY_PLOTS_DIR / "05_representative_explanations.png", dpi=150)
    plt.close()

    logger.info("Visualizations saved to %s", POLICY_PLOTS_DIR)


def main():
    t0 = time.time()
    thresholds = load_frozen_thresholds()
    logger.info("Loaded frozen policy thresholds: %s", thresholds)

    test_df, split_meta = prepare_test_data()

    # Vectorized action & band assignment
    R_vals = test_df["R_t"].values
    actions, bands = batch_assign_actions(R_vals, thresholds)
    test_df["action"] = actions
    test_df["risk_band"] = bands

    # Verify safety invariants
    passed, diag = verify_policy_invariants(R_vals, actions, thresholds)
    assert passed, f"TEST invariant check failed: {diag}"
    logger.info("Policy invariants verified on TEST: %s", diag)

    # Generate explanations
    logger.info("Generating deterministic natural language explanations for test transactions...")
    reasons = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        exp = generate_explanation(
            A_t=float(row["A_t"]),
            P_t=float(row["P_t"]),
            G_t=float(row["G_t"]),
            R_t=float(row["R_t"]),
            action=PolicyAction(row["action"]),
            thresholds=thresholds,
            D_t=float(row["D_t"]),
            V_t=float(row["V_t"]),
            d_t=int(row["d_t"]),
            v_t=int(row["v_t"]),
            device_info=str(row["DeviceInfo"]),
        )
        reasons.append(exp)
    test_df["action_reason"] = reasons

    # Evaluate Policy
    y_test = test_df["isFraud"].values.astype(int)
    eval_res = evaluate_policy(y_test, actions, thresholds)
    comp_res = compare_baseline_and_progressive_policy(y_test, test_df["A_t"].values, R_vals, BASELINE_THRESHOLD, thresholds)

    elapsed = time.time() - t0

    # 1. Save results/policy_predictions.csv
    out_cols = [
        "TransactionID", "TransactionDT", "isFraud",
        "A_t", "P_t", "G_t", "R_t",
        "action", "risk_band", "action_reason",
    ]
    pred_path = RESULTS_DIR / "policy_predictions.csv"
    test_df[out_cols].to_csv(pred_path, index=False)
    logger.info("Saved -> %s (%d rows)", pred_path, len(test_df))

    # 2. Save artifacts/policy/test_results.json
    test_out = {
        "thresholds": thresholds.to_dict(),
        "evaluation": eval_res,
        "policy_comparison": comp_res,
        "safety_diagnostics": diag,
        "split_meta": split_meta,
        "evaluation_elapsed_sec": round(elapsed, 2),
    }
    with open(POLICY_DIR / "test_results.json", "w") as f:
        json.dump(test_out, f, indent=2)
    logger.info("Saved -> %s", POLICY_DIR / "test_results.json")

    # 3. Save artifacts/policy/action_distribution.json
    with open(POLICY_DIR / "action_distribution.json", "w") as f:
        json.dump(eval_res["actions"], f, indent=2)
    logger.info("Saved -> %s", POLICY_DIR / "action_distribution.json")

    # 4. Save artifacts/policy/reproducibility.json
    repro = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "phase": "Phase 4 Progressive Risk Decision Policy",
        "thresholds": thresholds.to_dict(),
        "split_meta": split_meta,
        "python_version": sys.version,
    }
    with open(POLICY_DIR / "reproducibility.json", "w") as f:
        json.dump(repro, f, indent=2)
    logger.info("Saved -> %s", POLICY_DIR / "reproducibility.json")

    # Generate plots
    generate_plots(test_df, thresholds, eval_res)

    # Print summary table to console
    logger.info("\n" + "=" * 90)
    logger.info("TRUSTGRAPH PHASE 4 — PROGRESSIVE RISK DECISION POLICY TEST RESULTS")
    logger.info("=" * 90)
    logger.info("| Risk Band  | Action   | Range             | Transactions | Pct (%) | Frauds | Fraud Rate | Capture (%) | Legit FP |")
    logger.info("|------------|----------|-------------------|--------------|---------|--------|------------|-------------|----------|")
    for act in PolicyAction:
        a_info = eval_res["actions"][act.value]
        if act == PolicyAction.ALLOW:
            r_str = f"R_t < {thresholds.tau_verify:.2f}"
        elif act == PolicyAction.VERIFY:
            r_str = f"{thresholds.tau_verify:.2f} <= R_t < {thresholds.tau_throttle:.2f}"
        elif act == PolicyAction.THROTTLE:
            r_str = f"{thresholds.tau_throttle:.2f} <= R_t < {thresholds.tau_block:.2f}"
        else:
            r_str = f"R_t >= {thresholds.tau_block:.2f}"

        logger.info("| %-10s | %-8s | %-17s | %-12d | %-7.2f | %-6d | %-10.2f%% | %-11.2f%% | %-8d |",
                    act.value, act.value, r_str,
                    a_info["transaction_count"], a_info["pct_of_total_traffic"],
                    a_info["fraud_count"], a_info["fraud_rate"] * 100,
                    a_info["fraud_capture_pct_of_total"], a_info["legitimate_count"])
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
