"""
analyze_incremental_relational.py — Phase 3.2 Incremental Contribution Analysis
================================================================================

Performs the formal incremental set analysis of relational risk (G_t)
beyond entity-scoped temporal risk (P_t) on the held-out TEST partition.

Outputs:
  - artifacts/fusion/incremental_relational_analysis.json
  - results/incremental_relational_cases.csv
  - artifacts/fusion/plots/14_incremental_set_analysis.png
  - artifacts/fusion/plots/15_signal_correlation_matrix.png
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.fusion.config import (
    FUSION_DIR, FUSION_PLOTS_DIR, RESULTS_DIR,
    BASELINE_THRESHOLD, TEMPORAL_THRESHOLD, RELATIONAL_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_incremental")

FUSION_DIR.mkdir(parents=True, exist_ok=True)
FUSION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load test predictions, relational predictions, and raw transaction features."""
    logger.info("Loading predictions and test metadata...")
    fusion_path = RESULTS_DIR / "fusion_predictions.csv"
    rel_path = RESULTS_DIR / "relational_predictions.csv"

    if not fusion_path.exists() or not rel_path.exists():
        raise FileNotFoundError("Prediction files not found. Ensure Phase 3 and 3.1 have executed.")

    df_fusion = pd.read_csv(fusion_path)
    df_rel = pd.read_csv(rel_path)

    # Load raw test transactions to get DeviceInfo and other attributes
    df_raw, _ = load_train_data()
    _, _, test_raw, _ = chronological_split(df_raw)
    del df_raw

    # Merge into a single comprehensive test analysis dataframe
    df = df_fusion.merge(
        df_rel[["TransactionID", "entity_proxy", "d_t", "v_t", "D_t", "V_t"]],
        on="TransactionID", how="left"
    )
    df = df.merge(
        test_raw[["TransactionID", "DeviceInfo", "card1", "addr1", "P_emaildomain"]],
        on="TransactionID", how="left"
    )

    logger.info("Merged analysis dataframe: %d rows", len(df))
    return df


def perform_set_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute formal set operations on detected frauds and false positives."""
    is_fraud = df["isFraud"].values.astype(bool)
    is_legit = ~is_fraud

    b0_pos = df["baseline_prediction"].values.astype(bool)
    b1_pos = df["temporal_prediction"].values.astype(bool)
    b2_pos = df["relational_prediction"].values.astype(bool)
    b3_pos = df["combined_prediction"].values.astype(bool)

    # Fraud Detection Sets (True Positives)
    S0 = set(df[is_fraud & b0_pos]["TransactionID"])
    S1 = set(df[is_fraud & b1_pos]["TransactionID"])
    S2 = set(df[is_fraud & b2_pos]["TransactionID"])
    S3 = set(df[is_fraud & b3_pos]["TransactionID"])
    All_Frauds = set(df[is_fraud]["TransactionID"])

    # False Positive Sets
    FP0 = set(df[is_legit & b0_pos]["TransactionID"])
    FP1 = set(df[is_legit & b1_pos]["TransactionID"])
    FP2 = set(df[is_legit & b2_pos]["TransactionID"])
    FP3 = set(df[is_legit & b3_pos]["TransactionID"])
    All_Legit = set(df[is_legit]["TransactionID"])

    # Set operations - Frauds
    b3_incremental_over_b1 = S3 - S1  # |B3 \ B1| (G_t incremental frauds)
    b1_incremental_over_b0 = S1 - S0  # |B1 \ B0|
    b2_incremental_over_b0 = S2 - S0  # |B2 \ B0|
    b3_incremental_over_b0 = S3 - S0  # |B3 \ B0|
    b3_intersect_b1        = S3 & S1
    b3_minus_b2            = S3 - S2
    b2_minus_b1            = S2 - S1
    b0_missed_frauds       = All_Frauds - S0
    b3_missed_frauds       = All_Frauds - S3

    # Set operations - False Positives
    fp3_incremental_over_fp1 = FP3 - FP1  # |FP3 \ FP1| (G_t incremental FPs)
    fp1_incremental_over_fp0 = FP1 - FP0  # |FP1 \ FP0|
    fp2_incremental_over_fp0 = FP2 - FP0  # |FP2 \ FP0|
    fp3_incremental_over_fp0 = FP3 - FP0  # |FP3 \ FP0|

    # System comparison table data
    total_frauds = len(All_Frauds)
    total_legit = len(All_Legit)

    systems = {
        "B0": {"tp": len(S0), "fp": len(FP0)},
        "B1": {"tp": len(S1), "fp": len(FP1)},
        "B2": {"tp": len(S2), "fp": len(FP2)},
        "B3": {"tp": len(S3), "fp": len(FP3)},
    }
    for k, v in systems.items():
        v["recall"] = round(v["tp"] / total_frauds, 6)
        v["precision"] = round(v["tp"] / (v["tp"] + v["fp"]), 6) if (v["tp"] + v["fp"]) > 0 else 0.0
        v["fpr"] = round(v["fp"] / total_legit, 6)
        v["f1"] = round(2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"]), 6) if (v["precision"] + v["recall"]) > 0 else 0.0

    return {
        "fraud_set_counts": {
            "total_frauds": total_frauds,
            "S0_baseline_detected": len(S0),
            "S1_temporal_detected": len(S1),
            "S2_relational_detected": len(S2),
            "S3_fused_detected": len(S3),
            "S1_minus_S0_temporal_boost": len(b1_incremental_over_b0),
            "S2_minus_S0_relational_boost": len(b2_incremental_over_b0),
            "S3_minus_S0_fused_boost": len(b3_incremental_over_b0),
            "S3_intersect_S1": len(b3_intersect_b1),
            "S3_minus_S1_incremental_G_t": len(b3_incremental_over_b1),
            "S3_minus_S2": len(b3_minus_b2),
            "S2_minus_S1": len(b2_minus_b1),
            "B0_missed_frauds": len(b0_missed_frauds),
            "B3_missed_frauds": len(b3_missed_frauds),
        },
        "false_positive_set_counts": {
            "total_legitimate": total_legit,
            "FP0_baseline": len(FP0),
            "FP1_temporal": len(FP1),
            "FP2_relational": len(FP2),
            "FP3_fused": len(FP3),
            "FP1_minus_FP0_temporal_extra": len(fp1_incremental_over_fp0),
            "FP2_minus_FP0_relational_extra": len(fp2_incremental_over_fp0),
            "FP3_minus_FP0_fused_extra": len(fp3_incremental_over_fp0),
            "FP3_minus_FP1_incremental_G_t_extra": len(fp3_incremental_over_fp1),
        },
        "system_comparison_table": systems,
        "incremental_fraud_ids": sorted(list(b3_incremental_over_b1)),
        "incremental_fp_ids": sorted(list(fp3_incremental_over_fp1)),
    }


def perform_coverage_aware_incremental(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze incremental impact in the 4 disjoint context regimes."""
    P_zero = (df["P_t"] == 0.0)
    P_active = (df["P_t"] > 0.0)
    G_zero = (df["G_t"] == 0.0)
    G_active = (df["G_t"] > 0.0)

    regimes = {
        "regime_1_P0_G0_uncontextualized": P_zero & G_zero,
        "regime_2_Pactive_G0_temporal_only": P_active & G_zero,
        "regime_3_P0_Gactive_relational_only": P_zero & G_active,
        "regime_4_Pactive_Gactive_joint_context": P_active & G_active,
    }

    out = {}
    for name, mask in regimes.items():
        sub = df[mask]
        n_tx = len(sub)
        n_fraud = int(sub["isFraud"].sum())
        n_legit = n_tx - n_fraud

        is_fraud = sub["isFraud"].values.astype(bool)
        is_legit = ~is_fraud

        b0_pos = sub["baseline_prediction"].values.astype(bool)
        b1_pos = sub["temporal_prediction"].values.astype(bool)
        b2_pos = sub["relational_prediction"].values.astype(bool)
        b3_pos = sub["combined_prediction"].values.astype(bool)

        s0 = int(np.sum(is_fraud & b0_pos))
        s1 = int(np.sum(is_fraud & b1_pos))
        s2 = int(np.sum(is_fraud & b2_pos))
        s3 = int(np.sum(is_fraud & b3_pos))

        fp0 = int(np.sum(is_legit & b0_pos))
        fp1 = int(np.sum(is_legit & b1_pos))
        fp2 = int(np.sum(is_legit & b2_pos))
        fp3 = int(np.sum(is_legit & b3_pos))

        out[name] = {
            "transaction_count": n_tx,
            "pct_of_total": round(100.0 * n_tx / len(df), 2),
            "fraud_count": n_fraud,
            "legit_count": n_legit,
            "fraud_prevalence": round(n_fraud / n_sub, 6) if (n_sub := n_tx) > 0 else 0.0,
            "frauds_detected_B0": s0,
            "frauds_detected_B1": s1,
            "frauds_detected_B2": s2,
            "frauds_detected_B3": s3,
            "incremental_frauds_B3_minus_B1": s3 - s1,
            "false_positives_B0": fp0,
            "false_positives_B1": fp1,
            "false_positives_B2": fp2,
            "false_positives_B3": fp3,
            "incremental_fps_B3_minus_B1": fp3 - fp1,
            "mean_A_t": float(sub["A_t"].mean()) if len(sub) > 0 else 0.0,
            "mean_P_t": float(sub["P_t"].mean()) if len(sub) > 0 else 0.0,
            "mean_G_t": float(sub["G_t"].mean()) if len(sub) > 0 else 0.0,
            "mean_R_t": float(sub["R_t"].mean()) if len(sub) > 0 else 0.0,
        }
    return out


def perform_correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate correlation, mutual information, and overlap metrics among A_t, P_t, G_t."""
    A_t = df["A_t"].values
    P_t = df["P_t"].values
    G_t = df["G_t"].values
    y_true = df["isFraud"].values

    # Pearson & Spearman correlations
    pearson_AP, _ = stats.pearsonr(A_t, P_t)
    pearson_AG, _ = stats.pearsonr(A_t, G_t)
    pearson_PG, _ = stats.pearsonr(P_t, G_t)

    spearman_AP, _ = stats.spearmanr(A_t, P_t)
    spearman_AG, _ = stats.spearmanr(A_t, G_t)
    spearman_PG, _ = stats.spearmanr(P_t, G_t)

    # Activity overlap
    p_active = (P_t > 0.0)
    g_active = (G_t > 0.0)
    jaccard_active = float(np.sum(p_active & g_active) / np.sum(p_active | g_active)) if np.sum(p_active | g_active) > 0 else 0.0

    # High risk overlap
    p_high = (P_t >= TEMPORAL_THRESHOLD)
    g_high = (G_t >= RELATIONAL_THRESHOLD)
    jaccard_high = float(np.sum(p_high & g_high) / np.sum(p_high | g_high)) if np.sum(p_high | g_high) > 0 else 0.0

    # Conditional probabilities
    prob_G_given_P = float(np.mean(g_active[p_active])) if np.sum(p_active) > 0 else 0.0
    prob_G_given_notP = float(np.mean(g_active[~p_active])) if np.sum(~p_active) > 0 else 0.0
    prob_P_given_G = float(np.mean(p_active[g_active])) if np.sum(g_active) > 0 else 0.0
    prob_P_given_notG = float(np.mean(p_active[~g_active])) if np.sum(~g_active) > 0 else 0.0

    return {
        "pearson_correlations": {
            "r_A_P": round(float(pearson_AP), 6),
            "r_A_G": round(float(pearson_AG), 6),
            "r_P_G": round(float(pearson_PG), 6),
        },
        "spearman_rank_correlations": {
            "rho_A_P": round(float(spearman_AP), 6),
            "rho_A_G": round(float(spearman_AG), 6),
            "rho_P_G": round(float(spearman_PG), 6),
        },
        "signal_overlap": {
            "total_transactions": len(df),
            "P_active_count": int(np.sum(p_active)),
            "G_active_count": int(np.sum(g_active)),
            "both_active_count": int(np.sum(p_active & g_active)),
            "jaccard_active_context": round(jaccard_active, 6),
            "jaccard_high_risk_threshold": round(jaccard_high, 6),
            "P_G_given_P_active": round(prob_G_given_P, 6),
            "P_G_given_P_zero": round(prob_G_given_notP, 6),
            "P_P_given_G_active": round(prob_P_given_G, 6),
            "P_P_given_G_zero": round(prob_P_given_notG, 6),
        },
    }


def inspect_incremental_cases(df: pd.DataFrame, incremental_ids: List[int]) -> List[Dict[str, Any]]:
    """Detailed case-by-case audit of frauds in S3 \ S1."""
    sub = df[df["TransactionID"].isin(incremental_ids)].copy()
    cases = []
    for _, row in sub.iterrows():
        txn_id = int(row["TransactionID"])
        A = float(row["A_t"])
        P = float(row["P_t"])
        G = float(row["G_t"])
        R = float(row["R_t"])
        D = float(row["D_t"])
        V = float(row["V_t"])
        d = int(row["d_t"])
        v = int(row["v_t"])
        device = str(row["DeviceInfo"])
        entity = str(row["entity_proxy"])

        # Determine primary recovery driver
        # F1 formula: R_t = clip(A_t + 1.0*P_t + 0.05*G_t, 0, 1)
        # S3 \ S1 means B1 (A_t >= tau or P_t >= 0.70) was False, but R_t >= tau was True.
        # This occurs either because:
        #  (a) G_t provided the incremental push (when P_t = 0 or low)
        #  (b) P_t provided sub-threshold additive push (P_t in (0, 0.70)) + G_t joint push
        boost_from_P = 1.0 * P
        boost_from_G = 0.05 * G
        gap_to_threshold = BASELINE_THRESHOLD - A

        if P == 0.0 and G > 0.0:
            driver = "Pure Relational Boost (G_t push with zero P_t)"
        elif P > 0.0 and G > 0.0:
            driver = "Joint Contextual Boost (P_t additive + G_t additive)"
        elif P > 0.0 and G == 0.0:
            driver = "Sub-threshold Temporal Boost (P_t < 0.70 but A_t + P_t >= tau)"
        else:
            driver = "Other"

        case_entry = {
            "TransactionID": txn_id,
            "entity_proxy": entity,
            "DeviceInfo": device,
            "A_t": round(A, 6),
            "P_t": round(P, 6),
            "G_t": round(G, 6),
            "R_t": round(R, 6),
            "d_t_connected_entities": d,
            "v_t_new_relationships": v,
            "D_t_norm_degree": round(D, 6),
            "V_t_norm_velocity": round(V, 6),
            "boost_from_temporal": round(boost_from_P, 6),
            "boost_from_relational": round(boost_from_G, 6),
            "gap_to_baseline_threshold": round(gap_to_threshold, 6),
            "recovery_driver": driver,
        }
        cases.append(case_entry)

    return cases


def generate_plots(set_analysis: Dict[str, Any], corr_analysis: Dict[str, Any], cov_analysis: Dict[str, Any]):
    """Generate diagnostic plots for Phase 3.2."""
    # Plot 14: Incremental Set Comparison & Coverage Regimes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar 1: Fraud recovery breakdown across systems
    systems = ["B0", "B1", "B2", "B3"]
    tps = [set_analysis["system_comparison_table"][s]["tp"] for s in systems]
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#7ED321"]
    bars = axes[0].bar(systems, tps, color=colors, width=0.5)
    for bar, val in zip(bars, tps):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                     f"{val}", ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylabel("Detected Frauds on Test (Total=3083)")
    axes[0].set_title("Fraud Detections Across Systems")
    axes[0].set_ylim(0, max(tps) * 1.15)

    # Bar 2: Incremental Frauds & FPs by Context Regime
    regimes = ["P=0, G=0\n(Uncontext)", "P>0, G=0\n(Temp Only)", "P=0, G>0\n(Rel Only)", "P>0, G>0\n(Joint)"]
    inc_frauds = [
        cov_analysis["regime_1_P0_G0_uncontextualized"]["incremental_frauds_B3_minus_B1"],
        cov_analysis["regime_2_Pactive_G0_temporal_only"]["incremental_frauds_B3_minus_B1"],
        cov_analysis["regime_3_P0_Gactive_relational_only"]["incremental_frauds_B3_minus_B1"],
        cov_analysis["regime_4_Pactive_Gactive_joint_context"]["incremental_frauds_B3_minus_B1"],
    ]
    inc_fps = [
        cov_analysis["regime_1_P0_G0_uncontextualized"]["incremental_fps_B3_minus_B1"],
        cov_analysis["regime_2_Pactive_G0_temporal_only"]["incremental_fps_B3_minus_B1"],
        cov_analysis["regime_3_P0_Gactive_relational_only"]["incremental_fps_B3_minus_B1"],
        cov_analysis["regime_4_Pactive_Gactive_joint_context"]["incremental_fps_B3_minus_B1"],
    ]
    x = np.arange(len(regimes))
    w = 0.35
    axes[1].bar(x - w/2, inc_frauds, width=w, label="Incremental Frauds (B3 \\ B1)", color="mediumseagreen")
    axes[1].bar(x + w/2, inc_fps, width=w, label="Incremental FPs (B3 \\ B1)", color="tomato")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regimes)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Incremental Performance by Context Regime (B3 vs B1)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FUSION_PLOTS_DIR / "14_incremental_set_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 15: Signal Correlation Matrix
    corr_matrix = np.array([
        [1.0, corr_analysis["pearson_correlations"]["r_A_P"], corr_analysis["pearson_correlations"]["r_A_G"]],
        [corr_analysis["pearson_correlations"]["r_A_P"], 1.0, corr_analysis["pearson_correlations"]["r_P_G"]],
        [corr_analysis["pearson_correlations"]["r_A_G"], corr_analysis["pearson_correlations"]["r_P_G"], 1.0],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix, cmap="Blues", vmin=0, vmax=1)
    labels = ["A_t (Point)", "P_t (Temporal)", "G_t (Relational)"]
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr_matrix[i, j]:.4f}", ha="center", va="center", color="black" if corr_matrix[i, j] < 0.6 else "white")
    plt.colorbar(im, ax=ax)
    plt.title("Pearson Correlation Matrix (A_t, P_t, G_t) on Test")
    plt.tight_layout()
    plt.savefig(FUSION_PLOTS_DIR / "15_signal_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Plots saved to %s", FUSION_PLOTS_DIR)


def main():
    logger.info("=" * 60)
    logger.info("TRUSTGRAPH Phase 3.2 — Incremental Relational Contribution")
    logger.info("=" * 60)

    df = load_data()

    set_res = perform_set_analysis(df)
    cov_res = perform_coverage_aware_incremental(df)
    corr_res = perform_correlation_analysis(df)
    cases_res = inspect_incremental_cases(df, set_res["incremental_fraud_ids"])

    # Save results/incremental_relational_cases.csv
    cases_df = pd.DataFrame(cases_res)
    cases_path = RESULTS_DIR / "incremental_relational_cases.csv"
    cases_df.to_csv(cases_path, index=False)
    logger.info("Incremental cases saved -> %s (%d cases)", cases_path, len(cases_df))

    # Save artifacts/fusion/incremental_relational_analysis.json
    full_output = {
        "research_question": "Incremental contribution of relational risk G_t beyond temporal risk P_t",
        "set_analysis": set_res,
        "coverage_regimes": cov_res,
        "correlation_and_overlap": corr_res,
        "incremental_recovered_cases_summary": {
            "total_incremental_frauds_B3_minus_B1": len(cases_res),
            "drivers_breakdown": {
                driver: int(sum(1 for c in cases_res if c["recovery_driver"] == driver))
                for driver in sorted(list(set(c["recovery_driver"] for c in cases_res)))
            },
            "cases": cases_res,
        },
    }
    out_json_path = FUSION_DIR / "incremental_relational_analysis.json"
    with open(out_json_path, "w") as f:
        json.dump(full_output, f, indent=2)
    logger.info("Incremental analysis saved -> %s", out_json_path)

    # Generate plots
    generate_plots(set_res, corr_res, cov_res)

    # Print summary table
    logger.info("\n" + "=" * 65)
    logger.info("SUMMARY TABLE: 4-WAY SYSTEM COMPARISON ON TEST SET")
    logger.info("=" * 65)
    logger.info("| System | Recall   | Precision | F1       | FPR      | Fraud Cases |")
    logger.info("|--------|----------|-----------|----------|----------|-------------|")
    for s in ["B0", "B1", "B2", "B3"]:
        row = set_res["system_comparison_table"][s]
        logger.info("| %-6s | %-8.4f | %-9.4f | %-8.4f | %-8.4f | %-11d |",
                    s, row["recall"], row["precision"], row["f1"], row["fpr"], row["tp"])
    logger.info("-" * 65)
    logger.info("Incremental Frauds (B3 \\ B1):         %d", set_res["fraud_set_counts"]["S3_minus_S1_incremental_G_t"])
    logger.info("Incremental False Positives (B3 \\ B1): %d", set_res["false_positive_set_counts"]["FP3_minus_FP1_incremental_G_t_extra"])
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
