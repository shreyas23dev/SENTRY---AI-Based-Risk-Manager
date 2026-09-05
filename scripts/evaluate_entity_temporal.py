"""
evaluate_entity_temporal.py — Phase 2.1 Final Evaluation on Held-Out Test Partition
====================================================================================

Usage:
    python scripts/evaluate_entity_temporal.py

Evaluates:
  1. Baseline (B0, Frozen LightGBM)
  2. Global Temporal Memory (B1_global, Phase 2 Negative Control)
  3. Entity-Scoped Temporal Memory (B1_entity, Phase 2.1)
  4. Entity Key Ablation on Test (card_email, card_composite, card1, card_addr)

Generates 6 Visualizations:
  1. Global temporal state example
  2. Entity-scoped temporal state trace for representative entity
  3. Controlled slow-burn attack sequence (A_t, E_t, P_t)
  4. 3-way Comparative Precision-Recall Curve (B0 vs B1_global vs B1_entity)
  5. Distribution of transactions per tracked entity
  6. Distribution of entity-scoped temporal states (E_t, P_t)

LEAKAGE / FREEZE SAFETY:
  - Consumes existing frozen A_t from results/test_predictions.csv directly.
  - Phase 1 baseline artifacts and model are NOT modified.
"""

import json
import logging
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.baseline import config as base_cfg
from trustgraph.temporal.engine import TemporalRiskEngine
from trustgraph.temporal.entity_tracker import EntityTemporalRiskEngine, resolve_entity_key
from trustgraph.temporal.evaluator import (
    evaluate_temporal_comparison,
    make_temporal_prediction,
    plot_temporal_sequence,
    plot_slow_burn_demonstration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_entity_temporal")

OUT_DIR   = base_cfg.PROJECT_ROOT / "artifacts" / "temporal_entity"
PLOTS_DIR = OUT_DIR / "plots"
PRED_PATH = base_cfg.PROJECT_ROOT / "results" / "temporal_entity_predictions.csv"

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.size":        11,
}


def main():
    t_start = time.perf_counter()
    logger.info("=" * 75)
    logger.info("TRUSTGRAPH Phase 2.1 — Final Entity-Scoped Temporal Evaluation")
    logger.info("=" * 75)

    # 1. Load frozen baseline predictions
    test_pred_path = base_cfg.PROJECT_ROOT / "results" / "test_predictions.csv"
    if not test_pred_path.exists():
        raise FileNotFoundError(f"Missing {test_pred_path}. Run Phase 1 baseline first.")

    logger.info("Loading frozen Phase 1 predictions from: %s", test_pred_path)
    df_test = pd.read_csv(test_pred_path)
    logger.info("Loaded %d rows. Columns: %s", len(df_test), list(df_test.columns))

    # Join raw card/address attributes for entity resolution
    raw_cols = ["TransactionID", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "P_emaildomain"]
    raw_txn = pd.read_csv(base_cfg.TRAIN_TRANSACTION_CSV, usecols=raw_cols)
    df_test = df_test.merge(raw_txn, on="TransactionID", how="left")
    del raw_txn

    # Ensure chronological order
    if not df_test["TransactionDT"].is_monotonic_increasing:
        df_test = df_test.sort_values("TransactionDT").reset_index(drop=True)

    y_test   = df_test["isFraud"].values
    A_test   = df_test["A_t"].values
    base_pred = df_test["baseline_prediction"].values

    # Load frozen baseline threshold
    with open(base_cfg.ARTIFACTS_DIR / "threshold.json") as f:
        base_thr_data = json.load(f)
    base_thr = float(base_thr_data["threshold"])

    # Load tuned parameters
    with open(OUT_DIR / "parameters.json") as f:
        tuning_data = json.load(f)
    params = tuning_data["selected_parameters"]
    selected_key = params["entity_key"]
    beta     = float(params["beta"])
    gamma    = float(params["gamma"])
    lambda_  = float(params["lambda"])
    delta    = float(params["delta"])
    temp_thr = float(params["temporal_threshold"])

    logger.info("\nFrozen Phase 2.1 Selected Configuration:")
    logger.info("  Entity Key:                    %s", selected_key)
    logger.info("  Baseline Threshold (tau_base): %.6f", base_thr)
    logger.info("  Beta (EMA):                    %.4f", beta)
    logger.info("  Gamma (Suspicion):             %.4f", gamma)
    logger.info("  Lambda (Accumulation):         %.4f", lambda_)
    logger.info("  Delta (Decay):                 %.4f", delta)
    logger.info("  Temporal Threshold (tau_temp): %.4f", temp_thr)

    # ------------------------------------------------------------------
    # 2. CONFIGURATION A: Global Temporal Stream (Phase 2 Negative Control)
    # ------------------------------------------------------------------
    with open(base_cfg.PROJECT_ROOT / "artifacts" / "temporal" / "parameters.json") as f:
        g_data = json.load(f)
    gp = g_data["selected_parameters"]
    g_engine = TemporalRiskEngine(beta=float(gp["beta"]), gamma=float(gp["gamma"]), lambda_=float(gp["lambda"]), delta=float(gp["delta"]))
    E_global, P_global = g_engine.process_stream(A_test)

    metrics_global = evaluate_temporal_comparison(
        y_test, A_test, E_global, P_global,
        baseline_threshold=base_thr,
        temporal_threshold=float(gp["temporal_threshold"]),
        partition_name="test_global_control",
    )

    # ------------------------------------------------------------------
    # 3. CONFIGURATION B: Entity-Scoped Temporal Memory (Phase 2.1)
    # ------------------------------------------------------------------
    logger.info("\nResolving entity key '%s' on test partition...", selected_key)
    ent_series = resolve_entity_key(df_test, key_type=selected_key)
    unresolved_mask = ent_series.str.startswith("unresolved_")
    n_unresolved = int(unresolved_mask.sum())
    unique_entities = int(ent_series.nunique())

    logger.info("Entity statistics for %s on Test:", selected_key)
    logger.info("  Unique entities: %d", unique_entities)
    logger.info("  Unresolved transactions: %d (%.2f%%)", n_unresolved, 100 * n_unresolved / len(df_test))

    # Measure runtime
    ent_engine = EntityTemporalRiskEngine(beta=beta, gamma=gamma, lambda_=lambda_, delta=delta)
    t_ent_start = time.perf_counter()
    E_entity, P_entity = ent_engine.process_stream(df_test, ent_series, score_col="A_t")
    t_ent_elapsed = time.perf_counter() - t_ent_start
    ent_throughput = len(A_test) / t_ent_elapsed
    ent_latency_ms = (t_ent_elapsed / len(A_test)) * 1000

    metrics_entity = evaluate_temporal_comparison(
        y_test, A_test, E_entity, P_entity,
        baseline_threshold=base_thr,
        temporal_threshold=temp_thr,
        partition_name="test_entity_scoped",
    )

    # ------------------------------------------------------------------
    # 4. Mandatory 3-Way Comparison Output
    # ------------------------------------------------------------------
    b0 = metrics_global["B0_baseline"]
    b1_g = metrics_global["B1_temporal"]
    d_g = metrics_global["comparative_delta"]
    b1_e = metrics_entity["B1_temporal"]
    d_e = metrics_entity["comparative_delta"]

    logger.info("=" * 80)
    logger.info("MANDATORY 3-WAY TEST EVALUATION COMPARISON:")
    logger.info("Metric                  B0 (Baseline)    B1 (Global Control)    B1 (Entity-Scoped)")
    logger.info("-" * 80)
    logger.info("ROC-AUC                 %10.6f       %10.6f            %10.6f", b0["roc_auc"], b1_g["roc_auc"], b1_e["roc_auc"])
    logger.info("PR-AUC (Avg Precision)  %10.6f       %10.6f            %10.6f", b0["pr_auc"], b1_g["pr_auc"], b1_e["pr_auc"])
    logger.info("Precision               %10.6f       %10.6f            %10.6f", b0["precision"], b1_g["precision"], b1_e["precision"])
    logger.info("Recall                  %10.6f       %10.6f            %10.6f", b0["recall"], b1_g["recall"], b1_e["recall"])
    logger.info("F1 Score                %10.6f       %10.6f            %10.6f", b0["f1"], b1_g["f1"], b1_e["f1"])
    logger.info("FPR                     %10.6f       %10.6f            %10.6f", b0["fpr"], b1_g["fpr"], b1_e["fpr"])
    logger.info("FNR                     %10.6f       %10.6f            %10.6f", b0["fnr"], b1_g["fnr"], b1_e["fnr"])
    logger.info("-" * 80)
    logger.info("Frauds Recovered                 -               %4d                  %4d", d_g["additional_frauds_recovered"], d_e["additional_frauds_recovered"])
    logger.info("%% Baseline FNs Recovered         -              %5.2f%%               %5.2f%%", d_g["pct_baseline_fn_recovered"], d_e["pct_baseline_fn_recovered"])
    logger.info("Additional False Positives       -               %4d                  %4d", d_g["additional_false_positives"], d_e["additional_false_positives"])
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # 5. Entity Key Ablation on Held-Out Test
    # ------------------------------------------------------------------
    logger.info("\nRunning Entity Key Ablation on TEST partition...")
    ablation_test = {}
    with open(OUT_DIR / "entity_key_ablation.json") as f:
        val_ablation = json.load(f)

    for k_name in ["card_email", "card_composite", "card1", "card_addr"]:
        k_series = resolve_entity_key(df_test, key_type=k_name)
        k_unresolved = int(k_series.str.startswith("unresolved_").sum())
        k_unique = int(k_series.nunique())

        kp = val_ablation[k_name]["best_params"]
        eng = EntityTemporalRiskEngine(beta=kp["beta"], gamma=kp["gamma"], lambda_=kp["lambda"], delta=kp["delta"])
        e_k, p_k = eng.process_stream(df_test, k_series, score_col="A_t")
        m_k = evaluate_temporal_comparison(y_test, A_test, e_k, p_k, base_thr, kp["temporal_threshold"], partition_name=f"test_{k_name}")

        ablation_test[k_name] = {
            "unique_entities_test": k_unique,
            "unresolved_count_test": k_unresolved,
            "unresolved_pct_test": round(100 * k_unresolved / len(df_test), 2),
            "test_metrics": m_k["B1_temporal"],
            "comparative_delta": m_k["comparative_delta"],
            "parameters": kp,
        }

        logger.info("  Key [%s]: Unique=%d, Unresolved=%.1f%% -> F1=%.6f, Recov=%d (%.2f%%), ExtraFP=%d, Prec=%.4f, Rec=%.4f, FPR=%.4f",
                    k_name, k_unique, 100 * k_unresolved / len(df_test),
                    m_k["B1_temporal"]["f1"],
                    m_k["comparative_delta"]["additional_frauds_recovered"],
                    m_k["comparative_delta"]["pct_baseline_fn_recovered"],
                    m_k["comparative_delta"]["additional_false_positives"],
                    m_k["B1_temporal"]["precision"], m_k["B1_temporal"]["recall"], m_k["B1_temporal"]["fpr"])

    # ------------------------------------------------------------------
    # 6. Generate 6 Required Visualizations
    # ------------------------------------------------------------------
    logger.info("\nGenerating 6 required Phase 2.1 visualizations...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Global temporal state trace example
    sample_slice = slice(100, 160)
    plot_temporal_sequence(
        A_test[sample_slice], E_global[sample_slice], P_global[sample_slice], y_test[sample_slice],
        base_thr, float(gp["temporal_threshold"]),
        PLOTS_DIR / "1_global_temporal_state.png",
        title="1. Global Temporal Memory Trace (Negative Control)",
    )

    # Plot 2: Entity-scoped temporal state for representative entity with multiple transactions
    entity_counts = ent_series[~unresolved_mask].value_counts()
    rep_entity = entity_counts.index[0]
    ent_idx = np.where(ent_series.values == rep_entity)[0]
    if len(ent_idx) > 30:
        ent_idx = ent_idx[:30]

    plot_temporal_sequence(
        A_test[ent_idx], E_entity[ent_idx], P_entity[ent_idx], y_test[ent_idx],
        base_thr, temp_thr,
        PLOTS_DIR / "2_entity_temporal_state_trace.png",
        title=f"2. Entity-Scoped Temporal State Trace (Entity: {rep_entity})",
    )

    # Plot 3: Controlled Slow-Burn Attack Sequence
    slow_burn_results = plot_slow_burn_demonstration(
        gamma=gamma, lambda_=lambda_, delta=delta, beta=beta,
        baseline_thr=base_thr, temporal_thr=temp_thr,
        save_path=PLOTS_DIR / "3_slow_burn_attack_sequence.png",
    )

    # Plot 4: 3-Way Precision-Recall Curve Comparison
    prec_b0, rec_b0, _ = precision_recall_curve(y_test, A_test)
    prec_b1_g, rec_b1_g, _ = precision_recall_curve(y_test, np.maximum(A_test, P_global))
    prec_b1_e, rec_b1_e, _ = precision_recall_curve(y_test, np.maximum(A_test, P_entity))

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(rec_b0, prec_b0, lw=2, color="#1f77b4", label=f"B0 Baseline (AP={b0['pr_auc']:.4f})")
        ax.plot(rec_b1_g, prec_b1_g, lw=1.5, color="#ff7f0e", linestyle="--", label=f"B1 Global Temporal (AP={b1_g['pr_auc']:.4f})")
        ax.plot(rec_b1_e, prec_b1_e, lw=2, color="#2ca02c", linestyle="-.", label=f"B1 Entity Temporal (AP={b1_e['pr_auc']:.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("4. Precision-Recall Curve (B0 vs B1_global vs B1_entity)")
        ax.legend(loc="upper right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "4_pr_curve_comparison.png", dpi=150)
        plt.close(fig)

    # Plot 5: Distribution of number of transactions per tracked entity
    counts_all = ent_series.value_counts().values
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(counts_all, bins=np.logspace(0, 3, 40), color="#3470a3", edgecolor="black", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Transactions per Entity (Log Scale)")
        ax.set_ylabel("Number of Entities")
        ax.set_title(f"5. Distribution of Transactions per Entity ({selected_key})")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "5_entity_transaction_distribution.png", dpi=150)
        plt.close(fig)

    # Plot 6: Distribution of entity-scoped temporal states (E_t, P_t)
    with plt.style.context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.hist(E_entity[y_test == 0], bins=50, alpha=0.6, color="#1f77b4", density=True, label="Legit")
        ax1.hist(E_entity[y_test == 1], bins=50, alpha=0.7, color="#d62728", density=True, label="Fraud")
        ax1.set_title(r"EMA Evidence $E_t$ Distribution")
        ax1.set_xlabel(r"$E_t$")
        ax1.legend()

        ax2.hist(P_entity[y_test == 0], bins=50, alpha=0.6, color="#1f77b4", density=True, label="Legit")
        ax2.hist(P_entity[y_test == 1], bins=50, alpha=0.7, color="#d62728", density=True, label="Fraud")
        ax2.set_title(r"Persistent Accumulator $P_t$ Distribution")
        ax2.set_xlabel(r"$P_t$")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "6_temporal_state_distributions.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 7. Save Output Predictions & Metrics
    # ------------------------------------------------------------------
    df_out = pd.DataFrame({
        "TransactionID":       df_test["TransactionID"],
        "TransactionDT":       df_test["TransactionDT"],
        "entity_id":           ent_series,
        "isFraud":             df_test["isFraud"],
        "A_t":                 df_test["A_t"],
        "E_t":                 E_entity,
        "P_t":                 P_entity,
        "baseline_prediction": base_pred,
        "temporal_prediction": make_temporal_prediction(A_test, P_entity, base_thr, temp_thr),
    })
    df_out.to_csv(PRED_PATH, index=False)
    logger.info("Saved entity temporal predictions → %s (%d rows)", PRED_PATH, len(df_out))

    # Full metrics JSON
    full_metrics = {
        "baseline_metrics": b0,
        "global_temporal_metrics": b1_g,
        "entity_temporal_metrics": b1_e,
        "delta_global_vs_baseline": d_g,
        "delta_entity_vs_baseline": d_e,
        "entity_key_ablation_test": ablation_test,
        "selected_configuration": params,
        "runtime_performance": {
            "total_test_transactions": len(df_test),
            "unique_entities": unique_entities,
            "unresolved_transactions": n_unresolved,
            "processing_time_seconds": round(t_ent_elapsed, 4),
            "throughput_txn_per_s": round(ent_throughput, 1),
            "per_transaction_latency_ms": round(ent_latency_ms, 6),
        },
        "slow_burn_results": slow_burn_results,
    }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    with open(OUT_DIR / "config.json", "w") as f:
        json.dump({
            "phase": 2.1,
            "name": "Entity-Scoped Temporal Risk Memory",
            "selected_entity_key": selected_key,
            "parameters": params,
            "baseline_threshold": base_thr,
            "bounds": {"E_t": [0.0, 1.0], "P_t": [0.0, 1.0]},
            "complexity": {"time_per_transaction": "O(1)", "memory_per_entity": "O(1)"},
        }, f, indent=2)

    with open(OUT_DIR / "reproducibility.json", "w") as f:
        json.dump({
            "dataset": "IEEE-CIS Fraud Detection (DOI: 10.21227/y5e7-wp63)",
            "input_file": str(test_pred_path),
            "output_file": str(PRED_PATH),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)

    logger.info("All Phase 2.1 artifacts saved to: %s", OUT_DIR)
    logger.info("Evaluation completed in %.2f s", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
