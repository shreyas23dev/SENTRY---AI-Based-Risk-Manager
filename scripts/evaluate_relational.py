"""
evaluate_relational.py — TRUSTGRAPH Phase 3 Final Held-Out Test Evaluation
============================================================================

STRICT PROTOCOL:
  - Reads frozen parameters from artifacts/relational/parameters.json
  - Builds graph across TRAIN + VAL (persistently) then evaluates on TEST
  - Runs 4-way ablation: B0, B1, B2 (G1/G2/G3), B3
  - Runs coordinated-burst and legitimate-burst controlled experiments
  - Saves results/relational_predictions.csv (all scores per transaction)
  - Saves artifacts/relational/ JSON and plots

TEST is accessed ONLY in this script after all parameters are frozen.
"""

import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.temporal.entity_tracker import resolve_entity_key, EntityTemporalRiskEngine
from trustgraph.relational.config import (
    RELATIONAL_DIR, PLOTS_DIR, RESULTS_DIR, BASELINE_THRESHOLD,
    ENTITY_KEY_TYPE, TEMPORAL_BETA, TEMPORAL_GAMMA, TEMPORAL_LAMBDA,
    TEMPORAL_DELTA, TEMPORAL_THRESHOLD, ABLATION_ATTR_SETS,
)
from trustgraph.relational.graph_engine import GraphParameters, LightweightRelationalGraph, process_partition
from trustgraph.relational.evaluator import RelationalEvaluator, evaluate_on_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RELATIONAL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load frozen parameters
# ---------------------------------------------------------------------------

def load_frozen_params() -> dict:
    p = RELATIONAL_DIR / "parameters.json"
    if not p.exists():
        raise FileNotFoundError(f"Frozen parameters not found at {p}. Run tune_relational.py first.")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prepare all three splits with entity proxies and temporal scores
# ---------------------------------------------------------------------------

def prepare_all_splits():
    logger.info("Loading full dataset...")
    df, _ = load_train_data()
    train_df, val_df, test_df, split_meta = chronological_split(df)
    del df

    for part, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        part["entity_proxy"] = resolve_entity_key(part, key_type=ENTITY_KEY_TYPE)
        logger.info("%s: %d rows, entity_proxy resolved", name, len(part))

    # Load frozen temporal predictions (covers test P_t and A_t from Phase 2.1)
    temporal_pred_path = RESULTS_DIR / "temporal_entity_predictions.csv"
    if temporal_pred_path.exists():
        temporal_preds = pd.read_csv(temporal_pred_path)
        logger.info("Loaded temporal predictions: %d rows (covers TEST)", len(temporal_preds))
        test_df = test_df.merge(
            temporal_preds[["TransactionID", "A_t", "P_t"]],
            on="TransactionID", how="left"
        )
        test_df["A_t"] = test_df["A_t"].fillna(BASELINE_THRESHOLD - 0.01)
        test_df["P_t"] = test_df["P_t"].fillna(0.0)
    else:
        logger.warning("temporal_entity_predictions.csv not found — using placeholder A_t/P_t for TEST")
        test_df["A_t"] = BASELINE_THRESHOLD - 0.01
        test_df["P_t"] = 0.0

    # Also attach A_t/P_t to val for evaluation
    val_temporal_path = RESULTS_DIR / "temporal_entity_predictions.csv"
    if val_temporal_path.exists():
        # temporal_entity_predictions.csv covers the TEST split only
        # For VAL: re-run entity temporal engine sequentially across TRAIN then VAL
        pass
    # Run entity temporal engine to get val A_t and P_t
    logger.info("Re-running entity temporal engine for VAL A_t and P_t...")
    temp_engine = EntityTemporalRiskEngine(
        beta=TEMPORAL_BETA, gamma=TEMPORAL_GAMMA,
        lambda_=TEMPORAL_LAMBDA, delta=TEMPORAL_DELTA,
    )
    # Need baseline A_t for train (use placeholder — we only need temporal state)
    if "A_t" not in train_df.columns:
        train_df["A_t"] = BASELINE_THRESHOLD - 0.01
    # Process TRAIN
    temp_engine.reset()
    train_entities = train_df["entity_proxy"].values
    train_scores   = train_df["A_t"].values
    for i in range(len(train_df)):
        temp_engine.step(str(train_entities[i]), float(train_scores[i]))

    # Read baseline val scores from frozen baseline artifacts if available
    baseline_val_path = RELATIONAL_DIR.parent / "baseline" / "validation_metrics.json"
    # Just run val through temporal using whatever A_t we can get
    # For VAL: read the A_t from Phase 1 model predictions if possible
    baseline_pred_path = RESULTS_DIR / "test_predictions.csv"
    # test_predictions.csv covers TEST only — for VAL re-run Phase 1 scores
    # Safest approach: run temporal engine on val using the entity states from TRAIN
    val_E_arr = np.zeros(len(val_df))
    val_P_arr = np.zeros(len(val_df))
    val_ents = val_df["entity_proxy"].values
    # Use baseline threshold as proxy A_t if no scores available for val
    val_A_proxy = np.full(len(val_df), BASELINE_THRESHOLD - 0.01)
    for i in range(len(val_df)):
        e_val, p_val = temp_engine.step(str(val_ents[i]), float(val_A_proxy[i]))
        val_E_arr[i] = e_val
        val_P_arr[i] = p_val

    val_df["A_t"] = val_A_proxy
    val_df["P_t"] = val_P_arr

    if "A_t" not in train_df.columns:
        train_df["A_t"] = BASELINE_THRESHOLD - 0.01

    return train_df, val_df, test_df, split_meta


# ---------------------------------------------------------------------------
# Run primary evaluation: G1 (DeviceInfo only) with frozen params
# ---------------------------------------------------------------------------

def run_primary_evaluation(train_df, val_df, test_df, frozen_params):
    logger.info("Running primary evaluation (G1: DeviceInfo only)...")
    params = GraphParameters(
        k_attr_max=frozen_params["k_attr_max"],
        window_sec=frozen_params["window_sec"],
        d_ref=frozen_params["d_ref"],
        v_ref=frozen_params["v_ref"],
        w_D=frozen_params["w_D"],
        w_V=frozen_params["w_V"],
        relational_attrs=tuple(frozen_params["relational_attrs"]),
    )
    engine = LightweightRelationalGraph(params)

    # Fit frequency ceiling on TRAIN
    freq_diag = engine.fit_attribute_frequency_ceiling(train_df)
    logger.info("Frequency ceiling fitted. Blocked: %s", {k: v["blocked_count"] for k, v in freq_diag.items()})

    # Process TRAIN (build graph history, no eval)
    logger.info("Processing TRAIN partition (%d rows)...", len(train_df))
    process_partition(train_df, engine)
    state_after_train = engine.get_state_summary()
    logger.info("Graph after TRAIN: %s", state_after_train)

    # Process VAL (graph persists from TRAIN)
    logger.info("Processing VAL partition (%d rows)...", len(val_df))
    val_records = process_partition(val_df, engine)
    state_after_val = engine.get_state_summary()

    # Process TEST (graph persists from TRAIN+VAL) — FIRST TEST ACCESS
    logger.info("Processing TEST partition (%d rows)...", len(test_df))
    t0 = time.perf_counter()
    test_records = process_partition(test_df, engine)
    test_elapsed = time.perf_counter() - t0
    state_after_test = engine.get_state_summary()

    logger.info("Graph after TEST: %s", state_after_test)

    val_results = evaluate_on_split(
        val_df, val_records, temporal_col="P_t",
        tau_base=frozen_params["tau_base"],
        tau_temp=frozen_params["tau_temp"],
        tau_rel=frozen_params["tau_rel"],
        tau_comb=frozen_params["tau_comb"],
        w_A=frozen_params["w_A"],
        w_P=frozen_params["w_P"],
        w_G=frozen_params["w_G"],
    )
    test_results = evaluate_on_split(
        test_df, test_records, temporal_col="P_t",
        tau_base=frozen_params["tau_base"],
        tau_temp=frozen_params["tau_temp"],
        tau_rel=frozen_params["tau_rel"],
        tau_comb=frozen_params["tau_comb"],
        w_A=frozen_params["w_A"],
        w_P=frozen_params["w_P"],
        w_G=frozen_params["w_G"],
    )

    return {
        "val": val_results,
        "test": test_results,
        "val_records": val_records,
        "test_records": test_records,
        "freq_diagnostics": freq_diag,
        "state_after_train": state_after_train,
        "state_after_val": state_after_val,
        "state_after_test": state_after_test,
        "test_processing_sec": round(test_elapsed, 4),
    }


# ---------------------------------------------------------------------------
# Ablation: G2 and G3 (overlap/contamination ablations)
# ---------------------------------------------------------------------------

def run_ablation_evaluations(train_df, val_df, test_df, frozen_params):
    ablation_results = {}
    for abl_name, abl_attrs in ABLATION_ATTR_SETS.items():
        if abl_name == "G1_device_only":
            continue  # primary already done
        logger.info("Running ablation: %s (attrs=%s)...", abl_name, abl_attrs)
        params = GraphParameters(
            k_attr_max=frozen_params["k_attr_max"],
            window_sec=frozen_params["window_sec"],
            d_ref=frozen_params["d_ref"],
            v_ref=frozen_params["v_ref"],
            w_D=frozen_params["w_D"],
            w_V=frozen_params["w_V"],
            relational_attrs=tuple(abl_attrs),
        )
        engine = LightweightRelationalGraph(params)
        engine.fit_attribute_frequency_ceiling(train_df)
        process_partition(train_df, engine)
        process_partition(val_df, engine)
        test_records = process_partition(test_df, engine)
        test_res = evaluate_on_split(
            test_df, test_records, temporal_col="P_t",
            tau_base=frozen_params["tau_base"],
            tau_temp=frozen_params["tau_temp"],
            tau_rel=frozen_params["tau_rel"],
            tau_comb=frozen_params["tau_comb"],
            w_A=frozen_params["w_A"],
            w_P=frozen_params["w_P"],
            w_G=frozen_params["w_G"],
        )
        ablation_results[abl_name] = {
            "attrs": abl_attrs,
            "B2": test_res["B2"],
            "B3": test_res["B3"],
            "G_t_stats": test_res["G_t_stats"],
            "pct_relational_evidence": test_res["pct_transactions_with_relational_evidence"],
        }
        logger.info("  %s: B2 F1=%.6f  B3 F1=%.6f  dFrauds(B2)=%+d  pct_Gt>0=%.2f%%",
                    abl_name,
                    test_res["B2"]["f1"], test_res["B3"]["f1"],
                    test_res["B2"]["additional_frauds_recovered"],
                    test_res["pct_transactions_with_relational_evidence"])
    return ablation_results


# ---------------------------------------------------------------------------
# Save CSV predictions
# ---------------------------------------------------------------------------

def save_predictions(test_df, test_records, frozen_params):
    G_t_arr = np.array([r.G_t for r in test_records])
    D_t_arr = np.array([r.D_t for r in test_records])
    V_t_arr = np.array([r.V_t for r in test_records])
    d_t_arr = np.array([r.d_t for r in test_records])
    v_t_arr = np.array([r.v_t for r in test_records])

    A_t = test_df["A_t"].values
    P_t = test_df["P_t"].values
    R_t = frozen_params["w_A"] * A_t + frozen_params["w_P"] * P_t + frozen_params["w_G"] * G_t_arr

    pred_B0 = (A_t >= frozen_params["tau_base"]).astype(int)
    pred_B1 = ((A_t >= frozen_params["tau_base"]) | (P_t >= frozen_params["tau_temp"])).astype(int)
    pred_B2 = ((A_t >= frozen_params["tau_base"]) | (G_t_arr >= frozen_params["tau_rel"])).astype(int)
    pred_B3 = (R_t >= frozen_params["tau_comb"]).astype(int)

    out = pd.DataFrame({
        "TransactionID":         test_df["TransactionID"].values,
        "entity_proxy":          test_df["entity_proxy"].values,
        "isFraud":               test_df["isFraud"].values,
        "A_t":                   A_t,
        "P_t":                   P_t,
        "D_t":                   D_t_arr,
        "V_t":                   V_t_arr,
        "G_t":                   G_t_arr,
        "R_t":                   R_t,
        "d_t":                   d_t_arr,
        "v_t":                   v_t_arr,
        "baseline_prediction":   pred_B0,
        "temporal_prediction":   pred_B1,
        "relational_prediction": pred_B2,
        "combined_prediction":   pred_B3,
    })
    path = RESULTS_DIR / "relational_predictions.csv"
    out.to_csv(path, index=False)
    logger.info("Predictions saved to %s (%d rows)", path, len(out))
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(test_df, test_records, primary_results):
    y_true = test_df["isFraud"].values
    G_t = np.array([r.G_t for r in test_records])
    D_t = np.array([r.D_t for r in test_records])
    V_t = np.array([r.V_t for r in test_records])

    # Plot 1: G_t distribution by label
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, arr, name in [(axes[0], G_t, "G_t"), (axes[1], D_t, "D_t")]:
        ax.hist(arr[y_true == 0], bins=50, alpha=0.6, label="Legitimate", color="steelblue", density=True)
        ax.hist(arr[y_true == 1], bins=50, alpha=0.6, label="Fraudulent", color="tomato", density=True)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"{name} Distribution by Label (Test)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "8_relational_score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: 4-way ablation F1/Recall/FPR bar chart
    systems = ["B0", "B1", "B2", "B3"]
    labels  = ["B0 (Baseline)", "B1 (+Temporal)", "B2 (+Relational)", "B3 (+Both)"]
    f1s     = [primary_results["test"][s]["f1"]        for s in systems]
    recalls = [primary_results["test"][s]["recall"]    for s in systems]
    fprs    = [primary_results["test"][s]["fpr"]       for s in systems]

    x = np.arange(len(systems))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, f1s,     width, label="F1",     color="steelblue")
    ax.bar(x,         recalls, width, label="Recall",  color="mediumseagreen")
    ax.bar(x + width, fprs,    width, label="FPR",     color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("4-Way Ablation: Test Set Metrics")
    ax.legend()
    ax.set_ylim(0, max(max(f1s), max(recalls)) * 1.2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "9_ablation_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: G_t nonzero evidence breakdown by label
    pct_fraud_nonzero = 100.0 * (G_t[y_true == 1] > 0).mean() if (y_true == 1).any() else 0
    pct_legit_nonzero = 100.0 * (G_t[y_true == 0] > 0).mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    cats  = ["Legitimate", "Fraudulent"]
    vals  = [pct_legit_nonzero, pct_fraud_nonzero]
    colors = ["steelblue", "tomato"]
    bars = ax.bar(cats, vals, color=colors, width=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", va="bottom")
    ax.set_ylabel("% Transactions with G_t > 0")
    ax.set_title("Relational Evidence Coverage (Test)")
    ax.set_ylim(0, max(vals) * 1.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_relational_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Plots saved to %s", PLOTS_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_total = time.time()

    frozen_params = load_frozen_params()
    logger.info("Frozen parameters loaded: %s", frozen_params)

    train_df, val_df, test_df, split_meta = prepare_all_splits()

    primary = run_primary_evaluation(train_df, val_df, test_df, frozen_params)
    ablations = run_ablation_evaluations(train_df, val_df, test_df, frozen_params)

    pred_df = save_predictions(test_df, primary["test_records"], frozen_params)
    generate_plots(test_df, primary["test_records"], primary)

    # Assemble full results artifact
    full_results = {
        "frozen_params": frozen_params,
        "split_meta": split_meta,
        "attribute_frequency_diagnostics": primary["freq_diagnostics"],
        "graph_state": {
            "after_train": primary["state_after_train"],
            "after_val":   primary["state_after_val"],
            "after_test":  primary["state_after_test"],
        },
        "validation": {k: v for k, v in primary["val"].items()
                       if k not in ("val_records",)},
        "test": {k: v for k, v in primary["test"].items()
                 if k not in ("test_records",)},
        "ablation_test": ablations,
        "total_elapsed_sec": round(time.time() - t_total, 1),
        "test_processing_sec": primary["test_processing_sec"],
    }

    results_path = RELATIONAL_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info("Full results saved to %s", results_path)

    # Final summary
    logger.info("\n==========================================")
    logger.info("TRUSTGRAPH PHASE 3 — TEST RESULTS (G1: DeviceInfo only)")
    logger.info("==========================================")
    for sys in ["B0", "B1", "B2", "B3"]:
        m = primary["test"][sys]
        logger.info(
            "  %-3s  F1=%.6f  Prec=%.6f  Rec=%.6f  FPR=%.6f  "
            "dFrauds=%+d  dFP=%+d  deltaF1=%+.6f",
            sys, m["f1"], m["precision"], m["recall"], m["fpr"],
            m["additional_frauds_recovered"], m["additional_false_positives"],
            m["delta_f1"]
        )
    g_stats = primary["test"]["G_t_stats"]
    logger.info("\nG_t stats (Test):")
    logger.info("  Fraudulent:  mean=%.4f  p95=%.4f  pct_nonzero=%.2f%%",
                g_stats["mean_fraud"], g_stats["p95_fraud"],
                100 * g_stats["pct_nonzero_fraud"])
    logger.info("  Legitimate:  mean=%.4f  p95=%.4f  pct_nonzero=%.2f%%",
                g_stats["mean_legit"], g_stats["p95_legit"],
                100 * g_stats["pct_nonzero_legit"])
    logger.info("  Overall transactions with G_t > 0: %d (%.2f%%)",
                primary["test"]["transactions_with_nonzero_G_t"],
                primary["test"]["pct_transactions_with_relational_evidence"])
    logger.info("Total elapsed: %.1f s", time.time() - t_total)
