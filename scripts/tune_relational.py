"""
tune_relational.py — TRUSTGRAPH Phase 3 Validation-Only Staged Parameter Search
================================================================================

EFFICIENCY DESIGN:
  The TRAIN graph build is the bottleneck. This script avoids re-running TRAIN
  for every configuration by:
    1. Loading data and resolving entity proxies ONCE.
    2. For each k_attr_max: build a TRAIN graph ONCE, then deepcopy the internal
       graph state dictionaries for each (W, d_ref, v_ref) combination before
       running the lightweight VAL pass.
    3. VAL pass only (no TRAIN rebuild) for each configuration.

PROTOCOL (strictly validation-only — TEST never accessed):
  Stage 1: k_attr_max × W × d_ref × v_ref  →  best val B2 F1
  Stage 2: w_D × w_V                         →  best val B2 F1
  Stage 3: w_A × w_P × w_G                   →  best val B3 F1
  Stage 4: tau_rel, tau_comb                  →  best val B2 F1, B3 F1

All results saved to artifacts/relational/validation_tuning.json.
Frozen best params saved to artifacts/relational/parameters.json.
"""

import sys
import json
import copy
import time
import logging
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np

from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.temporal.entity_tracker import resolve_entity_key
from trustgraph.relational.config import (
    RELATIONAL_DIR, BASELINE_THRESHOLD, ENTITY_KEY_TYPE,
    TEMPORAL_THRESHOLD,
    K_ATTR_MAX_GRID, WINDOW_GRID_SEC, D_REF_GRID, V_REF_GRID,
    WD_WV_GRID, COMBINED_WEIGHT_GRID, REL_THRESHOLD_GRID, COMB_THRESHOLD_GRID,
    RELATIONAL_ATTRIBUTES,
)
from trustgraph.relational.graph_engine import (
    GraphParameters, LightweightRelationalGraph, process_partition,
)
from trustgraph.relational.evaluator import evaluate_on_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RELATIONAL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading (once)
# ---------------------------------------------------------------------------

def prepare_splits():
    logger.info("Loading train data...")
    df, _ = load_train_data()
    train_df, val_df, _, _ = chronological_split(df)
    del df

    logger.info("Resolving entity proxies (%s)...", ENTITY_KEY_TYPE)
    train_df["entity_proxy"] = resolve_entity_key(train_df, key_type=ENTITY_KEY_TYPE)
    val_df["entity_proxy"]   = resolve_entity_key(val_df,   key_type=ENTITY_KEY_TYPE)

    # Load frozen temporal predictions for VAL (A_t, P_t from Phase 2.1)
    temporal_pred_path = Path(__file__).resolve().parent.parent / "results" / "temporal_entity_predictions.csv"
    if temporal_pred_path.exists():
        temporal_preds = pd.read_csv(temporal_pred_path)
        logger.info("Loaded frozen temporal predictions: %d rows", len(temporal_preds))
        val_df = val_df.merge(
            temporal_preds[["TransactionID", "A_t", "P_t"]],
            on="TransactionID", how="left"
        )
        val_df["A_t"] = val_df["A_t"].fillna(BASELINE_THRESHOLD - 0.01)
        val_df["P_t"] = val_df["P_t"].fillna(0.0)
    else:
        logger.warning("temporal_entity_predictions.csv not found — using placeholder A_t/P_t for VAL")
        val_df["A_t"] = BASELINE_THRESHOLD - 0.01
        val_df["P_t"] = 0.0

    if "A_t" not in train_df.columns:
        train_df["A_t"] = BASELINE_THRESHOLD - 0.01

    return train_df, val_df


# ---------------------------------------------------------------------------
# Graph state snapshot / restore via deepcopy
# ---------------------------------------------------------------------------

def snapshot_engine_state(engine: LightweightRelationalGraph) -> dict:
    """Capture all mutable graph state as a deep copy."""
    return {
        "entity_to_attrs":           copy.deepcopy(dict(engine._entity_to_attrs)),
        "attr_to_entities":          copy.deepcopy(dict(engine._attr_to_entities)),
        "entity_neighbors":          copy.deepcopy(dict(engine._entity_neighbors)),
        "relationship_first_seen":   copy.deepcopy(dict(engine._relationship_first_seen)),
        "entity_velocity_events":    copy.deepcopy({k: list(v) for k, v in engine._entity_velocity_events.items()}),
        "blocked_attr_values":       copy.deepcopy(engine._blocked_attr_values),
    }


def restore_engine_state(engine: LightweightRelationalGraph, state: dict) -> None:
    """Restore a previously snapshotted state into engine (in-place)."""
    from collections import defaultdict, deque
    engine._entity_to_attrs         = defaultdict(set, {k: set(v) for k, v in state["entity_to_attrs"].items()})
    engine._attr_to_entities        = defaultdict(set, {k: set(v) for k, v in state["attr_to_entities"].items()})
    engine._entity_neighbors        = defaultdict(set, {k: set(v) for k, v in state["entity_neighbors"].items()})
    engine._relationship_first_seen = dict(state["relationship_first_seen"])
    engine._entity_velocity_events  = defaultdict(deque, {k: deque(v) for k, v in state["entity_velocity_events"].items()})
    engine._blocked_attr_values     = set(state["blocked_attr_values"])
    engine._attr_freq_cache_fitted  = True


# ---------------------------------------------------------------------------
# Run one VAL evaluation (without rebuilding TRAIN)
# ---------------------------------------------------------------------------

def eval_val(engine_template: LightweightRelationalGraph,
             train_state: dict,
             val_df: pd.DataFrame,
             k_attr_max: int,
             window_sec: float,
             d_ref: float,
             v_ref: float,
             w_D: float,
             w_V: float,
             tau_rel: float,
             tau_comb: float,
             w_A: float,
             w_P: float,
             w_G: float) -> dict:
    """Restore TRAIN state, apply new params, run VAL pass, return metrics."""
    # Create fresh params with potentially different W/d_ref/v_ref/weights
    new_params = GraphParameters(
        k_attr_max=k_attr_max,
        window_sec=window_sec,
        d_ref=d_ref,
        v_ref=v_ref,
        w_D=w_D,
        w_V=w_V,
        relational_attrs=tuple(RELATIONAL_ATTRIBUTES),
    )
    # New engine; restore state from TRAIN snapshot
    engine = LightweightRelationalGraph(new_params)
    restore_engine_state(engine, train_state)

    val_records = process_partition(val_df, engine)
    results = evaluate_on_split(
        val_df, val_records,
        temporal_col="P_t",
        tau_base=BASELINE_THRESHOLD,
        tau_temp=TEMPORAL_THRESHOLD,
        tau_rel=tau_rel,
        tau_comb=tau_comb,
        w_A=w_A, w_P=w_P, w_G=w_G,
    )
    return results


# ---------------------------------------------------------------------------
# Staged grid search
# ---------------------------------------------------------------------------

def stage1_search(train_df, val_df):
    """
    Outer loop: k_attr_max (5 values)
      Inner loop: W × d_ref × v_ref (2×4×4 = 32 per k)
    Total: 160 configs, but TRAIN is rebuilt only 5 times (once per k).
    """
    best_f1, best_cfg, all_results = -1.0, None, []
    inner_grid = list(product(WINDOW_GRID_SEC, D_REF_GRID, V_REF_GRID))

    for k in K_ATTR_MAX_GRID:
        logger.info("Stage 1: Building TRAIN graph for k_attr_max=%d...", k)
        params = GraphParameters(
            k_attr_max=k, window_sec=86_400.0, d_ref=5.0, v_ref=3.0,
            w_D=0.5, w_V=0.5, relational_attrs=tuple(RELATIONAL_ATTRIBUTES),
        )
        engine = LightweightRelationalGraph(params)
        engine.fit_attribute_frequency_ceiling(train_df)
        t0 = time.time()
        process_partition(train_df, engine)
        logger.info("  TRAIN graph built in %.1f s. Snapshotting...", time.time() - t0)
        train_state = snapshot_engine_state(engine)
        logger.info("  Snapshot done. Running %d inner configs...", len(inner_grid))

        for idx, (w, d, v) in enumerate(inner_grid):
            results = eval_val(
                engine, train_state, val_df,
                k_attr_max=k, window_sec=w, d_ref=d, v_ref=v,
                w_D=0.5, w_V=0.5, tau_rel=0.5, tau_comb=0.5,
                w_A=0.5, w_P=0.25, w_G=0.25,
            )
            f1 = results["B2"]["f1"]
            entry = {
                "k_attr_max": k, "window_sec": w, "d_ref": d, "v_ref": v,
                "val_B2_f1": f1,
                "val_B2_recall": results["B2"]["recall"],
                "val_B2_fpr":    results["B2"]["fpr"],
                "additional_frauds": results["B2"]["additional_frauds_recovered"],
                "additional_fps":    results["B2"]["additional_false_positives"],
                "pct_nonzero_Gt":    results["pct_transactions_with_relational_evidence"],
            }
            all_results.append(entry)
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = entry

        logger.info("  k=%d done. Best so far: k=%s W=%s d=%s v=%s F1=%.6f",
                    k, best_cfg["k_attr_max"], best_cfg["window_sec"],
                    best_cfg["d_ref"], best_cfg["v_ref"], best_f1)

    logger.info("Stage 1 best: k=%s W=%s d_ref=%s v_ref=%s → B2 F1=%.6f",
                best_cfg["k_attr_max"], best_cfg["window_sec"],
                best_cfg["d_ref"], best_cfg["v_ref"], best_f1)
    return best_cfg, all_results


def stage2_search(train_state_cache, val_df, s1):
    """w_D × w_V: reuse the k_attr_max TRAIN state already built during Stage 1."""
    best_f1, best_cfg, all_results = -1.0, None, []
    logger.info("Stage 2: %d configurations (reusing TRAIN state)...", len(WD_WV_GRID))
    for w_D, w_V in WD_WV_GRID:
        results = eval_val(
            None, train_state_cache, val_df,
            k_attr_max=s1["k_attr_max"], window_sec=s1["window_sec"],
            d_ref=s1["d_ref"], v_ref=s1["v_ref"],
            w_D=w_D, w_V=w_V, tau_rel=0.5, tau_comb=0.5,
            w_A=0.5, w_P=0.25, w_G=0.25,
        )
        f1 = results["B2"]["f1"]
        entry = {"w_D": w_D, "w_V": w_V, "val_B2_f1": f1,
                 "additional_frauds": results["B2"]["additional_frauds_recovered"]}
        all_results.append(entry)
        if f1 > best_f1:
            best_f1 = f1
            best_cfg = entry
    logger.info("Stage 2 best: w_D=%.2f w_V=%.2f → B2 F1=%.6f",
                best_cfg["w_D"], best_cfg["w_V"], best_f1)
    return best_cfg, all_results


def stage3_search(train_state_cache, val_df, s1, s2):
    """w_A × w_P × w_G for B3 F1."""
    best_f1, best_cfg, all_results = -1.0, None, []
    logger.info("Stage 3: %d configurations (reusing TRAIN state)...", len(COMBINED_WEIGHT_GRID))
    for w_A, w_P, w_G in COMBINED_WEIGHT_GRID:
        results = eval_val(
            None, train_state_cache, val_df,
            k_attr_max=s1["k_attr_max"], window_sec=s1["window_sec"],
            d_ref=s1["d_ref"], v_ref=s1["v_ref"],
            w_D=s2["w_D"], w_V=s2["w_V"], tau_rel=0.5, tau_comb=0.5,
            w_A=w_A, w_P=w_P, w_G=w_G,
        )
        f1 = results["B3"]["f1"]
        entry = {"w_A": w_A, "w_P": w_P, "w_G": w_G, "val_B3_f1": f1,
                 "additional_frauds": results["B3"]["additional_frauds_recovered"]}
        all_results.append(entry)
        if f1 > best_f1:
            best_f1 = f1
            best_cfg = entry
    logger.info("Stage 3 best: w_A=%.2f w_P=%.2f w_G=%.2f → B3 F1=%.6f",
                best_cfg["w_A"], best_cfg["w_P"], best_cfg["w_G"], best_f1)
    return best_cfg, all_results


def stage4_search(train_state_cache, val_df, s1, s2, s3):
    """tau_rel (B2) then tau_comb (B3)."""
    best_rel_f1, best_tau_rel, rel_results = -1.0, 0.5, []
    logger.info("Stage 4a: tau_rel grid (%d)...", len(REL_THRESHOLD_GRID))
    for tau in REL_THRESHOLD_GRID:
        results = eval_val(
            None, train_state_cache, val_df,
            k_attr_max=s1["k_attr_max"], window_sec=s1["window_sec"],
            d_ref=s1["d_ref"], v_ref=s1["v_ref"],
            w_D=s2["w_D"], w_V=s2["w_V"], tau_rel=tau, tau_comb=0.5,
            w_A=s3["w_A"], w_P=s3["w_P"], w_G=s3["w_G"],
        )
        f1 = results["B2"]["f1"]
        rel_results.append({"tau_rel": tau, "val_B2_f1": f1,
                             "additional_frauds": results["B2"]["additional_frauds_recovered"],
                             "additional_fps": results["B2"]["additional_false_positives"]})
        if f1 > best_rel_f1:
            best_rel_f1 = f1
            best_tau_rel = tau

    best_comb_f1, best_tau_comb, comb_results = -1.0, 0.5, []
    logger.info("Stage 4b: tau_comb grid (%d)...", len(COMB_THRESHOLD_GRID))
    for tau in COMB_THRESHOLD_GRID:
        results = eval_val(
            None, train_state_cache, val_df,
            k_attr_max=s1["k_attr_max"], window_sec=s1["window_sec"],
            d_ref=s1["d_ref"], v_ref=s1["v_ref"],
            w_D=s2["w_D"], w_V=s2["w_V"], tau_rel=best_tau_rel, tau_comb=tau,
            w_A=s3["w_A"], w_P=s3["w_P"], w_G=s3["w_G"],
        )
        f1 = results["B3"]["f1"]
        comb_results.append({"tau_comb": tau, "val_B3_f1": f1,
                              "additional_frauds": results["B3"]["additional_frauds_recovered"],
                              "additional_fps": results["B3"]["additional_false_positives"]})
        if f1 > best_comb_f1:
            best_comb_f1 = f1
            best_tau_comb = tau

    logger.info("Stage 4 best: tau_rel=%.2f (B2 F1=%.6f), tau_comb=%.2f (B3 F1=%.6f)",
                best_tau_rel, best_rel_f1, best_tau_comb, best_comb_f1)
    return best_tau_rel, best_tau_comb, rel_results, comb_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    train_df, val_df = prepare_splits()

    # Stage 1 — also captures the final best-k TRAIN state for reuse
    s1_best, s1_all = stage1_search(train_df, val_df)

    # Rebuild TRAIN graph once with best k_attr_max for all subsequent stages
    logger.info("Rebuilding TRAIN graph with best k_attr_max=%d for Stages 2-4...",
                s1_best["k_attr_max"])
    best_params = GraphParameters(
        k_attr_max=s1_best["k_attr_max"],
        window_sec=s1_best["window_sec"],
        d_ref=s1_best["d_ref"],
        v_ref=s1_best["v_ref"],
        w_D=0.5, w_V=0.5,
        relational_attrs=tuple(RELATIONAL_ATTRIBUTES),
    )
    best_engine = LightweightRelationalGraph(best_params)
    best_engine.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, best_engine)
    best_train_state = snapshot_engine_state(best_engine)
    logger.info("TRAIN state cached. Running Stages 2-4...")

    s2_best, s2_all = stage2_search(best_train_state, val_df, s1_best)
    s3_best, s3_all = stage3_search(best_train_state, val_df, s1_best, s2_best)
    best_tau_rel, best_tau_comb, s4a_all, s4b_all = stage4_search(
        best_train_state, val_df, s1_best, s2_best, s3_best)

    # Final validation pass with all frozen params
    final_results = eval_val(
        None, best_train_state, val_df,
        k_attr_max=s1_best["k_attr_max"],
        window_sec=s1_best["window_sec"],
        d_ref=s1_best["d_ref"],
        v_ref=s1_best["v_ref"],
        w_D=s2_best["w_D"],
        w_V=s2_best["w_V"],
        tau_rel=best_tau_rel,
        tau_comb=best_tau_comb,
        w_A=s3_best["w_A"],
        w_P=s3_best["w_P"],
        w_G=s3_best["w_G"],
    )

    elapsed = time.time() - t0

    frozen_params = {
        "k_attr_max":  s1_best["k_attr_max"],
        "window_sec":  s1_best["window_sec"],
        "d_ref":       s1_best["d_ref"],
        "v_ref":       s1_best["v_ref"],
        "w_D":         s2_best["w_D"],
        "w_V":         s2_best["w_V"],
        "w_A":         s3_best["w_A"],
        "w_P":         s3_best["w_P"],
        "w_G":         s3_best["w_G"],
        "tau_base":    BASELINE_THRESHOLD,
        "tau_temp":    TEMPORAL_THRESHOLD,
        "tau_rel":     best_tau_rel,
        "tau_comb":    best_tau_comb,
        "relational_attrs": RELATIONAL_ATTRIBUTES,
        "entity_key":  ENTITY_KEY_TYPE,
    }

    tuning_output = {
        "frozen_params": frozen_params,
        "stage1_best": s1_best,
        "stage1_all": s1_all,
        "stage2_best": s2_best,
        "stage2_all": s2_all,
        "stage3_best": s3_best,
        "stage3_all": s3_all,
        "stage4a_tau_rel_all":  s4a_all,
        "stage4b_tau_comb_all": s4b_all,
        "final_validation": {
            "B0": final_results["B0"],
            "B1": final_results["B1"],
            "B2": final_results["B2"],
            "B3": final_results["B3"],
            "G_t_stats": final_results["G_t_stats"],
            "D_t_stats": final_results["D_t_stats"],
            "V_t_stats": final_results["V_t_stats"],
            "pct_relational_evidence": final_results["pct_transactions_with_relational_evidence"],
        },
        "tuning_elapsed_sec": round(elapsed, 1),
    }

    tuning_path = RELATIONAL_DIR / "validation_tuning.json"
    with open(tuning_path, "w") as f:
        json.dump(tuning_output, f, indent=2)
    logger.info("Tuning results → %s", tuning_path)

    params_path = RELATIONAL_DIR / "parameters.json"
    with open(params_path, "w") as f:
        json.dump(frozen_params, f, indent=2)
    logger.info("Frozen parameters → %s", params_path)

    logger.info("\n=== FINAL VALIDATION RESULTS ===")
    for sys in ["B0", "B1", "B2", "B3"]:
        m = final_results[sys]
        logger.info("  %-3s  F1=%.6f  Prec=%.6f  Rec=%.6f  FPR=%.6f  dFrauds=%+d  dFP=%+d",
                    sys, m["f1"], m["precision"], m["recall"], m["fpr"],
                    m["additional_frauds_recovered"], m["additional_false_positives"])
    logger.info("Total tuning time: %.1f s", elapsed)
