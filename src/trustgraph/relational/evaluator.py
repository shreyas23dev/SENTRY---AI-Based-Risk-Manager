"""
evaluator.py — TRUSTGRAPH Phase 3 Relational Evaluator
=======================================================

Runs the 4-way ablation study and computes per-system metrics.

Systems:
  B0 — Baseline LightGBM (A_t >= tau_base)
  B1 — Baseline + Entity Temporal (A_t >= tau_base OR P_t >= tau_temp)
  B2 — Baseline + Relational (A_t >= tau_base OR G_t >= tau_rel) — no P_t
  B3 — Baseline + Temporal + Relational (R_t = w_A*A_t + w_P*P_t + w_G*G_t >= tau_comb)

Graph state persists across TRAIN → VAL → TEST partitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BASELINE_THRESHOLD
from .graph_engine import (
    GraphParameters,
    LightweightRelationalGraph,
    RelationalRecord,
    process_partition,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6),
        "recall":    round(recall, 6),
        "f1":        round(f1, 6),
        "fpr":       round(fpr, 6),
    }


def _delta(v_new: float, v_base: float) -> str:
    d = v_new - v_base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.6f}"


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_on_split(
    df: pd.DataFrame,
    records: List[RelationalRecord],
    temporal_col: str = "P_t",
    *,
    tau_base: float = BASELINE_THRESHOLD,
    tau_temp: float = 0.70,
    tau_rel: float  = 0.50,
    tau_comb: float = 0.50,
    w_A: float = 0.50,
    w_P: float = 0.25,
    w_G: float = 0.25,
) -> Dict[str, Any]:
    """
    Compute 4-way ablation metrics for one partition.
    df must be aligned (same row order) with records.
    """
    assert len(df) == len(records), "df / records length mismatch"

    A_t   = df["A_t"].values.astype(float)
    P_t   = df[temporal_col].values.astype(float) if temporal_col in df.columns else np.zeros(len(df))
    G_t   = np.array([r.G_t for r in records], dtype=float)
    y_true = df["isFraud"].values.astype(int)

    R_t = w_A * A_t + w_P * P_t + w_G * G_t

    pred_B0 = (A_t >= tau_base).astype(int)
    pred_B1 = ((A_t >= tau_base) | (P_t >= tau_temp)).astype(int)
    pred_B2 = ((A_t >= tau_base) | (G_t >= tau_rel)).astype(int)
    pred_B3 = (R_t >= tau_comb).astype(int)

    base_m = _compute_metrics(y_true, pred_B0)

    results: Dict[str, Any] = {}
    for name, pred in [("B0", pred_B0), ("B1", pred_B1), ("B2", pred_B2), ("B3", pred_B3)]:
        m = _compute_metrics(y_true, pred)
        delta_frauds = m["tp"] - base_m["tp"]
        delta_fps    = m["fp"] - base_m["fp"]
        results[name] = {
            **m,
            "additional_frauds_recovered":  delta_frauds,
            "additional_false_positives":   delta_fps,
            "delta_f1":  round(m["f1"]  - base_m["f1"],  6),
            "delta_fpr": round(m["fpr"] - base_m["fpr"], 6),
        }

    # G_t distribution
    results["G_t_stats"] = {
        "mean_fraud":    float(G_t[y_true == 1].mean()) if (y_true == 1).any() else 0.0,
        "mean_legit":    float(G_t[y_true == 0].mean()),
        "p95_fraud":     float(np.percentile(G_t[y_true == 1], 95)) if (y_true == 1).any() else 0.0,
        "p95_legit":     float(np.percentile(G_t[y_true == 0], 95)),
        "pct_nonzero_fraud": float((G_t[y_true == 1] > 0).mean()) if (y_true == 1).any() else 0.0,
        "pct_nonzero_legit": float((G_t[y_true == 0] > 0).mean()),
    }

    results["D_t_stats"] = {
        "mean_fraud": float(np.array([r.D_t for r in records])[y_true == 1].mean()) if (y_true == 1).any() else 0.0,
        "mean_legit": float(np.array([r.D_t for r in records])[y_true == 0].mean()),
    }

    results["V_t_stats"] = {
        "mean_fraud": float(np.array([r.V_t for r in records])[y_true == 1].mean()) if (y_true == 1).any() else 0.0,
        "mean_legit": float(np.array([r.V_t for r in records])[y_true == 0].mean()),
    }

    results["transactions_with_nonzero_G_t"] = int((G_t > 0).sum())
    results["pct_transactions_with_relational_evidence"] = round(100.0 * (G_t > 0).mean(), 4)

    return results


# ---------------------------------------------------------------------------
# Relational evaluator class
# ---------------------------------------------------------------------------

class RelationalEvaluator:
    """High-level evaluator: builds the full persistent graph, runs ablations."""

    def __init__(
        self,
        params: GraphParameters,
        tau_base: float = BASELINE_THRESHOLD,
        tau_temp: float = 0.70,
        tau_rel:  float = 0.50,
        tau_comb: float = 0.50,
        w_A: float = 0.50,
        w_P: float = 0.25,
        w_G: float = 0.25,
    ) -> None:
        self.params   = params
        self.tau_base = tau_base
        self.tau_temp = tau_temp
        self.tau_rel  = tau_rel
        self.tau_comb = tau_comb
        self.w_A = w_A
        self.w_P = w_P
        self.w_G = w_G

    def run(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        test_df:  Optional[pd.DataFrame] = None,
        entity_col: str = "entity_proxy",
        timestamp_col: str = "TransactionDT",
        id_col:        str = "TransactionID",
        temporal_col:  str = "P_t",
    ) -> Dict[str, Any]:
        """
        Build graph across TRAIN, evaluate on VAL; optionally evaluate on TEST.

        TRAIN → fit frequency ceiling + process (build graph history)
        VAL   → initialized from final TRAIN graph state; process + evaluate
        TEST  → initialized from final TRAIN+VAL graph state; process + evaluate
        """
        engine = LightweightRelationalGraph(self.params)

        # Step 1: Fit attribute-frequency ceiling on TRAIN only
        freq_diag = engine.fit_attribute_frequency_ceiling(train_df, entity_col=entity_col)

        # Step 2: Process TRAIN chronologically to build graph history
        # (we collect records but only use them for graph state, not for eval)
        _ = process_partition(
            train_df, engine,
            entity_col=entity_col, timestamp_col=timestamp_col, id_col=id_col,
        )
        train_state = engine.get_state_summary()

        # Step 3: Process VALIDATION — graph continues from final TRAIN state
        val_records = process_partition(
            val_df, engine,
            entity_col=entity_col, timestamp_col=timestamp_col, id_col=id_col,
        )
        val_state = engine.get_state_summary()

        val_results = evaluate_on_split(
            val_df, val_records,
            temporal_col=temporal_col,
            tau_base=self.tau_base, tau_temp=self.tau_temp,
            tau_rel=self.tau_rel, tau_comb=self.tau_comb,
            w_A=self.w_A, w_P=self.w_P, w_G=self.w_G,
        )

        output: Dict[str, Any] = {
            "params": {
                "k_attr_max": self.params.k_attr_max,
                "window_sec": self.params.window_sec,
                "d_ref": self.params.d_ref,
                "v_ref": self.params.v_ref,
                "w_D": self.params.w_D,
                "w_V": self.params.w_V,
                "tau_base": self.tau_base,
                "tau_temp": self.tau_temp,
                "tau_rel": self.tau_rel,
                "tau_comb": self.tau_comb,
                "w_A": self.w_A,
                "w_P": self.w_P,
                "w_G": self.w_G,
                "relational_attrs": list(self.params.relational_attrs),
            },
            "attribute_frequency_diagnostics": freq_diag,
            "graph_state_after_train": train_state,
            "graph_state_after_val": val_state,
            "validation": val_results,
        }

        # Step 4: TEST — only if requested (all params must be frozen before calling)
        if test_df is not None:
            test_records = process_partition(
                test_df, engine,
                entity_col=entity_col, timestamp_col=timestamp_col, id_col=id_col,
            )
            test_state = engine.get_state_summary()
            test_results = evaluate_on_split(
                test_df, test_records,
                temporal_col=temporal_col,
                tau_base=self.tau_base, tau_temp=self.tau_temp,
                tau_rel=self.tau_rel, tau_comb=self.tau_comb,
                w_A=self.w_A, w_P=self.w_P, w_G=self.w_G,
            )
            output["graph_state_after_test"] = test_state
            output["test"] = test_results
            output["test_records"] = test_records

        output["val_records"] = val_records
        return output
