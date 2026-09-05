"""
scorer.py — TRUSTGRAPH Phase 5A Unified RuntimeScorer
======================================================

Load-once, score-many service that integrates all frozen pipeline components:
  - LightGBM model (A_t)
  - FastPreprocessor (single-row path, no DataFrame)
  - EntityTemporalRiskEngine (P_t)
  - LightweightRelationalGraph (G_t)
  - Conditional fusion rule (R_t)
  - Progressive risk decision policy (action, explanation)

Usage:
    scorer = RuntimeScorer.load(artifacts_root)

    # Single transaction
    result = scorer.score_transaction(raw_dict)
    print(result.action, result.R_t, result.explanation)

    # Batch
    df_out = scorer.score_batch(df)

The scorer maintains internal mutable state (temporal memory + graph state).
Transactions MUST be presented in chronological order for causal correctness.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.preprocessing import BaselinePreprocessor
from trustgraph.runtime.fast_preprocessor import FastPreprocessor
from trustgraph.temporal.entity_tracker import EntityTemporalRiskEngine, resolve_entity_key
from trustgraph.relational.graph_engine import (
    GraphParameters, LightweightRelationalGraph,
)
from trustgraph.policy.config import PolicyAction
from trustgraph.policy.decision_engine import (
    PolicyThresholds, generate_explanation,
)
from trustgraph.fusion.config import ENTITY_KEY_TYPE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Complete scoring output for a single transaction."""
    transaction_id: Optional[int]
    entity_id: str
    A_t: float       # Point-wise model risk
    P_t: float       # Entity temporal risk
    G_t: float       # Relational graph risk
    D_t: float       # Normalised relational degree
    V_t: float       # Normalised relational velocity
    d_t: int         # Raw degree (distinct connected entities)
    v_t: int         # Raw velocity (new connections in window)
    R_t: float       # Fused combined risk
    action: str      # Policy action string (ALLOW/VERIFY/THROTTLE/BLOCK)
    explanation: str # Auditable human-readable explanation


# ---------------------------------------------------------------------------
# RuntimeScorer
# ---------------------------------------------------------------------------

class RuntimeScorer:
    """
    Load-once, score-many TRUSTGRAPH runtime service.

    Internal state (temporal memory + graph) accumulates across calls.
    Transactions must be presented in ascending TransactionDT order.
    """

    def __init__(
        self,
        model: BaselineModel,
        fast_prep: FastPreprocessor,
        temp_engine: EntityTemporalRiskEngine,
        graph_engine: LightweightRelationalGraph,
        thresholds: PolicyThresholds,
        alpha: float = 1.0,
        beta_fusion: float = 0.05,
        graph_attrs: List[str] = ("DeviceInfo",),
    ) -> None:
        self._model = model
        self._fast_prep = fast_prep
        self._temp_engine = temp_engine
        self._graph_engine = graph_engine
        self._thresholds = thresholds
        self._alpha = alpha
        self._beta_fusion = beta_fusion
        self._graph_attrs = list(graph_attrs)

    # ------------------------------------------------------------------
    # Factory loader
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, artifacts_root: Path) -> "RuntimeScorer":
        """
        Load all frozen artifacts from artifacts_root.

        artifacts_root is typically PROJECT_ROOT / "artifacts".
        Loads:
            artifacts/baseline/model/lgbm_model.pkl
            artifacts/baseline/preprocessing/
            artifacts/policy/thresholds.json
        """
        artifacts_root = Path(artifacts_root)

        logger.info("RuntimeScorer: loading frozen model...")
        model = BaselineModel.load(artifacts_root / "baseline" / "model" / "lgbm_model.pkl")

        logger.info("RuntimeScorer: loading preprocessor...")
        raw_prep = BaselinePreprocessor.load(artifacts_root / "baseline" / "preprocessing")
        fast_prep = FastPreprocessor(raw_prep)

        logger.info("RuntimeScorer: initialising temporal engine (frozen params)...")
        temp_engine = EntityTemporalRiskEngine(beta=0.3, gamma=0.5, lambda_=0.05, delta=0.05)

        logger.info("RuntimeScorer: initialising relational graph (frozen params)...")
        rel_params = GraphParameters(
            k_attr_max=25,
            window_sec=86_400.0,
            d_ref=3.0,
            v_ref=10.0,
            w_D=0.6,
            w_V=0.4,
            relational_attrs=("DeviceInfo",),
        )
        graph_engine = LightweightRelationalGraph(rel_params)

        logger.info("RuntimeScorer: loading policy thresholds...")
        thresh_path = artifacts_root / "policy" / "thresholds.json"
        with open(thresh_path) as f:
            td = json.load(f)
        thresholds = PolicyThresholds(
            tau_verify=td["tau_verify"],
            tau_throttle=td["tau_throttle"],
            tau_block=td["tau_block"],
        )

        # NOTE: blocked attribute values must be loaded separately if graph
        # state is being seeded from a checkpoint. For a fresh scorer, call
        # scorer.seed_from_history(train_df, val_df) before scoring test data.

        logger.info("RuntimeScorer ready.")
        return cls(
            model=model,
            fast_prep=fast_prep,
            temp_engine=temp_engine,
            graph_engine=graph_engine,
            thresholds=thresholds,
        )

    def seed_blocked_values(self, train_df: pd.DataFrame) -> None:
        """
        Fit the attribute frequency ceiling from TRAIN data.
        Must be called once before processing any transactions.
        """
        self._graph_engine.fit_attribute_frequency_ceiling(train_df)
        logger.info("RuntimeScorer: attribute frequency ceiling fitted.")

    # ------------------------------------------------------------------
    # Single-transaction scoring (hot path)
    # ------------------------------------------------------------------

    def score_transaction(
        self,
        raw_dict: Dict[str, Any],
        entity_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        transaction_id: Optional[int] = None,
    ) -> ScoringResult:
        """
        Score one transaction. O(1) per call after loading.

        Parameters
        ----------
        raw_dict : dict of {column_name: value} for all raw features
        entity_id : pre-computed entity proxy string (optional; derived from raw_dict if None)
        timestamp : TransactionDT float (optional; read from raw_dict if None)
        transaction_id : TransactionID int (optional; read from raw_dict if None)

        Returns
        -------
        ScoringResult with all risk components and policy action.
        """
        # -- Resolve identifiers --
        if timestamp is None:
            timestamp = float(raw_dict.get("TransactionDT", 0.0))
        if transaction_id is None:
            txn_id_val = raw_dict.get("TransactionID")
            transaction_id = int(txn_id_val) if txn_id_val is not None else -1
        if entity_id is None:
            entity_id = _resolve_entity_id_from_dict(raw_dict, transaction_id)

        # -- A_t: Fast single-row preprocessing + LightGBM --
        x_vec = self._fast_prep.transform_single_row(raw_dict)  # (1, n_features)
        A_t = float(self._model.predict_risk(x_vec)[0])

        # -- P_t: Entity temporal risk (stateful O(1)) --
        _, P_t = self._temp_engine.step(entity_id, A_t)

        # -- G_t: Relational graph risk (stateful O(k)) --
        attr_dict = {a: (_safe_str(raw_dict.get(a))) for a in self._graph_attrs}
        rec = self._graph_engine.score(entity_id, timestamp, transaction_id, attr_dict)
        self._graph_engine.update(entity_id, timestamp, attr_dict)
        G_t = rec.G_t

        # -- R_t: Conditional fusion (frozen formula) --
        R_t = float(np.clip(A_t + self._alpha * P_t + self._beta_fusion * G_t, 0.0, 1.0))

        # -- Policy: O(1) threshold comparison --
        t = self._thresholds
        if R_t >= t.tau_block:
            action = PolicyAction.BLOCK
        elif R_t >= t.tau_throttle:
            action = PolicyAction.THROTTLE
        elif R_t >= t.tau_verify:
            action = PolicyAction.VERIFY
        else:
            action = PolicyAction.ALLOW

        explanation = generate_explanation(
            A_t=A_t, P_t=P_t, G_t=G_t, R_t=R_t,
            action=action, thresholds=t,
            D_t=rec.D_t, V_t=rec.V_t, d_t=rec.d_t, v_t=rec.v_t,
            device_info=raw_dict.get("DeviceInfo"),
        )

        return ScoringResult(
            transaction_id=transaction_id,
            entity_id=entity_id,
            A_t=A_t, P_t=P_t, G_t=G_t,
            D_t=rec.D_t, V_t=rec.V_t, d_t=rec.d_t, v_t=rec.v_t,
            R_t=R_t,
            action=action.value,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Batch scoring (uses fast_prep batch path + vectorised fusion/policy)
    # ------------------------------------------------------------------

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a chronologically ordered batch DataFrame.

        Returns df with additional columns:
            A_t, P_t, G_t, D_t, V_t, d_t, v_t, R_t, action, explanation
        """
        df = df.copy()

        # Entity proxy
        df["entity_proxy"] = resolve_entity_key(df, key_type=ENTITY_KEY_TYPE)

        # A_t — batch preprocessing (unchanged batch path)
        X = self._fast_prep.transform(df)
        df["A_t"] = self._model.predict_risk(X)

        # P_t — sequential temporal
        ents = df["entity_proxy"].values
        scores = df["A_t"].values
        P_arr = np.zeros(len(df), dtype=float)
        for i in range(len(df)):
            _, P_arr[i] = self._temp_engine.step(str(ents[i]), float(scores[i]))
        df["P_t"] = P_arr

        # G_t — sequential graph
        attrs = self._graph_attrs
        entities = df["entity_proxy"].astype(str).tolist()
        timestamps = df["TransactionDT"].astype(float).tolist()
        txn_ids = df["TransactionID"].astype(int).tolist()
        attr_col_vals = [df[a].tolist() for a in attrs]
        G_arr = np.zeros(len(df), dtype=float)
        D_arr = np.zeros(len(df), dtype=float)
        V_arr = np.zeros(len(df), dtype=float)
        d_arr = np.zeros(len(df), dtype=int)
        v_arr = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            attr_dict = {attrs[j]: _safe_str(attr_col_vals[j][i]) for j in range(len(attrs))}
            rec = self._graph_engine.score(entities[i], timestamps[i], txn_ids[i], attr_dict)
            self._graph_engine.update(entities[i], timestamps[i], attr_dict)
            G_arr[i] = rec.G_t
            D_arr[i] = rec.D_t
            V_arr[i] = rec.V_t
            d_arr[i] = rec.d_t
            v_arr[i] = rec.v_t
        df["G_t"] = G_arr
        df["D_t"] = D_arr
        df["V_t"] = V_arr
        df["d_t"] = d_arr
        df["v_t"] = v_arr

        # R_t — vectorised
        df["R_t"] = np.clip(
            df["A_t"].values + self._alpha * df["P_t"].values + self._beta_fusion * df["G_t"].values,
            0.0, 1.0
        )

        # Policy — vectorised
        t = self._thresholds
        R = df["R_t"].values
        acts = np.full(len(df), PolicyAction.ALLOW.value, dtype=object)
        acts[R >= t.tau_verify] = PolicyAction.VERIFY.value
        acts[R >= t.tau_throttle] = PolicyAction.THROTTLE.value
        acts[R >= t.tau_block] = PolicyAction.BLOCK.value
        df["action"] = acts

        return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_entity_id_from_dict(raw: Dict[str, Any], txn_id: int) -> str:
    """Derive card_addr_email entity proxy from a raw transaction dict."""
    card1 = raw.get("card1")
    addr1 = raw.get("addr1")
    email = raw.get("P_emaildomain")
    if (card1 is not None and not _is_nan(card1) and
            addr1 is not None and not _is_nan(addr1) and
            email is not None and not _is_nan(email)):
        return f"{card1}_{addr1}_{email}"
    return f"unresolved_{txn_id}"


def _is_nan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _safe_str(v: Any) -> Optional[str]:
    """Return str(v) unless v is None/NaN, in which case return None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return str(v)
