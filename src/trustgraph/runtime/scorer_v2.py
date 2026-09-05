"""
scorer_v2.py — TRUSTGRAPH V2 High-Performance Runtime Scorer
============================================================

Unified load-once, score-many runtime service for Baseline V2:
  - Fast single-row feature extractor for 452 features (base + time + amt + freq + running stream)
  - LightGBM V2 model inference
  - Frozen entity temporal memory (P_t)
  - Frozen lightweight relational graph (G_t)
  - Frozen conditional risk fusion (R_t)
  - Frozen progressive decision policy (ALLOW/VERIFY/THROTTLE/BLOCK)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.preprocessing import BaselinePreprocessor
from trustgraph.runtime.fast_preprocessor import FastPreprocessor
from trustgraph.features_v2.causal_features import FrequencyEncoder, EntityAmountStats
from trustgraph.temporal.entity_tracker import EntityTemporalRiskEngine, resolve_entity_key
from trustgraph.relational.graph_engine import GraphParameters, LightweightRelationalGraph
from trustgraph.policy.config import PolicyAction
from trustgraph.policy.decision_engine import PolicyThresholds, generate_explanation

logger = logging.getLogger(__name__)


@dataclass
class ScoringResultV2:
    transaction_id: Optional[int]
    entity_id: str
    A_t: float
    P_t: float
    G_t: float
    D_t: float
    V_t: float
    R_t: float
    action: str
    explanation: str


class FastPreprocessorV2:
    """
    Zero-DataFrame fast feature extractor for all 452 V2 features.
    """

    def __init__(
        self,
        base_prep: BaselinePreprocessor,
        freq_encoder: FrequencyEncoder,
        track_attrs: Tuple[str, ...] = ("card1", "addr1", "P_emaildomain", "DeviceInfo"),
    ) -> None:
        self.fast_base = FastPreprocessor(base_prep)
        self.freq_encoder = freq_encoder
        self.track_attrs = track_attrs
        self.total_features = len(base_prep.feature_cols) + len(freq_encoder.cols) + len(track_attrs) + 5 + 3 + 4  # 432 + 4 + 4 + 5 + 3 + 4 = 452

        # Causal state trackers for running features
        self.attr_counts: Dict[str, Dict[str, int]] = {a: {} for a in track_attrs}
        self.entity_stats: Dict[str, EntityAmountStats] = {}

    def transform_single_row(
        self,
        raw_dict: Dict[str, Any],
        entity_id: str,
        timestamp: float,
        amt: float,
    ) -> np.ndarray:
        """
        Extracts full 452-dim float32 feature vector with 0 DataFrame creation.
        """
        out = np.empty((1, self.total_features), dtype=np.float32)
        
        # 1. Base 432 features
        x_base = self.fast_base.transform_single_row(raw_dict)
        n_base = x_base.shape[1]
        out[0, :n_base] = x_base[0]
        offset = n_base

        # 2. Frequency encoding (4 features)
        for c in self.freq_encoder.cols:
            val = raw_dict.get(c)
            fval = self.freq_encoder.freq_maps.get(c, {}).get(str(val), 0.0) if val is not None and not (isinstance(val, float) and math.isnan(val)) else 0.0
            out[0, offset] = float(fval)
            offset += 1

        # 3. Running attribute prior counts (4 features)
        for a in self.track_attrs:
            val = raw_dict.get(a)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                out[0, offset] = float(self.attr_counts[a].get(str(val), 0))
            else:
                out[0, offset] = 0.0
            offset += 1

        # 4. Time features (5 features)
        dt = timestamp
        hour_of_day = (dt % 86400) // 3600
        day_of_week = (dt // 86400) % 7
        hour_angle = 2.0 * math.pi * hour_of_day / 24.0
        out[0, offset] = float(hour_of_day)
        out[0, offset + 1] = float(day_of_week)
        out[0, offset + 2] = float(math.sin(hour_angle))
        out[0, offset + 3] = float(math.cos(hour_angle))

        # Query entity stats for dt_elapsed
        st = self.entity_stats.get(entity_id)
        if st is not None and st.last_timestamp >= 0:
            out[0, offset + 4] = float(dt - st.last_timestamp)
        else:
            out[0, offset + 4] = -1.0
        offset += 5

        # 5. Amount features (3 features)
        log_amt = math.log1p(max(amt, 0.0))
        amt_decimal = round(amt - math.floor(amt), 4)
        amt_is_int = 1.0 if amt_decimal == 0.0 else 0.0
        out[0, offset] = float(log_amt)
        out[0, offset + 1] = float(amt_decimal)
        out[0, offset + 2] = float(amt_is_int)
        offset += 3

        # 6. Entity historical stats (4 features)
        if st is not None:
            p_cnt, p_mean, p_std, _ = st.get_stats_before(amt, dt)
        else:
            p_cnt, p_mean, p_std = 0.0, amt, 0.0

        out[0, offset] = float(p_cnt)
        out[0, offset + 1] = float(p_mean)
        out[0, offset + 2] = float(p_std)
        out[0, offset + 3] = float(amt / (p_mean + 1e-4))

        return out

    def update_state(self, raw_dict: Dict[str, Any], entity_id: str, timestamp: float, amt: float) -> None:
        """Causal state update performed strictly AFTER scoring."""
        if not entity_id.startswith("unresolved_"):
            if entity_id not in self.entity_stats:
                self.entity_stats[entity_id] = EntityAmountStats()
            self.entity_stats[entity_id].update(amt, timestamp)

            for a in self.track_attrs:
                val = raw_dict.get(a)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    val_str = str(val)
                    self.attr_counts[a][val_str] = self.attr_counts[a].get(val_str, 0) + 1


class RuntimeScorerV2:
    """
    Unified high-performance runtime service for TRUSTGRAPH V2.
    """

    def __init__(
        self,
        model: BaselineModel,
        fast_prep: FastPreprocessorV2,
        temp_engine: EntityTemporalRiskEngine,
        graph_engine: LightweightRelationalGraph,
        thresholds: PolicyThresholds,
        alpha: float = 1.0,
        beta_fusion: float = 0.05,
    ) -> None:
        self.model = model
        self.fast_prep = fast_prep
        self.temp_engine = temp_engine
        self.graph_engine = graph_engine
        self.thresholds = thresholds
        self.alpha = alpha
        self.beta_fusion = beta_fusion

    @classmethod
    def load(cls, artifacts_root: Path, base_train_df: pd.DataFrame) -> "RuntimeScorerV2":
        artifacts_root = Path(artifacts_root)
        model = BaselineModel.load(artifacts_root / "baseline_v2" / "model" / "lgbm_model.pkl")
        base_prep = BaselinePreprocessor.load(artifacts_root / "baseline" / "preprocessing")

        freq_enc = FrequencyEncoder(["card1", "addr1", "P_emaildomain", "DeviceInfo"]).fit(base_train_df)
        fast_prep = FastPreprocessorV2(base_prep, freq_enc)

        temp_engine = EntityTemporalRiskEngine(beta=0.3, gamma=0.5, lambda_=0.05, delta=0.05)
        rel_params = GraphParameters(k_attr_max=25, window_sec=86400.0, d_ref=3.0, v_ref=10.0, w_D=0.6, w_V=0.4, relational_attrs=("DeviceInfo",))
        graph_engine = LightweightRelationalGraph(rel_params)
        graph_engine.fit_attribute_frequency_ceiling(base_train_df)

        thresh_path = artifacts_root / "policy" / "thresholds.json"
        with open(thresh_path) as f:
            td = json.load(f)
        thresholds = PolicyThresholds(tau_verify=td["tau_verify"], tau_throttle=td["tau_throttle"], tau_block=td["tau_block"])

        return cls(
            model=model,
            fast_prep=fast_prep,
            temp_engine=temp_engine,
            graph_engine=graph_engine,
            thresholds=thresholds,
        )

    def score_transaction(
        self,
        raw_dict: Dict[str, Any],
        entity_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        transaction_id: Optional[int] = None,
    ) -> ScoringResultV2:
        if timestamp is None:
            timestamp = float(raw_dict.get("TransactionDT", 0.0))
        if transaction_id is None:
            transaction_id = int(raw_dict.get("TransactionID", -1))
        amt = float(raw_dict.get("TransactionAmt", 0.0))
        if entity_id is None:
            c1, a1, em = raw_dict.get("card1"), raw_dict.get("addr1"), raw_dict.get("P_emaildomain")
            if c1 is not None and a1 is not None and em is not None and not (isinstance(c1, float) and math.isnan(c1)) and not (isinstance(a1, float) and math.isnan(a1)) and not (isinstance(em, float) and math.isnan(em)):
                entity_id = f"{c1}_{a1}_{em}"
            else:
                entity_id = f"unresolved_{transaction_id}"

        # 1. Feature extraction & LightGBM predict
        x_vec = self.fast_prep.transform_single_row(raw_dict, entity_id, timestamp, amt)
        A_t = float(self.model.predict_risk(x_vec)[0])

        # 2. Temporal engine
        _, P_t = self.temp_engine.step(entity_id, A_t)

        # 3. Relational graph
        attr_dict = {"DeviceInfo": str(raw_dict["DeviceInfo"]) if raw_dict.get("DeviceInfo") is not None and not (isinstance(raw_dict.get("DeviceInfo"), float) and math.isnan(raw_dict["DeviceInfo"])) else None}
        rec = self.graph_engine.score(entity_id, timestamp, transaction_id, attr_dict)
        G_t = rec.G_t

        # Causal State updates (performed strictly after scoring)
        self.fast_prep.update_state(raw_dict, entity_id, timestamp, amt)
        self.graph_engine.update(entity_id, timestamp, attr_dict)

        # 4. Fused risk
        R_t = float(np.clip(A_t + self.alpha * P_t + self.beta_fusion * G_t, 0.0, 1.0))

        # 5. Policy assignment
        if R_t >= self.thresholds.tau_block:
            action = PolicyAction.BLOCK
        elif R_t >= self.thresholds.tau_throttle:
            action = PolicyAction.THROTTLE
        elif R_t >= self.thresholds.tau_verify:
            action = PolicyAction.VERIFY
        else:
            action = PolicyAction.ALLOW

        explanation = generate_explanation(
            A_t=A_t, P_t=P_t, G_t=G_t, R_t=R_t,
            action=action, thresholds=self.thresholds,
            D_t=rec.D_t, V_t=rec.V_t, d_t=rec.d_t, v_t=rec.v_t,
            device_info=raw_dict.get("DeviceInfo"),
        )

        return ScoringResultV2(
            transaction_id=transaction_id,
            entity_id=entity_id,
            A_t=A_t, P_t=P_t, G_t=G_t,
            D_t=rec.D_t, V_t=rec.V_t,
            R_t=R_t,
            action=action.value,
            explanation=explanation,
        )
