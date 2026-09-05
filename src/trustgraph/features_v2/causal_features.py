"""
causal_features.py — TRUSTGRAPH Causal Feature Engineering Engine (V2)
======================================================================

Implements strictly causal, point-in-time and sequential feature generators:
  1. Time features (hour_of_day, day_of_week, hour_sin, hour_cos, entity elapsed time)
  2. Amount features (log_TransactionAmt, amt_decimal_cents, amt_is_integer)
  3. Frequency encoding (fitted on TRAIN only)
  4. Causal running frequencies (entity/group transaction counts prior to t)
  5. Causal running amount statistics (Welford's running mean, std, ratio prior to t)

Guarantees:
  - Strict causality: features for transaction t only observe events before t.
  - Zero target leakage: isFraud is never referenced.
  - Test/Validation isolation: No future lookahead.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Point-in-time stateless transforms (Pure functions of current row)
# ---------------------------------------------------------------------------

def compute_point_in_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-of-day, cyclics, and amount transforms that depend
    strictly on the transaction's own timestamp and amount.
    """
    dt = df["TransactionDT"].values
    amt = df["TransactionAmt"].values

    # Seconds in day = 86400, seconds in hour = 3600
    hour_of_day = (dt % 86400) // 3600
    day_of_week = (dt // 86400) % 7

    hour_angle = 2.0 * math.pi * hour_of_day / 24.0
    hour_sin = np.sin(hour_angle)
    hour_cos = np.cos(hour_angle)

    # Safe log transform (amounts >= 0)
    log_amt = np.log1p(np.maximum(amt, 0.0))

    # Decimal cents component (e.g. 125.95 -> 0.95)
    amt_decimal = np.round(amt - np.floor(amt), 4)
    amt_is_integer = (amt_decimal == 0.0).astype(np.float32)

    return pd.DataFrame({
        "hour_of_day": hour_of_day.astype(np.float32),
        "day_of_week": day_of_week.astype(np.float32),
        "hour_sin": hour_sin.astype(np.float32),
        "hour_cos": hour_cos.astype(np.float32),
        "log_TransactionAmt": log_amt.astype(np.float32),
        "amt_decimal_cents": amt_decimal.astype(np.float32),
        "amt_is_integer": amt_is_integer,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Frequency Encoder (Fit on TRAIN only)
# ---------------------------------------------------------------------------

class FrequencyEncoder:
    """
    Computes normalized frequency count of categorical values on TRAIN only.
    Unseen categories in validation/test are mapped to 0.0.
    """

    def __init__(self, cols: List[str]) -> None:
        self.cols = cols
        self.freq_maps: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> "FrequencyEncoder":
        n_rows = float(len(df))
        self.freq_maps = {}
        for c in self.cols:
            if c in df.columns:
                counts = df[c].dropna().astype(str).value_counts()
                self.freq_maps[c] = {k: float(v) / n_rows for k, v in counts.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out_dict = {}
        for c in self.cols:
            if c in self.freq_maps:
                fmap = self.freq_maps[c]
                vals = df[c].astype(object)
                out_dict[f"freq_{c}"] = vals.map(lambda v: fmap.get(str(v), 0.0) if pd.notna(v) else 0.0).astype(np.float32)
        return pd.DataFrame(out_dict, index=df.index)


# ---------------------------------------------------------------------------
# Stateful Causal Stream Feature Generator (Causal running stats prior to t)
# ---------------------------------------------------------------------------

@dataclass
class EntityAmountStats:
    """Welford's online running mean and M2 variance accumulator."""
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0
    last_timestamp: float = -1.0

    def update(self, amt: float, timestamp: float) -> None:
        self.count += 1
        delta = amt - self.mean
        self.mean += delta / self.count
        delta2 = amt - self.mean
        self.M2 += delta * delta2
        self.last_timestamp = timestamp

    def get_stats_before(self, current_amt: float, current_ts: float) -> Tuple[float, float, float, float]:
        """
        Returns (prior_count, prior_mean, prior_std, dt_elapsed)
        strictly from history before the current update.
        """
        prior_count = float(self.count)
        prior_mean = self.mean if self.count > 0 else current_amt
        prior_std = math.sqrt(self.M2 / self.count) if self.count > 1 else 0.0
        dt_elapsed = (current_ts - self.last_timestamp) if self.last_timestamp >= 0 else -1.0
        return prior_count, prior_mean, prior_std, dt_elapsed


class CausalStreamFeatureEngine:
    """
    Stateful chronological processor that tracks prior frequencies and
    historical amount statistics for entities/attributes strictly before t.
    """

    def __init__(self, track_attrs: Tuple[str, ...] = ("card1", "addr1", "P_emaildomain", "DeviceInfo")) -> None:
        self.track_attrs = track_attrs
        # Attribute value running frequencies: attr_name -> {attr_val: count}
        self.attr_counts: Dict[str, Dict[str, int]] = {a: defaultdict(int) for a in track_attrs}
        # Entity proxy running amount stats: entity_proxy -> EntityAmountStats
        self.entity_stats: Dict[str, EntityAmountStats] = defaultdict(EntityAmountStats)

    def reset(self) -> None:
        self.attr_counts = {a: defaultdict(int) for a in self.track_attrs}
        self.entity_stats = defaultdict(EntityAmountStats)

    def process_partition(
        self,
        df: pd.DataFrame,
        entity_col: str = "entity_proxy",
        timestamp_col: str = "TransactionDT",
        amt_col: str = "TransactionAmt",
    ) -> pd.DataFrame:
        """
        Processes a dataframe chronologically, generating causal features:
          - prior_count_<attr> : count of prior transactions with this attribute value
          - entity_prior_count : count of prior transactions for this entity proxy
          - entity_hist_mean_amt : running mean amount for this entity prior to t
          - entity_hist_std_amt : running std amount for this entity prior to t
          - entity_amt_ratio : current_amt / (hist_mean + 1e-4)
          - entity_dt_elapsed : seconds since last transaction by this entity
        """
        N = len(df)
        entities = df[entity_col].astype(str).values
        timestamps = df[timestamp_col].astype(float).values
        amounts = df[amt_col].astype(float).values

        # Prepare column arrays for attribute values
        attr_vals = {}
        for a in self.track_attrs:
            if a in df.columns:
                attr_vals[a] = df[a].values
            else:
                attr_vals[a] = np.full(N, None)

        # Output feature buffers
        out_freqs = {f"prior_count_{a}": np.zeros(N, dtype=np.float32) for a in self.track_attrs}
        out_ent_count = np.zeros(N, dtype=np.float32)
        out_hist_mean = np.zeros(N, dtype=np.float32)
        out_hist_std = np.zeros(N, dtype=np.float32)
        out_amt_ratio = np.zeros(N, dtype=np.float32)
        out_dt_elapsed = np.zeros(N, dtype=np.float32)

        for i in range(N):
            ent = entities[i]
            ts = timestamps[i]
            amt = amounts[i]

            # 1. Query prior frequencies (strictly BEFORE t)
            for a in self.track_attrs:
                val = attr_vals[a][i]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    val_str = str(val)
                    out_freqs[f"prior_count_{a}"][i] = float(self.attr_counts[a][val_str])
                else:
                    out_freqs[f"prior_count_{a}"][i] = 0.0

            # 2. Query entity historical amount stats (strictly BEFORE t)
            st = self.entity_stats[ent]
            p_cnt, p_mean, p_std, dt_el = st.get_stats_before(amt, ts)
            out_ent_count[i] = p_cnt
            out_hist_mean[i] = p_mean
            out_hist_std[i] = p_std
            out_amt_ratio[i] = amt / (p_mean + 1e-4)
            out_dt_elapsed[i] = dt_el

            # 3. Insert / update state (AFTER scoring row i)
            if not ent.startswith("unresolved_"):
                st.update(amt, ts)
                for a in self.track_attrs:
                    val = attr_vals[a][i]
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        self.attr_counts[a][str(val)] += 1

        result_dict = {
            "entity_prior_count": out_ent_count,
            "entity_hist_mean_amt": out_hist_mean,
            "entity_hist_std_amt": out_hist_std,
            "entity_amt_ratio": out_amt_ratio,
            "entity_dt_elapsed": out_dt_elapsed,
        }
        result_dict.update(out_freqs)
        return pd.DataFrame(result_dict, index=df.index)
