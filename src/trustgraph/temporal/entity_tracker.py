"""
entity_tracker.py — Entity-Scoped Temporal Risk Memory
========================================================

Responsibilities:
  - Extract and resolve defensible entity keys (card1, card_composite, card_email, card_addr)
  - Explicit missing/unresolved entity policy: unresolved transactions receive unique
    isolated keys (e.g. 'unresolved_<TransactionID>'), preventing cross-user contamination.
  - Stateful O(1) per-entity temporal tracking:
      E_{e, t}   = β * A_t + (1 - β) * E_{e, t-1}
      P_{e, t+1} = clip(P_{e, t} + Δ_t, 0.0, 1.0)
      where Δ_t = +λ if E_{e, t} > γ else -δ
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from trustgraph.temporal.engine import compute_ema, compute_bounded_accumulator


def resolve_entity_key(
    df: pd.DataFrame,
    key_type: str = "card_email",
) -> pd.Series:
    """
    Construct entity key series with strict missing value isolation.

    Candidate Keys:
      - 'card1': Primary payment card attribute (100% non-null)
      - 'card_composite': card1 + card2 + card3 + card4 + card5 + card6
      - 'card_email': card1 + P_emaildomain (User Account proxy)
      - 'card_addr': card1 + addr1 (Billing Address / Card proxy)

    Missing Value Policy:
      If any component required for the composite key is missing (NaN),
      the transaction is assigned a unique isolated identifier:
      'unresolved_<TransactionID>'.
      This prevents artificial cross-user contamination from a shared 'UNKNOWN' entity.
    """
    txn_ids = df["TransactionID"].astype(str)

    if key_type == "card1":
        # card1 has 0% missing in IEEE-CIS
        return df["card1"].astype(str)

    elif key_type == "card_composite":
        card_cols = ["card1", "card2", "card3", "card4", "card5", "card6"]
        available_cols = [c for c in card_cols if c in df.columns]
        # Check for missing values in any component
        has_null = df[available_cols].isna().any(axis=1)
        composite = df[available_cols[0]].astype(str)
        for c in available_cols[1:]:
            composite = composite + "_" + df[c].astype(str)
        # Isolate unresolved
        return composite.where(~has_null, "unresolved_" + txn_ids)

    elif key_type == "card_email":
        has_card = df["card1"].notna() if "card1" in df.columns else False
        has_email = df["P_emaildomain"].notna() if "P_emaildomain" in df.columns else False
        valid_mask = has_card & has_email
        combined = df["card1"].astype(str) + "_" + df["P_emaildomain"].astype(str)
        return combined.where(valid_mask, "unresolved_" + txn_ids)

    elif key_type == "card_addr":
        has_card = df["card1"].notna() if "card1" in df.columns else False
        has_addr = df["addr1"].notna() if "addr1" in df.columns else False
        valid_mask = has_card & has_addr
        combined = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        return combined.where(valid_mask, "unresolved_" + txn_ids)

    elif key_type == "card_addr_email":
        has_card = df["card1"].notna() if "card1" in df.columns else False
        has_addr = df["addr1"].notna() if "addr1" in df.columns else False
        has_email = df["P_emaildomain"].notna() if "P_emaildomain" in df.columns else False
        valid_mask = has_card & has_addr & has_email
        combined = df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + df["P_emaildomain"].astype(str)
        return combined.where(valid_mask, "unresolved_" + txn_ids)

    else:
        raise ValueError(f"Unknown key_type: {key_type}")


class EntityTemporalRiskEngine:
    """
    Entity-Scoped Temporal Risk Memory Engine.

    Maintains independent O(1) state (E_t, P_t) per unique entity.
    """

    def __init__(
        self,
        beta: float = 0.50,
        gamma: float = 0.50,
        lambda_: float = 0.05,
        delta: float = 0.05,
    ) -> None:
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta = delta
        # entity_id -> (E_state, P_state)
        self.states: Dict[str, Tuple[float, float]] = {}

    def reset(self) -> None:
        """Clear all entity states."""
        self.states.clear()

    def step(self, entity_id: str, A_t: float) -> Tuple[float, float]:
        """
        Execute one chronological update step for a specific entity.

        Parameters
        ----------
        entity_id : str — Entity identifier (e.g. '13553_gmail.com' or 'unresolved_3488960')
        A_t       : float in [0, 1] — Instantaneous model risk score

        Returns
        -------
        E_t : float in [0, 1] — Current entity EMA evidence
        P_t : float in [0, 1] — Current entity persistent risk
        """
        prev_E, prev_P = self.states.get(entity_id, (0.0, 0.0))

        E_t = compute_ema(A_t, prev_E, self.beta)
        current_P = prev_P

        next_P = compute_bounded_accumulator(
            E_t, prev_P, self.gamma, self.lambda_, self.delta
        )
        self.states[entity_id] = (E_t, next_P)

        return E_t, current_P

    def process_stream(
        self,
        df: pd.DataFrame,
        entity_col_or_series: Union[str, pd.Series],
        score_col: str = "A_t",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a DataFrame chronologically, maintaining entity-specific states.
        """
        N = len(df)
        E_out = np.empty(N, dtype=np.float64)
        P_out = np.empty(N, dtype=np.float64)

        self.reset()
        if isinstance(entity_col_or_series, str):
            entities = df[entity_col_or_series].values
        else:
            entities = entity_col_or_series.values

        scores = df[score_col].values

        for i in range(N):
            ent = str(entities[i])
            score = float(scores[i])
            e_val, p_val = self.step(ent, score)
            E_out[i] = e_val
            P_out[i] = p_val

        return E_out, P_out
