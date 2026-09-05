"""
engine.py — TRUSTGRAPH Phase 2 Temporal Risk Memory Engine
===========================================================

Formulation:
  1. Exponential Moving Average (EMA) Evidence:
       E_t = β * A_t + (1 - β) * E_{t-1}
       E_0 = 0.0

  2. Bounded Persistent-Risk Accumulator:
       Δ_t = λ    if E_t > γ
            -δ    otherwise

       P_{t+1} = clip(P_t + Δ_t, 0.0, 1.0)
       P_0 = 0.0

Guarantees:
  - Strict bounds: 0.0 <= E_t <= 1.0, 0.0 <= P_t <= 1.0
  - O(1) time complexity per transaction
  - O(1) memory state per tracked sequence/stream
  - Causal: depends only on current A_t and previous state
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_ema(A_t: float, prev_E: float, beta: float) -> float:
    """
    Recursive Exponential Moving Average of fraud-risk scores.

    Parameters
    ----------
    A_t    : float in [0, 1] — Instantaneous model risk score P(isFraud=1 | X_t)
    prev_E : float in [0, 1] — Previous EMA state E_{t-1}
    beta   : float in (0, 1] — Memory weight parameter

    Returns
    -------
    E_t    : float in [0, 1]
    """
    if not (0.0 <= A_t <= 1.0):
        raise ValueError(f"A_t must be in [0, 1], got {A_t}")
    if not (0.0 <= prev_E <= 1.0):
        raise ValueError(f"prev_E must be in [0, 1], got {prev_E}")
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    E_t = beta * A_t + (1.0 - beta) * prev_E
    return float(np.clip(E_t, 0.0, 1.0))


def compute_bounded_accumulator(
    E_t: float,
    prev_P: float,
    gamma: float,
    lambda_: float,
    delta: float,
) -> float:
    """
    Asymmetric Bounded Persistent-Risk Accumulator step.

    Parameters
    ----------
    E_t     : float in [0, 1] — Current EMA evidence
    prev_P  : float in [0, 1] — Previous persistent-risk state P_t
    gamma   : float in (0, 1) — Suspicion threshold for accumulation
    lambda_ : float > 0       — Upward accumulation step size
    delta   : float > 0       — Downward decay step size

    Returns
    -------
    next_P  : float in [0, 1] — Updated bounded persistent-risk state P_{t+1}
    """
    if not (0.0 <= E_t <= 1.0):
        raise ValueError(f"E_t must be in [0, 1], got {E_t}")
    if not (0.0 <= prev_P <= 1.0):
        raise ValueError(f"prev_P must be in [0, 1], got {prev_P}")

    if E_t > gamma:
        delta_t = lambda_
    else:
        delta_t = -delta

    next_P = np.clip(prev_P + delta_t, 0.0, 1.0)
    return float(next_P)


class TemporalRiskEngine:
    """
    Stateful Temporal Risk Memory Engine.

    Maintains O(1) state (E_t, P_t) across a sequential stream of transactions.
    """

    def __init__(
        self,
        beta: float = 0.40,
        gamma: float = 0.30,
        lambda_: float = 0.20,
        delta: float = 0.05,
    ) -> None:
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta = delta

        self.E_state: float = 0.0
        self.P_state: float = 0.0
        self.step_count: int = 0

    def reset(self) -> None:
        """Reset internal state to E_0 = 0.0, P_0 = 0.0."""
        self.E_state = 0.0
        self.P_state = 0.0
        self.step_count = 0

    def step(self, A_t: float) -> Tuple[float, float]:
        """
        Execute one chronological update step.

        Parameters
        ----------
        A_t : float in [0, 1] — Current transaction risk score

        Returns
        -------
        E_t : float in [0, 1] — Current EMA evidence
        P_t : float in [0, 1] — Current persistent-risk state (prior to transition)
        """
        # Current EMA evidence
        self.E_state = compute_ema(A_t, self.E_state, self.beta)
        current_P = self.P_state

        # Update persistent risk state for next transaction
        self.P_state = compute_bounded_accumulator(
            self.E_state, self.P_state, self.gamma, self.lambda_, self.delta
        )
        self.step_count += 1

        return self.E_state, current_P

    def process_stream(self, A_t_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a 1D sequence of A_t scores in chronological order.

        Parameters
        ----------
        A_t_array : np.ndarray, shape (N,)

        Returns
        -------
        E_array : np.ndarray, shape (N,) — EMA evidence at each step
        P_array : np.ndarray, shape (N,) — Persistent-risk state at each step
        """
        N = len(A_t_array)
        E_out = np.empty(N, dtype=np.float64)
        P_out = np.empty(N, dtype=np.float64)

        self.reset()
        for i in range(N):
            E_val, P_val = self.step(float(A_t_array[i]))
            E_out[i] = E_val
            P_out[i] = P_val

        return E_out, P_out


class EntityTemporalRiskTracker:
    """
    Multi-entity Temporal Risk Memory Tracker.

    Maintains independent O(1) state (E_t, P_t) per unique entity
    (e.g., payment card, user account, email domain).
    """

    def __init__(
        self,
        beta: float = 0.40,
        gamma: float = 0.30,
        lambda_: float = 0.20,
        delta: float = 0.05,
    ) -> None:
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta = delta
        # entity_id -> (E_state, P_state)
        self.states: Dict[Union[str, int, float], Tuple[float, float]] = {}

    def reset(self) -> None:
        """Clear all entity states."""
        self.states.clear()

    def step(self, entity_id: Union[str, int, float], A_t: float) -> Tuple[float, float]:
        """
        Update temporal state for a specific entity.

        Parameters
        ----------
        entity_id : identifier of the entity
        A_t       : float in [0, 1]

        Returns
        -------
        E_t, P_t : float in [0, 1]
        """
        prev_E, prev_P = self.states.get(entity_id, (0.0, 0.0))
        E_t = compute_ema(A_t, prev_E, self.beta)
        current_P = prev_P
        next_P = compute_bounded_accumulator(
            E_t, prev_P, self.gamma, self.lambda_, self.delta
        )
        self.states[entity_id] = (E_t, next_P)
        return E_t, current_P

    def process_dataframe(
        self,
        df: pd.DataFrame,
        entity_col: str,
        score_col: str = "A_t",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a DataFrame chronologically, maintaining entity-specific states.
        """
        N = len(df)
        E_out = np.empty(N, dtype=np.float64)
        P_out = np.empty(N, dtype=np.float64)

        self.reset()
        entities = df[entity_col].values
        scores   = df[score_col].values

        for i in range(N):
            ent = entities[i]
            score = float(scores[i])
            e_val, p_val = self.step(ent, score)
            E_out[i] = e_val
            P_out[i] = p_val

        return E_out, P_out
