"""
fast_preprocessor.py — TRUSTGRAPH Phase 5A Single-Row Fast Preprocessing Path
===============================================================================

Adds a zero-DataFrame single-row scoring path to the frozen BaselinePreprocessor.
The full batch path (transform()) is UNCHANGED — it is still used for bulk evaluation.

Optimization strategy:
  The bottleneck in single-transaction scoring is that transform() wraps a single
  row in a full pd.DataFrame and runs pandas column operations (copy, map, reindex)
  — this costs ~11 ms per call due to Python/pandas object overhead, not computation.

  FastPreprocessor.transform_single_row(raw_dict) instead:
    1. Allocates a single float32 numpy array of shape (432,)
    2. For numeric features: direct float() cast, NaN for missing
    3. For categorical features: O(1) dict lookup into pre-built mapping tables
    4. Reorders into the frozen feature_cols order in one numpy index operation

  No DataFrame is created. No pandas is involved in the hot path.

Correctness guarantee:
  transform_single_row must produce values within float32 tolerance of transform()
  for every transaction. Verified in tests/test_runtime.py.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from trustgraph.baseline.preprocessing import BaselinePreprocessor

logger = logging.getLogger(__name__)

_NAN = float("nan")


class FastPreprocessor:
    """
    Thin wrapper around the frozen BaselinePreprocessor that adds a
    zero-DataFrame single-row preprocessing path.

    Usage:
        fp = FastPreprocessor(preprocessor)
        x_vec = fp.transform_single_row(raw_dict)  # shape (1, n_features)
    """

    def __init__(self, preprocessor: BaselinePreprocessor) -> None:
        self._preprocessor = preprocessor
        self.feature_cols: List[str] = preprocessor.feature_cols
        self.cat_cols_set: frozenset = frozenset(preprocessor.cat_cols)
        # Pre-build per-column category lookup dicts (str -> float code)
        # For missing/unknown -> NaN (same as batch path)
        self._cat_lookup: Dict[str, Dict[str, float]] = {}
        for col, mapping in preprocessor.encoder.mappings.items():
            # mapping is {str_val: int_code} — we want {str_val: float32(code)}
            self._cat_lookup[col] = {k: float(v) for k, v in mapping.items()}
        # Feature column index — numpy integer array for fast gather
        self._n_features = len(self.feature_cols)
        logger.debug(
            "FastPreprocessor ready: %d features, %d categorical lookup tables",
            self._n_features, len(self._cat_lookup),
        )

    def transform_single_row(self, raw_dict: Dict[str, Any]) -> np.ndarray:
        """
        Convert a single transaction dict to a (1, n_features) float32 numpy array.

        Parameters
        ----------
        raw_dict : dict mapping column_name -> raw value (str, int, float, or None/NaN)

        Returns
        -------
        np.ndarray of shape (1, n_features), dtype float32
        Identical to BaselinePreprocessor.transform(single_row_df) within float32 precision.
        """
        out = np.empty(self._n_features, dtype=np.float32)

        for i, col in enumerate(self.feature_cols):
            val = raw_dict.get(col)

            # Treat Python None and float NaN as missing
            if val is None or (isinstance(val, float) and math.isnan(val)):
                out[i] = _NAN
                continue

            if col in self._cat_lookup:
                # Categorical: look up string representation in mapping table
                out[i] = self._cat_lookup[col].get(str(val), _NAN)
            else:
                # Numeric: direct float cast; coerce errors to NaN
                try:
                    out[i] = float(val)
                except (TypeError, ValueError):
                    out[i] = _NAN

        return out.reshape(1, self._n_features)

    # ------------------------------------------------------------------
    # Passthrough to batch path (unchanged)
    # ------------------------------------------------------------------

    def transform(self, df) -> "pd.DataFrame":
        """Batch transform — delegates directly to the frozen BaselinePreprocessor."""
        return self._preprocessor.transform(df)
