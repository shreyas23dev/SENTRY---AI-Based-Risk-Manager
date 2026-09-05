"""
config.py — TRUSTGRAPH Phase 2 Temporal Risk Memory Configuration
==================================================================

Constants, default parameter ranges, and artifact paths for Phase 2.
"""

from pathlib import Path
from trustgraph.baseline.config import PROJECT_ROOT, ARTIFACTS_DIR as BASELINE_ARTIFACTS_DIR, RESULTS_DIR

# Artifacts & Results
TEMPORAL_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "temporal"
TEMPORAL_PLOTS_DIR     = TEMPORAL_ARTIFACTS_DIR / "plots"
TEST_PREDICTIONS_CSV   = RESULTS_DIR / "test_predictions.csv"
TEMPORAL_PREDICTIONS_CSV = RESULTS_DIR / "temporal_predictions.csv"

# Candidate search ranges for VALIDATION ONLY tuning
CANDIDATE_BETAS   = [0.2, 0.3, 0.4, 0.5, 0.6]
CANDIDATE_GAMMAS  = [0.10, 0.20, 0.30, 0.40, 0.50]
CANDIDATE_LAMBDAS = [0.05, 0.10, 0.20, 0.30]
CANDIDATE_DELTAS  = [0.01, 0.02, 0.05, 0.10]
CANDIDATE_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Default / Fallback parameter values (recalibrated on validation)
DEFAULT_BETA    = 0.40
DEFAULT_GAMMA   = 0.30
DEFAULT_LAMBDA  = 0.20
DEFAULT_DELTA   = 0.05
DEFAULT_TEMPORAL_THRESHOLD = 0.50
