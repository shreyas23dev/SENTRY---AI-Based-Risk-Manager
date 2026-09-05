"""
config.py — TRUSTGRAPH Phase 3.1 Conditional Risk Fusion Configuration
======================================================================

Configuration and candidate search grids for conditional risk fusion.
Validation-only selection. TEST partition accessed only after freeze.
"""

from pathlib import Path
from trustgraph.baseline.config import PROJECT_ROOT, RESULTS_DIR

BASELINE_THRESHOLD = 0.594298

# Artifact directories
FUSION_DIR       = PROJECT_ROOT / "artifacts" / "fusion"
FUSION_PLOTS_DIR = FUSION_DIR / "plots"
BASELINE_DIR     = PROJECT_ROOT / "artifacts" / "baseline"
TEMPORAL_DIR     = PROJECT_ROOT / "artifacts" / "temporal_entity"
RELATIONAL_DIR   = PROJECT_ROOT / "artifacts" / "relational"

# Frozen upstream parameters
TEMPORAL_BETA       = 0.30
TEMPORAL_GAMMA      = 0.50
TEMPORAL_LAMBDA     = 0.05
TEMPORAL_DELTA      = 0.05
TEMPORAL_THRESHOLD  = 0.70

RELATIONAL_K_MAX    = 25
RELATIONAL_WINDOW   = 86_400.0   # 24 h
RELATIONAL_D_REF    = 3.0
RELATIONAL_V_REF    = 10.0
RELATIONAL_WD       = 0.60
RELATIONAL_WV       = 0.40
RELATIONAL_THRESHOLD= 0.60
ENTITY_KEY_TYPE     = "card_addr_email"

# Candidate search grids for Validation-Only Tuning
ALPHA_GRID = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
BETA_GRID  = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
CP_GRID    = [0.40, 0.60, 0.80, 1.00, 1.20]
CG_GRID    = [0.40, 0.60, 0.80, 1.00, 1.20]
TAU_COMB_GRID = [0.45, 0.50, 0.55, BASELINE_THRESHOLD, 0.65, 0.70]
