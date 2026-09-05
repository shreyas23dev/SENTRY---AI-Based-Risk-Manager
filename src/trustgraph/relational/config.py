"""
config.py — Phase 3 Relational Risk Configuration
===================================================

All candidate grids for validation-only staged parameter selection.
Graph state persists across TRAIN → VALIDATION → TEST.
Attribute-frequency ceiling fitted on TRAIN only.
"""

from pathlib import Path

from trustgraph.baseline.config import PROJECT_ROOT, RESULTS_DIR

# Artifact directories
RELATIONAL_DIR     = PROJECT_ROOT / "artifacts" / "relational"
PLOTS_DIR          = RELATIONAL_DIR / "plots"
BASELINE_DIR       = PROJECT_ROOT / "artifacts" / "baseline"
TEMPORAL_ENTITY_DIR= PROJECT_ROOT / "artifacts" / "temporal_entity"

# Frozen upstream inputs
FROZEN_TEST_PREDS  = RESULTS_DIR / "test_predictions.csv"
TEMPORAL_ENTITY_PREDS = RESULTS_DIR / "temporal_entity_predictions.csv"
TRAIN_TRANSACTION_CSV = PROJECT_ROOT / "ieee-fraud-detection" / "train_transaction.csv"
TRAIN_IDENTITY_CSV    = PROJECT_ROOT / "ieee-fraud-detection" / "train_identity.csv"

# Frozen baseline threshold (Phase 1)
BASELINE_THRESHOLD = 0.594298

# Frozen temporal parameters (Phase 2.1 — card_addr_email entity proxy)
TEMPORAL_BETA       = 0.30
TEMPORAL_GAMMA      = 0.50
TEMPORAL_LAMBDA     = 0.05
TEMPORAL_DELTA      = 0.05
TEMPORAL_THRESHOLD  = 0.70
ENTITY_KEY_TYPE     = "card_addr_email"   # card1 + addr1 + P_emaildomain

# Relational attributes for graph construction
# addr1 and P_emaildomain EXCLUDED: already embedded in entity proxy, far too coarse.
# DeviceInfo: primary relational attribute, specificity-filtered by k_attr_max.
RELATIONAL_ATTRIBUTES = ["DeviceInfo"]

# Stage 1 candidate grids
K_ATTR_MAX_GRID  = [10, 25, 50, 100, 250]
WINDOW_GRID_SEC  = [86_400, 604_800]   # 24 h, 7 days
D_REF_GRID       = [3, 5, 10, 20]
V_REF_GRID       = [2, 3, 5, 10]

# Stage 2 candidate grids (w_D, w_V where w_D + w_V = 1)
WD_WV_GRID = [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6), (0.7, 0.3)]

# Stage 3 candidate grids for combined weights (w_A + w_P + w_G = 1)
COMBINED_WEIGHT_GRID = [
    (0.60, 0.20, 0.20),
    (0.50, 0.25, 0.25),
    (0.40, 0.30, 0.30),
    (0.34, 0.33, 0.33),
]

# Stage 4 thresholds
REL_THRESHOLD_GRID  = [0.4, 0.5, 0.6, 0.7]
COMB_THRESHOLD_GRID = [0.4, 0.5, 0.6, 0.7]

# Ablation attribute sets (G1 primary; G2/G3 overlap-contamination ablations)
ABLATION_ATTR_SETS = {
    "G1_device_only":              ["DeviceInfo"],
    "G2_device_addr":              ["DeviceInfo", "addr1"],
    "G3_device_addr_email":        ["DeviceInfo", "addr1", "P_emaildomain"],
}
