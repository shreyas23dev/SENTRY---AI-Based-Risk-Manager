"""
config.py — TRUSTGRAPH Phase 4 Progressive Risk Decision Policy Configuration
=============================================================================

Constants, action enums, candidate threshold grids, and artifact paths
for the progressive risk decision policy layer.
"""

from enum import Enum
from pathlib import Path
from trustgraph.baseline.config import PROJECT_ROOT, RESULTS_DIR

BASELINE_THRESHOLD = 0.594298

# Artifact directories
POLICY_DIR       = PROJECT_ROOT / "artifacts" / "policy"
POLICY_PLOTS_DIR = POLICY_DIR / "plots"
FUSION_DIR       = PROJECT_ROOT / "artifacts" / "fusion"
BASELINE_DIR     = PROJECT_ROOT / "artifacts" / "baseline"

# Action Enumeration with natural severity ordering
class PolicyAction(str, Enum):
    ALLOW    = "ALLOW"
    VERIFY   = "VERIFY"
    THROTTLE = "THROTTLE"
    BLOCK    = "BLOCK"

    @property
    def severity_rank(self) -> int:
        ranks = {
            PolicyAction.ALLOW: 0,
            PolicyAction.VERIFY: 1,
            PolicyAction.THROTTLE: 2,
            PolicyAction.BLOCK: 3,
        }
        return ranks[self]


class RiskBand(str, Enum):
    LOW       = "LOW"
    MODERATE  = "MODERATE"
    HIGH      = "HIGH"
    VERY_HIGH = "VERY_HIGH"


# Candidate threshold search grids for Validation-Only Selection
CANDIDATE_TAU_VERIFY   = [0.50, 0.55, 0.60]
CANDIDATE_TAU_THROTTLE = [0.65, 0.70, 0.75]
CANDIDATE_TAU_BLOCK    = [0.80, 0.85, 0.90, 0.95]
