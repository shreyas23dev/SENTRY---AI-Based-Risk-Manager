"""
explanation.py — Causal Signal-to-Explanation Engine for TRUSTGRAPH
====================================================================

Generates human-readable, auditable explanation strings strictly derived
from the real features and risk components ($A_t, P_t, G_t, R_t, d_t, v_t$)
produced by the engine. Zero fabricated statements.
"""

from typing import List, Optional


def generate_signal_explanations(
    A_t: float,
    P_t: float,
    G_t: float,
    R_t: float,
    d_t: int,
    v_t: int,
    decision: str,
    device_info: Optional[str] = None,
    threshold_verify: float = 0.60,
    threshold_block: float = 0.80,
) -> List[str]:
    """
    Produce structured, human-readable reasons derived from the actual signals.

    Parameters
    ----------
    A_t : float (Point-wise LightGBM baseline risk)
    P_t : float (Entity temporal risk accumulator)
    G_t : float (Relational graph risk)
    R_t : float (Combined fused risk)
    d_t : int (Distinct entities connected via DeviceInfo)
    v_t : int (New connection events in 24h window)
    decision : str ("ALLOW", "VERIFY", "THROTTLE", "BLOCK")
    device_info : Optional[str] (Raw device string if present)
    threshold_verify : float (Default 0.60)
    threshold_block : float (Default 0.80)

    Returns
    -------
    List[str] of specific reasons explaining why the transaction received this score and action.
    """
    reasons: List[str] = []

    # 1. Point-wise tabular baseline risk (A_t)
    if A_t >= threshold_block:
        reasons.append(f"Severe baseline transaction risk (point-wise tabular score: {A_t:.4f})")
    elif A_t >= threshold_verify:
        reasons.append(f"Elevated baseline transaction risk (point-wise tabular score: {A_t:.4f})")
    elif A_t >= 0.40 and decision != "ALLOW":
        reasons.append(f"Moderate baseline transaction risk (point-wise tabular score: {A_t:.4f})")

    # 2. Entity temporal accumulated velocity (P_t)
    if P_t >= 0.30:
        reasons.append(f"Elevated recent entity risk with persistent longitudinal velocity (P_t: {P_t:.4f})")
    elif P_t > 0.0:
        reasons.append(f"Mild recent entity risk detected on historical proxy (P_t: {P_t:.4f})")

    # 3. Relational graph risk & topological features (G_t, d_t, v_t)
    if G_t >= 0.50:
        dev_tag = f" on device '{device_info}'" if device_info and str(device_info) != "nan" else ""
        reasons.append(f"High relational risk across device network (G_t: {G_t:.4f}{dev_tag})")
    elif G_t > 0.0:
        reasons.append(f"Elevated relational risk from shared device context (G_t: {G_t:.4f})")

    if d_t > 1:
        reasons.append(f"Device linked to multiple entities ({d_t} distinct entities connected)")
    elif d_t == 1 and G_t > 0.0:
        reasons.append("Device linked to known entity network")

    if v_t > 1:
        reasons.append(f"High device transaction velocity ({v_t} new connection events in 24h window)")

    # 4. Contextual evidence uplift dynamics
    contextual_uplift = R_t - A_t
    if contextual_uplift >= 0.01:
        reasons.append(f"Risk increased by contextual evidence (+{contextual_uplift:.4f} combined temporal and graph uplift)")
        if A_t < threshold_verify and R_t >= threshold_verify:
            reasons.append(f"Contextual evidence escalated decision from baseline ALLOW to {decision}")

    # 5. Fallback for clean low-risk transactions
    if not reasons:
        if decision == "ALLOW":
            reasons.append("All baseline and contextual risk signals within low-risk operating thresholds")
        else:
            reasons.append(f"Transaction evaluated at combined risk score {R_t:.4f}")

    return reasons
