# TRUSTGRAPH Phase 3: Mathematical Risk Engine + Cost-Aware Decision Engine

## Overview

Phase 3 builds the mathematical decision layer that combines:
- $A_t \in [0, 1]$: IEEE-CIS XGBoost transaction-level fraud probability (Phase 1, frozen)
- $G_t \in [0, 1]$: Point-in-time Knowledge Graph contextual entity risk (Phase 2, frozen)

into a final continuous risk score $R_t \in [0, 1]$ and maps that risk into a merchant-facing action (`ALLOW`, `VERIFY`, `THROTTLE`, `BLOCK`) that minimises expected financial and operational loss.

---

## 1. Upstream Signals (Frozen)

Both signals are treated as strictly immutable inputs:

1. **Transaction-Level ML Baseline ($A_t$)**:
   - Model: IEEE-CIS XGBoost classifier (611 trees, 263 engineered features).
   - Character: High precision on transaction-level behavioral anomalies, tabular signals, and amount/card interactions.
   - Scale: Tree-ensemble posterior probability with mean $\approx 0.0234$ on TEST ($0.2873$ on fraud, $0.0138$ on legitimate).

2. **Knowledge Graph Entity Risk ($G_t$)**:
   - Model: Causal point-in-time knowledge graph with $O(1)$ deque and counter state indexes.
   - Character: Strong recall on recidivist entities, shared devices, and 2-hop graph contamination rings.
   - Scale: Hand-crafted continuous risk score in $[0, 1]$ with mean $\approx 0.1905$ on TEST ($0.4396$ on fraud, $0.1816$ on legitimate).

---

## 2. Signal Calibration Methodology

### Rationale
$A_t$ is an ensemble posterior probability, whereas $G_t$ is an analytically weighted structural risk index. Prior to fusion, we evaluate whether calibration via Platt Scaling (logistic sigmoid) or Isotonic Regression is required to align them on a common empirical probability scale.

### Calibration Protocol
- **Fitting**: Calibrators are fitted strictly on the **TRAIN** partition ($N = 413,379$).
- **Selection Criterion**: Evaluated on **VALIDATION** ($N = 88,581$). A calibrator is only activated if Expected Calibration Error (ECE) improves by at least $\Delta \ge 0.005$ (0.5 percentage points):
  $$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
- **Monotonicity**: Both Platt and Isotonic calibration are strictly monotonic, preserving signal rank order.
- **Leakage Prevention**: TEST labels are never seen by the calibrators.

---

## 3. Mathematical Fusion Formulations

Four candidate fusion equations were formulated and evaluated:

### F1 — Additive Fusion
$$R_t = \text{clip}(A_t + \alpha \cdot G_t, \; 0, \; 1)$$
- **Rationale**: Simple additive uplift.
- **Limitation**: Can push already high $A_t$ into saturation and aggressively inflates false positives when graph context is noisy.
- **Search Space**: $\alpha \in [0.05, 0.50]$.

### F2 — Conditional Graph Contribution (Preferred)
$$R_t = \text{clip}\left(A_t + \beta \cdot G_t \cdot (1 - A_t), \; 0, \; 1\right)$$
- **Rationale**: The knowledge graph provides **residual information** that the tabular ML model does not capture:
  - When $A_t \to 1$ (high ML certainty), $(1 - A_t) \to 0$: the graph cannot push the transaction beyond certainty, preventing double-counting.
  - When $A_t \to 0$ (ML sees nothing wrong), the graph can contribute up to $\beta \cdot G_t$ of residual risk uplift to catch repeat fraudsters.
  - When $G_t = 0$ (cold-start entity), $R_t = A_t$ exactly.
- **Search Space**: $\beta \in [0.05, 0.50]$.

### F3 — Multi-Signal Conditional Fusion
$$R_t = \text{clip}\left(A_t + \alpha \cdot P_t \cdot (1 - A_t) + \beta \cdot G_t \cdot (1 - A_t), \; 0, \; 1\right)$$
- **Status**: Skipped. The legacy relational temporal signal $P_t$ was trained on a different feature distribution and is not a valid posterior estimate under the Phase 1/2 architecture.

### F4 — Conservative Maximum
$$R_t = \text{clip}\left(\max(A_t, \; c \cdot G_t), \; 0, \; 1\right)$$
- **Rationale**: Never dilutes high ML risk, but allows strong graph evidence alone to trigger intervention.
- **Search Space**: $c \in [0.50, 1.20]$.

---

## 4. Parameter Selection Protocol (Validation Only)

1. A grid search over all parameter combinations is evaluated strictly on **VALIDATION** ($N = 88,581$, $3,042$ frauds).
2. Primary selection metric: **PR-AUC** (Precision-Recall Area Under Curve), the gold standard for highly imbalanced fraud detection ($3.4\%$ base rate).
3. The optimal decision threshold $\tau^*$ is selected by finding the F1-maximising cutoff on VALIDATION.
4. Once selected, all weights and thresholds are **FROZEN** and committed to `artifacts/risk/frozen_params.json`.
5. Only then is a single, one-shot evaluation executed on held-out **TEST** ($N = 88,580$, $3,083$ frauds).

---

## 5. Cost-Aware Decision Engine

Rather than using arbitrary score thresholds, the decision engine models the merchant's expected financial loss for each possible action given risk $R_t$ and transaction amount $\text{Amt}$:

$$\text{Action}^* = \arg\min_{a \in \{\text{ALLOW}, \text{VERIFY}, \text{THROTTLE}, \text{BLOCK}\}} \mathbb{E}[\text{Cost}(a \mid R_t, \text{Amt})]$$

### Action Expected Cost Equations

1. **ALLOW**:
   $$\mathbb{E}[\text{Cost} \mid \text{ALLOW}] = R_t \cdot (C_{\text{fraud\_rate}} \cdot \text{Amt} + C_{\text{chargeback\_fee}})$$
   - If fraud passes, merchant suffers full chargeback plus network penalty fees.
   - If legitimate, operational cost is zero.

2. **VERIFY** (e.g., Step-up 3DS / SMS OTP):
   $$\mathbb{E}[\text{Cost} \mid \text{VERIFY}] = R_t \cdot (1 - \eta_v) \cdot (C_{\text{fraud\_rate}} \cdot \text{Amt} + C_{\text{cb}}) + (1 - R_t) \cdot C_{\text{fp\_friction\_v}} + C_{\text{verify\_fixed}}$$
   - Reduces fraud loss by efficacy $\eta_v$ (default: 70% fraud intercepted).
   - Interrupted legitimate users experience cart abandonment / friction cost ($C_{\text{fp\_friction\_v}}$).
   - Fixed infrastructure cost per verification ($C_{\text{verify\_fixed}} \approx ₹2$).

3. **THROTTLE** (Velocity / Limit restrictions):
   $$\mathbb{E}[\text{Cost} \mid \text{THROTTLE}] = R_t \cdot (1 - \eta_t) \cdot (C_{\text{fraud\_rate}} \cdot \text{Amt} + C_{\text{cb}}) + (1 - R_t) \cdot C_{\text{fp\_friction\_t}} + C_{\text{throttle\_fixed}}$$
   - Partially dampens fraud loss ($\eta_t \approx 30\%$) with lower customer friction than full step-up verification.

4. **BLOCK** (Hard Decline):
   $$\mathbb{E}[\text{Cost} \mid \text{BLOCK}] = (1 - R_t) \cdot C_{\text{fp\_block}}$$
   - Fraud blocked: ₹0 chargeback loss.
   - Legitimate customer falsely declined: severe cost from lost commission, lifetime customer churn, and brand damage ($C_{\text{fp\_block}}$).

### Pre-Defined Merchant Cost Scenarios

| Parameter | Conservative (Low Friction) | Balanced (E-Commerce) | Aggressive (High Value / FinTech) |
|:---|:---:|:---:|:---:|
| **$C_{\text{fraud\_rate}}$** | 1.0 | 1.0 | 1.0 |
| **$C_{\text{chargeback\_fee}}$** | ₹1,000 | ₹1,000 | ₹2,000 |
| **Verify Efficacy ($\eta_v$)** | 70% | 70% | 80% |
| **Throttle Efficacy ($\eta_t$)** | 30% | 30% | 40% |
| **Legit Verify Friction** | ₹400 | ₹150 | ₹75 |
| **Legit Throttle Friction**| ₹150 | ₹50 | ₹25 |
| **Legit Block Cost** | ₹2,000 | ₹500 | ₹200 |
| **Fixed Verify Cost** | ₹2 | ₹2 | ₹2 |
| **Avg Transaction Amt** | ₹8,000 | ₹3,500 | ₹2,000 |

---

## 6. Explainable Output Contract

Every decision produces a structured payload with mathematical attribution:

```json
{
  "transaction_id": 3488979,
  "base_risk": 0.4125,
  "graph_risk": 0.7310,
  "final_risk": 0.5843,
  "action": "VERIFY",
  "expected_cost": 312.45,
  "action_costs": {
    "allow": 1870.0,
    "verify": 312.45,
    "throttle": 1320.5,
    "block": 500.0
  },
  "risk_contributors": [
    "Elevated ML fraud probability (A_t = 0.4125)",
    "High graph risk (G_t = 0.7310): entity/device has confirmed fraud history",
    "Graph contextual uplift: +0.1718 (formula: F2_conditional)",
    "Final risk R_t = 0.5843 [MEDIUM RISK]"
  ],
  "formula": "F2_conditional",
  "graph_contribution": 0.1718,
  "scenario_name": "balanced"
}
```

This structured output feeds directly into the Phase 4 GraphRAG investigator without requiring any LLM to synthesize numerical facts.

---

## 7. Final Frozen Parameters (Validation Selected)

All parameters were selected on VALIDATION ($N = 88,581$) and frozen into `artifacts/risk/frozen_params.json` prior to TEST evaluation:

| Component | Selected Method / Parameter | Rationale |
|:---|:---:|:---|
| **$A_t$ Calibration** | `none` | $A_t$ is already well-calibrated (ECE = 0.0102). Platt calibration degraded ECE to 0.0175. |
| **$G_t$ Calibration** | `platt` (logistic) | Graph risk had high raw ECE (0.1435). Platt scaling reduced ECE to **0.0078** ($\Delta = +0.1357$). |
| **Fusion Formula** | **F2 (Conditional)** | Highest VALIDATION PR-AUC (0.6689 vs 0.6688 for F1 and 0.6269 for F4). |
| **Selected $\beta$** | **0.05** | Bounded, conservative graph uplift preventing false-positive explosion. |
| **Optimal Decision $\tau^*$** | **0.1244** | F1-maximising threshold on VALIDATION ($F_1 = 0.6699$). |

---

## 8. TEST Evaluation Results ($N = 88,580$, $3,083$ Frauds)

Evaluated once on the untouched held-out TEST partition after freezing all parameters:

| Configuration | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | FPR | TP | FP | FN | TN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$A_t$ Baseline** (XGBoost) | 0.8892 | 0.5658 | 73.08% | 47.45% | 0.5754 | 0.63% | 1,463 | 539 | 1,620 | 84,958 |
| **$G_t$ Baseline** (Graph Only) | 0.7480 | 0.1101 | 12.78% | 39.05% | 0.1926 | 9.61% | 1,204 | 8,216 | 1,879 | 77,281 |
| **F1 Additive** ($\alpha = 0.05$) | 0.8909 | 0.5667 | 72.34% | 47.68% | 0.5748 | 0.66% | 1,470 | 562 | 1,613 | 84,935 |
| **F2 Conditional** ($\beta = 0.05$) | **0.8909** | **0.5667** | **72.77%** | **47.58%** | **0.5754** | **0.64%** | **1,467** | **549** | **1,616** | **84,948** |
| **F4 Conservative Max** ($c = 0.5$) | 0.8688 | 0.5262 | 79.65% | 44.05% | 0.5673 | 0.41% | 1,358 | 347 | 1,725 | 85,150 |
| **Final System ($R_t$)** | **0.8909** | **0.5667** | **72.77%** | **47.58%** | **0.5754** | **0.64%** | **1,467** | **549** | **1,616** | **84,948** |

### Key Observations
1. **F2 Conditional Fusion outperforms $A_t$ alone**:
   - Captures **+4 additional confirmed frauds** (TP: 1,463 $\to$ 1,467) with only +10 false positives.
   - PR-AUC improves from 0.5658 to **0.5667**; ROC-AUC improves from 0.8892 to **0.8909**.
2. **Conservative weighting avoids FP explosion**:
   - Unlike naive equal weighting ($0.70 A_t + 0.30 G_t$) which produced 24,000+ FPs in Phase 2, the validation-tuned conditional formulation maintains sub-1% FPR (0.64%).
3. **Complementary error correction**:
   - $G_t$ recalibrates borderline cases where XGBoost assigns $A_t \approx 0.10 - 0.20$ but graph recidivism is high.

---

## 9. Business Loss Evaluation (Cost Scenarios)

Total financial loss evaluated across all $N = 88,580$ test transactions under predefined merchant cost profiles:

$$\text{Total Loss} = \text{Fraud Loss (Allowed + Residual)} + \text{False Positive Friction Loss} + \text{Operational Verification Cost}$$

| Metric | Conservative | Balanced (Default) | Aggressive | $A_t$-Only (Balanced) |
|:---|:---:|:---:|:---:|:---:|
| **Total Expected Loss** | **₹2,673,586** | **₹2,295,621** | **₹3,105,446** | **₹2,304,525** |
| **Fraud Loss (Total)** | ₹2,606,754 | ₹2,210,835 | ₹2,670,555 | ₹2,223,497 |
| - *From Allowed Fraud* | ₹2,319,844 | ₹1,953,205 | ₹1,825,721 | ₹1,968,148 |
| - *Residual from Verify/Throttle*| ₹286,910 | ₹257,630 | ₹844,834 | ₹255,350 |
| **False-Positive Loss** | ₹65,250 | ₹83,250 | ₹429,325 | ₹79,550 |
| - *From False Blocks* | ₹22,000 | ₹32,500 | ₹44,000 | ₹32,000 |
| - *From Verify/Throttle Friction*| ₹43,250 | ₹50,750 | ₹385,325 | ₹47,550 |
| **Operational Costs (SMS OTP)**| ₹1,582 | ₹1,536 | ₹5,566 | ₹1,478 |
| **Action Distribution** | | | | |
| - `ALLOW` | 87,326 | 86,563 | 75,201 | 86,612 |
| - `VERIFY` | 791 | 768 | 2,783 | 739 |
| - `THROTTLE` | 102 | 360 | 9,120 | 344 |
| - `BLOCK` | 361 | 889 | 1,476 | 885 |
| **Frauds Blocked** | 350 | 824 | 1,256 | 821 |
| **Legitimate Blocked** | 11 | 65 | 220 | 64 |

### Financial Impact vs. $A_t$ Baseline
In the **Balanced Scenario**, the integrated mathematical risk engine saves **₹8,904 in net loss** over $A_t$ alone on the 88k test partition:
- Fraud loss is reduced by **₹12,662** due to +3 more frauds blocked and +29 more frauds routed to step-up verification.
- FP friction increases by only ₹3,700, yielding net-positive ROI.

---

## 10. Limitations & Design Considerations

1. **Transaction Amount Sensitivity**:
   - The cost engine scales fraud penalty directly with `TransactionAmt`. Small frauds (e.g. ₹50) with high risk might be routed to `THROTTLE` rather than `BLOCK` if $C_{\text{fp\_block}} = ₹500 > ₹50 + ₹1000$. Merchants can tune $C_{\text{chargeback\_fee}}$ to adjust minimum decline penalties.
2. **Fixed OTP Efficacy Assumption**:
   - We assumed $\eta_v = 70\%$ verification fraud reduction. In practice, automated bots bypassing OTP (SIM swap) may achieve lower efficacy.
3. **Platt Scaling of $G_t$**:
   - Platt scaling maps $G_t \in [0, 1]$ into a compressed probability space ($\max G_t^{\text{cal}} \approx 0.35$). This is why $\beta = 0.05$ creates subtle, disciplined uplifts rather than dominating $A_t$.

