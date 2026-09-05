# TRUSTGRAPH: Final Empirical Evaluation & System Performance Report
**Evaluation Manifest:** `artifacts/final_evaluation/evaluation_manifest.json`  
**Dataset & Partition:** IEEE-CIS Fraud Detection — Held-Out Chronological TEST  
**Partition Timestamp Range:** `TransactionDT` $\in [13,151,945, 15,811,131]$  
**Evaluation Scope:** Final Frozen System (B0, B1, B2, B3 & Progressive Risk Policy)  
**Population Size:** $N = 88,580$ ($3,083$ frauds, $85,497$ legitimate transactions, base fraud prevalence $= 3.4805\%$)  

---

## 1. Evaluation Protocol & Data Integrity

All evaluations presented in this report adhere to strict temporal split protocols to prevent information leakage:
- **Chronological Isolation:** The dataset is partitioned chronologically into **TRAIN** ($N=413,379$, $DT \le 10,438,003$), **VALIDATION** ($N=88,581$, $10,438,017 \le DT \le 13,151,880$), and **TEST** ($N=88,580$, $13,151,945 \le DT \le 15,811,131$).
- **Zero Test Tuning Guarantee:** Model fitting and categorical preprocessor mappings were performed strictly on TRAIN. Operating thresholds (baseline $\tau = 0.594298$, fusion $\tau = 0.594298$, policy $\tau = [0.60, 0.65, 0.80]$) were tuned strictly on VALIDATION. The held-out TEST partition was evaluated untouched only after freezing all models and parameters.
- **Statistical Uncertainty Estimation:** 95% confidence intervals are computed using the **Wilson score interval for binomial proportions**, providing rigorous finite-sample coverage without relying on asymptotic normal approximations.

---

## 2. Table 1: Model Evolution Across Development Phases

Evaluated on held-out TEST ($N=88,580$) at the frozen baseline threshold $\tau = 0.594298$:

| System Architecture | Operating Rule / Threshold | TP | FP | FN | TN | Precision [95% CI] | Recall [95% CI] | $F_1$ Score | FPR [95% CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$B_0$ Baseline LightGBM** | $A_t \ge 0.594298$ | 1,313 | 755 | 1,770 | 84,742 | 63.49% [61.41%, 65.54%] | 42.59% [40.85%, 44.34%] | 0.5098 | 0.883% [0.823%, 0.948%] |
| **$B_1$ Entity Temporal** | $A_t \ge \tau \lor P_t \ge 0.70$ | 1,316 | 771 | 1,767 | 84,726 | 63.06% [60.98%, 65.10%] | 42.69% [40.95%, 44.44%] | 0.5091 | 0.902% [0.841%, 0.967%] |
| **$B_2$ Relational OR** | $A_t \ge \tau \lor G_t \ge 0.60$ | 1,521 | 3,887 | 1,562 | 81,610 | 28.13% [26.94%, 29.34%] | 49.34% [47.57%, 51.11%] | 0.3583 | 4.546% [4.408%, 4.689%] |
| **$B_3$ Conditional Fusion** | $R_t \ge 0.594298$ | **1,346** | **813** | **1,737** | **84,684** | **62.34% [60.29%, 64.36%]** | **43.66% [41.92%, 45.42%]** | **0.5135** | **0.951% [0.888%, 1.018%]** |

*Methodological Note:* B2 illustrates that simple disjunctive OR rules on relational graph signals degrade precision unacceptably ($63.49\% \to 28.13\%$). In contrast, B3 conditional fusion ($R_t = \text{clip}(A_t + 1.0 P_t + 0.05 G_t, 0, 1)$) integrates temporal and graph uplifts non-destructively.

---

## 3. Table 2: Progressive Decision Policy (Cumulative Operational Tiers)

Interventions stratified by operational severity thresholds:

| Operational Tier | Decision Threshold | Intervened Actions Included | TP | FP | FN | TN | Precision [95% CI] | Recall [95% CI] | $F_1$ Score | FPR [95% CI] | Legitimate Customer Friction |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tier 1** | $R_t \ge 0.60$ | VERIFY + THROTTLE + BLOCK | 1,343 | 798 | 1,740 | 84,699 | 62.73% [60.67%, 64.75%] | 43.56% [41.82%, 45.32%] | 0.5142 | 0.933% [0.871%, 1.000%] | 0.933% ($798 / 85,497$) |
| **Tier 2** | $R_t \ge 0.65$ | THROTTLE + BLOCK | 1,274 | 627 | 1,809 | 84,870 | 67.02% [64.88%, 69.10%] | 41.32% [39.60%, 43.07%] | 0.5112 | 0.733% [0.678%, 0.793%] | 0.733% ($627 / 85,497$) |
| **Tier 3** | $R_t \ge 0.80$ | BLOCK only | **1,081** | **295** | 2,002 | 85,202 | **78.56% [76.32%, 80.66%]** | **35.06% [33.40%, 36.77%]** | **0.4849** | **0.345% [0.308%, 0.387%]** | **0.345% ($295 / 85,497$)** |

---

## 4. Table 3: Mutually Exclusive Operational Action Stratification

Every transaction in the stream receives exactly one policy action:

| Action Tier | Score Range | Total Transactions | Fraud Count | Legit Count | Empirical Fraud Rate | Enrichment vs Base ($3.4805\%$) | False Positives | Operational Workflow |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **ALLOW** | $[0.00, 0.60)$ | 86,439 | 1,740 | 84,699 | 2.0130% | $0.58\times$ | 0 | Frictionless authorization |
| **VERIFY** | $[0.60, 0.65)$ | 240 | 69 | 171 | 28.7500% | $8.26\times$ | 171 | Step-up 3DS / OTP challenge |
| **THROTTLE** | $[0.65, 0.80)$ | 525 | 193 | 332 | 36.7619% | $10.56\times$ | 332 | Velocity capping / delayed clearing |
| **BLOCK** | $[0.80, 1.00]$ | 1,376 | 1,081 | 295 | 78.5610% | **$22.57\times$** | 295 | Hard transaction decline |
| **TOTAL** | **$[0.00, 1.00]$** | **88,580** | **3,083** | **85,497** | **3.4805%** | **$1.00\times$** | **798** | — |

*Integrity Check:*
- Transactions: $86,439 + 240 + 525 + 1,376 = 88,580$ (100.0%)
- Frauds: $1,740 + 69 + 193 + 1,081 = 3,083$ (100.0%)
- Legitimate: $84,699 + 171 + 332 + 295 = 85,497$ (100.0%)

---

## 5. Transaction-Level Discrepancy Reconciliation

### A. Mathematical Reconciliation: B3 ($1,346$ TP) vs Progressive Tier 1 ($1,343$ TP)
- $B_3$ evaluates conditional fusion as a single binary classifier at $\tau = 0.594298$.
- Progressive Policy Tier 1 initiates interventions at $\tau_{\text{verify}} = 0.600000$.
- Exactly **18 transactions** lie in the half-open interval $[0.594298, 0.600000)$:
  - **3 Fraudulent transactions** (TransactionIDs: `3385732`, `3407764`, `3467412`)
  - **15 Legitimate transactions**
- Under policy rules ($R_t < 0.60$), these 18 transactions are assigned to **ALLOW**.
- **Reconciliation Identity:**
  $$\text{Tier 1 Intervened Frauds} = \text{B3 Frauds} - 3 = 1,346 - 3 = \mathbf{1,343}$$
  $$\text{Tier 1 False Alarms} = \text{B3 False Positives} - 15 = 813 - 15 = \mathbf{798}$$

### B. Transaction-Level Tracking of 755 Baseline False Positives
Baseline $B_0$ hard-declined **755 legitimate transactions** ($A_t \ge 0.594298, \text{isFraud}=0$). Under the Progressive Policy:
- **14 transactions** ($1.85\%$) are de-escalated to **ALLOW** (clean pass)
- **147 transactions** ($19.47\%$) are diverted to **VERIFY** (step-up challenge rather than rejection)
- **317 transactions** ($41.99\%$) are diverted to **THROTTLE** (velocity pacing rather than rejection)
- **277 transactions** ($36.69\%$) are retained as hard **BLOCK**
- **478 legitimate transactions ($63.31\%$) are successfully diverted away from hard declines.**
- **18 new legitimate transactions** were escalated into BLOCK due to contextual graph/temporal spikes ($277 + 18 = 295$ total BLOCK FPs).
- **Net reduction in catastrophic customer hard declines: $755 - 295 = \mathbf{460}$ transactions ($60.93\%$ reduction).**

---

## 6. Publication Figures

1. **Figure 1 (Risk Distribution):** [`artifacts/final_evaluation/plots/01_risk_action_distribution.png`](file:///c:/Users/Shreyas%20A/Trustgraph/artifacts/final_evaluation/plots/01_risk_action_distribution.png)  
   Demonstrates risk score distribution on a log scale across ALLOW, VERIFY, THROTTLE, and BLOCK with threshold boundaries.
2. **Figure 2 (Fraud Enrichment):** [`artifacts/final_evaluation/plots/02_fraud_enrichment_by_tier.png`](file:///c:/Users/Shreyas%20A/Trustgraph/artifacts/final_evaluation/plots/02_fraud_enrichment_by_tier.png)  
   Shows monotonic surge in empirical fraud concentration from $2.01\%$ (ALLOW) to $78.56\%$ (BLOCK, $22.57\times$ enrichment).
3. **Figure 3 (Precision-Recall Tradeoff):** [`artifacts/final_evaluation/plots/03_progressive_precision_recall_tradeoff.png`](file:///c:/Users/Shreyas%20A/Trustgraph/artifacts/final_evaluation/plots/03_progressive_precision_recall_tradeoff.png)  
   Maps operational frontier across baseline and policy tiers.
4. **Figure 4 (Operational Friction):** [`artifacts/final_evaluation/plots/04_fpr_vs_fraud_capture.png`](file:///c:/Users/Shreyas%20A/Trustgraph/artifacts/final_evaluation/plots/04_fpr_vs_fraud_capture.png)  
   Illustrates dramatic reduction in false alarm exposure down to $0.345\%$ at the BLOCK tier.

---

## 7. Defensible Research Claims

### Model-Level Claim:
> *"Conditional relational fusion ($B_3$) captures **33 additional fraud cases** over the point-wise baseline while maintaining FPR below 1% ($0.9509\%$)."*

### Policy-Level Claim:
> *"The progressive policy concentrates high-risk transactions into increasingly severe interventions, with the BLOCK tier reaching **78.56% precision** and **22.57x fraud enrichment** while hard-blocking only **0.345%** of legitimate transactions."*

### False-Decline Claim:
> *"The progressive policy reduces legitimate hard declines from 755 to 295, a net reduction of **460 false declines (60.93%)**."*

---

## 8. Limitations & Boundary Conditions

1. **Offline Retrospective Evaluation:** Metrics reflect historical replay; live merchant transaction re-routing may induce downstream behavioral adaptation.
2. **Attribute Availability:** Relational graph connectivity relies on `DeviceInfo` availability ($24.4\%$ populated); transactions lacking device metadata fall back to temporal and tabular features alone.
3. **Action Execution Dependency:** The business benefit of VERIFY ($28.75\%$ fraud) assumes step-up challenges successfully authenticate genuine users and deter fraudsters.
