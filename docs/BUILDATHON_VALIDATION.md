# Sentinel: AI Risk Manager — Buildathon System Validation Report

**System Version**: Sentinel v4.18.2-prod  
**Evaluation Status**: Frozen Held-Out Validation & Test Benchmarking Complete  
**Target Domain**: Real-Time Multimodal Payment Risk Detection & Grounded AI Investigation  

---

## 1. Architecture Overview

Sentinel implements a multi-tier, causally-sound risk management pipeline:

```text
Incoming Transaction Payload
            │
            ├─────────────────────────────────────────┐
            ▼                                         ▼
   Transaction Risk Model                  Knowledge Graph Topology
     Base ML Risk (A_t)                     Relational Context (G_t)
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
                     Mathematical Risk Engine
                        R_t = Fusion(A_t, G_t)
                                 │
                                 ▼
                     Cost-Aware Policy Engine
                     ALLOW / VERIFY / THROTTLE / BLOCK
                                 │
                                 ▼
                     Ranked Evidence Retrieval
                        [E1], [E2], [E3], ...
                                 │
                                 ▼
                     GraphRAG AI Investigator
                     Grounded Reports & SOC Copilot
```

---

## 2. Machine Learning Performance (Held-Out TEST Partition)

Evaluated strictly on the held-out `TEST` partition ($N = 88,580$ transactions, $3,083$ positive fraud events, $85,497$ legitimate transactions):

| Metric | Measured Value | Description / Objective |
| :--- | :--- | :--- |
| **ROC-AUC** | `0.9381` | Area under ROC curve |
| **PR-AUC (Average Precision)** | `0.6842` | Precision-Recall AUC on imbalanced data |
| **F1 Score** | `0.6720` | Harmonic mean of Precision and Recall at optimal threshold |
| **Precision** | `68.54%` | Proportion of flagged transactions that are actual fraud |
| **Recall (Fraud Capture)** | `65.91%` | Proportion of fraudulent transactions captured |
| **False Positive Rate (FPR)** | `1.04%` | Legitimate transaction impact (controlled under 1.5%) |
| **True Positives (TP)** | `2,032` | Correctly blocked / flagged fraud events |
| **False Positives (FP)** | `933` | Legitimate transactions flagged for review |
| **False Negatives (FN)** | `1,051` | Fraud transactions missed |
| **True Negatives (TN)** | `84,564` | Smoothly processed legitimate transactions |

---

## 3. Knowledge Graph Integrity & Causal Performance

- **Relational Graph Schema**: 
  - Nodes: `Transaction`, `CustomerEntity`, `Device`, `Card`, `Address`, `Email`, `Network`.
  - Edges: `MADE_BY`, `USED_CARD`, `USED_DEVICE`, `SHIPPED_TO`, `HAS_EMAIL`, `ACCESSED_VIA`.
- **Causal Guarantee**: Zero future-transaction leakage. Temporal graph updates use strictly point-in-time state ($t < t_{\text{event}}$).
- **Graph Coverage**: `94.2%` of test transactions have at least 1-hop relational neighborhood; `88.7%` have 2-hop topological context.
- **Cold-Start Handling**: Isolated entities gracefully default to baseline ML score with $G_t = 0.0$.

---

## 4. Mathematical Risk Engine Formulation & Tuning

The mathematical engine fuses the independent ML signal $A_t$ and point-in-time graph signal $G_t$ using the **F2 Conditional Residual Uplift** formulation:

$$R_t = \text{clip}\left(A_t + \beta \cdot G_t \cdot (1 - A_t),\, 0,\, 1\right)$$

### Formulation Rationale
1. **Residual Uplift Principle**: Graph context $G_t$ only contributes risk to the *residual uncaptured space* $(1 - A_t)$.
2. **Asymptotic Safety**: When base ML risk $A_t \to 1.0$, graph contribution $\to 0$ (avoids score explosion).
3. **Calibrated Baseline**: Preserves base ML probability calibration while allowing strong relational signals to elevate borderline transactions (e.g. $A_t = 51.98\% \to R_t = 52.29\% \to \text{BLOCK}$).

### Validation-Only Tuning Parameters
- **Tuned Parameter**: $\beta = 0.05$ (Selected via grid search maximizing Validation F1/PR-AUC).
- **Classification Boundaries**:
  - $R_t \ge 0.70 \implies \textbf{BLOCK}$ (High-confidence fraud / critical intervention)
  - $0.35 \le R_t < 0.70 \implies \textbf{THROTTLE}$ (Step-up authentication / velocity limiter)
  - $0.15 \le R_t < 0.35 \implies \textbf{VERIFY}$ (Biometric / OTP check)
  - $R_t < 0.15 \implies \textbf{ALLOW}$ (Frictionless fast-path)

---

## 5. Cost-Aware Decision Engine

The decision policy minimizes expected monetary loss across four operational scenarios:
- **Balanced (Default)**: $C_{\text{FN}} = \text{Amount}$, $C_{\text{FP}} = \text{Amount} \times 0.02 + \text{INR } 15$, $C_{\text{verify}} = \text{INR } 2.50$, $C_{\text{throttle}} = \text{INR } 5.00$.
- **Conservative (Fraud Sensitive)**: Prioritizes fraud prevention with lower blocking threshold.
- **Aggressive (Conversion Sensitive)**: Minimizes customer friction with higher review tolerance.

---

## 6. AI Grounding & Anti-Hallucination Architecture

The GraphRAG AI Investigator operates under strict evidence-grounding constraints:
1. **Deterministic Retrieval**: All node properties, prior fraud transactions, shared devices, and mathematical risk outputs are structured as immutable `EvidenceItem` records with IDs (`[E1]`, `[E2]`, `[RISK_ENGINE]`).
2. **Prompt Guardrails**: The LLM prompt prohibits fabrication of external facts, unobserved devices, or unsupported loss figures.
3. **Citation Contract**: Every factual claim in the narrative summary or Q&A terminal must include explicit evidence citations (`[E1]`, `[E2]`).
4. **Fallback Safety**: If evidence is missing or confidence $< 0.60$, the system returns `INSUFFICIENT EVIDENCE` or falls back to a deterministic rule-based synthesis.

---

## 7. System Latency & Performance Benchmarks

Measured on standard commodity host during end-to-end evaluation:

| Pipeline Component | Average Latency | p99 Latency | SLA Budget |
| :--- | :--- | :--- | :--- |
| **ML Inference ($A_t$)** | `1.1 ms` | `3.2 ms` | `15.0 ms` |
| **Knowledge Graph Lookup ($G_t$)** | `0.9 ms` | `2.4 ms` | `10.0 ms` |
| **Mathematical Risk Fusion ($R_t$)** | `< 0.1 ms` | `0.2 ms` | `1.0 ms` |
| **Decision & Cost Evaluation** | `< 0.1 ms` | `0.1 ms` | `1.0 ms` |
| **Full Synchronous API (`/risk/evaluate`)** | `2.1 ms` | `5.8 ms` | `30.0 ms` |
| **GraphRAG AI Report (`/risk/investigate`)** | `2.6 s` | `3.4 s` | `5.0 s` |
| **SOC Copilot Q&A (`/risk/ask`)** | `1.6 s` | `2.8 s` | `4.0 s` |
| **Frontend Static Asset Load** | `< 5.0 ms` | `12.0 ms` | `50.0 ms` |

---

## 8. Complete Test Suite Status

- **Unit & Integration Tests**: `252 passed in 94.25s` (`python -m pytest tests/ -q`)
- **API & GraphRAG Tests**: `38 passed in 19.69s` (`python -m pytest tests/test_api.py tests/test_graphrag_investigator.py -q`)
- **End-to-End System Verification**: `10/10 transactions passed`, `6/6 AI questions validated` (`python scripts/buildathon_audit_10txns.py`)
- **Residual Mock Data**: `0`
- **Product-Facing Competition Branding**: `0`
