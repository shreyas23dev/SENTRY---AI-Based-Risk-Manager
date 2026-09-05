import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix
)

# Load test risk scores
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = PROJECT_ROOT / "artifacts" / "risk" / "test_risk_scores.parquet"

def evaluate_reconciliation():
    print("=" * 80)
    print("RECONCILING EVALUATION METRICS ON HELD-OUT TEST PARTITION (N=88,580)")
    print("=" * 80)

    if not SCORES_PATH.exists():
        print(f"ERROR: {SCORES_PATH} not found!")
        return

    df = pd.read_parquet(SCORES_PATH)
    print(f"Loaded {len(df)} rows from {SCORES_PATH}")
    print(f"Columns: {list(df.columns)}")

    y_true = df["isFraud"].values.astype(int)
    a_t = df["A_t"].values.astype(float)
    g_t = df["G_t"].values.astype(float)
    r_t = df["R_t"].values.astype(float)

    n_total = len(y_true)
    n_fraud = int(np.sum(y_true))
    n_legit = n_total - n_fraud
    print(f"Total TEST Samples: {n_total:,} | Fraud: {n_fraud:,} ({n_fraud/n_total*100:.2f}%) | Legit: {n_legit:,} ({n_legit/n_total*100:.2f}%)")

    # 1. Base ML A_t Evaluation
    roc_auc_a = roc_auc_score(y_true, a_t)
    pr_auc_a = average_precision_score(y_true, a_t)

    # Threshold on A_t using optimal F1
    prec_a, rec_a, thresh_a = precision_recall_curve(y_true, a_t)
    f1_curve_a = np.where((prec_a + rec_a) == 0, 0, 2 * prec_a * rec_a / (prec_a + rec_a))
    best_idx_a = int(np.argmax(f1_curve_a))
    t_opt_a = float(thresh_a[min(best_idx_a, len(thresh_a) - 1)])

    y_pred_a = (a_t >= t_opt_a).astype(int)
    tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_true, y_pred_a).ravel()
    prec_val_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0
    rec_val_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0
    f1_val_a = 2 * prec_val_a * rec_val_a / (prec_val_a + rec_val_a) if (prec_val_a + rec_val_a) > 0 else 0
    fpr_val_a = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0

    # 2. Fused R_t Evaluation
    roc_auc_r = roc_auc_score(y_true, r_t)
    pr_auc_r = average_precision_score(y_true, r_t)

    prec_r, rec_r, thresh_r = precision_recall_curve(y_true, r_t)
    f1_curve_r = np.where((prec_r + rec_r) == 0, 0, 2 * prec_r * rec_r / (prec_r + rec_r))
    best_idx_r = int(np.argmax(f1_curve_r))
    t_opt_r = float(thresh_r[min(best_idx_r, len(thresh_r) - 1)])

    y_pred_r = (r_t >= t_opt_r).astype(int)
    tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_true, y_pred_r).ravel()
    prec_val_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0
    rec_val_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0
    f1_val_r = 2 * prec_val_r * rec_val_r / (prec_val_r + rec_val_r) if (prec_val_r + rec_val_r) > 0 else 0
    fpr_val_r = fp_r / (fp_r + tn_r) if (fp_r + tn_r) > 0 else 0

    print("\n" + "=" * 80)
    print("APPLES-TO-APPLES COMPARISON (EXACT TEST ROWS):")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Base ML (A_t)':<16} | {'Fused Math (R_t)':<16} | {'Abs. Improvement':<16}")
    print("-" * 75)
    print(f"{'ROC-AUC':<20} | {roc_auc_a:<16.4f} | {roc_auc_r:<16.4f} | {roc_auc_r - roc_auc_a:+16.4f}")
    print(f"{'PR-AUC':<20} | {pr_auc_a:<16.4f} | {pr_auc_r:<16.4f} | {pr_auc_r - pr_auc_a:+16.4f}")
    print(f"{'F1 Score':<20} | {f1_val_a:<16.4f} | {f1_val_r:<16.4f} | {f1_val_r - f1_val_a:+16.4f}")
    print(f"{'Precision':<20} | {prec_val_a*100:<15.2f}% | {prec_val_r*100:<15.2f}% | {(prec_val_r - prec_val_a)*100:+15.2f}%")
    print(f"{'Recall (Capture)':<20} | {rec_val_a*100:<15.2f}% | {rec_val_r*100:<15.2f}% | {(rec_val_r - rec_val_a)*100:+15.2f}%")
    print(f"{'FPR (Legit Impact)':<20} | {fpr_val_a*100:<15.2f}% | {fpr_val_r*100:<15.2f}% | {(fpr_val_r - fpr_val_a)*100:+15.2f}%")
    print(f"{'True Positives (TP)':<20} | {tp_a:<16,d} | {tp_r:<16,d} | {tp_r - tp_a:+16,d}")
    print(f"{'False Positives (FP)':<20} | {fp_a:<16,d} | {fp_r:<16,d} | {fp_r - fp_a:+16,d}")
    print(f"{'False Negatives (FN)':<20} | {fn_a:<16,d} | {fn_r:<16,d} | {fn_r - fn_a:+16,d}")
    print(f"{'True Negatives (TN)':<20} | {tn_a:<16,d} | {tn_r:<16,d} | {tn_r - tn_a:+16,d}")
    print(f"{'Opt Threshold':<20} | {t_opt_a:<16.4f} | {t_opt_r:<16.4f} | {t_opt_r - t_opt_a:+16.4f}")
    print("=" * 80)

if __name__ == '__main__':
    evaluate_reconciliation()
