import pandas as pd
import numpy as np
import json

df_pol = pd.read_csv('results/policy_predictions.csv')
df_fus = pd.read_csv('results/fusion_predictions.csv')

y = df_pol['isFraud'].values
A = df_fus['A_t'].values
R = df_fus['R_t'].values
actions = df_pol['action'].values

N = len(y)
frauds = int(np.sum(y == 1))
legit = int(np.sum(y == 0))
base_rate = frauds / N

print(f"Independent Recompute:")
print(f"  N = {N:,}, Frauds = {frauds:,}, Legit = {legit:,}, Base Rate = {base_rate:.6f}")

# B0
b0_pred = (A >= 0.594298).astype(int)
tp0 = int(np.sum((y == 1) & (b0_pred == 1)))
fp0 = int(np.sum((y == 0) & (b0_pred == 1)))
fn0 = int(np.sum((y == 1) & (b0_pred == 0)))
tn0 = int(np.sum((y == 0) & (b0_pred == 0)))
prec0 = tp0 / (tp0 + fp0)
rec0 = tp0 / frauds
f1_0 = 2 * prec0 * rec0 / (prec0 + rec0)
fpr0 = fp0 / legit

print(f"B0: TP={tp0}, FP={fp0}, FN={fn0}, TN={tn0}, Prec={prec0:.6f}, Rec={rec0:.6f}, F1={f1_0:.6f}, FPR={fpr0:.6f}")

# B3
b3_pred = (R >= 0.594298).astype(int)
tp3 = int(np.sum((y == 1) & (b3_pred == 1)))
fp3 = int(np.sum((y == 0) & (b3_pred == 1)))
fn3 = int(np.sum((y == 1) & (b3_pred == 0)))
tn3 = int(np.sum((y == 0) & (b3_pred == 0)))
prec3 = tp3 / (tp3 + fp3)
rec3 = tp3 / frauds
f1_3 = 2 * prec3 * rec3 / (prec3 + rec3)
fpr3 = fp3 / legit

print(f"B3: TP={tp3}, FP={fp3}, FN={fn3}, TN={tn3}, Prec={prec3:.6f}, Rec={rec3:.6f}, F1={f1_3:.6f}, FPR={fpr3:.6f}")

# Tier 1 (R >= 0.60)
t1_pred = (R >= 0.60).astype(int)
tp1 = int(np.sum((y == 1) & (t1_pred == 1)))
fp1 = int(np.sum((y == 0) & (t1_pred == 1)))
fn1 = int(np.sum((y == 1) & (t1_pred == 0)))
tn1 = int(np.sum((y == 0) & (t1_pred == 0)))
prec1 = tp1 / (tp1 + fp1)
rec1 = tp1 / frauds
f1_1 = 2 * prec1 * rec1 / (prec1 + rec1)
fpr1 = fp1 / legit

print(f"Tier 1: TP={tp1}, FP={fp1}, FN={fn1}, TN={tn1}, Prec={prec1:.6f}, Rec={rec1:.6f}, F1={f1_1:.6f}, FPR={fpr1:.6f}")

# Tier 3 (BLOCK, R >= 0.80)
tp_blk = int(np.sum((y == 1) & (actions == 'BLOCK')))
fp_blk = int(np.sum((y == 0) & (actions == 'BLOCK')))
fn_blk = int(np.sum((y == 1) & (actions != 'BLOCK')))
tn_blk = int(np.sum((y == 0) & (actions != 'BLOCK')))
prec_blk = tp_blk / (tp_blk + fp_blk)
rec_blk = tp_blk / frauds
f1_blk = 2 * prec_blk * rec_blk / (prec_blk + rec_blk)
fpr_blk = fp_blk / legit
enrich_blk = (tp_blk / (tp_blk + fp_blk)) / base_rate

print(f"BLOCK: TP={tp_blk}, FP={fp_blk}, FN={fn_blk}, TN={tn_blk}, Prec={prec_blk:.6f}, Rec={rec_blk:.6f}, F1={f1_blk:.6f}, FPR={fpr_blk:.6f}, Enrich={enrich_blk:.2f}x")

# Check against manifest
with open('artifacts/final_evaluation/evaluation_manifest.json') as f:
    man = json.load(f)

assert tp0 == man['headline_metrics']['B0_baseline']['tp']
assert fp0 == man['headline_metrics']['B0_baseline']['fp']
assert tp3 == man['headline_metrics']['B3_conditional_fusion']['tp']
assert fp3 == man['headline_metrics']['B3_conditional_fusion']['fp']
assert tp1 == man['headline_metrics']['progressive_tier_1_verify_plus']['tp']
assert fp1 == man['headline_metrics']['progressive_tier_1_verify_plus']['fp']
assert tp_blk == man['headline_metrics']['progressive_tier_3_block']['tp']
assert fp_blk == man['headline_metrics']['progressive_tier_3_block']['fp']

print("\nALL INDEPENDENT RECOMPUTATIONS MATCH MANIFEST EXACTLY (0 DISCREPANCY)!")
