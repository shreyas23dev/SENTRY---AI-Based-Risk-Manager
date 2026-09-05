import pandas as pd
import numpy as np

df = pd.read_csv('results/policy_predictions.csv')
y = df['isFraud'].values
R = df['R_t'].values
n_total = len(y)
total_frauds = int(np.sum(y == 1))
total_legit = int(np.sum(y == 0))
base_fraud_rate = total_frauds / n_total

actions = [
    ('ALLOW', (R < 0.60), "[0.00, 0.60)"),
    ('VERIFY', (R >= 0.60) & (R < 0.65), "[0.60, 0.65)"),
    ('THROTTLE', (R >= 0.65) & (R < 0.80), "[0.65, 0.80)"),
    ('BLOCK', (R >= 0.80), "[0.80, 1.00]"),
]

tot_txns = 0
tot_f = 0
tot_l = 0

print(f"{'Action':8s} | {'Score Range':14s} | {'Txns':6s} | {'Frauds':6s} | {'Legit':6s} | {'Fraud Rate':10s} | {'Enrichment':10s} | {'Legit FPs':9s}")
print('-' * 85)
for name, m, rng in actions:
    cnt = int(np.sum(m))
    f = int(np.sum((y == 1) & m))
    l = int(np.sum((y == 0) & m))
    rate = f / cnt if cnt > 0 else 0.0
    enrichment = rate / base_fraud_rate
    fps = l if name != 'ALLOW' else 0
    tot_txns += cnt
    tot_f += f
    tot_l += l
    print(f"{name:8s} | {rng:14s} | {cnt:6d} | {f:6d} | {l:6d} | {rate:9.4%} | {enrichment:9.2f}x | {fps:9d}")

print('-' * 85)
print(f"{'TOTALS':8s} | {'[0.00, 1.00]':14s} | {tot_txns:6d} | {tot_f:6d} | {tot_l:6d}")

assert tot_txns == 88580
assert tot_f == 3083
assert tot_l == 85497
print("ALL SUMS ASSERTIONS PASSED!")
