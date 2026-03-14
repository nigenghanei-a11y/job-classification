# ============================================
# SCRIPT 3: COMBINE_RESULTS.py
# Run this AFTER all 13 workers complete
# ============================================

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.stats.proportion import proportions_ztest

# ============================================
# CONFIGURATION
# ============================================
NUM_CHUNKS = 13
INPUT_DIR = Path("../data/")
OUTPUT_FILE = "../data/mistral_zero_shot_predictions_combined.csv"
SUMMARY_FILE = "../data/mistral_ztest_summary_combined.json"

# ============================================
# LOAD ALL CHUNKS
# ============================================
print("Loading all chunks...")
all_chunks = []

for i in range(1, NUM_CHUNKS + 1):
    chunk_file = INPUT_DIR / f"results_chunk_{i:02d}.csv"
    if chunk_file.exists():
        chunk = pd.read_csv(chunk_file)
        all_chunks.append(chunk)
        print(f"  Loaded chunk {i}: {len(chunk)} rows")
    else:
        print(f"  ⚠️ Missing chunk {i}: {chunk_file}")

if len(all_chunks) == 0:
    raise FileNotFoundError("No result chunks found!")

# ============================================
# COMBINE ALL CHUNKS
# ============================================
df_pred = pd.concat(all_chunks).reset_index(drop=True)
df_pred.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Combined {len(df_pred)} rows")
print(f"Saved: {OUTPUT_FILE}")

# ============================================
# EVALUATION
# ============================================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

df_eval = df_pred[df_pred["predicted_level"].isin(["junior", "mid", "senior"])].copy()

print(f"Total rows: {len(df_pred)}")
print(f"Valid predicted rows: {len(df_eval)}")
print(f"Unknown/invalid predictions removed: {len(df_pred) - len(df_eval)}")

if len(df_eval) > 0:
    acc = accuracy_score(df_eval["experience_level"], df_eval["predicted_level"])
    print(f"\nAccuracy: {round(acc, 4)}")
    
    print("\nClassification Report:")
    print(classification_report(df_eval["experience_level"], df_eval["predicted_level"], digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(df_eval["experience_level"], df_eval["predicted_level"]))
    
    # ============================================
    # Z-TEST
    # ============================================
    observed_correct = (df_eval["experience_level"] == df_eval["predicted_level"]).sum()
    n = len(df_eval)
    p0 = 0.80
    
    stat, p_value = proportions_ztest(
        count=observed_correct,
        nobs=n,
        value=p0,
        alternative="larger"
    )
    
    observed_accuracy = observed_correct / n
    
    print(f"\nZ-Test Results:")
    print(f"  Observed correct: {observed_correct}")
    print(f"  Total evaluated: {n}")
    print(f"  Observed accuracy: {round(observed_accuracy, 4)}")
    print(f"  Hypothesized accuracy: {p0}")
    print(f"  Z-statistic: {round(stat, 4)}")
    print(f"  P-value: {p_value}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"\n✅ Reject H0: The model accuracy is significantly greater than {p0}")
    else:
        print(f"\n❌ Fail to reject H0: Not enough evidence that accuracy > {p0}")
    
    # ============================================
    # SAVE SUMMARY
    # ============================================
    summary = {
        "total_rows_used": int(len(df_pred)),
        "total_rows_evaluated": int(len(df_eval)),
        "observed_correct": int(observed_correct),
        "observed_accuracy": float(observed_accuracy),
        "hypothesized_accuracy": float(p0),
        "z_statistic": float(stat),
        "p_value": float(p_value),
        "num_chunks": NUM_CHUNKS,
        "chunk_sizes": [len(c) for c in all_chunks]
    }
    
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Saved summary: {SUMMARY_FILE}")
else:
    print("\n⚠️ No valid predictions to evaluate!")
