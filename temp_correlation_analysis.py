import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("outputs/features_directional.csv", index_col="Date", parse_dates=True)

exclude = ["raw_open", "raw_high", "raw_low", "raw_close", "raw_daily_return", "raw_log_return", "target_direction"]
features = [c for c in df.columns if c not in exclude]
X = df[features].dropna()
y = df.loc[X.index, "target_direction"]

print(f"Samples: {len(X)}, Target balance: {y.mean():.3f} (1=UP), Features: {len(features)}\n")

# Pearson (point-biserial) correlation
pearson = X.corrwith(y).sort_values(ascending=False)

# Spearman rank correlation
spearman = pd.Series(
    {col: X[col].corr(y, method="spearman") for col in features}
).sort_values(ascending=False)

# Mutual information
mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
mi_series = pd.Series(mi, index=features).sort_values(ascending=False)

# Combined report
report = pd.DataFrame({
    "pearson_r": pearson,
    "spearman_r": spearman.reindex(pearson.index),
    "mi": mi_series.reindex(pearson.index),
}).sort_values("mi", ascending=False)

print("=" * 85)
print(f"{'Feature':<35} {'Pearson r':>10} {'Spearman r':>11} {'Mutual Info':>12}")
print("=" * 85)
for feat, row in report.iterrows():
    print(f"{feat:<35} {row['pearson_r']:>10.4f} {row['spearman_r']:>11.4f} {row['mi']:>12.4f}")

print("\n--- Top 15 by Mutual Information ---")
print(report.head(15).to_string())

print("\n--- Top 15 by |Pearson| ---")
print(report.reindex(report["pearson_r"].abs().sort_values(ascending=False).index).head(15).to_string())

print("\n--- Top 15 by |Spearman| ---")
print(report.reindex(report["spearman_r"].abs().sort_values(ascending=False).index).head(15).to_string())

report.to_csv("outputs/correlation_analysis.csv")
print("\nSaved full report to outputs/correlation_analysis.csv")

# ==================== DISTRIBUTION DRIFT CHECK ====================
print("\n\n" + "=" * 85)
print("DISTRIBUTION DRIFT CHECK (Early 2008-2010 vs Late 2023-2025)")
print("=" * 85)

df_w = df.copy()
df_w["year"] = df_w.index.year

early_mask = df_w["year"] <= 2010
late_mask = df_w["year"] >= 2023

drift_results = []
for col in features:
    early = df_w.loc[early_mask, col].dropna()
    late = df_w.loc[late_mask, col].dropna()
    if len(early) < 10 or len(late) < 10:
        continue

    e_min, e_max = early.min(), early.max()
    l_min, l_max = late.min(), late.max()

    pct_outside = ((late < e_min) | (late > e_max)).mean() * 100
    
    e_mean, e_std = early.mean(), early.std() + 1e-10
    l_mean, l_std = late.mean(), late.std() + 1e-10
    
    mean_shift = abs(l_mean - e_mean) / e_std
    std_ratio = l_std / e_std if e_std > 1e-10 else 0
    
    drift_results.append({
        "feature": col,
        "pct_oob": pct_outside,
        "mean_shift_sigma": mean_shift,
        "std_ratio": std_ratio,
        "early_mean": e_mean,
        "late_mean": l_mean,
    })

drift_df = pd.DataFrame(drift_results).sort_values("pct_oob", ascending=False)

print(f"\n{'Feature':<35} {'% Late OOB':>10} {'Mean Shift(σ)':>14} {'Std Ratio':>10}")
print("-" * 75)
for _, row in drift_df.iterrows():
    flag = " ***" if row["pct_oob"] > 10 or row["mean_shift_sigma"] > 1.0 else ""
    print(f"{row['feature']:<35} {row['pct_oob']:>9.1f}% {row['mean_shift_sigma']:>14.2f} {row['std_ratio']:>10.2f}{flag}")

any_flagged = drift_df[(drift_df["pct_oob"] > 10) | (drift_df["mean_shift_sigma"] > 1.0)]
if len(any_flagged) == 0:
    print("\n✓ No significant distribution drift detected — all features stationary!")
else:
    print(f"\n✗ {len(any_flagged)} features still have distribution drift issues")