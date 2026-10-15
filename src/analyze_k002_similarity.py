from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed")
METRICS_DIR = Path("results/metrics")
PLOTS_DIR = Path("results/plots")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

print("=" * 65)
print("STAGE 11: K002 FEATURE SIMILARITY ANALYSIS")
print("=" * 65)

train_df = pd.read_csv(
    DATA_DIR / "train.csv",
    low_memory=False
)

val_df = pd.read_csv(
    DATA_DIR / "validation.csv",
    low_memory=False
)

test_df = pd.read_csv(
    DATA_DIR / "test.csv",
    low_memory=False
)

development_df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

k002_df = test_df[
    test_df["bearing_code"] == "K002"
].copy()

if k002_df.empty:
    raise ValueError("K002 not found.")

features = [
    col
    for col in development_df.columns
    if col.startswith("vib_")
]

print("\nFeatures:", len(features))
print("K002 windows:", len(k002_df))


# =====================================================
# STANDARDIZE USING DEVELOPMENT DATA ONLY
# =====================================================

# Standardization converts features with different
# numerical scales into comparable units.
#
# IMPORTANT:
# Fit the scaler ONLY on development data.

scaler = StandardScaler()

scaler.fit(
    development_df[features]
)

development_scaled = scaler.transform(
    development_df[features]
)

k002_scaled = scaler.transform(
    k002_df[features]
)


# =====================================================
# CALCULATE FEATURE AVERAGES
# =====================================================

healthy_mask = (
    development_df["target"] == 0
).to_numpy()

faulty_mask = (
    development_df["target"] == 1
).to_numpy()

healthy_mean = development_scaled[
    healthy_mask
].mean(axis=0)

faulty_mean = development_scaled[
    faulty_mask
].mean(axis=0)

k002_mean = k002_scaled.mean(axis=0)


# =====================================================
# COMPARE FEATURE DISTANCES
# =====================================================

results = []

for index, feature in enumerate(features):

    distance_to_healthy = abs(
        k002_mean[index] - healthy_mean[index]
    )

    distance_to_faulty = abs(
        k002_mean[index] - faulty_mean[index]
    )

    closer_to = (
        "Healthy"
        if distance_to_healthy < distance_to_faulty
        else "Faulty"
    )

    results.append({
        "feature": feature,
        "k002_mean_scaled": k002_mean[index],
        "healthy_mean_scaled": healthy_mean[index],
        "faulty_mean_scaled": faulty_mean[index],
        "distance_to_healthy": distance_to_healthy,
        "distance_to_faulty": distance_to_faulty,
        "closer_to": closer_to
    })


results_df = pd.DataFrame(results)

results_df["distance_difference"] = (
    results_df["distance_to_healthy"]
    - results_df["distance_to_faulty"]
)

results_df = results_df.sort_values(
    "distance_difference",
    ascending=False
)

print("\n" + "=" * 65)
print("K002 FEATURE SIMILARITY")
print("=" * 65)

print(
    results_df[
        [
            "feature",
            "distance_to_healthy",
            "distance_to_faulty",
            "closer_to"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_df.to_csv(
    METRICS_DIR / "k002_feature_similarity.csv",
    index=False
)


# =====================================================
# SUMMARY
# =====================================================

closer_to_faulty = int(
    (results_df["closer_to"] == "Faulty").sum()
)

closer_to_healthy = int(
    (results_df["closer_to"] == "Healthy").sum()
)

print("\n" + "=" * 65)
print("SIMILARITY SUMMARY")
print("=" * 65)

print(
    "Features closer to healthy mean:",
    closer_to_healthy
)

print(
    "Features closer to faulty mean:",
    closer_to_faulty
)


# =====================================================
# CREATE COMPARISON PLOT
# =====================================================

plot_df = results_df.sort_values(
    "distance_difference"
)

plt.figure(figsize=(11, 8))

plt.barh(
    plot_df["feature"],
    plot_df["distance_difference"]
)

plt.axvline(
    0,
    color="black",
    linewidth=1
)

plt.xlabel(
    "Distance to healthy mean minus distance to faulty mean"
)

plt.ylabel("Vibration feature")

plt.title(
    "K002 Feature Similarity to Healthy and Faulty Development Data"
)

plt.tight_layout()

plot_path = (
    PLOTS_DIR / "k002_feature_similarity.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print("\nPlot saved to:", plot_path)

print("\nSTAGE 11 COMPLETED!")