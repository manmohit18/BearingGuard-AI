from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = Path("dataset/processed")
PLOT_DIR = Path("results/plots")
RESULT_DIR = Path("results/metrics")

PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

print("=" * 60)
print("STAGE 7: FEATURE AND GENERALIZATION ANALYSIS")
print("=" * 60)

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

train_df["split"] = "Train"
val_df["split"] = "Validation"
test_df["split"] = "Test"

df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

print("\nTotal records:", len(df))
print("Unique bearings:", df["bearing_code"].nunique())


# ==================================================
# BEARING INFORMATION
# ==================================================

print("\n" + "=" * 60)
print("BEARING DISTRIBUTION")
print("=" * 60)

bearing_summary = (
    df.groupby(["split", "bearing_code", "target"])
    .size()
    .reset_index(name="records")
)

print(bearing_summary.to_string(index=False))

bearing_summary.to_csv(
    RESULT_DIR / "bearing_distribution.csv",
    index=False
)


# ==================================================
# OPERATING CONDITION DISTRIBUTION
# ==================================================

if "speed_rpm" in df.columns:

    print("\n" + "=" * 60)
    print("SPEED DISTRIBUTION")
    print("=" * 60)

    speed_summary = (
        df.groupby(["split", "target", "speed_rpm"])
        .size()
        .reset_index(name="records")
    )

    print(speed_summary.to_string(index=False))

    speed_summary.to_csv(
        RESULT_DIR / "speed_distribution.csv",
        index=False
    )


# ==================================================
# FEATURE STATISTICS
# ==================================================

selected_features = [
    "vib_mean",
    "vib_rms",
    "vib_std",
    "vib_spectral_centroid",
    "vib_band_energy_0"
]

available_features = [
    feature
    for feature in selected_features
    if feature in df.columns
]

feature_stats = (
    df.groupby(["split", "bearing_code", "target"])[
        available_features
    ]
    .agg(["mean", "std", "min", "max"])
)

feature_stats.to_csv(
    RESULT_DIR / "bearing_feature_statistics.csv"
)

print("\nFeature statistics saved.")


# ==================================================
# FEATURE DISTRIBUTION GRAPHS
# ==================================================

for feature in available_features:

    plt.figure(figsize=(14, 6))

    sns.boxplot(
        data=df,
        x="bearing_code",
        y=feature,
        hue="split",
        showfliers=False
    )

    plt.title(
        f"{feature} Distribution Across Bearings"
    )

    plt.xlabel("Bearing")
    plt.ylabel(feature)

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{feature}_bearing_distribution.png",
        dpi=300
    )

    plt.close()


# ==================================================
# COMPARE HEALTHY BEARINGS
# ==================================================

healthy_df = df[df["target"] == 0].copy()

print("\n" + "=" * 60)
print("HEALTHY BEARING FEATURE MEANS")
print("=" * 60)

healthy_means = (
    healthy_df.groupby(["split", "bearing_code"])[
        available_features
    ]
    .mean()
    .reset_index()
)

print(healthy_means.to_string(index=False))

healthy_means.to_csv(
    RESULT_DIR / "healthy_bearing_feature_means.csv",
    index=False
)


# ==================================================
# K002 DIAGNOSTIC SUMMARY
# ==================================================

k002_df = test_df[
    test_df["bearing_code"] == "K002"
]

if not k002_df.empty:

    print("\n" + "=" * 60)
    print("K002 DIAGNOSTIC SUMMARY")
    print("=" * 60)

    print("\nRecords:", len(k002_df))

    print("\nFeature means:")

    print(
        k002_df[available_features]
        .mean()
        .to_string()
    )

    if "speed_rpm" in k002_df.columns:

        print("\nOperating speeds:")

        print(
            k002_df["speed_rpm"]
            .value_counts()
            .to_string()
        )


print("\nDiagnostic graphs saved to:", PLOT_DIR)
print("Diagnostic tables saved to:", RESULT_DIR)

print("\nSTAGE 7 DIAGNOSTICS COMPLETED!")