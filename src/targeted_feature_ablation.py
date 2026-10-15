from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed")
OUTPUT_DIR = Path("results/metrics")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42
THRESHOLD = 0.85


# =====================================================
# LOAD DEVELOPMENT DATA ONLY
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 5")
print("TARGETED FEATURE ABLATION")
print("=" * 70)

train_df = pd.read_csv(
    DATA_DIR / "train.csv",
    low_memory=False
)

val_df = pd.read_csv(
    DATA_DIR / "validation.csv",
    low_memory=False
)

df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

all_features = [
    col
    for col in df.columns
    if col.startswith("vib_")
]

print("\nDevelopment records:", len(df))
print(
    "Development bearings:",
    df["bearing_code"].nunique()
)

print("Original features:", len(all_features))


# =====================================================
# FEATURE CONFIGURATIONS
# =====================================================

feature_sets = {
    "All_19": all_features,

    "Without_Band_4": [
        f for f in all_features
        if f != "vib_band_energy_4"
    ],

    "Without_Band_3": [
        f for f in all_features
        if f != "vib_band_energy_3"
    ],

    "Without_Bands_3_4": [
        f for f in all_features
        if f not in [
            "vib_band_energy_3",
            "vib_band_energy_4"
        ]
    ]
}


# =====================================================
# SAME BEARING-LEVEL FOLDS AS STEP 2
# =====================================================

healthy_bearings = sorted(
    df.loc[
        df["target"] == 0,
        "bearing_code"
    ].unique()
)

faulty_bearings = sorted(
    df.loc[
        df["target"] == 1,
        "bearing_code"
    ].unique()
)

rng = np.random.default_rng(
    RANDOM_STATE
)

faulty_groups = np.array_split(
    rng.permutation(faulty_bearings),
    4
)

folds = []

for i in range(4):

    validation_bearings = [
        healthy_bearings[i],
        *faulty_groups[i].tolist()
    ]

    mask = df[
        "bearing_code"
    ].isin(validation_bearings)

    folds.append((
        np.flatnonzero(
            (~mask).to_numpy()
        ),
        np.flatnonzero(
            mask.to_numpy()
        )
    ))


# =====================================================
# TRAIN AND EVALUATE
# =====================================================

results = []
bearing_results = []

for feature_set_name, features in feature_sets.items():

    print("\n" + "=" * 70)
    print("FEATURE SET:", feature_set_name)
    print("Number of features:", len(features))
    print("=" * 70)

    for fold_number, (
        train_idx,
        val_idx
    ) in enumerate(folds, start=1):

        train_part = df.iloc[train_idx]
        val_part = df.iloc[val_idx]

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ])

        model.fit(
            train_part[features],
            train_part["target"]
        )

        probabilities = model.predict_proba(
            val_part[features]
        )[:, 1]

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        actual = val_part[
            "target"
        ].to_numpy()

        accuracy = accuracy_score(
            actual,
            predictions
        )

        macro_f1 = f1_score(
            actual,
            predictions,
            average="macro",
            zero_division=0
        )

        healthy_recall = recall_score(
            actual,
            predictions,
            pos_label=0,
            zero_division=0
        )

        faulty_recall = recall_score(
            actual,
            predictions,
            pos_label=1,
            zero_division=0
        )

        results.append({
            "feature_set": feature_set_name,
            "fold": fold_number,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "healthy_recall": healthy_recall,
            "faulty_recall": faulty_recall
        })

        print(
            f"Fold {fold_number} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro-F1: {macro_f1:.4f} | "
            f"Healthy Recall: {healthy_recall:.4f} | "
            f"Faulty Recall: {faulty_recall:.4f}"
        )

        temp = val_part[
            ["bearing_code", "target"]
        ].copy()

        temp["prediction"] = predictions

        for bearing, group in temp.groupby(
            "bearing_code"
        ):

            bearing_results.append({
                "feature_set": feature_set_name,
                "fold": fold_number,
                "bearing_code": bearing,
                "actual_class": int(
                    group["target"].iloc[0]
                ),
                "accuracy": accuracy_score(
                    group["target"],
                    group["prediction"]
                )
            })


# =====================================================
# CREATE SUMMARY
# =====================================================

results_df = pd.DataFrame(results)
bearing_df = pd.DataFrame(bearing_results)

summary = (
    results_df.groupby("feature_set")
    .agg(
        mean_accuracy=(
            "accuracy", "mean"
        ),
        mean_macro_f1=(
            "macro_f1", "mean"
        ),
        mean_healthy_recall=(
            "healthy_recall", "mean"
        ),
        mean_faulty_recall=(
            "faulty_recall", "mean"
        )
    )
    .reset_index()
)

healthy_results = bearing_df[
    bearing_df["actual_class"] == 0
]

worst_healthy = (
    healthy_results.groupby(
        "feature_set"
    )["accuracy"]
    .min()
    .reset_index(
        name="worst_healthy_accuracy"
    )
)

summary = summary.merge(
    worst_healthy,
    on="feature_set"
)

summary = summary.sort_values(
    "mean_macro_f1",
    ascending=False
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 70)
print("TARGETED FEATURE ABLATION SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# SAVE RESULTS
# =====================================================

results_df.to_csv(
    OUTPUT_DIR / "targeted_ablation_folds.csv",
    index=False
)

bearing_df.to_csv(
    OUTPUT_DIR / "targeted_ablation_per_bearing.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "targeted_ablation_summary.csv",
    index=False
)

print("\nSTEP 5 COMPLETED!")