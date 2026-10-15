from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score
)


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed")
OUTPUT_DIR = Path("results/metrics")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

REMOVED_BANDS = [
    "vib_band_energy_2",
    "vib_band_energy_3",
    "vib_band_energy_4"
]

REMOVED_FREQUENCY = REMOVED_BANDS + [
    "vib_dominant_freq"
]


# =====================================================
# LOAD DEVELOPMENT DATA
# =====================================================

print("=" * 65)
print("STAGE 12: FREQUENCY FEATURE EXPERIMENT")
print("=" * 65)

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
    column
    for column in df.columns
    if column.startswith("vib_")
]

feature_sets = {
    "All_19": all_features,

    "Without_3_Bands": [
        feature
        for feature in all_features
        if feature not in REMOVED_BANDS
    ],

    "Without_4_Frequency_Features": [
        feature
        for feature in all_features
        if feature not in REMOVED_FREQUENCY
    ]
}


# =====================================================
# CREATE MODEL
# =====================================================

def create_model():

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])


# =====================================================
# BEARING-LEVEL CROSS-VALIDATION
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

if len(healthy_bearings) != 4:
    raise ValueError(
        "Expected four healthy development bearings."
    )

rng = np.random.default_rng(RANDOM_STATE)

shuffled_faulty = rng.permutation(
    faulty_bearings
)

faulty_groups = np.array_split(
    shuffled_faulty,
    4
)

folds = []

for fold_index in range(4):

    validation_bearings = [
        healthy_bearings[fold_index],
        *faulty_groups[fold_index].tolist()
    ]

    validation_mask = df[
        "bearing_code"
    ].isin(validation_bearings)

    folds.append((
        np.flatnonzero(
            (~validation_mask).to_numpy()
        ),
        np.flatnonzero(
            validation_mask.to_numpy()
        )
    ))


# =====================================================
# RUN EXPERIMENTS
# =====================================================

results = []
bearing_results = []

for feature_set_name, features in feature_sets.items():

    print("\n" + "=" * 65)
    print("FEATURE SET:", feature_set_name)
    print("Number of features:", len(features))
    print("=" * 65)

    for fold_number, (train_idx, val_idx) in enumerate(
        folds,
        start=1
    ):

        model = create_model()

        model.fit(
            df.iloc[train_idx][features],
            df.iloc[train_idx]["target"]
        )

        predictions = model.predict(
            df.iloc[val_idx][features]
        )

        actual = df.iloc[val_idx][
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

        # Evaluate each validation bearing separately.

        fold_df = df.iloc[val_idx].copy()

        fold_df["prediction"] = predictions

        for bearing, group in fold_df.groupby(
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
# SUMMARY
# =====================================================

results_df = pd.DataFrame(results)

bearing_df = pd.DataFrame(
    bearing_results
)

summary = (
    results_df.groupby("feature_set")
    .agg(
        mean_accuracy=("accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
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

print("\n" + "=" * 65)
print("FREQUENCY FEATURE EXPERIMENT SUMMARY")
print("=" * 65)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_df.to_csv(
    OUTPUT_DIR / "frequency_feature_folds.csv",
    index=False
)

bearing_df.to_csv(
    OUTPUT_DIR / "frequency_feature_per_bearing.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "frequency_feature_summary.csv",
    index=False
)

print("\nSTAGE 12 COMPLETED!")