from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

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


# =====================================================
# LOAD DEVELOPMENT DATA
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 3")
print("HEALTHY-ONLY ISOLATION FOREST EXPERIMENT")
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

features = [
    col
    for col in df.columns
    if col.startswith("vib_")
]

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

print("\nHealthy bearings:")
print(healthy_bearings)

print("\nFaulty bearings:")
print(faulty_bearings)

print("\nFeatures:", len(features))


# =====================================================
# CREATE SAME 4 BEARING-LEVEL FOLDS
# =====================================================

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

    validation_mask = df[
        "bearing_code"
    ].isin(validation_bearings)

    train_mask = ~validation_mask

    folds.append((
        np.flatnonzero(
            train_mask.to_numpy()
        ),
        np.flatnonzero(
            validation_mask.to_numpy()
        )
    ))

    print(
        f"\nFold {i + 1} validation bearings:",
        validation_bearings
    )


# =====================================================
# CONTAMINATION VALUES
# =====================================================

# contamination controls the boundary used by
# Isolation Forest.
#
# We test several values using DEVELOPMENT data only.

contamination_values = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05
]


# =====================================================
# RUN EXPERIMENT
# =====================================================

results = []
bearing_results = []

for contamination in contamination_values:

    print("\n" + "=" * 70)
    print(
        "CONTAMINATION:",
        contamination
    )
    print("=" * 70)

    for fold_number, (
        train_idx,
        val_idx
    ) in enumerate(
        folds,
        start=1
    ):

        train_part = df.iloc[
            train_idx
        ].copy()

        val_part = df.iloc[
            val_idx
        ].copy()

        # =============================================
        # TRAIN USING HEALTHY DATA ONLY
        # =============================================

        healthy_train = train_part[
            train_part["target"] == 0
        ]

        train_healthy_bearings = set(
            healthy_train[
                "bearing_code"
            ]
        )

        validation_bearings = set(
            val_part[
                "bearing_code"
            ]
        )

        assert train_healthy_bearings.isdisjoint(
            validation_bearings
        )

        # =============================================
        # STANDARDIZATION
        # =============================================

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            healthy_train[features]
        )

        X_val = scaler.transform(
            val_part[features]
        )

        # =============================================
        # ISOLATION FOREST
        # =============================================

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(
            X_train
        )

        # Isolation Forest:
        #
        # +1 = normal
        # -1 = anomaly
        #
        # Our project:
        #
        # 0 = healthy
        # 1 = faulty

        raw_prediction = model.predict(
            X_val
        )

        predictions = np.where(
            raw_prediction == -1,
            1,
            0
        )

        actual = val_part[
            "target"
        ].to_numpy()

        # =============================================
        # METRICS
        # =============================================

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
            "contamination": contamination,
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
            f"Healthy Recall: "
            f"{healthy_recall:.4f} | "
            f"Faulty Recall: "
            f"{faulty_recall:.4f}"
        )

        # =============================================
        # PER-BEARING RESULTS
        # =============================================

        temp = val_part[
            [
                "bearing_code",
                "target"
            ]
        ].copy()

        temp["prediction"] = predictions

        for bearing, group in temp.groupby(
            "bearing_code"
        ):

            bearing_results.append({
                "contamination": contamination,
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

results_df = pd.DataFrame(
    results
)

bearing_df = pd.DataFrame(
    bearing_results
)

summary = (
    results_df.groupby(
        "contamination"
    )
    .agg(
        mean_accuracy=(
            "accuracy",
            "mean"
        ),
        mean_macro_f1=(
            "macro_f1",
            "mean"
        ),
        mean_healthy_recall=(
            "healthy_recall",
            "mean"
        ),
        mean_faulty_recall=(
            "faulty_recall",
            "mean"
        )
    )
    .reset_index()
)


# =====================================================
# WORST HEALTHY BEARING
# =====================================================

healthy_bearing_results = bearing_df[
    bearing_df[
        "actual_class"
    ] == 0
]

worst_healthy = (
    healthy_bearing_results
    .groupby(
        "contamination"
    )["accuracy"]
    .min()
    .reset_index(
        name="worst_healthy_accuracy"
    )
)


# =====================================================
# WORST FAULTY BEARING
# =====================================================

faulty_bearing_results = bearing_df[
    bearing_df[
        "actual_class"
    ] == 1
]

worst_faulty = (
    faulty_bearing_results
    .groupby(
        "contamination"
    )["accuracy"]
    .min()
    .reset_index(
        name="worst_faulty_accuracy"
    )
)


summary = summary.merge(
    worst_healthy,
    on="contamination"
)

summary = summary.merge(
    worst_faulty,
    on="contamination"
)

summary = summary.sort_values(
    "mean_macro_f1",
    ascending=False
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 70)
print("ISOLATION FOREST SUMMARY")
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
    OUTPUT_DIR /
    "isolation_forest_folds.csv",
    index=False
)

bearing_df.to_csv(
    OUTPUT_DIR /
    "isolation_forest_per_bearing.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR /
    "isolation_forest_summary.csv",
    index=False
)

print(
    "\nResults saved successfully."
)

print(
    "\nSTEP 3 COMPLETED!"
)