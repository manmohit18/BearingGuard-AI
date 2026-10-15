from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
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

CLUSTER_COUNTS = [4, 8, 16, 32]

# Percentiles of training-healthy anomaly scores.
# Higher percentile = fewer training false alarms.
PERCENTILES = [95, 99, 99.5]


# =====================================================
# LOAD DEVELOPMENT DATA ONLY
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 12")
print("HEALTHY-ONLY K-MEANS EXPERIMENT")
print("=" * 70)

train_df = pd.read_csv(
    DATA_DIR / "train.csv",
    low_memory=False
)

validation_df = pd.read_csv(
    DATA_DIR / "validation.csv",
    low_memory=False
)

df = pd.concat(
    [train_df, validation_df],
    ignore_index=True
)

features = [
    column
    for column in df.columns
    if column.startswith("vib_")
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

print("\nDevelopment windows:", len(df))
print("Features:", len(features))
print("Healthy bearings:", healthy_bearings)
print("Faulty bearings:", faulty_bearings)


# =====================================================
# BEARING-DISJOINT CROSS-VALIDATION
# =====================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

faulty_groups = np.array_split(
    rng.permutation(faulty_bearings),
    4
)

folds = []

for fold_index in range(4):

    held_out_bearings = [
        healthy_bearings[fold_index],
        *faulty_groups[fold_index].tolist()
    ]

    validation_mask = df[
        "bearing_code"
    ].isin(held_out_bearings)

    train_indices = np.flatnonzero(
        (~validation_mask).to_numpy()
    )

    validation_indices = np.flatnonzero(
        validation_mask.to_numpy()
    )

    folds.append(
        (
            train_indices,
            validation_indices,
            held_out_bearings
        )
    )


# =====================================================
# EXPERIMENT
# =====================================================

fold_results = []
bearing_results = []

for cluster_count in CLUSTER_COUNTS:

    print("\n" + "=" * 70)
    print("CLUSTERS:", cluster_count)
    print("=" * 70)

    for fold_number, (
        train_idx,
        val_idx,
        held_out_bearings
    ) in enumerate(folds, start=1):

        train_part = df.iloc[train_idx]
        val_part = df.iloc[val_idx]

        # Train clustering only on healthy windows.
        healthy_train = train_part[
            train_part["target"] == 0
        ]

        scaler = StandardScaler()

        X_healthy = scaler.fit_transform(
            healthy_train[features]
        )

        X_val = scaler.transform(
            val_part[features]
        )

        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_STATE,
            batch_size=512,
            n_init=10,
            max_iter=200
        )

        model.fit(X_healthy)

        # Distance to nearest healthy cluster.
        train_distances = model.transform(
            X_healthy
        ).min(axis=1)

        val_distances = model.transform(
            X_val
        ).min(axis=1)

        actual = val_part[
            "target"
        ].to_numpy()

        for percentile in PERCENTILES:

            threshold = np.percentile(
                train_distances,
                percentile
            )

            predictions = (
                val_distances > threshold
            ).astype(int)

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

            fold_results.append({
                "clusters": cluster_count,
                "percentile": percentile,
                "fold": fold_number,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "healthy_recall": healthy_recall,
                "faulty_recall": faulty_recall
            })

            temp = val_part[
                ["bearing_code", "target"]
            ].copy()

            temp["prediction"] = predictions

            for bearing, group in temp.groupby(
                "bearing_code"
            ):

                bearing_results.append({
                    "clusters": cluster_count,
                    "percentile": percentile,
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

            print(
                f"Fold {fold_number} | "
                f"Percentile {percentile:5.1f} | "
                f"Accuracy: {accuracy:.4f} | "
                f"Macro-F1: {macro_f1:.4f} | "
                f"Healthy: {healthy_recall:.4f} | "
                f"Faulty: {faulty_recall:.4f}"
            )


# =====================================================
# SUMMARY
# =====================================================

fold_df = pd.DataFrame(
    fold_results
)

bearing_df = pd.DataFrame(
    bearing_results
)

summary = (
    fold_df.groupby(
        ["clusters", "percentile"]
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

healthy_results = bearing_df[
    bearing_df["actual_class"] == 0
]

faulty_results = bearing_df[
    bearing_df["actual_class"] == 1
]

worst_healthy = (
    healthy_results.groupby(
        ["clusters", "percentile"]
    )["accuracy"]
    .min()
    .reset_index(
        name="worst_healthy_accuracy"
    )
)

worst_faulty = (
    faulty_results.groupby(
        ["clusters", "percentile"]
    )["accuracy"]
    .min()
    .reset_index(
        name="worst_faulty_accuracy"
    )
)

summary = summary.merge(
    worst_healthy,
    on=["clusters", "percentile"]
)

summary = summary.merge(
    worst_faulty,
    on=["clusters", "percentile"]
)

summary = summary.sort_values(
    "mean_macro_f1",
    ascending=False
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 70)
print("K-MEANS EXPERIMENT SUMMARY")
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

fold_df.to_csv(
    OUTPUT_DIR / "kmeans_folds.csv",
    index=False
)

bearing_df.to_csv(
    OUTPUT_DIR / "kmeans_per_bearing.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "kmeans_summary.csv",
    index=False
)

print("\nResults saved successfully.")
print("\nSTEP 12 COMPLETED!")