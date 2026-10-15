from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

LOGISTIC_THRESHOLD = 0.85

KMEANS_CLUSTERS = 16
KMEANS_PERCENTILE = 99.5


# =====================================================
# LOAD DEVELOPMENT DATA
# =====================================================

print("=" * 70)
print("STEP 13 - LOGISTIC REGRESSION VS K-MEANS")
print("BEARING-DISJOINT DEVELOPMENT COMPARISON")
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

rng = np.random.default_rng(
    RANDOM_STATE
)

faulty_groups = np.array_split(
    rng.permutation(faulty_bearings),
    4
)

results = []
bearing_results = []


# =====================================================
# EVALUATE IDENTICAL FOLDS
# =====================================================

for fold_index in range(4):

    held_out = [
        healthy_bearings[fold_index],
        *faulty_groups[fold_index].tolist()
    ]

    val_mask = df[
        "bearing_code"
    ].isin(held_out)

    train_part = df.loc[
        ~val_mask
    ].copy()

    val_part = df.loc[
        val_mask
    ].copy()

    X_train = train_part[features]
    y_train = train_part["target"]

    X_val = val_part[features]
    y_val = val_part["target"].to_numpy()

    print(
        f"\nFold {fold_index + 1}"
    )

    print(
        "Held-out bearings:",
        held_out
    )

    # ---------------------------------------------
    # LOGISTIC REGRESSION
    # ---------------------------------------------

    logistic = Pipeline([
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

    logistic.fit(
        X_train,
        y_train
    )

    logistic_scores = (
        logistic.predict_proba(
            X_val
        )[:, 1]
    )

    logistic_predictions = (
        logistic_scores
        >= LOGISTIC_THRESHOLD
    ).astype(int)

    # ---------------------------------------------
    # K-MEANS
    # ---------------------------------------------

    healthy_train = train_part[
        train_part["target"] == 0
    ]

    scaler = StandardScaler()

    X_healthy = scaler.fit_transform(
        healthy_train[features]
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    kmeans = MiniBatchKMeans(
        n_clusters=KMEANS_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=512,
        n_init=10,
        max_iter=200
    )

    kmeans.fit(
        X_healthy
    )

    train_distances = (
        kmeans.transform(
            X_healthy
        ).min(axis=1)
    )

    val_distances = (
        kmeans.transform(
            X_val_scaled
        ).min(axis=1)
    )

    kmeans_threshold = np.percentile(
        train_distances,
        KMEANS_PERCENTILE
    )

    kmeans_predictions = (
        val_distances
        > kmeans_threshold
    ).astype(int)

    # ---------------------------------------------
    # METRICS
    # ---------------------------------------------

    predictions_by_model = {
        "Logistic Regression": (
            logistic_predictions
        ),
        "K-Means": (
            kmeans_predictions
        )
    }

    for model_name, predictions in (
        predictions_by_model.items()
    ):

        results.append({
            "model": model_name,
            "fold": fold_index + 1,
            "accuracy": accuracy_score(
                y_val,
                predictions
            ),
            "macro_f1": f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0
            ),
            "healthy_recall": recall_score(
                y_val,
                predictions,
                pos_label=0,
                zero_division=0
            ),
            "faulty_recall": recall_score(
                y_val,
                predictions,
                pos_label=1,
                zero_division=0
            )
        })

        temp = val_part[
            ["bearing_code", "target"]
        ].copy()

        temp["prediction"] = predictions

        for bearing, group in temp.groupby(
            "bearing_code"
        ):

            bearing_results.append({
                "model": model_name,
                "fold": fold_index + 1,
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

results_df = pd.DataFrame(
    results
)

bearing_df = pd.DataFrame(
    bearing_results
)

summary = (
    results_df.groupby(
        "model"
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

worst_healthy = (
    bearing_df[
        bearing_df["actual_class"] == 0
    ]
    .groupby("model")["accuracy"]
    .min()
    .reset_index(
        name="worst_healthy_accuracy"
    )
)

worst_faulty = (
    bearing_df[
        bearing_df["actual_class"] == 1
    ]
    .groupby("model")["accuracy"]
    .min()
    .reset_index(
        name="worst_faulty_accuracy"
    )
)

summary = summary.merge(
    worst_healthy,
    on="model"
)

summary = summary.merge(
    worst_faulty,
    on="model"
)


# =====================================================
# PRINT AND SAVE
# =====================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print("PER-BEARING COMPARISON")
print("=" * 70)

print(
    bearing_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison_folds.csv",
    index=False
)

bearing_df.to_csv(
    OUTPUT_DIR / "model_comparison_per_bearing.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "model_comparison_summary.csv",
    index=False
)

print("\nResults saved successfully.")
print("\nSTEP 13 COMPLETED!")
