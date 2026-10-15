from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score
)


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("results/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

N_CLUSTERS = 16
THRESHOLD_PERCENTILE = 99.5


# =====================================================
# LOAD DATA
# =====================================================

print("=" * 70)
print("STEP 14 - K-MEANS TEST-BEARING EVALUATION")
print("=" * 70)

train_df = pd.read_csv(
    DATA_DIR / "train.csv",
    low_memory=False
)

validation_df = pd.read_csv(
    DATA_DIR / "validation.csv",
    low_memory=False
)

test_df = pd.read_csv(
    DATA_DIR / "test.csv",
    low_memory=False
)

development_df = pd.concat(
    [train_df, validation_df],
    ignore_index=True
)

features = [
    column
    for column in development_df.columns
    if column.startswith("vib_")
]

healthy_development = development_df[
    development_df["target"] == 0
].copy()

print("\nTraining healthy bearings:")
print(
    sorted(
        healthy_development[
            "bearing_code"
        ].unique()
    )
)

print("\nTest bearings:")
print(
    sorted(
        test_df[
            "bearing_code"
        ].unique()
    )
)


# =====================================================
# TRAIN K-MEANS ON HEALTHY DEVELOPMENT DATA
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    healthy_development[features]
)

kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    batch_size=512,
    n_init=10,
    max_iter=200
)

kmeans.fit(X_train)

train_distances = (
    kmeans.transform(X_train)
    .min(axis=1)
)

threshold = np.percentile(
    train_distances,
    THRESHOLD_PERCENTILE
)

print("\nK-Means configuration:")
print("Clusters:", N_CLUSTERS)
print("Percentile:", THRESHOLD_PERCENTILE)
print("Distance threshold:", round(threshold, 6))


# =====================================================
# EVALUATE TEST BEARINGS
# =====================================================

X_test = scaler.transform(
    test_df[features]
)

test_distances = (
    kmeans.transform(X_test)
    .min(axis=1)
)

predictions = (
    test_distances > threshold
).astype(int)

actual = test_df[
    "target"
].to_numpy()

print("\n" + "=" * 70)
print("OVERALL TEST RESULTS")
print("=" * 70)

print(
    "Accuracy:",
    round(
        accuracy_score(actual, predictions),
        4
    )
)

print(
    "Macro-F1:",
    round(
        f1_score(
            actual,
            predictions,
            average="macro",
            zero_division=0
        ),
        4
    )
)

print(
    "Healthy recall:",
    round(
        recall_score(
            actual,
            predictions,
            pos_label=0
        ),
        4
    )
)

print(
    "Faulty recall:",
    round(
        recall_score(
            actual,
            predictions,
            pos_label=1
        ),
        4
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        actual,
        predictions,
        labels=[0, 1]
    )
)


# =====================================================
# PER-BEARING RESULTS
# =====================================================

results_df = test_df[
    ["bearing_code", "target"]
].copy()

results_df["prediction"] = predictions
results_df["anomaly_distance"] = test_distances

bearing_results = []

for bearing, group in results_df.groupby(
    "bearing_code"
):

    bearing_results.append({
        "bearing_code": bearing,
        "actual_class": int(
            group["target"].iloc[0]
        ),
        "windows": len(group),
        "correct_windows": int(
            (
                group["target"]
                == group["prediction"]
            ).sum()
        ),
        "accuracy": accuracy_score(
            group["target"],
            group["prediction"]
        ),
        "predicted_fault_rate": (
            group["prediction"].mean()
        )
    })

bearing_df = pd.DataFrame(
    bearing_results
)

print("\n" + "=" * 70)
print("PER-BEARING TEST RESULTS")
print("=" * 70)

print(
    bearing_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# SAVE MODEL AND RESULTS
# =====================================================

model_bundle = {
    "scaler": scaler,
    "kmeans": kmeans,
    "feature_columns": features,
    "distance_threshold": threshold,
    "n_clusters": N_CLUSTERS,
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "model_name": "Healthy-only MiniBatchKMeans"
}

joblib.dump(
    model_bundle,
    MODEL_DIR / "kmeans_fault_detector.joblib"
)

results_df.to_csv(
    OUTPUT_DIR / "kmeans_test_predictions.csv",
    index=False
)

bearing_df.to_csv(
    OUTPUT_DIR / "kmeans_test_per_bearing.csv",
    index=False
)

print("\nModel saved:")
print(
    MODEL_DIR / "kmeans_fault_detector.joblib"
)

print("\nResults saved successfully.")
print("\nSTEP 14 COMPLETED!")
