from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score
)


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = Path(
    "models/threshold_optimized_model.joblib"
)

TEST_PATH = Path(
    "dataset/processed/test.csv"
)

OUTPUT_DIR = Path(
    "results/metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 70)
print("STEP 15 - FINAL LOGISTIC REGRESSION EVALUATION")
print("=" * 70)

bundle = joblib.load(
    MODEL_PATH
)

model = bundle["pipeline"]

features = bundle["feature_columns"]

threshold = bundle.get(
    "decision_threshold",
    0.85
)

print("\nModel loaded successfully.")
print("Features:", len(features))
print("Decision threshold:", threshold)


# =====================================================
# LOAD TEST DATA
# =====================================================

test_df = pd.read_csv(
    TEST_PATH,
    low_memory=False
)

X_test = test_df[features]

actual = test_df[
    "target"
].to_numpy()


# =====================================================
# PREDICT
# =====================================================

fault_probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)

predictions = (
    fault_probabilities >= threshold
).astype(int)


# =====================================================
# OVERALL RESULTS
# =====================================================

print("\n" + "=" * 70)
print("FINAL LOGISTIC TEST RESULTS")
print("=" * 70)

print(
    "Accuracy:",
    round(
        accuracy_score(
            actual,
            predictions
        ),
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

results_df[
    "fault_probability"
] = fault_probabilities

results_df[
    "prediction"
] = predictions

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
print("FINAL LOGISTIC PER-BEARING RESULTS")
print("=" * 70)

print(
    bearing_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# SAVE RESULTS
# =====================================================

results_df.to_csv(
    OUTPUT_DIR / "final_logistic_test_predictions.csv",
    index=False
)

bearing_df.to_csv(
    OUTPUT_DIR / "final_logistic_test_per_bearing.csv",
    index=False
)

print("\nResults saved successfully.")
print("\nSTEP 15 COMPLETED!")