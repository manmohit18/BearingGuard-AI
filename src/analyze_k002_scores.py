from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = Path(
    "models/baseline_logistic_regression.joblib"
)

TEST_PATH = Path(
    "dataset/processed/test.csv"
)

OUTPUT_DIR = Path("results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Diagnostic thresholds only.
# These will NOT be used to select a new model.
THRESHOLDS = [0.50, 0.85]


# =====================================================
# LOAD ORIGINAL FROZEN MODEL
# =====================================================

print("=" * 65)
print("STAGE 10: K002 FAULT-SCORE DIAGNOSTICS")
print("=" * 65)

bundle = joblib.load(MODEL_PATH)

model = bundle["pipeline"]
features = bundle["feature_columns"]

print("\nLoaded original Logistic Regression model.")


# =====================================================
# LOAD TEST DATA
# =====================================================

test_df = pd.read_csv(
    TEST_PATH,
    low_memory=False
)

scores = model.predict_proba(
    test_df[features]
)[:, 1]

test_df["fault_score"] = scores


# =====================================================
# SUMMARIZE SCORES BY BEARING
# =====================================================

summary = (
    test_df.groupby(
        ["bearing_code", "target"]
    )["fault_score"]
    .agg(
        records="count",
        mean="mean",
        median="median",
        minimum="min",
        maximum="max"
    )
    .reset_index()
)

print("\nFAULT-SCORE DISTRIBUTION BY BEARING")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

summary.to_csv(
    OUTPUT_DIR / "test_bearing_score_summary.csv",
    index=False
)


# =====================================================
# COMPARE TWO PREDEFINED THRESHOLDS
# =====================================================

results = []

for threshold in THRESHOLDS:

    predictions = (
        scores >= threshold
    ).astype(int)

    for bearing, group in test_df.groupby(
        "bearing_code"
    ):

        mask = (
            test_df["bearing_code"] == bearing
        )

        bearing_predictions = predictions[
            mask.to_numpy()
        ]

        actual = group["target"].to_numpy()

        accuracy = float(
            np.mean(
                bearing_predictions == actual
            )
        )

        results.append({
            "threshold": threshold,
            "bearing_code": bearing,
            "actual_class": int(actual[0]),
            "records": len(group),
            "accuracy": accuracy,
            "fault_predictions": int(
                bearing_predictions.sum()
            )
        })


results_df = pd.DataFrame(results)

print("\nPER-BEARING DIAGNOSTIC COMPARISON")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_df.to_csv(
    OUTPUT_DIR / "test_threshold_diagnostic.csv",
    index=False
)


# =====================================================
# K002 SCORE DISTRIBUTION
# =====================================================

k002 = test_df[
    test_df["bearing_code"] == "K002"
]

if k002.empty:
    raise ValueError("K002 not found in test data.")

print("\nK002 FAULT-SCORE PERCENTILES")

print(
    k002["fault_score"].quantile(
        [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    ).to_string()
)

print("\nSTAGE 10 DIAGNOSTICS COMPLETED!")