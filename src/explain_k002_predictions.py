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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 4")
print("LOGISTIC REGRESSION FEATURE CONTRIBUTIONS")
print("=" * 70)

bundle = joblib.load(MODEL_PATH)

pipeline = bundle["pipeline"]
features = bundle["feature_columns"]

scaler = pipeline.named_steps["scaler"]
classifier = pipeline.named_steps["classifier"]

coefficients = classifier.coef_[0]

print("\nModel loaded successfully.")
print("Features:", len(features))


# =====================================================
# LOAD HEALTHY TEST BEARINGS
# =====================================================

df = pd.read_csv(
    TEST_PATH,
    low_memory=False
)

df = df[
    df["bearing_code"].isin(
        ["K001", "K002"]
    )
].copy()

if df.empty:
    raise ValueError(
        "K001 and K002 not found."
    )


# =====================================================
# CALCULATE STANDARDIZED FEATURES
# =====================================================

# Use the scaler already fitted inside the
# original trained model.
#
# Do not fit a new scaler using test data.

X_scaled = scaler.transform(
    df[features]
)


# =====================================================
# CALCULATE FEATURE CONTRIBUTIONS
# =====================================================

# Each feature contribution equals:
#
# standardized feature value × model coefficient

contributions = (
    X_scaled * coefficients
)

contribution_df = pd.DataFrame(
    contributions,
    columns=features,
    index=df.index
)

contribution_df["bearing_code"] = (
    df["bearing_code"]
)


# =====================================================
# COMPARE K001 AND K002
# =====================================================

mean_contributions = (
    contribution_df.groupby(
        "bearing_code"
    )[features]
    .mean()
    .T
)

mean_contributions["difference_K002_minus_K001"] = (
    mean_contributions["K002"]
    - mean_contributions["K001"]
)

mean_contributions = (
    mean_contributions.sort_values(
        "difference_K002_minus_K001",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("FEATURE CONTRIBUTION COMPARISON")
print("=" * 70)

print(
    mean_contributions.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# DISPLAY TOP CONTRIBUTING FEATURES
# =====================================================

print("\n" + "=" * 70)
print("TOP FEATURES PUSHING K002 TOWARD FAULTY")
print("=" * 70)

top_features = mean_contributions.head(8)

print(
    top_features[
        [
            "K001",
            "K002",
            "difference_K002_minus_K001"
        ]
    ].to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# VERIFY CONTRIBUTION CALCULATION
# =====================================================

# Logistic Regression decision score is:
#
# intercept + sum(feature contributions)

reconstructed_scores = (
    contributions.sum(axis=1)
    + classifier.intercept_[0]
)

original_scores = pipeline.decision_function(
    df[features]
)

assert np.allclose(
    reconstructed_scores,
    original_scores
)

print(
    "\nContribution calculation verified."
)


# =====================================================
# SAVE RESULTS
# =====================================================

mean_contributions.to_csv(
    OUTPUT_DIR / "k002_feature_contributions.csv"
)

print("\nResults saved successfully.")
print("\nSTEP 4 COMPLETED!")