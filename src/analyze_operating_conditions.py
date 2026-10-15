from pathlib import Path

import joblib
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


# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 1")
print("OPERATING-CONDITION ANALYSIS")
print("=" * 70)

bundle = joblib.load(MODEL_PATH)

model = bundle["pipeline"]
features = bundle["feature_columns"]

print("\nOriginal Logistic Regression model loaded.")


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    TEST_PATH,
    low_memory=False
)

# Compare only the two healthy test bearings.
df = df[
    df["bearing_code"].isin(["K001", "K002"])
].copy()

if df.empty:
    raise ValueError(
        "K001 and K002 not found in test.csv."
    )

required_columns = [
    "bearing_code",
    "speed_rpm",
    "torque_Nm",
    "force_N",
    "target"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# =====================================================
# GENERATE FAULT SCORES
# =====================================================

df["fault_score"] = model.predict_proba(
    df[features]
)[:, 1]

# Two predefined diagnostic thresholds.
df["prediction_050"] = (
    df["fault_score"] >= 0.50
).astype(int)

df["prediction_085"] = (
    df["fault_score"] >= 0.85
).astype(int)


# =====================================================
# GROUP BY OPERATING CONDITIONS
# =====================================================

condition_columns = [
    "bearing_code",
    "speed_rpm",
    "torque_Nm",
    "force_N"
]

summary = (
    df.groupby(condition_columns)
    .agg(
        windows=("target", "size"),

        mean_fault_score=(
            "fault_score", "mean"
        ),

        median_fault_score=(
            "fault_score", "median"
        ),

        false_alarm_rate_050=(
            "prediction_050", "mean"
        ),

        false_alarm_rate_085=(
            "prediction_085", "mean"
        )
    )
    .reset_index()
)

summary = summary.sort_values(
    [
        "bearing_code",
        "speed_rpm",
        "torque_Nm",
        "force_N"
    ]
)

# Convert rates to percentages.
summary["false_alarm_rate_050"] *= 100
summary["false_alarm_rate_085"] *= 100


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 70)
print("FALSE ALARMS BY OPERATING CONDITION")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# =====================================================
# OVERALL BEARING SUMMARY
# =====================================================

bearing_summary = (
    df.groupby("bearing_code")
    .agg(
        windows=("target", "size"),

        mean_fault_score=(
            "fault_score", "mean"
        ),

        false_alarm_rate_050=(
            "prediction_050", "mean"
        ),

        false_alarm_rate_085=(
            "prediction_085", "mean"
        )
    )
    .reset_index()
)

bearing_summary[
    ["false_alarm_rate_050", "false_alarm_rate_085"]
] *= 100

print("\n" + "=" * 70)
print("OVERALL HEALTHY-BEARING SUMMARY")
print("=" * 70)

print(
    bearing_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# =====================================================
# SAVE RESULTS
# =====================================================

summary.to_csv(
    OUTPUT_DIR / "k002_operating_condition_analysis.csv",
    index=False
)

bearing_summary.to_csv(
    OUTPUT_DIR / "k002_operating_condition_summary.csv",
    index=False
)

print("\nResults saved successfully.")
print("\nSTEP 1 COMPLETED!")