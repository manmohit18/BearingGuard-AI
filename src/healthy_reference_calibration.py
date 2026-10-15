from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed/calibration")

OUTPUT_DIR = Path("results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_PATH = (
    DATA_DIR / "healthy_calibration.csv"
)

EVALUATION_PATH = (
    DATA_DIR / "healthy_evaluation.csv"
)

FEATURES = [
    "vib_rms",
    "vib_std",
    "vib_kurtosis",
    "vib_crest_factor",
    "vib_band_energy_3",
    "vib_band_energy_4"
]

GROUP_COLUMNS = [
    "bearing_code",
    "speed_rpm",
    "torque_Nm",
    "force_N"
]

# Exploratory thresholds only.
# These are NOT finalized deployment thresholds.
THRESHOLDS = [2.0, 3.0, 4.0, 5.0]


# =====================================================
# LOAD DATA
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 9")
print("HEALTHY-REFERENCE CALIBRATION EXPERIMENT")
print("=" * 70)

calibration_df = pd.read_csv(
    CALIBRATION_PATH,
    low_memory=False
)

evaluation_df = pd.read_csv(
    EVALUATION_PATH,
    low_memory=False
)

print(
    "\nCalibration windows:",
    len(calibration_df)
)

print(
    "Evaluation windows:",
    len(evaluation_df)
)


# =====================================================
# CHECK FOR RECORDING LEAKAGE
# =====================================================

calibration_files = set(
    calibration_df["filename"]
)

evaluation_files = set(
    evaluation_df["filename"]
)

overlap = calibration_files.intersection(
    evaluation_files
)

if overlap:
    raise ValueError(
        "Recording leakage detected!"
    )

print(
    "\nRecording overlap:",
    len(overlap)
)


# =====================================================
# BUILD HEALTHY REFERENCES
# =====================================================

references = {}

for group_key, group in calibration_df.groupby(
    GROUP_COLUMNS
):

    X = group[FEATURES].to_numpy(
        dtype=float
    )

    means = X.mean(axis=0)

    stds = X.std(
        axis=0,
        ddof=0
    )

    # Avoid division by zero for features
    # with extremely small variation.
    stds = np.maximum(
        stds,
        1e-12
    )

    references[group_key] = {
        "mean": means,
        "std": stds
    }

print(
    "\nHealthy reference groups:",
    len(references)
)


# =====================================================
# EVALUATE SEPARATE RECORDINGS
# =====================================================

results = []

for group_key, group in evaluation_df.groupby(
    GROUP_COLUMNS
):

    if group_key not in references:
        raise ValueError(
            f"No reference for {group_key}"
        )

    reference = references[group_key]

    X = group[FEATURES].to_numpy(
        dtype=float
    )

    z_scores = np.abs(
        (
            X - reference["mean"]
        ) / reference["std"]
    )

    # Maximum standardized deviation across
    # the selected vibration features.
    anomaly_scores = z_scores.max(
        axis=1
    )

    for threshold in THRESHOLDS:

        false_alarms = (
            anomaly_scores > threshold
        )

        results.append({
            "bearing_code": group_key[0],
            "speed_rpm": group_key[1],
            "torque_Nm": group_key[2],
            "force_N": group_key[3],
            "threshold": threshold,
            "windows": len(group),
            "false_alarms": int(
                false_alarms.sum()
            ),
            "false_alarm_rate": float(
                false_alarms.mean()
            ),
            "mean_anomaly_score": float(
                anomaly_scores.mean()
            ),
            "median_anomaly_score": float(
                np.median(anomaly_scores)
            )
        })


# =====================================================
# RESULTS
# =====================================================

results_df = pd.DataFrame(results)

summary = (
    results_df.groupby("threshold")
    .agg(
        total_windows=(
            "windows", "sum"
        ),
        total_false_alarms=(
            "false_alarms", "sum"
        )
    )
    .reset_index()
)

summary["false_alarm_rate"] = (
    summary["total_false_alarms"]
    / summary["total_windows"]
)

print("\n" + "=" * 70)
print("OVERALL CALIBRATION RESULTS")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# PER-BEARING SUMMARY
# =====================================================

bearing_summary = (
    results_df.groupby(
        ["bearing_code", "threshold"]
    )
    .agg(
        windows=(
            "windows", "sum"
        ),
        false_alarms=(
            "false_alarms", "sum"
        )
    )
    .reset_index()
)

bearing_summary["false_alarm_rate"] = (
    bearing_summary["false_alarms"]
    / bearing_summary["windows"]
)

print("\n" + "=" * 70)
print("PER-BEARING CALIBRATION RESULTS")
print("=" * 70)

print(
    bearing_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# SAVE RESULTS
# =====================================================

results_df.to_csv(
    OUTPUT_DIR / "calibration_condition_results.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "calibration_overall_summary.csv",
    index=False
)

bearing_summary.to_csv(
    OUTPUT_DIR / "calibration_per_bearing.csv",
    index=False
)

print("\nResults saved successfully.")
print("\nSTEP 9 COMPLETED!")