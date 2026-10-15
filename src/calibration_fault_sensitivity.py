from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed")

CALIBRATION_PATH = (
    DATA_DIR
    / "calibration"
    / "healthy_calibration.csv"
)

OUTPUT_DIR = Path("results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "vib_rms",
    "vib_std",
    "vib_kurtosis",
    "vib_crest_factor",
    "vib_band_energy_3",
    "vib_band_energy_4"
]

CONDITIONS = [
    "speed_rpm",
    "torque_Nm",
    "force_N"
]

THRESHOLDS = [2.0, 3.0, 4.0, 5.0]


# =====================================================
# LOAD DEVELOPMENT DATA
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 10")
print("CALIBRATION FAULT SENSITIVITY")
print("=" * 70)

calibration_df = pd.read_csv(
    CALIBRATION_PATH,
    low_memory=False
)

train_df = pd.read_csv(
    DATA_DIR / "train.csv",
    low_memory=False
)

validation_df = pd.read_csv(
    DATA_DIR / "validation.csv",
    low_memory=False
)

development_df = pd.concat(
    [train_df, validation_df],
    ignore_index=True
)

faulty_df = development_df[
    development_df["target"] == 1
].copy()

print(
    "\nHealthy calibration windows:",
    len(calibration_df)
)

print(
    "Faulty development windows:",
    len(faulty_df)
)

print(
    "Faulty bearings:",
    faulty_df["bearing_code"].nunique()
)


# =====================================================
# BUILD CONDITION-MATCHED REFERENCES
# =====================================================

references = {}

group_columns = [
    "bearing_code",
    *CONDITIONS
]

for group_key, group in calibration_df.groupby(
    group_columns
):

    bearing = group_key[0]
    condition = tuple(
        group_key[1:]
    )

    X = group[FEATURES].to_numpy(
        dtype=float
    )

    means = X.mean(axis=0)

    stds = np.maximum(
        X.std(axis=0, ddof=0),
        1e-12
    )

    references.setdefault(
        condition,
        []
    ).append({
        "bearing": bearing,
        "mean": means,
        "std": stds
    })


# =====================================================
# SCORE FAULTY WINDOWS
# =====================================================

all_scores = []

for condition, group in faulty_df.groupby(
    CONDITIONS
):

    condition = tuple(
        condition
    )

    if condition not in references:
        raise ValueError(
            f"No healthy references for {condition}"
        )

    X = group[FEATURES].to_numpy(
        dtype=float
    )

    reference_scores = []

    for reference in references[condition]:

        z_scores = np.abs(
            (
                X - reference["mean"]
            ) / reference["std"]
        )

        # Largest feature deviation.
        scores = z_scores.max(
            axis=1
        )

        reference_scores.append(
            scores
        )

    # Use the closest healthy reference.
    #
    # A fault is considered anomalous only
    # if it differs from every reference.

    anomaly_scores = np.min(
        np.vstack(
            reference_scores
        ),
        axis=0
    )

    temp = group[
        ["bearing_code", "filename"]
    ].copy()

    temp["anomaly_score"] = (
        anomaly_scores
    )

    all_scores.append(
        temp
    )


scored_df = pd.concat(
    all_scores,
    ignore_index=True
)


# =====================================================
# EVALUATE FAULT DETECTION
# =====================================================

results = []

for threshold in THRESHOLDS:

    detected = (
        scored_df["anomaly_score"]
        > threshold
    )

    results.append({
        "threshold": threshold,
        "faulty_windows": len(scored_df),
        "detected_faulty_windows": int(
            detected.sum()
        ),
        "missed_faulty_windows": int(
            (~detected).sum()
        ),
        "faulty_recall": float(
            detected.mean()
        )
    })


summary = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("FAULT SENSITIVITY SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# PER-BEARING RESULTS
# =====================================================

bearing_results = []

for threshold in THRESHOLDS:

    for bearing, group in scored_df.groupby(
        "bearing_code"
    ):

        detected = (
            group["anomaly_score"]
            > threshold
        )

        bearing_results.append({
            "threshold": threshold,
            "bearing_code": bearing,
            "faulty_windows": len(group),
            "detected_windows": int(
                detected.sum()
            ),
            "faulty_recall": float(
                detected.mean()
            )
        })


bearing_summary = pd.DataFrame(
    bearing_results
)

print("\n" + "=" * 70)
print("PER-BEARING FAULT SENSITIVITY")
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

summary.to_csv(
    OUTPUT_DIR
    / "calibration_fault_sensitivity_summary.csv",
    index=False
)

bearing_summary.to_csv(
    OUTPUT_DIR
    / "calibration_fault_sensitivity_per_bearing.csv",
    index=False
)

scored_df.to_csv(
    OUTPUT_DIR
    / "calibration_fault_anomaly_scores.csv",
    index=False
)

print("\nResults saved successfully.")
print("\nSTEP 10 COMPLETED!")