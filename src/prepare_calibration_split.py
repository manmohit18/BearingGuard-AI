from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed")

OUTPUT_DIR = DATA_DIR / "calibration"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42

CALIBRATION_FRACTION = 0.25

CONDITION_COLUMNS = [
    "speed_rpm",
    "torque_Nm",
    "force_N"
]


# =====================================================
# LOAD DEVELOPMENT DATA ONLY
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 8")
print("RECORDING-LEVEL CALIBRATION SPLIT")
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

# Healthy development bearings only.
df = df[
    df["target"] == 0
].copy()

print("\nHealthy development bearings:")
print(
    sorted(
        df["bearing_code"].unique()
    )
)

print("\nTotal healthy windows:", len(df))


# =====================================================
# VERIFY REQUIRED COLUMNS
# =====================================================

required_columns = [
    "bearing_code",
    "filename",
    "window_start",
    "target",
    *CONDITION_COLUMNS
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# =====================================================
# SPLIT BY ENTIRE RECORDINGS
# =====================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

calibration_files = []
evaluation_files = []

group_columns = [
    "bearing_code",
    *CONDITION_COLUMNS
]

for group_key, group in df.groupby(
    group_columns
):

    filenames = sorted(
        group["filename"].unique()
    )

    if len(filenames) < 2:
        raise ValueError(
            f"Not enough recordings for {group_key}"
        )

    filenames = rng.permutation(
        filenames
    )

    n_calibration = max(
        1,
        int(
            round(
                len(filenames)
                * CALIBRATION_FRACTION
            )
        )
    )

    # Ensure at least one recording remains
    # for evaluation.
    n_calibration = min(
        n_calibration,
        len(filenames) - 1
    )

    calibration_files.extend(
        filenames[:n_calibration].tolist()
    )

    evaluation_files.extend(
        filenames[n_calibration:].tolist()
    )


# =====================================================
# CREATE DATASETS
# =====================================================

calibration_df = df[
    df["filename"].isin(
        calibration_files
    )
].copy()

evaluation_df = df[
    df["filename"].isin(
        evaluation_files
    )
].copy()


# =====================================================
# SAFETY CHECKS
# =====================================================

calibration_names = set(
    calibration_df["filename"]
)

evaluation_names = set(
    evaluation_df["filename"]
)

assert calibration_names.isdisjoint(
    evaluation_names
), "Recording leakage detected!"

assert len(calibration_df) + len(evaluation_df) == len(df)

assert (
    calibration_df["target"] == 0
).all()

assert (
    evaluation_df["target"] == 0
).all()


# =====================================================
# SORT WINDOWS
# =====================================================

sort_columns = [
    "bearing_code",
    *CONDITION_COLUMNS,
    "filename",
    "window_start"
]

calibration_df = calibration_df.sort_values(
    sort_columns
)

evaluation_df = evaluation_df.sort_values(
    sort_columns
)


# =====================================================
# DISPLAY SUMMARY
# =====================================================

summary_rows = []

for bearing in sorted(
    df["bearing_code"].unique()
):

    cal = calibration_df[
        calibration_df["bearing_code"] == bearing
    ]

    evaluation = evaluation_df[
        evaluation_df["bearing_code"] == bearing
    ]

    summary_rows.append({
        "bearing_code": bearing,
        "calibration_recordings": (
            cal["filename"].nunique()
        ),
        "calibration_windows": len(cal),
        "evaluation_recordings": (
            evaluation["filename"].nunique()
        ),
        "evaluation_windows": len(evaluation)
    })

summary = pd.DataFrame(
    summary_rows
)

print("\n" + "=" * 70)
print("CALIBRATION SPLIT SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)

print("\nRecording overlap:", len(
    calibration_names.intersection(
        evaluation_names
    )
))


# =====================================================
# SAVE DATASETS
# =====================================================

calibration_df.to_csv(
    OUTPUT_DIR / "healthy_calibration.csv",
    index=False
)

evaluation_df.to_csv(
    OUTPUT_DIR / "healthy_evaluation.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "calibration_split_summary.csv",
    index=False
)

print("\nFiles saved successfully.")
print("\nSTEP 8 COMPLETED!")