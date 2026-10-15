from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = Path(
    "dataset/processed/test.csv"
)

OUTPUT_DIR = Path("results/metrics")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURES = [
    "vib_rms",
    "vib_std",
    "vib_kurtosis",
    "vib_crest_factor",
    "vib_band_energy_3",
    "vib_band_energy_4"
]

CONDITIONS = [
    "bearing_code",
    "speed_rpm",
    "torque_Nm",
    "force_N"
]


# =====================================================
# LOAD DATA
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 6")
print("CALIBRATION FEASIBILITY ANALYSIS")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

df = df[
    df["bearing_code"].isin(
        ["K001", "K002"]
    )
].copy()


# =====================================================
# CALCULATE FEATURE STABILITY
# =====================================================

results = []

for condition, group in df.groupby(
    CONDITIONS
):

    record = dict(
        zip(
            CONDITIONS,
            condition
        )
    )

    record["windows"] = len(group)

    for feature in FEATURES:

        values = group[
            feature
        ].to_numpy()

        mean_value = np.mean(
            values
        )

        std_value = np.std(
            values
        )

        # Coefficient of variation can be
        # unstable when the mean is near zero.
        # We therefore report mean and std
        # separately rather than relying on CV.

        record[
            f"{feature}_mean"
        ] = mean_value

        record[
            f"{feature}_std"
        ] = std_value

    results.append(
        record
    )


# =====================================================
# CREATE RESULTS
# =====================================================

summary = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("CALIBRATION FEASIBILITY SUMMARY")
print("=" * 70)

display_columns = CONDITIONS + [
    "windows",
    "vib_rms_mean",
    "vib_rms_std",
    "vib_band_energy_3_mean",
    "vib_band_energy_3_std",
    "vib_band_energy_4_mean",
    "vib_band_energy_4_std"
]

print(
    summary[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =====================================================
# SAVE RESULTS
# =====================================================

summary.to_csv(
    OUTPUT_DIR /
    "calibration_feasibility.csv",
    index=False
)

print("\nResults saved successfully.")
print("\nSTEP 6 COMPLETED!")