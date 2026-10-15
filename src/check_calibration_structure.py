from pathlib import Path

import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = Path(
    "dataset/processed/test.csv"
)


# =====================================================
# LOAD DATA
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 7")
print("CALIBRATION DATA STRUCTURE CHECK")
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
# SHOW AVAILABLE COLUMNS
# =====================================================

print("\nALL AVAILABLE COLUMNS")
print("-" * 70)

for column in df.columns:
    print(column)


# =====================================================
# FIND POSSIBLE RECORDING IDENTIFIERS
# =====================================================

keywords = [
    "record",
    "file",
    "window",
    "sample",
    "segment",
    "index",
    "time"
]

candidate_columns = [
    column
    for column in df.columns
    if any(
        keyword in column.lower()
        for keyword in keywords
    )
]

print("\n" + "=" * 70)
print("POSSIBLE RECORDING / WINDOW IDENTIFIERS")
print("=" * 70)

if candidate_columns:

    for column in candidate_columns:

        print(
            f"\nColumn: {column}"
        )

        print(
            "Unique values:",
            df[column].nunique()
        )

        print(
            "First 10 values:"
        )

        print(
            df[column].head(10).tolist()
        )

else:

    print(
        "No obvious recording/window "
        "identifier found."
    )


# =====================================================
# SHOW FIRST RECORDS
# =====================================================

print("\n" + "=" * 70)
print("FIRST 10 ROWS - METADATA")
print("=" * 70)

metadata_columns = [
    "bearing_code",
    "speed_rpm",
    "torque_Nm",
    "force_N"
]

metadata_columns += [
    column
    for column in candidate_columns
    if column not in metadata_columns
]

print(
    df[metadata_columns]
    .head(10)
    .to_string(index=False)
)


print("\nSTEP 7 COMPLETED!")