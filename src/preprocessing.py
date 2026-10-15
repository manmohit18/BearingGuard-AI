from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path(
    "dataset/raw/paderborn_features_raw_no_overlap.csv"
)

OUTPUT_DIR = Path("dataset/processed")

RANDOM_STATE = 42


def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("Dataset shape:", df.shape)

    print("\nCondition distribution:")
    print(df["label"].value_counts())

    print("\nUnique bearings:")
    print(df["bearing_code"].nunique())

    return df


def prepare_binary_dataset(df):

    print("\nPreparing Healthy vs Real Damage dataset...")

    df = df[
        df["label"].isin(["Healthy", "Real"])
    ].copy()

    df["target"] = df["label"].map({
        "Healthy": 0,
        "Real": 1
    })

    print("\nBinary classification distribution:")
    print(df["target"].value_counts())

    return df


def split_by_bearing(df):

    print("\nCreating bearing-level splits...")

    bearing_info = (
        df[["bearing_code", "target"]]
        .drop_duplicates()
    )

    train_bearings = []
    val_bearings = []
    test_bearings = []

    for target in [0, 1]:

        bearings = bearing_info[
            bearing_info["target"] == target
        ]["bearing_code"].to_numpy()

        train_ids, remaining_ids = train_test_split(
            bearings,
            test_size=0.4,
            random_state=RANDOM_STATE
        )

        val_ids, test_ids = train_test_split(
            remaining_ids,
            test_size=0.5,
            random_state=RANDOM_STATE
        )

        train_bearings.extend(train_ids)
        val_bearings.extend(val_ids)
        test_bearings.extend(test_ids)

    train_df = df[
        df["bearing_code"].isin(train_bearings)
    ].copy()

    val_df = df[
        df["bearing_code"].isin(val_bearings)
    ].copy()

    test_df = df[
        df["bearing_code"].isin(test_bearings)
    ].copy()

    # Verify no bearing appears in multiple partitions.

    assert set(train_bearings).isdisjoint(val_bearings)
    assert set(train_bearings).isdisjoint(test_bearings)
    assert set(val_bearings).isdisjoint(test_bearings)

    return train_df, val_df, test_df


def save_datasets(train_df, val_df, test_df):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print("\nDatasets saved successfully!")


def main():

    df = load_dataset()

    df = prepare_binary_dataset(df)

    train_df, val_df, test_df = split_by_bearing(df)

    print("\n" + "=" * 50)
    print("DATASET SPLIT SUMMARY")
    print("=" * 50)

    for name, data in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df)
    ]:

        print(f"\n{name} Dataset:")
        print("Records:", len(data))
        print("Bearings:", data["bearing_code"].nunique())
        print("Classes:")
        print(data["target"].value_counts())

    save_datasets(train_df, val_df, test_df)

    print("\nSTAGE 3 DATA PREPARATION COMPLETED!")


if __name__ == "__main__":
    main()