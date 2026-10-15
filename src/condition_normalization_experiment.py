from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed")
OUTPUT_DIR = Path("results/metrics")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

CONDITION_COLUMNS = [
    "speed_rpm",
    "torque_Nm",
    "force_N"
]


# =====================================================
# CONDITION-AWARE NORMALIZER
# =====================================================

class ConditionAwareScaler(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        feature_columns,
        condition_columns
    ):

        self.feature_columns = feature_columns
        self.condition_columns = condition_columns

    def fit(self, X, y=None):

        X = X.copy()

        self.global_scaler_ = StandardScaler()

        self.global_scaler_.fit(
            X[self.feature_columns]
        )

        self.condition_scalers_ = {}

        grouped = X.groupby(
            self.condition_columns
        )

        for condition, group in grouped:

            scaler = StandardScaler()

            scaler.fit(
                group[self.feature_columns]
            )

            self.condition_scalers_[condition] = scaler

        return self

    def transform(self, X):

        X = X.copy()

        output = np.zeros(
            (
                len(X),
                len(self.feature_columns)
            )
        )

        grouped = X.groupby(
            self.condition_columns
        )

        for condition, group in grouped:

            positions = X.index.get_indexer(
                group.index
            )

            scaler = self.condition_scalers_.get(
                condition,
                self.global_scaler_
            )

            output[positions] = scaler.transform(
                group[self.feature_columns]
            )

        return output


# =====================================================
# LOAD DEVELOPMENT DATA
# =====================================================

print("=" * 70)
print("K002 IMPROVEMENT - STEP 2")
print("CONDITION-AWARE NORMALIZATION EXPERIMENT")
print("=" * 70)

train_df = pd.read_csv(
    DATA_DIR / "train.csv",
    low_memory=False
)

val_df = pd.read_csv(
    DATA_DIR / "validation.csv",
    low_memory=False
)

df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

features = [
    col
    for col in df.columns
    if col.startswith("vib_")
]

input_columns = (
    features + CONDITION_COLUMNS
)

print("\nDevelopment records:", len(df))
print("Development bearings:", df["bearing_code"].nunique())
print("Vibration features:", len(features))


# =====================================================
# CREATE MODELS
# =====================================================

def create_global_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ])


def create_condition_model():

    return Pipeline([
        (
            "scaler",
            ConditionAwareScaler(
                feature_columns=features,
                condition_columns=CONDITION_COLUMNS
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ])


# =====================================================
# BEARING-LEVEL FOLDS
# =====================================================

healthy_bearings = sorted(
    df.loc[
        df["target"] == 0,
        "bearing_code"
    ].unique()
)

faulty_bearings = sorted(
    df.loc[
        df["target"] == 1,
        "bearing_code"
    ].unique()
)

rng = np.random.default_rng(
    RANDOM_STATE
)

faulty_groups = np.array_split(
    rng.permutation(faulty_bearings),
    4
)

folds = []

for i in range(4):

    validation_bearings = [
        healthy_bearings[i],
        *faulty_groups[i].tolist()
    ]

    mask = df[
        "bearing_code"
    ].isin(validation_bearings)

    folds.append((
        np.flatnonzero(
            (~mask).to_numpy()
        ),
        np.flatnonzero(
            mask.to_numpy()
        )
    ))


# =====================================================
# RUN CROSS-VALIDATION
# =====================================================

models = {
    "Global_Standardization": create_global_model,
    "Condition_Aware_Standardization": create_condition_model
}

results = []
bearing_results = []

for model_name, model_factory in models.items():

    print("\n" + "=" * 70)
    print("MODEL:", model_name)
    print("=" * 70)

    for fold_number, (train_idx, val_idx) in enumerate(
        folds,
        start=1
    ):

        model = model_factory()

        train_part = df.iloc[train_idx]
        val_part = df.iloc[val_idx]

        if model_name == "Global_Standardization":

            X_train = train_part[features]
            X_val = val_part[features]

        else:

            X_train = train_part[input_columns]
            X_val = val_part[input_columns]

        model.fit(
            X_train,
            train_part["target"]
        )

        probabilities = model.predict_proba(
            X_val
        )[:, 1]

        # Use the threshold selected in Stage 9.
        predictions = (
            probabilities >= 0.85
        ).astype(int)

        actual = val_part[
            "target"
        ].to_numpy()

        accuracy = accuracy_score(
            actual,
            predictions
        )

        macro_f1 = f1_score(
            actual,
            predictions,
            average="macro",
            zero_division=0
        )

        healthy_recall = recall_score(
            actual,
            predictions,
            pos_label=0,
            zero_division=0
        )

        faulty_recall = recall_score(
            actual,
            predictions,
            pos_label=1,
            zero_division=0
        )

        results.append({
            "model": model_name,
            "fold": fold_number,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "healthy_recall": healthy_recall,
            "faulty_recall": faulty_recall
        })

        print(
            f"Fold {fold_number} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro-F1: {macro_f1:.4f} | "
            f"Healthy Recall: {healthy_recall:.4f} | "
            f"Faulty Recall: {faulty_recall:.4f}"
        )

        fold_df = val_part.copy()

        fold_df["prediction"] = predictions

        for bearing, group in fold_df.groupby(
            "bearing_code"
        ):

            bearing_results.append({
                "model": model_name,
                "fold": fold_number,
                "bearing_code": bearing,
                "actual_class": int(
                    group["target"].iloc[0]
                ),
                "accuracy": accuracy_score(
                    group["target"],
                    group["prediction"]
                )
            })


# =====================================================
# CREATE SUMMARY
# =====================================================

results_df = pd.DataFrame(
    results
)

bearing_df = pd.DataFrame(
    bearing_results
)

summary = (
    results_df.groupby("model")
    .agg(
        mean_accuracy=(
            "accuracy", "mean"
        ),
        mean_macro_f1=(
            "macro_f1", "mean"
        ),
        mean_healthy_recall=(
            "healthy_recall", "mean"
        ),
        mean_faulty_recall=(
            "faulty_recall", "mean"
        )
    )
    .reset_index()
)

worst_healthy = (
    bearing_df[
        bearing_df["actual_class"] == 0
    ]
    .groupby("model")["accuracy"]
    .min()
    .reset_index(
        name="worst_healthy_accuracy"
    )
)

summary = summary.merge(
    worst_healthy,
    on="model"
)

summary = summary.sort_values(
    "mean_macro_f1",
    ascending=False
)

print("\n" + "=" * 70)
print("CONDITION NORMALIZATION SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# SAVE RESULTS
# =====================================================

results_df.to_csv(
    OUTPUT_DIR / "condition_normalization_folds.csv",
    index=False
)

bearing_df.to_csv(
    OUTPUT_DIR / "condition_normalization_per_bearing.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "condition_normalization_summary.csv",
    index=False
)

print("\nSTEP 2 COMPLETED!")