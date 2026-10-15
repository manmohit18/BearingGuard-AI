from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed")
RESULT_DIR = Path("results/metrics")
MODEL_DIR = Path("models")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = np.round(
    np.arange(0.50, 0.96, 0.05),
    2
)

# Development targets, not industrial standards.
MIN_HEALTHY_RECALL = 0.99
MIN_FAULTY_RECALL = 0.95

RANDOM_STATE = 42


# =====================================================
# LOAD DEVELOPMENT DATA
# =====================================================

print("=" * 65)
print("STAGE 9: FALSE-ALARM REDUCTION")
print("=" * 65)

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
    column
    for column in df.columns
    if column.startswith("vib_")
]

print("\nDevelopment records:", len(df))
print("Development bearings:", df["bearing_code"].nunique())
print("Features:", len(features))


# =====================================================
# CREATE MODEL
# =====================================================

def create_model():

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])


# =====================================================
# CREATE BEARING-LEVEL FOLDS
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

print("\nHealthy development bearings:")
print(healthy_bearings)

print("\nFaulty development bearings:")
print(faulty_bearings)

if len(healthy_bearings) != 4:
    raise ValueError(
        "Expected four healthy development bearings."
    )

if len(faulty_bearings) < 4:
    raise ValueError(
        "At least four faulty development bearings required."
    )

# Assign one healthy bearing to each fold.
# Distribute faulty bearings across the same four folds.

rng = np.random.default_rng(RANDOM_STATE)

shuffled_faulty = rng.permutation(
    faulty_bearings
)

faulty_groups = np.array_split(
    shuffled_faulty,
    4
)

folds = []

for fold_index in range(4):

    validation_bearings = [
        healthy_bearings[fold_index],
        *faulty_groups[fold_index].tolist()
    ]

    validation_mask = df[
        "bearing_code"
    ].isin(validation_bearings)

    training_mask = ~validation_mask

    folds.append((
        np.flatnonzero(training_mask.to_numpy()),
        np.flatnonzero(validation_mask.to_numpy())
    ))

    print(
        f"\nFold {fold_index + 1} "
        f"validation bearings: {validation_bearings}"
    )


# =====================================================
# GENERATE OUT-OF-FOLD FAULT SCORES
# =====================================================

# Each bearing receives predictions from a model
# that was NOT trained using that bearing.

oof_scores = np.full(
    len(df),
    np.nan
)

oof_fold = np.zeros(
    len(df),
    dtype=int
)

for fold_number, (train_idx, val_idx) in enumerate(
    folds,
    start=1
):

    train_bearings = set(
        df.iloc[train_idx]["bearing_code"]
    )

    validation_bearings = set(
        df.iloc[val_idx]["bearing_code"]
    )

    assert train_bearings.isdisjoint(
        validation_bearings
    )

    model = create_model()

    model.fit(
        df.iloc[train_idx][features],
        df.iloc[train_idx]["target"]
    )

    probabilities = model.predict_proba(
        df.iloc[val_idx][features]
    )[:, 1]

    oof_scores[val_idx] = probabilities
    oof_fold[val_idx] = fold_number

    print(
        f"\nFold {fold_number} completed."
    )

if np.isnan(oof_scores).any():
    raise RuntimeError(
        "Some development rows did not receive predictions."
    )

print("\nOut-of-fold predictions completed.")


# =====================================================
# SAVE OUT-OF-FOLD PREDICTIONS
# =====================================================

oof_df = df[[
    "bearing_code",
    "target",
    "speed_rpm"
]].copy()

oof_df["fold"] = oof_fold
oof_df["fault_score"] = oof_scores

oof_df.to_csv(
    RESULT_DIR / "threshold_oof_predictions.csv",
    index=False
)


# =====================================================
# EVALUATE THRESHOLDS
# =====================================================

results = []
bearing_results = []

for threshold in THRESHOLDS:

    predictions = (
        oof_scores >= threshold
    ).astype(int)

    actual = df["target"].to_numpy()

    healthy_mask = actual == 0
    faulty_mask = actual == 1

    healthy_recall = recall_score(
        actual[healthy_mask],
        predictions[healthy_mask],
        pos_label=0,
        zero_division=0
    )

    faulty_recall = recall_score(
        actual[faulty_mask],
        predictions[faulty_mask],
        pos_label=1,
        zero_division=0
    )

    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(
        actual,
        predictions
    )

    false_alarms = int(
        np.sum(
            (actual == 0) &
            (predictions == 1)
        )
    )

    missed_faults = int(
        np.sum(
            (actual == 1) &
            (predictions == 0)
        )
    )

    # Calculate performance separately for each bearing.

    per_bearing_healthy = []

    for bearing in healthy_bearings:

        mask = (
            df["bearing_code"].to_numpy()
            == bearing
        )

        bearing_accuracy = accuracy_score(
            actual[mask],
            predictions[mask]
        )

        per_bearing_healthy.append(
            bearing_accuracy
        )

        bearing_results.append({
            "threshold": threshold,
            "bearing_code": bearing,
            "actual_class": 0,
            "accuracy": bearing_accuracy
        })

    worst_healthy_accuracy = min(
        per_bearing_healthy
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "healthy_recall": healthy_recall,
        "faulty_recall": faulty_recall,
        "worst_healthy_accuracy": (
            worst_healthy_accuracy
        ),
        "false_alarm_windows": false_alarms,
        "missed_fault_windows": missed_faults
    })


results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "threshold_comparison.csv",
    index=False
)

pd.DataFrame(
    bearing_results
).to_csv(
    RESULT_DIR / "threshold_per_healthy_bearing.csv",
    index=False
)


# =====================================================
# SELECT THRESHOLD
# =====================================================

# Keep thresholds that satisfy both development
# recall targets.

eligible = results_df[
    (results_df["healthy_recall"] >= MIN_HEALTHY_RECALL) &
    (results_df["faulty_recall"] >= MIN_FAULTY_RECALL)
].copy()

print("\n" + "=" * 65)
print("THRESHOLD COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

if eligible.empty:

    print(
        "\nWARNING: No threshold satisfies "
        "both recall targets."
    )

    # Preserve the original default threshold rather
    # than silently accepting an unsafe trade-off.
    best_threshold = 0.50

    print(
        "Keeping the default threshold:",
        best_threshold
    )

else:

    # Prioritize the weakest healthy bearing,
    # then overall balanced classification performance.
    eligible = eligible.sort_values(
        [
            "worst_healthy_accuracy",
            "macro_f1",
            "threshold"
        ],
        ascending=[False, False, True]
    )

    best_threshold = float(
        eligible.iloc[0]["threshold"]
    )

    print(
        "\nSelected development threshold:",
        best_threshold
    )


# =====================================================
# TRAIN MODEL ON ALL DEVELOPMENT BEARINGS
# =====================================================

final_model = create_model()

final_model.fit(
    df[features],
    df["target"]
)

model_bundle = {
    "pipeline": final_model,
    "feature_columns": features,
    "class_names": ["Healthy", "Faulty"],
    "decision_threshold": best_threshold,
    "model_name": "Logistic Regression",
    "selection_method": "Bearing-level OOF threshold experiment"
}

model_path = (
    MODEL_DIR / "threshold_optimized_model.joblib"
)

joblib.dump(
    model_bundle,
    model_path
)

print("\nModel saved to:", model_path)

print("\nSTAGE 9 COMPLETED!")