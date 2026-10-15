from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("dataset/processed")
RESULT_DIR = Path("results/metrics")
MODEL_DIR = Path("models")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 4


# =====================================================
# LOAD DEVELOPMENT DATA
# =====================================================

print("=" * 65)
print("STAGE 8: MODEL GENERALIZATION IMPROVEMENT")
print("=" * 65)

train_df = pd.read_csv(
    DATA_DIR / "train.csv",
    low_memory=False
)

val_df = pd.read_csv(
    DATA_DIR / "validation.csv",
    low_memory=False
)

# Combine only training and validation datasets.
# The test dataset is NOT loaded.

df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

print("\nDevelopment records:", len(df))
print("Development bearings:", df["bearing_code"].nunique())

print("\nClass distribution:")
print(df["target"].value_counts())

print("\nHealthy bearings:")
print(
    sorted(
        df.loc[
            df["target"] == 0,
            "bearing_code"
        ].unique()
    )
)

print("\nFaulty bearings:")
print(
    sorted(
        df.loc[
            df["target"] == 1,
            "bearing_code"
        ].unique()
    )
)


# =====================================================
# SELECT FEATURES
# =====================================================

all_features = [
    column
    for column in df.columns
    if column.startswith("vib_")
]

without_mean = [
    feature
    for feature in all_features
    if feature != "vib_mean"
]

selected_features = [
    "vib_rms",
    "vib_std",
    "vib_kurtosis",
    "vib_skewness",
    "vib_crest_factor",
    "vib_shape_factor",
    "vib_spectral_centroid",
    "vib_spectral_bandwidth",
    "vib_spectral_energy",
    "vib_dominant_freq"
]

# Ensure the selected features exist.

for feature in selected_features:
    if feature not in df.columns:
        raise ValueError(
            f"Missing feature: {feature}"
        )


feature_sets = {
    "All_19_Features": all_features,
    "Without_Vib_Mean": without_mean,
    "Selected_10_Features": selected_features
}

print("\nFeature sets:")

for name, features in feature_sets.items():
    print(f"{name}: {len(features)} features")


# =====================================================
# MODEL DEFINITIONS
# =====================================================

def create_model(model_name):

    if model_name == "Logistic Regression":

        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])

    elif model_name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:
        raise ValueError("Unknown model")


model_names = [
    "Logistic Regression",
    "Random Forest"
]


# =====================================================
# PREPARE CROSS-VALIDATION
# =====================================================

y = df["target"]
groups = df["bearing_code"]

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Build the splits once so every experiment uses
# the same training and validation bearings.

splits = list(
    cv.split(
        df,
        y,
        groups
    )
)

# A fold must contain both healthy and faulty
# bearings to calculate meaningful class metrics.

for fold_number, (train_idx, val_idx) in enumerate(
    splits,
    start=1
):

    train_classes = set(
        y.iloc[train_idx].unique()
    )

    val_classes = set(
        y.iloc[val_idx].unique()
    )

    if train_classes != {0, 1}:
        raise ValueError(
            f"Fold {fold_number} training data "
            "does not contain both classes."
        )

    if val_classes != {0, 1}:
        raise ValueError(
            f"Fold {fold_number} validation data "
            "does not contain both classes."
        )

    train_bearings = set(
        groups.iloc[train_idx]
    )

    val_bearings = set(
        groups.iloc[val_idx]
    )

    assert train_bearings.isdisjoint(
        val_bearings
    )


# =====================================================
# RUN EXPERIMENTS
# =====================================================

fold_results = []
bearing_results = []

for feature_set_name, features in feature_sets.items():

    print("\n" + "=" * 65)
    print("FEATURE SET:", feature_set_name)
    print("=" * 65)

    X = df[features]

    if X.isna().any().any():
        raise ValueError(
            f"Missing values in {feature_set_name}"
        )

    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Invalid values in {feature_set_name}"
        )

    for model_name in model_names:

        print("\nMODEL:", model_name)

        for fold_number, (train_idx, val_idx) in enumerate(
            splits,
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = create_model(model_name)

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(X_val)

            accuracy = accuracy_score(
                y_val,
                predictions
            )

            macro_f1 = f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0
            )

            healthy_recall = recall_score(
                y_val,
                predictions,
                pos_label=0,
                zero_division=0
            )

            faulty_recall = recall_score(
                y_val,
                predictions,
                pos_label=1,
                zero_division=0
            )

            fold_results.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "fold": fold_number,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "healthy_recall": healthy_recall,
                "faulty_recall": faulty_recall,
                "validation_bearings": (
                    df.iloc[val_idx][
                        "bearing_code"
                    ].nunique()
                )
            })

            print(
                f"Fold {fold_number} | "
                f"Accuracy={accuracy:.4f} | "
                f"Macro-F1={macro_f1:.4f} | "
                f"Healthy Recall={healthy_recall:.4f} | "
                f"Fault Recall={faulty_recall:.4f}"
            )

            # Per-bearing results

            fold_df = df.iloc[val_idx].copy()

            fold_df["prediction"] = predictions

            for bearing, group in fold_df.groupby(
                "bearing_code"
            ):

                bearing_results.append({
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "fold": fold_number,
                    "bearing_code": bearing,
                    "actual_class": int(
                        group["target"].iloc[0]
                    ),
                    "records": len(group),
                    "accuracy": accuracy_score(
                        group["target"],
                        group["prediction"]
                    )
                })


# =====================================================
# SAVE FOLD RESULTS
# =====================================================

fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    RESULT_DIR / "cross_validation_folds.csv",
    index=False
)

bearing_df = pd.DataFrame(bearing_results)

bearing_df.to_csv(
    RESULT_DIR / "cross_validation_per_bearing.csv",
    index=False
)


# =====================================================
# SUMMARIZE RESULTS
# =====================================================

summary = (
    fold_df.groupby(
        ["feature_set", "model"]
    )
    .agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_healthy_recall=(
            "healthy_recall", "mean"
        ),
        mean_faulty_recall=(
            "faulty_recall", "mean"
        )
    )
    .reset_index()
)

# Include worst healthy-bearing accuracy.
# This is important because our original model
# failed on one unseen healthy bearing.

healthy_bearing_df = bearing_df[
    bearing_df["actual_class"] == 0
]

worst_healthy = (
    healthy_bearing_df.groupby(
        ["feature_set", "model"]
    )["accuracy"]
    .min()
    .reset_index(
        name="worst_healthy_bearing_accuracy"
    )
)

summary = summary.merge(
    worst_healthy,
    on=["feature_set", "model"],
    how="left"
)

summary = summary.sort_values(
    [
        "mean_macro_f1",
        "worst_healthy_bearing_accuracy"
    ],
    ascending=False
)

print("\n" + "=" * 65)
print("CROSS-VALIDATION SUMMARY")
print("=" * 65)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

summary.to_csv(
    RESULT_DIR / "cross_validation_summary.csv",
    index=False
)


# =====================================================
# SELECT BEST CONFIGURATION
# =====================================================

# Choose the highest mean macro-F1.
# This is a development-only model selection.

best_row = summary.iloc[0]

best_feature_set = best_row["feature_set"]
best_model_name = best_row["model"]

print("\n" + "=" * 65)
print("BEST DEVELOPMENT CONFIGURATION")
print("=" * 65)

print("Feature set:", best_feature_set)
print("Model:", best_model_name)

print(
    "Mean CV Macro-F1:",
    round(best_row["mean_macro_f1"], 4)
)

print(
    "Worst healthy-bearing accuracy:",
    round(
        best_row["worst_healthy_bearing_accuracy"],
        4
    )
)


# =====================================================
# TRAIN FINAL DEVELOPMENT MODEL
# =====================================================

# Train on ALL development bearings.
# Do not include the original test bearings.

final_features = feature_sets[best_feature_set]

final_model = create_model(
    best_model_name
)

final_model.fit(
    df[final_features],
    df["target"]
)

model_bundle = {
    "pipeline": final_model,
    "feature_columns": final_features,
    "class_names": ["Healthy", "Faulty"],
    "model_name": best_model_name,
    "feature_set": best_feature_set
}

model_path = (
    MODEL_DIR / "improved_model_experimental.joblib"
)

joblib.dump(
    model_bundle,
    model_path
)

print("\nExperimental model saved to:", model_path)

print("\nSTAGE 8 COMPLETED!")