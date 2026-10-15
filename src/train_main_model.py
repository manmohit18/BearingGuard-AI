from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DATA_DIR = Path("dataset/processed")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STAGE 5: RANDOM FOREST MODEL")
print("=" * 60)

# Load data
train_df = pd.read_csv(
    DATA_DIR / "train.csv",
    low_memory=False
)

val_df = pd.read_csv(
    DATA_DIR / "validation.csv",
    low_memory=False
)

# Load the previously trained baseline
baseline_bundle = joblib.load(
    MODEL_DIR / "baseline_logistic_regression.joblib"
)

baseline_model = baseline_bundle["pipeline"]
features = baseline_bundle["feature_columns"]

X_train = train_df[features]
y_train = train_df["target"]

X_val = val_df[features]
y_val = val_df["target"]

# Check inputs
for name, X in [("train", X_train), ("validation", X_val)]:
    if X.isna().any().any():
        raise ValueError(f"Missing values in {name}")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid values in {name}")

# Main model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)
print("Training completed.")

# Predictions
baseline_pred = baseline_model.predict(X_val)
rf_pred = model.predict(X_val)


def evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro"
        )
    }


results = pd.DataFrame([
    evaluate(
        "Logistic Regression",
        y_val,
        baseline_pred
    ),
    evaluate(
        "Random Forest",
        y_val,
        rf_pred
    )
])

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

print("\nRANDOM FOREST CONFUSION MATRIX")
print(confusion_matrix(y_val, rf_pred, labels=[0, 1]))

print("\nRANDOM FOREST CLASSIFICATION REPORT")
print(
    classification_report(
        y_val,
        rf_pred,
        target_names=["Healthy", "Faulty"],
        zero_division=0
    )
)

# Save comparison
results.to_csv(
    RESULT_DIR / "model_comparison_validation.csv",
    index=False
)

# Save trained model and feature order
joblib.dump(
    {
        "pipeline": model,
        "feature_columns": features,
        "class_names": ["Healthy", "Faulty"]
    },
    MODEL_DIR / "random_forest.joblib"
)

# Per-bearing evaluation
bearing_results = []

for bearing, group in val_df.groupby("bearing_code"):

    X_group = group[features]
    y_group = group["target"]

    predictions = model.predict(X_group)

    bearing_results.append({
        "bearing_code": bearing,
        "actual_class": int(y_group.iloc[0]),
        "records": len(group),
        "accuracy": accuracy_score(
            y_group, predictions
        )
    })

bearing_df = pd.DataFrame(bearing_results)

print("\nPER-BEARING RESULTS")
print(bearing_df.to_string(index=False))

bearing_df.to_csv(
    RESULT_DIR / "per_bearing_validation.csv",
    index=False
)

# Per-speed evaluation
if "speed_rpm" in val_df.columns:

    speed_results = []

    for speed, group in val_df.groupby("speed_rpm"):

        predictions = model.predict(group[features])

        speed_results.append({
            "speed_rpm": speed,
            "records": len(group),
            "accuracy": accuracy_score(
                group["target"], predictions
            ),
            "macro_f1": f1_score(
                group["target"],
                predictions,
                average="macro",
                zero_division=0
            )
        })

    speed_df = pd.DataFrame(speed_results)

    print("\nPER-SPEED RESULTS")
    print(speed_df.to_string(index=False))

    speed_df.to_csv(
        RESULT_DIR / "per_speed_validation.csv",
        index=False
    )

# Feature importance
importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance_df.head(10).to_string(index=False))

importance_df.to_csv(
    RESULT_DIR / "feature_importance.csv",
    index=False
)

print("\nModel saved successfully.")
print("STAGE 5 COMPLETED!")