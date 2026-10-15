from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("dataset/processed")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("AIoT BEARING FAULT DETECTION")
print("STAGE 4: BASELINE MODEL TRAINING")
print("=" * 60)

train_df = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
val_df = pd.read_csv(DATA_DIR / "validation.csv", low_memory=False)
print("\nTraining records:", len(train_df))
print("Validation records:", len(val_df))


# --------------------------------------------------
# SELECT VIBRATION FEATURES
# --------------------------------------------------

feature_columns = [
    col for col in train_df.columns
    if col.startswith("vib_")
]

if not feature_columns:
    raise ValueError("No vibration features found!")

print("\nSelected vibration features:")

for feature in feature_columns:
    print("-", feature)

print("\nTotal features:", len(feature_columns))


# --------------------------------------------------
# PREPARE INPUTS AND TARGETS
# --------------------------------------------------

X_train = train_df[feature_columns]
y_train = train_df["target"]

X_val = val_df[feature_columns]
y_val = val_df["target"]

# Ensure validation data has the same feature columns.

assert list(X_train.columns) == list(X_val.columns)

# Check invalid values before training.

for name, X in [
    ("Training", X_train),
    ("Validation", X_val)
]:
    if X.isna().any().any():
        raise ValueError(f"{name} contains missing features.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains infinite values.")


# --------------------------------------------------
# BUILD MODEL PIPELINE
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# VALIDATION PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

precision = precision_score(
    y_val, y_pred, zero_division=0
)

recall = recall_score(
    y_val, y_pred, zero_division=0
)

macro_f1 = f1_score(
    y_val, y_pred, average="macro"
)

cm = confusion_matrix(
    y_val, y_pred, labels=[0, 1]
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Macro F1  : {macro_f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        target_names=["Healthy", "Faulty"],
        zero_division=0
    )
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "baseline_logistic_regression.joblib"

joblib.dump(
    {
        "pipeline": model,
        "feature_columns": feature_columns,
        "class_names": ["Healthy", "Faulty"]
    },
    model_path
)

print("\nModel saved to:", model_path)


# --------------------------------------------------
# SAVE VALIDATION RESULTS
# --------------------------------------------------

results = pd.DataFrame([{
    "model": "Logistic Regression",
    "dataset": "Validation",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "macro_f1": macro_f1
}])

results.to_csv(
    RESULT_DIR / "baseline_validation_results.csv",
    index=False
)

print("Validation results saved successfully!")

print("\nSTAGE 4 COMPLETED!")