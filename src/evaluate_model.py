from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = Path("dataset/processed")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/metrics")
PLOT_DIR = Path("results/plots")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD TEST DATA
# ==================================================

print("=" * 60)
print("STAGE 6: FINAL MODEL EVALUATION")
print("=" * 60)

test_df = pd.read_csv(
    DATA_DIR / "test.csv",
    low_memory=False
)

print("\nTest records:", len(test_df))
print("Test bearings:", test_df["bearing_code"].nunique())

print("\nTest class distribution:")
print(test_df["target"].value_counts())


# ==================================================
# LOAD TRAINED MODELS
# ==================================================

model_files = {
    "Logistic Regression": "baseline_logistic_regression.joblib",
    "Random Forest": "random_forest.joblib"
}

results = []
all_predictions = {}
all_bearing_results = []
all_speed_results = []


# ==================================================
# EVALUATE MODELS
# ==================================================

for model_name, filename in model_files.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    bundle = joblib.load(MODEL_DIR / filename)

    model = bundle["pipeline"]
    features = bundle["feature_columns"]

    X_test = test_df[features]
    y_test = test_df["target"]

    if X_test.isna().any().any():
        raise ValueError("Test data contains missing features.")

    if not np.isfinite(
        X_test.to_numpy(dtype=float)
    ).all():
        raise ValueError("Test data contains invalid values.")

    predictions = model.predict(X_test)

    all_predictions[model_name] = predictions

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1
    })

    print(f"\nAccuracy  : {accuracy:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"Macro-F1  : {macro_f1:.6f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Healthy", "Faulty"],
            zero_division=0
        )
    )

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Faulty"],
        yticklabels=["Healthy", "Faulty"]
    )

    plt.xlabel("Predicted Condition")
    plt.ylabel("Actual Condition")
    plt.title(f"{model_name} - Test Confusion Matrix")

    plt.tight_layout()

    plot_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        PLOT_DIR / f"{plot_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # PER-BEARING RESULTS
    # ==================================================

    evaluation_df = test_df.copy()
    evaluation_df["prediction"] = predictions

    print("\nPer-bearing results:")

    for bearing, group in evaluation_df.groupby("bearing_code"):

        bearing_accuracy = accuracy_score(
            group["target"],
            group["prediction"]
        )

        bearing_result = {
            "model": model_name,
            "bearing_code": bearing,
            "actual_class": int(group["target"].iloc[0]),
            "records": len(group),
            "accuracy": bearing_accuracy
        }

        all_bearing_results.append(bearing_result)

        print(
            f"{bearing}: "
            f"Class={bearing_result['actual_class']}, "
            f"Accuracy={bearing_accuracy:.4f}"
        )

    # ==================================================
    # PER-SPEED RESULTS
    # ==================================================

    if "speed_rpm" in evaluation_df.columns:

        print("\nPer-speed results:")

        for speed, group in evaluation_df.groupby("speed_rpm"):

            speed_accuracy = accuracy_score(
                group["target"],
                group["prediction"]
            )

            speed_macro_f1 = f1_score(
                group["target"],
                group["prediction"],
                average="macro",
                zero_division=0
            )

            all_speed_results.append({
                "model": model_name,
                "speed_rpm": speed,
                "records": len(group),
                "accuracy": speed_accuracy,
                "macro_f1": speed_macro_f1
            })

            print(
                f"{speed} RPM: "
                f"Accuracy={speed_accuracy:.4f}, "
                f"Macro-F1={speed_macro_f1:.4f}"
            )


# ==================================================
# SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "final_model_comparison.csv",
    index=False
)

pd.DataFrame(all_bearing_results).to_csv(
    RESULT_DIR / "final_per_bearing_results.csv",
    index=False
)

pd.DataFrame(all_speed_results).to_csv(
    RESULT_DIR / "final_per_speed_results.csv",
    index=False
)

# Save predictions for future analysis.

prediction_df = test_df[
    ["bearing_code", "target"]
].copy()

if "speed_rpm" in test_df.columns:
    prediction_df["speed_rpm"] = test_df["speed_rpm"]

for name, predictions in all_predictions.items():

    column_name = (
        name.lower().replace(" ", "_") + "_prediction"
    )

    prediction_df[column_name] = predictions

prediction_df.to_csv(
    RESULT_DIR / "final_test_predictions.csv",
    index=False
)


# ==================================================
# MODEL COMPARISON GRAPH
# ==================================================

comparison_plot = results_df.set_index("model")[
    ["accuracy", "precision", "recall", "macro_f1"]
]

ax = comparison_plot.plot(
    kind="bar",
    figsize=(10, 6),
    rot=0
)

plt.title("Model Performance on Unseen Test Bearings")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Metric")
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "model_comparison.png",
    dpi=300
)

plt.close()


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FINAL TEST COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nResults saved to:", RESULT_DIR)
print("Graphs saved to:", PLOT_DIR)

print("\nSTAGE 6 COMPLETED!")