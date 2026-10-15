from pathlib import Path

import time
import joblib
import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = Path(
    "models/threshold_optimized_model.joblib"
)

DATA_PATH = Path(
    "dataset/processed/validation.csv"
)

OUTPUT_PATH = Path(
    "results/metrics/fault_detection_replay.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

CONSECUTIVE_FAULT_WINDOWS = 3
MAX_WINDOWS = 30
PLAYBACK_DELAY = 0.1


# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 70)
print("STEP 16 - EDGE FAULT DETECTION")
print("=" * 70)

bundle = joblib.load(MODEL_PATH)

model = bundle["pipeline"]
features = bundle["feature_columns"]
threshold = bundle["decision_threshold"]

print("\nModel loaded successfully.")
print("Features:", len(features))
print("Decision threshold:", threshold)


# =====================================================
# LOAD DEMONSTRATION RECORDING
# =====================================================

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

faulty_df = df[
    df["target"] == 1
].copy()

bearing_code = sorted(
    faulty_df["bearing_code"].unique()
)[0]

bearing_df = faulty_df[
    faulty_df["bearing_code"] == bearing_code
]

filename = sorted(
    bearing_df["filename"].unique()
)[0]

demo_df = (
    bearing_df[
        bearing_df["filename"] == filename
    ]
    .sort_values("window_start")
    .head(MAX_WINDOWS)
    .copy()
)

if demo_df.empty:
    raise ValueError(
        "No recording available for demonstration."
    )

print("\nSelected bearing:", bearing_code)
print("Selected recording:", filename)
print("Windows:", len(demo_df))

print("\nStarting monitoring...\n")


# =====================================================
# INFERENCE LOOP
# =====================================================

consecutive_faults = 0
records = []

for window_number, (_, row) in enumerate(
    demo_df.iterrows(),
    start=1
):

    X = pd.DataFrame(
        [row[features].to_dict()],
        columns=features
    )

    start_time = time.perf_counter()

    fault_probability = float(
        model.predict_proba(X)[0, 1]
    )

    inference_ms = (
        time.perf_counter() - start_time
    ) * 1000

    prediction = int(
        fault_probability >= threshold
    )

    if prediction == 1:
        consecutive_faults += 1
    else:
        consecutive_faults = 0

    alarm_active = (
        consecutive_faults
        >= CONSECUTIVE_FAULT_WINDOWS
    )

    status = (
        "FAULT"
        if prediction == 1
        else "HEALTHY"
    )

    alarm_status = (
        "ALARM"
        if alarm_active
        else "NORMAL"
    )

    print(
        f"Window {window_number:02d} | "
        f"Probability: {fault_probability:.4f} | "
        f"Prediction: {status} | "
        f"Consecutive: {consecutive_faults} | "
        f"Alert: {alarm_status} | "
        f"Inference: {inference_ms:.3f} ms"
    )

    records.append({
        "bearing_code": bearing_code,
        "filename": filename,
        "window_number": window_number,
        "window_start": row["window_start"],
        "fault_probability": fault_probability,
        "prediction": prediction,
        "consecutive_faults": consecutive_faults,
        "alarm_active": alarm_active,
        "inference_ms": inference_ms
    })

    time.sleep(PLAYBACK_DELAY)


# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(records)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("FAULT DETECTION SUMMARY")
print("=" * 70)

print(
    "Total windows:",
    len(results_df)
)

print(
    "Fault predictions:",
    int(results_df["prediction"].sum())
)

print(
    "Alarm-active windows:",
    int(results_df["alarm_active"].sum())
)

print(
    "Average inference time:",
    f"{results_df['inference_ms'].mean():.3f} ms"
)

print("\nResults saved:", OUTPUT_PATH)
print("\nSTEP 16 COMPLETED!")