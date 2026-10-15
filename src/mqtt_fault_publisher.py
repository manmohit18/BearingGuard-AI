from pathlib import Path
import json
import time

import joblib
import pandas as pd
import paho.mqtt.client as mqtt


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = Path(
    "models/threshold_optimized_model.joblib"
)

DATA_PATH = Path(
    "dataset/processed/validation.csv"
)

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "bearing/monitoring"

CONSECUTIVE_FAULT_WINDOWS = 3

MAX_WINDOWS = 30
PUBLISH_INTERVAL = 1.0


# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 70)
print("STEP 18 - MQTT FAULT DETECTION PUBLISHER")
print("=" * 70)

bundle = joblib.load(MODEL_PATH)

model = bundle["pipeline"]
features = bundle["feature_columns"]
threshold = bundle["decision_threshold"]

print("\nModel loaded successfully.")
print("Decision threshold:", threshold)


# =====================================================
# LOAD DEMONSTRATION RECORDING
# =====================================================

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

# =====================================================
# SELECT BEARING FOR DEMONSTRATION
# =====================================================

# Choose "KA04" for faulty or "K006" for healthy
bearing_code = "KA04"

bearing_df = df[
    df["bearing_code"] == bearing_code
].copy()

if bearing_df.empty:
    raise ValueError(
        f"Bearing {bearing_code} not found."
    )

# Verify the selected bearing has a consistent label
if bearing_df["target"].nunique() != 1:
    raise ValueError(
        f"Bearing {bearing_code} contains mixed labels."
    )

actual_condition = (
    "FAULTY"
    if int(bearing_df["target"].iloc[0]) == 1
    else "HEALTHY"
)

print("Actual bearing condition:", actual_condition)

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
        "No demonstration data found."
    )

print("\nSelected bearing:", bearing_code)
print("Selected recording:", filename)
print("Windows:", len(demo_df))


# =====================================================
# CONNECT TO MQTT BROKER
# =====================================================

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="bearing_edge_publisher"
)

print("\nConnecting to MQTT broker...")

try:
    client.connect(
        MQTT_BROKER,
        MQTT_PORT,
        keepalive=60
    )
    client.loop_start()

except Exception as error:
    print("\nMQTT CONNECTION FAILED!")
    print(error)
    print(
        "Check that Mosquitto is running "
        "on localhost:1883."
    )
    raise SystemExit(1)

print("MQTT connection initiated.")


# =====================================================
# EDGE INFERENCE AND MQTT PUBLISHING
# =====================================================

consecutive_faults = 0

try:

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

        message = {
            "device_id": "edge_device_01",
            "bearing_code": bearing_code,
            "filename": filename,
            "window_number": window_number,
            "window_start": int(
                row["window_start"]
            ),
            "fault_probability": round(
                fault_probability,
                6
            ),
            "prediction": (
                "FAULT"
                if prediction == 1
                else "HEALTHY"
            ),
            "consecutive_faults": (
                consecutive_faults
            ),
            "alarm_active": bool(
                alarm_active
            ),
            "inference_ms": round(
                inference_ms,
                3
            ),
            "timestamp": pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
        }

        payload = json.dumps(message)

        publish_result = client.publish(
            MQTT_TOPIC,
            payload,
            qos=1
        )

        publish_result.wait_for_publish(
            timeout=5
        )

        if publish_result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                "MQTT publish failed: "
                + mqtt.error_string(
                    publish_result.rc
                )
            )

        status = (
            "ALARM"
            if alarm_active
            else "NORMAL"
        )

        print(
            f"Window {window_number:02d} | "
            f"Probability: {fault_probability:.4f} | "
            f"Prediction: {message['prediction']} | "
            f"Alert: {status} | "
            f"MQTT: SENT"
        )

        time.sleep(
            PUBLISH_INTERVAL
        )

finally:

    client.loop_stop()
    client.disconnect()


print("\n" + "=" * 70)
print("MQTT PUBLISHING COMPLETED")
print("=" * 70)

print("Topic:", MQTT_TOPIC)
print("Total windows:", len(demo_df))
print("\nSTEP 18 COMPLETED!")