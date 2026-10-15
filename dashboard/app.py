import json
import threading
import time
from collections import deque
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import paho.mqtt.client as mqtt


# ============================================================
# CONFIGURATION
# ============================================================

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "bearing/monitoring"

MAX_HISTORY = 500
FAULT_THRESHOLD = 0.85
FINAL_WINDOW = 30


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BearingGuard AI | Industrial Monitoring",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODERN DARK THEME
# ============================================================

st.markdown(
    """
    <style>

    /* Global background */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(18, 73, 110, 0.22),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #080d18 0%,
                #0d1424 55%,
                #090e19 100%
            );

        color: #e9f0ff;
    }

    /* Main content */

    .block-container {
    padding-top: 5rem !important;
    padding-bottom: 2rem;
    max-width: 1600px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

    /* Sidebar */

    [data-testid="stSidebar"] {
        background: #0c1322;
        border-right: 1px solid #243147;
    }

    /* Hide default Streamlit decorations */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Branding */

    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #ffffff;
        margin-bottom: 0;
    }

    .brand-accent {
        color: #38bdf8;
    }

    .brand-subtitle {
        color: #91a3bc;
        font-size: 0.93rem;
        margin-top: 4px;
        margin-bottom: 25px;
    }

    /* Section headings */

    .section-title {
        color: #dce9fb;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 24px 0 14px 0;
    }

    /* Metric cards */

    .metric-card {
        background: linear-gradient(
            145deg,
            #141e30,
            #101827
        );
        border: 1px solid #25334a;
        border-radius: 16px;
        padding: 22px;
        min-height: 145px;
        transition: all 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #38bdf8;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .metric-label {
        color: #8da0ba;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .metric-subtitle {
        color: #8294ad;
        font-size: 0.78rem;
        margin-top: 8px;
    }

    .healthy-value {
        color: #34d399;
    }

    .fault-value {
        color: #fb7185;
    }

    .cyan-value {
        color: #38bdf8;
    }

    /* Status pills */

    .status-pill {
        display: inline-block;
        padding: 7px 15px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.4px;
    }

    .status-online {
        background: rgba(16,185,129,0.12);
        color: #34d399;
        border: 1px solid rgba(16,185,129,0.3);
    }

    .status-offline {
        background: rgba(239,68,68,0.12);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.3);
    }

    /* Alarm panels */

    .alarm-panel {
        background: linear-gradient(
            120deg,
            rgba(127,29,29,0.45),
            rgba(69,10,10,0.35)
        );
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 22px 26px;
        margin: 15px 0 20px 0;
        animation: alarmPulse 2s infinite;
    }

    @keyframes alarmPulse {
        0% {
            box-shadow: 0 0 0 rgba(239,68,68,0);
        }
        50% {
            box-shadow: 0 0 22px rgba(239,68,68,0.23);
        }
        100% {
            box-shadow: 0 0 0 rgba(239,68,68,0);
        }
    }

    .alarm-title {
        color: #fda4af;
        font-size: 1.25rem;
        font-weight: 800;
    }

    .alarm-text {
        color: #fecdd3;
        font-size: 0.91rem;
        margin-top: 6px;
    }

    .healthy-panel {
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(52,211,153,0.35);
        border-radius: 16px;
        padding: 22px 26px;
        margin: 15px 0 20px 0;
    }

    .healthy-title {
        color: #6ee7b7;
        font-size: 1.25rem;
        font-weight: 800;
    }

    .healthy-text {
        color: #a7f3d0;
        font-size: 0.91rem;
        margin-top: 6px;
    }

    /* Dashboard panels */

    .dashboard-panel {
        background: #111b2b;
        border: 1px solid #25334a;
        border-radius: 16px;
        padding: 18px;
    }

    /* Sidebar section */

    .sidebar-heading {
        color: #38bdf8;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 1.4px;
        margin-top: 24px;
        margin-bottom: 12px;
    }

    /* Streamlit tabs */

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #111b2b;
        border-radius: 10px;
        color: #a4b5ca;
        padding: 10px 18px;
    }

    .stTabs [aria-selected="true"] {
        background: #12354b;
        color: #38bdf8;
    }

    /* Dataframe */

    [data-testid="stDataFrame"] {
        border: 1px solid #26364e;
        border-radius: 12px;
    }

    /* Buttons */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MQTT MONITOR
# ============================================================

class MQTTMonitor:

    def __init__(self):

        self.lock = threading.Lock()

        self.messages = deque(maxlen=MAX_HISTORY)

        self.connected = False
        self.error = None

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="bearingguard_dashboard_main",
            reconnect_on_failure=True,
        )

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

        try:

            self.client.connect(
                MQTT_BROKER,
                MQTT_PORT,
                keepalive=60,
            )

            self.client.loop_start()

        except Exception as error:

            self.error = str(error)

    def on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties,
    ):

        with self.lock:

            self.connected = reason_code == 0

            if self.connected:
                self.error = None
            else:
                self.error = str(reason_code)

        if reason_code == 0:

            client.subscribe(MQTT_TOPIC, qos=1)

    def on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties,
    ):

        with self.lock:
            self.connected = False

    def on_message(
        self,
        client,
        userdata,
        message,
    ):

        try:

            payload = json.loads(
                message.payload.decode("utf-8")
            )

            payload["received_at"] = (
                datetime.now().strftime("%H:%M:%S")
            )

            with self.lock:
                self.messages.append(payload)

        except Exception as error:

            with self.lock:
                self.error = str(error)

    def snapshot(self):

        with self.lock:

            return {
                "connected": self.connected,
                "error": self.error,
                "messages": list(self.messages),
            }


@st.cache_resource
def get_monitor():

    return MQTTMonitor()


monitor = get_monitor()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:10px 0 20px 0;">
            <div style="
                font-size:1.5rem;
                font-weight:800;
                color:white;
            ">
                ⚙️ BearingGuard
                <span style="color:#38bdf8;">AI</span>
            </div>
            <div style="
                color:#8193ad;
                font-size:0.78rem;
                margin-top:5px;
            ">
                PREDICTIVE MAINTENANCE PLATFORM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">MONITORING</div>',
        unsafe_allow_html=True,
    )

    st.write("📊 Live Dashboard")
    st.write("📈 Analytics")
    st.write("📋 Event History")

    st.markdown(
        '<div class="sidebar-heading">CONFIGURATION</div>',
        unsafe_allow_html=True,
    )

    st.info(
        f"MQTT Broker\n\n"
        f"{MQTT_BROKER}:{MQTT_PORT}"
    )

    st.caption(f"Topic: {MQTT_TOPIC}")

    st.markdown(
        '<div class="sidebar-heading">ALARM SETTINGS</div>',
        unsafe_allow_html=True,
    )

    sound_enabled = st.toggle(
        "Enable warning sound",
        value=True,
    )

    st.caption(
        "The 10-second warning beep is available "
        "when a faulty recording reaches window 30."
    )

    st.markdown(
        '<div class="sidebar-heading">MODEL INFORMATION</div>',
        unsafe_allow_html=True,
    )

    st.write("**Model:** Logistic Regression")
    st.write("**Decision threshold:** 0.85")
    st.write("**Alert rule:** 3 consecutive faults")
    st.write("**Dataset:** Paderborn")

    st.divider()

    st.caption(
        "BearingGuard AI · Edge AIoT Prototype"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="brand-title">
        Industrial <span class="brand-accent">Monitoring</span>
    </div>
    <div class="brand-subtitle">
        Real-time bearing health intelligence
        powered by lightweight Edge AI and MQTT
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHART THEME
# ============================================================

def apply_chart_theme(fig, height=350):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a8bad0"),
        height=height,
        margin=dict(l=15, r=15, t=25, b=20),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.10)",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.10)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0,
        ),
    )

    return fig


# ============================================================
# METRIC CARD
# ============================================================

def metric_card(label, value, subtitle, color_class=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">
                {value}
            </div>
            <div class="metric-subtitle">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# AUDIO COMPONENT
# ============================================================

def warning_audio():

    components.html(
        """
        <div style="
            background:#30121c;
            border:1px solid #ef4444;
            border-radius:12px;
            padding:14px;
            font-family:Arial,sans-serif;
            color:white;
        ">

            <div style="
                font-weight:bold;
                margin-bottom:10px;
            ">
                🔊 Critical Fault Warning
            </div>

            <button id="playAlarm" style="
                background:#ef4444;
                border:0;
                color:white;
                padding:12px 18px;
                border-radius:8px;
                font-weight:bold;
                cursor:pointer;
            ">
                ▶ Play 10-Second Alarm
            </button>

            <span id="alarmStatus" style="
                margin-left:12px;
                font-size:13px;
                color:#fda4af;
            "></span>

        </div>

        <script>

        const button = document.getElementById("playAlarm");
        const status = document.getElementById("alarmStatus");

        let playing = false;

        button.addEventListener("click", async () => {

            if (playing) return;

            playing = true;

            button.disabled = true;
            button.textContent = "🚨 Alarm Playing";

            const AudioContextClass =
                window.AudioContext ||
                window.webkitAudioContext;

            const context = new AudioContextClass();

            await context.resume();

            const oscillator = context.createOscillator();
            const gain = context.createGain();

            oscillator.type = "square";
            oscillator.frequency.value = 850;

            gain.gain.value = 0.05;

            oscillator.connect(gain);
            gain.connect(context.destination);

            oscillator.start();

            status.textContent = "10-second warning active";

            setTimeout(async () => {

                oscillator.stop();

                await context.close();

                status.textContent = "Alarm completed";

                button.textContent = "✓ Alarm Completed";

            }, 10000);

        });

        </script>
        """,
        height=120,
    )


# ============================================================
# LIVE DASHBOARD
# ============================================================

@st.fragment(run_every="1s")
def live_dashboard():

    snapshot = monitor.snapshot()

    messages = snapshot["messages"]
    connected = snapshot["connected"]

    # --------------------------------------------------------
    # CONNECTION STATUS
    # --------------------------------------------------------

    top_left, top_right = st.columns([3, 1])

    with top_left:

        st.markdown(
            '<div class="section-title">SYSTEM OVERVIEW</div>',
            unsafe_allow_html=True,
        )

    with top_right:

        if connected:

            st.markdown(
                '<span class="status-pill status-online">'
                '● MQTT CONNECTED</span>',
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<span class="status-pill status-offline">'
                '● MQTT DISCONNECTED</span>',
                unsafe_allow_html=True,
            )

    if snapshot["error"]:
        st.warning(snapshot["error"])

    if not messages:

        st.info(
            "Waiting for live MQTT data. "
            "Start the Python edge publisher."
        )

        return

    latest = messages[-1]

    bearing = latest.get("bearing_code", "UNKNOWN")
    prediction = latest.get("prediction", "UNKNOWN")

    probability = float(
        latest.get("fault_probability", 0)
    )

    inference_ms = float(
        latest.get("inference_ms", 0)
    )

    alarm_active = bool(
        latest.get("alarm_active", False)
    )

    window_number = int(
        latest.get("window_number", 0)
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Active Bearing",
            bearing,
            "Currently monitored unit",
            "cyan-value",
        )

    with col2:
        metric_card(
            "Bearing Status",
            prediction,
            "Latest AI classification",
            "fault-value"
            if prediction == "FAULT"
            else "healthy-value",
        )

    with col3:
        metric_card(
            "Fault Probability",
            f"{probability * 100:.2f}%",
            "Logistic Regression output",
            "fault-value"
            if probability >= FAULT_THRESHOLD
            else "healthy-value",
        )

    with col4:
        metric_card(
            "Inference Time",
            f"{inference_ms:.2f} ms",
            "Latest model inference",
            "cyan-value",
        )

    # --------------------------------------------------------
    # HEALTH / FAULT ALERT
    # --------------------------------------------------------

    if alarm_active:

        st.markdown(
            """
            <div class="alarm-panel">
                <div class="alarm-title">
                    🚨 CRITICAL BEARING FAULT DETECTED
                </div>
                <div class="alarm-text">
                    Three or more consecutive fault
                    predictions detected.
                    Immediate inspection recommended.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif prediction == "FAULT":

        st.warning(
            "⚠️ Fault prediction detected. "
            "Waiting for consecutive-window confirmation."
        )

    else:

        st.markdown(
            """
            <div class="healthy-panel">
                <div class="healthy-title">
                    ✅ BEARING OPERATING NORMALLY
                </div>
                <div class="healthy-text">
                    No active fault alarm.
                    Latest vibration features are
                    classified as healthy.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # AUDIO ALARM AT FINAL WINDOW
    # --------------------------------------------------------

    if (
        sound_enabled
        and alarm_active
        and window_number == FINAL_WINDOW
    ):

        warning_audio()

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    history = pd.DataFrame(messages)

    history["fault_probability"] = pd.to_numeric(
        history["fault_probability"],
        errors="coerce",
    )

    history["inference_ms"] = pd.to_numeric(
        history["inference_ms"],
        errors="coerce",
    )

    history["message_index"] = range(
        1,
        len(history) + 1,
    )

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Live Analytics",
            "⚡ Performance",
            "📋 Event History",
        ]
    )

    # --------------------------------------------------------
    # LIVE ANALYTICS
    # --------------------------------------------------------

    with tab1:

        st.markdown(
            '<div class="section-title">'
            'FAULT PROBABILITY TREND'
            '</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=history["message_index"],
                y=history["fault_probability"],
                mode="lines+markers",
                name="Fault Probability",
                line=dict(
                    color="#38bdf8",
                    width=3,
                ),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.08)",
            )
        )

        fig.add_hline(
            y=FAULT_THRESHOLD,
            line_dash="dash",
            line_color="#fb7185",
            annotation_text="Fault Threshold",
        )

        fig.update_yaxes(
            range=[0, 1.05],
            title="Fault Probability",
        )

        fig.update_xaxes(
            title="Received Message",
        )

        apply_chart_theme(fig, height=380)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Messages Received",
                len(history),
            )

        with b:
            st.metric(
                "Current Window",
                window_number,
            )

        with c:
            st.metric(
                "Alarm Status",
                "🚨 ACTIVE"
                if alarm_active
                else "NORMAL",
            )

    # --------------------------------------------------------
    # PERFORMANCE TAB
    # --------------------------------------------------------

    with tab2:

        st.markdown(
            '<div class="section-title">'
            'EDGE INFERENCE PERFORMANCE'
            '</div>',
            unsafe_allow_html=True,
        )

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=history["message_index"],
                y=history["inference_ms"],
                mode="lines+markers",
                name="Inference Latency",
                line=dict(
                    color="#a78bfa",
                    width=2,
                ),
            )
        )

        fig2.update_xaxes(
            title="Received Message",
        )

        fig2.update_yaxes(
            title="Inference Time (ms)",
        )

        apply_chart_theme(fig2, height=350)

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )

        x, y, z = st.columns(3)

        with x:
            st.metric(
                "Average Inference",
                f"{history['inference_ms'].mean():.2f} ms",
            )

        with y:
            st.metric(
                "Minimum Inference",
                f"{history['inference_ms'].min():.2f} ms",
            )

        with z:
            st.metric(
                "Maximum Inference",
                f"{history['inference_ms'].max():.2f} ms",
            )

    # --------------------------------------------------------
    # EVENT HISTORY TAB
    # --------------------------------------------------------

    with tab3:

        st.markdown(
            '<div class="section-title">'
            'MONITORING EVENT HISTORY'
            '</div>',
            unsafe_allow_html=True,
        )

        filter_option = st.selectbox(
            "Filter predictions",
            ["All", "FAULT", "HEALTHY"],
            key="history_filter",
        )

        filtered = history.copy()

        if filter_option != "All":

            filtered = filtered[
                filtered["prediction"] == filter_option
            ]

        columns = [
            "received_at",
            "bearing_code",
            "window_number",
            "fault_probability",
            "prediction",
            "consecutive_faults",
            "alarm_active",
            "inference_ms",
        ]

        available = [
            c for c in columns
            if c in filtered.columns
        ]

        st.dataframe(
            filtered[available].iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        csv_data = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download Monitoring Data",
            data=csv_data,
            file_name="bearing_monitoring_history.csv",
            mime="text/csv",
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "BearingGuard AI • Lightweight Edge AIoT "
        "Framework • Paderborn Dataset • "
        "MQTT Live Monitoring"
    )

    st.caption(
        "Research prototype using replayed dataset "
        "features, not a certified industrial "
        "safety-monitoring system."
    )


# ============================================================
# RUN DASHBOARD
# ============================================================

live_dashboard()