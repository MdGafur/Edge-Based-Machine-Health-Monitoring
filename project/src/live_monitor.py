import sys
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import serial
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src import config as C
from src.alerts import send_telegram
from src.common import parse_serial_line
from src.pipeline import HealthPipeline, StabilityTracker

st.set_page_config(page_title="Machine Health Monitor", layout="wide")
st_autorefresh(interval=1000, key="refresh")


@st.cache_resource
def get_pipeline():
    return HealthPipeline()


def init_state():
    s = st.session_state
    s.setdefault("ser", None)
    s.setdefault("tracker", StabilityTracker())
    s.setdefault("history", deque(maxlen=300))
    s.setdefault("last", None)
    s.setdefault("recent_flags", deque(maxlen=10))
    s.setdefault("notice", "")


def connect(port, baud):
    try:
        st.session_state.ser = serial.Serial(port, baud, timeout=0.1)
        st.session_state.notice = f"Connected to {port}"
    except serial.SerialException as e:
        st.session_state.ser = None
        st.session_state.notice = f"Serial error: {e}"


def disconnect():
    if st.session_state.ser is not None:
        st.session_state.ser.close()
    st.session_state.ser = None


def handle(pipe, reading, telegram_on):
    s = st.session_state
    result = pipe.assess(reading)
    confirmed, changed = s.tracker.update(result)
    if changed and telegram_on and confirmed not in ("NORMAL", "WAITING"):
        ok, info = send_telegram(confirmed, result.get("detail") or f"Reading: {reading}")
        s.notice = f"Telegram: {info}"
    if result["stage1"] is not None:
        s.recent_flags.append(result["stage1"] == "ANOMALY")
    s.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "status": StabilityTracker.label(result),
        "stage1": result["stage1"] or "-",
        "score": None if result["stage1_score"] is None else round(result["stage1_score"], 4),
        "stage2": result["stage2"] or "-",
    })
    s.last = {"reading": reading, "result": result}


def drain(pipe, telegram_on):
    ser = st.session_state.ser
    if ser is None:
        return
    try:
        while ser.in_waiting:
            reading = parse_serial_line(ser.readline().decode("utf-8", errors="ignore"))
            if reading is not None:
                handle(pipe, reading, telegram_on)
    except (serial.SerialException, OSError) as e:
        st.session_state.notice = f"Serial lost: {e}"
        disconnect()


init_state()
try:
    pipe = get_pipeline()
except Exception as e:
    st.error(f"Models not loaded: {e}")
    st.stop()

with st.sidebar:
    st.header("Connection")
    port = st.text_input("Port", C.DEFAULT_PORT)
    baud = st.number_input("Baud", value=C.DEFAULT_BAUD, step=1)
    c1, c2 = st.columns(2)
    if c1.button("Connect"):
        connect(port, int(baud))
    if c2.button("Disconnect"):
        disconnect()
    telegram_on = st.checkbox("Telegram alerts", value=False)
    st.caption(st.session_state.notice)
    st.caption("Connected" if st.session_state.ser else "Not connected")

drain(pipe, telegram_on)

st.title("Edge-Based Machine Health Monitoring")
status = st.session_state.tracker.confirmed
(st.success if status == "NORMAL" else st.info if status == "WAITING" else st.error)(f"Confirmed status: {status}")

last = st.session_state.last
if last is None:
    st.info("Waiting for data from the ESP32.")
else:
    r, res = last["reading"], last["result"]
    if res["status"] == "SENSOR_FAULT":
        st.warning(f"Sensor fault: {res['detail']}. Power-cycle the ESP32 and check the wiring.")
    cols = st.columns(6)
    cols[0].metric("Temperature (C)", f"{r['temperature_C']:.2f}")
    cols[1].metric("Current (mA)", f"{r['current_mA']:.1f}")
    cols[2].metric("Voltage (V)", f"{r['bus_voltage_v']:.2f}")
    cols[3].metric("Mic RMS", f"{r['microphone_rms_raw']:.0f}", help=f"gate {pipe.gates['microphone_rms_raw']:.0f}")
    cols[4].metric("Vibration RMS", f"{r['vibration_rms']:.4f}", help=f"gate {pipe.gates['vibration_rms']:.4f}")
    cols[5].metric("Vib p-p", f"{r['vibration_peak_to_peak']:.3f}")

    if res["stage1"]:
        st.subheader("Detection")
        flags = st.session_state.recent_flags
        st.write(f"Stage 1 (Isolation Forest): {res['stage1']} for this reading, score {res['stage1_score']:.4f}. "
                 f"Flagged in {sum(flags)} of the last {len(flags)} readings.")
        st.caption("Advisory only. On unseen normal data the forest flags roughly 5 to 20% of single readings, "
                   "so a lone flag means little. The score is not a probability.")
        st.write(f"Physical gates: vibration {'EXCEEDED' if res['vib_gate'] else 'ok'}, "
                 f"microphone {'EXCEEDED' if res['mic_gate'] else 'ok'}")
        if res["detail"]:
            st.caption(res["detail"])
        if res["stage2"]:
            st.write(f"Stage 2 best guess: {res['stage2']}")
            st.caption("Conditional on an anomaly already being confirmed. Not the probability that a real fault exists, "
                       "and sound and vibration faults overlap on this hardware.")
            st.json({k: f"{v:.1%}" for k, v in res["stage2_proba"].items()})
    st.subheader("Independent diagnostics")
    st.write(", ".join(res["issues"]) if res["issues"] else "No temperature, current or voltage warnings.")

if st.session_state.history:
    hist = pd.DataFrame(list(st.session_state.history))
    st.subheader("Recent readings")
    st.dataframe(hist.tail(15).iloc[::-1])
    st.subheader("Detector score history")
    st.line_chart(hist["score"])
