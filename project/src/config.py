from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

STAGE1_PATH = MODEL_DIR / "stage1_isolation_forest.joblib"
STAGE2_PATH = MODEL_DIR / "stage2_random_forest.joblib"
METADATA_PATH = MODEL_DIR / "two_stage_metadata.json"
REPORT_PATH = REPORT_DIR / "two_stage_report.txt"

SERIAL_COLUMNS = [
    "timestamp_ms",
    "acceleration_x_m_s2",
    "acceleration_y_m_s2",
    "acceleration_z_m_s2",
    "bus_voltage_v",
    "current_mA",
    "temperature_C",
    "microphone_rms_raw",
    "vibration_rms",
    "vibration_peak_to_peak",
    "vibration_x_rms",
    "vibration_y_rms",
    "vibration_z_rms",
    "accel_samples",
    "accel_window_ms",
]

# Model inputs: orientation-independent and not session-dependent.
# Raw acceleration and per-axis vibration depend on mounting. Bus voltage and
# temperature shift between sessions (supply level, warm-up), so they are used
# only by the independent threshold diagnostics, not by the models.
FEATURES = [
    "current_mA",
    "microphone_rms_raw",
    "vibration_rms",
    "vibration_peak_to_peak",
]
DIAGNOSTIC_FEATURES = ["bus_voltage_v", "temperature_C"]

# Stage 2 separates sound from vibration faults. Current carries no physical cue for
# that and can only encode which recording session a row came from, so it is excluded.
STAGE2_FEATURES = [
    "microphone_rms_raw",
    "vibration_rms",
    "vibration_peak_to_peak",
]

FIRMWARE = "fw2"

NORMAL_LABEL = "normal"
SOUND_FILE_LABEL = "abnormal_sound"
VIBRATION_FILE_LABEL = "abnormal_vibration"
SOUND_LABEL = "ABNORMAL_SOUND"
VIBRATION_LABEL = "ABNORMAL_VIBRATION"

# 0.01 chosen from unseen-session tests: false alarms 4.3% and 0.7% (was 17% and 7% at 0.05)
STAGE1_PARAMS = dict(n_estimators=300, contamination=0.01, random_state=42)
STAGE2_PARAMS = dict(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# Physical gates: a sound/vibration label is only trusted when the matching
# sensor reading is above the range seen in normal recordings.
GATE_QUANTILE = 0.995
GATE_FACTOR = 1.2

GRAVITY_RANGE = (8.0, 11.5)

# Below this Stage 2 confidence the abnormal type is reported as unspecified.
STAGE2_MIN_CONFIDENCE = 0.75

# False: a physical gate alone confirms an abnormal reading (the Isolation Forest is advisory).
# True: the Isolation Forest must also flag the reading. On the first real data the forest had
# 17% false alarms on a held-out normal session while the gates had under 0.4%.
REQUIRE_STAGE1_CONFIRMATION = False

STABILITY_COUNT = 3
ALERT_COOLDOWN_S = 120

TEMP_WARN_MARGIN_C = 5.0
CURRENT_HIGH_FACTOR = 1.5
CURRENT_LOW_FACTOR = 0.5
VOLTAGE_TOLERANCE_FRACTION = 0.10

DEFAULT_PORT = "COM3"
DEFAULT_BAUD = 115200
