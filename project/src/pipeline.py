import math

import joblib
import pandas as pd

from . import config as C
from .common import load_metadata


class HealthPipeline:
    def __init__(self):
        self.stage1 = joblib.load(C.STAGE1_PATH)
        self.stage2 = joblib.load(C.STAGE2_PATH)
        self.meta = load_metadata()
        if self.meta["features"] != C.FEATURES or self.meta.get("stage2_features") != C.STAGE2_FEATURES:
            raise RuntimeError("Saved models do not match config.py. Run scripts/train_two_stage.py again.")
        self.gates = self.meta["gates"]
        self.stats = self.meta["normal_stats"]

    def sensor_fault(self, r):
        for c in C.FEATURES + C.DIAGNOSTIC_FEATURES:
            if math.isnan(r[c]):
                return f"{c} is NaN"
        g = math.sqrt(r["acceleration_x_m_s2"] ** 2 + r["acceleration_y_m_s2"] ** 2 + r["acceleration_z_m_s2"] ** 2)
        if not C.GRAVITY_RANGE[0] <= g <= C.GRAVITY_RANGE[1]:
            return f"accelerometer gravity magnitude {g:.1f} m/s2 is impossible (sensor glitch)"
        if r["vibration_rms"] == 0 or r["vibration_peak_to_peak"] == 0:
            return "accelerometer frozen (vibration exactly 0)"
        return None

    def diagnostics(self, r):
        issues = []
        if r["temperature_C"] > self.stats["temperature_C"]["max"] + C.TEMP_WARN_MARGIN_C:
            issues.append("HIGH_TEMPERATURE")
        cur = self.stats["current_mA"]
        if r["current_mA"] > cur["p99"] * C.CURRENT_HIGH_FACTOR:
            issues.append("HIGH_CURRENT")
        if r["current_mA"] < cur["p01"] * C.CURRENT_LOW_FACTOR:
            issues.append("LOW_CURRENT")
        v = self.stats["bus_voltage_v"]["median"]
        if abs(r["bus_voltage_v"] - v) > v * C.VOLTAGE_TOLERANCE_FRACTION:
            issues.append("VOLTAGE_DEVIATION")
        return issues

    def assess(self, r):
        fault = self.sensor_fault(r)
        if fault:
            return {"status": "SENSOR_FAULT", "detail": fault, "issues": [], "stage1": None, "stage1_score": None,
                    "vib_gate": None, "mic_gate": None, "stage2": None, "stage2_proba": None}
        X = pd.DataFrame([[r[c] for c in C.FEATURES]], columns=C.FEATURES)
        flagged = int(self.stage1.predict(X)[0]) == -1
        score = float(self.stage1.decision_function(X)[0])
        vib_hit = r["vibration_rms"] > self.gates["vibration_rms"]
        mic_hit = r["microphone_rms_raw"] > self.gates["microphone_rms_raw"]

        status, detail, stage2, proba = "NORMAL", "", None, None
        gate_hit = vib_hit or mic_hit
        if gate_hit and (flagged or not C.REQUIRE_STAGE1_CONFIRMATION):
            X2 = pd.DataFrame([[r[c] for c in C.STAGE2_FEATURES]], columns=C.STAGE2_FEATURES)
            p = self.stage2.predict_proba(X2)[0]
            proba = {str(k): float(v) for k, v in zip(self.stage2.classes_, p)}
            stage2 = max(proba, key=proba.get)
            if mic_hit and not vib_hit:
                status = C.SOUND_LABEL
                detail = "microphone far above normal while vibration is normal"
            elif proba[stage2] >= C.STAGE2_MIN_CONFIDENCE:
                status = stage2
            else:
                status = "ABNORMAL_UNSPECIFIED"
                detail = f"type uncertain: best guess {stage2} at {proba[stage2]:.0%}"
            if not flagged:
                detail = (detail + "; " if detail else "") + "confirmed by the physical gate, the Isolation Forest scored it normal"
        elif flagged:
            detail = "detector flagged this reading but neither physical gate was exceeded (not confirmed)"
        elif gate_hit:
            detail = "a gate was exceeded but the Isolation Forest scored it normal (not confirmed)"
        return {"status": status, "detail": detail, "issues": self.diagnostics(r), "stage1": "ANOMALY" if flagged else "NORMAL",
                "stage1_score": score, "vib_gate": vib_hit, "mic_gate": mic_hit, "stage2": stage2, "stage2_proba": proba}


ABNORMAL_TYPES = ("ABNORMAL_SOUND", "ABNORMAL_VIBRATION", "ABNORMAL_UNSPECIFIED")


class StabilityTracker:
    def __init__(self, count=C.STABILITY_COUNT):
        self.count = count
        self.key = None
        self.run = 0
        self.confirmed = "WAITING"

    @staticmethod
    def label(result):
        parts = [] if result["status"] == "NORMAL" else [result["status"]]
        parts += result["issues"]
        return " + ".join(parts) if parts else "NORMAL"

    @staticmethod
    def group(label):
        for t in ABNORMAL_TYPES:
            label = label.replace(t, "ABNORMAL")
        return label

    def update(self, result):
        key = self.label(result)
        self.run = self.run + 1 if key == self.key else 1
        self.key = key
        changed = False
        if self.run >= self.count and key != self.confirmed:
            previous = self.group(self.confirmed)
            self.confirmed = key
            changed = self.group(key) != previous
        return self.confirmed, changed
