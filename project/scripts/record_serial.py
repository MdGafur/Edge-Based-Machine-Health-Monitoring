import argparse
import csv
import hashlib
import math
import os
import stat
import statistics
import time
from datetime import datetime
from pathlib import Path

import serial

COLUMNS = [
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
SENSOR_COLUMNS = COLUMNS[1:13]
LABELS = ["normal", "abnormal_sound", "abnormal_vibration"]
EXTRA = ["fault_type", "session", "firmware", "is_synthetic"]

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real"
MANIFEST = REAL_DIR / "manifest.csv"
HASHES = REAL_DIR / "hashes.csv"

WARMUP_ROWS = 10
EXPECTED_SAMPLES = 100
WINDOW_MS_RANGE = (950.0, 1050.0)
STUCK_ABORT = 10
GRAVITY_RANGE = (8.0, 11.5)


def parse(line):
    parts = line.strip().split(",")
    if len(parts) != len(COLUMNS):
        return None
    try:
        return dict(zip(COLUMNS, (float(p) for p in parts)))
    except ValueError:
        return None


def row_problem(r):
    if any(math.isnan(r[c]) for c in SENSOR_COLUMNS):
        return "nan"
    if r["vibration_rms"] == 0 or r["vibration_peak_to_peak"] == 0:
        return "stuck"
    g = math.sqrt(sum(r[c] ** 2 for c in ("acceleration_x_m_s2", "acceleration_y_m_s2", "acceleration_z_m_s2")))
    if not GRAVITY_RANGE[0] <= g <= GRAVITY_RANGE[1]:
        return "gravity"
    if int(r["accel_samples"]) != EXPECTED_SAMPLES:
        return "samples"
    if not WINDOW_MS_RANGE[0] <= r["accel_window_ms"] <= WINDOW_MS_RANGE[1]:
        return "window"
    return None


def summarize(rows):
    print("\nSession summary (mean +/- std)")
    for c in ["current_mA", "bus_voltage_v", "temperature_C", "microphone_rms_raw",
              "vibration_rms", "vibration_peak_to_peak", "accel_window_ms"]:
        vals = [r[c] for r in rows]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {c:24s} {statistics.fmean(vals):12.4f} +/- {sd:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Record real ESP32 readings to CSV.")
    ap.add_argument("--label", required=True, choices=LABELS)
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--session", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--notes", default="", help="how the condition was created, mounting, room")
    ap.add_argument("--firmware", default="fw2")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--countdown", type=int, default=5)
    args = ap.parse_args()

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    out = REAL_DIR / f"{args.label}_{args.session}.csv"
    if out.exists():
        raise SystemExit(f"{out.name} already exists. Use a different --session.")

    for i in range(args.countdown, 0, -1):
        print(f"Starting in {i}...", end="\r")
        time.sleep(1)

    started = datetime.now().isoformat(timespec="seconds")
    kept, seen = [], 0
    dropped = {"nan": 0, "stuck": 0, "gravity": 0, "samples": 0, "window": 0}
    stuck_run = 0
    silent = 0
    try:
        with serial.Serial(args.port, args.baud, timeout=3) as ser, open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS + EXTRA)
            while len(kept) < args.rows:
                raw = ser.readline().decode("utf-8", errors="ignore")
                if not raw:
                    silent += 1
                    if silent >= 5:
                        raise SystemExit("No data from the ESP32. Close Serial Monitor / dashboard and check the port.")
                    continue
                silent = 0
                r = parse(raw)
                if r is None:
                    continue
                seen += 1
                if seen <= WARMUP_ROWS:
                    continue
                problem = row_problem(r)
                if problem:
                    dropped[problem] += 1
                    stuck_run = stuck_run + 1 if problem in ("stuck", "gravity") else 0
                    if stuck_run >= STUCK_ABORT:
                        print(f"\nAccelerometer gave frozen or impossible values for {STUCK_ABORT} rows in a row. "
                              "Power-cycle the ESP32, check the wiring, then record again.")
                        break
                    continue
                stuck_run = 0
                w.writerow([r[c] for c in COLUMNS] + [args.label, args.session, args.firmware, False])
                f.flush()
                kept.append(r)
                print(f"{args.label}: {len(kept)}/{args.rows}", end="\r")
    except KeyboardInterrupt:
        print("\nStopped early.")

    if kept:
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        new_hash = not HASHES.exists()
        with open(HASHES, "a", newline="") as f:
            w = csv.writer(f)
            if new_hash:
                w.writerow(["file", "sha256", "recorded"])
            w.writerow([out.name, digest, started])
        os.chmod(out, stat.S_IREAD)

    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "label", "session", "firmware", "rows", "started", "notes"])
        w.writerow([out.name, args.label, args.session, args.firmware, len(kept), started, args.notes])

    print(f"\nSaved {len(kept)} rows to {out}")
    print(f"Dropped rows: {dropped}")
    if kept:
        summarize(kept)


if __name__ == "__main__":
    main()
