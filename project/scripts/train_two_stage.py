import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from src import config as C

REQUIRED = C.FEATURES + C.DIAGNOSTIC_FEATURES + ["fault_type", "session", "firmware", "is_synthetic"]
NOTES = []


def load_recordings(firmware):
    frames = []
    for path in sorted(C.REAL_DIR.glob("*.csv")):
        if path.name == "manifest.csv":
            continue
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise SystemExit(f"{path.name} is not a recorder file (missing {missing})")
        frames.append(df)
    if not frames:
        raise SystemExit(f"No recordings found in {C.REAL_DIR}")
    data = pd.concat(frames, ignore_index=True)
    if data["is_synthetic"].astype(str).str.lower().isin(["true", "1"]).any():
        raise SystemExit("data/real contains rows with is_synthetic=True. Remove them.")
    data = data[data["firmware"] == firmware]
    before = len(data)
    data = data.dropna(subset=C.FEATURES + C.DIAGNOSTIC_FEATURES)
    return data, before - len(data)


def split(df, label, allow_single, frac):
    sessions = sorted(df["session"].unique())
    if len(sessions) >= 2:
        hold = sessions[-1]
        NOTES.append(f"{label}: held out whole session {hold}")
        return df[df["session"] != hold], df[df["session"] == hold]
    if not allow_single:
        raise SystemExit(f"'{label}' has 1 session. Record another, or rerun with --allow-single-session.")
    df = df.sort_values("timestamp_ms")
    cut = int(len(df) * (1 - frac))
    NOTES.append(f"{label}: ONLY 1 session, used last {frac:.0%} of its rows (chronological) as holdout. "
                 "Optimistic: train and test rows share the same recording conditions.")
    return df.iloc[:cut], df.iloc[cut:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--firmware", default=C.FIRMWARE)
    ap.add_argument("--allow-single-session", action="store_true")
    ap.add_argument("--holdout-fraction", type=float, default=0.3)
    args = ap.parse_args()

    C.MODEL_DIR.mkdir(exist_ok=True)
    C.REPORT_DIR.mkdir(exist_ok=True)
    data, dropped = load_recordings(args.firmware)

    by = {}
    for label in (C.NORMAL_LABEL, C.SOUND_FILE_LABEL, C.VIBRATION_FILE_LABEL):
        part = data[data["fault_type"] == label]
        if part.empty:
            raise SystemExit(f"No '{label}' recordings for firmware {args.firmware}.")
        by[label] = part

    lines = [f"Two-stage training report  {datetime.now().isoformat(timespec='seconds')}",
             f"Firmware: {args.firmware}   model features: {C.FEATURES}",
             f"Rows dropped for NaN: {dropped}"]
    for label, part in by.items():
        lines.append(f"{label}: {len(part)} rows, sessions={sorted(part['session'].unique())}")

    lines += ["", "Class profile (mean +/- std) - how far apart the classes really are"]
    for label, part in by.items():
        desc = ", ".join(f"{f}={part[f].mean():.4g}+/-{part[f].std():.3g}" for f in C.FEATURES)
        lines.append(f"  {label}: {desc}")

    n_tr, n_ho = split(by[C.NORMAL_LABEL], C.NORMAL_LABEL, args.allow_single_session, args.holdout_fraction)
    s1 = IsolationForest(**C.STAGE1_PARAMS).fit(n_tr[C.FEATURES])
    false_alarms = int((s1.predict(n_ho[C.FEATURES]) == -1).sum())
    far = false_alarms / len(n_ho)
    abn = pd.concat([by[C.SOUND_FILE_LABEL], by[C.VIBRATION_FILE_LABEL]])
    flagged = float((s1.predict(abn[C.FEATURES]) == -1).mean())
    lines += ["", "Stage 1 (Isolation Forest, trained on normal only)",
              f"Held-out normal rows: {len(n_ho)}, false alarms {false_alarms} ({far:.2%})",
              f"Share of abnormal rows flagged ANOMALY: {flagged:.2%}"]

    no_cur = [f for f in C.FEATURES if f != "current_mA"]
    s1b = IsolationForest(**C.STAGE1_PARAMS).fit(n_tr[no_cur])
    lines.append(f"Stage 1 without current_mA (comparison): false alarms {(s1b.predict(n_ho[no_cur]) == -1).mean():.2%}, "
                 f"abnormal flagged {(s1b.predict(abn[no_cur]) == -1).mean():.2%}")

    tr_parts, ho_parts = [], []
    for file_label, out_label in ((C.SOUND_FILE_LABEL, C.SOUND_LABEL), (C.VIBRATION_FILE_LABEL, C.VIBRATION_LABEL)):
        tr, ho = split(by[file_label], file_label, args.allow_single_session, args.holdout_fraction)
        tr_parts.append(tr.assign(target=out_label))
        ho_parts.append(ho.assign(target=out_label))
    tr, ho = pd.concat(tr_parts), pd.concat(ho_parts)
    order = [C.SOUND_LABEL, C.VIBRATION_LABEL]
    s2 = RandomForestClassifier(**C.STAGE2_PARAMS).fit(tr[C.STAGE2_FEATURES], tr["target"])
    pred = s2.predict(ho[C.STAGE2_FEATURES])
    acc = accuracy_score(ho["target"], pred)
    cm = confusion_matrix(ho["target"], pred, labels=order)
    lines += ["", f"Stage 2 (Random Forest, sound vs vibration) inputs: {C.STAGE2_FEATURES}",
              f"Held-out rows: {len(ho)}, accuracy {acc:.2%}",
              f"Confusion matrix (rows=true, cols=pred, order={order}): {cm.tolist()}",
              "Which inputs carry the separation (same split, one input group at a time):"]
    for name, cols in (("all four inputs incl. current (comparison)", C.FEATURES),
                       ("microphone only", ["microphone_rms_raw"]),
                       ("vibration only", ["vibration_rms", "vibration_peak_to_peak"]),
                       ("current only", ["current_mA"])):
        m = RandomForestClassifier(**C.STAGE2_PARAMS).fit(tr[cols], tr["target"])
        lines.append(f"  {name}: {accuracy_score(ho['target'], m.predict(ho[cols])):.2%}")

    normal = by[C.NORMAL_LABEL]
    gates = {f: float(normal[f].quantile(C.GATE_QUANTILE) * C.GATE_FACTOR)
             for f in ("vibration_rms", "microphone_rms_raw")}
    vib_ho = ho[ho["target"] == C.VIBRATION_LABEL]
    snd_ho = ho[ho["target"] == C.SOUND_LABEL]
    lines += ["", "Physical gates (from all normal recordings)"]
    for f, ho_part, label in (("vibration_rms", vib_ho, "vibration"), ("microphone_rms_raw", snd_ho, "sound")):
        lines.append(f"  {f} gate {gates[f]:.4g}: held-out normal above {(n_ho[f] > gates[f]).mean():.2%}, "
                     f"held-out {label} above {(ho_part[f] > gates[f]).mean():.2%}")
    lines.append("Cross-check: share of each abnormal class that exceeds BOTH gates (classes overlap if both are high)")
    for label, part in (("sound", snd_ho), ("vibration", vib_ho)):
        both = ((part["vibration_rms"] > gates["vibration_rms"]) & (part["microphone_rms_raw"] > gates["microphone_rms_raw"])).mean()
        lines.append(f"  {label}: {both:.2%}")

    lines += ["", "Evaluation notes"] + [f"  - {n}" for n in NOTES]

    s1_final = IsolationForest(**C.STAGE1_PARAMS).fit(normal[C.FEATURES])
    all_abn = pd.concat([by[C.SOUND_FILE_LABEL].assign(target=C.SOUND_LABEL),
                         by[C.VIBRATION_FILE_LABEL].assign(target=C.VIBRATION_LABEL)])
    s2_final = RandomForestClassifier(**C.STAGE2_PARAMS).fit(all_abn[C.STAGE2_FEATURES], all_abn["target"])
    joblib.dump(s1_final, C.STAGE1_PATH)
    joblib.dump(s2_final, C.STAGE2_PATH)

    stats = {f: {"min": float(normal[f].min()), "p01": float(normal[f].quantile(0.01)),
                 "median": float(normal[f].median()), "p99": float(normal[f].quantile(0.99)),
                 "max": float(normal[f].max())} for f in C.FEATURES + C.DIAGNOSTIC_FEATURES}
    meta = {"trained_at": datetime.now().isoformat(timespec="seconds"), "firmware": args.firmware,
            "features": C.FEATURES, "stage2_features": C.STAGE2_FEATURES, "diagnostic_features": C.DIAGNOSTIC_FEATURES,
            "stage1_params": C.STAGE1_PARAMS, "stage2_params": C.STAGE2_PARAMS,
            "rows": {k: len(v) for k, v in by.items()},
            "sessions": {k: sorted(v["session"].unique()) for k, v in by.items()},
            "synthetic": False, "gates": gates, "normal_stats": stats, "evaluation_notes": NOTES,
            "stage1_holdout_false_alarm_rate": far, "stage2_holdout_accuracy": acc}
    with open(C.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    text = "\n".join(lines)
    C.REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
