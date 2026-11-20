import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import train_two_stage as T
from src import config as C


def table(cm, rows, cols):
    return pd.DataFrame(cm, index=[f"true {r}" for r in rows], columns=[f"pred {c}" for c in cols])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--firmware", default=C.FIRMWARE)
    ap.add_argument("--allow-single-session", action="store_true")
    ap.add_argument("--holdout-fraction", type=float, default=0.3)
    ap.add_argument("--png", action="store_true", help="also save reports/confusion_matrices.png")
    args = ap.parse_args()

    data, _ = T.load_recordings(args.firmware)
    by = {lab: data[data["fault_type"] == lab] for lab in (C.NORMAL_LABEL, C.SOUND_FILE_LABEL, C.VIBRATION_FILE_LABEL)}
    for lab, part in by.items():
        if part.empty:
            raise SystemExit(f"No '{lab}' recordings found.")

    n_tr, n_ho = T.split(by[C.NORMAL_LABEL], C.NORMAL_LABEL, args.allow_single_session, args.holdout_fraction)
    abn = pd.concat([by[C.SOUND_FILE_LABEL], by[C.VIBRATION_FILE_LABEL]])
    s1 = IsolationForest(**C.STAGE1_PARAMS).fit(n_tr[C.FEATURES])
    y_true1 = ["NORMAL"] * len(n_ho) + ["ABNORMAL"] * len(abn)
    X1 = pd.concat([n_ho[C.FEATURES], abn[C.FEATURES]])
    y_pred1 = ["ANOMALY" if p == -1 else "NORMAL" for p in s1.predict(X1)]
    cm1 = confusion_matrix([("ANOMALY" if t == "ABNORMAL" else "NORMAL") for t in y_true1], y_pred1, labels=["NORMAL", "ANOMALY"])

    print("STAGE 1 - Isolation Forest (trained on normal sessions only)")
    print(f"Test set: {len(n_ho)} held-out NORMAL rows + {len(abn)} abnormal rows (never used in training)")
    print(table(cm1, ["NORMAL", "ABNORMAL"], ["NORMAL", "ANOMALY"]).to_string())
    tn, fp, fn, tp = cm1[0, 0], cm1[0, 1], cm1[1, 0], cm1[1, 1]
    print(f"false-alarm rate {fp / (tn + fp):.1%} | detection rate {tp / (tp + fn):.1%} | precision {tp / max(tp + fp, 1):.1%}")

    tr_parts, ho_parts = [], []
    for file_label, out_label in ((C.SOUND_FILE_LABEL, C.SOUND_LABEL), (C.VIBRATION_FILE_LABEL, C.VIBRATION_LABEL)):
        tr, ho = T.split(by[file_label], file_label, args.allow_single_session, args.holdout_fraction)
        tr_parts.append(tr.assign(target=out_label))
        ho_parts.append(ho.assign(target=out_label))
    tr, ho = pd.concat(tr_parts), pd.concat(ho_parts)
    s2 = RandomForestClassifier(**C.STAGE2_PARAMS).fit(tr[C.STAGE2_FEATURES], tr["target"])
    pred2 = s2.predict(ho[C.STAGE2_FEATURES])
    order = [C.SOUND_LABEL, C.VIBRATION_LABEL]
    cm2 = confusion_matrix(ho["target"], pred2, labels=order)

    print("\nSTAGE 2 - Random Forest (sound vs vibration), inputs:", C.STAGE2_FEATURES)
    print(f"Test set: {len(ho)} held-out rows")
    print(table(cm2, order, order).to_string())
    print(classification_report(ho["target"], pred2, labels=order, digits=3, zero_division=0))
    for note in T.NOTES:
        print("NOTE:", note)

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; run: python -m pip install matplotlib")
            return
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, cm, rl, cl, title in ((axes[0], cm1, ["NORMAL", "ABNORMAL"], ["NORMAL", "ANOMALY"], "Stage 1 (held-out)"),
                                      (axes[1], cm2, order, order, "Stage 2 (held-out rows)")):
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(cl)), cl)
            ax.set_yticks(range(len(rl)), rl)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(title)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
        fig.tight_layout()
        out = C.REPORT_DIR / "confusion_matrices.png"
        fig.savefig(out, dpi=150)
        print("Saved", out)


if __name__ == "__main__":
    main()
