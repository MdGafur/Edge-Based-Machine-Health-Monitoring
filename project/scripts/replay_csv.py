import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.pipeline import HealthPipeline, StabilityTracker


def main():
    ap = argparse.ArgumentParser(description="Run recorded rows through the live decision logic.")
    ap.add_argument("csv")
    args = ap.parse_args()

    pipe = HealthPipeline()
    tracker = StabilityTracker()
    df = pd.read_csv(args.csv)
    raw, confirmed_counts, alerts = Counter(), Counter(), 0
    flagged = vib_gate = mic_gate = checked = 0
    for _, row in df.iterrows():
        result = pipe.assess(row.to_dict())
        raw[StabilityTracker.label(result)] += 1
        if result["stage1"] is not None:
            checked += 1
            flagged += result["stage1"] == "ANOMALY"
            vib_gate += bool(result["vib_gate"])
            mic_gate += bool(result["mic_gate"])
        confirmed, changed = tracker.update(result)
        confirmed_counts[confirmed] += 1
        if changed and confirmed not in ("NORMAL", "WAITING"):
            alerts += 1
    print(f"{args.csv}: {len(df)} rows")
    if checked:
        print(f"Stage 1 flagged ANOMALY: {flagged}/{checked} ({flagged / checked:.1%}) | "
              f"vibration gate exceeded: {vib_gate / checked:.1%} | microphone gate exceeded: {mic_gate / checked:.1%}")
    print("Per-row status:", dict(raw))
    print("After 3-reading stability:", dict(confirmed_counts))
    print("Alerts that would be sent:", alerts)


if __name__ == "__main__":
    main()
