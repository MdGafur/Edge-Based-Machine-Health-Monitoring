import argparse
import csv
import hashlib
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C


def main():
    ap = argparse.ArgumentParser(description="Check recordings against their saved SHA-256 fingerprints.")
    ap.add_argument("--baseline", action="store_true",
                    help="fingerprint files that have none yet (taken NOW, so it proves nothing about earlier changes)")
    args = ap.parse_args()

    hashes_path = C.REAL_DIR / "hashes.csv"
    known = {}
    if hashes_path.exists():
        with open(hashes_path, newline="") as f:
            known = {r["file"]: (r["sha256"], r["recorded"]) for r in csv.DictReader(f)}

    bad, new_rows = 0, []
    for path in sorted(C.REAL_DIR.glob("*.csv")):
        if path.name in ("manifest.csv", "hashes.csv"):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.name not in known:
            if args.baseline:
                new_rows.append([path.name, digest, "baseline-" + datetime.now().isoformat(timespec="seconds")])
                os.chmod(path, stat.S_IREAD)
                print(f"BASELINE  {path.name} (fingerprint taken now)")
            else:
                print(f"UNHASHED  {path.name} (no fingerprint; run with --baseline to start tracking it)")
        elif known[path.name][0] == digest:
            kind = " (baseline taken after recording)" if known[path.name][1].startswith("baseline") else ""
            print(f"OK        {path.name}{kind}")
        else:
            bad += 1
            print(f"MODIFIED  {path.name} (differs from its saved fingerprint)")

    if new_rows:
        write_header = not hashes_path.exists()
        with open(hashes_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["file", "sha256", "recorded"])
            w.writerows(new_rows)
    print("\nAll fingerprinted files match." if not bad else f"\n{bad} file(s) changed since they were fingerprinted.")


if __name__ == "__main__":
    main()
