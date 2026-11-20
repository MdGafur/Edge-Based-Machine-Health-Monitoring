import json

from .config import SERIAL_COLUMNS, METADATA_PATH


def parse_serial_line(line):
    parts = line.strip().split(",")
    if len(parts) != len(SERIAL_COLUMNS):
        return None
    try:
        return dict(zip(SERIAL_COLUMNS, (float(p) for p in parts)))
    except ValueError:
        return None


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
