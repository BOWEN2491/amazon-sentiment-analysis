"""Small shared helpers for the M1 data scripts."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd
from platformdirs import user_downloads_path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Reviews.csv"
EXPECTED_ROWS = 568454
EXPECTED_COLUMNS = [
    "Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator",
    "HelpfulnessDenominator", "Score", "Time", "Summary", "Text",
]


def sample_path():
    """Resolve the OS Downloads known folder, including redirected folders."""
    # https://platformdirs.readthedocs.io/en/latest/api.html#platformdirs.user_downloads_path
    return user_downloads_path() / "group_sample_seed47.csv"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def versions():
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name)
           for name in ("pandas", "numpy", "matplotlib", "platformdirs")},
    }


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_raw(path):
    """Validate the source without cleaning, filtering, or reordering it."""
    df = pd.read_csv(path)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} source rows, found {len(df):,}; sampling stopped.")
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected source columns: {list(df.columns)!r}")
    return df
