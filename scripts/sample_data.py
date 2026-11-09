"""Create or verify Group Phi's fixed official sample without overwriting it."""

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import ROOT, RAW_PATH, load_raw, sample_path, sha256, versions, write_json

SEED = 47
SAMPLE_SIZE = 50000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-path", type=Path, default=sample_path())
    args = parser.parse_args()
    reviews_path = args.reviews_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    manifest_path = (ROOT / "outputs" / "sample_manifest.json" if output_path == sample_path().resolve()
                     else output_path.with_name(output_path.name + ".manifest.json"))
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    raw_checksum = sha256(reviews_path)
    if previous.get("raw_sha256") and previous["raw_sha256"] != raw_checksum:
        raise ValueError("Source differs from the sample's recorded raw SHA-256; no sample was written.")
    # load_raw uses the professor's exact pd.read_csv(reviews_path) call and only validates.
    df = load_raw(reviews_path)
    sample = df.sample(n=50000, random_state=47)
    # Official API: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
    expected = pd.read_csv(io.StringIO(sample.to_csv(index=False)))
    reused = output_path.exists()
    if reused:
        actual = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(actual, expected)
        print(f"Existing sample exactly matches; preserving: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(output_path, index=False, mode="x")
        pd.testing.assert_frame_equal(pd.read_csv(output_path), expected)
    assert len(expected) == SAMPLE_SIZE
    checksum = sha256(output_path)
    if previous.get("sample_path") == str(output_path) and previous.get("sample_sha256") not in (None, checksum):
        raise ValueError("Sample bytes differ from the recorded SHA-256; existing manifest was preserved.")
    manifest = {
        "group": "Phi", "seed": SEED, "sample_size": SAMPLE_SIZE,
        "sampling_method": "pd.read_csv(reviews_path).sample(n=50000, random_state=47)",
        "raw_path": str(reviews_path), "raw_sha256": raw_checksum,
        "sample_path": str(output_path), "sample_sha256": checksum,
        "columns": list(expected.columns),
        "created_at_utc": previous.get("created_at_utc") if reused else datetime.now(timezone.utc).isoformat(),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "versions": versions(),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
