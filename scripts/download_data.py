"""Download and validate the public Amazon Fine Food Reviews source."""

import argparse
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, RAW_PATH, load_raw, sha256, versions, write_json

SOURCE = "https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/snap/amazon-fine-food-reviews"
AUTH_HELP = (
    "Kaggle authentication is required. Install the official Kaggle CLI, run "
    "`kaggle auth login`, then run `kaggle datasets download "
    "snap/amazon-fine-food-reviews -p data/raw` from the project root. "
    "Re-run this script afterwards. Never put credentials in source code. "
    "See https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md"
)


def validate_zip(path):
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"ZIP integrity check failed: {bad_member}")
        matches = [name for name in archive.namelist() if Path(name).name == "Reviews.csv"]
        if len(matches) != 1:
            raise ValueError(f"Expected one Reviews.csv in archive, found {matches!r}")
        return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    args = parser.parse_args()
    raw = args.raw_path.expanduser().resolve()
    raw.parent.mkdir(parents=True, exist_ok=True)
    archive_path = raw.parent / "amazon-fine-food-reviews.zip"
    manifest_path = (ROOT / "outputs" / "raw_manifest.json" if raw == RAW_PATH.resolve()
                     else raw.with_name(raw.name + ".manifest.json"))
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    downloaded_at = None

    if not raw.exists():
        if not archive_path.exists():
            partial = archive_path.with_suffix(".zip.partial")
            if partial.exists():
                raise FileExistsError(f"Incomplete download already exists: {partial}; inspect and remove it before retrying.")
            print("Downloading public Kaggle archive...", flush=True)
            request = urllib.request.Request(DOWNLOAD, headers={"User-Agent": "Group-Phi-M1/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if "zip" not in response.headers.get("Content-Type", "").lower():
                        raise RuntimeError("The public endpoint did not return a ZIP. " + AUTH_HELP)
                    expected_bytes = response.headers.get("Content-Length")
                    with partial.open("xb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                if expected_bytes and partial.stat().st_size != int(expected_bytes):
                    raise ValueError("Downloaded archive size does not match Content-Length.")
                validate_zip(partial)
                partial.rename(archive_path)
                downloaded_at = datetime.now(timezone.utc).isoformat()
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise RuntimeError(AUTH_HELP) from None
                raise
        member = validate_zip(archive_path)
        partial_csv = raw.with_suffix(raw.suffix + ".partial")
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(member) as source, partial_csv.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        load_raw(partial_csv)
        if previous.get("sha256") and previous["sha256"] != sha256(partial_csv):
            raise ValueError("Extracted source differs from recorded SHA-256; final source was not created.")
        partial_csv.rename(raw)
    else:
        print(f"Validating existing source without overwriting: {raw}", flush=True)

    df = load_raw(raw)
    checksum = sha256(raw)
    if previous.get("sha256") and previous["sha256"] != checksum:
        raise ValueError("Source differs from recorded SHA-256; existing provenance was preserved.")
    manifest = {
        "dataset": "Amazon Fine Food Reviews", "source_url": SOURCE,
        "original_source_url": "https://snap.stanford.edu/data/web-FineFoods.html",
        "download_url": DOWNLOAD, "raw_path": str(raw),
        "downloaded_at_utc": downloaded_at or previous.get("downloaded_at_utc"),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": checksum, "rows": len(df), "columns": list(df.columns),
        "versions": versions(),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
