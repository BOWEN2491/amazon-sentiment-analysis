# Data Engineering and Exploratory Analysis

[Back to the project homepage](../README.md)

This document preserves Group Phi's verified M1 data preparation and EDA. It does not establish completion or submission of the full Project Charter. No models have been trained and no application has been developed.

## Environment

Use Python 3.11. The verified environment uses Python 3.11.9, pandas 3.0.5, NumPy 2.3.3, Matplotlib 3.10.6, and platformdirs 4.14.5. The four package versions are pinned in [requirements.txt](../requirements.txt). Downloads, ZIP processing, and SHA-256 use the Python standard library.

## Reproduce the source and official sample

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/sample_data.py
```

The source is [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), originating from [Stanford SNAP](https://snap.stanford.edu/data/web-FineFoods.html). The script verifies **568,454 source rows** and the ten source columns. It downloads the public archive without credentials, validates ZIP integrity, and extracts only `Reviews.csv`. If authentication becomes necessary, use the official Kaggle CLI's `kaggle auth login` and follow the script's instructions; never store credentials in this repository.

The sampling method is exactly:

```python
df = pd.read_csv(reviews_path)
sample = df.sample(n=50000, random_state=47)
sample.to_csv(output_path, index=False, mode="x")
```

No cleaning, filtering, shuffling, or index reset occurs before sampling. The script uses `platformdirs.user_downloads_path()` to resolve the system Downloads folder, including redirected Windows folders. The official file is `<system Downloads>/group_sample_seed47.csv`; resolve the directory through the operating system rather than assuming a home-directory path. An existing sample is compared to the exact expected content and row order, reused if identical, and never silently overwritten.

`outputs/raw_manifest.json` and `outputs/sample_manifest.json` record source/sample hashes, row counts, columns, paths, timestamps, and Python/package versions. Pinning versions and preserving source bytes/order make the sample reproducible.

Optional `--raw-path` (download) and `--reviews-path` / `--output-path` (sample) arguments support alternate local locations. Custom raw/sample destinations receive a separate sidecar named `<filename>.manifest.json`, so they cannot replace the official manifests. A recorded source hash mismatch stops sampling before writing a sample.

## EDA

```powershell
.\.venv\Scripts\python.exe scripts/analyze_data.py
```

Analysis reads the official sample and verifies its SHA-256 against `outputs/sample_manifest.json`, its 50,000-row size, original columns, and scores 1–5. It labels sentiments in memory and checks the sample hash again after analysis. It does not clean, relabel, or rewrite the official CSV. Unlike download/sampling, analysis has no custom-path options: it always uses the official sample.

The **provisional** sentiment mapping is 1–2 = Negative, 3 = Neutral, 4–5 = Positive, pending group confirmation. Percentages use all 50,000 sampled reviews. Review length is `len(text.split())` on original, nonmissing `Text`; whitespace-only strings count as zero. HTML and punctuation are retained. Missingness follows pandas' default `read_csv` missing-value interpretation and `isna()`. Exact duplicate rows compare all ten original columns; duplicate review texts compare nonmissing `Text` exactly. Both duplicate measures count extra occurrences after the first. No missing values or duplicates are removed.

Class imbalance is described using the largest/smallest sentiment class count ratio, with a declared threshold of 2:1. This is a descriptive rule, not a statistical significance test. An absent class is flagged as imbalanced with an undefined ratio.

### Verified findings

| Rating | Count | Percentage |
| --- | ---: | ---: |
| 1 | 4,731 | 9.462% |
| 2 | 2,659 | 5.318% |
| 3 | 3,709 | 7.418% |
| 4 | 7,191 | 14.382% |
| 5 | 31,710 | 63.420% |

| Proposed sentiment | Count | Percentage |
| --- | ---: | ---: |
| Negative | 7,390 | 14.780% |
| Neutral | 3,709 | 7.418% |
| Positive | 38,901 | 77.802% |

Mean review length is **80.42688 words**. `Text` and `Score` have no missing values; `Summary` and `ProfileName` each have **2 missing values (0.004%)**; all other columns have none. There are **0 exact duplicate rows** and **3,674 additional occurrences of nonmissing review texts**. The largest/smallest sentiment class ratio is **10.4883:1**, showing class imbalance under the declared rule.

### Outputs

- [`outputs/eda_statistics.json`](../outputs/eda_statistics.json): full machine-readable results, metric definitions, versions, and sample hash.
- [`outputs/figures/score_distribution.png`](../outputs/figures/score_distribution.png): 300-DPI rating bar chart with counts and percentages, suitable for the M1 PDF.
- [`docs/m1_dataset_summary.md`](m1_dataset_summary.md): concise English text ready to paste into the group charter.
- `outputs/raw_manifest.json` and `outputs/sample_manifest.json`: provenance and reproducibility records.

## Complete reproduction

Clone the repository and run commands from its root:

```sh
git clone https://github.com/BOWEN2491/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/sample_data.py
.\.venv\Scripts\python.exe scripts/analyze_data.py
```

macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/sample_data.py
.venv/bin/python scripts/analyze_data.py
```

For an existing environment, skip environment creation and installation. Existing validated source/sample files are reused. Never delete or replace the fixed official sample merely to bypass a mismatch.

### Authentication fallback

The public Kaggle download succeeded without authentication in the verified run. If authentication becomes necessary, install the official CLI in the environment and log in using its supported flow:

```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
.\.venv\Scripts\kaggle.exe auth login
.\.venv\Scripts\kaggle.exe datasets download snap/amazon-fine-food-reviews -p data/raw
.\.venv\Scripts\python.exe scripts/download_data.py
```

On macOS/Linux use `.venv/bin/python` and `.venv/bin/kaggle` for the corresponding commands. The download script validates and extracts the CLI-downloaded archive. Follow the [official Kaggle authentication documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md); never paste credentials into source files or commit them.

## Reproducibility and implementation limitations

- Raw data must contain exactly 568,454 rows and the ten original columns: `Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Score`, `Time`, `Summary`, `Text`. A mismatch stops sampling.
- The original CSV is stored in local `data/raw/Reviews.csv`. The default raw and sample manifests are generated locally in `outputs/`; they contain machine-specific paths and are ignored by Git. A fresh clone must run download and sampling before analysis, even though verified EDA results are already committed.
- Analysis requires the official Downloads sample and its local `outputs/sample_manifest.json`. The committed EDA JSON is a historical result snapshot, not a substitute manifest or another member's configuration. Its provenance includes the original run's local path and environment.
- Source bytes and row order, seed, and pinned pandas/NumPy versions support sample reproducibility. SHA-256 protects the bytes of each local file; identical CSV byte hashes across operating systems are not guaranteed because serialization, including line endings, can differ.
- A conflicting existing sample is rejected without overwriting. Recorded source-hash changes stop sampling before creating a new sample. The analysis checks sample SHA-256 before and after its work.
- Custom download/sample destinations use separate `<filename>.manifest.json` sidecars. The analysis script has no custom-path options and continues to use the default official sample.
- An interrupted download or extraction can leave `.partial` files. Inspect incomplete files before removing them and retrying; scripts do not automatically resume or clean them up. ZIP integrity and extracted CSV structure are validated.
- Running analysis rewrites the tracked statistics JSON, chart, and M1 summary. Inspect those changes before committing regenerated results. This documentation refactor does not execute data scripts or regenerate outputs.
- No missing values or duplicates are removed. The sentiment mapping, final model choices, and remaining team roles require group confirmation; model training and deployment are later work.

Raw/sample CSVs, Kaggle archives, virtual environments, credentials, and local manifests remain outside Git. The shared repository retains the verified EDA statistics, rating chart, and M1 summary.

## M2 continuation

The M1 procedures and verified results above are preserved. The [M2 feature engineering report](m2_feature_engineering.md) documents the new reusable `data_pipeline.py` interface, training-only BoW/TF-IDF features, standard shared split, duplicate-leakage audit, group-aware comparison, association rankings, and Jacob's integration example.

M2 additionally uses pinned scikit-learn 1.9.1 and SciPy 1.16.3 from the updated requirements. With the existing official sample and local sample manifest, run `python -m data_pipeline` using the virtual-environment interpreter; there is no need to rerun M1 sampling or EDA. Full Windows/macOS/Linux commands and tests are in the M2 report. The pipeline preserves the official CSV and M1 generated results. Local M2 split indices and metadata are ignored by Git; only the new summary JSON, figures, source, tests, and documentation are shareable.

The standard stratified 40,000/10,000 split remains the default. Any switch to duplicate-isolated grouping, optional Summary input, or final sentiment mapping requires group confirmation. Models and the application remain future work.
