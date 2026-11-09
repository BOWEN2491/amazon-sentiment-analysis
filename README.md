# Amazon Food Review Sentiment Analysis — Group Phi

CSE 3521 (The Ohio State University), Autumn 2026. M1 covers data engineering and exploratory analysis only; no models are trained here. Group Phi's fixed official dataset is **50,000 reviews sampled with random seed 47**.

## Setup

Python 3.11 is used. From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

On macOS/Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

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

No cleaning, filtering, shuffling, or index reset occurs before sampling. The script uses `platformdirs.user_downloads_path()` to resolve the system Downloads folder, including redirected Windows folders. On this machine the official file is `D:\Downloads\group_sample_seed47.csv`. An existing sample is compared to the exact expected content and row order, reused if identical, and never silently overwritten.

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

- `outputs/eda_statistics.json`: full machine-readable results, metric definitions, versions, and sample hash.
- `outputs/figures/score_distribution.png`: 300-DPI rating bar chart with counts and percentages, suitable for the M1 PDF.
- `docs/m1_dataset_summary.md`: concise English text ready to paste into the group charter.
- `outputs/raw_manifest.json` and `outputs/sample_manifest.json`: provenance and reproducibility records.

### Complete reproduction

```powershell
Set-Location 'D:\Coursework\CSE 3521\amazon-sentiment-analysis'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/sample_data.py
.\.venv\Scripts\python.exe scripts/analyze_data.py
```

For an existing environment, skip its creation and installation. Existing validated source/sample files are reused. If a sample or recorded source hash differs, investigate rather than deleting or replacing the fixed official sample. An interrupted download leaves a `.partial` file; inspect that incomplete file before removing it and retrying. The public Kaggle download succeeded without authentication during this run.

## Project structure

```text
amazon-sentiment-analysis/
├── data/raw/                 # Local Reviews.csv and Kaggle ZIP (ignored)
├── scripts/
│   ├── common.py             # Paths, hashing, source validation
│   ├── download_data.py
│   ├── sample_data.py
│   └── analyze_data.py
├── outputs/
│   ├── figures/score_distribution.png
│   ├── raw_manifest.json
│   ├── sample_manifest.json
│   └── eda_statistics.json
├── docs/m1_dataset_summary.md
├── requirements.txt
├── README.md
├── DECISIONS.md
├── task_plan.md
├── findings.md
├── progress.md
└── .gitignore
```

The official sample is stored separately in the OS Downloads folder, not in the repository. The local `.venv/` folder is ignored.

## Remaining group decision

Confirm the proposed three-class sentiment mapping. There are no unresolved data acquisition or analysis blockers; model training and deployment belong to later milestones.

## Local repository and data handling

Data CSVs, archives, databases, virtual environments, and credentials are excluded by `.gitignore`. Keep raw/sample files local. This project has no remote and is not pushed to GitHub.
