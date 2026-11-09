# AI-Powered Text Intelligence Platform — Sentiment Analysis of Amazon Food Reviews

**Group Phi · CSE 3521 · The Ohio State University · Autumn 2026**

## Project Overview

We are building a sentiment analysis platform for Amazon food review text. The planned system will compare at least two text feature representations, at least two classical machine learning models, and at least one deep/modern NLP model. Evaluation will report accuracy, precision, recall, F1-score, and confusion matrices. A planned Streamlit application will support live sentiment prediction using at least two trained models.

**Current progress:** fixed-sample data preparation and exploratory data analysis (EDA) are complete. Models have not been trained, and the application has not been developed.

## Team Members

| Member | Confirmed role |
| --- | --- |
| Bowen Zheng | Data Engineer |
| Wesley Kamau | App Engineer |

Other team members and roles are pending confirmation.

## Technical Approach

Amazon Reviews → Text Preprocessing → Feature Engineering → ML Models → Evaluation → Streamlit App

- Proposed feature representations: **Bag of Words** and **TF-IDF**.
- Candidate classical models: **Naive Bayes** and **Logistic Regression**.
- Candidate deep model: **Feedforward Neural Network**.

These are plans, not implemented features. Final model selections and the sentiment labeling scheme remain subject to group confirmation. The provisional labels are scores 1–2 → Negative, 3 → Neutral, and 4–5 → Positive.

## Project Milestones

| Milestone | Due | Verified status |
| --- | --- | --- |
| M1: Project Charter & Scope | Nov 10, 2026 | Data preparation and EDA complete; full charter completion/submission unconfirmed |
| M2: Initial Solution | Nov 15, 2026 | Planned |
| M3: Refined Models & App Prototype | Nov 22, 2026 | Planned |
| M4: Final Report | Nov 29, 2026 | Planned |
| Final Demo | See Carmen | Planned |

## Dataset

We use [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews): **568,454 original reviews**, with an official Group Phi sample of exactly **50,000 reviews**, sampled using **random seed 47**. Original and sampled CSV files are not committed to GitHub.

Verified EDA finds 38,901 positive (77.802%), 7,390 negative (14.780%), and 3,709 neutral (7.418%) reviews under the provisional mapping. Mean review length is 80.43 words. Summary and ProfileName each have two missing values; there are no complete-row duplicates and 3,674 additional occurrences of review texts. The 10.49:1 largest/smallest class ratio indicates class imbalance.

See the [data engineering documentation](docs/data_engineering.md), [M1 dataset summary](docs/m1_dataset_summary.md), [full statistics](outputs/eda_statistics.json), and [rating distribution chart](outputs/figures/score_distribution.png).

## Getting Started

Use Python 3.11. Clone the shared repository:

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

The scripts download the source, create or verify the official sample in the operating system's Downloads directory, then analyze it. Run them in that order: local manifests are generated during reproduction and are not included in a clone. See [reproduction details and limitations](docs/data_engineering.md#complete-reproduction) before regenerating tracked outputs or handling a sample mismatch.

## Project Structure

Shared files committed to the repository:

```text
amazon-sentiment-analysis/
├── docs/
│   ├── data_engineering.md
│   └── m1_dataset_summary.md
├── outputs/
│   ├── eda_statistics.json
│   └── figures/score_distribution.png
├── scripts/
│   ├── common.py
│   ├── download_data.py
│   ├── sample_data.py
│   └── analyze_data.py
├── requirements.txt
├── README.md
└── .gitignore
```

Local runtime files are separate: `.venv/`, `data/raw/`, generated provenance manifests, and the official CSV in the system Downloads directory are not shared through Git.

## Contributing

After cloning, create a feature branch and commit only the files relevant to your work:

```sh
git switch -c feature/short-description
# Make and review your changes.
git status
git diff
git add <specific-files>
git commit -m "Describe your change"
git push -u origin feature/short-description
```

Open a pull request targeting `main` on [GitHub](https://github.com/BOWEN2491/amazon-sentiment-analysis) and request a teammate's review. Do not commit datasets, credentials, virtual environments, or local manifests. Coordinate changes to the fixed sample, proposed labels, and model choices with the group.
