"""Analyze Group Phi's fixed sample without changing its contents."""

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd

from common import EXPECTED_COLUMNS, ROOT, sample_path, sha256, versions, write_json

SENTIMENT_MAP = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]


def distribution(values, order, total):
    counts = values.value_counts().reindex(order, fill_value=0)
    return {str(label): {"count": int(counts[label]), "percentage": float(counts[label] / total * 100)}
            for label in order}


def analyze(df):
    """All percentages use the full sample; duplicate counts exclude first occurrences."""
    total = len(df)
    sentiments = df["Score"].map(SENTIMENT_MAP)
    sentiment_counts = distribution(sentiments, SENTIMENT_ORDER, total)
    sizes = [item["count"] for item in sentiment_counts.values()]
    ratio = max(sizes) / min(sizes) if min(sizes) else None
    # Python's default str.split() splits on whitespace; original text is not cleaned.
    word_counts = df["Text"].dropna().map(lambda text: len(text.split()))
    return {
        "total_reviews": total,
        "rating_distribution": distribution(df["Score"], range(1, 6), total),
        "sentiment_distribution": sentiment_counts,
        "sentiment_mapping": {str(score): label for score, label in SENTIMENT_MAP.items()},
        "sentiment_mapping_status": "Provisional; pending group confirmation",
        "word_count": {
            "mean": float(word_counts.mean()) if len(word_counts) else None,
            "nonmissing_texts": len(word_counts),
            "missing_texts_excluded": int(df["Text"].isna().sum()),
            "whitespace_only_texts": int((word_counts == 0).sum()),
            "definition": "Original Text split on whitespace; missing text excluded, whitespace-only text counted as zero; HTML and punctuation retained.",
        },
        "missing_values": {
            col: {"count": int(df[col].isna().sum()), "percentage": float(df[col].isna().mean() * 100)}
            for col in df.columns
        },
        "duplicates": {
            "exact_rows_extra_occurrences": int(df.duplicated().sum()),
            "review_texts_extra_occurrences": int(df["Text"].dropna().duplicated().sum()),
            "definition": "Extra occurrences after the first: exact rows compare all original columns; review texts compare nonmissing Text exactly.",
        },
        "class_imbalance": {
            "largest_to_smallest_ratio": ratio,
            "missing_classes": [label for label, item in sentiment_counts.items() if item["count"] == 0],
            "threshold": 2.0,
            "is_imbalanced": min(sizes) == 0 or ratio >= 2.0,
            "interpretation": "Descriptive largest/smallest class ratio >= 2; not an inferential significance test. A missing class is imbalanced and its ratio is undefined.",
        },
    }


def make_chart(stats, path):
    ratings = list(range(1, 6))
    counts = [stats["rating_distribution"][str(score)]["count"] for score in ratings]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        bars = ax.bar(ratings, counts, width=0.64, color="#315F85", zorder=3)
        ax.set_title("Amazon Food Review Ratings — Group Phi", fontsize=15, fontweight="bold", pad=20)
        ax.set_xlabel("Review rating (stars)", labelpad=10)
        ax.set_ylabel("Number of reviews", labelpad=10)
        ax.set_xticks(ratings)
        ax.set_ylim(0, max(counts) * 1.18)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#DDE3E8", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, score in zip(bars, ratings):
            item = stats["rating_distribution"][str(score)]
            ax.annotate(f"{item['count']:,}\n({item['percentage']:.2f}%)",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=10)
        fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
        fig.text(0.5, 0.02, "Fixed sample: 50,000 reviews | Seed: 47", ha="center", fontsize=10, color="#4D5963")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def dataset_summary(stats):
    classes = stats["sentiment_distribution"]
    label_text = ", ".join(f"{label.lower()} {classes[label]['count']:,} ({classes[label]['percentage']:.2f}%)"
                           for label in SENTIMENT_ORDER)
    missing = stats["missing_values"]
    other_missing = ", ".join(f"{col} ({item['count']:,})" for col, item in missing.items()
                              if item["count"] and col not in ("Text", "Summary", "Score")) or "none"
    ratio = stats["class_imbalance"]["largest_to_smallest_ratio"]
    ratio_text = f"{ratio:.2f}:1" if ratio is not None else "undefined because a class is absent"
    conclusion = "class imbalance" if stats["class_imbalance"]["is_imbalanced"] else "no class imbalance under this criterion"
    return (
        "# Dataset Summary\n\n"
        "We use Amazon Fine Food Reviews from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), "
        "originally collected by [Stanford SNAP](https://snap.stanford.edu/data/web-FineFoods.html). "
        "The source contains 568,454 reviews. Group Phi generated a fixed sample of 50,000 reviews "
        "using pandas random sampling without replacement and random_state=47, with no prior cleaning or reordering. "
        "The official sampled CSV is preserved unchanged for subsequent project work.\n\n"
        "The proposed mapping assigns scores 1–2 to Negative, 3 to Neutral, and 4–5 to Positive; "
        "this mapping remains subject to group confirmation. The resulting distribution is "
        f"{label_text}. Mean review length is {stats['word_count']['mean']:.2f} words, "
        "calculated by whitespace splitting of original Text; missing text is excluded and whitespace-only text counts as zero. "
        "HTML and punctuation are retained.\n\n"
        f"Missing values are Text: {missing['Text']['count']:,}, Summary: {missing['Summary']['count']:,}, "
        f"and Score: {missing['Score']['count']:,}; other missing fields are {other_missing}. "
        f"There are {stats['duplicates']['exact_rows_extra_occurrences']:,} duplicate rows across all original columns "
        f"and {stats['duplicates']['review_texts_extra_occurrences']:,} repeated nonmissing review texts, "
        "counting occurrences after the first. No missing values or duplicates were removed. "
        f"The largest-to-smallest sentiment class ratio is {ratio_text}, indicating {conclusion} "
        "using a descriptive 2:1 threshold, rather than a significance test. "
        "The accompanying 300-DPI bar chart displays counts and percentages for ratings 1–5.\n"
    )


def main():
    path = sample_path().resolve()
    before_hash = sha256(path)
    manifest = json.loads((ROOT / "outputs" / "sample_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("seed") != 47 or manifest.get("sample_size") != 50000:
        raise ValueError("Official sample manifest must specify seed 47 and 50,000 rows.")
    if manifest.get("sample_sha256") != before_hash:
        raise ValueError("Official sample SHA-256 does not match its manifest; analysis stopped.")
    df = pd.read_csv(path)
    if len(df) != 50000 or list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Expected the official sample with 50,000 rows and the ten original columns.")
    if not df["Score"].isin(range(1, 6)).all():
        raise ValueError("The official sample contains missing or invalid scores; investigate before labeling.")
    stats = analyze(df)
    stats.update({"group": "Phi", "seed": 47, "original_dataset_size": 568454,
                  "sample_path": str(path), "sample_sha256": before_hash,
                  "generated_at_utc": datetime.now(timezone.utc).isoformat(), "versions": versions()})
    chart = ROOT / "outputs" / "figures" / "score_distribution.png"
    summary = ROOT / "docs" / "m1_dataset_summary.md"
    make_chart(stats, chart)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(dataset_summary(stats), encoding="utf-8")
    if sha256(path) != before_hash:
        raise RuntimeError("Official sample changed during analysis; generated outputs must be discarded.")
    stats["sample_unchanged"] = True
    stats["rating_chart"] = str(chart.relative_to(ROOT))
    stats["dataset_summary"] = str(summary.relative_to(ROOT))
    write_json(ROOT / "outputs" / "eda_statistics.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
