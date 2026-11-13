"""Group Phi M2: fixed split, sparse text features, and training-only rankings.

Run ``python -m data_pipeline`` from the repository root. No classifier is trained.
"""

from dataclasses import dataclass
import hashlib
import html
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from scripts.common import EXPECTED_COLUMNS, ROOT, sample_path, sha256

SEED = 47
LABELS = ("Negative", "Neutral", "Positive")
LABEL_MAP = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}
LOCAL_DIR = ROOT / "outputs" / "m2" / "local"
VECTOR_CONFIG = dict(lowercase=False, stop_words=None, ngram_range=(1, 2),
                     min_df=2, max_df=1.0, max_features=20000)
PREPROCESSING = "html-unescape_tags-to-space_lowercase_whitespace-v1"


def runtime_versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("pandas", "numpy", "scipy", "scikit-learn")}}


def preprocess_text(text):
    """Stateless processing; preserve negation and reject invalid/empty inputs."""
    if not isinstance(text, str):
        raise ValueError("Text must be a nonmissing string.")
    text = re.sub(r"<[^>]*>", " ", html.unescape(text))
    text = " ".join(text.lower().split())
    if not text:
        raise ValueError("Text is empty after preprocessing.")
    return text


def load_official_sample():
    path = sample_path()
    before = sha256(path)
    manifest = json.loads((ROOT / "outputs" / "sample_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("seed") != SEED or manifest.get("sample_size") != 50000
            or manifest.get("sample_sha256") != before):
        raise ValueError("Official sample manifest/seed/size/SHA-256 mismatch.")
    df = pd.read_csv(path)
    if len(df) != 50000 or list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Expected the unchanged official 50,000-row, ten-column sample.")
    if not df.Score.isin(LABEL_MAP).all():
        raise ValueError("Score must be present and in 1–5.")
    cleaned = []
    for position, text in enumerate(df.Text):
        try:
            cleaned.append(preprocess_text(text))
        except ValueError as error:
            raise ValueError(f"Invalid Text at sample row position {position}: {error}") from error
    if sha256(path) != before:
        raise RuntimeError("Official sample changed while loading.")
    return df, np.asarray(cleaned, dtype=object), df.Score.map(LABEL_MAP).to_numpy(), before


def make_split(labels):
    """Exact row split: sklearn's stratification rounds class sizes to integers."""
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    return train_test_split(np.arange(len(labels)), test_size=0.2,
                            random_state=SEED, stratify=labels)


def shared_split(labels, sample_fingerprint, directory=LOCAL_DIR):
    """Validate/reuse local indices; changed state is never silently overwritten."""
    directory = Path(directory)
    indices_file, metadata_file = directory / "split_indices.npz", directory / "split_manifest.json"
    metadata = {"sample_sha256": sample_fingerprint, "rows": len(labels),
                "strategy": "stratified", "test_size": 0.2, "random_state": SEED,
                "preprocessing": PREPROCESSING, "versions": runtime_versions()}
    expected_train, expected_test = make_split(labels)
    if indices_file.exists() or metadata_file.exists():
        if not (indices_file.exists() and metadata_file.exists()):
            raise ValueError("Incomplete local split state; inspect it before retrying.")
        if json.loads(metadata_file.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Local split source/configuration/version mismatch; state preserved.")
        with np.load(indices_file, allow_pickle=False) as arrays:
            train, test = arrays["train_indices"].copy(), arrays["test_indices"].copy()
        if not (np.array_equal(train, expected_train) and np.array_equal(test, expected_test)):
            raise ValueError("Local split indices differ from the deterministic split; state preserved.")
        return train, test
    directory.mkdir(parents=True, exist_ok=True)
    with indices_file.open("xb") as stream:
        np.savez_compressed(stream, train_indices=expected_train, test_indices=expected_test)
    with metadata_file.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return expected_train, expected_test


def fit_representations(train_texts, test_texts, vector_config=None):
    """Only training text enters either vocabulary/IDF fit."""
    config = dict(VECTOR_CONFIG if vector_config is None else vector_config)
    bow = CountVectorizer(**config, dtype=np.int64)
    tfidf = TfidfVectorizer(**config, dtype=np.float64, sublinear_tf=True,
                           use_idf=True, smooth_idf=True, norm="l2")
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    return (csr_matrix(bow.fit_transform(train_texts)), csr_matrix(bow.transform(test_texts)),
            csr_matrix(tfidf.fit_transform(train_texts)), csr_matrix(tfidf.transform(test_texts)),
            bow, tfidf)


@dataclass
class FeatureBundle:
    X_train_bow: csr_matrix
    X_test_bow: csr_matrix
    X_train_tfidf: csr_matrix
    X_test_tfidf: csr_matrix
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names_bow: np.ndarray
    feature_names_tfidf: np.ndarray
    vectorizer_bow: CountVectorizer
    vectorizer_tfidf: TfidfVectorizer
    train_indices: np.ndarray
    test_indices: np.ndarray
    config: dict
    sample_fingerprint: str


def build_features():
    """Return Jacob's shared sparse inputs; reuse the protected local split."""
    _, texts, labels, fingerprint = load_official_sample()
    train, test = shared_split(labels, fingerprint)
    Xb, Xbt, Xt, Xtt, bow, tfidf = fit_representations(texts[train], texts[test])
    return FeatureBundle(Xb, Xbt, Xt, Xtt, labels[train], labels[test],
                         bow.get_feature_names_out(), tfidf.get_feature_names_out(),
                         bow, tfidf, train, test,
                         {"vectorizer": dict(VECTOR_CONFIG), "preprocessing": PREPROCESSING,
                          "sentiment_mapping_status": "Provisional; pending group confirmation"}, fingerprint)


def rank_features(X, labels, names, representation, top_n=10):
    """Descriptive association scores, NOT learned classifier coefficients.

    Each target is compared with BOTH other classes, including Neutral.
    Only one-dimensional vocabulary summaries become dense.
    """
    if representation not in ("bow", "tfidf"):
        raise ValueError("representation must be bow or tfidf")
    labels, names = np.asarray(labels), np.asarray(names)
    result = {}
    for target in ("Positive", "Negative"):
        mask = labels == target
        if not mask.any() or mask.all():
            raise ValueError(f"Both {target} and rest must occur in training labels.")
        if representation == "bow":
            target_sum = np.asarray(X[mask].sum(axis=0)).ravel()
            rest_sum = np.asarray(X[~mask].sum(axis=0)).ravel()
            score = (np.log((target_sum + 1) / (target_sum.sum() + X.shape[1]))
                     - np.log((rest_sum + 1) / (rest_sum.sum() + X.shape[1])))
        else:
            score = (np.asarray(X[mask].mean(axis=0)).ravel()
                     - np.asarray(X[~mask].mean(axis=0)).ravel())
        order = np.lexsort((names, -score))[:top_n]
        result[target] = [{"feature": str(names[i]), "score": float(score[i])} for i in order]
    return result


def class_distribution(labels):
    labels = np.asarray(labels)
    return {label: {"count": int(np.sum(labels == label)),
                    "percentage": float(np.mean(labels == label) * 100)} for label in LABELS}


def text_overlap(texts, train, test):
    texts = np.asarray(texts)
    shared = set(texts[train]) & set(texts[test])
    return {"distinct_overlapping_texts": len(shared),
            "affected_train_rows": sum(text in shared for text in texts[train]),
            "affected_test_rows": sum(text in shared for text in texts[test])}


def split_description(raw, cleaned, labels, train, test):
    return {"train_size": len(train), "test_size": len(test),
            "test_percentage": len(test) / len(labels) * 100,
            "train_classes": class_distribution(labels[train]),
            "test_classes": class_distribution(labels[test]),
            "raw_text_overlap": text_overlap(raw, train, test),
            "preprocessed_text_overlap": text_overlap(cleaned, train, test)}


def audit_splits(raw, cleaned, labels, train, test):
    # Fixed first fold: do not select a fold based on test performance or appearance.
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
    grouped_train, grouped_test = next(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels, groups=cleaned))
    groups = pd.DataFrame({"raw": raw, "cleaned": cleaned, "label": labels})
    group_sizes = groups.groupby("raw").size()
    return {"default_stratified": split_description(raw, cleaned, labels, train, test),
            "group_aware_comparison": split_description(raw, cleaned, labels, grouped_train, grouped_test),
            "group_aware_method": "StratifiedGroupKFold(5, shuffle=True, random_state=47), first fold, grouped by preprocessed Text; comparison only",
            "raw_distinct_texts": int(len(group_sizes)),
            "raw_repeated_text_groups": int((group_sizes > 1).sum()),
            "raw_extra_occurrences": int((group_sizes - 1).sum()),
            "raw_groups_with_conflicting_labels": int((groups.groupby("raw").label.nunique() > 1).sum()),
            "preprocessed_distinct_texts": int(groups.cleaned.nunique()),
            "preprocessed_groups_with_conflicting_labels": int((groups.groupby("cleaned").label.nunique() > 1).sum())}


def make_figure(rankings, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
    for ax, target, color in zip(axes, ("Positive", "Negative"), ("#287A58", "#B94B47")):
        rows = rankings[target][::-1]
        values = [row["score"] for row in rows]
        bars = ax.barh([row["feature"] for row in rows], values, color=color)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=9)
        ax.set_xlim(0, max(values) * 1.26)
        ax.set_title(f"{target} versus rest (including Neutral)", fontsize=11)
        ax.set_xlabel("Training-only association score")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    before = sha256(sample_path())
    bundle = build_features()
    df, cleaned, labels, fingerprint = load_official_sample()
    if fingerprint != bundle.sample_fingerprint:
        raise RuntimeError("Sample changed between feature construction and audit.")
    rankings = {"bow": rank_features(bundle.X_train_bow, bundle.y_train, bundle.feature_names_bow, "bow"),
                "tfidf": rank_features(bundle.X_train_tfidf, bundle.y_train, bundle.feature_names_tfidf, "tfidf")}
    split = audit_splits(df.Text.to_numpy(), cleaned, labels, bundle.train_indices, bundle.test_indices)
    report = {"group": "Phi", "sample_sha256": fingerprint, "sample_size": len(labels),
              "seed": SEED, "versions": runtime_versions(), "configuration": bundle.config,
              "tfidf_settings": {"sublinear_tf": True, "use_idf": True, "smooth_idf": True,
                                 "norm": "l2", "dtype": "float64"},
              "split_audit": split,
              "vocabulary_sizes": {"bow": len(bundle.feature_names_bow), "tfidf": len(bundle.feature_names_tfidf)},
              "matrix_shapes": {name: list(getattr(bundle, name).shape) for name in (
                  "X_train_bow", "X_test_bow", "X_train_tfidf", "X_test_tfidf")},
              "train_indices_sha256": hashlib.sha256(bundle.train_indices.astype("<i8").tobytes()).hexdigest(),
              "test_indices_sha256": hashlib.sha256(bundle.test_indices.astype("<i8").tobytes()).hexdigest(),
              "feature_names_sha256": {name: hashlib.sha256(json.dumps(getattr(bundle, "feature_names_" + name).tolist(), ensure_ascii=False).encode()).hexdigest() for name in ("bow", "tfidf")},
              "ranking_methods": {"bow": "Add-one smoothed class-versus-rest token log probability ratio (not log-odds)",
                                  "tfidf": "Class mean minus rest mean TF-IDF weight",
                                  "rest": "Both other sentiment classes, including Neutral",
                                  "tie_break": "Feature name ascending", "scope": "Training rows only; not classifier coefficients"},
              "top_features": rankings,
              "recommendation": "Keep exact 80/20 stratified baseline for shared M2 comparison pending group approval; recommend group-aware split for leakage-isolated evaluation, accepting approximate row/class proportions."}
    if sha256(sample_path()) != before:
        raise RuntimeError("Official sample changed; do not publish generated outputs.")
    for name, title in (("bow", "BoW: top sentiment-associated features"), ("tfidf", "TF-IDF: top sentiment-associated features")):
        make_figure(rankings[name], title, ROOT / "outputs" / "figures" / f"m2_{name}_features.png")
    output = ROOT / "outputs" / "m2" / "feature_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
