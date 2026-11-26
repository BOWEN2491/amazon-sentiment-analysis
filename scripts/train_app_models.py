"""Train the tuned classical models and export app-ready artifacts.

Fits the M3 tuned configurations from tune_classical.py on the official
training split and writes ``outputs/models/*.joblib`` artifacts that the
Streamlit app (app.py) loads for live prediction. Each artifact is a dict
with the fitted vectorizer, the fitted model, and evaluation metadata.

Run from the repository root:  python scripts/train_app_models.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline import build_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "models"

CANDIDATES = {
    "logistic_regression_tfidf": {
        "representation": "tfidf",
        "model": LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000),
        "hyperparameters": {"C": 10.0, "class_weight": "balanced", "max_iter": 1000},
    },
    "naive_bayes_tfidf": {
        "representation": "tfidf",
        "model": MultinomialNB(alpha=0.1),
        "hyperparameters": {"alpha": 0.1},
    },
    "logistic_regression_bow": {
        "representation": "bow",
        "model": LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000),
        "hyperparameters": {"C": 0.1, "class_weight": "balanced", "max_iter": 1000},
    },
}


def main():
    bundle = build_features()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, spec in CANDIDATES.items():
        if spec["representation"] == "tfidf":
            X_train, X_test = bundle.X_train_tfidf, bundle.X_test_tfidf
            vectorizer = bundle.vectorizer_tfidf
        else:
            X_train, X_test = bundle.X_train_bow, bundle.X_test_bow
            vectorizer = bundle.vectorizer_bow
        model = spec["model"]
        model.fit(X_train, bundle.y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(bundle.y_test, predictions)
        macro_f1 = f1_score(bundle.y_test, predictions, average="macro", zero_division=0)
        report = classification_report(bundle.y_test, predictions, zero_division=0)
        print(f"{name}: accuracy={accuracy:.4f} macro_f1={macro_f1:.4f}")
        print(report)
        artifact = {
            "vectorizer": vectorizer,
            "model": model,
            "metadata": {
                "name": name,
                "estimator": type(model).__name__,
                "representation": spec["representation"],
                "hyperparameters": spec["hyperparameters"],
                "test_accuracy": round(float(accuracy), 4),
                "test_macro_f1": round(float(macro_f1), 4),
                "trained_on": "Official 50,000-review sample, group split fold 0 (seed 47)",
                "sample_sha256": bundle.sample_fingerprint,
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        joblib.dump(artifact, MODEL_DIR / f"{name}.joblib")
        summary[name] = artifact["metadata"]
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(summary)} artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
