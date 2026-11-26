"""Group Phi: Streamlit app for live Amazon food review sentiment prediction.

App Engineer deliverable. Serves the team's trained classical models from
outputs/models/ (see scripts/train_app_models.py) with side-by-side live
predictions and per-class confidence. If no artifacts are committed yet it
runs in setup mode and shows the current dataset and feature-engineering
results.

Run from the repository root:  python -m streamlit run app.py
"""

from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "outputs" / "models"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"

SENTIMENT_COLORS = {"Negative": "#d9534f", "Neutral": "#8a8a8a", "Positive": "#3a9d5d"}


@st.cache_resource
def load_models():
    """Load committed model artifacts (joblib dicts with vectorizer + model)."""
    models = {}
    if not MODEL_DIR.is_dir():
        return models
    import joblib

    for path in sorted(MODEL_DIR.glob("*.joblib")):
        models[path.stem] = joblib.load(path)
    return models


def predict(text, artifact):
    from data_pipeline import preprocess_text

    cleaned = preprocess_text(text)
    features = artifact["vectorizer"].transform([cleaned])
    label = str(artifact["model"].predict(features)[0])
    probabilities = None
    if hasattr(artifact["model"], "predict_proba"):
        classes = [str(c) for c in artifact["model"].classes_]
        probabilities = dict(zip(classes, artifact["model"].predict_proba(features)[0]))
    return label, probabilities


def model_label(name, artifact):
    metadata = artifact.get("metadata", {})
    estimator = metadata.get("estimator", "")
    representation = metadata.get("representation", "")
    if estimator and representation:
        return f"{estimator} ({representation.upper()})"
    return name.replace("_", " ").title()


def prediction_page(models):
    st.header("Predict a review")
    if not models:
        st.info(
            "No trained models are committed yet. Live prediction turns on "
            "once the team's model artifacts land in outputs/models/ "
            "(M2/M3 model work)."
        )
        return
    review = st.text_area("Paste an Amazon food review", height=150)
    if st.button("Predict", type="primary"):
        if not review.strip():
            st.warning("Enter a review first.")
            return
        results = {}
        for name, artifact in sorted(models.items()):
            try:
                results[name] = predict(review, artifact)
            except ValueError as error:
                st.warning(str(error))
                return
        st.subheader("Model agreement")
        columns = st.columns(len(results))
        for column, (name, (label, probabilities)) in zip(columns, results.items()):
            with column:
                st.markdown(f"**{model_label(name, models[name])}**")
                color = SENTIMENT_COLORS.get(label, "#333333")
                st.markdown(
                    f"<span style='font-size:1.6rem;font-weight:700;color:{color}'>"
                    f"{label}</span>",
                    unsafe_allow_html=True,
                )
                if probabilities:
                    st.caption(f"Confidence: {max(probabilities.values()):.1%}")
        st.subheader("Per-class confidence")
        import pandas as pd

        chart_rows = []
        for name, (label, probabilities) in results.items():
            if not probabilities:
                continue
            for sentiment, probability in probabilities.items():
                chart_rows.append(
                    {
                        "Model": model_label(name, models[name]),
                        "Sentiment": sentiment,
                        "Probability": round(float(probability), 4),
                    }
                )
        if chart_rows:
            frame = pd.DataFrame(chart_rows)
            for sentiment in ("Negative", "Neutral", "Positive"):
                subset = frame[frame["Sentiment"] == sentiment].set_index("Model")[
                    ["Probability"]
                ]
                st.markdown(f"**{sentiment}**")
                st.bar_chart(subset, horizontal=True)
        with st.expander("Model details"):
            for name, artifact in sorted(models.items()):
                metadata = artifact.get("metadata", {})
                st.markdown(f"**{model_label(name, artifact)}** (`{name}.joblib`)")
                st.json(metadata)


def overview_page():
    st.header("Dataset and features")
    figures = [
        ("score_distribution.png", "Sentiment class distribution (50,000-review sample)"),
        ("m2_bow_features.png", "Top Bag of Words features"),
        ("m2_tfidf_features.png", "Top TF-IDF features"),
    ]
    shown = False
    for filename, caption in figures:
        path = FIGURES_DIR / filename
        if path.exists():
            st.image(str(path), caption=caption)
            shown = True
    if not shown:
        st.info("No output figures found. Run the pipeline scripts first.")


def main():
    st.set_page_config(page_title="Group Phi Sentiment Analysis")
    st.title("Amazon Food Review Sentiment Analysis")
    st.caption("Group Phi · CSE 3521 · The Ohio State University")
    page = st.sidebar.radio("Page", ["Predict", "Dataset overview"])
    if page == "Predict":
        prediction_page(load_models())
    else:
        overview_page()


if __name__ == "__main__":
    main()
