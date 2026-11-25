"""Group Phi: Streamlit app for live Amazon food review sentiment prediction.

App Engineer deliverable. Serves the team's trained models once their
artifacts are committed under outputs/models/; until then it runs in setup
mode and shows the current dataset and feature-engineering results.

Run from the repository root:  python -m streamlit run app.py
"""

from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "outputs" / "models"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"


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
    return str(artifact["model"].predict(features)[0])


def prediction_page(models):
    st.header("Predict a review")
    if not models:
        st.info(
            "No trained models are committed yet. Live prediction turns on "
            "once the team's model artifacts land in outputs/models/ "
            "(M2/M3 model work)."
        )
        return
    choice = st.selectbox("Model", sorted(models))
    review = st.text_area("Paste an Amazon food review", height=150)
    if st.button("Predict"):
        if review.strip():
            st.write(f"**Sentiment:** {predict(review, models[choice])}")
        else:
            st.warning("Enter a review first.")


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
