import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.naive_bayes import MultinomialNB

from data_pipeline import build_features, load_official_sample


def main():
    bundle = build_features()
    df, _, _, _ = load_official_sample()

    bow_model = MultinomialNB()
    bow_model.fit(bundle.X_train_bow, bundle.y_train)
    bow_predictions = bow_model.predict(bundle.X_test_bow)
    bow_accuracy = accuracy_score(bundle.y_test, bow_predictions)
    bow_cm = confusion_matrix(
        bundle.y_test, bow_predictions, labels=["Negative", "Neutral", "Positive"]
    )
    print(f"Naive Bayes (BoW) Accuracy: {bow_accuracy:.2%}")
    print(classification_report(bundle.y_test, bow_predictions, zero_division=0))
    print(bow_cm)

    tfidf_model = MultinomialNB()
    tfidf_model.fit(bundle.X_train_tfidf, bundle.y_train)
    tfidf_predictions = tfidf_model.predict(bundle.X_test_tfidf)
    tfidf_accuracy = accuracy_score(bundle.y_test, tfidf_predictions)
    tfidf_cm = confusion_matrix(
        bundle.y_test, tfidf_predictions, labels=["Negative", "Neutral", "Positive"]
    )
    print(f"Naive Bayes (TF-IDF) Accuracy: {tfidf_accuracy:.2%}")
    print(classification_report(bundle.y_test, tfidf_predictions, zero_division=0))
    print(tfidf_cm)

    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(bundle.X_train_bow, bundle.y_train)
    baseline_predictions = baseline_model.predict(bundle.X_test_bow)
    baseline_accuracy = accuracy_score(bundle.y_test, baseline_predictions)
    baseline_cm = confusion_matrix(
        bundle.y_test, baseline_predictions, labels=["Negative", "Neutral", "Positive"]
    )
    print(f"Majority-Class Baseline Accuracy: {baseline_accuracy:.2%}")
    print(classification_report(bundle.y_test, baseline_predictions, zero_division=0))
    print(baseline_cm)

    incorrect = bundle.y_test != bow_predictions
    incorrect_indices = np.where(incorrect)[0]
    print(incorrect_indices[:10])
    print(f"Total BoW misclassifications: {len(incorrect_indices)}")

    for review_number, test_position in enumerate(incorrect_indices[:10], start=1):
        original_position = bundle.test_indices[test_position]
        review_text = df.iloc[original_position]["Text"]
        rating = df.iloc[original_position]["Score"]
        actual = bundle.y_test[test_position]
        predicted = bow_predictions[test_position]

        print(f"Review: {review_number}")
        print(f"Rating: {rating}")
        print(f"Actual: {actual}")
        print(f"Predicted: {predicted}")
        print(f"Text: {review_text}")
        print("-" * 60)

    logistic_regression_model = LogisticRegression(max_iter=1000)
    logistic_regression_model.fit(bundle.X_train_tfidf, bundle.y_train)
    logistic_regression_predictions = logistic_regression_model.predict(
        bundle.X_test_tfidf
    )
    logistic_regression_accuracy = accuracy_score(
        bundle.y_test, logistic_regression_predictions
    )
    logistic_regression_cm = confusion_matrix(
        bundle.y_test,
        logistic_regression_predictions,
        labels=["Negative", "Neutral", "Positive"],
    )
    print(f"Logistic Regression (TF-IDF) Accuracy: {logistic_regression_accuracy:.2%}")
    print(
        classification_report(
            bundle.y_test, logistic_regression_predictions, zero_division=0
        )
    )
    print(logistic_regression_cm)

    logistic_bow_model = LogisticRegression(max_iter=1000)
    logistic_bow_model.fit(bundle.X_train_bow, bundle.y_train)
    logistic_bow_predictions = logistic_bow_model.predict(bundle.X_test_bow)
    logistic_bow_accuracy = accuracy_score(bundle.y_test, logistic_bow_predictions)
    logistic_bow_cm = confusion_matrix(
        bundle.y_test,
        logistic_bow_predictions,
        labels=["Negative", "Neutral", "Positive"],
    )
    print(f"Logistic Regression (BoW) Accuracy: {logistic_bow_accuracy:.2%}")
    print(
        classification_report(bundle.y_test, logistic_bow_predictions, zero_division=0)
    )
    print(logistic_bow_cm)


if __name__ == "__main__":
    main()
