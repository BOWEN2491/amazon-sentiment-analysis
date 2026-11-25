from time import perf_counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB

from data_pipeline import build_features, fit_representations, load_official_sample


def main():
    bundle = build_features()
    _, texts, _, _ = load_official_sample()

    training_texts = texts[bundle.train_indices]
    validation_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=47
    )
    train_idx, val_idx = next(
        validation_splitter.split(training_texts, bundle.y_train, groups=training_texts)
    )
    train_groups = set(training_texts[train_idx])
    val_groups = set(training_texts[val_idx])
    print(f"Overlapping texts: {len(train_groups & val_groups)}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")

    subtrain_texts = training_texts[train_idx]
    validation_texts = training_texts[val_idx]
    subtrain_labels = bundle.y_train[train_idx]
    validation_labels = bundle.y_train[val_idx]
    (X_subtrain_bow, X_val_bow, X_subtrain_tfidf, X_val_tfidf) = fit_representations(
        subtrain_texts, validation_texts
    )
    print(f"BoW training shape: {X_subtrain_bow.shape}")
    print(f"BoW validation shape: {X_val_bow.shape}")
    print(f"TF-IDF training shape: {X_subtrain_tfidf.shape}")
    print(f"TF-IDF validation shape: {X_val_tfidf.shape}")

    alpha_values = [0.1, 0.5, 1.0, 2.0]
    for alpha in alpha_values:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_subtrain_bow, subtrain_labels)
        predictions = model.predict(X_val_bow)
        macro_f1 = f1_score(
            validation_labels, predictions, average="macro", zero_division=0
        )
        neutral_recall = recall_score(
            validation_labels,
            predictions,
            labels=["Neutral"],
            average="macro",
            zero_division=0,
        )
        print(
            f"Naive Bayes (BoW) | Alpha: {alpha} | Validation Macro-F1: {macro_f1:.4f} | Neutral Recall: {neutral_recall:.4f}"
        )
    for alpha in alpha_values:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_subtrain_tfidf, subtrain_labels)
        predictions = model.predict(X_val_tfidf)
        macro_f1 = f1_score(
            validation_labels, predictions, average="macro", zero_division=0
        )
        neutral_recall = recall_score(
            validation_labels,
            predictions,
            labels=["Neutral"],
            average="macro",
            zero_division=0,
        )
        print(
            f"Naive Bayes (TF-IDF) | Alpha: {alpha} | Validation Macro-F1: {macro_f1:.4f} | Neutral Recall: {neutral_recall:.4f}"
        )

    c_values = [0.1, 1.0, 10.0]
    class_weights = [None, "balanced"]
    for c in c_values:
        for weight in class_weights:
            model = LogisticRegression(C=c, class_weight=weight, max_iter=1000)
            model.fit(X_subtrain_tfidf, subtrain_labels)
            predictions = model.predict(X_val_tfidf)
            macro_f1 = f1_score(
                validation_labels, predictions, average="macro", zero_division=0
            )
            neutral_recall = recall_score(
                validation_labels,
                predictions,
                labels=["Neutral"],
                average="macro",
                zero_division=0,
            )
            print(
                f"Logistic Regression (TF-IDF) | C: {c} | Class Weight: {weight} | Validation Macro-F1: {macro_f1:.4f} | Neutral Recall: {neutral_recall:.4f}"
            )
    for c in c_values:
        for weight in class_weights:
            model = LogisticRegression(C=c, class_weight=weight, max_iter=1000)
            model.fit(X_subtrain_bow, subtrain_labels)
            predictions = model.predict(X_val_bow)
            macro_f1 = f1_score(
                validation_labels, predictions, average="macro", zero_division=0
            )
            neutral_recall = recall_score(
                validation_labels,
                predictions,
                labels=["Neutral"],
                average="macro",
                zero_division=0,
            )
            print(
                f"Logistic Regression (BoW) | C: {c} | Class Weight: {weight} | Validation Macro-F1: {macro_f1:.4f} | Neutral Recall: {neutral_recall:.4f}"
            )

    tuned_logistic_tfidf_model = LogisticRegression(
        C=10.0, class_weight="balanced", max_iter=1000
    )
    start_time = perf_counter()
    tuned_logistic_tfidf_model.fit(bundle.X_train_tfidf, bundle.y_train)
    tuned_logistic_tfidf_training_time = perf_counter() - start_time
    print(
        f"Logistic Regression (TF-IDF) training time: {tuned_logistic_tfidf_training_time:.4f} seconds"
    )
    tuned_logistic_tfidf_predictions = tuned_logistic_tfidf_model.predict(
        bundle.X_test_tfidf
    )
    tuned_logistic_tfidf_accuracy = accuracy_score(
        bundle.y_test, tuned_logistic_tfidf_predictions
    )
    tuned_logistic_tfidf_cm = confusion_matrix(
        bundle.y_test,
        tuned_logistic_tfidf_predictions,
        labels=["Negative", "Neutral", "Positive"],
    )
    tuned_logistic_tfidf_f1 = f1_score(
        bundle.y_test,
        tuned_logistic_tfidf_predictions,
        average="macro",
        zero_division=0,
    )
    print(
        f"Tuned Logistic Regression (TF-IDF) Accuracy: {tuned_logistic_tfidf_accuracy:.2%} | Macro-F1: {tuned_logistic_tfidf_f1:.4f}"
    )
    print(
        classification_report(
            bundle.y_test, tuned_logistic_tfidf_predictions, zero_division=0
        )
    )
    print(tuned_logistic_tfidf_cm)

    tuned_logistic_bow_model = LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=1000
    )
    start_time = perf_counter()
    tuned_logistic_bow_model.fit(bundle.X_train_bow, bundle.y_train)
    tuned_logistic_bow_training_time = perf_counter() - start_time
    print(
        f"Logistic Regression (BoW) training time: {tuned_logistic_bow_training_time:.4f} seconds"
    )
    tuned_logistic_bow_predictions = tuned_logistic_bow_model.predict(bundle.X_test_bow)
    tuned_logistic_bow_accuracy = accuracy_score(
        bundle.y_test, tuned_logistic_bow_predictions
    )
    tuned_logistic_bow_cm = confusion_matrix(
        bundle.y_test,
        tuned_logistic_bow_predictions,
        labels=["Negative", "Neutral", "Positive"],
    )
    tuned_logistic_bow_f1 = f1_score(
        bundle.y_test, tuned_logistic_bow_predictions, average="macro", zero_division=0
    )
    print(
        f"Tuned Logistic Regression (BoW) Accuracy: {tuned_logistic_bow_accuracy:.2%} | Macro-F1: {tuned_logistic_bow_f1:.4f}"
    )
    print(
        classification_report(
            bundle.y_test, tuned_logistic_bow_predictions, zero_division=0
        )
    )
    print(tuned_logistic_bow_cm)

    tuned_nb_tfidf_model = MultinomialNB(alpha=0.1)
    start_time = perf_counter()
    tuned_nb_tfidf_model.fit(bundle.X_train_tfidf, bundle.y_train)
    tuned_nb_tfidf_training_time = perf_counter() - start_time
    print(
        f"Tuned Naive Bayes (TF-IDF) training time: {tuned_nb_tfidf_training_time:.4f} seconds"
    )
    tuned_nb_tfidf_predictions = tuned_nb_tfidf_model.predict(bundle.X_test_tfidf)
    tuned_nb_tfidf_accuracy = accuracy_score(bundle.y_test, tuned_nb_tfidf_predictions)
    tuned_nb_tfidf_cm = confusion_matrix(
        bundle.y_test,
        tuned_nb_tfidf_predictions,
        labels=["Negative", "Neutral", "Positive"],
    )
    tuned_nb_tfidf_f1 = f1_score(
        bundle.y_test,
        tuned_nb_tfidf_predictions,
        average="macro",
        zero_division=0,
    )
    print(
        f"Tuned Naive Bayes (TF-IDF) Accuracy: {tuned_nb_tfidf_accuracy:.2%} | Macro-F1: {tuned_nb_tfidf_f1:.4f}"
    )
    print(
        classification_report(
            bundle.y_test, tuned_nb_tfidf_predictions, zero_division=0
        )
    )
    print(tuned_nb_tfidf_cm)

    tuned_nb_bow_model = MultinomialNB(alpha=1.0)
    start_time = perf_counter()
    tuned_nb_bow_model.fit(bundle.X_train_bow, bundle.y_train)
    tuned_nb_bow_training_time = perf_counter() - start_time
    print(f"Naive Bayes (BoW) training time: {tuned_nb_bow_training_time:.4f} seconds")


if __name__ == "__main__":
    main()
