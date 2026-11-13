"""Small, synthetic checks for the M2 pipeline; no official data is rewritten."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr

import data_pipeline as pipeline


class PreprocessingTests(unittest.TestCase):
    def test_html_whitespace_case_and_negation(self):
        self.assertEqual(
            pipeline.preprocess_text("  NOT&nbsp;<b>Good</b>\n NEVER &amp; again "),
            "not good never & again",
        )

    def test_invalid_text_is_rejected_without_dropping_rows(self):
        for text in (None, float("nan"), 17, "", " \t\n", "<br><p></p>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                pipeline.preprocess_text(text)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(pipeline.LABELS, 20)
        self.groups = np.array([f"review-{i // 2}" for i in range(60)], dtype=object)

    def test_exact_deterministic_stratified_partition(self):
        train, test = pipeline.make_split(self.labels)
        repeated_train, repeated_test = pipeline.make_split(self.labels)
        np.testing.assert_array_equal(train, repeated_train)
        np.testing.assert_array_equal(test, repeated_test)
        self.assertEqual((len(train), len(test)), (48, 12))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(range(60)))
        for label in pipeline.LABELS:
            self.assertEqual(np.sum(self.labels[train] == label), 16)
            self.assertEqual(np.sum(self.labels[test] == label), 4)

    @staticmethod
    def snapshot(directory):
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in Path(directory).iterdir()}

    def test_saved_split_reuses_exact_indices_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            first = pipeline.shared_split(self.labels, "sample-a", directory, groups=self.groups)
            before = self.snapshot(directory)
            second = pipeline.shared_split(self.labels, "sample-a", directory, groups=self.groups)
            self.assertEqual(self.snapshot(directory), before)
            for expected, actual in zip(first, second):
                np.testing.assert_array_equal(actual, expected)

    def test_source_configuration_and_version_mismatches_preserve_state(self):
        for field in ("sample_sha256", "random_state", "versions", "strategy",
                      "n_splits", "fold_index", "label_mapping", "grouping_sha256", "preprocessing"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                pipeline.shared_split(self.labels, "sample-a", directory, groups=self.groups)
                path = Path(directory) / "group_split_manifest.json"
                metadata = json.loads(path.read_text(encoding="utf-8"))
                metadata[field] = "changed"
                path.write_text(json.dumps(metadata), encoding="utf-8")
                before = self.snapshot(directory)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    pipeline.shared_split(self.labels, "sample-a", directory, groups=self.groups)
                self.assertEqual(self.snapshot(directory), before)

    def test_different_source_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline.shared_split(self.labels, "sample-a", directory, groups=self.groups)
            before = self.snapshot(directory)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                pipeline.shared_split(self.labels, "sample-b", directory, groups=self.groups)
            self.assertEqual(self.snapshot(directory), before)

    def test_corrupt_indices_are_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            train, test = pipeline.shared_split(self.labels, "sample-a", directory, groups=self.groups)
            np.savez_compressed(Path(directory) / "group_split_indices.npz",
                                train_indices=train[::-1], test_indices=test)
            before = self.snapshot(directory)
            with self.assertRaisesRegex(ValueError, "indices differ"):
                pipeline.shared_split(self.labels, "sample-a", directory, groups=self.groups)
            self.assertEqual(self.snapshot(directory), before)

    def test_partial_state_is_rejected_without_creation(self):
        for filename in ("group_split_indices.npz", "group_split_manifest.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / filename).write_bytes(b"incomplete")
                before = self.snapshot(directory)
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    pipeline.shared_split(self.labels, "sample-a", directory, groups=self.groups)
                self.assertEqual(self.snapshot(directory), before)


    def test_group_partition_is_deterministic_complete_and_text_isolated(self):
        raw = ["<b>Shared</b>", " shared "] + [f"review {i // 2}" for i in range(58)]
        groups = np.array([pipeline.preprocess_text(t) for t in raw], dtype=object)
        labels = np.tile(pipeline.LABELS, 20)  # repeated groups can contain conflicting labels
        train, test = pipeline.make_group_split(labels, groups)
        expected = next(pipeline.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=47).split(
            np.zeros(len(labels)), labels, groups=groups))
        for actual, first_fold in zip((train, test), expected):
            np.testing.assert_array_equal(actual, first_fold)
        again = pipeline.make_group_split(labels, groups)
        for actual, expected in zip((train, test), again):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(sorted(np.r_[train, test].tolist()), list(range(60)))
        self.assertFalse(set(groups[train]) & set(groups[test]))
        self.assertEqual(0 in train, 1 in train)

    def test_old_and_new_caches_coexist_without_changing_legacy_files(self):
        with tempfile.TemporaryDirectory() as directory:
            old_indices = Path(directory) / "split_indices.npz"
            old_meta = Path(directory) / "split_manifest.json"
            old_indices.write_bytes(b"historical indices")
            old_meta.write_text('{"strategy":"stratified"}', encoding="utf-8")
            old = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (old_indices, old_meta)}
            pipeline.shared_split(self.labels, "sample-a", directory, groups=self.groups)
            self.assertEqual(old, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (old_indices, old_meta)})
            self.assertTrue((Path(directory) / "group_split_indices.npz").exists())

    def test_changed_grouping_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline.shared_split(self.labels, "sample-a", directory, groups=self.groups)
            before = self.snapshot(directory)
            changed = self.groups.copy()
            changed[0] = "new-group"
            with self.assertRaisesRegex(ValueError, "mismatch"):
                pipeline.shared_split(self.labels, "sample-a", directory, groups=changed)
            self.assertEqual(before, self.snapshot(directory))


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.train = ["not good tasty food", "never bad tasty food",
                      "bad bitter food", "good sweet food",
                      "plain ordinary food", "plain average food"]
        self.labels = np.array(["Positive", "Positive", "Negative", "Negative",
                                "Neutral", "Neutral"])
        self.config = dict(pipeline.VECTOR_CONFIG, min_df=1)

    def fit(self, test):
        return pipeline.fit_representations(self.train, test, self.config)

    def test_sparse_dimensions_values_and_negation_tokens(self):
        xb, xbt, xt, xtt, bow, tfidf = self.fit(["not good", "unknownword"])
        for matrix in (xb, xbt, xt, xtt):
            self.assertTrue(isspmatrix_csr(matrix))
            self.assertTrue(np.isfinite(matrix.data).all())
            self.assertTrue((matrix.data >= 0).all())
        self.assertEqual(xb.shape, (6, len(bow.get_feature_names_out())))
        self.assertEqual(xbt.shape, (2, xb.shape[1]))
        self.assertEqual(xt.shape, (6, len(tfidf.get_feature_names_out())))
        self.assertEqual(xtt.shape, (2, xt.shape[1]))
        self.assertEqual(xb.dtype, np.dtype("int64"))
        self.assertEqual(xt.dtype, np.dtype("float64"))
        for vectorizer in (bow, tfidf):
            self.assertTrue({"not", "never"}.issubset(vectorizer.vocabulary_))
            self.assertNotIn("unknownword", vectorizer.vocabulary_)
        self.assertEqual(xbt[1].nnz, 0)
        self.assertEqual(xtt[1].nnz, 0)

    def test_test_changes_do_not_change_vocabulary_idf_or_rankings(self):
        first = self.fit(["testonlytoken delicious"])
        changed = self.fit(["differenttesttoken bad bad bad", "tasty"])
        for feature_offset, vectorizer_offset, kind in ((0, 4, "bow"), (2, 5, "tfidf")):
            first_names = first[vectorizer_offset].get_feature_names_out()
            changed_names = changed[vectorizer_offset].get_feature_names_out()
            np.testing.assert_array_equal(first_names, changed_names)
            np.testing.assert_array_equal(first[feature_offset].toarray(), changed[feature_offset].toarray())
            self.assertEqual(
                pipeline.rank_features(first[feature_offset], self.labels, first_names, kind),
                pipeline.rank_features(changed[feature_offset], self.labels, changed_names, kind),
            )
            self.assertNotIn("testonlytoken", first[vectorizer_offset].vocabulary_)
            self.assertNotIn("differenttesttoken", changed[vectorizer_offset].vocabulary_)
        np.testing.assert_array_equal(first[5].idf_, changed[5].idf_)

    def test_bundle_uses_identical_ordered_labels_for_both_representations(self):
        texts = np.array(self.train * 3)
        labels = np.tile(self.labels, 3)
        train, test = pipeline.make_split(labels)
        with patch.object(pipeline, "load_official_sample", return_value=(None, texts, labels, "fingerprint")), \
                patch.object(pipeline, "shared_split", return_value=(train, test)) as shared:
            bundle = pipeline.build_features()
        shared.assert_called_once()
        np.testing.assert_array_equal(shared.call_args.kwargs["groups"], texts)
        self.assertEqual(bundle.config["official_split"]["strategy"], "StratifiedGroupKFold")
        np.testing.assert_array_equal(bundle.y_train, labels[train])
        np.testing.assert_array_equal(bundle.y_test, labels[test])
        np.testing.assert_array_equal(bundle.train_indices, train)
        np.testing.assert_array_equal(bundle.test_indices, test)
        self.assertEqual(bundle.X_train_bow.shape[0], len(bundle.y_train))
        self.assertEqual(bundle.X_train_tfidf.shape[0], len(bundle.y_train))
        self.assertEqual(bundle.X_test_bow.shape[0], len(bundle.y_test))
        self.assertEqual(bundle.X_test_tfidf.shape[0], len(bundle.y_test))


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array(["Positive", "Negative", "Neutral"])
        self.names = np.array(["alpha", "beta"])

    def test_bow_hand_calculated_smoothed_probability_ratio_includes_neutral(self):
        matrix = csr_matrix([[3, 1], [1, 3], [4, 0]])
        result = pipeline.rank_features(matrix, self.labels, self.names, "bow")
        positive = {row["feature"]: row["score"] for row in result["Positive"]}
        negative = {row["feature"]: row["score"] for row in result["Negative"]}
        self.assertAlmostEqual(positive["alpha"], np.log(4 / 6) - np.log(6 / 10))
        self.assertAlmostEqual(positive["beta"], np.log(2 / 6) - np.log(4 / 10))
        self.assertAlmostEqual(negative["alpha"], np.log(2 / 6) - np.log(8 / 10))
        self.assertAlmostEqual(negative["beta"], np.log(4 / 6) - np.log(2 / 10))

    def test_tfidf_hand_calculated_mean_difference_includes_neutral(self):
        matrix = csr_matrix([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4]])
        result = pipeline.rank_features(matrix, self.labels, self.names, "tfidf")
        positive = {row["feature"]: row["score"] for row in result["Positive"]}
        negative = {row["feature"]: row["score"] for row in result["Negative"]}
        self.assertAlmostEqual(positive["alpha"], 0.8 - (0.2 + 0.6) / 2)
        self.assertAlmostEqual(positive["beta"], 0.2 - (0.8 + 0.4) / 2)
        self.assertAlmostEqual(negative["alpha"], 0.2 - (0.8 + 0.6) / 2)
        self.assertAlmostEqual(negative["beta"], 0.8 - (0.2 + 0.4) / 2)

    def test_equal_scores_sort_by_feature_name(self):
        for kind in ("bow", "tfidf"):
            with self.subTest(kind=kind):
                rankings = pipeline.rank_features(csr_matrix(np.ones((3, 3))), self.labels,
                                                   np.array(["zulu", "alpha", "beta"]), kind, 2)
                for target in ("Positive", "Negative"):
                    self.assertEqual([row["feature"] for row in rankings[target]], ["alpha", "beta"])

    def test_invalid_representation_and_missing_target_fail_clearly(self):
        matrix = csr_matrix(np.ones((3, 2)))
        with self.assertRaises(ValueError):
            pipeline.rank_features(matrix, self.labels, self.names, "coefficients")
        with self.assertRaises(ValueError):
            pipeline.rank_features(matrix, np.array(["Positive"] * 3), self.names, "bow")


if __name__ == "__main__":
    unittest.main()
