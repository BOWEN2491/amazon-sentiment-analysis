# M2 Feature Engineering

[Project homepage](../README.md) · [M1 data engineering](data_engineering.md) · [Computed results](../outputs/m2/feature_report.json)

## Input and preprocessing

The fixed official sample contains 50,000 rows generated with seed 47. The M2 loader verifies its SHA-256 against the local sample manifest, its original ten columns, ratings 1–5, and row count. It does not modify the CSV or rerun sampling. Only `Text` is predictive input; `Score` supplies the provisional labels 1–2 → Negative, 3 → Neutral, and 4–5 → Positive. `Id`, product/user fields, and other metadata are excluded. `Summary` could be investigated later, but combining it with Text requires a shared group decision and consistent retraining.

Stateless preprocessing decodes HTML entities, replaces HTML tags with spaces, lowercases, and normalizes whitespace. Missing/non-string text and text empty after processing raise errors rather than being silently dropped. No stemming, lemmatization, or stopword removal is applied. Words such as `not` and `never` remain available. The default token pattern `(?u)\b\w\w+\b` keeps tokens of at least two word characters and splits apostrophe-containing contractions; therefore, this simple pipeline does not preserve every form of negation semantics.

## Official group-aware split and leakage audit

Group Phi has adopted the fixed first fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=47)` as the official split for M2 and subsequent milestones. Groups are defined by preprocessed review Text, preventing identical model-input texts from appearing in both training and testing sets. No fold is selected using model performance. Both representations receive the same ordered train/test indices and labels through `build_features()`.

The recomputed official partition contains 40,000 training and 10,000 test rows:

| Sentiment | Training count | Training % | Test count | Test % |
| --- | ---: | ---: | ---: | ---: |
| Negative | 5,912 | 14.7800 | 1,478 | 14.7800 |
| Neutral | 2,968 | 7.4200 | 741 | 7.4100 |
| Positive | 31,120 | 77.8000 | 7,781 | 77.8100 |

Training and test row positions are disjoint and cover every sample row exactly once. There are zero overlapping original texts and zero overlapping preprocessed texts; affected training and test row counts are also zero. The exact 80/20 count is an observed result for this sample, not a general guarantee of StratifiedGroupKFold. Group isolation constrains class balance and row counts, so exact proportions cannot generally both be guaranteed.

Official local files are `outputs/m2/local/group_split_indices.npz` and `group_split_manifest.json`. Metadata records the sample SHA-256, grouping fingerprint, row count, preprocessing version, label mapping, SGKF parameters and first-fold index, and runtime versions. Existing state is checked against a fresh deterministic computation; incomplete, altered, or incompatible state is rejected without silent overwriting. The old `split_indices.npz` and `split_manifest.json` are preserved unchanged. All local split files remain excluded from Git.

The sample still contains 46,326 distinct original texts, 2,978 repeated-text groups, and 3,674 extra occurrences. Preprocessing leaves 46,315 distinct texts. One original-text group and one preprocessed-text group contain conflicting labels; neither labels nor rows were changed. Such groups stay together in the official partition.

### Historical standard stratified baseline

The former `train_test_split(test_size=0.2, random_state=47, stratify=labels)` split is retained solely as an audit comparison, not as an alternative default for models:

| Sentiment | Training count | Training % | Test count | Test % |
| --- | ---: | ---: | ---: | ---: |
| Negative | 5,912 | 14.7800 | 1,478 | 14.7800 |
| Neutral | 2,967 | 7.4175 | 742 | 7.4200 |
| Positive | 31,121 | 77.8025 | 7,780 | 77.8000 |

| Historical baseline text definition | Distinct texts shared across splits | Affected training rows | Affected test rows |
| --- | ---: | ---: | ---: |
| Original Text | 1,077 | 1,382 | 1,153 |
| Preprocessed Text | 1,082 | 1,390 | 1,158 |

The earlier disjoint row indices did not prevent duplicate-text leakage. The official group-aware partition removes this exact-text intersection; these counts do not measure model performance or accuracy inflation. All future model comparisons must use the official ordered partition rather than mix the historical baseline with the adopted split.

## Sparse feature representations

Both vectorizers have been refitted from scratch on the official group-aware training text only; held-out text is transformed without fitting. Shared settings are `ngram_range=(1, 2)`, `min_df=2` training documents, `max_df=1.0`, `max_features=20000`, `stop_words=None`, and `lowercase=False` because the common preprocessor already lowercases. The default token pattern is retained. No metadata is concatenated into text.

| Representation | Actual vocabulary size | Training shape | Test shape | Values |
| --- | ---: | --- | --- | --- |
| BoW, CountVectorizer | 20,000 | 40,000 × 20,000 | 10,000 × 20,000 | int64 raw counts |
| TF-IDF, TfidfVectorizer | 20,000 | 40,000 × 20,000 | 10,000 × 20,000 | float64 weights |

The measured vocabularies reach the configured cap. TF-IDF uses `sublinear_tf=True` (1 + log count for nonzero terms), `use_idf=True`, `smooth_idf=True`, and `norm="l2"`. Unlike raw BoW counts, it adjusts within-document counts by training-set inverse document frequency and normalizes each row. The interface explicitly returns SciPy CSR matrices, compatible with Multinomial Naive Bayes and Logistic Regression. Full feature names are available through each fitted vectorizer and the returned bundle. Document-by-vocabulary matrices are never converted wholesale to dense arrays.

## Training-only feature associations

These are descriptive, class-discriminative rankings, not learned classifier coefficients or model-derived feature importance. For each target, Positive or Negative, rest contains both other classes, including Neutral. All four rankings were recomputed using only the official group-aware training rows and newly fitted training vocabulary; no historical rankings or feature statistics were reused. Scores are sorted descending, with feature names ascending for exact ties.

For BoW, let n(c,j) be the summed count of feature j in class c, and V the training vocabulary size. Add-one smoothing gives p(c,j) = (n(c,j) + 1) / (sum_j n(c,j) + V). The score is log p(c,j) − log p(rest,j). This is a smoothed token log probability ratio, not strict log-odds.

For TF-IDF, the score is the mean weight of a feature among target-class training reviews minus its mean among rest training reviews, including zero-valued documents. Only vocabulary-sized aggregate vectors become dense. The two score scales are different and should not be compared numerically across representations.

### BoW: Positive versus rest

| Rank | Feature | Score |
| ---: | --- | ---: |
| 1 | `great with` | 4.03187258 |
| 2 | `what more` | 3.43825500 |
| 3 | `definitely order` | 3.42767289 |
| 4 | `just love` | 3.28498113 |
| 5 | `worth every` | 3.27882726 |
| 6 | `to everyone` | 3.22818353 |
| 7 | `just great` | 3.16104423 |
| 8 | `this wonderful` | 3.13287335 |
| 9 | `can beat` | 3.08158006 |
| 10 | `every penny` | 3.02751283 |

### BoW: Negative versus rest

| Rank | Feature | Score |
| ---: | --- | ---: |
| 1 | `threw the` | 4.24855187 |
| 2 | `what waste` | 4.08603294 |
| 3 | `is horrible` | 3.65976471 |
| 4 | `buyer beware` | 3.64420019 |
| 5 | `save your` | 3.63928617 |
| 6 | `wasted my` | 3.60419485 |
| 7 | `waste your` | 3.55924347 |
| 8 | `don waste` | 3.54535435 |
| 9 | `just awful` | 3.45004417 |
| 10 | `very disappointing` | 3.45004417 |

### TF-IDF: Positive versus rest

| Rank | Feature | Score |
| ---: | --- | ---: |
| 1 | `great` | 0.01168601 |
| 2 | `love` | 0.00809811 |
| 3 | `best` | 0.00708764 |
| 4 | `the best` | 0.00673903 |
| 5 | `delicious` | 0.00570868 |
| 6 | `and` | 0.00516885 |
| 7 | `loves` | 0.00470497 |
| 8 | `perfect` | 0.00447909 |
| 9 | `my` | 0.00444684 |
| 10 | `favorite` | 0.00428081 |

### TF-IDF: Negative versus rest

| Rank | Feature | Score |
| ---: | --- | ---: |
| 1 | `not` | 0.01399695 |
| 2 | `was` | 0.01220038 |
| 3 | `were` | 0.00635540 |
| 4 | `had` | 0.00616055 |
| 5 | `disappointed` | 0.00612201 |
| 6 | `would` | 0.00600903 |
| 7 | `bad` | 0.00597517 |
| 8 | `money` | 0.00560197 |
| 9 | `what` | 0.00547137 |
| 10 | `product` | 0.00498999 |

### Figures and observations

- [BoW positive/negative associations, 300 DPI](../outputs/figures/m2_bow_features.png)
- [TF-IDF positive/negative associations, 300 DPI](../outputs/figures/m2_tfidf_features.png)

TF-IDF highlights positive terms such as `great`, `love`, and `delicious`, and negative associations including `not` and `disappointed`. The presence of `not` is consistent with preserving negation words. Function words such as `and`, `my`, and `was` also rank because stopwords were deliberately retained; descriptive association alone does not establish semantic sentiment or predictive utility. The recomputed BoW list includes positive associations such as `great with` and `just love`, and negative associations such as `threw the`, `is horrible`, and `buyer beware`. These remain context-specific and can be sensitive to low-frequency features, long reviews, and repetition within the training set; they are not universal sentiment indicators. No algorithm was adjusted to obtain more intuitive-looking words.

Positive reviews remain the majority: official training shares are 77.8000% Positive, 14.7800% Negative, and 7.4200% Neutral. Class imbalance remains despite zero exact-text overlap. Label mapping, model choices, and optional Summary input remain group decisions; group-aware splitting has already been adopted.

## Reproduction and ML integration

Run from the repository root using Python 3.11. The M2 pipeline is merged into `main`; use the default branch after cloning, without a feature-branch checkout. `build_features()` uses the official group-aware split by default. Existing users with the official sample and local sample manifest do not need to rerun M1 sampling or EDA. A new clone must first follow the existing [source/sample setup](data_engineering.md#complete-reproduction); do not replace a conflicting official sample just to bypass validation.

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m data_pipeline
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS/Linux:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m data_pipeline
.venv/bin/python -m unittest discover -s tests -v
```

M2 adds pinned scikit-learn 1.9.1 and SciPy 1.16.3. The command writes the shareable [feature report JSON](../outputs/m2/feature_report.json) and two figures, while keeping local split indices and metadata out of Git. It does not write the official CSV or any M1 result. Large matrices and serialized vectorizers are not committed; `build_features()` reconstructs them and reuses the checked official group-aware split. Rebuild all feature inputs after this switch. Do not reuse old vectorizers, rankings, or models fitted on the historical partition; any previously trained models must be retrained later by the ML Engineer on the official inputs. No classifier is trained by this pipeline.

```python
from data_pipeline import build_features

features = build_features()
# Shared CSR inputs: X_train_bow, X_test_bow,
# X_train_tfidf, X_test_tfidf; y_train and y_test are sentiment strings.
print(features.X_train_bow.shape, features.X_test_tfidf.shape)
print(features.feature_names_bow, features.feature_names_tfidf)
# Also available: vectorizer_bow, vectorizer_tfidf,
# train_indices, test_indices, config, and sample_fingerprint.
```

The following is an integration example for Jacob to run during the ML milestone; it was not executed for this data-engineering deliverable:

```python
from sklearn.naive_bayes import MultinomialNB
from data_pipeline import build_features

features = build_features()
model = MultinomialNB()
model.fit(features.X_train_bow, features.y_train)
predictions = model.predict(features.X_test_bow)
```

The same bundle provides TF-IDF inputs for Logistic Regression without changing row order or labels. Classifier training, evaluation metrics, and error analysis remain the ML Engineer's work. No classifier was trained in this session.

## Copy-ready Feature Engineering section

The unchanged Group Phi sample contains 50,000 Amazon food reviews. Text alone is used as input; HTML entities are decoded, tags replaced with spaces, and case and whitespace normalized. Stopwords, including not and never, are retained. Ratings provide provisional Negative (1–2), Neutral (3), and Positive (4–5) labels; Summary and metadata are excluded.

Group Phi adopted the first fold of StratifiedGroupKFold with five splits, shuffle=True, and random_state=47, grouping by preprocessed Text. The official partition contains 40,000 training and 10,000 test reviews; test labels comprise 1,478 negative, 741 neutral, and 7,781 positive reviews. Both original and preprocessed texts have zero cross-split overlap. The exact 80/20 ratio is observed here, not guaranteed by the algorithm. The earlier stratified split remains an audit baseline with 1,077 original-text and 1,082 preprocessed-text intersections.

BoW and TF-IDF were refitted only on official training rows, each yielding 20,000 unigram/bigram features in CSR matrices. TF-IDF uses sublinear term frequency, smoothed training IDF, and L2 normalization. Recomputed positive/negative rankings compare each class against both other classes, including Neutral, using smoothed token log probability ratios for BoW and class-mean TF-IDF differences. These are descriptive associations, not model coefficients. Class imbalance and context-specific phrases remain limitations.
