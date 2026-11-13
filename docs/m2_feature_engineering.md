# M2 Feature Engineering

[Project homepage](../README.md) · [M1 data engineering](data_engineering.md) · [Computed results](../outputs/m2/feature_report.json)

## Input and preprocessing

The fixed official sample contains 50,000 rows generated with seed 47. The M2 loader verifies its SHA-256 against the local sample manifest, its original ten columns, ratings 1–5, and row count. It does not modify the CSV or rerun sampling. Only `Text` is predictive input; `Score` supplies the provisional labels 1–2 → Negative, 3 → Neutral, and 4–5 → Positive. `Id`, product/user fields, and other metadata are excluded. `Summary` could be investigated later, but combining it with Text requires a shared group decision and consistent retraining.

Stateless preprocessing decodes HTML entities, replaces HTML tags with spaces, lowercases, and normalizes whitespace. Missing/non-string text and text empty after processing raise errors rather than being silently dropped. No stemming, lemmatization, or stopword removal is applied. Words such as `not` and `never` remain available. The default token pattern `(?u)\b\w\w+\b` keeps tokens of at least two word characters and splits apostrophe-containing contractions; therefore, this simple pipeline does not preserve every form of negation semantics.

## Shared split and duplicate leakage

The default split applies `train_test_split` to original sample row positions with `test_size=0.2`, `random_state=47`, and stratification on sentiment. It produces exactly 40,000 training and 10,000 held-out test rows. Both representations use the same ordered indices and labels. Integer class rounding gives:

| Sentiment | Training count | Training % | Test count | Test % |
| --- | ---: | ---: | ---: | ---: |
| Negative | 5,912 | 14.7800 | 1,478 | 14.7800 |
| Neutral | 2,967 | 7.4175 | 742 | 7.4200 |
| Positive | 31,121 | 77.8025 | 7,780 | 77.8000 |

All percentages within each split use that split's size. Training and test row positions are disjoint and cover the full sample. Local `outputs/m2/local/split_indices.npz` and `split_manifest.json` preserve the ordered split, sample fingerprint, configuration, and versions. Existing state is validated against a fresh deterministic computation; incomplete, altered, or incompatible state is rejected without silent overwriting. These local files are ignored by Git.

The sample has 46,326 distinct original texts, 2,978 repeated-text groups, and 3,674 extra text occurrences. Preprocessing leaves 46,315 distinct texts. One original-text group and one preprocessed-text group contain conflicting sentiment labels; neither labels nor rows were changed.

| Default-split comparison | Distinct texts shared across splits | Affected training rows | Affected test rows |
| --- | ---: | ---: | ---: |
| Original Text | 1,077 | 1,382 | 1,153 |
| Preprocessed Text | 1,082 | 1,390 | 1,158 |

Thus, disjoint row indices do not eliminate duplicate-text leakage. These counts describe text intersections, not model performance or an estimated amount of accuracy inflation.

### Group-aware comparison, not the shared default

We evaluated the fixed first fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=47)`, grouping by preprocessed Text. No fold was selected using test scores or a preferred-looking distribution. The comparison has 40,000 training and 10,000 test rows, with zero original- or preprocessed-text overlap and zero affected rows in either split.

| Sentiment | Training count | Training % | Test count | Test % |
| --- | ---: | ---: | ---: | ---: |
| Negative | 5,912 | 14.7800 | 1,478 | 14.7800 |
| Neutral | 2,968 | 7.4200 | 741 | 7.4100 |
| Positive | 31,120 | 77.8000 | 7,781 | 77.8100 |

The exact 80/20 count in this run is an observed result, not a guarantee of StratifiedGroupKFold. Group isolation constrains which rows can move together, so exact row counts and exact class stratification cannot generally both be guaranteed. The default retains exact 80/20 row counts and near-proportional sentiment counts but allows duplicate-text leakage. We recommend group-aware isolation for a future leakage-isolated evaluation, subject to group confirmation. Until then, every model must use the existing standard stratified baseline; it must not mix the two strategies when comparing results.

## Sparse feature representations

Both vectorizers are fitted on training text only; held-out text is transformed without fitting. Shared settings are `ngram_range=(1, 2)`, `min_df=2` training documents, `max_df=1.0`, `max_features=20000`, `stop_words=None`, and `lowercase=False` because the common preprocessor already lowercases. The default token pattern is retained. No metadata is concatenated into text.

| Representation | Actual vocabulary size | Training shape | Test shape | Values |
| --- | ---: | --- | --- | --- |
| BoW, CountVectorizer | 20,000 | 40,000 × 20,000 | 10,000 × 20,000 | int64 raw counts |
| TF-IDF, TfidfVectorizer | 20,000 | 40,000 × 20,000 | 10,000 × 20,000 | float64 weights |

The measured vocabularies reach the configured cap. TF-IDF uses `sublinear_tf=True` (1 + log count for nonzero terms), `use_idf=True`, `smooth_idf=True`, and `norm="l2"`. Unlike raw BoW counts, it adjusts within-document counts by training-set inverse document frequency and normalizes each row. The interface explicitly returns SciPy CSR matrices, compatible with Multinomial Naive Bayes and Logistic Regression. Full feature names are available through each fitted vectorizer and the returned bundle. Document-by-vocabulary matrices are never converted wholesale to dense arrays.

## Training-only feature associations

These are descriptive, class-discriminative rankings, not learned classifier coefficients or model-derived feature importance. For each target, Positive or Negative, rest contains both other classes, including Neutral. All calculations use only the default training rows and training vocabulary. Scores are sorted descending, with feature names ascending for exact ties.

For BoW, let n(c,j) be the summed count of feature j in class c, and V the training vocabulary size. Add-one smoothing gives p(c,j) = (n(c,j) + 1) / (sum_j n(c,j) + V). The score is log p(c,j) − log p(rest,j). This is a smoothed token log probability ratio, not strict log-odds.

For TF-IDF, the score is the mean weight of a feature among target-class training reviews minus its mean among rest training reviews, including zero-valued documents. Only vocabulary-sized aggregate vectors become dense. The two score scales are different and should not be compared numerically across representations.

### BoW: Positive versus rest

| Rank | Feature | Score |
| ---: | --- | ---: |
| 1 | `great with` | 3.46703796 |
| 2 | `definitely order` | 3.40835446 |
| 3 | `new favorite` | 3.38613132 |
| 4 | `worth every` | 3.32831175 |
| 5 | `to everyone` | 3.20156005 |
| 6 | `every penny` | 3.10218757 |
| 7 | `breakfast and` | 3.02462934 |
| 8 | `before bed` | 3.00836882 |
| 9 | `added bonus` | 2.99183951 |
| 10 | `say enough` | 2.99183951 |

### BoW: Negative versus rest

| Rank | Feature | Score |
| ---: | --- | ---: |
| 1 | `deliberately` | 5.18711922 |
| 2 | `knocking things` | 5.15813168 |
| 3 | `eats some` | 5.12827872 |
| 4 | `what waste` | 4.74528647 |
| 5 | `threw the` | 4.40435988 |
| 6 | `thing then` | 4.02966643 |
| 7 | `thought she` | 3.93435625 |
| 8 | `buyer beware` | 3.71121270 |
| 9 | `don waste` | 3.71121270 |
| 10 | `never order` | 3.54869377 |

### TF-IDF: Positive versus rest

| Rank | Feature | Score |
| ---: | --- | ---: |
| 1 | `great` | 0.01183898 |
| 2 | `love` | 0.00809192 |
| 3 | `best` | 0.00703767 |
| 4 | `the best` | 0.00676055 |
| 5 | `delicious` | 0.00567575 |
| 6 | `and` | 0.00503628 |
| 7 | `loves` | 0.00470476 |
| 8 | `perfect` | 0.00455061 |
| 9 | `favorite` | 0.00436818 |
| 10 | `my` | 0.00423346 |

### TF-IDF: Negative versus rest

| Rank | Feature | Score |
| ---: | --- | ---: |
| 1 | `not` | 0.01426573 |
| 2 | `was` | 0.01184367 |
| 3 | `disappointed` | 0.00612777 |
| 4 | `money` | 0.00607507 |
| 5 | `were` | 0.00606206 |
| 6 | `had` | 0.00598097 |
| 7 | `what` | 0.00586195 |
| 8 | `bad` | 0.00584788 |
| 9 | `would` | 0.00577472 |
| 10 | `it was` | 0.00513031 |

### Figures and observations

- [BoW positive/negative associations, 300 DPI](../outputs/figures/m2_bow_features.png)
- [TF-IDF positive/negative associations, 300 DPI](../outputs/figures/m2_tfidf_features.png)

TF-IDF highlights positive terms such as `great`, `love`, and `delicious`, and negative associations including `not` and `disappointed`. The presence of `not` is consistent with preserving negation words. Function words such as `and`, `my`, and `was` also rank because stopwords were deliberately retained; descriptive association alone does not establish semantic sentiment or predictive utility. BoW's high-scoring phrases, including `knocking things` and `eats some`, are context-specific and can be sensitive to low-frequency features, long reviews, and repeated text. They should not all be interpreted as generally negative vocabulary. No algorithm was adjusted to obtain more intuitive-looking words.

Positive reviews remain the majority: training shares are 77.8025% Positive, 14.7800% Negative, and 7.4175% Neutral. This imbalance and the documented duplicate overlap must accompany future model evaluation. Label mapping, model choices, optional Summary input, and a switch to group-aware splitting remain group decisions.

## Reproduction and ML integration

Run from the repository root using Python 3.11. Until M2 is merged into main, use the published `feature/m2-data-pipeline` branch; after a fresh clone, select it with `git switch feature/m2-data-pipeline`. Existing users with the official sample and local sample manifest do not need to rerun M1 sampling or EDA. A new clone must first follow the existing [source/sample setup](data_engineering.md#complete-reproduction); do not replace a conflicting official sample just to bypass validation.

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

M2 adds pinned scikit-learn 1.9.1 and SciPy 1.16.3. The command writes the shareable [feature report JSON](../outputs/m2/feature_report.json) and two figures, while keeping local split indices and metadata out of Git. It does not write the official CSV or any M1 result. Large matrices and serialized vectorizers are not committed; `build_features()` reconstructs them and reuses the checked local split.

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

The unchanged Group Phi sample contains 50,000 Amazon food reviews. We use Text alone, decode HTML entities, replace tags with spaces, lowercase, and normalize whitespace. We retain stopwords, including not and never; ratings supply provisional Negative (1–2), Neutral (3), and Positive (4–5) labels. Summary and metadata are excluded from predictive inputs.

A deterministic, sentiment-stratified split with random_state=47 yields 40,000 training and 10,000 test reviews. The test set contains 1,478 negative, 742 neutral, and 7,780 positive reviews. The baseline has 1,077 overlapping original texts and 1,082 overlapping preprocessed texts. A fixed first-fold StratifiedGroupKFold comparison isolates preprocessed-text groups with zero overlap. It also yields 80/20 here, although this ratio is not guaranteed; adopting it requires group confirmation.

Training-only CountVectorizer and TfidfVectorizer each produce 20,000 unigram/bigram features and CSR sparse matrices. Both use min_df=2 and no stopword removal; TF-IDF additionally applies sublinear term frequency, smoothed IDF, and L2 normalization. Positive/negative features are ranked against both other classes, including Neutral, using smoothed token log probability ratios for BoW and class-mean weight differences for TF-IDF. These descriptive associations are not model coefficients. Class imbalance, duplicate leakage, and context-specific phrases limit interpretation.
