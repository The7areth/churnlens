# Understand ChurnLens from end to end

This walkthrough explains the implemented system, the reasoning behind it, and the boundaries of its evidence. Links point to the code you should be able to navigate during a review.

## 1. The project in one minute

ChurnLens is a supervised binary-classification application built around IBM's fictional telecom customer dataset. Each historical row contains customer/account information and a label indicating churn. The training pipeline learns relationships between the available information and that label, then assigns a new, similarly structured record a churn score.

Three models are compared: Logistic Regression, Random Forest, and XGBoost. Logistic Regression achieved the highest validation ROC-AUC and was selected. Its preprocessing, classifier, threshold, and provenance are saved together. FastAPI loads that artifact and exposes predictions; Streamlit supplies a customer form and a results dashboard.

The strongest project story is the complete engineering workflow: reproducible data handling, careful evaluation, a consistent training/serving pipeline, validation, tests, and a working interface. The current results establish a benchmark on a sample dataset. Real retention outcomes would require additional validation and an intervention experiment.

## 2. The problem and intended decision

A retention team has limited time and cannot contact every customer. A model could help prioritize accounts for review. The output is a score and a High/Low classification, not an automatic instruction to discount, contact, or deny service.

There are two important mistakes:

- **False positive:** flag a customer who is actually retained. Outreach may be unnecessary and consume budget.
- **False negative:** miss a customer who actually churns. The team may lose an opportunity to investigate or help.

A useful threshold depends on the relative costs of these mistakes and the team's capacity. This demo uses validation F1 to make the choice explicit. It has not measured the costs or the effect of outreach.

## 3. What is in the dataset?

The full CSV contains **7,043 rows and 21 columns**: 19 predictors, customerID, and Churn. Approximately **26.5%** of rows have Churn=Yes.

| Group | Examples | Meaning |
|---|---|---|
| Customer characteristics | gender, SeniorCitizen, Partner, Dependents | Demographic and household categories |
| Account history | tenure | Months with the company |
| Services | PhoneService, InternetService, TechSupport, StreamingTV | Subscriptions and service options |
| Billing relationship | Contract, PaperlessBilling, PaymentMethod | Account and payment arrangements |
| Charges | MonthlyCharges, TotalCharges | Current monthly charge and accumulated charges |
| Target | Churn | Yes becomes 1; No becomes 0 |
| Identifier | customerID | Used for validation and split auditing, excluded from model inputs |

The downloader checks an expected SHA-256 hash, which identifies the exact source bytes. This supports reproducibility and detects a changed or partial download; it does not prove that the data is representative or unbiased. The source license and provenance are retained.

The included **180-row sample** is synthetic test data. It enables smoke tests without internet and has deliberately simplified labels. The API and reports identify non-IBM data so sample scores cannot be mistaken for benchmark results.

Read: [data.py](../src/churn/data.py), [config.py](../src/churn/config.py), [data provenance](../data/README.md).

## 4. Validation and cleaning

The data loader checks required columns, unique/nonmissing customer IDs, valid labels, valid customer fields, and sufficient examples of both classes. Numeric parsing fails on unexpected nonnumeric values. Blank TotalCharges becomes a missing value.

There are **11 blank TotalCharges entries** in the full dataset; all have zero tenure. Seven are in the training partition. The current pipeline uses a training-median imputer. That is a simple general-purpose choice, with a domain limitation: an established customer's median accumulated charge may be a poor representation of a new customer. A future development-only experiment could compare zero-tenure-specific treatment and a missingness indicator. Do not claim that comparison has already been performed.

Service rules prevent contradictory records. If PhoneService is No, MultipleLines must be No phone service. If InternetService is No, internet add-ons must be No internet service. The UI applies those conventions automatically; the API rejects contradictory direct requests.

Read: [schema.py](../src/churn/schema.py).

## 5. Why three data partitions?

| Partition | Rows | Its job |
|---|---:|---|
| Training | 4,225 | Fit preprocessing and model parameters |
| Validation | 1,409 | Choose the model and its decision threshold |
| Test | 1,409 | Report final performance after those choices |

The split is stratified, which approximately preserves the churn proportion in each partition. Seed 42 makes the split reproducible; 42 has no special statistical meaning. Customer IDs are unique, the partitions are disjoint, and the saved split IDs make them auditable.

**Data leakage** occurs when information unavailable at the intended training/prediction stage influences the model. Fitting an imputer or scaler using test rows would leak information about the test distribution. Choosing a model or threshold because it looks best on the test set would also bias the evaluation.

The project prevents those paths by fitting transformations on training data and making selection decisions on validation data. The validation set is reused for model and threshold selection, so its selected scores can be optimistic. The test set provides the final check. There is no claim of cross-validation or nested cross-validation in this release.

The test results have now been inspected. Further optimization should use a defined development protocol and eventually fresh final evaluation data, rather than repeatedly choosing changes from these same test scores.

Read: `split_data`, `train`, and `choose_threshold` in [train.py](../src/churn/train.py).

## 6. What preprocessing actually does

The scikit-learn **Pipeline** bundles the fitted transformations and classifier. A **ColumnTransformer** sends numeric and categorical columns through different processing steps.

For tenure, MonthlyCharges, and TotalCharges:

1. Replace missing values with the median learned from training rows.
2. Standardize each column using its training mean and standard deviation: `(value - mean) / standard deviation`.

Scaling matters for the regularized Logistic Regression model because the numeric variables have different units and ranges. Tree models generally do not need scaling; the shared pipeline keeps the comparison and serving interface simple.

For categorical fields:

1. Apply most-frequent imputation as a defensive pipeline step. The current data/API schema otherwise rejects missing required categories.
2. Apply one-hot encoding: turn a category into indicator columns. For example, Contract becomes separate indicators for Month-to-month, One year, and Two year.

One-hot encoding avoids inventing a numeric ordering between categories. SeniorCitizen is treated as categorical despite being stored as 0/1. `handle_unknown="ignore"` makes the pipeline robust to unseen values, but the public API deliberately accepts only declared categories. Those are different layers: general pipeline robustness and a strict application contract.

All medians, scaling statistics, and category mappings are fitted on training data. Predictions call `transform`, not `fit`.

Read: [pipeline.py](../src/churn/pipeline.py), [predict.py](../src/churn/predict.py).

## 7. The three models

### Logistic Regression

Despite its name, this is a classification model. It combines transformed inputs into a linear score `z = w·x + b`, then maps that score into a number between zero and one with `p = 1 / (1 + exp(-z))`.

It is a useful baseline for tabular data: fast, compact, and comparatively interpretable. This implementation uses scikit-learn's default L2 regularization, `C=1.0`, and a maximum of 2,000 optimization iterations. C controls inverse regularization strength; max_iter is a convergence limit, not the number of trees or data passes. The measured run converged in 38 iterations.

### Random Forest

A forest averages predictions from many decision trees. Trees can capture nonlinear rules and feature interactions; averaging reduces the instability of an individual tree.

This implementation uses 300 trees, maximum depth 10, and at least 5 samples per leaf. The depth and leaf limits constrain complexity. Bootstrap sampling and random feature selection are handled by the estimator's defaults. The model is not trained with custom class weights in this release.

### XGBoost

Gradient boosting builds trees sequentially. New trees improve the current model using gradients of the loss function. This implementation uses 250 trees, depth 3, learning rate 0.04, row subsampling 0.85, column subsampling 0.85, and histogram-based training.

The small learning rate controls each new tree's contribution. Subsampling and shallow trees limit complexity. There is no early stopping or exhaustive hyperparameter search in this release. XGBoost requires a compatible native runtime; an unavailable import is recorded as a two-model fallback.

### Why Logistic Regression won

The selection rule was highest **validation ROC-AUC**. Scores were approximately 0.860 for Logistic Regression and 0.857 for each tree model. The algorithm label did not determine the winner.

These differences are small and come from one split. The conclusion is that Logistic Regression won this fixed comparison—not that it is universally better or statistically proven superior. XGBoost's slightly higher test ROC-AUC at more decimal places does not justify changing the selection after inspecting test results.

Read: the estimator definitions in [train.py](../src/churn/train.py).

## 8. From score to risk band

The model produces a probability estimate. The application turns it into a class using a threshold:

```text
probability >= threshold → predicted churn / High risk
probability < threshold  → predicted retained / Low risk
```

The threshold search evaluates 181 values from 0.05 through 0.95 in steps of 0.005. It maximizes validation F1, choosing the first maximum if there is a tie. The selected threshold is approximately **0.495**. It is selected for this exact fitted model, then stored with it.

The comparison table uses 0.5 for every model, allowing a common operating-point comparison. The final selected-model summary and API use 0.495. ROC-AUC does not change with this classification threshold because it is computed from scores across thresholds.

The selected model remains fitted on the training partition. Retraining on training plus validation could change its scores, which would invalidate the assumption that the old threshold was tuned for that exact model. A future refit needs a revised calibration/threshold-selection procedure.

## 9. Read the metrics using real numbers

On the 1,409-row test set, the served model produces this confusion matrix:

| Actual / predicted | Predicted retained | Predicted churn |
|---|---:|---:|
| Actually retained | **910 true negatives** | **125 false positives** |
| Actually churned | **175 false negatives** | **199 true positives** |

**Precision = 199 / (199 + 125) = 61.4%.** Of the 324 flagged customers, 199 carry the churn label. Precision describes the concentration of actual churners among flagged cases.

**Recall = 199 / (199 + 175) = 53.2%.** Of the 374 churners, the model finds 199. It misses 175; say that openly.

**F1 = 2 × precision × recall / (precision + recall) = 0.570.** This harmonic mean summarizes both precision and recall at the chosen threshold. It does not encode real outreach costs.

**ROC-AUC = 0.832.** It measures ranking: roughly, a randomly chosen churner receives a higher score than a randomly chosen retained customer 83.2% of the time, with ties given half credit. It is not 83.2% accuracy and does not describe a specific threshold.

**Average precision = 0.622.** This summarizes the precision-recall relationship over score thresholds. It is useful when the positive class is less common. The prior-score baseline is approximately 0.265, reflecting class prevalence.

**Brier score = 0.141.** This is mean squared error between the probability estimates and 0/1 outcomes; lower is better. The prior-probability baseline is about 0.195. A lower Brier score alone does not prove calibration.

**Accuracy = (910 + 199) / 1,409 ≈ 78.7%.** Predicting everyone as retained already gives approximately 73.5% accuracy and zero churn recall. This is why accuracy alone would give an incomplete picture.

The results are useful as an honest baseline with a working service. They do not justify “production-ready retention prediction” or a claim that revenue was saved.

Read: [metrics.json](../reports/metrics.json), [comparison](../reports/comparison.md).

## 10. What the EDA and feature importance mean

EDA is descriptive exploration of the training partition. The plots show class imbalance, tenure distributions, and churn rates by contract. In training data, churn rates are approximately 43.1% for month-to-month, 11.0% for one-year, and 2.9% for two-year contracts.

The selected model's **permutation importance** is measured on validation data. Each raw input column is shuffled, predictions are recomputed, and the decrease in ROC-AUC is recorded. Repeating this five times gives a mean and standard deviation. Tenure, Contract, and InternetService lead the current ranking.

This shows which inputs the fitted model relies on for ranking. It does not show that a variable causes churn, explain an individual customer's prediction, or quantify the effect of changing a subscription. Correlated variables can share or obscure importance. The UI's what-if changes are illustrations of model response, not intervention estimates.

Read: [eda.py](../src/churn/eda.py), [feature_importance.csv](../reports/feature_importance.csv).

## 11. What happens when someone clicks the button?

1. Streamlit collects the customer inputs. It normalizes add-on values when phone or internet service is absent.
2. The UI sends JSON to FastAPI's `POST /predict` endpoint.
3. Pydantic validates categories, numeric bounds, required fields, extra fields, and service consistency. Invalid input gets HTTP 422.
4. The API uses the artifact already loaded at startup. If a usable artifact was not loaded, it returns HTTP 503 rather than inventing a score.
5. Shared inference code puts fields in the expected feature order, represents an unknown TotalCharges as missing, and calls the saved pipeline.
6. The pipeline applies the saved transformations and classifier. The score is compared with the saved threshold.
7. The API returns the score, class/risk, threshold, selected-model name, and dataset label. Streamlit displays these and offers a JSON download.

`GET /health` is a readiness check: it returns success only when the model is available. FastAPI also generates interactive OpenAPI documentation at `/docs`.

The joblib artifact contains the complete pipeline, threshold, feature list, model name, dataset name, and metadata. This avoids rebuilding preprocessing separately in the API. Only trusted artifacts should be loaded because joblib uses pickle-based deserialization.

Read: [api.py](../src/churn/api.py), [predict.py](../src/churn/predict.py), [Streamlit app](../app/streamlit_app.py).

## 12. Testing, Docker, and reproducibility

Tests cover data validity, duplicate IDs, disjoint reproducible splits, inference that does not refit preprocessing, persistence/API parity, invalid requests, missing-model handling, download retries/checksums, complete synthetic training, and UI-to-API service normalization.

These checks establish important functional properties; they do not prove future predictive performance, security, fairness, or calibration. The initial local suite had 17 passing tests. The live CI workflow is the current authority for the published commit's checks.

Docker packages Python, dependencies, source, and the model/report files present at build time. Compose runs separate API and UI services. The UI reaches `http://api:8000` on the container network; the host reaches localhost ports. Readiness checks prevent starting the dependent UI before the API model is ready. Rebuilding is required after retraining because the artifact is copied into the image.

Reproducibility comes from fixed input bytes, saved splits, a fixed seed, explicit model parameters, dependency records, and a persisted complete pipeline. It is not a guarantee of bit-for-bit identical results across every operating system and numerical backend.

## 13. The most valuable next improvements

Start with a concrete decision and better evidence: a defined future churn window, representative timestamped data, temporal validation, stable development-set evaluation, calibration assessment, and a cost/capacity-based operating threshold. Then evaluate demographic subgroup performance and whether those fields should be used at all.

If the product is deployed, add authentication, rate limits, artifact integrity/version controls, request monitoring, and drift checks. Measure retention impact through a controlled outreach experiment. Each of these is a future extension, not a completed feature.

See [the roadmap](roadmap.md) and [model card](model-card.md).
