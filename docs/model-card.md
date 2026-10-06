# Model card

## Intended use

An end-to-end binary classification system for exploring churn patterns in IBM's fictional Telco sample. The application supports reproducible model evaluation and interactive scoring of similarly structured customer records.

## Data and label

7,043 records, 19 input fields, and binary `Churn` (Yes = 1). `customerID` is excluded from predictors. This snapshot has no validated forward prediction horizon. A future-period churn forecast requires timestamped business data and temporal evaluation. The source checksum and partition IDs are retained in reports.

## Training and selection

Fixed seed 42. Stratified 60/20/20 train/validation/test. Numeric medians, scaling parameters, and categorical encodings are fitted only on training rows. Logistic Regression, Random Forest, and optional XGBoost are compared with fixed parameters. The winner maximizes validation ROC-AUC; its threshold maximizes validation F1. The exact fitted pipeline and threshold are persisted together. There is no refit on test data or retuning from final test scores.

## Performance

Read [comparison.md](../reports/comparison.md) for the current measured results and [metrics.json](../reports/metrics.json) for all metrics, versions, dataset hash, skipped models, and split sizes. Comparison rows use threshold 0.5; the selected-model summary uses the deployed threshold. ROC-AUC measures ranking; precision and recall quantify the threshold tradeoff. Brier score summarizes probability error but does not establish calibration.

## Limitations

- Fictional sample, limited domain representativeness, and one random split.
- No external or temporal validation, uncertainty intervals, or subgroup performance study.
- Uncalibrated scores: a displayed 70% is a model estimate, not an empirically verified 70-in-100 future frequency.
- F1 optimization does not model retention costs, intervention effectiveness, or capacity.
- Demographic inputs (including gender and senior status) require fairness and suitability review before any real customer application.
- API bounds allow some inputs beyond observed training support. Such predictions are extrapolations and may be unreliable.
- Permutation importance and what-if changes are associative, not causal. Correlated variables can dilute importance.
- Synthetic fixture results are functional checks and must never be cited as IBM benchmark results.

## Deployment requirements

The API and UI support local execution and Docker Compose. Authentication, request monitoring, drift detection, and persistent request logging are outside the current implementation. Hosted operation requires those controls, trusted and versioned model artifacts, and validation on target-business data. A rollout plan should include calibration, subgroup evaluation, cost-based thresholds, and a controlled retention experiment. Load artifacts from trusted sources and restart the service after retraining.
