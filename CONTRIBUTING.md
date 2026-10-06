# Contributing to ChurnLens

Use Python 3.12 and install `requirements-dev.txt`, followed by `pip install --no-deps -e .`.

Before opening a pull request:

```bash
black src app tests
pytest -q
python -m churn.train --data data/sample.csv --artifacts artifacts/sample --reports reports/sample
```

Use the synthetic fixture for routine checks and keep its reports in a separate directory. If changing the model or preprocessing, describe the selection protocol before running experiments. Keep the existing test results as a historical baseline; do not repeatedly optimize against the same held-out test set and present it as fresh evidence.

Include the problem, behavior change, and verification in your pull request. Explain any changes to schema, model compatibility, or dataset handling. Add a focused regression test when fixing a substantive bug. Avoid committing raw data, model binaries, credentials, environments, or sample-run output.

For new algorithms, preserve the same split and preprocessing boundary. For business improvements, specify the decision objective and how it will be evaluated. Performance claims must identify their dataset, split, threshold, and metric.
