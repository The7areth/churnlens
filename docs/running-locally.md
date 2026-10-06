# Running ChurnLens

Use Python 3.12. Run commands from the repository root. Follow the README quick start for the standard installation.

## Dependencies and model training

`requirements.txt` contains the core runtime; `requirements-xgboost.txt` adds XGBoost; `requirements-dev.txt` adds test and formatting tools. Install with `python -m pip install -r requirements-dev.txt`, then `python -m pip install --no-deps -e .`.

```bash
python -m churn.data
python -m churn.train
```

The downloader retries interrupted transfers and requires the pinned SHA-256. It refuses a changed source rather than silently training on a different file. Training generates `artifacts/model.joblib` and the reports. Downloading data needs internet; predicting with an existing artifact does not.

The original delivered ZIP also contains a verified full CSV and a trained model. A Git clone generates these locally. Retrain after changing ML library versions. `requirements-lock.txt` records the tested Python 3.12 development environment, including test and formatting tools.

## Start the services

Terminal 1:

```bash
uvicorn churn.api:app --host 127.0.0.1 --port 8000
```

Terminal 2, after activating the same environment:

```bash
streamlit run app/streamlit_app.py
```

- Application: http://localhost:8501
- API documentation: http://localhost:8000/docs
- Readiness: http://localhost:8000/health

`MODEL_PATH` overrides the API artifact. `API_URL` overrides the UI backend; default is `http://127.0.0.1:8000`. Restart the API after replacing an artifact. The model-evidence tab reads local reports, so those reports can differ if you point the UI at another API.

## Offline synthetic smoke test

```bash
python -m churn.train --data data/sample.csv --artifacts artifacts/sample --reports reports/sample
MODEL_PATH=artifacts/sample/model.joblib uvicorn churn.api:app --port 8000
```

On PowerShell, set `$env:MODEL_PATH="artifacts/sample/model.joblib"` before starting Uvicorn. The sample contains 180 synthetic rows and deliberately simplified labels. It verifies functionality; quote IBM benchmark results from the main reports, not sample scores.

## XGBoost on macOS

XGBoost can require an OpenMP library. If you use Homebrew, install it with `brew install libomp`. You can also run the two scikit-learn models explicitly:

```bash
python -m churn.train --skip-xgboost
```

Inspect `skipped_models` in the generated metrics. The three-model result in the repository was run successfully; a fallback run must be described as a two-model run. Docker includes Linux OpenMP.

## Docker after local training

```bash
docker compose up --build
```

Open the same UI and API URLs. Stop with Ctrl+C and then `docker compose down`. Images include the artifact and reports present at build time; rebuild after retraining. Services run as a non-root user and expose ports on localhost.

## Docker from a fresh clone

The following volume syntax is for macOS/Linux shells:

```bash
docker compose build
mkdir -p artifacts reports data/raw
docker compose run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" -v "$PWD/reports:/app/reports" \
  api sh -c 'python -m churn.data && python -m churn.train'
docker compose up --build
```

The temporary container writes data, the model, and reports to your checkout. The second build includes those outputs. The UI depends on a healthy API; a missing model keeps the API unready.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `/health` returns 503 | Train the model, check MODEL_PATH, then restart the API. |
| Prediction returns 422 | Compare inputs with `examples/customer.json`; check service consistency and categories. |
| UI cannot reach API | Start Uvicorn; check API_URL and ports. In Compose, use `http://api:8000`. |
| Port is occupied | Choose another port and update API_URL for the UI. |
| XGBoost import fails | Install OpenMP, use Docker, or run the documented two-model fallback. |
| Dataset download fails | Retry; use the supplied CSV or synthetic sample if offline. Never bypass the checksum without reviewing the changed source. |
| Numerical/deprecation warnings | Check versions and the local verification notes; retrain in a consistent environment. |

Only load trusted joblib files. This is a local demonstration, with no authentication or production hardening.
