# Verification — October 6, 2026

Environment: macOS ARM64, Python 3.12.14. Exact Python package versions are in `requirements-lock.txt`.

- Installed the project as an editable package; `pip check` reported no broken requirements.
- Verified the downloaded IBM CSV against its pinned SHA-256; validated all 7,043 records and unique customer IDs.
- Completed the Logistic Regression / Random Forest / XGBoost comparison. Logistic Regression won by validation ROC-AUC. Test results are in `metrics.json`; no test-driven retuning was performed.
- **17 automated tests passed.** After shortening one UI label, the affected UI integration test passed again.
- The tests exercise invalid inputs, persistence/API parity, missing model readiness, disjoint splits, inference isolation, interrupted download retries, checksum protection, complete synthetic training, and UI-to-API service normalization.
- Started FastAPI and Streamlit locally. Live `/health` returned ready, and the documented example returned a probability of 0.2877686 and Low risk. A default customer submitted in the browser returned 61.1% and High risk at a 49.5% threshold. Different inputs intentionally produce different results.
- Visually reviewed EDA/evaluation charts and the live browser prediction. `demo.jpg` records the working interface.

## Environment observations

The Mac did not have a system OpenMP installation. For the three-model benchmark, `DYLD_LIBRARY_PATH` pointed to the OpenMP runtime shipped with the installed scikit-learn wheel (`.venv/lib/python3.12/site-packages/sklearn/.dylibs`). No system library or security setting was changed. The selected Logistic Regression artifact serves without this override. For normal macOS setup, install `libomp` as described in the README; Docker includes Linux OpenMP. Missing XGBoost imports are handled by a recorded two-model fallback.

This combination of numerical libraries on macOS emitted matrix-operation warnings, and some dependencies emitted deprecation/solver-option warnings. Every validation probability was finite. Independent Python scalar dot products and sigmoid calculations matched the persisted Logistic Regression pipeline with maximum absolute difference **2.22e-16**; batch versus individual-row predictions agreed within **1.11e-16**. The solver converged in 38 iterations. These checks support the reported inference results, but do not establish portability to every platform.

## Not verified here

- Docker image build and container execution: Docker CLI/engine was unavailable in this environment. Dockerfile, Compose configuration, and CI build steps are included.
- Current GitHub Actions execution: consult the [live workflow](https://github.com/The7areth/churnlens/actions/workflows/ci.yml); publication-time results are recorded below when verified.
- Production deployment, external validation, calibration, fairness, or measured retention/revenue impact.
