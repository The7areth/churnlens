import json
import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from churn.config import ROOT, FEATURES
from churn.data import load_data
from churn.pipeline import build_pipeline


@pytest.fixture
def customer():
    return json.loads((ROOT / "examples/customer.json").read_text())


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    df = load_data(ROOT / "data/sample.csv")
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(
        df[FEATURES], df.Churn.eq("Yes")
    )
    path = tmp_path_factory.mktemp("artifact") / "model.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": FEATURES,
            "threshold": 0.4,
            "model_name": "Test Logistic Regression",
            "dataset": "Synthetic test fixture",
        },
        path,
    )
    return path
