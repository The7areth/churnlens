import pytest
from fastapi.testclient import TestClient
from churn.api import create_app
from churn.predict import load_bundle, predict_customer
from churn.schema import Customer


def test_persistence_api_parity_and_missing_total(model_path, customer):
    customer["TotalCharges"] = None
    expected = predict_customer(load_bundle(model_path), Customer(**customer))
    with TestClient(create_app(model_path)) as client:
        assert client.get("/health").status_code == 200
        result = client.post("/predict", json=customer)
        assert result.status_code == 200
        assert result.json() == expected
        assert 0 <= result.json()["churn_probability"] <= 1
        assert result.json()["predicted_churn"] == (
            result.json()["churn_probability"] >= 0.4
        )


@pytest.mark.parametrize(
    "change",
    [
        {"tenure": -1},
        {"MonthlyCharges": -5},
        {"Contract": "unknown"},
        {"Churn": "Yes"},
        {"PhoneService": "No", "MultipleLines": "Yes"},
        {"InternetService": "No", "OnlineSecurity": "Yes"},
    ],
)
def test_bad_inputs_return_422(model_path, customer, change):
    with TestClient(create_app(model_path)) as client:
        result = client.post("/predict", json={**customer, **change})
        assert result.status_code == 422


def test_missing_model_returns_503(tmp_path, customer):
    with TestClient(create_app(tmp_path / "absent.joblib")) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=customer).status_code == 503


def test_documented_example_is_valid(model_path, customer):
    with TestClient(create_app(model_path)) as client:
        assert client.post("/predict", json=customer).status_code == 200
