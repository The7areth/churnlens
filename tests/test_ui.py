from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
from churn.api import create_app
from churn.config import ROOT


def test_ui_submits_to_api_and_normalizes_absent_services(model_path, monkeypatch):
    with TestClient(create_app(model_path)) as client:
        monkeypatch.setattr("requests.get", lambda url, **kwargs: client.get("/health"))
        monkeypatch.setattr(
            "requests.post",
            lambda url, json, **kwargs: client.post("/predict", json=json),
        )
        app = AppTest.from_file(
            str(ROOT / "app/streamlit_app.py"), default_timeout=30
        ).run()
        assert not app.exception
        for widget in app.selectbox:
            if widget.label == "Phone service":
                widget.set_value("No")
            if widget.label == "Internet service":
                widget.set_value("No")
        app.button[0].click().run()
        assert not app.exception
        assert not app.error
        assert app.metric[0].label == "Churn probability"
        assert app.metric[1].value in ["High", "Low"]
