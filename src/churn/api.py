import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from .config import ROOT
from .predict import load_bundle, predict_customer
from .schema import Customer, Prediction

logger = logging.getLogger(__name__)


def create_app(model_path=None):
    path = Path(model_path or os.getenv("MODEL_PATH", ROOT / "artifacts/model.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = None
        try:
            app.state.bundle = load_bundle(path)
        except Exception:
            logger.exception(
                "Model unavailable. Train a compatible artifact before serving."
            )
        yield

    app = FastAPI(
        title="Hareth's Customer Churn API",
        version="1.0.0",
        lifespan=lifespan,
        description="Portfolio demo: estimated churn risk on IBM Telco-like records.",
    )

    @app.get("/health")
    def health(request: Request):
        bundle = request.app.state.bundle
        if bundle is None:
            raise HTTPException(
                503, "Model unavailable. Run python -m churn.train and restart the API."
            )
        return {
            "status": "ready",
            "model": bundle["model_name"],
            "dataset": bundle["dataset"],
            "threshold": bundle["threshold"],
        }

    @app.post("/predict", response_model=Prediction)
    def predict(customer: Customer, request: Request):
        bundle = request.app.state.bundle
        if bundle is None:
            raise HTTPException(
                503, "Model unavailable. Train the model and restart the API."
            )
        return predict_customer(bundle, customer)

    return app


app = create_app()
