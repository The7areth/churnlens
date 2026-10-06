from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]
SEED = 42
