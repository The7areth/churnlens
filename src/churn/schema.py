from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float | None = Field(default=None, ge=0, le=120000)

    @model_validator(mode="after")
    def consistent_services(self):
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError("MultipleLines must match PhoneService")
        for field in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            if (self.InternetService == "No") != (
                getattr(self, field) == "No internet service"
            ):
                raise ValueError(f"{field} must match InternetService")
        return self


class Prediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    risk: Literal["High", "Low"]
    predicted_churn: bool
    threshold: float
    model: str
    dataset: str
