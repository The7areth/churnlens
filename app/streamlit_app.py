import json
import os
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
from churn.config import ROOT

st.set_page_config(
    page_title="ChurnLens · Hareth Al-Fawaz", page_icon="◈", layout="wide"
)
st.markdown(
    """<style>
.stApp {background:#f8fafc} h1 {letter-spacing:-1.5px} div[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px}
</style>""",
    unsafe_allow_html=True,
)
st.caption("HARETH AL-FAWAZ  /  APPLIED MACHINE LEARNING")
st.title("ChurnLens")
st.write("Turn customer signals into a focused retention conversation.")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
predict_tab, evidence_tab, about_tab = st.tabs(
    ["Customer prediction", "Model evidence", "About the project"]
)
with predict_tab:
    st.info(
        "Portfolio demo using IBM's fictional telecom sample. Scores estimate churn likelihood; they do not prove why a customer will leave."
    )
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        health_response.raise_for_status()
        health = health_response.json()
        st.caption(
            f"API ready · {health['model']} · {health['dataset']} · threshold {health['threshold']:.3f}"
        )
        if health["dataset"] != "IBM Telco":
            st.warning(
                "This API is running a custom/synthetic demo model. Do not quote its scores as an IBM benchmark."
            )
    except requests.RequestException:
        st.warning(
            "API unavailable. Start the trained FastAPI service on port 8000, then submit again."
        )
    with st.form("customer"):
        left, middle, right = st.columns(3)
        with left:
            st.subheader("Customer")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.checkbox("Senior citizen")
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly charges ($)", 0.0, 1000.0, 75.0)
            total = st.number_input("Total charges ($)", 0.0, 120000.0, 900.0)
            total_unknown = st.checkbox("Total charges unknown (use training median)")
        with middle:
            st.subheader("Services")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox(
                "Multiple lines (ignored without phone)", ["No", "Yes"]
            )
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online security", ["No", "Yes"])
            backup = st.selectbox("Online backup", ["No", "Yes"])
            protection = st.selectbox("Device protection", ["No", "Yes"])
            support = st.selectbox("Tech support", ["No", "Yes"])
        with right:
            st.subheader("Account")
            tv = st.selectbox("Streaming TV", ["No", "Yes"])
            movies = st.selectbox("Streaming movies", ["No", "Yes"])
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            st.caption(
                "If internet service is No, all internet add-ons are automatically marked ‘No internet service’."
            )
        submitted = st.form_submit_button(
            "Estimate churn risk", type="primary", use_container_width=True
        )
    if submitted:
        payload = dict(
            gender=gender,
            SeniorCitizen=int(senior),
            Partner=partner,
            Dependents=dependents,
            tenure=tenure,
            PhoneService=phone,
            MultipleLines=multiple if phone == "Yes" else "No phone service",
            InternetService=internet,
            Contract=contract,
            PaperlessBilling=paperless,
            PaymentMethod=payment,
            MonthlyCharges=monthly,
            TotalCharges=None if total_unknown else total,
        )
        for key, value in zip(
            [
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ],
            [security, backup, protection, support, tv, movies],
        ):
            payload[key] = value if internet != "No" else "No internet service"
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            a, b, c = st.columns(3)
            a.metric("Churn probability", f"{result['churn_probability']:.1%}")
            b.metric("Risk band", result["risk"])
            c.metric("Decision threshold", f"{result['threshold']:.1%}")
            st.progress(result["churn_probability"])
            st.write(
                "Suggested next step: review service experience and suitability for retention outreach."
                if result["predicted_churn"]
                else "Suggested next step: continue routine engagement and monitor changes."
            )
            st.caption(
                "Risk bands use the validation-selected threshold. Changing inputs illustrates model behavior, not the causal effect of an intervention."
            )
            st.download_button(
                "Download prediction JSON",
                json.dumps({"customer": payload, "prediction": result}, indent=2),
                "churn-prediction.json",
                "application/json",
            )
        except requests.RequestException as exc:
            st.error(
                f"Could not obtain a prediction. Check API availability and customer fields. {exc}"
            )
with evidence_tab:
    report_path = ROOT / "reports/metrics.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        st.subheader("Held-out model comparison")
        st.caption(
            f"Saved training report · {report['dataset']} · seed {report['seed']} · {report['split_counts']}"
        )
        st.dataframe(
            pd.DataFrame(report["test_at_0_5"])
            .T.drop(columns="confusion_matrix")
            .astype(float)
            .round(3),
            use_container_width=True,
        )
        st.write(
            f"Selected by validation ROC-AUC: **{report['selected_model']}**. Threshold selected by validation F1: **{report['decision_threshold']:.3f}**."
        )
        st.caption(
            "Table uses threshold 0.5 for comparison. The API uses the tuned threshold shown above. Local reports may differ from an API using a custom MODEL_PATH."
        )
        for file in ["evaluation.png", "eda.png"]:
            if (ROOT / "reports" / file).exists():
                st.image(str(ROOT / "reports" / file), use_container_width=True)
        importance = ROOT / "reports/feature_importance.csv"
        if importance.exists():
            st.write("Validation permutation importance · mean ROC-AUC decrease")
            st.bar_chart(
                pd.read_csv(importance)
                .head(8)
                .set_index("feature")["mean_auc_decrease"]
            )
    else:
        st.info("Train a model to generate the comparison and EDA charts.")
with about_tab:
    st.markdown(
        """### From raw data to a working service
**Dataset validation → stratified split → preprocessing pipeline → three-model comparison → persisted model → FastAPI → Streamlit**

Preprocessing learns only from training data. Model choice and the decision threshold use validation data. Final metrics come from a held-out test partition.

This is an educational benchmark, not a deployed retention program. The sample does not establish a future churn horizon; production use would require temporal validation, calibration, fairness checks, monitoring, and measured outreach costs.

Built for Hareth Al-Fawaz's applied AI portfolio. See the repository README for reproducibility and model limitations.
"""
    )
