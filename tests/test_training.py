import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from churn.config import FEATURES, ROOT
from churn.data import load_data
from churn.pipeline import build_pipeline
from churn.train import choose_threshold, split_data, train


def test_splits_are_disjoint_stratified_and_reproducible():
    df = load_data(ROOT / "data/sample.csv")
    parts = split_data(df)
    ids = [set(part.customerID) for part in parts]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == set(df.customerID)
    for a, b in zip(parts, split_data(df)):
        assert a.customerID.tolist() == b.customerID.tolist()
        assert abs(a.Churn.eq("Yes").mean() - df.Churn.eq("Yes").mean()) < 0.07


def test_preprocessing_does_not_learn_from_inference():
    df = load_data(ROOT / "data/sample.csv")
    training, validation, _ = split_data(df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(
        training[FEATURES], training.Churn.eq("Yes")
    )
    imputer = (
        pipe.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
    )
    medians = imputer.statistics_.copy()
    changed = validation[FEATURES].copy()
    changed["TotalCharges"] = 1e9
    changed["PaymentMethod"] = "unseen category"
    assert np.isfinite(pipe.predict_proba(changed)).all()
    np.testing.assert_array_equal(imputer.statistics_, medians)
    assert "customerID" not in pipe.feature_names_in_
    assert "Churn" not in pipe.feature_names_in_


def test_duplicate_ids_are_rejected(tmp_path):
    df = pd.read_csv(ROOT / "data/sample.csv")
    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_data(path)


def test_threshold_uses_probabilities():
    assert 0.2 < choose_threshold([0, 0, 1, 1], np.array([0.1, 0.2, 0.4, 0.8])) <= 0.4


def test_training_emits_loadable_model_and_honest_reports(tmp_path):
    from churn.predict import load_bundle

    report = train(
        ROOT / "data/sample.csv",
        tmp_path / "models",
        tmp_path / "reports",
        skip_xgboost=True,
    )
    assert report["dataset"] != "IBM Telco"
    assert set(report["test_at_0_5"]) == {"Logistic Regression", "Random Forest"}
    assert report["selected_model"] == max(
        report["validation_at_0_5"],
        key=lambda n: report["validation_at_0_5"][n]["roc_auc"],
    )
    matrix = np.array(report["selected_test_at_tuned_threshold"]["confusion_matrix"])
    assert matrix.shape == (2, 2)
    assert matrix.sum() == report["split_counts"]["test"]
    assert (
        load_bundle(tmp_path / "models/model.joblib")["threshold"]
        == report["decision_threshold"]
    )
    assert (tmp_path / "reports/evaluation.png").exists()
