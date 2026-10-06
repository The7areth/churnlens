"""Reproduce the model comparison; never select models using test scores."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    brier_score_loss,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from .config import FEATURES, ROOT, SEED
from .data import load_data, SHA256
from .pipeline import build_pipeline
from .eda import write_eda
import matplotlib.pyplot as plt


def split_data(df):
    train, rest = train_test_split(
        df, test_size=0.4, stratify=df.Churn, random_state=SEED
    )
    validation, test = train_test_split(
        rest, test_size=0.5, stratify=rest.Churn, random_state=SEED
    )
    return train, validation, test


def evaluate(y, probability, threshold):
    predicted = probability >= threshold
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def choose_threshold(y, probability):
    grid = np.linspace(0.05, 0.95, 181)
    # First maximum breaks ties deterministically in favor of recall.
    scores = [
        f1_score(y, probability >= threshold, zero_division=0) for threshold in grid
    ]
    return float(grid[int(np.argmax(scores))])


def train(data_path, artifact_dir, report_dir, skip_xgboost=False):
    df = load_data(data_path)
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    dataset = (
        "IBM Telco"
        if digest == SHA256
        else "Custom / synthetic demo (not IBM benchmark)"
    )
    train_df, val_df, test_df = split_data(df)
    X, y = train_df[FEATURES], train_df.Churn.eq("Yes").astype(int)
    Xv, yv = val_df[FEATURES], val_df.Churn.eq("Yes").astype(int)
    Xt, yt = test_df[FEATURES], test_df.Churn.eq("Yes").astype(int)
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=2,
        ),
    }
    skipped = {}
    if not skip_xgboost:
        try:
            from xgboost import XGBClassifier

            estimators["XGBoost"] = XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=2,
            )
        except (ImportError, OSError, ValueError) as exc:
            skipped["XGBoost"] = str(exc)
            print(f"XGBoost unavailable; continuing with sklearn: {exc}")
    else:
        skipped["XGBoost"] = "Explicitly disabled"
    pipelines, validation = {}, {}
    for name, estimator in estimators.items():
        print(f"Training {name}…", flush=True)
        pipeline = build_pipeline(estimator).fit(X, y)
        pipelines[name] = pipeline
        validation[name] = evaluate(yv, pipeline.predict_proba(Xv)[:, 1], 0.5)
    winner = max(validation, key=lambda name: validation[name]["roc_auc"])
    model = pipelines[winner]
    threshold = choose_threshold(yv, model.predict_proba(Xv)[:, 1])
    # Keep this fitted pipeline unchanged: threshold was chosen for this exact model.
    tests = {
        name: evaluate(yt, pipe.predict_proba(Xt)[:, 1], 0.5)
        for name, pipe in pipelines.items()
    }
    final_test = evaluate(yt, model.predict_proba(Xt)[:, 1], threshold)
    baseline = DummyClassifier(strategy="prior").fit(X, y)
    reports = {
        "dataset": dataset,
        "dataset_sha256": digest,
        "seed": SEED,
        "split_counts": {"train": len(X), "validation": len(Xv), "test": len(Xt)},
        "selected_model": winner,
        "selection_metric": "validation ROC-AUC at fixed model parameters",
        "decision_threshold": threshold,
        "threshold_selection": "maximum validation F1 on 0.05–0.95 grid, step 0.005",
        "validation_at_0_5": validation,
        "test_at_0_5": tests,
        "selected_test_at_tuned_threshold": final_test,
        "baseline_test": evaluate(yt, baseline.predict_proba(Xt)[:, 1], 0.5),
        "skipped_models": skipped,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["pandas", "numpy", "scikit-learn", "joblib"]
        },
        "python": platform.python_version(),
    }
    if "XGBoost" in pipelines:
        reports["versions"]["xgboost"] = importlib.metadata.version("xgboost")
    artifact_dir, report_dir = Path(artifact_dir), Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": model,
        "threshold": threshold,
        "model_name": winner,
        "dataset": dataset,
        "features": FEATURES,
        "metadata": reports,
    }
    joblib.dump(bundle, artifact_dir / "model.joblib")
    (report_dir / "metrics.json").write_text(json.dumps(reports, indent=2))
    (report_dir / "split_ids.json").write_text(
        json.dumps(
            {
                "train": train_df.customerID.tolist(),
                "validation": val_df.customerID.tolist(),
                "test": test_df.customerID.tolist(),
            },
            indent=2,
        )
    )
    write_eda(train_df, report_dir)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, pipe in pipelines.items():
        RocCurveDisplay.from_predictions(
            yt, pipe.predict_proba(Xt)[:, 1], name=name, ax=axes[0]
        )
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title("Held-out test ROC curves")
    ConfusionMatrixDisplay(
        np.asarray(final_test["confusion_matrix"]),
        display_labels=["Retained", "Churned"],
    ).plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title(f"{winner} · threshold {threshold:.3f}")
    fig.tight_layout()
    fig.savefig(report_dir / "evaluation.png", dpi=160)
    plt.close(fig)
    importance = permutation_importance(
        model, Xv, yv, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1
    )
    import pandas as pd

    pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_auc_decrease": importance.importances_mean,
            "std": importance.importances_std,
        }
    ).sort_values("mean_auc_decrease", ascending=False).to_csv(
        report_dir / "feature_importance.csv", index=False
    )
    lines = [
        "# Model comparison",
        "",
        f"Dataset: {dataset}. Seed: {SEED}. Train / validation / test: {len(X)} / {len(Xv)} / {len(Xt)}.",
        "",
        "All rows below use threshold 0.5; the model was selected using validation ROC-AUC.",
        "",
        "| Model | Validation AUC | Test AUC | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in tests.items():
        lines.append(
            f"| {name} | {validation[name]['roc_auc']:.3f} | {metrics['roc_auc']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |"
        )
    lines += [
        "",
        f"Selected: **{winner}**, validation-tuned threshold **{threshold:.3f}**.",
        f"Final test precision {final_test['precision']:.3f}, recall {final_test['recall']:.3f}, F1 {final_test['f1']:.3f}, ROC-AUC {final_test['roc_auc']:.3f}.",
        f"Confusion matrix [[TN, FP], [FN, TP]]: `{final_test['confusion_matrix']}`.",
        "",
        "Risk is High at or above the selected threshold; Low otherwise. Scores are model estimates, not calibrated guarantees.",
        "No test-based retuning or refitting after threshold selection. Feature importance uses validation data and is associative, not causal.",
    ]
    (report_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"selected_model": winner, "threshold": threshold, "test": final_test},
            indent=2,
        )
    )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/telco.csv")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--skip-xgboost", action="store_true")
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(
            "Dataset missing. Run python -m churn.data or pass --data data/sample.csv for a synthetic smoke demo."
        )
    train(args.data, args.artifacts, args.reports, args.skip_xgboost)


if __name__ == "__main__":
    main()
