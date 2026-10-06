"""Descriptive analysis of the training partition only."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_eda(df, output):
    output.mkdir(parents=True, exist_ok=True)
    target = df.Churn.eq("Yes")
    summary = {
        "scope": "training partition only",
        "rows": len(df),
        "churn_rate": float(target.mean()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "contract_churn_rate": df.assign(churn=target)
        .groupby("Contract")
        .churn.mean()
        .to_dict(),
    }
    (output / "eda.json").write_text(json.dumps(summary, indent=2))
    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    df.Churn.value_counts().reindex(["No", "Yes"]).plot.bar(
        ax=axes[0], color=["#94a3b8", "#0d9488"], rot=0
    )
    axes[0].set(title="Training class balance", ylabel="Customers", xlabel="Churn")
    df.assign(churn=target).groupby("Contract").churn.mean().sort_values().plot.barh(
        ax=axes[1], color="#0d9488"
    )
    axes[1].set(title="Churn rate by contract", xlabel="Fraction", ylabel="")
    axes[2].hist(
        [df.loc[~target, "tenure"], df.loc[target, "tenure"]],
        bins=12,
        label=["Retained", "Churned"],
        color=["#94a3b8", "#0d9488"],
    )
    axes[2].set(title="Tenure distribution", xlabel="Months", ylabel="Customers")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "eda.png", dpi=160)
    plt.close(fig)
