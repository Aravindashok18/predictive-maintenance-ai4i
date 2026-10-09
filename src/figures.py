"""EDA and result figures -> reports/figures/*.png (run after src.experiment)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import data
from .experiment import REPORTS

FIG = REPORTS / "figures"
OK, BAD = "#4c78a8", "#e45756"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)


def eda(df):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    by_type = df.groupby("type")[data.TARGET].agg(["mean", "sum", "count"]).loc[["L", "M", "H"]]
    ax[0].bar(by_type.index, by_type["mean"] * 100, color=OK)
    for i, (m, s, c) in enumerate(by_type.itertuples(index=False)):
        ax[0].text(i, m * 100 + 0.05, f"{int(s)}/{int(c)}", ha="center", fontsize=9)
    ax[0].set(title="Failure rate by product quality type", ylabel="% of machines that failed")

    modes = df[data.MODE_COLS].sum().sort_values()
    ax[1].barh(modes.index, modes.values, color=OK)
    ax[1].set(title="Failures by mode", xlabel="count (a machine can have more than one)")

    for label, col in ((0, OK), (1, BAD)):
        sub = df[df[data.TARGET] == label]
        ax[2].scatter(sub["rpm"], sub["torque_nm"], s=6 if label == 0 else 12, c=col,
                      alpha=0.25 if label == 0 else 0.9, label="failed" if label else "ok")
    ax[2].set(title="Speed vs torque", xlabel="rpm", ylabel="torque (Nm)")
    ax[2].legend(markerscale=2)
    save(fig, "eda_overview.png")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for col, a, title in (("power_w", ax[0], "Mechanical power (W)"),
                          ("temp_diff_k", ax[1], "Process - air temperature (K)"),
                          ("tool_wear_min", ax[2], "Tool wear (min)")):
        bins = np.linspace(df[col].min(), df[col].max(), 40)
        a.hist(df.loc[df[data.TARGET] == 0, col], bins=bins, density=True, alpha=0.6, color=OK, label="ok")
        a.hist(df.loc[df[data.TARGET] == 1, col], bins=bins, density=True, alpha=0.6, color=BAD, label="failed")
        a.set(title=title, ylabel="density")
    ax[0].legend()
    save(fig, "feature_distributions.png")


def results(df):
    imp = pd.read_csv(REPORTS / "feature_importance.csv").sort_values("permutation_drop_in_pr_auc")
    pred = pd.read_csv(REPORTS / "test_predictions.csv")
    res = json.loads((REPORTS / "results.json").read_text())
    final = res["headline"]["final_model"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].barh(imp["feature"], imp["permutation_drop_in_pr_auc"], xerr=imp["permutation_std"], color=OK)
    ax[0].set(title="Permutation importance on held-out data", xlabel="drop in PR-AUC when the feature is shuffled")
    for name, col, c in ((f"XGBoost ({final.split('/')[1]})", "p_xgboost", BAD), ("Logistic regression (raw)", "p_logreg_raw", OK)):
        p, r, _ = precision_recall_curve(pred["failure"], pred[col])
        ax[1].plot(r, p, color=c, label=name)
    ax[1].set(title="Precision-recall on held-out data", xlabel="recall", ylabel="precision", ylim=(0, 1.02))
    ax[1].legend(loc="lower left")
    save(fig, "results.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = data.add_features(data.load())
    eda(df)
    results(df)
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
