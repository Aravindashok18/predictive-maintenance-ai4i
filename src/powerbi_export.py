"""Write the tables the Power BI report is built on -> powerbi/*.csv.

Risk scores for every machine are out-of-fold: each row is scored by a model that
did not see it in training, so the dashboard does not show memorised answers.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import data, models
from .experiment import REPORTS

OUT = Path(__file__).resolve().parent.parent / "powerbi"


def main():
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((REPORTS / "final_model.json").read_text())
    res = json.loads((REPORTS / "results.json").read_text())
    df = data.add_features(data.load())
    X, y = df[cfg["features"]], df[data.TARGET].to_numpy()

    model = models.xgb_model(res["seed"], **cfg["params"])
    risk = models.oof_probabilities(model, X, y, res["seed"])

    fact = df[["product_id", "type", "air_temp_k", "process_temp_k", "rpm", "torque_nm",
               "tool_wear_min", "power_w", "temp_diff_k", "wear_torque", data.TARGET] + data.MODE_COLS].copy()
    fact["failure_risk"] = risk.round(4)
    fact["flagged"] = (risk >= cfg["threshold"]).astype(int)
    fact["wear_band"] = pd.cut(fact["tool_wear_min"], [-1, 50, 100, 150, 200, 300],
                               labels=["0-50", "51-100", "101-150", "151-200", "201+"]).astype(str)
    fact["rpm_band"] = pd.cut(fact["rpm"], [0, 1300, 1400, 1500, 1700, 3000],
                              labels=["<1300", "1300-1399", "1400-1499", "1500-1699", "1700+"]).astype(str)
    fact = fact.round({"power_w": 1, "wear_torque": 1})
    fact.to_csv(OUT / "fact_machines.csv", index=False)

    modes = (df.melt(id_vars=["product_id", "type"], value_vars=data.MODE_COLS,
                     var_name="failure_mode", value_name="occurred")
               .query("occurred == 1")[["product_id", "type", "failure_mode"]])
    modes.to_csv(OUT / "fact_failure_modes.csv", index=False)

    pd.read_csv(REPORTS / "feature_importance.csv").to_csv(OUT / "feature_importance.csv", index=False)

    rows = [{"model": k, **{m: v[m] for m in ("threshold", "precision", "recall", "f1", "pr_auc", "roc_auc")}}
            for k, v in res["main"]["models"].items()]
    pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False)

    p, r, t = precision_recall_curve(y, risk)
    curve = pd.DataFrame({"threshold": list(t) + [1.0], "precision": p, "recall": r}).round(4)
    curve.drop_duplicates(subset=["precision", "recall"]).to_csv(OUT / "pr_curve.csv", index=False)
    print("wrote", sorted(f.name for f in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
