"""Run the full experiment: split, tune on the training part, score on held-out data.

    python -m src.experiment            # main run (seed 42) -> reports/results.json
    python -m src.experiment --repeats 10   # also repeat the whole protocol on 10 other splits

Protocol, applied identically to every model:
  1. stratified 80/20 split, the 20% is never used for any decision;
  2. XGBoost hyperparameters are chosen by 5-fold CV on the training part (average precision);
  3. the decision threshold is the F1-maximising one on out-of-fold training predictions;
  4. the held-out part is scored once.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from . import data, models

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def evaluate_split(df, seed, n_iter, keep=False):
    train, test = train_test_split(df, test_size=0.2, stratify=df[data.TARGET], random_state=seed)
    y_tr, y_te = train[data.TARGET].to_numpy(), test[data.TARGET].to_numpy()
    out = {"split": {"train_rows": len(train), "test_rows": len(test),
                     "train_failures": int(y_tr.sum()), "test_failures": int(y_te.sum())},
           "models": {}}
    fitted = {}

    for fs_name, cols in data.FEATURE_SETS.items():
        X_tr, X_te = train[cols], test[cols]

        # logistic regression: default 0.5 threshold, and the CV-tuned threshold
        lr = models.logistic_baseline()
        lr.fit(X_tr, y_tr)
        p_lr = lr.predict_proba(X_te)[:, 1]
        thr_lr = models.best_f1_threshold(y_tr, models.oof_probabilities(models.logistic_baseline(), X_tr, y_tr, seed))
        out["models"][f"logreg/{fs_name}/thr=0.5"] = models.metrics(y_te, p_lr, 0.5)
        out["models"][f"logreg/{fs_name}/thr=cv"] = models.metrics(y_te, p_lr, thr_lr)

        # xgboost
        params, cv_ap = models.tune_xgb(X_tr, y_tr, seed, n_iter=n_iter)
        xgb = models.xgb_model(seed, **params).fit(X_tr, y_tr)
        p_xgb = xgb.predict_proba(X_te)[:, 1]
        oof = models.oof_probabilities(models.xgb_model(seed, **params), X_tr, y_tr, seed)
        thr_xgb = models.best_f1_threshold(y_tr, oof)
        res = models.metrics(y_te, p_xgb, thr_xgb)
        res["cv_average_precision"] = cv_ap
        res["params"] = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in params.items()}
        out["models"][f"xgboost/{fs_name}"] = res

        fitted[fs_name] = dict(lr=lr, p_lr=p_lr, thr_lr=thr_lr, xgb=xgb, p_xgb=p_xgb,
                               thr_xgb=thr_xgb, cv_ap=cv_ap, params=params, cols=cols)

    # the final feature set is picked from training-only CV, not from the test score
    final_fs = max(fitted, key=lambda k: fitted[k]["cv_ap"])
    out["final_feature_set"] = final_fs
    if keep:
        return out, fitted, final_fs, train, test
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=0)
    args = ap.parse_args()

    REPORTS.mkdir(exist_ok=True)
    raw = data.load()
    df = data.add_features(raw)
    results = {"audit": data.audit(raw), "seed": args.seed, "n_iter": args.n_iter}

    main_out, fitted, final_fs, train, test = evaluate_split(df, args.seed, args.n_iter, keep=True)
    results["main"] = main_out
    f = fitted[final_fs]
    y_te = test[data.TARGET].to_numpy()

    # logistic regression baseline = raw features, default threshold (the plain starting point)
    base = fitted["raw"]
    results["headline"] = {
        "final_model": f"xgboost/{final_fs}",
        "baseline": "logreg/raw/thr=0.5",
        "bootstrap_95ci": models.bootstrap_f1(y_te, f["p_xgb"], f["thr_xgb"], base["p_lr"], 0.5),
        "bootstrap_95ci_vs_tuned_logreg": models.bootstrap_f1(
            y_te, f["p_xgb"], f["thr_xgb"], fitted[final_fs]["p_lr"], fitted[final_fs]["thr_lr"])["f1_diff"],
    }

    # recall per failure mode on the held-out part
    pred = f["p_xgb"] >= f["thr_xgb"]
    modes = {}
    for m in data.MODE_COLS:
        mask = test[m].to_numpy() == 1
        modes[m] = {"cases": int(mask.sum()), "caught": int(pred[mask].sum())}
    unlabeled = (test[data.TARGET].to_numpy() == 1) & (test[data.MODE_COLS].sum(axis=1).to_numpy() == 0)
    modes["failure_without_mode_flag"] = {"cases": int(unlabeled.sum()), "caught": int(pred[unlabeled].sum())}
    results["per_failure_mode"] = modes

    # which inputs matter
    perm = permutation_importance(f["xgb"], test[f["cols"]], y_te, scoring="average_precision",
                                  n_repeats=20, random_state=args.seed)
    gain = f["xgb"].get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame({"feature": f["cols"],
                        "permutation_drop_in_pr_auc": perm.importances_mean,
                        "permutation_std": perm.importances_std,
                        "xgb_gain": [gain.get(c, 0.0) for c in f["cols"]]})
    imp = imp.sort_values("permutation_drop_in_pr_auc", ascending=False)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)

    pd.DataFrame({"product_id": test["product_id"].to_numpy(), "failure": y_te,
                  "p_xgboost": f["p_xgb"], "p_logreg_raw": base["p_lr"]}).to_csv(
        REPORTS / "test_predictions.csv", index=False)
    (REPORTS / "final_model.json").write_text(json.dumps(
        {"feature_set": final_fs, "features": f["cols"], "threshold": f["thr_xgb"],
         "params": main_out["models"][f"xgboost/{final_fs}"]["params"]}, indent=2))
    f["xgb"].save_model(str(REPORTS / "xgb_final.json"))

    if args.repeats:
        rows = []
        for r in range(args.repeats):
            s = 1000 + r
            o = evaluate_split(df, s, max(15, args.n_iter // 2))
            fs = o["final_feature_set"]
            rows.append({"seed": s,
                         "xgb_f1": o["models"][f"xgboost/{fs}"]["f1"],
                         "xgb_raw_f1": o["models"]["xgboost/raw"]["f1"],
                         "logreg_raw_f1": o["models"]["logreg/raw/thr=0.5"]["f1"],
                         "logreg_raw_tuned_thr_f1": o["models"]["logreg/raw/thr=cv"]["f1"],
                         "logreg_eng_tuned_thr_f1": o["models"][f"logreg/{fs}/thr=cv"]["f1"],
                         "final_feature_set": fs})
            print(f"repeat {r + 1}/{args.repeats}: xgb {rows[-1]['xgb_f1']:.3f} "
                  f"logreg {rows[-1]['logreg_raw_f1']:.3f}", flush=True)
        rep = pd.DataFrame(rows)
        rep.to_csv(REPORTS / "repeats.csv", index=False)
        num = rep.drop(columns=["seed", "final_feature_set"])
        results["repeats"] = {"n": args.repeats, "mean": num.mean().to_dict(), "std": num.std().to_dict(),
                              "min": num.min().to_dict(), "max": num.max().to_dict()}

    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    h = results["headline"]
    m = main_out["models"]
    print("final:", h["final_model"])
    print("xgb  F1 %.3f  P %.3f  R %.3f  PR-AUC %.3f" % (m[h["final_model"]]["f1"], m[h["final_model"]]["precision"],
                                                       m[h["final_model"]]["recall"], m[h["final_model"]]["pr_auc"]))
    b = m[h["baseline"]]
    print("base F1 %.3f  P %.3f  R %.3f  PR-AUC %.3f" % (b["f1"], b["precision"], b["recall"], b["pr_auc"]))


if __name__ == "__main__":
    main()
