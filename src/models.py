"""Model definitions, tuning and metrics."""
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def logistic_baseline(class_weight=None):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight=class_weight),
    )


def xgb_model(seed, **params):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
        **params,
    )


SEARCH_SPACE = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(2, 7),
    "learning_rate": loguniform(0.02, 0.3),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 8),
    "reg_lambda": loguniform(0.1, 10),
    "scale_pos_weight": [1, 3, 10, 28],
}


def tune_xgb(X, y, seed, n_iter=40, folds=5):
    """Random search scored on average precision, so no threshold is involved."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        xgb_model(seed), SEARCH_SPACE, n_iter=n_iter, scoring="average_precision",
        cv=cv, random_state=seed, n_jobs=1, refit=False,
    )
    search.fit(X, y)
    return search.best_params_, float(search.best_score_)


def oof_probabilities(model, X, y, seed, folds=5):
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def best_f1_threshold(y, prob):
    """Threshold with the highest F1 on the given (out-of-fold) predictions."""
    grid = np.unique(np.round(prob, 4))
    scores = [f1_score(y, prob >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics(y, prob, threshold):
    pred = prob >= threshold
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
    }


def bootstrap_f1(y, prob_a, thr_a, prob_b=None, thr_b=None, n=2000, seed=0):
    """95% percentile interval for F1 of A and, if B is given, for F1(A) - F1(B)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    a, b = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a.append(f1_score(y[idx], prob_a[idx] >= thr_a, zero_division=0))
        if prob_b is not None:
            b.append(f1_score(y[idx], prob_b[idx] >= thr_b, zero_division=0))
    a = np.array(a)
    out = {"f1_a": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
    if prob_b is not None:
        b = np.array(b)
        d = a - b
        out["f1_b"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        out["f1_diff"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out
