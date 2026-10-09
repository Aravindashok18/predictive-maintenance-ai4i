import numpy as np

from src import models


def test_best_threshold_separates_a_clean_problem():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    t = models.best_f1_threshold(y, p)
    assert 0.4 < t <= 0.8
    assert models.metrics(y, p, t)["f1"] == 1.0


def test_metrics_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1])
    m = models.metrics(y, p, 0.5)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(int)
    p = np.clip(y * 0.6 + rng.random(400) * 0.5, 0, 1)
    lo, hi = models.bootstrap_f1(y, p, 0.5, n=300)["f1_a"]
    f1 = models.metrics(y, p, 0.5)["f1"]
    assert lo <= f1 <= hi


def test_xgboost_learns_a_simple_rule():
    rng = np.random.default_rng(1)
    X = rng.random((600, 3))
    y = (X[:, 0] > 0.7).astype(int)
    model = models.xgb_model(0, n_estimators=50, max_depth=3).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
