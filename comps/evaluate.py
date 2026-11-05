"""Baselines, metrics and prediction intervals.

A GNN result means nothing on its own. It has to beat what a careful engineer would build
first: the median of nearby sales, and a gradient-boosted model with location features.
"""
from __future__ import annotations

import numpy as np

from comps.graph import CompsGraph


# ---------- metrics (industry style for automated valuations) ----------

def pct_errors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(predicted - actual) / actual


def summarise(actual: np.ndarray, predicted: np.ndarray) -> dict:
    e = pct_errors(actual, predicted)
    return {
        "median_abs_pct_error": round(float(np.median(e)) * 100, 2),
        "within_10pct": round(float(np.mean(e <= 0.10)) * 100, 1),
        "within_20pct": round(float(np.mean(e <= 0.20)) * 100, 1),
    }


# ---------- baselines ----------

def comps_features(log_price: np.ndarray, g: CompsGraph) -> np.ndarray:
    """Median log price of each sale's comparables, how many there are, and how far away.

    NaN where a sale has no comparables. Uses only earlier sales, like the GNN.
    """
    vals = np.where(g.mask, log_price[np.clip(g.nbr, 0, None)], np.nan)
    count = g.mask.sum(1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(count[:, None] > 0, vals, 0.0), axis=1)
        dist = np.where(count > 0, (g.dist_km * g.mask).sum(1) / np.maximum(count, 1), np.nan)
    med = np.where(count > 0, med, np.nan)
    return np.column_stack([med, count, dist])


def comps_median_baseline(log_price: np.ndarray, g: CompsGraph, fallback: float) -> np.ndarray:
    """Baseline 1: the median price of nearby earlier sales, with no adjustment at all."""
    med = comps_features(log_price, g)[:, 0]
    return np.where(np.isnan(med), fallback, med)


def fit_lightgbm(x_train: np.ndarray, y_train: np.ndarray, seed: int = 0):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.9, random_state=seed, verbose=-1)
    return model.fit(x_train, y_train)


# ---------- uncertainty ----------

def conformal_halfwidth(val_log_actual: np.ndarray, val_log_pred: np.ndarray, coverage: float = 0.9) -> float:
    """Split conformal prediction: from validation errors, the width that covers `coverage`
    of cases. It assumes only that future errors look like validation errors, not that
    they are normally distributed. Check the test coverage to see if that held."""
    resid = np.sort(np.abs(val_log_actual - val_log_pred))
    n = len(resid)
    rank = min(n - 1, int(np.ceil((n + 1) * coverage)) - 1)
    return float(resid[rank])
