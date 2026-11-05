"""Turn a sales table into numbers. Scaling is fitted on training sales only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from comps.data import TYPES

FEATURE_NAMES = ["beds", "baths", "cars", "log_land"] + [f"type_{t}" for t in TYPES] + ["lat", "lon", "years"]


def raw_features(df: pd.DataFrame) -> np.ndarray:
    """One row per sale. Never includes the price: the price is what we predict."""
    cols = [df["beds"], df["baths"], df["cars"], np.log1p(df["land_m2"] / 100)]
    cols += [(df["type"] == t).astype(float) for t in TYPES]
    years = (df["date"] - pd.Timestamp("2000-01-01")).dt.days / 365.0
    return np.column_stack(cols + [df["lat"], df["lon"], years]).astype(np.float32)


class Scaler:
    """Standardise each column. Fitted on training sales only, so test data leaks nothing."""

    def fit(self, x: np.ndarray) -> "Scaler":
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-6
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)
