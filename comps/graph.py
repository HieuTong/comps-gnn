"""Build the comparables graph: each sale points to nearby sales that happened BEFORE it.

This is the most important file for honesty. If a sale could see a later sale, the model
would be using the future, and its test accuracy would be a lie. So an edge j -> i exists
only when j sold strictly earlier than i, within a distance and a time window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0


@dataclass
class CompsGraph:
    nbr: np.ndarray       # (n, k) index of each comparable sale, -1 where there are fewer than k
    mask: np.ndarray      # (n, k) True where a comparable exists
    dist_km: np.ndarray   # (n, k) distance to each comparable
    days_ago: np.ndarray  # (n, k) how many days before this sale each comparable sold


def build_comps_graph(df: pd.DataFrame, k: int = 10, radius_km: float = 2.0,
                      lookback_days: int = 365, search: int = 300) -> CompsGraph:
    """For each sale, its k nearest earlier sales within radius_km and lookback_days.

    `search` is how many nearest sales to look through before filtering by date. Most of
    the nearest sales are later or too old, so it has to be much larger than k.
    """
    coords = np.radians(df[["lat", "lon"]].to_numpy())
    t = df["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    tree = BallTree(coords, metric="haversine")
    m = min(search, len(df))
    dist, idx = tree.query(coords, k=m)  # nearest first; includes the sale itself
    dist = dist * EARTH_KM

    n = len(df)
    nbr = np.full((n, k), -1, dtype=np.int64)
    out_d = np.zeros((n, k), dtype=np.float32)
    out_t = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        gap = t[i] - t[idx[i]]
        ok = (gap > 0) & (gap <= lookback_days) & (dist[i] <= radius_km)  # strictly earlier
        chosen = np.flatnonzero(ok)[:k]
        nbr[i, : len(chosen)] = idx[i, chosen]
        out_d[i, : len(chosen)] = dist[i, chosen]
        out_t[i, : len(chosen)] = gap[chosen]
    return CompsGraph(nbr=nbr, mask=nbr >= 0, dist_km=out_d, days_ago=out_t)
