"""Sales data: a simulated market for development, and a loader for real sales.

Every model here needs the same columns, so swapping simulated data for real data only
changes where the table comes from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["sale_id", "date", "lat", "lon", "price", "beds", "baths", "cars", "land_m2", "type"]
TYPES = ["house", "townhouse", "unit"]
CBD = (-37.8136, 144.9631)  # Melbourne CBD, used only to place simulated suburbs


def km_between(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres (haversine)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def simulate_sales(n: int = 15000, seed: int = 7) -> pd.DataFrame:
    """A simulated city. SIMULATED DATA: for building and testing the pipeline only.

    Prices combine things a model can see (rooms, land, type, suburb, date) with a hidden
    street-level premium it can't see directly. The only way to learn that premium is from
    nearby sales, which is exactly the situation where comparables help. Because I wrote
    the simulator, results on it show the pipeline works, not that GNNs beat other models
    on real property: that needs real data.
    """
    rng = np.random.default_rng(seed)

    # 30 suburbs; ones nearer the CBD are dearer, plus each suburb's own character.
    n_sub = 30
    sub_dist = rng.uniform(2, 30, n_sub)
    sub_angle = rng.uniform(0, 2 * np.pi, n_sub)
    sub_lat = CBD[0] + (sub_dist / 111.0) * np.sin(sub_angle)
    sub_lon = CBD[1] + (sub_dist / (111.0 * np.cos(np.radians(CBD[0])))) * np.cos(sub_angle)
    sub_level = 13.9 - 0.025 * sub_dist + rng.normal(0, 0.12, n_sub)

    sub = rng.integers(0, n_sub, n)
    lat = sub_lat[sub] + rng.normal(0, 0.012, n)
    lon = sub_lon[sub] + rng.normal(0, 0.015, n)

    # Hidden street-level premium: many small bumps, each a few hundred metres wide.
    n_bumps = 600
    bumps = rng.integers(0, n_sub, n_bumps)
    b_lat = sub_lat[bumps] + rng.normal(0, 0.012, n_bumps)
    b_lon = sub_lon[bumps] + rng.normal(0, 0.015, n_bumps)
    b_amp = rng.normal(0, 0.10, n_bumps)
    b_rad = rng.uniform(0.3, 0.8, n_bumps)
    street = np.zeros(n)
    for i in range(n_bumps):
        d = km_between(lat, lon, b_lat[i], b_lon[i])
        street += b_amp[i] * np.exp(-((d / b_rad[i]) ** 2))

    ptype = rng.choice(TYPES, n, p=[0.6, 0.15, 0.25])
    beds = np.where(ptype == "unit", rng.integers(1, 4, n), rng.integers(2, 6, n))
    baths = np.clip(beds - rng.integers(0, 3, n), 1, 4)
    cars = np.where(ptype == "unit", rng.integers(0, 2, n), rng.integers(1, 3, n))
    land = np.where(ptype == "house", rng.uniform(300, 900, n),
                    np.where(ptype == "townhouse", rng.uniform(150, 350, n), 0.0)).round()

    # Three years of a market that rises, dips, then recovers.
    days = rng.integers(0, 3 * 365, n)
    date = pd.Timestamp("2023-01-01") + pd.to_timedelta(days, unit="D")
    market = np.interp(days, [0, 365, 600, 3 * 365], [0.0, 0.06, 0.02, 0.09])

    type_effect = np.select([ptype == "house", ptype == "townhouse"], [0.0, -0.15], -0.45)
    log_price = (sub_level[sub] + street + market + type_effect
                 + 0.09 * beds + 0.05 * baths + 0.03 * cars
                 + 0.12 * np.log1p(land / 100) + rng.normal(0, 0.07, n))

    df = pd.DataFrame({
        "sale_id": np.arange(n), "date": date, "lat": lat, "lon": lon,
        "price": np.round(np.exp(log_price), -3), "beds": beds, "baths": baths,
        "cars": cars, "land_m2": land, "type": ptype,
    })
    return df.sort_values(["date", "sale_id"]).reset_index(drop=True)


def load_sales(path: str) -> pd.DataFrame:
    """Load real sales from a CSV with REQUIRED_COLUMNS, and check it before anything trains."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[(df["price"] > 10_000) & df["type"].isin(TYPES)]
    return df.sort_values(["date", "sale_id"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15) -> np.ndarray:
    """Label each sale train, val or test BY DATE: the model is always tested on the future.

    Expects df sorted by date (simulate_sales and load_sales both sort).
    """
    n = len(df)
    split = np.empty(n, dtype=object)
    a, b = int(n * (1 - val_frac - test_frac)), int(n * (1 - test_frac))
    split[:a], split[a:b], split[b:] = "train", "val", "test"
    return split
