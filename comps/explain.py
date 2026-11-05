"""Turn predictions into things a person can see: accuracy colours and comparable lines.

Kept separate from the Streamlit page so tests can check the logic without a browser.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Accuracy bands: (upper limit of absolute % error, name, RGBA colour).
BANDS = [
    (5.0, "within 5%", [46, 125, 50, 200]),      # green
    (15.0, "5–15% off", [245, 166, 35, 200]),    # amber
    (float("inf"), "over 15% off", [209, 73, 91, 210]),  # red
]
BAND_NAMES = [name for _, name, _ in BANDS]


def pct_error(actual, estimate) -> np.ndarray:
    """Signed % error: positive means the model guessed too high."""
    actual = np.asarray(actual, dtype=float)
    return (np.asarray(estimate, dtype=float) - actual) / actual * 100


def error_band(actual, estimate) -> np.ndarray:
    """The band name for each home, from the size of its error."""
    err = np.abs(pct_error(actual, estimate))
    names = np.empty(len(err), dtype=object)
    lower = -np.inf
    for limit, name, _ in BANDS:
        names[(err > lower) & (err <= limit)] = name
        lower = limit
    return names


def band_colour(name: str) -> list[int]:
    for _, band, colour in BANDS:
        if band == name:
            return colour
    raise KeyError(name)


def homes_for_map(sales: pd.DataFrame, estimate_col: str) -> pd.DataFrame:
    """One row per home with everything the map and its hover box need."""
    err = pct_error(sales["price"], sales[estimate_col])
    out = pd.DataFrame({
        "sale_id": sales["sale_id"].to_numpy(),
        "lat": sales["lat"].to_numpy(),
        "lon": sales["lon"].to_numpy(),
        "band": error_band(sales["price"], sales[estimate_col]),
        "pct_error": err,
        "price_txt": sales["price"].map("${:,.0f}".format).to_numpy(),
        "estimate_txt": sales[estimate_col].map("${:,.0f}".format).to_numpy(),
        "error_txt": [f"{e:+.1f}%" for e in err],
        "home_txt": (sales["beds"].astype(str) + " bed " + sales["type"]).to_numpy(),
    })
    out["colour"] = [band_colour(b) for b in out["band"]]
    return out


def comparable_lines(home: pd.Series, comps: pd.DataFrame, max_px: float = 12.0) -> pd.DataFrame:
    """A line from the home to each comparable; thicker means the model trusted it more."""
    if comps.empty:
        return pd.DataFrame(columns=["from", "to", "width", "weight"])
    return pd.DataFrame({
        "from": [[float(home["lon"]), float(home["lat"])]] * len(comps),
        "to": [[float(lon), float(lat)] for lon, lat in zip(comps["lon"], comps["lat"])],
        "width": np.maximum(1.0, comps["weight"].to_numpy() * max_px * 3),
        "weight": comps["weight"].to_numpy(),
    })
