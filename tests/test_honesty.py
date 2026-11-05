"""Tests for the things that would make the results dishonest if they broke.

Run from the project root:  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comps.data import REQUIRED_COLUMNS, load_sales, simulate_sales, time_split  # noqa: E402
from comps.evaluate import comps_median_baseline, conformal_halfwidth, summarise  # noqa: E402
from comps.features import Scaler, raw_features  # noqa: E402
from comps.graph import build_comps_graph  # noqa: E402
from comps.model import CompsGNN  # noqa: E402
from comps.train import to_tensors  # noqa: E402


@pytest.fixture(scope="module")
def small():
    df = simulate_sales(n=1500, seed=1)
    g = build_comps_graph(df, k=8, radius_km=2.0, lookback_days=365)
    return df, g


def test_comparables_always_sold_strictly_earlier(small):
    df, g = small
    t = df["date"].to_numpy()
    i, slot = np.nonzero(g.mask)
    assert len(i) > 0
    assert np.all(t[g.nbr[i, slot]] < t[i]), "a sale can see a later or same-day sale: future leakage"


def test_comparables_respect_radius_and_lookback(small):
    df, g = small
    assert g.dist_km[g.mask].max() <= 2.0 + 1e-6
    assert g.days_ago[g.mask].max() <= 365
    assert g.days_ago[g.mask].min() >= 1


def test_padding_is_consistent(small):
    _, g = small
    assert np.array_equal(g.mask, g.nbr >= 0)
    assert np.all(g.dist_km[~g.mask] == 0)


def test_time_split_never_trains_on_the_future():
    df = simulate_sales(n=1000, seed=2)
    split = time_split(df)
    dates = df["date"]
    assert dates[split == "train"].max() <= dates[split == "val"].min()
    assert dates[split == "val"].max() <= dates[split == "test"].min()


def _model_inputs(df, g):
    raw = raw_features(df)
    x = Scaler().fit(raw)(raw)
    y = np.log(df["price"].to_numpy()).astype(np.float32)
    return to_tensors(x, (y - y.mean()) / y.std(), g)


def test_a_sales_own_price_never_affects_its_own_estimate(small):
    """The core leakage test: change sale i's price; sale i's estimate must not move,
    but a later sale that uses i as a comparable should."""
    df, g = small
    torch.manual_seed(0)
    model = CompsGNN(n_features=len(raw_features(df)[0])).eval()
    t = _model_inputs(df, g)

    with torch.no_grad():
        before, _ = model(**t)
        i = int(np.flatnonzero(g.mask.any(1))[-1])
        t2 = dict(t, y=t["y"].clone())
        t2["y"][i] += 5.0
        after, _ = model(**t2)

    assert torch.isclose(before[i], after[i]), "sale i's own price leaked into its estimate"
    users = np.flatnonzero((g.nbr == i).any(1))
    if len(users):
        assert not torch.allclose(before[users], after[users]), "comparables should pass information on"


def test_attention_weights_are_a_proper_average(small):
    df, g = small
    model = CompsGNN(n_features=len(raw_features(df)[0])).eval()
    with torch.no_grad():
        _, w = model(**_model_inputs(df, g))
    assert torch.allclose(w.sum(1), torch.ones(len(df)), atol=1e-5)
    padded = ~torch.tensor(g.mask)
    assert torch.all(w[:, 1:][padded] == 0), "a missing comparable got weight"


def test_a_sale_with_no_comparables_uses_only_its_own_guess(small):
    df, g = small
    model = CompsGNN(n_features=len(raw_features(df)[0])).eval()
    with torch.no_grad():
        _, w = model(**_model_inputs(df, g))
    lonely = np.flatnonzero(~g.mask.any(1))
    assert len(lonely) > 0  # the earliest sales have nothing before them
    assert torch.allclose(w[lonely, 0], torch.ones(len(lonely)))


def test_comps_median_baseline_uses_neighbours():
    df = simulate_sales(n=800, seed=3)
    g = build_comps_graph(df, k=5)
    lp = np.log(df["price"].to_numpy())
    pred = comps_median_baseline(lp, g, fallback=-1.0)
    has = g.mask.any(1)
    assert np.all(pred[~has] == -1.0)
    i = int(np.flatnonzero(has)[0])
    assert pred[i] == pytest.approx(np.median(lp[g.nbr[i][g.mask[i]]]))


def test_conformal_interval_covers_about_the_target_share():
    rng = np.random.default_rng(0)
    actual = rng.normal(size=20000)
    pred = actual + rng.normal(scale=0.3, size=20000)
    half = conformal_halfwidth(actual[:10000], pred[:10000], coverage=0.9)
    covered = np.mean(np.abs(actual[10000:] - pred[10000:]) <= half)
    assert 0.88 <= covered <= 0.92


def test_metrics():
    r = summarise(np.array([100.0, 100.0, 100.0, 100.0]), np.array([105.0, 95.0, 115.0, 130.0]))
    assert r["median_abs_pct_error"] == 10.0
    assert r["within_10pct"] == 50.0 and r["within_20pct"] == 75.0


def test_load_sales_rejects_missing_columns(tmp_path):
    path = tmp_path / "sales.csv"
    pd.DataFrame({"sale_id": [1], "date": ["2024-01-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_sales(str(path))


def test_load_sales_accepts_the_documented_format(tmp_path):
    path = tmp_path / "sales.csv"
    simulate_sales(n=50, seed=4)[REQUIRED_COLUMNS].to_csv(path, index=False)
    assert len(load_sales(str(path))) == 50
