"""Tests for what the map shows: the colours must match the errors, and the lines the comparables."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comps.explain import (  # noqa: E402
    BAND_NAMES, band_colour, comparable_lines, error_band, homes_for_map, pct_error,
)


def test_pct_error_sign_means_too_high_or_too_low():
    assert pct_error([100.0], [110.0])[0] == pytest.approx(10.0)
    assert pct_error([100.0], [90.0])[0] == pytest.approx(-10.0)


def test_bands_follow_the_size_of_the_error_both_ways():
    actual = np.full(6, 100.0)
    estimate = np.array([103.0, 95.0, 110.0, 85.0, 120.0, 70.0])
    assert list(error_band(actual, estimate)) == [
        "within 5%", "within 5%", "5–15% off", "5–15% off", "over 15% off", "over 15% off"]


def test_band_edges_belong_to_the_better_band():
    assert error_band([100.0], [105.0])[0] == "within 5%"
    assert error_band([100.0], [115.0])[0] == "5–15% off"


def test_every_band_has_a_distinct_colour():
    colours = [tuple(band_colour(b)) for b in BAND_NAMES]
    assert len(set(colours)) == len(BAND_NAMES)
    with pytest.raises(KeyError):
        band_colour("not a band")


def test_homes_for_map_matches_each_home_to_its_own_colour():
    sales = pd.DataFrame({"sale_id": [1, 2], "lat": [-37.8, -37.9], "lon": [145.0, 145.1],
                          "price": [1_000_000, 500_000], "est": [1_020_000, 600_000],
                          "beds": [3, 2], "type": ["house", "unit"]})
    out = homes_for_map(sales, "est")
    assert list(out["band"]) == ["within 5%", "over 15% off"]
    assert out["colour"].iloc[1] == band_colour("over 15% off")
    assert list(out["error_txt"]) == ["+2.0%", "+20.0%"]
    assert list(out["estimate_txt"]) == ["$1,020,000", "$600,000"]


def test_lines_go_from_the_home_to_each_comparable_and_trust_sets_thickness():
    home = pd.Series({"lat": -37.80, "lon": 145.00})
    comps = pd.DataFrame({"lat": [-37.81, -37.82], "lon": [145.01, 145.02], "weight": [0.6, 0.05]})
    lines = comparable_lines(home, comps)
    assert lines["from"].tolist() == [[145.00, -37.80]] * 2
    assert lines["to"].tolist() == [[145.01, -37.81], [145.02, -37.82]]
    assert lines["width"].iloc[0] > lines["width"].iloc[1] >= 1.0


def test_no_comparables_means_no_lines():
    assert comparable_lines(pd.Series({"lat": 0.0, "lon": 0.0}), pd.DataFrame()).empty
