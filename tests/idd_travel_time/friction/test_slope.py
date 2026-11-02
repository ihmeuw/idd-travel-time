"""Slope and altitude multipliers reproduce Weiss 2018 eq. 1-3 at known points."""

from __future__ import annotations

import math

import numpy as np
import pytest

from idd_travel_time.friction.params import SlopeRule, load_params
from idd_travel_time.friction.slope import (
    elevation_factor,
    slope_degrees,
    tobler_factor,
)


@pytest.fixture
def rule() -> SlopeRule:
    return load_params().slope


def test_tobler_flat_is_one_and_steeper_is_less(rule: SlopeRule) -> None:
    f = tobler_factor(
        np.array([0.0, 5.71, 16.7, 45.0]), rule
    )  # 0 %, 10 %, 30 %, 100 % grade
    assert f[0] == pytest.approx(1.0)  # 6*exp(-0.175)=5.04 capped to 5 -> 1.0
    assert f[1] == pytest.approx(6 * math.exp(-3.5 * (0.1 + 0.05)) / 5, abs=1e-3)
    assert np.all(np.diff(f) < 0)


def test_tobler_treats_slope_as_unsigned_magnitude(rule: SlopeRule) -> None:
    up = tobler_factor(np.array([10.0]), rule)
    down = tobler_factor(np.array([-10.0]), rule)
    assert up[0] == pytest.approx(down[0])
    expected = 6 * math.exp(-3.5 * abs(math.tan(math.radians(10.0)) + 0.05)) / 5
    assert up[0] == pytest.approx(expected, abs=1e-4)


def test_elevation_factor_matches_equation_one(rule: SlopeRule) -> None:
    f = elevation_factor(np.array([0.0, 1000.0, 4000.0]), rule)
    assert f[0] == pytest.approx(1.016)
    assert f[1] == pytest.approx(1.016 * math.exp(-0.1072), abs=1e-4)
    assert f[2] < f[1] < f[0]


def test_slope_degrees_uses_per_axis_ground_lengths() -> None:
    # a ramp rising 1 m per column; cells 40 m wide -> slope atan(1/40)
    z = np.tile(np.arange(5, dtype=float), (4, 1))
    s = slope_degrees(z, dx_m=40.0, dy_m=40.0)
    assert s[1, 2] == pytest.approx(math.degrees(math.atan(1 / 40)), abs=1e-3)
    # same ramp, but cells only 20 m wide on the ground -> twice the slope
    s2 = slope_degrees(z, dx_m=np.full(4, 20.0), dy_m=np.full(4, 80.0))
    assert s2[1, 2] == pytest.approx(math.degrees(math.atan(1 / 20)), abs=1e-3)


def test_slope_degrees_propagates_nan() -> None:
    z = np.ones((3, 3))
    z[1, 1] = np.nan
    s = slope_degrees(z, 40.0, 40.0)
    assert np.isnan(s[1, 1])
    assert s.dtype == np.float32
