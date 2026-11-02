"""Slope and altitude multipliers on off-road walking speed (Weiss 2018 eq. 1-3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from idd_travel_time.friction.params import SlopeRule


def slope_degrees(
    elevation_m: NDArray[np.floating],
    dx_m: NDArray[np.floating] | float,
    dy_m: NDArray[np.floating] | float,
) -> NDArray[np.float32]:
    """Slope angle in degrees from an elevation grid with per-row ground spacings.

    *dx_m* and *dy_m* are the ground lengths of one cell east-west and
    north-south, either scalars or one value per row (the equal-area frame's
    cells change shape with latitude). NaN elevation gives NaN slope.
    """
    dx = (
        np.asarray(dx_m, dtype=np.float64).reshape(-1, 1)
        if np.ndim(dx_m)
        else float(dx_m)
    )
    dy = (
        np.asarray(dy_m, dtype=np.float64).reshape(-1, 1)
        if np.ndim(dy_m)
        else float(dy_m)
    )
    z = np.asarray(elevation_m, dtype=np.float64)
    gy, gx = np.gradient(z)  # per-cell differences along rows (y) and columns (x)
    rise = np.hypot(gx / dx, gy / dy)
    deg = np.degrees(np.arctan(rise))
    deg[np.isnan(z)] = np.nan  # a missing cell has no slope, not a slope of zero
    return np.asarray(deg, dtype=np.float32)


def tobler_factor(
    slope_deg: NDArray[np.floating], rule: SlopeRule
) -> NDArray[np.float32]:
    """Tobler's walking speed, capped and divided by the cap: a multiplier in (0, 1].

    *slope_deg* is an unsigned slope magnitude (Weiss 2018 treats slope
    isotropically: steep terrain always slows). The absolute value inside the
    formula is Tobler's own; with unsigned input it never sees a downhill.
    """
    slope = np.asarray(slope_deg, dtype=np.float64)
    if rule.symmetric:
        slope = np.abs(slope)
    term = np.abs(np.tan(np.radians(slope)) + rule.offset)
    speed = rule.base_kph * np.exp(rule.coefficient * term)
    return (np.minimum(speed, rule.cap_kph) / rule.cap_kph).astype(np.float32)


def elevation_factor(
    elevation_m: NDArray[np.floating], rule: SlopeRule
) -> NDArray[np.float32]:
    """Altitude multiplier: 1.016 * exp(-0.0001072 * elevation) in Weiss 2018."""
    z = np.asarray(elevation_m, dtype=np.float64)
    return (rule.elevation_scale * np.exp(rule.elevation_rate * z)).astype(np.float32)
