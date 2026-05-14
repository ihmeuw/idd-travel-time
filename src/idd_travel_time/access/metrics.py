"""Surface evaluation metrics.

Compares a predicted TravelTimeSurface (or ensemble) against ground-truth
travel-time observations.  Ground truth can come from:
- GPS-tracked travel time surveys
- Validated routing APIs (Google Maps, OSRM with speed adjustments)
- Previously published reference surfaces

Two evaluation modes
--------------------
``score_at_points(surface, ground_truth_gdf)``
    Point-based: match predicted surface pixels to GT point observations.
    Returns MAE, RMSE, correlation, bias, coverage_error.
    Used for in-sample and OOS validation.

``compare_surfaces(surface_a, surface_b)``
    Pixel-based: compare two surfaces over their common extent.
    Returns mean difference, RMSE, correlation, pixel coverage agreement.
    Used for inter-backend comparison.

``benchmark_table(surfaces_dict, ground_truth_gdf)``
    Run score_at_points for each named surface; return a tidy DataFrame.
    The main entry point for multi-backend comparison.

Ground truth format
-------------------
A GeoDataFrame with:
- ``geometry``: Point geometries (WGS-84)
- ``travel_time_min``: observed travel time in minutes (float)
- ``split`` (optional): ``"train"`` / ``"test"`` for IS/OOS split

Any extra columns are preserved and ignored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd

    from idd_travel_time.access.surface import TravelTimeSurface


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def score_at_points(
    surface: TravelTimeSurface,
    ground_truth: gpd.GeoDataFrame,
    *,
    obs_col: str = "travel_time_min",
    split: str | None = None,
) -> dict[str, float]:
    """Score a surface against point observations.

    Parameters
    ----------
    surface:
        Predicted TravelTimeSurface.
    ground_truth:
        GeoDataFrame with a Point geometry column and an observed travel
        time column.
    obs_col:
        Column name for observed travel times (default ``"travel_time_min"``).
    split:
        If ``"train"`` or ``"test"``, filter ground_truth by its ``"split"``
        column before scoring.

    Returns
    -------
    dict with keys: n, mae, rmse, bias, r2, pearson_r, spearman_r,
    coverage_error_30, coverage_error_60 (fraction of points where the
    predicted threshold differs from observed threshold at 30 and 60 min).
    """
    import geopandas as gpd  # noqa: PLC0415
    from scipy.stats import pearsonr, spearmanr  # noqa: PLC0415

    gt = ground_truth.copy()
    if split is not None and "split" in gt.columns:
        gt = gt[gt["split"] == split]

    if gt.empty:
        return _empty_scores()

    # Extract predicted travel time at each observation point
    predicted = _sample_surface_at_points(surface, gt)
    valid = ~np.isnan(predicted)
    if valid.sum() < 2:
        return _empty_scores()

    obs = gt[obs_col].values[valid].astype(np.float64)
    pred = predicted[valid].astype(np.float64)
    residuals = pred - obs

    n = int(valid.sum())
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bias = float(np.mean(residuals))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    pearson_r, _ = pearsonr(obs, pred)
    spearman_r, _ = spearmanr(obs, pred)

    cov_30 = _coverage_error(obs, pred, threshold=30.0)
    cov_60 = _coverage_error(obs, pred, threshold=60.0)

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "coverage_error_30min": cov_30,
        "coverage_error_60min": cov_60,
    }


def compare_surfaces(
    surface_a: TravelTimeSurface,
    surface_b: TravelTimeSurface,
) -> dict[str, float]:
    """Pixel-wise comparison between two surfaces over their common extent.

    Resamples both to the finer resolution and clips to the intersection.

    Returns
    -------
    dict with: n_pixels, mean_diff (a - b), abs_mean_diff, rmsd,
    pearson_r, agreement_30min, agreement_60min.
    """
    from scipy.stats import pearsonr  # noqa: PLC0415

    res = min(surface_a.resolution_deg, surface_b.resolution_deg)
    a = surface_a.resample(res)
    b = surface_b.resample(res)

    # Clip to intersection
    from idd_travel_time.access.surface import BoundingBox  # noqa: PLC0415
    common = BoundingBox(
        west=max(a.bbox.west, b.bbox.west),
        south=max(a.bbox.south, b.bbox.south),
        east=min(a.bbox.east, b.bbox.east),
        north=min(a.bbox.north, b.bbox.north),
    )
    a = a.clip(common)
    b = b.clip(common)

    # Align shapes (minor off-by-one can occur at edges)
    min_rows = min(a.nrows, b.nrows)
    min_cols = min(a.ncols, b.ncols)
    da = a.data[:min_rows, :min_cols]
    db = b.data[:min_rows, :min_cols]

    valid = ~(np.isnan(da) | np.isnan(db))
    if valid.sum() < 2:
        return {"n_pixels": 0, "mean_diff": float("nan"), "abs_mean_diff": float("nan"),
                "rmsd": float("nan"), "pearson_r": float("nan"),
                "agreement_30min": float("nan"), "agreement_60min": float("nan")}

    fa = da[valid].astype(np.float64)
    fb = db[valid].astype(np.float64)
    diff = fa - fb

    pearson_r, _ = pearsonr(fa, fb)

    agree_30 = float(np.mean((fa <= 30.0) == (fb <= 30.0)))
    agree_60 = float(np.mean((fa <= 60.0) == (fb <= 60.0)))

    return {
        "n_pixels": int(valid.sum()),
        "mean_diff": float(np.mean(diff)),
        "abs_mean_diff": float(np.mean(np.abs(diff))),
        "rmsd": float(np.sqrt(np.mean(diff ** 2))),
        "pearson_r": float(pearson_r),
        "agreement_30min": agree_30,
        "agreement_60min": agree_60,
    }


def benchmark_table(
    surfaces: dict[str, TravelTimeSurface],
    ground_truth: gpd.GeoDataFrame,
    *,
    obs_col: str = "travel_time_min",
    splits: list[str] | None = None,
) -> pd.DataFrame:
    """Score multiple named surfaces against ground truth.

    Parameters
    ----------
    surfaces:
        Dict of ``{name: TravelTimeSurface}``.
    ground_truth:
        GeoDataFrame with observed travel times.
    obs_col:
        Column for observed values.
    splits:
        List of split names to evaluate (e.g. ``["train", "test"]``).
        If ``None``, evaluates over all points regardless of split.

    Returns
    -------
    pd.DataFrame with one row per (backend, split) combination.
    """
    rows = []
    eval_splits: list[str | None] = splits if splits else [None]

    for name, surface in surfaces.items():
        for split in eval_splits:
            scores = score_at_points(
                surface, ground_truth, obs_col=obs_col, split=split
            )
            rows.append({
                "backend": name,
                "split": split or "all",
                **scores,
            })

    return pd.DataFrame(rows).set_index(["backend", "split"])


def residual_geodataframe(
    surface: TravelTimeSurface,
    ground_truth: gpd.GeoDataFrame,
    *,
    obs_col: str = "travel_time_min",
) -> gpd.GeoDataFrame:
    """Return ground-truth GDF augmented with predicted values and residuals.

    Useful for mapping where a backend over- or under-predicts.
    """
    gt = ground_truth.copy()
    predicted = _sample_surface_at_points(surface, gt)
    gt["predicted_min"] = predicted
    gt["residual_min"] = predicted - gt[obs_col].values
    gt["abs_residual_min"] = np.abs(gt["residual_min"])
    gt["backend"] = surface.backend
    return gt


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sample_surface_at_points(
    surface: TravelTimeSurface,
    gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """Sample surface travel times at GeoDataFrame point locations.

    Returns a float array of length len(gdf). NaN where the point falls
    outside the surface extent or lands on a NaN pixel.
    """
    lons = np.array([geom.x for geom in gdf.geometry], dtype=np.float64)
    lats = np.array([geom.y for geom in gdf.geometry], dtype=np.float64)

    # Convert (lat, lon) → (row, col)
    col_f = (lons - surface.bbox.west) / surface.resolution_deg
    row_f = (surface.bbox.north - lats) / surface.resolution_deg

    col_i = np.floor(col_f).astype(int)
    row_i = np.floor(row_f).astype(int)

    in_bounds = (
        (row_i >= 0) & (row_i < surface.nrows) &
        (col_i >= 0) & (col_i < surface.ncols)
    )

    result = np.full(len(gdf), np.nan, dtype=np.float32)
    valid_mask = in_bounds
    result[valid_mask] = surface.data[row_i[valid_mask], col_i[valid_mask]]
    return result


def _coverage_error(
    obs: np.ndarray,
    pred: np.ndarray,
    threshold: float,
) -> float:
    """Fraction of points where the coverage classification differs.

    i.e. predicted says "within threshold" but observed says "outside", or vice versa.
    """
    obs_within = obs <= threshold
    pred_within = pred <= threshold
    return float(np.mean(obs_within != pred_within))


def _empty_scores() -> dict:
    return {
        "n": 0,
        "mae": float("nan"),
        "rmse": float("nan"),
        "bias": float("nan"),
        "r2": float("nan"),
        "pearson_r": float("nan"),
        "spearman_r": float("nan"),
        "coverage_error_30min": float("nan"),
        "coverage_error_60min": float("nan"),
    }
