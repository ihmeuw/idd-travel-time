"""Structured benchmark runner for comparing SurfaceRouter backends.

Ties together metrics.benchmark_table with train/test splitting and
performance timing. Designed to be the top-level entry point for
running a systematic comparison across backends and destinations.

Usage
-----
>>> from idd_travel_time.access.benchmark import run_benchmark
>>> from idd_travel_time.access.backends.osmnx_surface import OsmnxSurfaceRouter
>>> from idd_travel_time.access.backends.friction_surface import FrictionSurfaceRouter
>>>
>>> result = run_benchmark(
...     routers={
...         "osmnx": OsmnxSurfaceRouter(),
...         "friction_5kph": FrictionSurfaceRouter(speed_kph_override=5.0),
...     },
...     destination=(-1.3006, 36.8073),
...     bbox=BoundingBox(-1.35, 36.76, -1.26, 36.86),
...     resolution_deg=0.01,
...     mode=TransportMode.WALKING,
...     ground_truth=gt_gdf,
...     test_fraction=0.3,
...     random_seed=42,
... )
>>> result.scores        # pd.DataFrame
>>> result.surfaces      # dict[str, TravelTimeSurface]
>>> result.timing        # dict[str, float]  (seconds per backend)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from idd_travel_time.access.metrics import benchmark_table, compare_surfaces
from idd_travel_time.access.surface import TravelTimeSurface

if TYPE_CHECKING:
    import geopandas as gpd

    from idd_travel_time.access.surface import BoundingBox
    from idd_travel_time.access.surface_router import SurfaceRouter
    from idd_travel_time.constants import TransportMode
    from idd_travel_time.models import TransitProfile


@dataclass
class BenchmarkResult:
    """Container for all outputs of run_benchmark.

    Attributes
    ----------
    surfaces:
        Dict of computed surfaces, one per backend.
    scores:
        DataFrame of metric scores (from metrics.benchmark_table).
    timing:
        Wall-clock seconds per backend.
    pairwise:
        Dict of pixel-wise comparison results between all backend pairs.
    ground_truth_with_preds:
        The input ground_truth GDF augmented with predicted values and
        residuals for each backend.
    config:
        Dict recording the benchmark configuration for provenance.
    """

    surfaces: dict[str, TravelTimeSurface]
    scores: pd.DataFrame
    timing: dict[str, float] = field(default_factory=dict)
    pairwise: dict[str, dict[str, Any]] = field(default_factory=dict)
    ground_truth_with_preds: gpd.GeoDataFrame | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of scores."""
        lines = ["=== Benchmark Summary ==="]
        for (backend, split), row in self.scores.iterrows():
            t = self.timing.get(backend, float("nan"))
            lines.append(
                f"  {backend:<30} [{split}]  "
                f"MAE={row.get('mae', float('nan')):.1f}min  "
                f"RMSE={row.get('rmse', float('nan')):.1f}min  "
                f"R²={row.get('r2', float('nan')):.3f}  "
                f"time={t:.1f}s"
            )
        return "\n".join(lines)


def run_benchmark(
    routers: dict[str, SurfaceRouter],
    destination: tuple[float, float],
    bbox: BoundingBox,
    resolution_deg: float,
    mode: TransportMode,
    *,
    ground_truth: gpd.GeoDataFrame | None = None,
    obs_col: str = "travel_time_min",
    test_fraction: float = 0.3,
    random_seed: int = 42,
    transit_profile: TransitProfile | None = None,
) -> BenchmarkResult:
    """Run all backends and score against ground truth.

    Parameters
    ----------
    routers:
        Dict of ``{name: SurfaceRouter}`` to benchmark.
    destination:
        ``(lat, lon)`` of the target facility.
    bbox:
        Bounding box for the output rasters.
    resolution_deg:
        Pixel size in degrees.
    mode:
        Transport mode.
    ground_truth:
        Optional GeoDataFrame with observed travel times.  If None, only
        surfaces and timing are returned (no metric scores).
    obs_col:
        Column name for observed values.
    test_fraction:
        Fraction of ground_truth to hold out for OOS evaluation.
    random_seed:
        RNG seed for train/test split.
    transit_profile:
        Required when ``mode == TRANSIT``.

    Returns
    -------
    BenchmarkResult
    """
    # --- Train/test split -----------------------------------------------
    gt = None
    if ground_truth is not None:
        gt = _add_split(ground_truth, test_fraction, random_seed)

    # --- Compute surfaces -----------------------------------------------
    surfaces: dict[str, TravelTimeSurface] = {}
    timing: dict[str, float] = {}
    errors: dict[str, str] = {}

    for name, router in routers.items():
        t0 = time.monotonic()
        try:
            surfaces[name] = router.compute_surface(
                destination,
                bbox,
                resolution_deg,
                mode,
                transit_profile=transit_profile,
            )
        except Exception as exc:  # noqa: BLE001
            errors[name] = str(exc)
        timing[name] = time.monotonic() - t0

    if errors:
        import warnings

        for name, msg in errors.items():
            warnings.warn(
                f"Backend {name!r} failed: {msg}", RuntimeWarning, stacklevel=2
            )

    # --- Score against ground truth -------------------------------------
    scores = pd.DataFrame()
    gt_with_preds = None

    if gt is not None and surfaces:
        splits = ["train", "test"] if "split" in gt.columns else None
        scores = benchmark_table(surfaces, gt, obs_col=obs_col, splits=splits)

        # Augment ground truth with all predictions
        gt_aug = gt.copy()
        for name, surface in surfaces.items():
            from idd_travel_time.access.metrics import (
                _sample_surface_at_points,
            )

            preds = _sample_surface_at_points(surface, gt_aug)
            gt_aug[f"pred_{name}_min"] = preds
            gt_aug[f"resid_{name}_min"] = preds - gt_aug[obs_col].to_numpy()
        gt_with_preds = gt_aug

    # --- Pairwise pixel comparisons ------------------------------------
    pairwise: dict[str, dict[str, Any]] = {}
    names = list(surfaces.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}_vs_{b}"
            pairwise[key] = compare_surfaces(surfaces[a], surfaces[b])

    config = {
        "destination": destination,
        "bbox": bbox.to_dict(),
        "resolution_deg": resolution_deg,
        "mode": mode.value,
        "backends": list(routers.keys()),
        "n_gt_points": len(ground_truth) if ground_truth is not None else 0,
        "test_fraction": test_fraction,
        "random_seed": random_seed,
    }

    return BenchmarkResult(
        surfaces=surfaces,
        scores=scores,
        timing=timing,
        pairwise=pairwise,
        ground_truth_with_preds=gt_with_preds,
        config=config,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_ground_truth(
    surface: TravelTimeSurface,
    n_points: int = 100,
    noise_std_min: float = 2.0,
    random_seed: int = 42,
) -> gpd.GeoDataFrame:
    """Generate synthetic ground-truth observations from a reference surface.

    Samples *n_points* random valid pixels from *surface*, adds Gaussian
    noise, and returns a GeoDataFrame.  Useful for unit tests, vignettes,
    and demonstrating the benchmark workflow without real GT data.

    Parameters
    ----------
    surface:
        Reference surface to sample from.
    n_points:
        Number of synthetic observations.
    noise_std_min:
        Standard deviation of additive Gaussian noise (minutes).
    random_seed:
        RNG seed.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    rng = np.random.default_rng(random_seed)

    valid_mask = ~np.isnan(surface.data)
    valid_rows, valid_cols = np.where(valid_mask)

    if len(valid_rows) == 0:
        msg = "Surface has no valid pixels — cannot generate synthetic GT"
        raise ValueError(msg)

    n = min(n_points, len(valid_rows))
    chosen = rng.choice(len(valid_rows), size=n, replace=False)
    rows = valid_rows[chosen]
    cols = valid_cols[chosen]

    lats = surface.bbox.north - (rows + 0.5) * surface.resolution_deg
    lons = surface.bbox.west + (cols + 0.5) * surface.resolution_deg
    true_times = surface.data[rows, cols].astype(np.float64)
    noisy_times = true_times + rng.normal(0, noise_std_min, size=n)
    noisy_times = np.clip(noisy_times, 0.0, None)

    geometries = [Point(lon, lat) for lon, lat in zip(lons, lats, strict=False)]
    return gpd.GeoDataFrame(
        {
            "travel_time_min": noisy_times.astype(np.float32),
            "true_travel_time_min": true_times.astype(np.float32),
        },
        geometry=geometries,
        crs="EPSG:4326",
    )


def _add_split(
    gdf: gpd.GeoDataFrame,
    test_fraction: float,
    random_seed: int,
) -> gpd.GeoDataFrame:
    """Add a ``split`` column (``"train"`` / ``"test"``) to the GDF."""
    rng = np.random.default_rng(random_seed)
    n = len(gdf)
    n_test = max(1, round(n * test_fraction))
    test_idx = rng.choice(n, size=n_test, replace=False)
    split = np.full(n, "train", dtype=object)
    split[test_idx] = "test"
    out = gdf.copy()
    out["split"] = split
    return out
