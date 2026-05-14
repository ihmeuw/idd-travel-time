"""EnsembleSurfaceRouter — combine multiple backends into one surface.

Strategies
----------
``"mean"``      : pixel-wise (optionally weighted) mean of all backends.
``"median"``    : pixel-wise median (equal weight, robust to outliers).
``"min"``       : pixel-wise minimum (fastest estimate — optimistic).
``"max"``       : pixel-wise maximum (slowest estimate — conservative).
``"stack"``     : return all surfaces as a dict instead of one combined surface.

NaN handling
------------
By default (``nan_policy="propagate"``) a NaN in ANY backend makes the
ensemble pixel NaN.  With ``nan_policy="ignore"`` the ensemble is computed
over the non-NaN backends at each pixel, allowing partial coverage.

Weights
-------
For ``"mean"``, pass ``weights=[w0, w1, …]`` (one per backend).  Weights
are normalised to sum to 1 before aggregation.  ``None`` = equal weights.

Uncertainty
-----------
``compute_uncertainty()`` returns a surface of pixel-wise standard deviation
across backends — a simple measure of inter-model spread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from idd_travel_time.access.surface import TravelTimeSurface
from idd_travel_time.access.surface_router import SurfaceRouter

if TYPE_CHECKING:
    from idd_travel_time.access.surface import BoundingBox
    from idd_travel_time.constants import TransportMode
    from idd_travel_time.models import TransitProfile

_VALID_STRATEGIES = ("mean", "median", "min", "max", "stack")
_VALID_NAN_POLICIES = ("propagate", "ignore")


class EnsembleSurfaceRouter(SurfaceRouter):
    """Combine multiple SurfaceRouter backends into an ensemble.

    Parameters
    ----------
    routers:
        List of SurfaceRouter instances to ensemble.
    strategy:
        Aggregation strategy: ``"mean"``, ``"median"``, ``"min"``,
        ``"max"``, or ``"stack"``.
    weights:
        Per-router weights for ``"mean"`` strategy.  Normalised internally.
        ``None`` = equal weights.
    nan_policy:
        ``"propagate"`` (default): NaN in any backend → NaN in output.
        ``"ignore"``: skip NaN backends at each pixel.
    """

    def __init__(
        self,
        routers: list[SurfaceRouter],
        strategy: str = "mean",
        weights: list[float] | None = None,
        nan_policy: str = "propagate",
    ) -> None:
        if len(routers) < 2:
            msg = "EnsembleSurfaceRouter requires at least 2 routers."
            raise ValueError(msg)
        if strategy not in _VALID_STRATEGIES:
            msg = f"strategy must be one of {_VALID_STRATEGIES}, got {strategy!r}"
            raise ValueError(msg)
        if nan_policy not in _VALID_NAN_POLICIES:
            msg = f"nan_policy must be one of {_VALID_NAN_POLICIES}, got {nan_policy!r}"
            raise ValueError(msg)
        if weights is not None and len(weights) != len(routers):
            msg = f"weights length ({len(weights)}) must match routers length ({len(routers)})"
            raise ValueError(msg)

        self._routers = routers
        self._strategy = strategy
        self._nan_policy = nan_policy

        if weights is not None:
            w = np.array(weights, dtype=np.float64)
            if (w < 0).any():
                msg = "All weights must be >= 0"
                raise ValueError(msg)
            self._weights: np.ndarray | None = w / w.sum()
        else:
            self._weights = None

    @property
    def name(self) -> str:
        names = "+".join(r.name for r in self._routers)
        return f"ensemble[{self._strategy}]({names})"

    @property
    def routers(self) -> list[SurfaceRouter]:
        return list(self._routers)

    def compute_surface(
        self,
        destination: tuple[float, float],
        bbox: BoundingBox,
        resolution_deg: float,
        mode: TransportMode,
        *,
        transit_profile: TransitProfile | None = None,
    ) -> TravelTimeSurface:
        """Compute and aggregate surfaces from all backends."""
        surfaces = self._compute_all(
            destination, bbox, resolution_deg, mode,
            transit_profile=transit_profile,
        )

        if self._strategy == "stack":
            # For stack, return the mean surface (caller should use
            # compute_all_surfaces() to get the dict)
            return self._aggregate(surfaces, destination, bbox, resolution_deg)

        return self._aggregate(surfaces, destination, bbox, resolution_deg)

    def compute_all_surfaces(
        self,
        destination: tuple[float, float],
        bbox: BoundingBox,
        resolution_deg: float,
        mode: TransportMode,
        *,
        transit_profile: TransitProfile | None = None,
    ) -> dict[str, TravelTimeSurface]:
        """Compute and return one surface per backend (no aggregation).

        Useful for visualising inter-backend differences.
        """
        return self._compute_all(
            destination, bbox, resolution_deg, mode,
            transit_profile=transit_profile,
        )

    def compute_uncertainty(
        self,
        destination: tuple[float, float],
        bbox: BoundingBox,
        resolution_deg: float,
        mode: TransportMode,
        *,
        transit_profile: TransitProfile | None = None,
    ) -> TravelTimeSurface:
        """Return pixel-wise standard deviation across backends.

        This is a simple measure of inter-model spread / uncertainty.
        """
        surfaces = self._compute_all(
            destination, bbox, resolution_deg, mode,
            transit_profile=transit_profile,
        )
        aligned = _align_to_common_grid(list(surfaces.values()), resolution_deg)
        stack = np.stack([s.data for s in aligned], axis=0)

        if self._nan_policy == "propagate":
            std_data = np.nanstd(stack, axis=0, ddof=1).astype(np.float32)
            any_nan = np.any(np.isnan(stack), axis=0)
            std_data[any_nan] = np.nan
        else:
            with np.errstate(invalid="ignore"):
                std_data = np.nanstd(stack, axis=0, ddof=1).astype(np.float32)
            all_nan = np.all(np.isnan(stack), axis=0)
            std_data[all_nan] = np.nan

        ref = aligned[0]
        return TravelTimeSurface.from_arrays(
            data=std_data,
            west=ref.bbox.west,
            north=ref.bbox.north,
            resolution_deg=resolution_deg,
            backend=f"uncertainty({self.name})",
            destination=destination,
            metadata={"backends": [r.name for r in self._routers]},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_all(
        self,
        destination: tuple[float, float],
        bbox: BoundingBox,
        resolution_deg: float,
        mode: TransportMode,
        *,
        transit_profile: TransitProfile | None = None,
    ) -> dict[str, TravelTimeSurface]:
        surfaces: dict[str, TravelTimeSurface] = {}
        for router in self._routers:
            surfaces[router.name] = router.compute_surface(
                destination, bbox, resolution_deg, mode,
                transit_profile=transit_profile,
            )
        return surfaces

    def _aggregate(
        self,
        surfaces: dict[str, TravelTimeSurface],
        destination: tuple[float, float],
        bbox: BoundingBox,
        resolution_deg: float,
    ) -> TravelTimeSurface:
        aligned = _align_to_common_grid(list(surfaces.values()), resolution_deg)
        stack = np.stack([s.data for s in aligned], axis=0)  # (n_backends, nrows, ncols)

        if self._strategy == "mean":
            if self._weights is not None:
                w = self._weights[:, np.newaxis, np.newaxis]
                if self._nan_policy == "propagate":
                    result = np.sum(stack * w, axis=0)
                    result[np.any(np.isnan(stack), axis=0)] = np.nan
                else:
                    weight_sum = np.where(~np.isnan(stack), w, 0.0).sum(axis=0)
                    result = np.where(~np.isnan(stack), stack * w, 0.0).sum(axis=0)
                    result = np.where(weight_sum > 0, result / weight_sum, np.nan)
            else:
                fn = np.nanmean if self._nan_policy == "ignore" else _strict_mean
                result = fn(stack, axis=0)

        elif self._strategy == "median":
            fn = np.nanmedian if self._nan_policy == "ignore" else _strict_median
            result = fn(stack, axis=0)

        elif self._strategy == "min":
            fn = np.nanmin if self._nan_policy == "ignore" else _strict_min
            result = fn(stack, axis=0)

        elif self._strategy == "max":
            fn = np.nanmax if self._nan_policy == "ignore" else _strict_max
            result = fn(stack, axis=0)

        else:  # stack → default to mean
            result = np.nanmean(stack, axis=0)

        result = result.astype(np.float32)
        if self._nan_policy == "propagate":
            result[np.any(np.isnan(stack), axis=0)] = np.nan

        ref = aligned[0]
        return TravelTimeSurface.from_arrays(
            data=result,
            west=ref.bbox.west,
            north=ref.bbox.north,
            resolution_deg=resolution_deg,
            backend=self.name,
            destination=destination,
            metadata={"strategy": self._strategy,
                       "backends": [r.name for r in self._routers]},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_to_common_grid(
    surfaces: list[TravelTimeSurface],
    resolution_deg: float,
) -> list[TravelTimeSurface]:
    """Resample all surfaces to the same resolution and clip to common bbox."""
    resampled = [s.resample(resolution_deg) if abs(s.resolution_deg - resolution_deg) / resolution_deg > 0.01
                 else s for s in surfaces]
    # Find intersection bbox
    west = max(s.bbox.west for s in resampled)
    south = max(s.bbox.south for s in resampled)
    east = min(s.bbox.east for s in resampled)
    north = min(s.bbox.north for s in resampled)

    from idd_travel_time.access.surface import BoundingBox  # noqa: PLC0415
    common = BoundingBox(west=west, south=south, east=east, north=north)
    return [s.clip(common) for s in resampled]


def _strict_mean(arr: np.ndarray, axis: int) -> np.ndarray:
    result = np.mean(arr, axis=axis)
    result[np.any(np.isnan(arr), axis=axis)] = np.nan
    return result


def _strict_median(arr: np.ndarray, axis: int) -> np.ndarray:
    result = np.median(arr, axis=axis)
    result[np.any(np.isnan(arr), axis=axis)] = np.nan
    return result


def _strict_min(arr: np.ndarray, axis: int) -> np.ndarray:
    result = np.min(arr, axis=axis)
    result[np.any(np.isnan(arr), axis=axis)] = np.nan
    return result


def _strict_max(arr: np.ndarray, axis: int) -> np.ndarray:
    result = np.max(arr, axis=axis)
    result[np.any(np.isnan(arr), axis=axis)] = np.nan
    return result
