"""FrictionSurfaceRouter — travel-time surface via raster least-cost path.

Algorithm (Weiss / MAP / Watmough approach)
-------------------------------------------
1. Load a friction raster: each cell value = minutes per metre of travel.
   Can be a MAP global friction surface (1 km) or a custom high-resolution
   raster (100 m, 20 m, etc.) built from OSM + land cover + DEM.
2. Clip friction raster to the expanded bbox.
3. Mark the destination pixel as the source.
4. Run ``skimage.graph.MCP_Geometric.find_costs()`` from the source pixel.
   MCP_Geometric correctly weights diagonal moves by √2.
5. The output ``cumulative_costs`` array has units: minutes (since friction
   is in min/m and MCP integrates over pixel-metre distances).
6. Clip to the requested bbox and return as TravelTimeSurface.

This approach is naturally multi-source: pass multiple destination pixels
to ``find_costs()`` and it returns the minimum cost from any source — exactly
what ``nearest_facility_surface()`` needs later.

Friction raster format
----------------------
- Float32 GeoTIFF in EPSG:4326.
- Cell value = minutes required to travel 1 metre across that cell.
- Example: walking at 5 km/h = 5000 m/60 min → 0.012 min/m.
- NoData (unreachable, e.g. open water) = a very large value or NaN.
- The MAP global friction surface uses exactly this convention.

Using MAP friction surfaces
---------------------------
The Malaria Atlas Project provides global 1 km and 100 m friction rasters
downloadable via the malariaAtlas R package or directly from GEE. Supply
the clipped GeoTIFF via ``friction_path``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from idd_travel_time.access.surface import BoundingBox, TravelTimeSurface
from idd_travel_time.access.surface_router import SurfaceRouter

if TYPE_CHECKING:
    from idd_travel_time.constants import TransportMode
    from idd_travel_time.models import TransitProfile

# Cells with friction > this are treated as impassable (open water, cliffs, …)
_IMPASSABLE_THRESHOLD_MIN_PER_M: float = 1.0   # > 1 min/m = impassable


class FrictionSurfaceRouter(SurfaceRouter):
    """Travel-time surface backend using raster least-cost path (MCP_Geometric).

    Parameters
    ----------
    friction_path:
        Path to a float32 GeoTIFF friction raster (minutes per metre).
        Must cover the full bbox requested in ``compute_surface()``.
    speed_kph_override:
        If provided, ignore the friction raster and use a uniform speed
        (useful for testing and for informal-transit mode approximation).
    """

    def __init__(
        self,
        friction_path: str | Path | None = None,
        speed_kph_override: float | None = None,
    ) -> None:
        if friction_path is None and speed_kph_override is None:
            msg = (
                "Provide either friction_path (a friction GeoTIFF) "
                "or speed_kph_override (uniform speed in km/h)."
            )
            raise ValueError(msg)
        self._friction_path = Path(friction_path) if friction_path else None
        self._speed_kph_override = speed_kph_override

    @property
    def name(self) -> str:
        return "friction"

    def compute_surface(
        self,
        destination: tuple[float, float],
        bbox: BoundingBox,
        resolution_deg: float,
        mode: TransportMode,
        *,
        transit_profile: TransitProfile | None = None,
    ) -> TravelTimeSurface:
        """Compute travel-time surface via MCP_Geometric least-cost path."""
        try:
            from skimage.graph import MCP_Geometric  # noqa: PLC0415
        except ImportError as exc:
            msg = "scikit-image is required for FrictionSurfaceRouter: pip install scikit-image"
            raise ImportError(msg) from exc

        try:
            import rasterio  # noqa: PLC0415
            from rasterio.transform import rowcol  # noqa: PLC0415
        except ImportError as exc:
            msg = "rasterio is required for FrictionSurfaceRouter: pip install rasterio"
            raise ImportError(msg) from exc

        dest_lat, dest_lon = destination

        # --- 1. Build or load the friction array --------------------------
        friction, affine, src_res = self._load_friction(bbox, resolution_deg)

        # --- 2. Mark destination pixel ------------------------------------
        # Convert (lat, lon) → (row, col) in the friction array
        dest_row, dest_col = _lonlat_to_rowcol(
            dest_lon, dest_lat, affine, friction.shape
        )
        if dest_row is None or dest_col is None:
            warnings.warn(
                f"Destination ({dest_lat:.4f}, {dest_lon:.4f}) is outside "
                "the friction raster extent. Surface will be all-NaN.",
                UserWarning,
                stacklevel=2,
            )
            nrows = max(1, int(np.ceil(bbox.height_deg / resolution_deg)))
            ncols = max(1, int(np.ceil(bbox.width_deg / resolution_deg)))
            data = np.full((nrows, ncols), np.nan, dtype=np.float32)
            return TravelTimeSurface.from_arrays(
                data=data, west=bbox.west, north=bbox.north,
                resolution_deg=resolution_deg, backend=self.name,
                destination=destination,
                metadata={"mode": mode.value, "friction_source": "none"},
            )

        # --- 3. Handle impassable cells -----------------------------------
        impassable = friction > _IMPASSABLE_THRESHOLD_MIN_PER_M
        friction_mcp = friction.copy()
        friction_mcp[impassable] = np.inf

        # --- 4. Run MCP_Geometric from destination pixel ------------------
        # MCP_Geometric treats friction values as cost per unit distance.
        # With sampling=(dy_m, dx_m) set to pixel sizes in metres,
        # cumulative_costs has units = minutes.
        pixel_size_m = src_res * 111_000.0  # degrees → metres (equatorial approx)
        sampling = (pixel_size_m, pixel_size_m)

        mcp = MCP_Geometric(friction_mcp, sampling=sampling, fully_connected=True)
        cumulative_costs, _ = mcp.find_costs(
            starts=[(dest_row, dest_col)]
        )
        cumulative_costs = cumulative_costs.astype(np.float32)

        # Mark unreachable (inf cost) and off-network as NaN
        cumulative_costs[np.isinf(cumulative_costs)] = np.nan
        cumulative_costs[impassable] = np.nan

        # --- 5. Resample to requested resolution if different -------------
        surface = TravelTimeSurface.from_arrays(
            data=cumulative_costs,
            west=bbox.west,
            north=bbox.north,
            resolution_deg=src_res,
            backend=self.name,
            destination=destination,
            metadata={
                "mode": mode.value,
                "friction_source": str(self._friction_path) if self._friction_path else "uniform",
                "speed_kph_override": self._speed_kph_override,
            },
        )

        if abs(src_res - resolution_deg) / resolution_deg > 0.01:
            surface = surface.resample(resolution_deg)

        return surface.clip(bbox)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_friction(
        self,
        bbox: BoundingBox,
        resolution_deg: float,
    ) -> tuple[np.ndarray, object, float]:
        """Return (friction_array, affine_transform, actual_resolution_deg).

        If speed_kph_override is set, builds a uniform friction array at
        the requested resolution. Otherwise clips the GeoTIFF to the bbox.
        """
        if self._speed_kph_override is not None:
            return self._uniform_friction(bbox, resolution_deg)
        return self._load_geotiff_friction(bbox)

    def _uniform_friction(
        self,
        bbox: BoundingBox,
        resolution_deg: float,
    ) -> tuple[np.ndarray, object, float]:
        """Build a uniform friction raster from a speed override."""
        from affine import Affine  # noqa: PLC0415

        speed_m_per_s = self._speed_kph_override * 1000.0 / 3600.0
        # friction = min/m = 1 / (m/s * 60)
        friction_val = np.float32(1.0 / (speed_m_per_s * 60.0))

        nrows = max(1, int(np.ceil(bbox.height_deg / resolution_deg)))
        ncols = max(1, int(np.ceil(bbox.width_deg / resolution_deg)))
        friction = np.full((nrows, ncols), friction_val, dtype=np.float32)
        affine = Affine.translation(bbox.west, bbox.north) * Affine.scale(
            resolution_deg, -resolution_deg
        )
        return friction, affine, resolution_deg

    def _load_geotiff_friction(
        self,
        bbox: BoundingBox,
    ) -> tuple[np.ndarray, object, float]:
        """Load and clip a GeoTIFF friction raster to bbox."""
        import rasterio  # noqa: PLC0415
        from rasterio.windows import from_bounds  # noqa: PLC0415

        if self._friction_path is None or not self._friction_path.exists():
            msg = f"Friction raster not found: {self._friction_path}"
            raise FileNotFoundError(msg)

        with rasterio.open(self._friction_path) as src:
            window = from_bounds(
                bbox.west, bbox.south, bbox.east, bbox.north,
                transform=src.transform,
            )
            data = src.read(1, window=window)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.inf, data)
            data = data.astype(np.float32)
            transform = src.window_transform(window)
            resolution_deg = float(abs(src.transform.a))

        return data, transform, resolution_deg


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _lonlat_to_rowcol(
    lon: float,
    lat: float,
    affine: object,
    shape: tuple[int, int],
) -> tuple[int | None, int | None]:
    """Convert geographic (lon, lat) to (row, col) pixel indices.

    Returns (None, None) if the point is outside the raster extent.
    """
    from affine import Affine  # noqa: PLC0415

    a = affine
    col = (lon - a.c) / a.a
    row = (lat - a.f) / a.e
    r, c = int(row), int(col)
    if 0 <= r < shape[0] and 0 <= c < shape[1]:
        return r, c
    return None, None
