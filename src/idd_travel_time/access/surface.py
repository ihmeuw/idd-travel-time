"""TravelTimeSurface — the core raster output of surface routing.

A ``TravelTimeSurface`` wraps a 2-D numpy array of travel times (in minutes)
together with the geographic metadata needed to locate every pixel on the Earth's
surface.  It is the common return type of all ``SurfaceRouter`` backends.

Layout
------
- ``data``:  float32 array shaped (nrows, ncols), units = minutes.
             NaN where travel time is unknown / unreachable.
- ``affine``: rasterio / affine ``Affine`` transform (top-left corner,
              pixel width, pixel height).
- ``crs``:    coordinate reference system (always EPSG:4326 for outputs).
- ``bbox``:   ``BoundingBox(west, south, east, north)`` in degrees.
- ``resolution_deg``: nominal pixel size in degrees (square pixels assumed).
- ``backend``: name of the router that produced this surface.
- ``destination``: (lat, lon) of the facility this surface measures time to.

Public API
----------
TravelTimeSurface.from_arrays(data, west, south, resolution_deg, ...)
TravelTimeSurface.from_geotiff(path) -> TravelTimeSurface
surface.to_geotiff(path) -> Path
surface.clip(bbox) -> TravelTimeSurface
surface.resample(resolution_deg) -> TravelTimeSurface
surface.mask_nodata(nodata_value) -> TravelTimeSurface
surface.threshold(minutes) -> np.ndarray  (bool mask)
surface.summary_stats() -> dict
surface.to_geodataframe() -> gpd.GeoDataFrame  (point grid with travel_time column)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from affine import Affine

if TYPE_CHECKING:
    import geopandas as gpd

_WGS84 = "EPSG:4326"
_NODATA = np.float32(np.nan)

# When resampling, the NaN mask is zoomed as a float array and re-binarised
# via this threshold — anything > 0.5 of the source pixel is treated as NaN.
_NAN_RESAMPLE_THRESHOLD: float = 0.5


@dataclass
class BoundingBox:
    """Geographic bounding box in WGS-84 degrees."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if self.west >= self.east:
            msg = f"west ({self.west}) must be < east ({self.east})"
            raise ValueError(msg)
        if self.south >= self.north:
            msg = f"south ({self.south}) must be < north ({self.north})"
            raise ValueError(msg)

    @property
    def width_deg(self) -> float:
        return self.east - self.west

    @property
    def height_deg(self) -> float:
        return self.north - self.south

    def buffer(self, deg: float) -> BoundingBox:
        """Expand bounding box by *deg* degrees on all sides."""
        return BoundingBox(
            west=self.west - deg,
            south=self.south - deg,
            east=self.east + deg,
            north=self.north + deg,
        )

    def contains_point(self, lat: float, lon: float) -> bool:
        return self.west <= lon <= self.east and self.south <= lat <= self.north

    def to_dict(self) -> dict[str, float]:
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> BoundingBox:
        return cls(west=d["west"], south=d["south"], east=d["east"], north=d["north"])

    @classmethod
    def from_point(cls, lat: float, lon: float, radius_deg: float) -> BoundingBox:
        """Square bbox centred on (lat, lon) with half-width *radius_deg*."""
        return cls(
            west=lon - radius_deg,
            south=lat - radius_deg,
            east=lon + radius_deg,
            north=lat + radius_deg,
        )


@dataclass
class TravelTimeSurface:
    """Raster travel-time surface from one destination to every grid cell.

    Parameters
    ----------
    data:
        float32 array shaped (nrows, ncols).  Units: minutes.  NaN = no data.
    affine:
        Rasterio-compatible Affine transform.  Pixel (col=0, row=0) is the
        top-left corner.  ``affine.c`` = west edge, ``affine.f`` = north edge.
    crs:
        Coordinate reference system string, e.g. ``"EPSG:4326"``.
    bbox:
        Geographic extent of the output (matches data extent).
    resolution_deg:
        Nominal pixel size in degrees (square pixels assumed).
    backend:
        Name of the router that produced this surface.
    destination:
        ``(lat, lon)`` of the facility this surface measures time to.
    metadata:
        Arbitrary key-value store for provenance (mode, time_context, etc.).
    """

    data: np.ndarray  # float32, shape (nrows, ncols)
    affine: Affine
    crs: str
    bbox: BoundingBox
    resolution_deg: float
    backend: str
    destination: tuple[float, float]  # (lat, lon)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    @property
    def nrows(self) -> int:
        return int(self.data.shape[0])

    @property
    def ncols(self) -> int:
        return int(self.data.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrows, self.ncols)

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        data: np.ndarray,
        west: float,
        north: float,
        resolution_deg: float,
        backend: str,
        destination: tuple[float, float],
        crs: str = _WGS84,
        metadata: dict[str, Any] | None = None,
    ) -> TravelTimeSurface:
        """Construct from a data array and geographic corner coordinates.

        Parameters
        ----------
        data:
            2-D float32 array, shape (nrows, ncols).
        west, north:
            Top-left corner of the raster in degrees.
        resolution_deg:
            Pixel size (square).
        """
        nrows, ncols = data.shape
        affine = Affine.translation(west, north) * Affine.scale(
            resolution_deg, -resolution_deg
        )
        south = north - nrows * resolution_deg
        east = west + ncols * resolution_deg
        bbox = BoundingBox(west=west, south=south, east=east, north=north)
        return cls(
            data=data.astype(np.float32),
            affine=affine,
            crs=crs,
            bbox=bbox,
            resolution_deg=resolution_deg,
            backend=backend,
            destination=destination,
            metadata=metadata or {},
        )

    @classmethod
    def from_geotiff(cls, path: str | Path) -> TravelTimeSurface:
        """Load a TravelTimeSurface from a GeoTIFF written by ``to_geotiff()``.

        The file must have been written by this class (reads JSON sidecar for
        backend / destination / metadata).
        """
        import rasterio

        p = Path(path)
        with rasterio.open(p) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            affine = src.transform
            crs = str(src.crs)
            west, south, east, north = src.bounds
            resolution_deg = float(affine.a)

        bbox = BoundingBox(west=west, south=south, east=east, north=north)

        # Read sidecar JSON for provenance
        sidecar = p.with_suffix(".json")
        meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}

        backend = meta.pop("backend", "unknown")
        destination_raw = meta.pop("destination", [0.0, 0.0])
        destination = (float(destination_raw[0]), float(destination_raw[1]))

        return cls(
            data=data,
            affine=affine,
            crs=crs,
            bbox=bbox,
            resolution_deg=resolution_deg,
            backend=backend,
            destination=destination,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_geotiff(self, path: str | Path) -> Path:
        """Write to a GeoTIFF file with a JSON sidecar for provenance.

        NaN values are written as the nodata value (-9999.0).
        """
        import rasterio
        from rasterio.crs import CRS

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        nodata_val = -9999.0
        data_out = self.data.copy()
        data_out[np.isnan(data_out)] = nodata_val

        with rasterio.open(
            out,
            "w",
            driver="GTiff",
            height=self.nrows,
            width=self.ncols,
            count=1,
            dtype="float32",
            crs=CRS.from_string(self.crs),
            transform=self.affine,
            nodata=nodata_val,
            compress="lzw",
        ) as dst:
            dst.write(data_out, 1)

        # Write JSON sidecar
        sidecar = out.with_suffix(".json")
        meta = dict(self.metadata)
        meta["backend"] = self.backend
        meta["destination"] = list(self.destination)
        sidecar.write_text(json.dumps(meta, indent=2))

        return out

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def clip(self, bbox: BoundingBox) -> TravelTimeSurface:
        """Return a new surface clipped to *bbox*.

        Clips to pixel boundaries — the output bbox will be a pixel-aligned
        subset of the requested bbox.
        """
        # Convert geo coords to pixel indices (row, col)
        col_start = max(0, int((bbox.west - self.bbox.west) / self.resolution_deg))
        row_start = max(0, int((self.bbox.north - bbox.north) / self.resolution_deg))
        col_end = min(
            self.ncols, int(np.ceil((bbox.east - self.bbox.west) / self.resolution_deg))
        )
        row_end = min(
            self.nrows,
            int(np.ceil((self.bbox.north - bbox.south) / self.resolution_deg)),
        )

        clipped = self.data[row_start:row_end, col_start:col_end]
        new_west = self.bbox.west + col_start * self.resolution_deg
        new_north = self.bbox.north - row_start * self.resolution_deg

        return TravelTimeSurface.from_arrays(
            data=clipped,
            west=new_west,
            north=new_north,
            resolution_deg=self.resolution_deg,
            backend=self.backend,
            destination=self.destination,
            crs=self.crs,
            metadata=dict(self.metadata),
        )

    def resample(self, resolution_deg: float) -> TravelTimeSurface:
        """Resample to a different resolution using nearest-neighbour.

        Useful for comparing surfaces produced at different resolutions.
        """
        from scipy.ndimage import zoom

        factor = self.resolution_deg / resolution_deg
        resampled = zoom(
            np.where(np.isnan(self.data), 0.0, self.data),
            factor,
            order=1,
        ).astype(np.float32)
        # Restore NaN mask
        nan_mask = (
            zoom(np.isnan(self.data).astype(np.float32), factor, order=0)
            > _NAN_RESAMPLE_THRESHOLD
        )
        resampled[nan_mask] = np.nan

        return TravelTimeSurface.from_arrays(
            data=resampled,
            west=self.bbox.west,
            north=self.bbox.north,
            resolution_deg=resolution_deg,
            backend=self.backend,
            destination=self.destination,
            crs=self.crs,
            metadata=dict(self.metadata),
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def threshold(self, minutes: float) -> np.ndarray:
        """Return a boolean mask: True where travel_time <= *minutes*.

        NaN cells are False (unreachable).
        """
        valid = ~np.isnan(self.data)
        return np.asarray(valid & (self.data <= minutes))

    def summary_stats(self) -> dict[str, Any]:
        """Return descriptive statistics over valid (non-NaN) pixels."""
        valid = self.data[~np.isnan(self.data)]
        if valid.size == 0:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p25": None,
                "p75": None,
                "nan_fraction": 1.0,
            }
        return {
            "count": int(valid.size),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "p25": float(np.percentile(valid, 25)),
            "p75": float(np.percentile(valid, 75)),
            "nan_fraction": float(np.isnan(self.data).mean()),
        }

    def coverage_at(self, threshold_minutes: float) -> float:
        """Fraction of valid pixels reachable within *threshold_minutes*."""
        valid = ~np.isnan(self.data)
        if not valid.any():
            return 0.0
        reachable = self.threshold(threshold_minutes)
        return float(reachable.sum()) / float(valid.sum())

    def pixel_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lats, lons) arrays of pixel centre coordinates.

        Both arrays have shape (nrows, ncols).
        """
        cols = np.arange(self.ncols)
        rows = np.arange(self.nrows)
        col_grid, row_grid = np.meshgrid(cols, rows)

        lons = self.bbox.west + (col_grid + 0.5) * self.resolution_deg
        lats = self.bbox.north - (row_grid + 0.5) * self.resolution_deg
        return lats.astype(np.float64), lons.astype(np.float64)

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert to a point GeoDataFrame with a ``travel_time_min`` column.

        Only valid (non-NaN) pixels are included.  Useful for joining to
        population grids or plotting with geopandas.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        lats, lons = self.pixel_coords()
        mask = ~np.isnan(self.data)
        valid_lats = lats[mask]
        valid_lons = lons[mask]
        valid_times = self.data[mask]

        geometries = [
            Point(lon, lat) for lon, lat in zip(valid_lons, valid_lats, strict=False)
        ]
        return gpd.GeoDataFrame(
            {"travel_time_min": valid_times.astype(np.float32)},
            geometry=geometries,
            crs=_WGS84,
        )

    def __repr__(self) -> str:
        s = self.summary_stats()
        return (
            f"TravelTimeSurface("
            f"{self.nrows}×{self.ncols} px, "
            f"{self.resolution_deg:.4f}°/px, "
            f"backend={self.backend!r}, "
            f"mean={s['mean']:.1f} min, "
            f"nan={s['nan_fraction']:.1%})"
        )
