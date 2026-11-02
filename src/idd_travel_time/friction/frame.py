"""The equal-area block grid shared with the rra population model.

Blocks come from the rra building-density tile index (ESRI:54034, 512-cell
tiles, 16 x 16 tiles per block). A block's raster is ``cells x cells`` at
``resolution_m`` with its origin at the block's top-left corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
from affine import Affine
from pyproj import CRS, Transformer

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

FRAME_CRS = "ESRI:54034"
LONLAT_CRS = "EPSG:4326"
TILE_CELLS = 512
TILES_PER_BLOCK = 16
BLOCK_CELLS = TILE_CELLS * TILES_PER_BLOCK  # 8192


@dataclass(frozen=True)
class BlockGrid:
    """Raster geometry of one block: bounds in the frame, transform, shape and per-row latitude.

    ``xmin`` / ``ymax`` are the block's top-left corner, which lies on the
    frame's cell lattice. ``xmax`` / ``ymin`` are the block's extent in the
    tile index. Edge blocks are clipped to the world bounds there, so their
    span is not a whole number of cells: the raster then rounds the cell
    count up and its last row or column overhangs the world bound by a
    fraction of a cell, keeping every cell on the same lattice as the
    interior blocks rather than shrinking cells at the edge.
    """

    block_key: str
    resolution_m: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def transform(self) -> Affine:
        """Affine transform of the block raster (north-up, origin top-left)."""
        return Affine(
            self.resolution_m, 0.0, self.xmin, 0.0, -self.resolution_m, self.ymax
        )

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols): whole cells covering the block, rounded up at clipped edges."""
        rows = math.ceil(round((self.ymax - self.ymin) / self.resolution_m, 6))
        cols = math.ceil(round((self.xmax - self.xmin) / self.resolution_m, 6))
        return rows, cols

    @property
    def raster_bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) actually covered by the raster cells."""
        rows, cols = self.shape
        return (
            self.xmin,
            self.ymax - rows * self.resolution_m,
            self.xmin + cols * self.resolution_m,
            self.ymax,
        )

    @property
    def clipped(self) -> bool:
        """True for an edge block whose index extent is not a whole number of cells."""
        _, ymin, xmax, _ = self.raster_bounds
        return not (
            math.isclose(ymin, self.ymin, abs_tol=1e-6)
            and math.isclose(xmax, self.xmax, abs_tol=1e-6)
        )

    @property
    def crs(self) -> str:
        """Frame CRS identifier."""
        return FRAME_CRS

    def lonlat_bounds(self, pad_deg: float = 0.0) -> tuple[float, float, float, float]:
        """(lon_min, lat_min, lon_max, lat_max) of the block, optionally padded."""
        poly = gpd.GeoSeries.from_xy(
            [self.xmin, self.xmax], [self.ymin, self.ymax], crs=FRAME_CRS
        )
        b = poly.to_crs(LONLAT_CRS).total_bounds
        corners = (
            gpd.GeoSeries.from_xy(
                [self.xmin, self.xmax, self.xmin, self.xmax],
                [self.ymin, self.ymin, self.ymax, self.ymax],
                crs=FRAME_CRS,
            )
            .to_crs(LONLAT_CRS)
            .total_bounds
        )
        lo = np.minimum(b, corners)[:2]
        hi = np.maximum(b, corners)[2:]
        return (
            float(lo[0] - pad_deg),
            float(lo[1] - pad_deg),
            float(hi[0] + pad_deg),
            float(hi[1] + pad_deg),
        )

    def row_latitudes(self) -> NDArray[np.float64]:
        """Latitude (degrees) of each row's cell centres."""
        rows, _ = self.shape
        ys = self.ymax - (np.arange(rows) + 0.5) * self.resolution_m
        xs = np.full(rows, (self.xmin + self.xmax) / 2.0)
        _, lats = _to_lonlat().transform(xs, ys)
        return np.asarray(lats, dtype=np.float64)

    def ground_lengths(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per row: ground metres spanned by one cell east-west and north-south.

        On the cylindrical equal-area frame a cell of *resolution_m* projected
        metres covers ``r * cos(lat)`` metres east-west and ``r / cos(lat)``
        north-south; the product is always ``r * r``.
        """
        cos_lat = np.cos(np.radians(self.row_latitudes()))
        return self.resolution_m * cos_lat, self.resolution_m / cos_lat


@lru_cache(maxsize=1)
def _to_lonlat() -> Transformer:
    return Transformer.from_crs(
        CRS.from_user_input(FRAME_CRS), CRS.from_user_input(LONLAT_CRS), always_xy=True
    )


def load_tile_index(tile_index_root: Path, resolution_m: int) -> gpd.GeoDataFrame:
    """The rra tile index for *resolution_m* (40 or 100): one row per tile with block_key and tile_key."""
    path = tile_index_root / f"{resolution_m}m" / "tile_index.parquet"
    idx = gpd.read_parquet(path)
    if idx.crs is None or CRS.from_user_input(idx.crs) != CRS.from_user_input(
        FRAME_CRS
    ):
        msg = f"tile index CRS {idx.crs} is not {FRAME_CRS}"
        raise ValueError(msg)
    return idx


def block_grid(
    tile_index: gpd.GeoDataFrame, block_key: str, resolution_m: float
) -> BlockGrid:
    """Grid of one block from the index rows that carry its key.

    The top-left corner must sit on the frame lattice (a whole number of
    cells from the frame's top-left corner); the far edges may be clipped.
    """
    tiles = tile_index[tile_index["block_key"] == block_key]
    if tiles.empty:
        msg = f"block {block_key!r} not in tile index"
        raise KeyError(msg)
    xmin, ymin, xmax, ymax = (float(v) for v in tiles.total_bounds)
    frame_xmin, _, _, frame_ymax = (float(v) for v in tile_index.total_bounds)
    for name, offset in (("x", xmin - frame_xmin), ("y", frame_ymax - ymax)):
        cells = offset / resolution_m
        if not math.isclose(cells, round(cells), abs_tol=1e-6):
            msg = f"block {block_key} top-left is off the frame lattice in {name} by {offset % resolution_m:.3f} m"
            raise ValueError(msg)
    return BlockGrid(
        block_key=block_key,
        resolution_m=resolution_m,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )


def block_for_point(tile_index: gpd.GeoDataFrame, lon: float, lat: float) -> str:
    """Block key containing a longitude/latitude point."""
    pt = gpd.GeoSeries.from_xy([lon], [lat], crs=LONLAT_CRS).to_crs(FRAME_CRS).iloc[0]
    hit = tile_index[tile_index.intersects(pt)]
    if hit.empty:
        msg = f"no block contains ({lon}, {lat})"
        raise KeyError(msg)
    return str(hit["block_key"].iloc[0])


def land_block_keys(tile_index: gpd.GeoDataFrame, land: gpd.GeoDataFrame) -> list[str]:
    """Block keys whose extent intersects any geometry of *land* (same CRS required)."""
    blocks = tile_index.dissolve("block_key")[["geometry"]].reset_index()
    hit = blocks.sjoin(
        land[["geometry"]].to_crs(blocks.crs), how="inner", predicate="intersects"
    )
    return sorted(hit["block_key"].unique().tolist())
