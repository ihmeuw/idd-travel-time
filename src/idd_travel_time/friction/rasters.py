"""Rasterising the vector inputs onto a block grid as speed (km/h) layers.

Within one layer, overlapping features resolve to the fastest speed
(DECISIONS 2026-10-20/21). Layer precedence is applied in ``build``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import shapely
from rasterio.features import rasterize

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from idd_travel_time.friction.frame import BlockGrid
    from idd_travel_time.friction.params import FrictionParams

NO_SPEED = np.float32(np.nan)
CLIP_MARGIN_M = 100.0  # keep a little beyond the block so edge cells still get touched


def _empty(grid: BlockGrid) -> NDArray[np.float32]:
    return np.full(grid.shape, np.nan, dtype=np.float32)


def clip_to_grid(gdf: gpd.GeoDataFrame, grid: BlockGrid) -> gpd.GeoDataFrame:
    """Reproject to the frame and clip to the block (plus a margin), dropping empties.

    Rasterising unclipped country, lake or land-cover polygons that extend far
    beyond the block is what drove the probe's 45 GB peak; clipping first
    keeps only the vertices the block can see.
    """
    g = gdf.to_crs(grid.crs)
    clipped = shapely.clip_by_rect(
        g.geometry.values,
        grid.xmin - CLIP_MARGIN_M,
        grid.ymin - CLIP_MARGIN_M,
        grid.xmax + CLIP_MARGIN_M,
        grid.ymax + CLIP_MARGIN_M,
    )
    keep = ~shapely.is_empty(clipped) & ~shapely.is_missing(clipped)
    out = g.loc[keep].copy()
    out["geometry"] = gpd.GeoSeries(clipped[keep], index=out.index, crs=grid.crs)
    return out


def rasterize_fastest(
    grid: BlockGrid,
    groups: Iterable[tuple[float, gpd.GeoSeries]],
    *,
    all_touched: bool,
    out: NDArray[np.float32] | None = None,
    count: NDArray[np.uint8] | None = None,
) -> NDArray[np.float32]:
    """Burn each (speed, geometries) group; where groups overlap the fastest speed wins.

    Geometries must already be in the grid CRS. NaN marks cells no group
    touched. If *count* is given it receives, per cell, how many groups
    touched it (2+ means the fastest-wins tie-break decided that cell).
    """
    acc = _empty(grid) if out is None else out
    hits = np.zeros(grid.shape, dtype=np.uint8) if count is not None else None
    for speed, geoms in groups:
        if len(geoms) == 0:
            continue
        burned = rasterize(
            ((g, 1) for g in geoms.values),
            out_shape=grid.shape,
            transform=grid.transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
        )
        hit = burned == 1
        acc[hit] = np.fmax(acc[hit], np.float32(speed))
        if hits is not None:
            hits[hit] += 1
    if count is not None and hits is not None:
        count[...] = hits
    return acc


def land_cover_speed(
    grid: BlockGrid,
    land_cover: gpd.GeoDataFrame,
    params: FrictionParams,
    count: NDArray[np.uint8] | None = None,
) -> NDArray[np.float32]:
    """Walking km/h from land cover; overlapping subtypes resolve to the fastest (V0 rule)."""
    lc = clip_to_grid(land_cover, grid)
    unknown = set(lc["subtype"].unique()) - set(params.land_cover_kph)
    if unknown:
        msg = f"land-cover subtypes without a speed: {sorted(unknown)}"
        raise ValueError(msg)
    groups = [
        (params.land_cover_kph[st], lc.loc[lc["subtype"] == st, "geometry"])
        for st in params.land_cover_kph
    ]
    return rasterize_fastest(grid, groups, all_touched=False, count=count)


def country_raster(
    grid: BlockGrid, countries: gpd.GeoDataFrame
) -> tuple[NDArray[np.int32], list[str | None]]:
    """Per-cell country index (0 = none) and the ISO3 list it indexes (index i -> iso3s[i-1])."""
    c = clip_to_grid(countries, grid)
    iso3s: list[str | None] = []
    shapes = []
    for i, (geom, iso3) in enumerate(
        zip(c.geometry.values, c["iso3"].tolist(), strict=True), start=1
    ):
        iso3s.append(iso3)
        shapes.append((geom, i))
    if not shapes:
        return np.zeros(grid.shape, dtype=np.int32), iso3s
    arr = rasterize(
        shapes, out_shape=grid.shape, transform=grid.transform, fill=0, dtype="int32"
    )
    return np.asarray(arr, dtype=np.int32), iso3s


def road_speed(
    grid: BlockGrid,
    roads: dict[str, gpd.GeoDataFrame],
    country_idx: NDArray[np.int32],
    iso3s: list[str | None],
    params: FrictionParams,
    *,
    walking_kph: float | None = None,
) -> NDArray[np.float32]:
    """Road km/h per cell: per-country motorised speed by class, or a flat walking speed.

    Each class is burned as a mask (every cell a line touches); the speed then
    comes from the cell's country. Overlapping classes resolve to the fastest.
    """
    acc = _empty(grid)
    for cls, gdf in roads.items():
        if len(gdf) == 0:
            continue
        geoms = clip_to_grid(gdf, grid).geometry
        if len(geoms) == 0:
            continue
        burned = rasterize(
            ((g, 1) for g in geoms.values),
            out_shape=grid.shape,
            transform=grid.transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        hit = burned == 1
        if not hit.any():
            continue
        if walking_kph is not None:
            speed = np.full(grid.shape, np.float32(walking_kph), dtype=np.float32)
        else:
            lookup = np.array(
                [params.road_speed(None, cls)]
                + [params.road_speed(iso, cls) for iso in iso3s],
                dtype=np.float32,
            )
            speed = lookup[country_idx]
        acc[hit] = np.fmax(acc[hit], speed[hit])
    return acc


def water_speed(
    grid: BlockGrid,
    water: dict[str, gpd.GeoDataFrame],
    params: FrictionParams,
    *,
    mode: str,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Water km/h per cell for *mode* ('motorised' | 'walking') and the cells to mask (ocean when walking)."""
    acc = _empty(grid)
    mask = np.zeros(grid.shape, dtype=bool)
    for cls, gdf in water.items():
        rule = params.water[cls]
        if len(gdf) == 0:
            continue
        geoms = clip_to_grid(gdf, grid).geometry
        if len(geoms) == 0:
            continue
        is_line = geoms.geom_type.isin(["LineString", "MultiLineString"]).to_numpy()
        burned = np.zeros(grid.shape, dtype=np.uint8)
        for subset, touched in ((geoms[~is_line], False), (geoms[is_line], True)):
            if len(subset) == 0:
                continue
            burned |= rasterize(
                ((g, 1) for g in subset.values),
                out_shape=grid.shape,
                transform=grid.transform,
                fill=0,
                all_touched=touched,
                dtype="uint8",
            )
        hit = burned == 1
        speed = rule.motorised_kph if mode == "motorised" else rule.walking_kph
        if mode == "walking" and rule.walking_masked:
            mask |= hit
            continue
        if speed is None:
            continue
        acc[hit] = np.fmax(acc[hit], np.float32(speed))
    return acc, mask
