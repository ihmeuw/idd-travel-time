"""Readers for the friction inputs: Overture layers by bounding box, elevation tiles by block.

All readers are bounding-box scoped so one block never loads a global file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pyarrow.dataset as ds
import pycountry
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import rowcol
from rasterio.warp import reproject

from idd_travel_time.friction.frame import LONLAT_CRS
from idd_travel_time.friction.params import ROAD_CLASSES, WATER_CLASSES

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from idd_travel_time.friction.frame import BlockGrid

LAND_COVER_DETAIL_MIN_ZOOM = (
    8  # Overture's detailed band; the zoom 0-7 band duplicates it coarsely
)
_TILE_NAME = "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM"
# The GLO-90 release names its 3 arc-second tiles COG_30; the gap-fill mirror
# keeps those names. Looking for the COG_10 name there found nothing, so the
# gap-fill never served a tile in the 2026-11-05 build (Armenia and Azerbaijan
# had no elevation).
_GLO90_NAME = "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM"
_FABDEM_NAME = "{ns}{lat:02d}{ew}{lon:03d}_FABDEM_V1-2"
_TILE_RE = re.compile(r"_([NS])(\d{2})_00_([EW])(\d{3})_00_")
ELEVATION_SOURCE_ORDER = ("fabdem", "glo30", "glo90")


@dataclass(frozen=True)
class InputPaths:
    """Where the inputs live (from ``paths.yaml``; see ``load_input_paths``)."""

    roads_dir: Path
    water_dir: Path
    land_cover_dir: Path
    divisions_dir: Path
    elevation_dir: Path
    elevation_fallback_dir: Path | None
    tile_index_root: Path
    fabdem_dir: Path | None = None


def load_input_paths(paths: dict[str, str]) -> InputPaths:
    """Build :class:`InputPaths` from the ``paths.yaml`` dict."""
    land_cover = Path(paths["overture_land_cover_dir"])
    divisions = land_cover.parents[1] / "theme=divisions" / "type=division_area"
    fallback = paths.get("elevation_fallback_dir")
    fabdem = paths.get("fabdem_dir")
    return InputPaths(
        roads_dir=Path(paths["overture_roads_dir"]),
        water_dir=Path(paths["overture_water_dir"]),
        land_cover_dir=land_cover,
        divisions_dir=divisions,
        elevation_dir=Path(paths["elevation_dir"]),
        elevation_fallback_dir=Path(fallback) if fallback else None,
        tile_index_root=Path(paths["rra_tile_index_root"]),
        fabdem_dir=Path(fabdem) if fabdem else None,
    )


def _bbox_filter(bounds: tuple[float, float, float, float]) -> ds.Expression:
    xmin, ymin, xmax, ymax = bounds
    return (
        (ds.field("bbox", "xmin") < xmax)
        & (ds.field("bbox", "xmax") > xmin)
        & (ds.field("bbox", "ymin") < ymax)
        & (ds.field("bbox", "ymax") > ymin)
    )


def read_geoparquet_bbox(
    files: list[Path],
    bounds: tuple[float, float, float, float],
    columns: list[str],
    extra_filter: ds.Expression | None = None,
) -> gpd.GeoDataFrame:
    """Rows of Overture GeoParquet files whose bbox column intersects *bounds* (lon/lat)."""
    if not files:
        msg = "no parquet files given"
        raise FileNotFoundError(msg)
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    flt = _bbox_filter(bounds)
    if extra_filter is not None:
        flt = flt & extra_filter
    table = dataset.to_table(filter=flt, columns=[*columns, "geometry"])
    geometry = gpd.GeoSeries.from_wkb(table["geometry"].to_numpy(zero_copy_only=False))
    frame = table.drop(["geometry"]).to_pandas()
    out = gpd.GeoDataFrame(frame, geometry=geometry, crs=LONLAT_CRS)
    # What the block actually scanned: the input-volume covariate for sizing
    # (DECISIONS 2026-11-03); cell counts of the output do not carry it.
    out.attrs["input_volume"] = {
        "rows": int(table.num_rows),
        "mb": table.nbytes / 1e6,
    }
    return out


def read_roads(
    paths: InputPaths, bounds: tuple[float, float, float, float]
) -> dict[str, gpd.GeoDataFrame]:
    """Road geometries per Overture class (one GeoParquet per class on disk)."""
    out: dict[str, gpd.GeoDataFrame] = {}
    for cls in ROAD_CLASSES:
        f = paths.roads_dir / f"{cls}.parquet"
        if not f.exists():
            msg = f"roads file missing: {f}"
            raise FileNotFoundError(msg)
        out[cls] = read_geoparquet_bbox([f], bounds, columns=["id"])
    return out


def read_water(
    paths: InputPaths, bounds: tuple[float, float, float, float]
) -> dict[str, gpd.GeoDataFrame]:
    """Water geometries per Overture water class; points are dropped (they carry no extent)."""
    out: dict[str, gpd.GeoDataFrame] = {}
    for cls in WATER_CLASSES:
        f = paths.water_dir / f"{cls}.parquet"
        if not f.exists():
            msg = f"water file missing: {f}"
            raise FileNotFoundError(msg)
        g = read_geoparquet_bbox([f], bounds, columns=["id"])
        out[cls] = g[~g.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    return out


def read_land_cover(
    paths: InputPaths, bounds: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Land-cover polygons of the detailed zoom band with their subtype."""
    files = sorted(paths.land_cover_dir.glob("*.parquet"))
    return read_geoparquet_bbox(
        files,
        bounds,
        columns=["subtype"],
        extra_filter=ds.field("cartography", "min_zoom") == LAND_COVER_DETAIL_MIN_ZOOM,
    )


def read_countries(
    paths: InputPaths, bounds: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Country polygons with an ``iso3`` column (None where the ISO2 code has no ISO3)."""
    files = sorted(paths.divisions_dir.glob("*.parquet"))
    g = read_geoparquet_bbox(
        files,
        bounds,
        columns=["country"],
        extra_filter=ds.field("subtype") == "country",
    )
    g["iso3"] = [iso2_to_iso3(c) for c in g["country"].tolist()]
    return g


def iso2_to_iso3(code: str | None) -> str | None:
    """ISO 3166-1 alpha-2 -> alpha-3, or None when unknown (e.g. Kosovo's XK)."""
    if not code:
        return None
    hit = pycountry.countries.get(alpha_2=code.upper())
    return None if hit is None else str(hit.alpha_3)


def tile_corners(bounds: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """South-west corners (lat, lon) of the one-degree tiles covering *bounds*."""
    lon0, lat0, lon1, lat1 = bounds
    return [
        (lat, lon)
        for lat in range(int(np.floor(lat0)), int(np.ceil(lat1)))
        for lon in range(int(np.floor(lon0)), int(np.ceil(lon1)))
    ]


def copernicus_tile_name(lat: int, lon: int) -> str:
    """Copernicus DEM tile stem for the tile whose south-west corner is (lat, lon)."""
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return _TILE_NAME.format(ns=ns, lat=abs(lat), ew=ew, lon=abs(lon))


def fabdem_tile_name(lat: int, lon: int) -> str:
    """FABDEM V1-2 tile stem for the tile whose south-west corner is (lat, lon)."""
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return _FABDEM_NAME.format(ns=ns, lat=abs(lat), ew=ew, lon=abs(lon))


def glo90_tile_name(lat: int, lon: int) -> str:
    """Copernicus GLO-90 tile stem (``COG_30``) for the tile whose south-west corner is (lat, lon)."""
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return _GLO90_NAME.format(ns=ns, lat=abs(lat), ew=ew, lon=abs(lon))


def elevation_tile_names(bounds: tuple[float, float, float, float]) -> list[str]:
    """Copernicus one-degree tile names covering *bounds*; the name is the south-west corner."""
    return [copernicus_tile_name(lat, lon) for lat, lon in tile_corners(bounds)]


Corner = tuple[int, int]  # (lat, lon) of a one-degree tile's south-west corner


@dataclass(frozen=True)
class ElevationTiles:
    """Resolved elevation files for a block: every source that has each tile, in order."""

    files: list[Path]  # the first source's file per tile, in tile order
    sources: dict[str, int]  # tiles per first source, in ELEVATION_SOURCE_ORDER
    missing: list[str]  # Copernicus-style names of tiles no source has
    by_source: dict[str, dict[Corner, Path]]  # every existing file, by source


def _tile_candidates(paths: InputPaths, lat: int, lon: int) -> list[tuple[str, Path]]:
    """The file each source would hold for the tile at (lat, lon), in ``ELEVATION_SOURCE_ORDER``."""
    candidates: list[tuple[str, Path]] = []
    if paths.fabdem_dir is not None:
        candidates.append(
            ("fabdem", paths.fabdem_dir / "tiles" / f"{fabdem_tile_name(lat, lon)}.tif")
        )
    candidates.append(
        (
            "glo30",
            paths.elevation_dir / "tiles" / f"{copernicus_tile_name(lat, lon)}.tif",
        )
    )
    if paths.elevation_fallback_dir is not None:
        candidates.append(
            (
                "glo90",
                paths.elevation_fallback_dir
                / "tiles"
                / f"{glo90_tile_name(lat, lon)}.tif",
            )
        )
    return candidates


def resolve_elevation_tiles(
    paths: InputPaths, bounds: tuple[float, float, float, float]
) -> ElevationTiles:
    """Per tile: FABDEM first, then Copernicus GLO-30, then the GLO-90 gap-fill.

    FABDEM (Hawker 2022) removes buildings and forest from the Copernicus DSM
    but covers 60S-80N only; outside that, and for any tile it lacks, the
    Copernicus chain stands in. ``by_source`` keeps every file that exists so
    :func:`elevation_with_fallback` can fill a cell the first source leaves
    empty from the next one.
    """
    files: list[Path] = []
    sources = dict.fromkeys(ELEVATION_SOURCE_ORDER, 0)
    missing: list[str] = []
    by_source: dict[str, dict[Corner, Path]] = {s: {} for s in ELEVATION_SOURCE_ORDER}
    for lat, lon in tile_corners(bounds):
        existing = [
            (source, candidate)
            for source, candidate in _tile_candidates(paths, lat, lon)
            if candidate.exists()
        ]
        for source, candidate in existing:
            by_source[source][(lat, lon)] = candidate
        if existing:
            files.append(existing[0][1])
            sources[existing[0][0]] += 1
        else:
            missing.append(copernicus_tile_name(lat, lon))
    return ElevationTiles(
        files=files, sources=sources, missing=missing, by_source=by_source
    )


@dataclass(frozen=True)
class ElevationFill:
    """How :func:`elevation_with_fallback` filled the block grid."""

    cells_by_source: dict[str, int]  # grid cells whose value came from each source
    nodata_tiles: list[str]  # file stems whose data had empty cells inside the block


def _merge_files(files: list[Path]) -> tuple[NDArray[np.float32], Any, Any]:
    """Mosaic of *files* on the first file's grid: (array, transform, crs); NaN where no file covers."""
    handles = [rasterio.open(f) for f in files]
    try:
        mosaic, transform = merge(handles, nodata=np.nan, dtype="float32")
        crs = handles[0].crs
    finally:
        for h in handles:
            h.close()
    return np.asarray(mosaic[0], dtype=np.float32), transform, crs


def _reproject_to_grid(
    mosaic: NDArray[np.float32], transform: Any, crs: Any, grid: BlockGrid
) -> NDArray[np.float32]:
    """*mosaic* resampled bilinearly onto the block grid; NaN where it has no value."""
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    reproject(
        source=mosaic,
        destination=out,
        src_transform=transform,
        src_crs=crs,
        src_nodata=np.nan,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return out


def _tile_has_nodata(
    mosaic: NDArray[np.float32], transform: Any, corner: Corner
) -> bool:
    """True when the one-degree tile at *corner* holds a NaN inside the mosaic."""
    lat, lon = corner
    (r0, c0), (r1, c1) = (
        rowcol(transform, lon, lat + 1),
        rowcol(transform, lon + 1, lat),
    )
    rows, cols = mosaic.shape
    r0, r1 = max(int(r0), 0), min(int(r1), rows)
    c0, c1 = max(int(c0), 0), min(int(c1), cols)
    return bool(np.isnan(mosaic[r0:r1, c0:c1]).any())


def elevation_on_grid(grid: BlockGrid, files: list[Path]) -> NDArray[np.float32]:
    """Elevation (m) resampled bilinearly onto the block grid; NaN where no tile covers."""
    if not files:
        return np.full(grid.shape, np.nan, dtype=np.float32)
    return _reproject_to_grid(*_merge_files(files), grid)


def elevation_with_fallback(
    grid: BlockGrid, tiles: ElevationTiles
) -> tuple[NDArray[np.float32], ElevationFill]:
    """Elevation on the block grid, each cell from the first source that has a value for it.

    One pass per source in ``ELEVATION_SOURCE_ORDER``: the source's files for
    the tiles still pending are merged on their own grid (so a 3 arc-second
    GLO-90 tile never coarsens a 1 arc-second neighbour) and resampled onto the
    block grid, and the cells still NaN take its values. A tile leaves the
    pending set once a source covered it without an empty cell; a tile whose
    file has empty cells (FABDEM N29W086 is nodata on 80 percent of its
    cells) stays pending for the next source. A cell no source has stays NaN.
    """
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    pending: set[Corner] = {c for files in tiles.by_source.values() for c in files}
    cells_by_source: dict[str, int] = {}
    nodata_tiles: list[str] = []
    for source in ELEVATION_SOURCE_ORDER:
        have = {
            c: p for c, p in tiles.by_source.get(source, {}).items() if c in pending
        }
        if not have:
            continue
        mosaic, transform, crs = _merge_files(list(have.values()))
        layer = _reproject_to_grid(mosaic, transform, crs, grid)
        take = np.isnan(out) & np.isfinite(layer)
        out[take] = layer[take]
        cells_by_source[source] = int(take.sum())
        del layer, take
        for corner, path in have.items():
            if _tile_has_nodata(mosaic, transform, corner):
                nodata_tiles.append(path.stem)
            else:
                pending.discard(corner)
        del mosaic
    return out, ElevationFill(
        cells_by_source=cells_by_source, nodata_tiles=nodata_tiles
    )
