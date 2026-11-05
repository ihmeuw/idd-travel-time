"""Input helpers that need no shared-drive access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from idd_travel_time.friction import inputs as io_
from idd_travel_time.friction.frame import BlockGrid


def test_elevation_tile_names_cover_bounds_and_name_sw_corner() -> None:
    names = io_.elevation_tile_names((31.9, -1.7, 34.9, 1.2))
    assert len(names) == 16
    assert "Copernicus_DSM_COG_10_S02_00_E031_00_DEM" in names
    assert "Copernicus_DSM_COG_10_N01_00_E034_00_DEM" in names
    assert io_.elevation_tile_names((-150.2, 61.1, -149.8, 61.3)) == [
        "Copernicus_DSM_COG_10_N61_00_W151_00_DEM",
        "Copernicus_DSM_COG_10_N61_00_W150_00_DEM",
    ]


def test_iso2_to_iso3() -> None:
    assert io_.iso2_to_iso3("UG") == "UGA"
    assert io_.iso2_to_iso3("ug") == "UGA"
    assert io_.iso2_to_iso3("XK") is None  # Kosovo: no ISO 3166-1 code
    assert io_.iso2_to_iso3(None) is None


def test_load_input_paths_derives_divisions_dir() -> None:
    p = io_.load_input_paths(
        {
            "overture_roads_dir": "/r",
            "overture_water_dir": "/w",
            "overture_land_cover_dir": "/x/2025-04-23.0/theme=base/type=land_cover",
            "elevation_dir": "/e",
            "rra_tile_index_root": "/t",
        }
    )
    assert p.divisions_dir == Path("/x/2025-04-23.0/theme=divisions/type=division_area")
    assert p.elevation_fallback_dir is None


def _lonlat_tile(
    path: Path,
    data: np.ndarray,
    *,
    lon: float = 32.0,
    lat_top: float = 1.0,
    nodata: float | None = None,
) -> Path:
    """Write *data* as a one-degree EPSG:4326 tile whose north-west corner is (lon, lat_top)."""
    n = data.shape[0]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=Affine(1 / n, 0, lon, 0, -1 / n, lat_top),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


def _grid_near(lon: float, lat: float, half_m: float = 2000.0) -> BlockGrid:
    """A small block grid (ESRI:54034 metres) centred on (lon, lat)."""
    import geopandas as gpd

    cx, cy = (
        gpd.GeoSeries.from_xy([lon], [lat], crs="EPSG:4326")
        .to_crs("ESRI:54034")
        .iloc[0]
        .coords[0]
    )
    return BlockGrid("T", 40.0, cx - half_m, cy - half_m, cx + half_m, cy + half_m)


def test_elevation_with_fallback_fills_empty_cells_from_the_next_source(
    tmp_path: Path,
) -> None:
    n = 360
    # FABDEM-like tile: 100 m everywhere, nodata (-9999, as the real tiles) in a
    # band about 1.2 km tall through the 4 km grid's middle; GLO-30-like tile:
    # 50 m everywhere.
    fab_data = np.full((n, n), 100.0, dtype=np.float32)
    fab_data[178:182, 165:195] = -9999.0
    fab = _lonlat_tile(tmp_path / "N00E032_FABDEM_V1-2.tif", fab_data, nodata=-9999.0)
    glo = _lonlat_tile(
        tmp_path / "Copernicus_DSM_COG_10_N00_00_E032_00_DEM.tif",
        np.full((n, n), 50.0, dtype=np.float32),
    )
    grid = _grid_near(32.5, 0.5)
    tiles = io_.ElevationTiles(
        files=[fab],
        sources={"fabdem": 1, "glo30": 0, "glo90": 0},
        missing=[],
        by_source={"fabdem": {(0, 32): fab}, "glo30": {(0, 32): glo}, "glo90": {}},
    )
    z, fill = io_.elevation_with_fallback(grid, tiles)
    assert z.shape == grid.shape
    assert not np.isnan(z).any()  # every hole cell took the next source
    values = set(np.unique(z.round(3)).tolist())
    assert values == {50.0, 100.0}
    n_hole = int((z == 50.0).sum())
    assert n_hole > 0
    assert fill.cells_by_source == {"fabdem": z.size - n_hole, "glo30": n_hole}
    assert fill.nodata_tiles == [fab.stem]
    # the same tile with no second source: the hole stays NaN and is reported
    alone = io_.ElevationTiles(
        files=[fab],
        sources={"fabdem": 1, "glo30": 0, "glo90": 0},
        missing=[],
        by_source={"fabdem": {(0, 32): fab}, "glo30": {}, "glo90": {}},
    )
    z_alone, fill_alone = io_.elevation_with_fallback(grid, alone)
    assert int(np.isnan(z_alone).sum()) == n_hole
    assert fill_alone.cells_by_source == {"fabdem": z.size - n_hole}
    assert fill_alone.nodata_tiles == [fab.stem]
    # a clean first source leaves nothing pending: the second source is never read
    clean = io_.ElevationTiles(
        files=[glo],
        sources={"fabdem": 0, "glo30": 1, "glo90": 0},
        missing=[],
        by_source={
            "fabdem": {},
            "glo30": {(0, 32): glo},
            "glo90": {(0, 32): tmp_path / "absent.tif"},
        },
    )
    z_clean, fill_clean = io_.elevation_with_fallback(grid, clean)
    assert float(z_clean.min()) == float(z_clean.max()) == 50.0
    assert fill_clean.cells_by_source == {"glo30": z.size}
    assert fill_clean.nodata_tiles == []
    # nothing at all: all NaN, nothing taken
    nothing = io_.ElevationTiles(files=[], sources={}, missing=["x"], by_source={})
    z_none, fill_none = io_.elevation_with_fallback(grid, nothing)
    assert np.isnan(z_none).all()
    assert fill_none == io_.ElevationFill(cells_by_source={}, nodata_tiles=[])


def test_elevation_on_grid_resamples_a_lonlat_tile(tmp_path: Path) -> None:
    # a 1-degree tile with elevation = 100 * latitude-fraction, in EPSG:4326
    tile = tmp_path / "t.tif"
    n = 360
    data = np.tile(np.linspace(100, 0, n, dtype=np.float32).reshape(-1, 1), (1, n))
    with rasterio.open(
        tile,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=Affine(1 / n, 0, 32.0, 0, -1 / n, 1.0),
    ) as dst:
        dst.write(data, 1)
    # a small block grid inside that tile (ESRI:54034 metres near 32.5E, 0.5N)
    import geopandas as gpd

    cx, cy = (
        gpd.GeoSeries.from_xy([32.5], [0.5], crs="EPSG:4326")
        .to_crs("ESRI:54034")
        .iloc[0]
        .coords[0]
    )
    grid = BlockGrid("T", 40.0, cx - 2000, cy - 2000, cx + 2000, cy + 2000)
    z = io_.elevation_on_grid(grid, [tile])
    assert z.shape == (100, 100)
    assert np.isfinite(z).all()
    assert 40 < float(z.mean()) < 60  # mid-tile latitude -> ~50
    assert z[0].mean() > z[-1].mean()  # north rows higher
    assert io_.elevation_on_grid(grid, []).shape == (100, 100)
    assert np.isnan(io_.elevation_on_grid(grid, [])).all()


def test_fabdem_and_copernicus_tile_names_share_the_corner_convention() -> None:
    assert io_.fabdem_tile_name(0, 10) == "N00E010_FABDEM_V1-2"
    assert io_.fabdem_tile_name(-1, 10) == "S01E010_FABDEM_V1-2"
    assert io_.fabdem_tile_name(61, -151) == "N61W151_FABDEM_V1-2"
    assert (
        io_.copernicus_tile_name(-2, 31) == "Copernicus_DSM_COG_10_S02_00_E031_00_DEM"
    )
    # the GLO-90 release (and the gap-fill mirror) names its tiles COG_30
    assert io_.glo90_tile_name(39, 44) == "Copernicus_DSM_COG_30_N39_00_E044_00_DEM"
    assert io_.tile_corners((-150.2, 61.1, -149.8, 61.3)) == [(61, -151), (61, -150)]


def test_resolve_elevation_tiles_prefers_fabdem_then_glo30_then_glo90(
    tmp_path: Path,
) -> None:
    fab, prim, fb = (
        tmp_path / "fabdem" / "tiles",
        tmp_path / "glo30" / "tiles",
        tmp_path / "glo90" / "tiles",
    )
    for d in (fab, prim, fb):
        d.mkdir(parents=True)
    # four tiles over (31..33E, -1..1N): A has all three sources, B glo30+glo90, C glo90 only, D nothing
    (fab / "S01E031_FABDEM_V1-2.tif").touch()
    (prim / "Copernicus_DSM_COG_10_S01_00_E031_00_DEM.tif").touch()
    (fb / "Copernicus_DSM_COG_30_S01_00_E031_00_DEM.tif").touch()
    (prim / "Copernicus_DSM_COG_10_S01_00_E032_00_DEM.tif").touch()
    (fb / "Copernicus_DSM_COG_30_S01_00_E032_00_DEM.tif").touch()
    (fb / "Copernicus_DSM_COG_30_N00_00_E031_00_DEM.tif").touch()
    # a COG_10 name in the gap-fill dir is not a GLO-90 tile and must not be found
    (fb / "Copernicus_DSM_COG_10_N00_00_E032_00_DEM.tif").touch()
    paths = io_.InputPaths(
        roads_dir=tmp_path,
        water_dir=tmp_path,
        land_cover_dir=tmp_path,
        divisions_dir=tmp_path,
        elevation_dir=prim.parent,
        elevation_fallback_dir=fb.parent,
        tile_index_root=tmp_path,
        fabdem_dir=fab.parent,
    )
    tiles = io_.resolve_elevation_tiles(paths, (31.2, -0.8, 32.9, 0.7))
    assert tiles.files == [
        fab / "S01E031_FABDEM_V1-2.tif",
        prim / "Copernicus_DSM_COG_10_S01_00_E032_00_DEM.tif",
        fb / "Copernicus_DSM_COG_30_N00_00_E031_00_DEM.tif",
    ]
    assert tiles.sources == {"fabdem": 1, "glo30": 1, "glo90": 1}
    assert tiles.missing == ["Copernicus_DSM_COG_10_N00_00_E032_00_DEM"]
    # every existing file is kept by source, keyed by the tile's (lat, lon) corner
    assert tiles.by_source == {
        "fabdem": {(-1, 31): fab / "S01E031_FABDEM_V1-2.tif"},
        "glo30": {
            (-1, 31): prim / "Copernicus_DSM_COG_10_S01_00_E031_00_DEM.tif",
            (-1, 32): prim / "Copernicus_DSM_COG_10_S01_00_E032_00_DEM.tif",
        },
        "glo90": {
            (-1, 31): fb / "Copernicus_DSM_COG_30_S01_00_E031_00_DEM.tif",
            (-1, 32): fb / "Copernicus_DSM_COG_30_S01_00_E032_00_DEM.tif",
            (0, 31): fb / "Copernicus_DSM_COG_30_N00_00_E031_00_DEM.tif",
        },
    }
    # without a FABDEM dir the Copernicus chain alone serves
    no_fab = io_.InputPaths(**{**paths.__dict__, "fabdem_dir": None})
    assert (
        io_.resolve_elevation_tiles(no_fab, (31.2, -0.8, 32.9, 0.7)).sources["fabdem"]
        == 0
    )


def test_load_input_paths_reads_optional_fabdem_dir() -> None:
    base = {
        "overture_roads_dir": "/r",
        "overture_water_dir": "/w",
        "overture_land_cover_dir": "/lc/theme=base/type=land_cover",
        "elevation_dir": "/e",
        "rra_tile_index_root": "/t",
    }
    assert io_.load_input_paths(base).fabdem_dir is None
    assert io_.load_input_paths({**base, "fabdem_dir": "/f"}).fabdem_dir == Path("/f")


def test_read_geoparquet_bbox_filters_and_records_input_volume(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import box

    from idd_travel_time.friction.inputs import read_geoparquet_bbox

    geoms = [box(0, 0, 1, 1), box(10, 10, 11, 11), box(0.5, 0.5, 2, 2)]
    table = pa.table(
        {
            "id": ["a", "b", "c"],
            "bbox": [
                {
                    "xmin": g.bounds[0],
                    "ymin": g.bounds[1],
                    "xmax": g.bounds[2],
                    "ymax": g.bounds[3],
                }
                for g in geoms
            ],
            "geometry": [g.wkb for g in geoms],
        }
    )
    f = tmp_path / "layer.parquet"
    pq.write_table(table, f)
    got = read_geoparquet_bbox([f], (0.0, 0.0, 1.5, 1.5), columns=["id"])
    assert sorted(got["id"]) == ["a", "c"]
    assert got.attrs["input_volume"]["rows"] == 2
    assert got.attrs["input_volume"]["mb"] > 0
