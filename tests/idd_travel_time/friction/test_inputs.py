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


def test_find_elevation_tiles_prefers_primary_then_fallback(tmp_path: Path) -> None:
    prim, fb = tmp_path / "glo30" / "tiles", tmp_path / "glo90" / "tiles"
    prim.mkdir(parents=True)
    fb.mkdir(parents=True)
    (prim / "A.tif").touch()
    (fb / "A.tif").touch()
    (fb / "B.tif").touch()
    paths = io_.InputPaths(
        roads_dir=tmp_path,
        water_dir=tmp_path,
        land_cover_dir=tmp_path,
        divisions_dir=tmp_path,
        elevation_dir=prim.parent,
        elevation_fallback_dir=fb.parent,
        tile_index_root=tmp_path,
    )
    files, missing = io_.find_elevation_tiles(paths, ["A", "B", "C"])
    assert files == [prim / "A.tif", fb / "B.tif"]
    assert missing == ["C"]


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
    (fb / "Copernicus_DSM_COG_10_S01_00_E031_00_DEM.tif").touch()
    (prim / "Copernicus_DSM_COG_10_S01_00_E032_00_DEM.tif").touch()
    (fb / "Copernicus_DSM_COG_10_S01_00_E032_00_DEM.tif").touch()
    (fb / "Copernicus_DSM_COG_10_N00_00_E031_00_DEM.tif").touch()
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
        fb / "Copernicus_DSM_COG_10_N00_00_E031_00_DEM.tif",
    ]
    assert tiles.sources == {"fabdem": 1, "glo30": 1, "glo90": 1}
    assert tiles.missing == ["Copernicus_DSM_COG_10_N00_00_E032_00_DEM"]
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
