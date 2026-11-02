"""Block grid geometry on a synthetic two-block tile index."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely import box

from idd_travel_time.friction import frame as fr

RES = 40.0
TILE = fr.TILE_CELLS * RES  # 20480 m


@pytest.fixture
def tile_index() -> gpd.GeoDataFrame:
    rows = []
    for bx, by in [(0, 0), (1, 0)]:
        for ti in range(fr.TILES_PER_BLOCK):
            for tj in range(fr.TILES_PER_BLOCK):
                xi, yi = bx * fr.TILES_PER_BLOCK + ti, by * fr.TILES_PER_BLOCK + tj
                x0 = 3_000_000 + xi * TILE
                y1 = 500_000 - yi * TILE
                rows.append(
                    (
                        f"B-{bx:04d}X-{by:04d}Y",
                        f"T-{xi:04d}X-{yi:04d}Y",
                        box(x0, y1 - TILE, x0 + TILE, y1),
                    )
                )
    return gpd.GeoDataFrame(
        rows, columns=["block_key", "tile_key", "geometry"], crs=fr.FRAME_CRS
    )


def test_block_grid_shape_transform_and_bounds(tile_index: gpd.GeoDataFrame) -> None:
    g = fr.block_grid(tile_index, "B-0001X-0000Y", RES)
    assert g.shape == (fr.BLOCK_CELLS, fr.BLOCK_CELLS)
    assert g.xmin == pytest.approx(3_000_000 + 16 * TILE)
    assert g.ymax == pytest.approx(500_000)
    t = g.transform
    assert (t.a, t.e) == (RES, -RES)
    assert (t.c, t.f) == (g.xmin, g.ymax)


def test_unknown_block_raises(tile_index: gpd.GeoDataFrame) -> None:
    with pytest.raises(KeyError):
        fr.block_grid(tile_index, "B-9999X-9999Y", RES)


def test_ground_lengths_keep_cell_area(tile_index: gpd.GeoDataFrame) -> None:
    g = fr.block_grid(tile_index, "B-0000X-0000Y", RES)
    dx, dy = g.ground_lengths()
    assert dx.shape == (fr.BLOCK_CELLS,)
    assert np.allclose(dx * dy, RES * RES)
    lats = g.row_latitudes()
    assert np.all(np.diff(lats) < 0)  # rows go south
    assert (
        dx[0] < RES < dy[0]
    )  # north of the equator: narrower east-west, taller north-south


def test_lonlat_bounds_contain_centre(tile_index: gpd.GeoDataFrame) -> None:
    g = fr.block_grid(tile_index, "B-0000X-0000Y", RES)
    lo_lon, lo_lat, hi_lon, hi_lat = g.lonlat_bounds(pad_deg=0.1)
    assert lo_lon < hi_lon
    assert lo_lat < hi_lat
    assert (
        fr.block_for_point(tile_index, (lo_lon + hi_lon) / 2, (lo_lat + hi_lat) / 2)
        == "B-0000X-0000Y"
    )


def test_land_block_keys(tile_index: gpd.GeoDataFrame) -> None:
    land = gpd.GeoDataFrame(
        geometry=[box(3_000_000 + 100, 400_000, 3_000_000 + 200, 400_100)],
        crs=fr.FRAME_CRS,
    )
    assert fr.land_block_keys(tile_index, land) == ["B-0000X-0000Y"]


def test_clipped_edge_block_rounds_up_to_whole_cells() -> None:
    # the frame's bottom row: 6898.267 cells tall at 40 m
    g = fr.BlockGrid("edge", RES, 0.0, -275930.66, 327680.0, 0.0)
    assert g.shape == (6899, 8192)
    assert g.clipped is True
    assert g.transform.f == 0.0  # anchored at the top-left corner
    full = fr.BlockGrid("full", RES, 0.0, -327680.0, 327680.0, 0.0)
    assert full.shape == (8192, 8192)
    assert full.clipped is False


def test_edge_block_rounds_cells_up_and_stays_on_lattice() -> None:
    # a block clipped at the world's south edge: 8192 x 8191.6 cells of 40 m
    g = fr.BlockGrid("E", RES, 0.0, 500_000.0 - 8191.6 * RES, 8192 * RES, 500_000.0)
    assert g.shape == (8192, 8192)
    assert g.raster_bounds[1] == pytest.approx(
        500_000.0 - 8192 * RES
    )  # overhangs the clipped bound
    assert g.transform.f == 500_000.0


def test_block_grid_rejects_top_left_off_lattice(tile_index: gpd.GeoDataFrame) -> None:
    shifted = tile_index.copy()
    one = shifted["block_key"] == "B-0001X-0000Y"
    shifted.loc[one, "geometry"] = shifted.loc[one, "geometry"].translate(xoff=7.0)
    with pytest.raises(ValueError, match="off the frame lattice"):
        fr.block_grid(shifted, "B-0001X-0000Y", RES)
