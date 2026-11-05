"""Rasterisers and composition on a tiny synthetic block."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString, box

from idd_travel_time.friction import build as bd
from idd_travel_time.friction import rasters as rs
from idd_travel_time.friction.frame import FRAME_CRS, BlockGrid
from idd_travel_time.friction.params import FrictionParams, load_params

# 10 x 10 cells of 40 m at the equator (x from 0, y from 0 to 400)
GRID = BlockGrid("T", 40.0, 0.0, 0.0, 400.0, 400.0)


@pytest.fixture(scope="module")
def params() -> FrictionParams:
    return load_params()


def _gdf(geoms: list[Any], **cols: list[Any]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(cols, geometry=geoms, crs=FRAME_CRS)


def test_rasterize_fastest_takes_max_and_counts_overlap() -> None:
    a = box(0, 0, 200, 400)  # left half
    b = box(100, 0, 400, 400)  # right 3/4, overlaps the middle quarter
    count = np.zeros(GRID.shape, dtype=np.uint8)
    out = rs.rasterize_fastest(
        GRID,
        [
            (2.0, gpd.GeoSeries([a], crs=FRAME_CRS)),
            (5.0, gpd.GeoSeries([b], crs=FRAME_CRS)),
        ],
        all_touched=False,
        count=count,
    )
    assert out[5, 1] == 2.0  # only a
    assert out[5, 3] == 5.0  # overlap -> fastest
    assert out[5, 8] == 5.0  # only b
    assert count[5, 3] == 2
    assert count[5, 1] == 1
    assert np.isnan(out).sum() == 0


def test_land_cover_speed_rejects_unknown_subtype(params: FrictionParams) -> None:
    lc = _gdf([box(0, 0, 400, 400)], subtype=["lava"])
    with pytest.raises(ValueError, match="lava"):
        rs.land_cover_speed(GRID, lc, params)


def test_road_speed_uses_country_lookup_and_walking_cap(params: FrictionParams) -> None:
    roads = {c: _gdf([], id=[]) for c in params.road_crosswalk}
    roads["motorway"] = _gdf(
        [LineString([(0, 200), (400, 200)])], id=["m"]
    )  # across row 4/5
    roads["track"] = _gdf(
        [LineString([(200, 0), (200, 400)])], id=["t"]
    )  # down column 5
    countries = _gdf([box(0, 0, 400, 400)], country=["UG"], iso3=["UGA"])
    idx, iso3s = rs.country_raster(GRID, countries)
    assert iso3s == ["UGA"]
    assert idx.min() == 1
    motor = rs.road_speed(GRID, roads, idx, iso3s, params)
    assert motor[4, 1] == 100.0 or motor[5, 1] == 100.0  # Uganda motorway
    assert np.nanmax(motor[:, 5]) == 100.0  # crossing cell: fastest wins
    assert motor[1, 5] == 30.0  # Uganda track
    walk = rs.road_speed(GRID, roads, idx, iso3s, params, walking_kph=5.0)
    assert np.nanmax(walk) == 5.0
    assert np.isfinite(walk).sum() == np.isfinite(motor).sum()
    # no country polygon -> global default (105 motorway)
    none_idx = np.zeros(GRID.shape, dtype=np.int32)
    assert np.nanmax(rs.road_speed(GRID, roads, none_idx, [], params)) == 105.0


def test_water_speed_by_mode(params: FrictionParams) -> None:
    water = {c: _gdf([], id=[]) for c in params.water}
    water["inland_water"] = _gdf([box(0, 0, 200, 200)], id=["lake"])
    water["coastal_ocean"] = _gdf([box(200, 200, 400, 400)], id=["sea"])
    water["stream_water"] = _gdf([LineString([(0, 300), (200, 300)])], id=["s"])
    motor, mask = rs.water_speed(GRID, water, params, mode="motorised")
    assert motor[7, 2] == 10.0  # lake (rows count from the top)
    assert motor[2, 7] == 19.0  # ocean
    assert np.isnan(motor[2, 2])  # stream ignored
    assert not mask.any()
    walk, mask = rs.water_speed(GRID, water, params, mode="walking")
    assert walk[7, 2] == 1.0
    assert np.isnan(walk[2, 7])
    assert mask[2, 7]
    assert not mask[7, 2]


def test_compose_mode_layer_precedence(params: FrictionParams) -> None:
    data = bd.BlockInputs(
        land_cover=_gdf([box(0, 0, 400, 400)], subtype=["grass"]),
        roads={c: _gdf([], id=[]) for c in params.road_crosswalk},
        water={c: _gdf([], id=[]) for c in params.water},
        countries=_gdf([box(0, 0, 400, 400)], country=["UG"], iso3=["UGA"]),
        elevation=np.zeros(GRID.shape, dtype=np.float32),
    )
    data.water["inland_water"] = _gdf([box(0, 0, 200, 200)], id=["lake"])
    data.roads["primary"] = _gdf(
        [LineString([(0, 100), (400, 100)])], id=["p"]
    )  # through the lake
    stats: dict[str, Any] = {}
    off, has_lc = bd.offroad_speed(GRID, data, params, stats)
    assert has_lc.all()
    assert off[2, 8] == pytest.approx(4.86 * 1.016, abs=1e-3)  # grass, flat, sea level
    walk = bd.compose_mode(GRID, data, params, off, "walking", stats, has_lc)
    assert walk[8, 1] == 1.0  # water replaces grass on the walking surface
    assert walk[7, 1] == 5.0  # road replaces water (bridge), capped to walking
    motor = bd.compose_mode(GRID, data, params, off, "motorised", stats, has_lc)
    assert motor[8, 1] == 10.0
    assert motor[7, 1] == 50.0  # Uganda primary
    assert stats["walking_cells_road"] == stats["motorised_cells_road"] > 0
    assert stats["walking_cells_nodata_unexpected"] == 0
    assert stats["motorised_cells_nodata_unexpected"] == 0
    bd.check_unexpected_nodata("T", stats, bd.MODES)  # nothing to refuse
    f = bd.speed_kph_to_friction(np.array([60.0, np.nan], dtype=np.float32))
    assert f[0] == pytest.approx(0.001)
    assert np.isnan(f[1])


def test_missing_elevation_gives_nan_friction_and_is_refused(
    params: FrictionParams,
) -> None:
    # grass everywhere, a road down column 5, no elevation on rows 0-3
    elevation = np.zeros(GRID.shape, dtype=np.float32)
    elevation[:4, :] = np.nan
    data = bd.BlockInputs(
        land_cover=_gdf([box(0, 0, 400, 400)], subtype=["grass"]),
        roads={c: _gdf([], id=[]) for c in params.road_crosswalk},
        water={c: _gdf([], id=[]) for c in params.water},
        countries=_gdf([box(0, 0, 400, 400)], country=["UG"], iso3=["UGA"]),
        elevation=elevation,
    )
    data.roads["primary"] = _gdf([LineString([(200, 0), (200, 400)])], id=["p"])
    stats: dict[str, Any] = {}
    off, has_lc = bd.offroad_speed(GRID, data, params, stats)
    # (a) no elevation -> no slope or altitude factor -> NaN, never a flat-ground guess
    assert np.isnan(off[:4, :]).all()
    assert np.isfinite(off[5:, :]).all()  # row 4 borders the gap: its gradient is NaN
    assert stats["cells_elevation_missing"] == 40
    walk = bd.compose_mode(GRID, data, params, off, "walking", stats, has_lc)
    # a road has its own speed, elevation or not; the grass cells beside it are NaN
    assert np.isfinite(walk[:4, 5]).all()
    assert np.isnan(walk[:4, 0]).all()
    # (b) those cells have land cover and still no value: counted and refused
    n_road_rows = int(np.isfinite(walk[:5, :]).sum())
    assert stats["walking_cells_nodata_unexpected"] == 50 - n_road_rows
    assert bd.unexpected_nodata(stats, ("walking",)) == {"walking": 50 - n_road_rows}
    with pytest.raises(bd.UnexpectedNoDataError, match="T: cells with land cover"):
        bd.check_unexpected_nodata("T", stats, ("walking",))
    # without the flag the count is 0 and nothing is refused (the stat is unavailable)
    legacy: dict[str, Any] = {}
    bd.compose_mode(GRID, data, params, off, "walking", legacy)
    assert legacy["walking_cells_nodata_unexpected"] == 0
    assert bd.unexpected_nodata(legacy, ("walking",)) == {}


def test_unexpected_nodata_report_reads_block_stats(tmp_path: Path) -> None:
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    modes = ("motorised", "walking")
    checked_ok = {f"{m}_cells_nodata_unexpected": 0 for m in modes}
    checked_bad = {
        "motorised_cells_nodata_unexpected": 0,
        "walking_cells_nodata_unexpected": 3,
    }
    legacy = {"walking_cells_nodata": 5}  # built before the check existed
    for key, payload in (
        ("B-0001X-0001Y", checked_ok),
        ("B-0002X-0001Y", checked_bad),
        ("B-0003X-0001Y", legacy),
    ):
        (stats_dir / f"{key}.json").write_text(json.dumps(payload))
    report = bd.unexpected_nodata_report(
        tmp_path,
        ["B-0001X-0001Y", "B-0002X-0001Y", "B-0003X-0001Y", "B-0004X-0001Y"],
        modes,
    )
    assert report == {"unexpected": ["B-0002X-0001Y"], "unchecked": ["B-0003X-0001Y"]}


def test_compose_mode_rejects_unknown_mode(params: FrictionParams) -> None:
    data = bd.BlockInputs(
        _gdf([], subtype=[]),
        {},
        {},
        _gdf([], country=[], iso3=[]),
        np.zeros(GRID.shape, np.float32),
    )
    with pytest.raises(ValueError, match="unknown mode"):
        bd.compose_mode(
            GRID, data, params, np.zeros(GRID.shape, np.float32), "flying", {}
        )


def test_write_raster_is_atomic_and_readable(tmp_path: Path) -> None:
    import rasterio

    arr = np.full(GRID.shape, 0.012, dtype=np.float32)
    out = tmp_path / "m" / "T.tif"
    bd.write_raster(out, arr, GRID)
    assert out.exists()
    assert not list(out.parent.glob("T.tif.tmp-*"))
    with rasterio.open(out) as src:
        assert src.shape == GRID.shape
        assert src.transform == GRID.transform
        assert np.allclose(src.read(1), 0.012)


def test_clip_to_grid_drops_outside_and_keeps_margin() -> None:
    far = box(5000, 5000, 6000, 6000)  # entirely outside the 400 m block
    near = box(-50, -50, 100, 100)  # straddles the edge; margin keeps the overlap
    g = _gdf([far, near], subtype=["grass", "grass"])
    c = rs.clip_to_grid(g, GRID)
    assert len(c) == 1
    assert c.geometry.iloc[0].bounds[0] >= -rs.CLIP_MARGIN_M
    assert c.geometry.iloc[0].bounds[2] == 100.0


def test_use_system_memory_pool_switches_arrow_allocator() -> None:
    assert bd.use_system_memory_pool() == "system"


def test_write_raster_registers_in_sidecar_and_block_done_sees_it(
    tmp_path: Path,
) -> None:
    from idd_tools.jobmon.atomic_io import AtomicRegistry

    arr = np.ones(GRID.shape, dtype=np.float32)
    out_dir = tmp_path / "slot"
    with AtomicRegistry(out_dir, task_id="t1", task_template="friction_block") as reg:
        bd.write_raster(out_dir / "walking" / "B-1.tif", arr, GRID, registry=reg)
        bd.write_raster(out_dir / "motorised" / "B-1.tif", arr, GRID, registry=reg)
        assert not bd.block_done(out_dir, "B-1")  # stats json still missing
        bd.write_json(
            out_dir / "stats" / "B-1.json", {"block_key": "B-1"}, registry=reg
        )
    assert bd.block_done(out_dir, "B-1")
    sidecar = json.loads((out_dir / ".atomic" / "t1.json").read_text())
    registered = json.dumps(sidecar)
    assert "walking/B-1.tif" in registered
    assert "stats/B-1.json" in registered
    assert not list((out_dir / "walking").glob("*.tmp"))


# ----------------------------------------------------------------------------
# Instrumentation: per-step seconds, input volume, the process peak
# ----------------------------------------------------------------------------


def test_timed_accumulates_seconds_under_a_label() -> None:
    steps: dict[str, float] = {}
    with bd.timed(steps, "a"):
        pass
    with bd.timed(steps, "a"):
        pass
    with bd.timed(steps, "b"):
        pass
    assert set(steps) == {"a", "b"}
    assert steps["a"] >= 0.0
    msg = "x"
    with pytest.raises(RuntimeError):  # the step is recorded even when the body raises
        with bd.timed(steps, "c"):
            raise RuntimeError(msg)
    assert "c" in steps


def test_layer_volume_reads_attrs_and_sums_classes() -> None:
    g1 = _gdf([box(0, 0, 1, 1)], id=[1])
    g1.attrs["input_volume"] = {"rows": 3, "mb": 1.5}
    g2 = _gdf([box(0, 0, 1, 1)], id=[2])
    g2.attrs["input_volume"] = {"rows": 2, "mb": 0.25}
    assert bd.layer_volume(g1) == {"rows": 3, "mb": 1.5}
    total = bd.layer_volume({"a": g1, "b": g2})
    assert total["rows"] == 5
    assert total["mb"] == pytest.approx(1.75)
    assert total["classes"]["b"] == {"rows": 2, "mb": 0.25}
    bare = _gdf([box(0, 0, 1, 1)], id=[3])  # a frame that did not come from the reader
    assert bd.layer_volume(bare) == {"rows": 0, "mb": 0.0}


def test_process_peak_gib_is_positive_and_monotone() -> None:
    before = bd.process_peak_gib()
    assert before > 0
    _ballast = np.ones(32 * 1024 * 1024, dtype=np.uint8)  # 32 MiB touched
    assert bd.process_peak_gib() >= before
