"""Tests for idd_travel_time.access.surface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from idd_travel_time.access.surface import BoundingBox, TravelTimeSurface

# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------


class TestBoundingBox:
    def test_basic_construction(self) -> None:
        bb = BoundingBox(west=36.80, south=-1.35, east=36.90, north=-1.25)
        assert bb.width_deg == pytest.approx(0.10)
        assert bb.height_deg == pytest.approx(0.10)

    def test_invalid_west_east(self) -> None:
        with pytest.raises(ValueError, match="west"):
            BoundingBox(west=37.0, south=-1.35, east=36.80, north=-1.25)

    def test_invalid_south_north(self) -> None:
        with pytest.raises(ValueError, match="south"):
            BoundingBox(west=36.80, south=-1.25, east=36.90, north=-1.35)

    def test_buffer(self) -> None:
        bb = BoundingBox(west=36.80, south=-1.35, east=36.90, north=-1.25)
        buffered = bb.buffer(0.05)
        assert buffered.west == pytest.approx(36.75)
        assert buffered.north == pytest.approx(-1.20)

    def test_contains_point_inside(self) -> None:
        bb = BoundingBox(west=36.80, south=-1.35, east=36.90, north=-1.25)
        assert bb.contains_point(lat=-1.30, lon=36.85)

    def test_contains_point_outside(self) -> None:
        bb = BoundingBox(west=36.80, south=-1.35, east=36.90, north=-1.25)
        assert not bb.contains_point(lat=-1.50, lon=36.85)

    def test_from_point(self) -> None:
        bb = BoundingBox.from_point(lat=-1.30, lon=36.85, radius_deg=0.05)
        assert bb.west == pytest.approx(36.80)
        assert bb.east == pytest.approx(36.90)

    def test_roundtrip_dict(self) -> None:
        bb = BoundingBox(west=36.80, south=-1.35, east=36.90, north=-1.25)
        assert BoundingBox.from_dict(bb.to_dict()) == bb


# ---------------------------------------------------------------------------
# TravelTimeSurface construction
# ---------------------------------------------------------------------------


def _make_surface(
    nrows: int = 10,
    ncols: int = 12,
    nan_fraction: float = 0.1,
    backend: str = "test",
) -> TravelTimeSurface:
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 60, (nrows, ncols)).astype(np.float32)
    n_nan = int(nrows * ncols * nan_fraction)
    flat = data.ravel()
    flat[:n_nan] = np.nan
    data = flat.reshape(nrows, ncols)
    return TravelTimeSurface.from_arrays(
        data=data,
        west=36.80,
        north=-1.25,
        resolution_deg=0.01,
        backend=backend,
        destination=(-1.30, 36.85),
    )


class TestTravelTimeSurfaceConstruction:
    def test_from_arrays_shape(self) -> None:
        s = _make_surface(10, 12)
        assert s.shape == (10, 12)
        assert s.nrows == 10
        assert s.ncols == 12

    def test_bbox_consistent(self) -> None:
        s = _make_surface(10, 12)
        assert s.bbox.west == pytest.approx(36.80)
        assert s.bbox.north == pytest.approx(-1.25)
        assert s.bbox.east == pytest.approx(36.80 + 12 * 0.01)
        assert s.bbox.south == pytest.approx(-1.25 - 10 * 0.01)

    def test_data_dtype_float32(self) -> None:
        s = _make_surface()
        assert s.data.dtype == np.float32

    def test_repr(self) -> None:
        s = _make_surface()
        r = repr(s)
        assert "TravelTimeSurface" in r
        assert "osmnx" in r or "test" in r


# ---------------------------------------------------------------------------
# TravelTimeSurface.summary_stats
# ---------------------------------------------------------------------------


class TestSummaryStats:
    def test_keys_present(self) -> None:
        s = _make_surface()
        stats = s.summary_stats()
        for k in ("count", "min", "max", "mean", "median", "nan_fraction"):
            assert k in stats

    def test_nan_fraction(self) -> None:
        s = _make_surface(nan_fraction=0.2)
        stats = s.summary_stats()
        assert stats["nan_fraction"] == pytest.approx(0.2, abs=0.05)

    def test_all_nan_surface(self) -> None:
        data = np.full((5, 5), np.nan, dtype=np.float32)
        s = TravelTimeSurface.from_arrays(
            data=data,
            west=0.0,
            north=0.0,
            resolution_deg=0.01,
            backend="test",
            destination=(0.0, 0.0),
        )
        stats = s.summary_stats()
        assert stats["count"] == 0
        assert stats["nan_fraction"] == pytest.approx(1.0)

    def test_coverage_at(self) -> None:
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        s = TravelTimeSurface.from_arrays(
            data=data,
            west=0.0,
            north=0.0,
            resolution_deg=0.01,
            backend="test",
            destination=(0.0, 0.0),
        )
        # 31 pixels (0–30) out of 100
        assert s.coverage_at(30.0) == pytest.approx(31 / 100)


# ---------------------------------------------------------------------------
# TravelTimeSurface.threshold
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_threshold_shape(self) -> None:
        s = _make_surface(10, 10, nan_fraction=0.0)
        mask = s.threshold(30.0)
        assert mask.shape == s.shape
        assert mask.dtype == bool

    def test_nan_cells_are_false(self) -> None:
        s = _make_surface(10, 10, nan_fraction=0.5)
        mask = s.threshold(999.0)
        # NaN pixels must be False even with a generous threshold
        assert not mask[np.isnan(s.data)].any()


# ---------------------------------------------------------------------------
# TravelTimeSurface.clip
# ---------------------------------------------------------------------------


class TestClip:
    def test_clipped_smaller(self) -> None:
        s = _make_surface(20, 20, nan_fraction=0.0)
        inner = BoundingBox(west=36.82, south=-1.43, east=36.88, north=-1.27)
        clipped = s.clip(inner)
        assert clipped.nrows <= s.nrows
        assert clipped.ncols <= s.ncols

    def test_clipped_bbox_within_original(self) -> None:
        s = _make_surface(20, 20, nan_fraction=0.0)
        inner = BoundingBox(west=36.82, south=-1.43, east=36.88, north=-1.27)
        clipped = s.clip(inner)
        assert clipped.bbox.west >= s.bbox.west
        assert clipped.bbox.east <= s.bbox.east

    def test_clip_preserves_resolution(self) -> None:
        s = _make_surface()
        inner = BoundingBox(west=36.82, south=-1.34, east=36.88, north=-1.27)
        clipped = s.clip(inner)
        assert clipped.resolution_deg == pytest.approx(s.resolution_deg)


# ---------------------------------------------------------------------------
# TravelTimeSurface.resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_coarser_resolution_smaller_array(self) -> None:
        s = _make_surface(40, 40, nan_fraction=0.0)
        coarse = s.resample(0.02)  # 2× coarser
        assert coarse.nrows < s.nrows
        assert coarse.ncols < s.ncols

    def test_finer_resolution_larger_array(self) -> None:
        s = _make_surface(10, 10, nan_fraction=0.0)
        fine = s.resample(0.005)  # 2× finer
        assert fine.nrows > s.nrows

    def test_same_resolution_same_shape(self) -> None:
        s = _make_surface()
        same = s.resample(0.01)
        assert same.shape == s.shape


# ---------------------------------------------------------------------------
# TravelTimeSurface.pixel_coords
# ---------------------------------------------------------------------------


class TestPixelCoords:
    def test_shape(self) -> None:
        s = _make_surface(8, 10)
        lats, lons = s.pixel_coords()
        assert lats.shape == (8, 10)
        assert lons.shape == (8, 10)

    def test_coords_within_bbox(self) -> None:
        s = _make_surface(8, 10)
        lats, lons = s.pixel_coords()
        assert lons.min() >= s.bbox.west
        assert lons.max() <= s.bbox.east
        assert lats.min() >= s.bbox.south
        assert lats.max() <= s.bbox.north


# ---------------------------------------------------------------------------
# TravelTimeSurface.to_geodataframe
# ---------------------------------------------------------------------------


class TestToGeoDataFrame:
    def test_returns_geodataframe(self) -> None:
        import geopandas as gpd

        s = _make_surface(5, 5, nan_fraction=0.0)
        gdf = s.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_only_valid_pixels(self) -> None:
        s = _make_surface(5, 5, nan_fraction=0.5)
        gdf = s.to_geodataframe()
        int(~np.isnan(s.data).sum() == gdf.__len__())
        # all rows are non-NaN
        assert not gdf["travel_time_min"].isna().any()

    def test_crs_wgs84(self) -> None:
        s = _make_surface(5, 5)
        gdf = s.to_geodataframe()
        assert gdf.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# GeoTIFF round-trip
# ---------------------------------------------------------------------------


class TestGeoTIFFRoundtrip:
    def test_saves_and_loads(self, tmp_path: Path) -> None:
        s = _make_surface(10, 12)
        p = s.to_geotiff(tmp_path / "surface.tif")
        assert p.exists()
        loaded = TravelTimeSurface.from_geotiff(p)
        assert loaded.shape == s.shape

    def test_data_preserved(self, tmp_path: Path) -> None:
        s = _make_surface(10, 12, nan_fraction=0.0)
        p = s.to_geotiff(tmp_path / "surface.tif")
        loaded = TravelTimeSurface.from_geotiff(p)
        np.testing.assert_allclose(loaded.data, s.data, rtol=1e-4)

    def test_nan_preserved(self, tmp_path: Path) -> None:
        s = _make_surface(10, 12, nan_fraction=0.3)
        p = s.to_geotiff(tmp_path / "surface.tif")
        loaded = TravelTimeSurface.from_geotiff(p)
        orig_nan = np.isnan(s.data)
        load_nan = np.isnan(loaded.data)
        np.testing.assert_array_equal(orig_nan, load_nan)

    def test_sidecar_json_written(self, tmp_path: Path) -> None:
        s = _make_surface()
        p = s.to_geotiff(tmp_path / "surface.tif")
        sidecar = p.with_suffix(".json")
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert "backend" in meta
        assert "destination" in meta

    def test_backend_and_destination_roundtrip(self, tmp_path: Path) -> None:
        s = _make_surface(backend="my_backend")
        p = s.to_geotiff(tmp_path / "surface.tif")
        loaded = TravelTimeSurface.from_geotiff(p)
        assert loaded.backend == "my_backend"
        assert loaded.destination == pytest.approx(s.destination)
