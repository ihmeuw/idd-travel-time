"""Tests for idd_travel_time.io.parquet."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from idd_travel_time.constants import (
    TimeContext,
    TransitCoverage,
    TransportMode,
    VizLayer,
)
from idd_travel_time.io import read_result, result_id, write_result
from idd_travel_time.models import TransitProfile, TravelTimeRequest, TravelTimeResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SQUARE = Polygon([
    (36.80, -1.30), (36.82, -1.30),
    (36.82, -1.28), (36.80, -1.28),
    (36.80, -1.30),
])

_WALKING_REQUEST = TravelTimeRequest(
    origin_lat=-1.286389,
    origin_lon=36.817223,
    mode=TransportMode.WALKING,
    duration_minutes=30,
    time_context=TimeContext.OFF_PEAK,
)

_TRANSIT_REQUEST = TravelTimeRequest(
    origin_lat=6.5244,
    origin_lon=3.3792,
    mode=TransportMode.TRANSIT,
    duration_minutes=45,
    time_context=TimeContext.RUSH_HOUR,
    transit_profile=TransitProfile.from_preset(
        "informal_minibus",
        time_context=TimeContext.RUSH_HOUR,
        typical_speed_kph=(10.0, 18.0),
    ),
)


def _make_iso_gdf(request: TravelTimeRequest) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "geometry": [_SQUARE],
            "duration_min": [request.duration_minutes],
            "mode": [request.mode.value],
            "time_context": [request.time_context.value],
        },
        crs="EPSG:4326",
    )


def _make_network_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[
            LineString([(36.80, -1.30), (36.81, -1.29)]),
            LineString([(36.81, -1.29), (36.82, -1.28)]),
        ],
        crs="EPSG:4326",
    )


def _make_poi_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "geometry": [Point(36.812, -1.288), Point(36.815, -1.285)],
            "name": ["Nairobi Hospital", "KFC"],
            "amenity": ["hospital", "fast_food"],
        },
        crs="EPSG:4326",
    )


def _make_result(
    request: TravelTimeRequest = _WALKING_REQUEST,
    network: bool = False,
    pois: bool = False,
) -> TravelTimeResult:
    return TravelTimeResult(
        request=request,
        isochrone_gdf=_make_iso_gdf(request),
        network_gdf=_make_network_gdf() if network else None,
        poi_gdf=_make_poi_gdf() if pois else None,
        router_name="osmnx",
    )


# ---------------------------------------------------------------------------
# result_id
# ---------------------------------------------------------------------------


class TestResultId:
    def test_deterministic(self) -> None:
        assert result_id(_WALKING_REQUEST) == result_id(_WALKING_REQUEST)

    def test_contains_mode(self) -> None:
        assert "walking" in result_id(_WALKING_REQUEST)

    def test_contains_duration(self) -> None:
        assert "30min" in result_id(_WALKING_REQUEST)

    def test_contains_time_context(self) -> None:
        assert "off_peak" in result_id(_WALKING_REQUEST)

    def test_negative_lat_safe(self) -> None:
        rid = result_id(_WALKING_REQUEST)
        assert "-" not in rid

    def test_different_requests_different_ids(self) -> None:
        req2 = TravelTimeRequest(
            origin_lat=0.0, origin_lon=0.0, mode=TransportMode.DRIVING
        )
        assert result_id(_WALKING_REQUEST) != result_id(req2)

    def test_transit_includes_speed_in_id(self) -> None:
        rid = result_id(_TRANSIT_REQUEST)
        assert "transit" in rid
        assert "10.0" in rid or "18.0" in rid

    def test_no_filesystem_unsafe_chars(self) -> None:
        rid = result_id(_WALKING_REQUEST)
        for ch in r'\/:*?"<>|':
            assert ch not in rid


# ---------------------------------------------------------------------------
# write_result / read_result — walking (no network, no POIs)
# ---------------------------------------------------------------------------


class TestWriteReadMinimal:
    def test_returns_result_dir_path(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        assert isinstance(out, Path)
        assert out.is_dir()

    def test_result_dir_under_output_dir(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        assert out.parent == tmp_path

    def test_result_dir_name_matches_result_id(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        assert out.name == result_id(_WALKING_REQUEST)

    def test_meta_json_exists(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        assert (out / "meta.json").exists()

    def test_isochrone_parquet_exists(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        assert (out / "isochrone.parquet").exists()

    def test_network_parquet_absent_when_none(self, tmp_path: Path) -> None:
        result = _make_result(network=False)
        out = write_result(result, tmp_path)
        assert not (out / "network.parquet").exists()

    def test_poi_parquet_absent_when_none(self, tmp_path: Path) -> None:
        result = _make_result(pois=False)
        out = write_result(result, tmp_path)
        assert not (out / "poi.parquet").exists()

    def test_roundtrip_isochrone_geometry(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.isochrone_gdf.geometry.iloc[0].equals(
            result.isochrone_gdf.geometry.iloc[0]
        )

    def test_roundtrip_isochrone_crs(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.isochrone_gdf.crs.to_epsg() == 4326

    def test_roundtrip_request_lat_lon(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.request.origin_lat == pytest.approx(_WALKING_REQUEST.origin_lat)
        assert loaded.request.origin_lon == pytest.approx(_WALKING_REQUEST.origin_lon)

    def test_roundtrip_request_mode(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.request.mode is TransportMode.WALKING

    def test_roundtrip_request_duration(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.request.duration_minutes == 30

    def test_roundtrip_router_name(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.router_name == "osmnx"

    def test_roundtrip_network_none(self, tmp_path: Path) -> None:
        result = _make_result(network=False)
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.network_gdf is None

    def test_roundtrip_poi_none(self, tmp_path: Path) -> None:
        result = _make_result(pois=False)
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.poi_gdf is None


# ---------------------------------------------------------------------------
# write_result / read_result — with network and POIs
# ---------------------------------------------------------------------------


class TestWriteReadFull:
    def test_network_parquet_exists(self, tmp_path: Path) -> None:
        result = _make_result(network=True)
        out = write_result(result, tmp_path)
        assert (out / "network.parquet").exists()

    def test_poi_parquet_exists(self, tmp_path: Path) -> None:
        result = _make_result(pois=True)
        out = write_result(result, tmp_path)
        assert (out / "poi.parquet").exists()

    def test_roundtrip_network_row_count(self, tmp_path: Path) -> None:
        result = _make_result(network=True)
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert len(loaded.network_gdf) == len(result.network_gdf)

    def test_roundtrip_poi_names(self, tmp_path: Path) -> None:
        result = _make_result(pois=True)
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert list(loaded.poi_gdf["name"]) == ["Nairobi Hospital", "KFC"]

    def test_roundtrip_poi_crs(self, tmp_path: Path) -> None:
        result = _make_result(pois=True)
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.poi_gdf.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# Transit request roundtrip
# ---------------------------------------------------------------------------


class TestTransitRoundtrip:
    def test_transit_profile_speed_preserved(self, tmp_path: Path) -> None:
        result = _make_result(request=_TRANSIT_REQUEST)
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.request.transit_profile is not None
        assert loaded.request.transit_profile.typical_speed_kph == pytest.approx(
            (10.0, 18.0)
        )

    def test_transit_profile_coverage_preserved(self, tmp_path: Path) -> None:
        result = _make_result(request=_TRANSIT_REQUEST)
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.request.transit_profile.coverage is TransitCoverage.ROAD_NETWORK

    def test_transit_profile_time_context_preserved(self, tmp_path: Path) -> None:
        result = _make_result(request=_TRANSIT_REQUEST)
        out = write_result(result, tmp_path)
        loaded = read_result(out)
        assert loaded.request.transit_profile.time_context is TimeContext.RUSH_HOUR


# ---------------------------------------------------------------------------
# Idempotency and overwrite
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_second_write_overwrites_first(self, tmp_path: Path) -> None:
        result = _make_result()
        write_result(result, tmp_path)
        out2 = write_result(result, tmp_path)
        assert out2.is_dir()
        # Still only one result directory
        children = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(children) == 1

    def test_meta_schema_version_present(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        meta = json.loads((out / "meta.json").read_text())
        assert meta["schema_version"] == "1"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_read_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_result(tmp_path / "does_not_exist")

    def test_read_missing_meta_raises(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        (out / "meta.json").unlink()
        with pytest.raises(FileNotFoundError, match="meta.json"):
            read_result(out)

    def test_read_wrong_schema_version_raises(self, tmp_path: Path) -> None:
        result = _make_result()
        out = write_result(result, tmp_path)
        meta_path = out / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["schema_version"] = "99"
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="schema version"):
            read_result(out)

    def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        result = _make_result()
        new_dir = tmp_path / "deep" / "nested" / "dir"
        out = write_result(result, new_dir)
        assert out.is_dir()


# ---------------------------------------------------------------------------
# TravelTimeResult convenience methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    def test_to_parquet_returns_path(self, tmp_path: Path) -> None:
        result = _make_result()
        out = result.to_parquet(tmp_path)
        assert isinstance(out, Path)
        assert out.is_dir()

    def test_from_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = _make_result(network=True, pois=True)
        out = result.to_parquet(tmp_path)
        loaded = TravelTimeResult.from_parquet(out)
        assert loaded.router_name == result.router_name
        assert loaded.request.mode is result.request.mode
        assert loaded.network_gdf is not None
        assert loaded.poi_gdf is not None

    def test_from_parquet_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TravelTimeResult.from_parquet(tmp_path / "no_such_dir")
