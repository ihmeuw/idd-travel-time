"""Tests for idd_travel_time.routers.

External API calls (ORS, osmnx Overpass) are mocked at the boundary.
All tests run fully offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from idd_travel_time.constants import (
    TransitCoverage,
    TransportMode,
    TimeContext,
    VizLayer,
)
from idd_travel_time.models import TransitProfile, TravelTimeRequest
from idd_travel_time.routers._helpers import (
    effective_speed_kph,
    make_isochrone_gdf,
    make_poi_gdf,
    ors_profile_for_request,
)
from idd_travel_time.routers.base import Router
from idd_travel_time.routers.google import GoogleMapsRouter
from idd_travel_time.routers.openroute import OpenRouteServiceRouter
from idd_travel_time.routers.osmnx_router import (
    OsmnxRouter,
    _alpha_shape_or_convex_hull,
    _travel_distance_meters,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NAIROBI_LAT = -1.286389
_NAIROBI_LON = 36.817223

WALKING_REQUEST = TravelTimeRequest(
    origin_lat=_NAIROBI_LAT,
    origin_lon=_NAIROBI_LON,
    mode=TransportMode.WALKING,
    duration_minutes=30,
)

TRANSIT_REQUEST = TravelTimeRequest(
    origin_lat=_NAIROBI_LAT,
    origin_lon=_NAIROBI_LON,
    mode=TransportMode.TRANSIT,
    duration_minutes=30,
    transit_profile=TransitProfile.from_preset("informal_minibus"),
)

_SQUARE = Polygon([(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)])


# ---------------------------------------------------------------------------
# _helpers: make_isochrone_gdf
# ---------------------------------------------------------------------------


class TestMakeIsochroneGdf:
    def test_returns_geodataframe(self) -> None:
        gdf = make_isochrone_gdf(_SQUARE, WALKING_REQUEST)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_single_row(self) -> None:
        gdf = make_isochrone_gdf(_SQUARE, WALKING_REQUEST)
        assert len(gdf) == 1

    def test_crs_wgs84(self) -> None:
        gdf = make_isochrone_gdf(_SQUARE, WALKING_REQUEST)
        assert gdf.crs.to_epsg() == 4326

    def test_schema_columns(self) -> None:
        gdf = make_isochrone_gdf(_SQUARE, WALKING_REQUEST)
        assert {"geometry", "duration_min", "mode", "time_context"}.issubset(gdf.columns)

    def test_duration_value(self) -> None:
        gdf = make_isochrone_gdf(_SQUARE, WALKING_REQUEST)
        assert gdf["duration_min"].iloc[0] == 30

    def test_mode_value(self) -> None:
        gdf = make_isochrone_gdf(_SQUARE, WALKING_REQUEST)
        assert gdf["mode"].iloc[0] == "walking"


# ---------------------------------------------------------------------------
# _helpers: make_poi_gdf
# ---------------------------------------------------------------------------


class TestMakePoiGdf:
    def test_basic(self) -> None:
        gdf = make_poi_gdf([(36.8, -1.3)], ["Hospital"], ["hospital"])
        assert len(gdf) == 1
        assert gdf["name"].iloc[0] == "Hospital"

    def test_empty(self) -> None:
        gdf = make_poi_gdf([], [], [])
        assert len(gdf) == 0

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            make_poi_gdf([(0.0, 0.0)], ["A"], [])

    def test_crs_wgs84(self) -> None:
        gdf = make_poi_gdf([(36.8, -1.3)], ["X"], ["cafe"])
        assert gdf.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# _helpers: ors_profile_for_request
# ---------------------------------------------------------------------------


class TestOrsProfile:
    def test_walking(self) -> None:
        assert ors_profile_for_request(WALKING_REQUEST) == "foot-walking"

    def test_cycling(self) -> None:
        req = TravelTimeRequest(
            origin_lat=0.0, origin_lon=0.0, mode=TransportMode.CYCLING
        )
        assert ors_profile_for_request(req) == "cycling-regular"

    def test_driving(self) -> None:
        req = TravelTimeRequest(
            origin_lat=0.0, origin_lon=0.0, mode=TransportMode.DRIVING
        )
        assert ors_profile_for_request(req) == "driving-car"

    def test_transit_uses_driving_proxy(self) -> None:
        assert ors_profile_for_request(TRANSIT_REQUEST) == "driving-car"


# ---------------------------------------------------------------------------
# _helpers: effective_speed_kph
# ---------------------------------------------------------------------------


class TestEffectiveSpeed:
    def test_walking_midpoint(self) -> None:
        speed = effective_speed_kph(WALKING_REQUEST)
        assert speed == pytest.approx(4.5)

    def test_transit_uses_profile_midpoint(self) -> None:
        speed = effective_speed_kph(TRANSIT_REQUEST)
        # informal_minibus: (15, 25) → midpoint 20
        assert speed == pytest.approx(20.0)

    def test_driving(self) -> None:
        req = TravelTimeRequest(
            origin_lat=0.0, origin_lon=0.0, mode=TransportMode.DRIVING
        )
        speed = effective_speed_kph(req)
        assert speed == pytest.approx(45.0)  # (30+60)/2


# ---------------------------------------------------------------------------
# Router ABC
# ---------------------------------------------------------------------------


def test_router_is_abstract() -> None:
    with pytest.raises(TypeError):
        Router()  # type: ignore[abstract]


def test_concrete_router_must_implement_compute_isochrone() -> None:
    class Incomplete(Router):
        @property
        def name(self) -> str:
            return "incomplete"

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# GoogleMapsRouter (stub)
# ---------------------------------------------------------------------------


class TestGoogleMapsRouter:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key"):
            GoogleMapsRouter()

    def test_accepts_explicit_key(self) -> None:
        r = GoogleMapsRouter(api_key="fake-key")
        assert r.name == "google_maps"

    def test_compute_isochrone_raises_not_implemented(self) -> None:
        r = GoogleMapsRouter(api_key="fake-key")
        with pytest.raises(NotImplementedError):
            r.compute_isochrone(WALKING_REQUEST)


# ---------------------------------------------------------------------------
# OpenRouteServiceRouter
# ---------------------------------------------------------------------------


class TestOpenRouteServiceRouter:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ORS_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ORS_API_KEY"):
            OpenRouteServiceRouter()

    def test_accepts_explicit_key(self) -> None:
        with patch("openrouteservice.Client"):
            r = OpenRouteServiceRouter(api_key="fake-key")
        assert r.name == "openroute"

    def test_effective_range_seconds_walking(self) -> None:
        with patch("openrouteservice.Client"):
            r = OpenRouteServiceRouter(api_key="fake-key")
        assert r._effective_range_seconds(WALKING_REQUEST) == 30 * 60

    def test_effective_range_seconds_transit_scaled(self) -> None:
        with patch("openrouteservice.Client"):
            r = OpenRouteServiceRouter(api_key="fake-key")
        # informal_minibus midpoint=20 kph, ORS driving=30 kph → scale=2/3
        seconds = r._effective_range_seconds(TRANSIT_REQUEST)
        expected = round(30 * 60 * (20.0 / 30.0))
        assert seconds == expected

    def test_compute_isochrone_happy_path(self) -> None:
        """Mock ORS client and osmnx; verify result schema."""
        fake_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [36.80, -1.30],
                    [36.82, -1.30],
                    [36.82, -1.28],
                    [36.80, -1.28],
                    [36.80, -1.30],
                ]],
            },
            "properties": {},
        }
        fake_iso_response = {"type": "FeatureCollection", "features": [fake_feature]}

        with patch("openrouteservice.Client") as MockClient:
            instance = MockClient.return_value
            instance.isochrones.return_value = fake_iso_response

            # Suppress network/POI fetches
            with patch.object(OpenRouteServiceRouter, "_fetch_network", return_value=None), \
                 patch.object(OpenRouteServiceRouter, "_fetch_pois", return_value=None):
                r = OpenRouteServiceRouter(api_key="fake-key")
                result = r.compute_isochrone(WALKING_REQUEST)

        assert result.router_name == "openroute"
        assert len(result.isochrone_gdf) == 1
        assert result.isochrone_gdf.crs.to_epsg() == 4326
        assert result.network_gdf is None
        assert result.poi_gdf is None


# ---------------------------------------------------------------------------
# OsmnxRouter
# ---------------------------------------------------------------------------


class TestTravelDistanceMeters:
    def test_walking_30min(self) -> None:
        d = _travel_distance_meters(4.5, 30)
        assert d == pytest.approx(2250.0)

    def test_fast_30min(self) -> None:
        d = _travel_distance_meters(60.0, 30)
        assert d == pytest.approx(30_000.0)


class TestAlphaShapeOrConvexHull:
    def test_single_point_returns_buffered_polygon(self) -> None:
        gdf = gpd.GeoDataFrame(geometry=[Point(0.0, 0.0)], crs="EPSG:4326")
        poly = _alpha_shape_or_convex_hull(gdf, buffer_deg=0.001)
        assert not poly.is_empty

    def test_empty_gdf_raises(self) -> None:
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with pytest.raises(ValueError, match="No reachable nodes"):
            _alpha_shape_or_convex_hull(gdf, buffer_deg=0.001)

    def test_multiple_points_returns_union(self) -> None:
        points = [Point(i * 0.01, 0.0) for i in range(5)]
        gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
        poly = _alpha_shape_or_convex_hull(gdf, buffer_deg=0.001)
        assert poly.area > 0


class TestOsmnxRouter:
    def test_name(self) -> None:
        assert OsmnxRouter().name == "osmnx"

    def test_compute_isochrone_mocked(self) -> None:
        """Mock all osmnx calls to verify result schema without network."""
        import networkx as nx
        from shapely.geometry import Point

        # Build a minimal fake projected graph with 4 nodes in a small square
        fake_graph = nx.MultiDiGraph()
        fake_graph.graph["crs"] = "EPSG:32637"
        coords = [(0, 0), (100, 0), (100, 100), (0, 100)]
        for i, (x, y) in enumerate(coords):
            fake_graph.add_node(i, x=x, y=y)
        fake_graph.add_edge(0, 1, length=100.0)
        fake_graph.add_edge(1, 2, length=100.0)
        fake_graph.add_edge(2, 3, length=100.0)
        fake_graph.add_edge(3, 0, length=100.0)

        # Fake projected graph (same structure, projected CRS)
        fake_proj_graph = fake_graph.copy()
        fake_proj_graph.graph["crs"] = "EPSG:32637"

        fake_nodes = gpd.GeoDataFrame(
            {"x": [0, 100, 100, 0], "y": [0, 0, 100, 100]},
            geometry=[Point(x, y) for x, y in coords],
            crs="EPSG:32637",
        )
        fake_edges = gpd.GeoDataFrame(
            geometry=[Point(50, 0), Point(100, 50), Point(50, 100), Point(0, 50)],
            crs="EPSG:32637",
        )

        with patch("idd_travel_time.routers.osmnx_router.ox") as mock_ox:
            mock_ox.graph_from_point.return_value = fake_graph
            mock_ox.project_graph.return_value = fake_proj_graph
            mock_ox.nearest_nodes.return_value = 0
            mock_ox.graph_to_gdfs.return_value = (fake_nodes, fake_edges)

            with patch.object(OsmnxRouter, "_extract_pois", return_value=None):
                r = OsmnxRouter()
                result = r.compute_isochrone(WALKING_REQUEST)

        assert result.router_name == "osmnx"
        assert len(result.isochrone_gdf) == 1
        assert result.isochrone_gdf.crs.to_epsg() == 4326
