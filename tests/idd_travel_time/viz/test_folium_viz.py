"""Tests for idd_travel_time.viz.folium_viz."""

from __future__ import annotations

import tempfile
from pathlib import Path

import folium
import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from idd_travel_time.constants import TransportMode, VizLayer
from idd_travel_time.models import TravelTimeRequest, TravelTimeResult
from idd_travel_time.viz.folium_viz import (
    _resolve_layers,
    _sample_polygon_points,
    render,
    save,
)

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
)


def _make_result(
    network: bool = False,
    pois: bool = False,
    request: TravelTimeRequest | None = None,
) -> TravelTimeResult:
    req = request or _WALKING_REQUEST
    iso_gdf = gpd.GeoDataFrame(
        {
            "geometry": [_SQUARE],
            "duration_min": [30],
            "mode": ["walking"],
            "time_context": ["off_peak"],
        },
        crs="EPSG:4326",
    )
    network_gdf = None
    if network:
        network_gdf = gpd.GeoDataFrame(
            geometry=[Point(36.81, -1.29), Point(36.815, -1.285)],
            crs="EPSG:4326",
        )
    poi_gdf = None
    if pois:
        poi_gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(36.812, -1.288)],
                "name": ["Nairobi Hospital"],
                "amenity": ["hospital"],
            },
            crs="EPSG:4326",
        )
    return TravelTimeResult(
        request=req,
        isochrone_gdf=iso_gdf,
        network_gdf=network_gdf,
        poi_gdf=poi_gdf,
        router_name="test",
    )


# ---------------------------------------------------------------------------
# _resolve_layers
# ---------------------------------------------------------------------------


class TestResolveLayers:
    def test_no_network_no_pois(self) -> None:
        result = _make_result()
        layers = _resolve_layers(result, None)
        assert VizLayer.ISOCHRONE in layers
        assert VizLayer.HEATMAP in layers
        assert VizLayer.NETWORK not in layers
        assert VizLayer.POI_SCATTER not in layers

    def test_with_network_and_pois(self) -> None:
        result = _make_result(network=True, pois=True)
        layers = _resolve_layers(result, None)
        assert VizLayer.NETWORK in layers
        assert VizLayer.POI_SCATTER in layers

    def test_explicit_subset(self) -> None:
        result = _make_result(network=True, pois=True)
        layers = _resolve_layers(result, [VizLayer.ISOCHRONE])
        assert layers == {VizLayer.ISOCHRONE}

    def test_requested_unavailable_silently_excluded(self) -> None:
        result = _make_result(network=False)
        layers = _resolve_layers(result, [VizLayer.ISOCHRONE, VizLayer.NETWORK])
        assert VizLayer.NETWORK not in layers
        assert VizLayer.ISOCHRONE in layers


# ---------------------------------------------------------------------------
# _sample_polygon_points
# ---------------------------------------------------------------------------


class TestSamplePolygonPoints:
    def test_returns_correct_count(self) -> None:
        pts = _sample_polygon_points(_SQUARE, n=50)
        assert len(pts) == 50

    def test_points_are_lat_lon_tuples(self) -> None:
        pts = _sample_polygon_points(_SQUARE, n=10)
        for lat, lon in pts:
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180

    def test_empty_polygon_returns_empty(self) -> None:
        from shapely.geometry import Polygon
        empty = Polygon()
        pts = _sample_polygon_points(empty, n=50)
        assert pts == []

    def test_multipolygon_uses_largest(self) -> None:
        from shapely.geometry import MultiPolygon
        small = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0)])
        large = _SQUARE
        mp = MultiPolygon([small, large])
        pts = _sample_polygon_points(mp, n=20)
        assert len(pts) == 20


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


class TestRender:
    def test_returns_folium_map(self) -> None:
        result = _make_result()
        m = render(result)
        assert isinstance(m, folium.Map)

    def test_with_all_layers(self) -> None:
        result = _make_result(network=True, pois=True)
        m = render(result)
        assert isinstance(m, folium.Map)

    def test_explicit_layer_subset(self) -> None:
        result = _make_result(network=True, pois=True)
        m = render(result, layers=[VizLayer.ISOCHRONE])
        assert isinstance(m, folium.Map)

    def test_empty_layer_list(self) -> None:
        result = _make_result()
        # No layers at all — should still return a map (just origin marker)
        m = render(result, layers=[])
        assert isinstance(m, folium.Map)

    def test_map_html_contains_origin_coords(self) -> None:
        result = _make_result()
        m = render(result)
        html = m._repr_html_()
        # origin lat/lon should appear in the rendered HTML
        assert "-1.286389" in html or "36.817223" in html

    def test_map_html_nonempty(self) -> None:
        result = _make_result()
        html = m = render(result)
        assert m._repr_html_()


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


class TestSave:
    def test_writes_html_file(self) -> None:
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_map.html"
            returned = save(result, out)
            assert returned == out.resolve()
            assert out.exists()
            assert out.stat().st_size > 0

    def test_returns_absolute_path(self) -> None:
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "map.html"
            returned = save(result, out)
            assert returned.is_absolute()

    def test_output_is_valid_html(self) -> None:
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "map.html"
            save(result, out)
            content = out.read_text()
            assert "<html" in content.lower()
            assert "leaflet" in content.lower()
