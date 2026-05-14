"""Tests for idd_travel_time.viz.static_viz."""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
from matplotlib.figure import Figure
from shapely.geometry import LineString, Point, Polygon

from idd_travel_time.constants import TransportMode, VizLayer
from idd_travel_time.models import TravelTimeRequest, TravelTimeResult
from idd_travel_time.viz.static_viz import (
    _resolve_layers,
    render_static,
    save_static,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SQUARE = Polygon(
    [
        (36.80, -1.30),
        (36.82, -1.30),
        (36.82, -1.28),
        (36.80, -1.28),
        (36.80, -1.30),
    ]
)

_WALKING_REQUEST = TravelTimeRequest(
    origin_lat=-1.286389,
    origin_lon=36.817223,
    mode=TransportMode.WALKING,
    duration_minutes=30,
)


def _make_result(
    network: bool = False,
    pois: bool = False,
) -> TravelTimeResult:
    iso = gpd.GeoDataFrame(
        {
            "geometry": [_SQUARE],
            "duration_min": [30],
            "mode": ["walking"],
            "time_context": ["off_peak"],
        },
        crs="EPSG:4326",
    )
    net = None
    if network:
        net = gpd.GeoDataFrame(
            geometry=[LineString([(36.80, -1.30), (36.82, -1.28)])],
            crs="EPSG:4326",
        )
    poi = None
    if pois:
        poi = gpd.GeoDataFrame(
            {
                "geometry": [Point(36.812, -1.288)],
                "name": ["Hospital"],
                "amenity": ["hospital"],
            },
            crs="EPSG:4326",
        )
    return TravelTimeResult(
        request=_WALKING_REQUEST,
        isochrone_gdf=iso,
        network_gdf=net,
        poi_gdf=poi,
        router_name="test",
    )


# ---------------------------------------------------------------------------
# _resolve_layers
# ---------------------------------------------------------------------------


class TestResolveLayersStatic:
    def test_isochrone_and_heatmap_always_available(self) -> None:
        result = _make_result()
        layers = _resolve_layers(result, None)
        assert VizLayer.ISOCHRONE in layers
        assert VizLayer.HEATMAP in layers

    def test_network_excluded_when_none(self) -> None:
        result = _make_result(network=False)
        layers = _resolve_layers(result, None)
        assert VizLayer.NETWORK not in layers

    def test_network_included_when_present(self) -> None:
        result = _make_result(network=True)
        layers = _resolve_layers(result, None)
        assert VizLayer.NETWORK in layers

    def test_poi_excluded_when_none(self) -> None:
        result = _make_result(pois=False)
        layers = _resolve_layers(result, None)
        assert VizLayer.POI_SCATTER not in layers

    def test_explicit_subset(self) -> None:
        result = _make_result(network=True, pois=True)
        layers = _resolve_layers(result, [VizLayer.ISOCHRONE])
        assert layers == {VizLayer.ISOCHRONE}

    def test_unavailable_layer_silently_excluded(self) -> None:
        result = _make_result(network=False)
        layers = _resolve_layers(result, [VizLayer.ISOCHRONE, VizLayer.NETWORK])
        assert VizLayer.NETWORK not in layers


# ---------------------------------------------------------------------------
# render_static
# ---------------------------------------------------------------------------


class TestRenderStatic:
    def test_returns_figure(self) -> None:
        result = _make_result()
        fig = render_static(result)
        assert isinstance(fig, Figure)

    def test_with_all_layers(self) -> None:
        result = _make_result(network=True, pois=True)
        fig = render_static(result)
        assert isinstance(fig, Figure)

    def test_explicit_layer_subset(self) -> None:
        result = _make_result(network=True, pois=True)
        fig = render_static(result, layers=[VizLayer.ISOCHRONE])
        assert isinstance(fig, Figure)

    def test_empty_layers(self) -> None:
        result = _make_result()
        fig = render_static(result, layers=[])
        assert isinstance(fig, Figure)

    def test_custom_figsize(self) -> None:
        result = _make_result()
        fig = render_static(result, figsize=(6, 6))
        w, h = fig.get_size_inches()
        assert (w, h) == (6.0, 6.0)

    def test_custom_dpi(self) -> None:
        result = _make_result()
        fig = render_static(result, dpi=72)
        assert fig.get_dpi() == 72

    def test_has_exactly_one_axes(self) -> None:
        result = _make_result()
        fig = render_static(result)
        assert len(fig.axes) == 1

    def test_axes_has_title_with_duration(self) -> None:
        result = _make_result()
        fig = render_static(result)
        title = fig.axes[0].get_title()
        assert "30" in title
        assert "walking" in title


# ---------------------------------------------------------------------------
# save_static
# ---------------------------------------------------------------------------


class TestSaveStatic:
    def test_writes_png(self) -> None:
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "map.png"
            returned = save_static(result, out)
            assert returned == out.resolve()
            assert out.exists()
            assert out.stat().st_size > 0

    def test_returns_absolute_path(self) -> None:
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "map.png"
            returned = save_static(result, out)
            assert returned.is_absolute()

    def test_png_is_valid_image(self) -> None:
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "map.png"
            save_static(result, out)
            # Check PNG magic bytes
            header = out.read_bytes()[:8]
            assert header == b"\x89PNG\r\n\x1a\n"

    def test_saves_pdf(self) -> None:
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "map.pdf"
            save_static(result, out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_saves_with_layers_subset(self) -> None:
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "map.png"
            returned = save_static(result, out, layers=[VizLayer.ISOCHRONE])
            assert returned.exists()


# ---------------------------------------------------------------------------
# viz/__init__ re-exports
# ---------------------------------------------------------------------------


def test_render_static_importable_from_viz() -> None:
    from idd_travel_time import viz

    assert hasattr(viz, "render_static")
    assert hasattr(viz, "save_static")
