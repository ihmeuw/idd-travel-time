"""Folium visualisation for travel-time isochrone results.

Renders up to four layers on a single folium Map:

    ISOCHRONE   — shaded reachable-area polygon
    NETWORK     — reachable road/path edges
    POI_SCATTER — named places within the isochrone
    HEATMAP     — reachability density heatmap

Public API
----------
render(result, layers=None) -> folium.Map
    Build and return a folium Map for *result*.
    Pass ``layers=[VizLayer.ISOCHRONE, VizLayer.HEATMAP]`` to render a subset.

save(result, path, layers=None) -> Path
    Render and save to an HTML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import folium
from folium.plugins import HeatMap

from idd_travel_time.constants import VizLayer

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from idd_travel_time.models import TravelTimeResult

# ---------------------------------------------------------------------------
# Style constants — all visual parameters live here, not inline
# ---------------------------------------------------------------------------

_ISOCHRONE_COLOR = "#3388ff"
_ISOCHRONE_FILL_OPACITY = 0.25
_ISOCHRONE_LINE_OPACITY = 0.8
_ISOCHRONE_WEIGHT = 2

_NETWORK_COLOR = "#ff7800"
_NETWORK_WEIGHT = 1
_NETWORK_OPACITY = 0.6

_POI_RADIUS = 5
_POI_COLOR = "#d62728"
_POI_FILL_COLOR = "#d62728"
_POI_FILL_OPACITY = 0.8

_HEATMAP_RADIUS = 15
_HEATMAP_BLUR = 10
_HEATMAP_MIN_OPACITY = 0.3

_TILE_PROVIDER = "CartoDB positron"
_ZOOM_START = 13


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render(
    result: TravelTimeResult,
    layers: list[VizLayer] | None = None,
) -> folium.Map:
    """Build a folium Map for *result*.

    Parameters
    ----------
    result:
        Completed TravelTimeResult from any router.
    layers:
        Which layers to render. Defaults to all layers present in *result*
        (i.e. whatever the router populated). Pass an explicit list to
        restrict to a subset, e.g. ``[VizLayer.ISOCHRONE]``.

    Returns
    -------
    folium.Map
        Interactive Leaflet map. Call ``.save("out.html")`` or display
        inline in a notebook with ``display(m)`` / return value.
    """
    active_layers = _resolve_layers(result, layers)

    lat, lon = result.origin
    m = folium.Map(
        location=[lat, lon],
        zoom_start=_ZOOM_START,
        tiles=_TILE_PROVIDER,
    )

    # Layer order: heatmap first (bottom), then network, then isochrone
    # polygon (semi-transparent), then POIs on top.
    if VizLayer.HEATMAP in active_layers:
        _add_heatmap(m, result)

    if VizLayer.NETWORK in active_layers and result.network_gdf is not None:
        _add_network(m, result)

    if VizLayer.ISOCHRONE in active_layers:
        _add_isochrone(m, result)

    if VizLayer.POI_SCATTER in active_layers and result.poi_gdf is not None:
        _add_pois(m, result)

    _add_origin_marker(m, result)
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def save(
    result: TravelTimeResult,
    path: str | Path,
    layers: list[VizLayer] | None = None,
) -> Path:
    """Render and save the map to an HTML file.

    Parameters
    ----------
    result:
        Completed TravelTimeResult.
    path:
        Output file path (will be created; parent dirs must exist).
    layers:
        Subset of layers to render. Defaults to all available.

    Returns
    -------
    Path
        Resolved absolute path of the saved file.
    """
    out = Path(path).resolve()
    m = render(result, layers=layers)
    m.save(str(out))
    return out


# ---------------------------------------------------------------------------
# Private layer builders
# ---------------------------------------------------------------------------


def _resolve_layers(
    result: TravelTimeResult,
    requested: list[VizLayer] | None,
) -> set[VizLayer]:
    """Intersect requested layers with what's actually available in result."""
    if requested is None:
        requested = list(result.request.viz_layers)

    available: set[VizLayer] = {VizLayer.ISOCHRONE}  # always present
    if result.network_gdf is not None:
        available.add(VizLayer.NETWORK)
    if result.poi_gdf is not None:
        available.add(VizLayer.POI_SCATTER)
    # Heatmap derives from the isochrone polygon nodes — always computable.
    available.add(VizLayer.HEATMAP)

    return available.intersection(requested)


def _add_isochrone(m: folium.Map, result: TravelTimeResult) -> None:
    """Add the reachable-area polygon as a semi-transparent GeoJSON layer."""
    layer = folium.FeatureGroup(name="Isochrone", show=True)
    folium.GeoJson(
        result.isochrone_gdf.__geo_interface__,
        style_function=lambda _: {
            "fillColor": _ISOCHRONE_COLOR,
            "color": _ISOCHRONE_COLOR,
            "weight": _ISOCHRONE_WEIGHT,
            "fillOpacity": _ISOCHRONE_FILL_OPACITY,
            "opacity": _ISOCHRONE_LINE_OPACITY,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["duration_min", "mode", "time_context"],
            aliases=["Duration (min)", "Mode", "Time context"],
        ),
    ).add_to(layer)
    layer.add_to(m)


def _add_network(m: folium.Map, result: TravelTimeResult) -> None:
    """Add reachable road/path edges as a polyline layer."""
    assert result.network_gdf is not None  # noqa: S101  # caller checks VizLayer.NETWORK
    layer = folium.FeatureGroup(name="Road network", show=False)
    folium.GeoJson(
        result.network_gdf.__geo_interface__,
        style_function=lambda _: {
            "color": _NETWORK_COLOR,
            "weight": _NETWORK_WEIGHT,
            "opacity": _NETWORK_OPACITY,
        },
    ).add_to(layer)
    layer.add_to(m)


def _add_pois(m: folium.Map, result: TravelTimeResult) -> None:
    """Add named POIs as circle markers with popups."""
    assert result.poi_gdf is not None  # noqa: S101  # caller checks VizLayer.POI_SCATTER
    layer = folium.FeatureGroup(name="Places of interest", show=True)

    for _, row in result.poi_gdf.iterrows():
        name = str(row.get("name", "")) or "Unnamed"
        amenity = str(row.get("amenity", "")) or ""
        label = f"{name} ({amenity})" if amenity else name

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=_POI_RADIUS,
            color=_POI_COLOR,
            fill=True,
            fill_color=_POI_FILL_COLOR,
            fill_opacity=_POI_FILL_OPACITY,
            popup=folium.Popup(label, max_width=200),
            tooltip=label,
        ).add_to(layer)

    layer.add_to(m)


def _add_heatmap(m: folium.Map, result: TravelTimeResult) -> None:
    """Add a reachability heatmap derived from the isochrone polygon boundary.

    We sample points along the polygon exterior as heat sources, weighted
    uniformly — this gives a density impression of the reachable boundary
    without requiring the full road network.
    """
    polygon = result.isochrone_gdf.geometry.iloc[0]
    heat_points = _sample_polygon_points(polygon, n=300)

    if not heat_points:
        return

    layer = folium.FeatureGroup(name="Heatmap", show=False)
    # folium.plugins.HeatMap ships without annotations. Calling it through a
    # typed alias satisfies mypy --strict in both the project env and the
    # stub-less pre-commit env; a `type: ignore` would be flagged unused there.
    heat_map: Callable[..., Any] = HeatMap
    heat_map(
        heat_points,
        radius=_HEATMAP_RADIUS,
        blur=_HEATMAP_BLUR,
        min_opacity=_HEATMAP_MIN_OPACITY,
    ).add_to(layer)
    layer.add_to(m)


def _add_origin_marker(m: folium.Map, result: TravelTimeResult) -> None:
    """Add a pin at the origin point."""
    lat, lon = result.origin
    mode = result.request.mode.value
    duration = result.request.duration_minutes
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(f"Origin<br>{mode}, {duration} min", max_width=150),
        tooltip="Origin",
        icon=folium.Icon(color="red", icon="play", prefix="fa"),
    ).add_to(m)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _sample_polygon_points(
    polygon: object,
    n: int,
) -> list[tuple[float, float]]:
    """Sample *n* points from the polygon exterior as (lat, lon) tuples.

    Uses uniform interpolation along the exterior ring. Falls back to
    an empty list if the polygon has no usable exterior.
    """
    try:
        from shapely.geometry import MultiPolygon, Polygon

        if isinstance(polygon, MultiPolygon):
            # Use the largest component
            polygon = max(polygon.geoms, key=lambda p: p.area)

        if not isinstance(polygon, Polygon) or polygon.is_empty:
            return []

        exterior = polygon.exterior
        total_length = exterior.length
        if total_length == 0:
            return []

        points: list[tuple[float, float]] = []
        for i in range(n):
            frac = i / n
            pt = exterior.interpolate(frac, normalized=True)
            points.append((pt.y, pt.x))  # folium wants (lat, lon)

    except Exception:  # noqa: BLE001
        return []
    return points
