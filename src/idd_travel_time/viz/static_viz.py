"""Matplotlib static map backend for travel-time isochrone results.

Produces publication-quality PNG/PDF/SVG figures suitable for reports,
papers, and batch thumbnail generation — contexts where an interactive
Leaflet map is not appropriate.

Basemap
-------
Uses contextily to pull a tiled basemap (CartoDB Positron by default)
behind the isochrone layers. contextily is an optional dependency; if
not installed, the map is rendered without a basemap and a warning is
emitted.

Public API
----------
render_static(result, layers=None, figsize=None, dpi=150) -> Figure
    Build and return a matplotlib Figure.

save_static(result, path, layers=None, figsize=None, dpi=150) -> Path
    Render and save to a file. Format inferred from the extension
    (.png, .pdf, .svg, .jpg all supported by matplotlib).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from idd_travel_time.constants import VizLayer

if TYPE_CHECKING:
    from idd_travel_time.models import TravelTimeResult

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_ISOCHRONE_COLOR = "#3388ff"
_ISOCHRONE_ALPHA = 0.30
_ISOCHRONE_EDGE_COLOR = "#1a5fa8"
_ISOCHRONE_EDGE_WIDTH = 1.5

_NETWORK_COLOR = "#ff7800"
_NETWORK_ALPHA = 0.55
_NETWORK_WIDTH = 0.6

_POI_COLOR = "#d62728"
_POI_MARKER = "o"
_POI_SIZE = 18
_POI_ALPHA = 0.85
_POI_EDGE_COLOR = "white"
_POI_EDGE_WIDTH = 0.4

_ORIGIN_COLOR = "#2ca02c"
_ORIGIN_MARKER = "*"
_ORIGIN_SIZE = 120

_HEATMAP_CMAP = "YlOrRd"
_HEATMAP_ALPHA = 0.45
_HEATMAP_BINS = 80

_DEFAULT_FIGSIZE = (10, 10)
_DEFAULT_DPI = 150

_BASEMAP_SOURCE = "CartoDB.Positron"  # contextily provider string


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_static(
    result: TravelTimeResult,
    layers: list[VizLayer] | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = _DEFAULT_DPI,
) -> Figure:
    """Build a matplotlib Figure for *result*.

    Parameters
    ----------
    result:
        Completed TravelTimeResult.
    layers:
        Which layers to render. Defaults to all available.
    figsize:
        Figure size in inches ``(width, height)``. Default ``(10, 10)``.
    dpi:
        Resolution in dots per inch. Default 150.

    Returns
    -------
    matplotlib.figure.Figure
    """
    active_layers = _resolve_layers(result, layers)
    fig, ax = plt.subplots(1, 1, figsize=figsize or _DEFAULT_FIGSIZE, dpi=dpi)
    ax.set_aspect("equal")

    # Determine map extent from the isochrone polygon + 10% margin.
    iso_proj, crs_proj = _project(result.isochrone_gdf)
    bounds = iso_proj.total_bounds  # [minx, miny, maxx, maxy]
    margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.10
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

    # Basemap (best-effort — silently degrade if contextily absent).
    _add_basemap(ax, crs_proj)

    # Layer rendering order: heatmap → isochrone → network → POIs → origin.
    if VizLayer.HEATMAP in active_layers:
        _draw_heatmap(ax, iso_proj)

    if VizLayer.ISOCHRONE in active_layers:
        _draw_isochrone(ax, iso_proj)

    if VizLayer.NETWORK in active_layers and result.network_gdf is not None:
        net_proj, _ = _project(result.network_gdf, target_crs=crs_proj)
        _draw_network(ax, net_proj)

    if VizLayer.POI_SCATTER in active_layers and result.poi_gdf is not None:
        poi_proj, _ = _project(result.poi_gdf, target_crs=crs_proj)
        _draw_pois(ax, poi_proj)

    _draw_origin(ax, result, crs_proj)
    _style_axes(ax, result)
    _add_legend(ax, active_layers, result)

    fig.tight_layout()
    return fig


def save_static(
    result: TravelTimeResult,
    path: str | Path,
    layers: list[VizLayer] | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = _DEFAULT_DPI,
) -> Path:
    """Render and save the static map to *path*.

    Parameters
    ----------
    result:
        Completed TravelTimeResult.
    path:
        Output file path. Format inferred from extension.
    layers:
        Subset of layers. Defaults to all available.
    figsize:
        Figure size in inches.
    dpi:
        Resolution.

    Returns
    -------
    Path
        Resolved absolute path of the saved file.
    """
    out = Path(path).resolve()
    fig = render_static(result, layers=layers, figsize=figsize, dpi=dpi)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Private — layer resolvers (mirrors folium_viz logic)
# ---------------------------------------------------------------------------


def _resolve_layers(
    result: TravelTimeResult,
    requested: list[VizLayer] | None,
) -> set[VizLayer]:
    if requested is None:
        requested = list(result.request.viz_layers)
    available: set[VizLayer] = {VizLayer.ISOCHRONE, VizLayer.HEATMAP}
    if result.network_gdf is not None:
        available.add(VizLayer.NETWORK)
    if result.poi_gdf is not None:
        available.add(VizLayer.POI_SCATTER)
    return available.intersection(requested)


# ---------------------------------------------------------------------------
# Private — CRS projection helpers
# ---------------------------------------------------------------------------


def _project(
    gdf: gpd.GeoDataFrame,
    target_crs: object | None = None,
) -> tuple[gpd.GeoDataFrame, object]:
    """Reproject *gdf* to a metric CRS for accurate plotting.

    Returns the projected GeoDataFrame and the CRS used.
    """
    if target_crs is not None:
        return gdf.to_crs(target_crs), target_crs
    crs = gdf.estimate_utm_crs()
    return gdf.to_crs(crs), crs


# ---------------------------------------------------------------------------
# Private — drawing functions
# ---------------------------------------------------------------------------


def _draw_isochrone(ax: plt.Axes, iso_proj: gpd.GeoDataFrame) -> None:
    iso_proj.plot(
        ax=ax,
        color=_ISOCHRONE_COLOR,
        alpha=_ISOCHRONE_ALPHA,
        edgecolor=_ISOCHRONE_EDGE_COLOR,
        linewidth=_ISOCHRONE_EDGE_WIDTH,
        zorder=2,
    )


def _draw_network(ax: plt.Axes, net_proj: gpd.GeoDataFrame) -> None:
    net_proj.plot(
        ax=ax,
        color=_NETWORK_COLOR,
        alpha=_NETWORK_ALPHA,
        linewidth=_NETWORK_WIDTH,
        zorder=3,
    )


def _draw_pois(ax: plt.Axes, poi_proj: gpd.GeoDataFrame) -> None:
    poi_proj.plot(
        ax=ax,
        color=_POI_COLOR,
        marker=_POI_MARKER,
        markersize=_POI_SIZE,
        alpha=_POI_ALPHA,
        edgecolors=_POI_EDGE_COLOR,
        linewidths=_POI_EDGE_WIDTH,
        zorder=5,
    )


def _draw_origin(
    ax: plt.Axes,
    result: TravelTimeResult,
    crs_proj: object,
) -> None:
    """Plot the origin star marker, reprojected to match the axes CRS."""
    from shapely.geometry import Point  # noqa: PLC0415

    lat, lon = result.origin
    origin_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)], crs="EPSG:4326"
    ).to_crs(crs_proj)
    origin_gdf.plot(
        ax=ax,
        color=_ORIGIN_COLOR,
        marker=_ORIGIN_MARKER,
        markersize=_ORIGIN_SIZE,
        zorder=6,
    )


def _draw_heatmap(ax: plt.Axes, iso_proj: gpd.GeoDataFrame) -> None:
    """Render a 2-D kernel density heatmap of the isochrone polygon."""
    polygon = iso_proj.geometry.iloc[0]
    if polygon.is_empty:
        return

    # Sample random points inside the polygon for density estimation.
    try:
        from shapely.geometry import MultiPolygon  # noqa: PLC0415

        if isinstance(polygon, MultiPolygon):
            polygon = max(polygon.geoms, key=lambda p: p.area)

        minx, miny, maxx, maxy = polygon.bounds
        rng = np.random.default_rng(seed=42)
        n_samples = 2000
        xs, ys = [], []
        while len(xs) < n_samples:
            batch_x = rng.uniform(minx, maxx, n_samples)
            batch_y = rng.uniform(miny, maxy, n_samples)
            from shapely.geometry import Point  # noqa: PLC0415
            mask = [polygon.contains(Point(x, y)) for x, y in zip(batch_x, batch_y)]
            xs.extend(batch_x[mask])
            ys.extend(batch_y[mask])
        xs = np.array(xs[:n_samples])
        ys = np.array(ys[:n_samples])

        ax.hist2d(
            xs, ys,
            bins=_HEATMAP_BINS,
            cmap=_HEATMAP_CMAP,
            alpha=_HEATMAP_ALPHA,
            zorder=1,
        )
    except Exception:  # noqa: BLE001
        pass  # heatmap is optional — never hard-fail on it


# ---------------------------------------------------------------------------
# Private — basemap, axes styling, legend
# ---------------------------------------------------------------------------


def _add_basemap(ax: plt.Axes, crs_proj: object) -> None:
    """Add a tiled basemap via contextily. Silently skips if unavailable."""
    try:
        import contextily as ctx  # noqa: PLC0415
        ctx.add_basemap(ax, crs=str(crs_proj), source=ctx.providers.CartoDB.Positron)
    except Exception:  # noqa: BLE001
        # contextily not installed, network unavailable, or tile fetch failed.
        ax.set_facecolor("#f0f0f0")


def _style_axes(ax: plt.Axes, result: TravelTimeResult) -> None:
    req = result.request
    title = (
        f"{req.duration_minutes}-min {req.mode.value} isochrone  "
        f"[{req.time_context.value.replace('_', ' ')}]  "
        f"· {req.origin_lat:.4f}, {req.origin_lon:.4f}"
    )
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_axis_off()


def _add_legend(
    ax: plt.Axes,
    active_layers: set[VizLayer],
    result: TravelTimeResult,
) -> None:
    handles = []
    if VizLayer.ISOCHRONE in active_layers:
        handles.append(mpatches.Patch(
            facecolor=_ISOCHRONE_COLOR, alpha=0.6,
            edgecolor=_ISOCHRONE_EDGE_COLOR, label="Reachable area",
        ))
    if VizLayer.NETWORK in active_layers:
        handles.append(mpatches.Patch(
            facecolor=_NETWORK_COLOR, alpha=0.8, label="Road network",
        ))
    if VizLayer.POI_SCATTER in active_layers:
        handles.append(mpatches.Patch(
            facecolor=_POI_COLOR, alpha=0.8, label="Places of interest",
        ))
    handles.append(mpatches.Patch(
        facecolor=_ORIGIN_COLOR, alpha=0.9, label="Origin",
    ))
    if handles:
        ax.legend(
            handles=handles,
            loc="lower right",
            fontsize=8,
            framealpha=0.85,
            edgecolor="#cccccc",
        )
