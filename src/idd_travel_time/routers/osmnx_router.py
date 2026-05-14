"""osmnx local routing backend.

Local fallback — no API key, no rate limits. Good for batch / cluster runs
where ORS quotas would be exhausted.

How it works
------------
1. Download (or load from cache) the street graph around the origin point.
2. Find all nodes reachable within the time budget at the effective speed.
3. Build a polygon around those nodes via node-buffer union.

Network / POI extraction reuses the same graph — no second download.

Graph caching
-------------
Pass a ``GraphCache`` instance to avoid re-downloading the same area::

    from idd_travel_time.io.graph_cache import GraphCache
    router = OsmnxRouter(graph_cache=GraphCache("/data/graph_cache"))

Or set ``IDD_GRAPH_CACHE_DIR`` in the environment and the router picks it
up automatically via ``default_cache_from_env()``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

from idd_travel_time.constants import VizLayer
from idd_travel_time.models import TravelTimeResult
from idd_travel_time.routers._helpers import (
    _WGS84,
    effective_speed_kph,
    make_isochrone_gdf,
    make_poi_gdf,
)
from idd_travel_time.routers.base import Router

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from idd_travel_time.io.graph_cache import GraphCache
    from idd_travel_time.models import TravelTimeRequest

# Buffer radius added around each reachable node before unioning, in degrees.
# ~50m at equator. Ensures the polygon covers roads, not just node centroids.
_NODE_BUFFER_DEG: float = 0.0005


class OsmnxRouter(Router):
    """Isochrone router using a locally downloaded OSM street graph.

    No API key required. Downloads graph data from the Overpass API
    on first call for a given area; subsequent calls for the same area
    return instantly when a ``GraphCache`` is provided.

    Parameters
    ----------
    network_type:
        osmnx network type (``"all"``, ``"drive"``, ``"walk"``, etc.).
        Default ``"all"`` covers roads, paths, and cycleways.
    graph_cache:
        Optional :class:`~idd_travel_time.io.graph_cache.GraphCache`.
        If ``None``, checks ``IDD_GRAPH_CACHE_DIR`` env var. If that is
        also unset, each call downloads fresh from Overpass.
    """

    def __init__(
        self,
        network_type: str = "all",
        graph_cache: GraphCache | None = None,
    ) -> None:
        self._network_type = network_type

        if graph_cache is not None:
            self._cache: GraphCache | None = graph_cache
        else:
            from idd_travel_time.io.graph_cache import (
                default_cache_from_env,
            )

            self._cache = default_cache_from_env()  # None if env var not set

    @property
    def name(self) -> str:
        return "osmnx"

    def compute_isochrone(self, request: TravelTimeRequest) -> TravelTimeResult:
        speed_kph = effective_speed_kph(request)
        dist_m = _travel_distance_meters(speed_kph, request.duration_minutes)
        radius_m = dist_m * 1.2  # 20% buffer to avoid edge effects

        graph = self._get_graph(request, radius_m)

        origin_node = ox.nearest_nodes(graph, request.origin_lon, request.origin_lat)
        speed_ms = speed_kph / 3.6
        budget_seconds = request.duration_minutes * 60.0

        for _, _, data in graph.edges(data=True):
            length_m: float = data.get("length", 0.0)
            data["travel_time"] = length_m / speed_ms if speed_ms > 0 else math.inf

        reachable_nodes = nx.single_source_dijkstra_path_length(
            graph,
            origin_node,
            cutoff=budget_seconds,
            weight="travel_time",
        )

        node_geoms = [
            graph.nodes[n]["geometry"]
            if "geometry" in graph.nodes[n]
            else _node_point(graph, n)
            for n in reachable_nodes
        ]

        node_gdf_wgs = gpd.GeoDataFrame(geometry=node_geoms, crs=_WGS84)

        polygon = _alpha_shape_or_convex_hull(node_gdf_wgs, buffer_deg=_NODE_BUFFER_DEG)
        isochrone_gdf = make_isochrone_gdf(polygon, request)

        network_gdf: gpd.GeoDataFrame | None = None
        poi_gdf: gpd.GeoDataFrame | None = None

        if VizLayer.NETWORK in request.viz_layers:
            network_gdf = self._extract_network(graph, reachable_nodes)

        if VizLayer.POI_SCATTER in request.viz_layers:
            poi_gdf = self._extract_pois(isochrone_gdf)

        return TravelTimeResult(
            request=request,
            isochrone_gdf=isochrone_gdf,
            network_gdf=network_gdf,
            poi_gdf=poi_gdf,
            router_name=self.name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_graph(
        self,
        request: TravelTimeRequest,
        radius_m: float,
    ) -> nx.MultiDiGraph:
        """Return the street graph — from cache if available, else download."""
        if self._cache is not None:
            return self._cache.get_or_download(
                lat=request.origin_lat,
                lon=request.origin_lon,
                radius_m=radius_m,
                network_type=self._network_type,
            )
        return ox.graph_from_point(
            (request.origin_lat, request.origin_lon),
            dist=round(radius_m),
            network_type=self._network_type,
        )

    def _extract_network(
        self,
        graph: nx.MultiDiGraph,
        reachable_nodes: dict[int, float],
    ) -> gpd.GeoDataFrame | None:
        """Return edges whose both endpoints are in the reachable set."""
        try:
            subgraph = graph.subgraph(reachable_nodes.keys())
            _, edges = ox.graph_to_gdfs(subgraph)
            return edges.to_crs(_WGS84)[["geometry"]].reset_index(drop=True)
        except Exception:  # noqa: BLE001
            return None

    def _extract_pois(
        self,
        isochrone_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame | None:
        """Fetch named POIs within the isochrone polygon via osmnx."""
        try:
            polygon = isochrone_gdf.geometry.iloc[0]
            tags = {"amenity": True, "name": True}
            pois = ox.features_from_polygon(polygon, tags=tags)
            pois = pois[pois.geometry.geom_type == "Point"].copy()

            names = pois.get("name", "").fillna("").tolist()
            amenities = pois.get("amenity", "").fillna("").tolist()
            points = [(geom.x, geom.y) for geom in pois.geometry]
            return make_poi_gdf(points, names, amenities)
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------


def _travel_distance_meters(speed_kph: float, duration_minutes: int) -> float:
    """Maximum straight-line distance reachable at *speed_kph* in *duration_minutes*."""
    return (speed_kph / 60.0) * duration_minutes * 1000.0


def _node_point(graph: nx.MultiDiGraph, node_id: int) -> object:
    """Build a Shapely Point from a graph node's x/y attributes."""
    from shapely.geometry import Point

    data = graph.nodes[node_id]
    return Point(data["x"], data["y"])


def _alpha_shape_or_convex_hull(
    node_gdf: gpd.GeoDataFrame,
    buffer_deg: float,
) -> BaseGeometry:
    """Build a tight polygon around reachable nodes.

    Strategy:
    1. Buffer each node point by *buffer_deg* to create small discs.
    2. Union all discs — this naturally produces a polygon that follows
       the road network shape rather than a convex hull.
    3. Fall back to convex hull of all nodes if the union is empty.
    """
    if node_gdf.empty:
        msg = "No reachable nodes — cannot build isochrone polygon"
        raise ValueError(msg)

    # Buffer in a projected CRS (metres) to avoid geographic-CRS distortion,
    # then reproject the union back to WGS-84.
    buffer_m = buffer_deg * 111_000.0
    node_proj = node_gdf.to_crs(node_gdf.estimate_utm_crs())
    buffered = node_proj.geometry.buffer(buffer_m)
    union_proj = unary_union(buffered)
    union_gdf = gpd.GeoDataFrame(geometry=[union_proj], crs=node_proj.crs)
    union = union_gdf.to_crs(_WGS84).geometry.iloc[0]

    if union.is_empty:
        return MultiPoint(list(node_gdf.geometry)).convex_hull

    return union
