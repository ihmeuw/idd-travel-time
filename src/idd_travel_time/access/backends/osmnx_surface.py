"""OsmnxSurfaceRouter — travel-time surface via street-network Dijkstra.

Algorithm
---------
1. Download (or load from GraphCache) the OSM street graph for
   ``bbox`` expanded by a guard buffer.
2. Project to UTM; impute edge travel times from edge lengths + speed.
3. Snap the destination point to its nearest network node.
4. Run single-source Dijkstra from the destination node → travel time
   in seconds to every reachable node (no cutoff).
5. Build a KD-tree over node (lat, lon) positions.
6. For every grid cell centre in ``bbox`` * ``resolution_deg``, find
   its nearest network node and assign that node's travel time.
7. Return a TravelTimeSurface (minutes, float32, NaN = unreachable).

Speed assumption
----------------
Uses ``effective_speed_kph`` from the existing router helpers.  For TRANSIT
mode the TransitProfile midpoint speed is used.  This is the same approach
as OsmnxRouter (isochrone) — consistent across the two surface types.

Graph guard buffer
------------------
The graph must extend beyond ``bbox`` by at least
``speed_kph / 60 * max_expected_minutes * 1.2`` km to ensure paths
that enter and leave the bbox are captured correctly.  Default guard is
``speed * 60 min * 1.2``, capped at 25 km.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import osmnx as ox

from idd_travel_time.access.surface import BoundingBox, TravelTimeSurface
from idd_travel_time.access.surface_router import SurfaceRouter
from idd_travel_time.routers._helpers import effective_speed_kph

if TYPE_CHECKING:
    from idd_travel_time.constants import TransportMode
    from idd_travel_time.io.graph_cache import GraphCache
    from idd_travel_time.models import TransitProfile

_WGS84 = "EPSG:4326"
_MAX_GUARD_KM: float = 25.0
_GUARD_FACTOR: float = 1.2
_MAX_SNAP_DIST_M: float = 500.0  # warn if destination snaps > this far


class OsmnxSurfaceRouter(SurfaceRouter):
    """Travel-time surface backend using osmnx + single-source Dijkstra.

    Parameters
    ----------
    network_type:
        osmnx network type (``"all"``, ``"walk"``, ``"drive"``).
    graph_cache:
        Optional :class:`~idd_travel_time.io.graph_cache.GraphCache`.
        If ``None``, checks ``IDD_GRAPH_CACHE_DIR`` env var.
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

            self._cache = default_cache_from_env()

    @property
    def name(self) -> str:
        return "osmnx"

    def compute_surface(
        self,
        destination: tuple[float, float],
        bbox: BoundingBox,
        resolution_deg: float,
        mode: TransportMode,
        *,
        transit_profile: TransitProfile | None = None,
    ) -> TravelTimeSurface:
        """Compute travel-time surface via Dijkstra on OSM street graph."""
        from idd_travel_time.models import TravelTimeRequest

        dest_lat, dest_lon = destination

        # Build a minimal TravelTimeRequest just to reuse effective_speed_kph.
        _dummy_req = TravelTimeRequest(
            origin_lat=dest_lat,
            origin_lon=dest_lon,
            mode=mode,
            transit_profile=transit_profile,
        )
        speed_kph = effective_speed_kph(_dummy_req)
        speed_ms = speed_kph / 3.6

        # --- 1. Download / load graph over expanded bbox ------------------
        guard_km = min(speed_kph * _GUARD_FACTOR, _MAX_GUARD_KM)
        guard_deg = guard_km / 111.0
        expanded = bbox.buffer(guard_deg)

        graph = self._get_graph(expanded, dest_lat, dest_lon, guard_km * 1000.0)

        # --- 2. Impute travel times ---------------------------------------
        for _, _, data in graph.edges(data=True):
            length_m: float = data.get("length", 0.0)
            data["travel_time_s"] = length_m / speed_ms if speed_ms > 0 else math.inf

        # --- 3. Snap destination to nearest node --------------------------
        dest_node = ox.nearest_nodes(graph, dest_lon, dest_lat)

        # --- 4. Single-source Dijkstra (no cutoff) ------------------------
        node_times_s: dict[int, float] = nx.single_source_dijkstra_path_length(
            graph,
            dest_node,
            weight="travel_time_s",
        )

        # --- 5. Build KD-tree over node positions (WGS-84 lon/lat) --------
        node_ids = list(node_times_s.keys())
        node_lons = np.array([graph.nodes[n]["x"] for n in node_ids], dtype=np.float64)
        node_lats = np.array([graph.nodes[n]["y"] for n in node_ids], dtype=np.float64)
        node_times_arr = np.array(
            [node_times_s[n] / 60.0 for n in node_ids], dtype=np.float32
        )  # convert to minutes

        from scipy.spatial import cKDTree

        # KD-tree in (lat, lon) space — good enough for small areas
        node_coords = np.column_stack([node_lats, node_lons])
        kd_tree = cKDTree(node_coords)

        # --- 6. Assign travel times to raster grid cells ------------------
        lats, lons = _make_grid(bbox, resolution_deg)
        grid_coords = np.column_stack([lats.ravel(), lons.ravel()])

        # Max snap distance in degrees (~1 degree ≈ 111 km at equator)
        max_dist_deg = _MAX_SNAP_DIST_M / 111_000.0
        distances, indices = kd_tree.query(grid_coords, workers=-1)

        travel_times = node_times_arr[indices].reshape(lats.shape)
        # Mark cells that snapped to a node further than max_snap as NaN
        too_far = distances.reshape(lats.shape) > max_dist_deg
        travel_times[too_far] = np.nan

        # --- 7. Build and return TravelTimeSurface ------------------------
        return TravelTimeSurface.from_arrays(
            data=travel_times,
            west=bbox.west,
            north=bbox.north,
            resolution_deg=resolution_deg,
            backend=self.name,
            destination=destination,
            metadata={
                "mode": mode.value,
                "speed_kph": speed_kph,
                "network_type": self._network_type,
                "n_nodes": len(node_ids),
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_graph(
        self,
        expanded_bbox: BoundingBox,
        dest_lat: float,
        dest_lon: float,
        radius_m: float,
    ) -> nx.MultiDiGraph:
        """Download or load graph for the expanded bbox."""
        if self._cache is not None:
            return self._cache.get_or_download(
                lat=dest_lat,
                lon=dest_lon,
                radius_m=radius_m,
                network_type=self._network_type,
            )
        return ox.graph_from_bbox(
            bbox=(
                expanded_bbox.west,
                expanded_bbox.south,
                expanded_bbox.east,
                expanded_bbox.north,
            ),
            network_type=self._network_type,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_grid(
    bbox: BoundingBox,
    resolution_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lats, lons) centre-coordinate arrays for a regular grid.

    Shape is (nrows, ncols) where
    nrows = ceil(height / resolution) and ncols = ceil(width / resolution).
    """
    nrows = max(1, int(np.ceil(bbox.height_deg / resolution_deg)))
    ncols = max(1, int(np.ceil(bbox.width_deg / resolution_deg)))

    row_indices = np.arange(nrows)
    col_indices = np.arange(ncols)
    col_grid, row_grid = np.meshgrid(col_indices, row_indices)

    lons = bbox.west + (col_grid + 0.5) * resolution_deg
    lats = bbox.north - (row_grid + 0.5) * resolution_deg
    return lats.astype(np.float64), lons.astype(np.float64)
