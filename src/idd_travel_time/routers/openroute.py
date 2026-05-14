"""OpenRouteService (ORS) routing backend.

Default backend. Uses the ORS isochrones REST API for polygon generation
and osmnx for network/POI extraction within the resulting polygon.

API key
-------
Read from the ``ORS_API_KEY`` environment variable. If not set, a
``RuntimeError`` is raised at construction time (fail-fast rather than at
the first API call).

Rate limits
-----------
ORS free tier: 500 requests/day, 40 requests/minute. This router makes
no attempt to throttle — callers doing batch work should add their own
rate-limiting wrapper.

Transit
-------
ORS has no transit profile. For TRANSIT requests we use ``driving-car``
as the road-network proxy and scale the isochrone range by the ratio of
the TransitProfile midpoint speed to the ORS driving speed assumption
(~30 km/h urban). This is an approximation — not schedule-exact — which
is intentional per the project design.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import geopandas as gpd
import openrouteservice
from shapely.geometry import shape

from idd_travel_time.constants import VizLayer
from idd_travel_time.models import TravelTimeResult
from idd_travel_time.routers._helpers import (
    effective_speed_kph,
    make_isochrone_gdf,
    make_poi_gdf,
    ors_profile_for_request,
)
from idd_travel_time.routers.base import Router

if TYPE_CHECKING:
    from idd_travel_time.models import TravelTimeRequest

# ORS driving-car assumes ~30 kph average urban speed for range scaling.
_ORS_DRIVING_SPEED_KPH: float = 30.0


class OpenRouteServiceRouter(Router):
    """Isochrone router backed by the OpenRouteService REST API.

    Parameters
    ----------
    api_key:
        ORS API key. Defaults to the ``ORS_API_KEY`` environment variable.

    Raises
    ------
    RuntimeError
        If no API key is found.
    """

    def __init__(self, api_key: str | None = None) -> None:
        resolved = api_key or os.environ.get("ORS_API_KEY")
        if not resolved:
            msg = (
                "ORS API key is required. Set the ORS_API_KEY environment variable "
                "or pass api_key= explicitly."
            )
            raise RuntimeError(msg)
        self._client = openrouteservice.Client(key=resolved)

    @property
    def name(self) -> str:
        return "openroute"

    def compute_isochrone(self, request: TravelTimeRequest) -> TravelTimeResult:
        """Call ORS isochrones API and return a TravelTimeResult.

        The ORS API takes (lon, lat) — note the reversed order vs WGS-84
        convention. We handle the flip internally.
        """
        profile = ors_profile_for_request(request)
        range_seconds = self._effective_range_seconds(request)

        iso_response = self._client.isochrones(
            locations=[[request.origin_lon, request.origin_lat]],
            profile=profile,
            range=[range_seconds],
            range_type="time",
            attributes=["area"],
        )

        polygon = shape(iso_response["features"][0]["geometry"])
        isochrone_gdf = make_isochrone_gdf(polygon, request)

        network_gdf: gpd.GeoDataFrame | None = None
        poi_gdf: gpd.GeoDataFrame | None = None

        if VizLayer.NETWORK in request.viz_layers:
            network_gdf = self._fetch_network(isochrone_gdf)

        if VizLayer.POI_SCATTER in request.viz_layers:
            poi_gdf = self._fetch_pois(isochrone_gdf)

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

    def _effective_range_seconds(self, request: TravelTimeRequest) -> int:
        """Convert duration_minutes to ORS range in seconds.

        For TRANSIT mode, scale by the ratio of the TransitProfile speed
        to ORS's implicit driving speed so the polygon covers the correct
        real-world area.
        """
        from idd_travel_time.constants import TransportMode  # noqa: PLC0415

        base_seconds = request.duration_minutes * 60

        if request.mode is TransportMode.TRANSIT:
            speed = effective_speed_kph(request)
            scale = speed / _ORS_DRIVING_SPEED_KPH
            return round(base_seconds * scale)

        return base_seconds

    def _fetch_network(
        self,
        isochrone_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame | None:
        """Fetch road network edges within the isochrone polygon via osmnx.

        Falls back to None (with a warning) if osmnx is unavailable or the
        query fails — network is optional in the result.
        """
        try:
            import osmnx as ox  # noqa: PLC0415
            from idd_travel_time.routers._helpers import _WGS84  # noqa: PLC0415

            polygon = isochrone_gdf.geometry.iloc[0]
            graph = ox.graph_from_polygon(polygon, network_type="all")
            _, edges = ox.graph_to_gdfs(graph)
            return edges.to_crs(_WGS84)[["geometry"]].reset_index(drop=True)
        except Exception:  # noqa: BLE001
            return None

    def _fetch_pois(
        self,
        isochrone_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame | None:
        """Fetch named POIs within the isochrone polygon via osmnx.

        Returns None on any failure — POIs are optional.
        """
        try:
            import osmnx as ox  # noqa: PLC0415

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
