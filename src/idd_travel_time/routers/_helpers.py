"""Internal helpers shared across router backends.

Not part of the public API — import from routers/ submodules instead.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

from idd_travel_time.constants import (
    COL_DURATION_MIN,
    COL_GEOMETRY,
    COL_MODE,
    COL_POI_NAME,
    COL_POI_TYPE,
    COL_TIME_CONTEXT,
)
from idd_travel_time.models import TravelTimeRequest

_WGS84 = "EPSG:4326"


def make_isochrone_gdf(
    polygon: Polygon,
    request: TravelTimeRequest,
) -> gpd.GeoDataFrame:
    """Wrap a Shapely polygon in the canonical isochrone GeoDataFrame schema.

    Schema
    ------
    geometry      : Polygon  (EPSG:4326)
    duration_min  : int
    mode          : str
    time_context  : str
    """
    df = pd.DataFrame(  # noqa: PD901  # transient — wrapped as GeoDataFrame on return
        {
            COL_GEOMETRY: [polygon],
            COL_DURATION_MIN: pd.array([request.duration_minutes], dtype="int32"),
            COL_MODE: pd.array([request.mode.value], dtype="string"),
            COL_TIME_CONTEXT: pd.array([request.time_context.value], dtype="string"),
        }
    )
    return gpd.GeoDataFrame(df, geometry=COL_GEOMETRY, crs=_WGS84)


def make_poi_gdf(
    points: list[tuple[float, float]],
    names: list[str],
    amenity_types: list[str],
) -> gpd.GeoDataFrame:
    """Build a POI GeoDataFrame from parallel lists.

    Parameters
    ----------
    points:
        List of (lon, lat) tuples.
    names, amenity_types:
        Parallel lists of the same length as *points*.
    """
    if not (len(points) == len(names) == len(amenity_types)):
        msg = "points, names, and amenity_types must have the same length"
        raise ValueError(msg)
    geometries = [Point(lon, lat) for lon, lat in points]
    df = pd.DataFrame(  # noqa: PD901  # transient — wrapped as GeoDataFrame on return
        {
            COL_GEOMETRY: geometries,
            COL_POI_NAME: pd.array(names, dtype="string"),
            COL_POI_TYPE: pd.array(amenity_types, dtype="string"),
        }
    )
    return gpd.GeoDataFrame(df, geometry=COL_GEOMETRY, crs=_WGS84)


def ors_profile_for_request(request: TravelTimeRequest) -> str:
    """Map a TravelTimeRequest to an ORS routing profile string.

    For TRANSIT mode, we use the road-network profile that matches
    the TransitProfile coverage type (driving for road_network,
    driving for fixed_routes as a conservative fallback — ORS has
    no transit profile).

    Returns
    -------
    str
        One of: ``"foot-walking"``, ``"cycling-regular"``,
        ``"driving-car"``.
    """
    from idd_travel_time.constants import (
        TransportMode,
    )

    match request.mode:
        case TransportMode.WALKING:
            return "foot-walking"
        case TransportMode.CYCLING:
            return "cycling-regular"
        case TransportMode.DRIVING:
            return "driving-car"
        case TransportMode.TRANSIT:
            # ORS has no transit profile; use driving as the network proxy.
            # Speed overriding is handled by the caller via range_type=time
            # and the midpoint_speed from TransitProfile.
            return "driving-car"
        case _:
            msg = f"Unsupported transport mode: {request.mode}"
            raise ValueError(msg)


def effective_speed_kph(request: TravelTimeRequest) -> float:
    """Return the effective routing speed in km/h for this request.

    For WALKING, CYCLING, DRIVING we use the midpoint of the mode's
    default speed range from constants. For TRANSIT we use the
    TransitProfile midpoint.
    """
    from idd_travel_time.constants import (
        DEFAULT_CYCLING_SPEED_KPH,
        DEFAULT_DRIVING_SPEED_KPH,
        DEFAULT_WALKING_SPEED_KPH,
        TransportMode,
    )

    match request.mode:
        case TransportMode.WALKING:
            lo, hi = DEFAULT_WALKING_SPEED_KPH
            return (lo + hi) / 2.0
        case TransportMode.CYCLING:
            lo, hi = DEFAULT_CYCLING_SPEED_KPH
            return (lo + hi) / 2.0
        case TransportMode.DRIVING:
            lo, hi = DEFAULT_DRIVING_SPEED_KPH
            return (lo + hi) / 2.0
        case TransportMode.TRANSIT:
            assert request.transit_profile is not None  # noqa: S101  # validated in TravelTimeRequest.__post_init__
            return request.transit_profile.midpoint_speed_kph
        case _:
            msg = f"Unsupported transport mode: {request.mode}"
            raise ValueError(msg)
