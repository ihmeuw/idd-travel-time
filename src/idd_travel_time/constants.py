"""Project-wide constants, enumerations, and path roots.

Never hardcode these values elsewhere in the package.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of installed package (src/idd_travel_time/)
PACKAGE_ROOT: Path = Path(__file__).parent

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TransportMode(str, Enum):
    """Top-level transport mode passed into TravelTimeRequest."""

    WALKING = "walking"
    CYCLING = "cycling"
    DRIVING = "driving"
    TRANSIT = "transit"


class TimeContext(str, Enum):
    """Time-of-day context that drives congestion / speed assumptions."""

    RUSH_HOUR = "rush_hour"
    OFF_PEAK = "off_peak"
    WEEKEND = "weekend"
    NIGHT = "night"


class TransitCoverage(str, Enum):
    """Whether transit follows the road network freely or fixed lines only."""

    ROAD_NETWORK = "road_network"
    FIXED_ROUTES = "fixed_routes"


class VizLayer(str, Enum):
    """Named visualisation layers that folium_viz can render."""

    ISOCHRONE = "isochrone"
    NETWORK = "network"
    POI_SCATTER = "poi_scatter"
    HEATMAP = "heatmap"


# ---------------------------------------------------------------------------
# Column names (GeoDataFrame outputs)
# ---------------------------------------------------------------------------

COL_GEOMETRY = "geometry"
COL_DURATION_MIN = "duration_min"
COL_MODE = "mode"
COL_TIME_CONTEXT = "time_context"
COL_POI_NAME = "name"
COL_POI_TYPE = "amenity"

# ---------------------------------------------------------------------------
# Numeric defaults
# ---------------------------------------------------------------------------

DEFAULT_DURATION_MINUTES: int = 30
DEFAULT_WALKING_SPEED_KPH: tuple[float, float] = (4.0, 5.0)
DEFAULT_CYCLING_SPEED_KPH: tuple[float, float] = (12.0, 20.0)
DEFAULT_DRIVING_SPEED_KPH: tuple[float, float] = (30.0, 60.0)


class SurfaceBackend(str, Enum):
    """Named surface-routing backends."""

    OSMNX = "osmnx"
    FRICTION = "friction"
    ENSEMBLE = "ensemble"
