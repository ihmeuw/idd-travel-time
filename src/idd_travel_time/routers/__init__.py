"""Routing backends for idd-travel-time.

Public surface
--------------
Router                  : abstract base class
OpenRouteServiceRouter  : default free backend (ORS REST API)
OsmnxRouter             : local fallback (no API key)
GoogleMapsRouter        : paid stub (not yet implemented)
"""

from idd_travel_time.routers.base import Router
from idd_travel_time.routers.google import GoogleMapsRouter
from idd_travel_time.routers.openroute import OpenRouteServiceRouter
from idd_travel_time.routers.osmnx_router import OsmnxRouter

__all__ = [
    "GoogleMapsRouter",
    "OpenRouteServiceRouter",
    "OsmnxRouter",
    "Router",
]
