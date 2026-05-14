"""Location lookup for idd-travel-time.

Public surface
--------------
geocode       : (place, *, country_hint?, language?) -> (lat, lon)
geocode_many  : (places, *, country_hint?, language?, on_error?) -> list[(lat, lon) | None]
GeocodingError: exception raised when a place cannot be resolved
"""

from idd_travel_time.locations.geocode import GeocodingError, geocode, geocode_many

__all__ = ["GeocodingError", "geocode", "geocode_many"]
