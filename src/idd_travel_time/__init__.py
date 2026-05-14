"""idd-travel-time: travel-time isochrones, globally.

Quickstart
----------
>>> from idd_travel_time.models import TravelTimeRequest, TransitProfile
>>> from idd_travel_time.constants import TransportMode, TimeContext
>>>
>>> # Walking isochrone, 30 minutes
>>> req = TravelTimeRequest(
...     origin_lat=-1.286389,
...     origin_lon=36.817223,
...     mode=TransportMode.WALKING,
... )
>>>
>>> # Informal minibus in Nairobi
>>> req = TravelTimeRequest(
...     origin_lat=-1.286389,
...     origin_lon=36.817223,
...     mode=TransportMode.TRANSIT,
...     time_context=TimeContext.RUSH_HOUR,
...     transit_profile=TransitProfile.from_preset(
...         "informal_minibus",
...         time_context=TimeContext.RUSH_HOUR,
...         typical_speed_kph=(10.0, 18.0),
...     ),
... )
"""

from idd_travel_time.models import TransitProfile, TravelTimeRequest, TravelTimeResult

__all__ = ["TransitProfile", "TravelTimeRequest", "TravelTimeResult"]
