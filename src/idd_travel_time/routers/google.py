"""Google Maps routing backend (paid stub).

This module implements the Router interface using the Google Maps
Distance Matrix / Routes API. It is a stub — the interface is wired up
and ready to use, but the underlying HTTP calls are not implemented.

To activate
-----------
1. Obtain a Google Maps Platform API key with the Routes API enabled.
2. Install the client: ``pip install googlemaps``
3. Set ``GOOGLE_MAPS_API_KEY`` in your environment.
4. Replace the ``NotImplementedError`` bodies below with real API calls.

Why keep the stub?
------------------
Having the class in place means callers can write:

    router: Router = OpenRouteServiceRouter()   # default
    # or:
    router: Router = GoogleMapsRouter()         # paid swap-in

…without any other code changes. The Router interface is the contract;
the backend is a one-line swap.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from idd_travel_time.routers.base import Router

if TYPE_CHECKING:
    from idd_travel_time.models import TravelTimeRequest, TravelTimeResult


class GoogleMapsRouter(Router):
    """Isochrone router backed by the Google Maps Platform (stub).

    Parameters
    ----------
    api_key:
        Google Maps API key. Defaults to ``GOOGLE_MAPS_API_KEY`` env var.

    Raises
    ------
    RuntimeError
        If no API key is found at construction time.
    """

    def __init__(self, api_key: str | None = None) -> None:
        resolved = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not resolved:
            msg = (
                "Google Maps API key is required. Set the GOOGLE_MAPS_API_KEY "
                "environment variable or pass api_key= explicitly."
            )
            raise RuntimeError(msg)
        self._api_key = resolved

    @property
    def name(self) -> str:
        return "google_maps"

    def compute_isochrone(self, request: TravelTimeRequest) -> TravelTimeResult:
        """Not yet implemented.

        When implemented, this should:
        1. Use the Google Routes API or Distance Matrix API to compute
           travel times from the origin to a grid of candidate points.
        2. Interpolate a polygon at the requested duration contour.
        3. Optionally extract POIs via the Google Places API.

        Raises
        ------
        NotImplementedError
            Always, until this stub is implemented.
        """
        msg = (
            "GoogleMapsRouter.compute_isochrone is not yet implemented. "
            "Use OpenRouteServiceRouter (free) or OsmnxRouter (local) instead."
        )
        raise NotImplementedError(msg)
