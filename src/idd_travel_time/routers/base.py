"""Abstract Router base class.

Every routing backend must implement this interface. Swapping ORS for Google
Maps (or any other backend) is a one-line change at the call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from idd_travel_time.models import TravelTimeResult, TravelTimeRequest


class Router(ABC):
    """Abstract base for isochrone routing backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend, e.g. ``"openroute"``."""

    @abstractmethod
    def compute_isochrone(self, request: TravelTimeRequest) -> TravelTimeResult:
        """Compute an isochrone for *request* and return a TravelTimeResult.

        Parameters
        ----------
        request:
            Fully populated TravelTimeRequest.

        Returns
        -------
        TravelTimeResult
            Isochrone polygon (always present), plus optional network/POI GeoDataFrames.
        """
