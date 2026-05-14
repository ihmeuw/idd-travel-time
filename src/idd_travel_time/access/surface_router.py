"""Abstract SurfaceRouter interface.

A SurfaceRouter takes a single destination point and a geographic bounding box
and returns a TravelTimeSurface: a raster where every pixel value is the
travel time in minutes from that pixel to the destination.

Design notes
------------
- The destination is where you want to *arrive* (a clinic, hospital, etc.).
- Travel time is computed from all grid cells *to* the destination.
  On a symmetric undirected graph this equals time from destination to all
  cells, which is how single-source Dijkstra / MCP naturally works.
- Resolution is specified in degrees for simplicity; ~0.01° ≈ 1.1 km,
  ~0.001° ≈ 111 m, ~0.0004° ≈ 44 m.
- mode and transit_profile are optional; backends that don't support a
  given mode should raise NotImplementedError.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from idd_travel_time.access.surface import BoundingBox, TravelTimeSurface
    from idd_travel_time.constants import TransportMode
    from idd_travel_time.models import TransitProfile


class SurfaceRouter(ABC):
    """Abstract base for travel-time surface backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend, e.g. ``"osmnx"``."""

    @abstractmethod
    def compute_surface(
        self,
        destination: tuple[float, float],
        bbox: BoundingBox,
        resolution_deg: float,
        mode: TransportMode,
        *,
        transit_profile: TransitProfile | None = None,
    ) -> TravelTimeSurface:
        """Compute a travel-time surface to *destination* over *bbox*.

        Parameters
        ----------
        destination:
            ``(lat, lon)`` of the target facility.
        bbox:
            Geographic bounding box of the output raster.
        resolution_deg:
            Pixel size in degrees (square).  Typical values:
            0.01 (~1.1 km), 0.001 (~111 m), 0.0004 (~44 m).
        mode:
            Transport mode.
        transit_profile:
            Required when ``mode == TRANSIT``.

        Returns
        -------
        TravelTimeSurface
            Raster with travel times in minutes, NaN where unreachable.
        """
