"""Core dataclasses for idd-travel-time.

TravelTimeRequest  — what the caller wants
TravelTimeResult   — what comes back
TransitProfile     — parameterised transit speed assumptions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from idd_travel_time.constants import (
    DEFAULT_DURATION_MINUTES,
    TimeContext,
    TransitCoverage,
    TransportMode,
    VizLayer,
)

if TYPE_CHECKING:
    import geopandas as gpd


# ---------------------------------------------------------------------------
# TransitProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitProfile:
    """Parameterised transit speed assumption for one time-of-day context.

    Not schedule-exact — intentionally uses typical-speed ranges to support
    global use including cities with informal / unscheduled transit.

    Parameters
    ----------
    typical_speed_kph:
        (min, max) speed range in km/h under the given congestion context.
    coverage:
        Whether transit can use the full road network (informal buses,
        mototaxis) or only fixed lines (metro, formal bus).
    time_context:
        Time-of-day context that drove these assumptions.
    mode_name:
        Human-readable label, e.g. "Informal minibus (off-peak)".
    """

    typical_speed_kph: tuple[float, float]
    coverage: TransitCoverage
    time_context: TimeContext
    mode_name: str

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        lo, hi = self.typical_speed_kph
        if lo <= 0:
            msg = f"typical_speed_kph lower bound must be > 0, got {lo}"
            raise ValueError(msg)
        if lo > hi:
            msg = (
                f"typical_speed_kph lower bound ({lo}) must be <= upper bound ({hi})"
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def midpoint_speed_kph(self) -> float:
        """Midpoint of the speed range; used as the point estimate for routing."""
        lo, hi = self.typical_speed_kph
        return (lo + hi) / 2.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        typical_speed_kph: tuple[float, float] | None = None,
        coverage: TransitCoverage | None = None,
        time_context: TimeContext | None = None,
        mode_name: str | None = None,
    ) -> TransitProfile:
        """Construct from a named preset, with optional per-parameter overrides.

        Parameters
        ----------
        name:
            One of the registered preset names (e.g. ``"informal_minibus"``).
        typical_speed_kph:
            Override the speed range tuple.
        coverage:
            Override the coverage type.
        time_context:
            Override the time context.
        mode_name:
            Override the human-readable label.

        Returns
        -------
        TransitProfile
            A new (frozen) TransitProfile instance.

        Raises
        ------
        KeyError
            If *name* is not found in the preset registry.
        """
        # Import here to avoid circular import at module load time.
        from idd_travel_time.profiles import get_preset  # noqa: PLC0415

        base = get_preset(name)
        return cls(
            typical_speed_kph=typical_speed_kph
            if typical_speed_kph is not None
            else base.typical_speed_kph,
            coverage=coverage if coverage is not None else base.coverage,
            time_context=time_context if time_context is not None else base.time_context,
            mode_name=mode_name if mode_name is not None else base.mode_name,
        )


# ---------------------------------------------------------------------------
# TravelTimeRequest
# ---------------------------------------------------------------------------


@dataclass
class TravelTimeRequest:
    """Everything needed to compute one isochrone.

    Parameters
    ----------
    origin_lat, origin_lon:
        WGS-84 coordinates of the origin point.
    mode:
        Top-level transport mode.
    duration_minutes:
        Reachability horizon in minutes.
    time_context:
        Time-of-day context; drives speed assumptions for driving / transit.
    transit_profile:
        Required when *mode* is ``TransportMode.TRANSIT``. Ignored otherwise.
    viz_layers:
        Which visualisation layers to include in the result. Defaults to all four.
    """

    origin_lat: float
    origin_lon: float
    mode: TransportMode
    duration_minutes: int = DEFAULT_DURATION_MINUTES
    time_context: TimeContext = TimeContext.OFF_PEAK
    transit_profile: TransitProfile | None = None
    viz_layers: list[VizLayer] = field(
        default_factory=lambda: list(VizLayer)
    )

    def __post_init__(self) -> None:
        if not (-90.0 <= self.origin_lat <= 90.0):
            msg = f"origin_lat must be in [-90, 90], got {self.origin_lat}"
            raise ValueError(msg)
        if not (-180.0 <= self.origin_lon <= 180.0):
            msg = f"origin_lon must be in [-180, 180], got {self.origin_lon}"
            raise ValueError(msg)
        if self.duration_minutes <= 0:
            msg = f"duration_minutes must be > 0, got {self.duration_minutes}"
            raise ValueError(msg)
        if self.mode is TransportMode.TRANSIT and self.transit_profile is None:
            msg = (
                "transit_profile is required when mode=TransportMode.TRANSIT. "
                "Use TransitProfile.from_preset('informal_minibus') or construct one directly."
            )
            raise ValueError(msg)

    @classmethod
    def from_place(
        cls,
        place: str,
        mode: TransportMode,
        *,
        country_hint: str | None = None,
        duration_minutes: int = DEFAULT_DURATION_MINUTES,
        time_context: TimeContext = TimeContext.OFF_PEAK,
        transit_profile: TransitProfile | None = None,
        viz_layers: list[VizLayer] | None = None,
    ) -> TravelTimeRequest:
        """Construct a request by geocoding a place name or address.

        Resolves *place* to (lat, lon) via OpenStreetMap Nominatim, then
        returns a fully constructed ``TravelTimeRequest``.

        Parameters
        ----------
        place:
            Free-text place name or address, e.g.
            ``"Kenyatta National Hospital, Nairobi"`` or
            ``"Tafawa Balewa Square, Lagos"``.
        mode:
            Transport mode.
        country_hint:
            ISO 3166-1 alpha-2 code to bias geocoding, e.g. ``"NG"``.
        duration_minutes:
            Isochrone duration. Default 30.
        time_context:
            Time-of-day context. Default ``OFF_PEAK``.
        transit_profile:
            Required when *mode* is ``TRANSIT``.
        viz_layers:
            Layers to include. Defaults to all four.

        Returns
        -------
        TravelTimeRequest

        Raises
        ------
        idd_travel_time.locations.GeocodingError
            If the place name cannot be resolved.

        Examples
        --------
        >>> req = TravelTimeRequest.from_place(
        ...     "Kenyatta National Hospital, Nairobi",
        ...     TransportMode.WALKING,
        ...     duration_minutes=20,
        ... )
        >>> req = TravelTimeRequest.from_place(
        ...     "Lagos Island",
        ...     TransportMode.TRANSIT,
        ...     country_hint="NG",
        ...     transit_profile=TransitProfile.from_preset("informal_minibus"),
        ... )
        """
        from idd_travel_time.locations import geocode  # noqa: PLC0415

        lat, lon = geocode(place, country_hint=country_hint)
        return cls(
            origin_lat=lat,
            origin_lon=lon,
            mode=mode,
            duration_minutes=duration_minutes,
            time_context=time_context,
            transit_profile=transit_profile,
            viz_layers=viz_layers if viz_layers is not None else list(VizLayer),
        )


# ---------------------------------------------------------------------------
# TravelTimeResult
# ---------------------------------------------------------------------------


@dataclass
class TravelTimeResult:
    """Output of one isochrone computation.

    Parameters
    ----------
    request:
        The originating request (for provenance).
    isochrone_gdf:
        GeoDataFrame with a single polygon row representing the reachable area.
        CRS is EPSG:4326. Columns: geometry, duration_min, mode, time_context.
    network_gdf:
        GeoDataFrame of the reachable road/path edges. May be None if the
        router does not support network extraction.
    poi_gdf:
        GeoDataFrame of named places (POIs) within the isochrone. May be None
        if POI extraction was not requested or not available.
    router_name:
        Name of the backend that produced this result (e.g. ``"openroute"``).
    """

    request: TravelTimeRequest
    isochrone_gdf: gpd.GeoDataFrame
    network_gdf: gpd.GeoDataFrame | None
    poi_gdf: gpd.GeoDataFrame | None
    router_name: str

    @property
    def origin(self) -> tuple[float, float]:
        """(lat, lon) of the origin point."""
        return (self.request.origin_lat, self.request.origin_lon)

    def to_parquet(self, output_dir: str | Path) -> Path:
        """Serialise this result to a GeoParquet result directory.

        Convenience wrapper around :func:`idd_travel_time.io.write_result`.

        Parameters
        ----------
        output_dir:
            Parent directory. Created if it does not exist.

        Returns
        -------
        Path
            The result directory (``output_dir / result_id``).
        """
        from idd_travel_time.io import write_result  # noqa: PLC0415

        return write_result(self, Path(output_dir))

    @classmethod
    def from_parquet(cls, result_dir: str | Path) -> TravelTimeResult:
        """Deserialise a TravelTimeResult from a GeoParquet result directory.

        Convenience wrapper around :func:`idd_travel_time.io.read_result`.

        Parameters
        ----------
        result_dir:
            Path to a directory produced by :meth:`to_parquet`.

        Returns
        -------
        TravelTimeResult
        """
        from idd_travel_time.io import read_result  # noqa: PLC0415

        return read_result(Path(result_dir))
