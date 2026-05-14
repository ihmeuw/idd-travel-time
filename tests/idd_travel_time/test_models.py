"""Tests for idd_travel_time.models."""

from __future__ import annotations

import pytest

from idd_travel_time.constants import TimeContext, TransitCoverage, TransportMode, VizLayer
from idd_travel_time.models import TransitProfile, TravelTimeRequest


# ---------------------------------------------------------------------------
# TransitProfile
# ---------------------------------------------------------------------------


class TestTransitProfile:
    def test_basic_construction(self) -> None:
        p = TransitProfile(
            typical_speed_kph=(15.0, 25.0),
            coverage=TransitCoverage.ROAD_NETWORK,
            time_context=TimeContext.OFF_PEAK,
            mode_name="Test profile",
        )
        assert p.typical_speed_kph == (15.0, 25.0)
        assert p.coverage is TransitCoverage.ROAD_NETWORK
        assert p.time_context is TimeContext.OFF_PEAK
        assert p.mode_name == "Test profile"

    def test_midpoint_speed_kph(self) -> None:
        p = TransitProfile(
            typical_speed_kph=(10.0, 30.0),
            coverage=TransitCoverage.ROAD_NETWORK,
            time_context=TimeContext.OFF_PEAK,
            mode_name="Test",
        )
        assert p.midpoint_speed_kph == pytest.approx(20.0)

    def test_frozen(self) -> None:
        p = TransitProfile(
            typical_speed_kph=(15.0, 25.0),
            coverage=TransitCoverage.ROAD_NETWORK,
            time_context=TimeContext.OFF_PEAK,
            mode_name="Test",
        )
        with pytest.raises((AttributeError, TypeError)):
            p.mode_name = "mutated"  # type: ignore[misc]

    def test_invalid_speed_lower_bound_zero(self) -> None:
        with pytest.raises(ValueError, match="lower bound must be > 0"):
            TransitProfile(
                typical_speed_kph=(0.0, 20.0),
                coverage=TransitCoverage.ROAD_NETWORK,
                time_context=TimeContext.OFF_PEAK,
                mode_name="Bad",
            )

    def test_invalid_speed_lower_exceeds_upper(self) -> None:
        with pytest.raises(ValueError, match="lower bound.*<= upper bound"):
            TransitProfile(
                typical_speed_kph=(30.0, 10.0),
                coverage=TransitCoverage.ROAD_NETWORK,
                time_context=TimeContext.OFF_PEAK,
                mode_name="Bad",
            )

    def test_equal_lower_upper_is_valid(self) -> None:
        p = TransitProfile(
            typical_speed_kph=(20.0, 20.0),
            coverage=TransitCoverage.ROAD_NETWORK,
            time_context=TimeContext.OFF_PEAK,
            mode_name="Point speed",
        )
        assert p.midpoint_speed_kph == pytest.approx(20.0)

    def test_from_preset_no_override(self) -> None:
        p = TransitProfile.from_preset("informal_minibus")
        assert p.typical_speed_kph == (15.0, 25.0)
        assert p.coverage is TransitCoverage.ROAD_NETWORK

    def test_from_preset_speed_override(self) -> None:
        p = TransitProfile.from_preset("informal_minibus", typical_speed_kph=(10.0, 20.0))
        assert p.typical_speed_kph == (10.0, 20.0)
        # other fields unchanged
        assert p.coverage is TransitCoverage.ROAD_NETWORK

    def test_from_preset_time_context_override(self) -> None:
        p = TransitProfile.from_preset("formal_bus", time_context=TimeContext.RUSH_HOUR)
        assert p.time_context is TimeContext.RUSH_HOUR

    def test_from_preset_all_overrides(self) -> None:
        p = TransitProfile.from_preset(
            "metro",
            typical_speed_kph=(20.0, 40.0),
            coverage=TransitCoverage.ROAD_NETWORK,
            time_context=TimeContext.NIGHT,
            mode_name="Night metro hack",
        )
        assert p.typical_speed_kph == (20.0, 40.0)
        assert p.coverage is TransitCoverage.ROAD_NETWORK
        assert p.time_context is TimeContext.NIGHT
        assert p.mode_name == "Night metro hack"

    def test_from_preset_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown preset"):
            TransitProfile.from_preset("nonexistent_preset")


# ---------------------------------------------------------------------------
# TravelTimeRequest
# ---------------------------------------------------------------------------


class TestTravelTimeRequest:
    def _walking_request(self, **kwargs) -> TravelTimeRequest:  # type: ignore[no-untyped-def]
        defaults = {
            "origin_lat": -1.286389,
            "origin_lon": 36.817223,
            "mode": TransportMode.WALKING,
        }
        return TravelTimeRequest(**{**defaults, **kwargs})

    def test_basic_walking(self) -> None:
        req = self._walking_request()
        assert req.origin_lat == pytest.approx(-1.286389)
        assert req.mode is TransportMode.WALKING
        assert req.duration_minutes == 30  # default

    def test_default_viz_layers_all_four(self) -> None:
        req = self._walking_request()
        assert set(req.viz_layers) == set(VizLayer)

    def test_custom_duration(self) -> None:
        req = self._walking_request(duration_minutes=45)
        assert req.duration_minutes == 45

    def test_invalid_lat_too_high(self) -> None:
        with pytest.raises(ValueError, match="origin_lat"):
            self._walking_request(origin_lat=91.0)

    def test_invalid_lat_too_low(self) -> None:
        with pytest.raises(ValueError, match="origin_lat"):
            self._walking_request(origin_lat=-91.0)

    def test_invalid_lon_too_high(self) -> None:
        with pytest.raises(ValueError, match="origin_lon"):
            self._walking_request(origin_lon=181.0)

    def test_invalid_duration_zero(self) -> None:
        with pytest.raises(ValueError, match="duration_minutes"):
            self._walking_request(duration_minutes=0)

    def test_invalid_duration_negative(self) -> None:
        with pytest.raises(ValueError, match="duration_minutes"):
            self._walking_request(duration_minutes=-5)

    def test_transit_without_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="transit_profile is required"):
            TravelTimeRequest(
                origin_lat=0.0,
                origin_lon=0.0,
                mode=TransportMode.TRANSIT,
            )

    def test_transit_with_profile_ok(self) -> None:
        profile = TransitProfile.from_preset("mototaxi")
        req = TravelTimeRequest(
            origin_lat=6.5244,
            origin_lon=3.3792,
            mode=TransportMode.TRANSIT,
            transit_profile=profile,
        )
        assert req.transit_profile is profile

    def test_driving_without_profile_ok(self) -> None:
        req = TravelTimeRequest(
            origin_lat=0.0,
            origin_lon=0.0,
            mode=TransportMode.DRIVING,
        )
        assert req.transit_profile is None

    def test_boundary_coords_valid(self) -> None:
        # Corners of the world should be valid
        for lat, lon in [(-90, -180), (-90, 180), (90, -180), (90, 180)]:
            req = TravelTimeRequest(
                origin_lat=lat,
                origin_lon=lon,
                mode=TransportMode.WALKING,
            )
            assert req.origin_lat == lat
