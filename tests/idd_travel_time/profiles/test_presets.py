"""Tests for idd_travel_time.profiles.presets."""

from __future__ import annotations

import pytest

from idd_travel_time.constants import TransitCoverage
from idd_travel_time.models import TransitProfile
from idd_travel_time.profiles import PRESETS, get_preset, list_presets

# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

EXPECTED_PRESETS = {
    "formal_bus",
    "informal_minibus",
    "mototaxi",
    "metro",
    "tuk_tuk",
}


def test_all_expected_presets_registered() -> None:
    assert set(PRESETS.keys()) == EXPECTED_PRESETS


def test_list_presets_sorted() -> None:
    names = list_presets()
    assert names == sorted(names)


def test_list_presets_contains_all() -> None:
    assert set(list_presets()) == EXPECTED_PRESETS


# ---------------------------------------------------------------------------
# get_preset
# ---------------------------------------------------------------------------


def test_get_preset_returns_transit_profile() -> None:
    p = get_preset("metro")
    assert isinstance(p, TransitProfile)


def test_get_preset_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError, match="Unknown preset"):
        get_preset("flying_carpet")


def test_get_preset_error_message_lists_available() -> None:
    with pytest.raises(KeyError) as exc_info:
        get_preset("flying_carpet")
    assert "informal_minibus" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Individual preset spot-checks
# ---------------------------------------------------------------------------


class TestFormalBus:
    def test_speed_range(self) -> None:
        p = get_preset("formal_bus")
        assert p.typical_speed_kph == (15.0, 30.0)

    def test_coverage_fixed_routes(self) -> None:
        p = get_preset("formal_bus")
        assert p.coverage is TransitCoverage.FIXED_ROUTES


class TestInformalMinibus:
    def test_speed_range(self) -> None:
        p = get_preset("informal_minibus")
        assert p.typical_speed_kph == (15.0, 25.0)

    def test_coverage_road_network(self) -> None:
        p = get_preset("informal_minibus")
        assert p.coverage is TransitCoverage.ROAD_NETWORK


class TestMototaxi:
    def test_faster_than_informal_minibus(self) -> None:
        moto = get_preset("mototaxi")
        minibus = get_preset("informal_minibus")
        assert moto.midpoint_speed_kph > minibus.midpoint_speed_kph

    def test_coverage_road_network(self) -> None:
        p = get_preset("mototaxi")
        assert p.coverage is TransitCoverage.ROAD_NETWORK


class TestMetro:
    def test_fastest_preset(self) -> None:
        metro = get_preset("metro")
        others = [get_preset(n) for n in EXPECTED_PRESETS - {"metro"}]
        assert all(metro.midpoint_speed_kph > o.midpoint_speed_kph for o in others)

    def test_coverage_fixed_routes(self) -> None:
        p = get_preset("metro")
        assert p.coverage is TransitCoverage.FIXED_ROUTES


class TestTukTuk:
    def test_speed_range(self) -> None:
        p = get_preset("tuk_tuk")
        assert p.typical_speed_kph == (15.0, 25.0)

    def test_coverage_road_network(self) -> None:
        p = get_preset("tuk_tuk")
        assert p.coverage is TransitCoverage.ROAD_NETWORK


# ---------------------------------------------------------------------------
# All presets: invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED_PRESETS))
def test_all_presets_valid_speed_range(name: str) -> None:
    p = get_preset(name)
    lo, hi = p.typical_speed_kph
    assert lo > 0
    assert lo <= hi


@pytest.mark.parametrize("name", list(EXPECTED_PRESETS))
def test_all_presets_have_mode_name(name: str) -> None:
    p = get_preset(name)
    assert p.mode_name
    assert len(p.mode_name) > 0


@pytest.mark.parametrize("name", list(EXPECTED_PRESETS))
def test_all_presets_frozen(name: str) -> None:
    p = get_preset(name)
    with pytest.raises((AttributeError, TypeError)):
        p.mode_name = "mutated"  # type: ignore[misc]
