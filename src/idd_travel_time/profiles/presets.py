"""Named TransitProfile presets.

These are the five canonical presets documented in the project seed. Every
parameter is overridable via TransitProfile.from_preset(..., key=value).

The PRESETS dict is the single source of truth — do not duplicate entries
elsewhere. To add a new preset, add it here and nowhere else.
"""

from __future__ import annotations

from idd_travel_time.constants import TimeContext, TransitCoverage
from idd_travel_time.models import TransitProfile

# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------
# Key   : preset name (str, used in TransitProfile.from_preset("name"))
# Value : frozen TransitProfile instance
# ---------------------------------------------------------------------------

PRESETS: dict[str, TransitProfile] = {
    # ------------------------------------------------------------------
    # formal_bus
    # Developed-world fixed-route bus service.
    # Speed range reflects mixed urban/suburban operation with stops.
    # ------------------------------------------------------------------
    "formal_bus": TransitProfile(
        typical_speed_kph=(15.0, 30.0),
        coverage=TransitCoverage.FIXED_ROUTES,
        time_context=TimeContext.OFF_PEAK,
        mode_name="Formal bus (off-peak)",
    ),
    # ------------------------------------------------------------------
    # informal_minibus
    # Sub-Saharan Africa and SE Asia: danfo (Lagos), matatu (Nairobi),
    # tro-tro (Accra). No fixed timetable; follows road network.
    # Slightly slower than mototaxi due to passenger loading/unloading.
    # ------------------------------------------------------------------
    "informal_minibus": TransitProfile(
        typical_speed_kph=(15.0, 25.0),
        coverage=TransitCoverage.ROAD_NETWORK,
        time_context=TimeContext.OFF_PEAK,
        mode_name="Informal minibus (off-peak)",
    ),
    # ------------------------------------------------------------------
    # mototaxi
    # Latin America (mototaxi), West Africa (okada/boda-boda).
    # Faster than minibus because it filters through traffic.
    # Road-network coverage: can go anywhere a motorcycle can.
    # ------------------------------------------------------------------
    "mototaxi": TransitProfile(
        typical_speed_kph=(20.0, 35.0),
        coverage=TransitCoverage.ROAD_NETWORK,
        time_context=TimeContext.OFF_PEAK,
        mode_name="Mototaxi / okada (off-peak)",
    ),
    # ------------------------------------------------------------------
    # metro
    # Cities with subway or light rail (Nairobi SGR, Lagos Blue Line,
    # Jakarta MRT, etc.).
    # Fixed routes only; higher speed because grade-separated.
    # ------------------------------------------------------------------
    "metro": TransitProfile(
        typical_speed_kph=(30.0, 60.0),
        coverage=TransitCoverage.FIXED_ROUTES,
        time_context=TimeContext.OFF_PEAK,
        mode_name="Metro / light rail (off-peak)",
    ),
    # ------------------------------------------------------------------
    # tuk_tuk
    # South and SE Asia three-wheelers (auto-rickshaw, bajaj).
    # Road-network coverage; speed similar to informal minibus.
    # ------------------------------------------------------------------
    "tuk_tuk": TransitProfile(
        typical_speed_kph=(15.0, 25.0),
        coverage=TransitCoverage.ROAD_NETWORK,
        time_context=TimeContext.OFF_PEAK,
        mode_name="Tuk-tuk / auto-rickshaw (off-peak)",
    ),
}


def list_presets() -> list[str]:
    """Return sorted list of all registered preset names."""
    return sorted(PRESETS)


def get_preset(name: str) -> TransitProfile:
    """Return a preset by name, raising ``KeyError`` with helpful message if missing.

    Parameters
    ----------
    name:
        Preset name, e.g. ``"informal_minibus"``.

    Raises
    ------
    KeyError
        If *name* is not registered. The error message lists available presets.
    """
    if name not in PRESETS:
        available = ", ".join(list_presets())
        msg = f"Unknown preset {name!r}. Available presets: {available}"
        raise KeyError(msg)
    return PRESETS[name]
