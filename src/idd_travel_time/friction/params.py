"""Parameter tables for the friction surface, loaded from ``friction/tables``.

Every speed and rule comes from a file; the loader validates completeness
against the class vocabularies the builder expects and raises on any gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LAND_COVER_SUBTYPES: tuple[str, ...] = (
    "crop",
    "forest",
    "grass",
    "shrub",
    "wetland",
    "urban",
    "barren",
    "snow",
    "moss",
    "mangrove",
)
ROAD_CLASSES: tuple[str, ...] = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "service",
    "unclassified",
    "track",
    "unknown",
)
WATER_CLASSES: tuple[str, ...] = (
    "stream_water",
    "river_water",
    "inland_water",
    "manmade_freshwater",
    "waste_and_sewage",
    "coastal_ocean",
    "coastal_inlet",
)
GLOBAL_DEFAULT_ISO3 = "-"
WALKING_ROAD_SPEED_KPH = 5.0  # Weiss 2020 walking-only surface: every road at 5 km/h
MINUTES_PER_HOUR = 60.0
METRES_PER_KM = 1000.0


@dataclass(frozen=True)
class WaterRule:
    """Per water class: speed by mode, or ``None`` to ignore; ``mask`` marks ocean (walking)."""

    motorised_kph: float | None
    walking_kph: float | None
    walking_masked: bool
    role: str


@dataclass(frozen=True)
class SlopeRule:
    """Tobler's hiking function and the altitude factor (Weiss 2018 eq. 1-3)."""

    base_kph: float
    coefficient: float
    offset: float
    cap_kph: float
    symmetric: bool
    elevation_scale: float
    elevation_rate: float


@dataclass(frozen=True)
class FrictionParams:
    """All tables the builder needs, keyed by the Overture class vocabularies."""

    land_cover_kph: dict[str, float]
    road_kph: dict[
        tuple[str, str], float
    ]  # (iso3, weiss_osm_class) -> km/h; iso3 "-" = global default
    road_crosswalk: dict[str, str]  # overture class -> weiss osm class
    water: dict[str, WaterRule]
    slope: SlopeRule
    table_dir: Path

    def road_speed(self, iso3: str | None, overture_class: str) -> float:
        """Per-country speed for an Overture road class, falling back to the global default."""
        weiss_class = self.road_crosswalk[overture_class]
        if iso3 is not None and (iso3, weiss_class) in self.road_kph:
            return self.road_kph[(iso3, weiss_class)]
        return self.road_kph[(GLOBAL_DEFAULT_ISO3, weiss_class)]


def tables_dir() -> Path:
    """Directory holding the shipped parameter tables."""
    return Path(str(resources.files("idd_travel_time.friction") / "tables"))


def _check_complete(found: set[str], expected: tuple[str, ...], what: str) -> None:
    missing = set(expected) - found
    extra = found - set(expected)
    if missing or extra:
        msg = f"{what}: missing {sorted(missing)}, unexpected {sorted(extra)}"
        raise ValueError(msg)


def load_land_cover(path: Path) -> dict[str, float]:
    """Walking km/h per Overture land-cover subtype."""
    df = pd.read_csv(path)
    out = {
        str(r.land_cover_subtype): float(r.walking_speed_kph) for r in df.itertuples()
    }
    _check_complete(set(out), LAND_COVER_SUBTYPES, "land_cover_speeds")
    if any(v <= 0 for v in out.values()):
        msg = "land-cover speeds must be positive"
        raise ValueError(msg)
    return out


def load_road_table(path: Path) -> dict[tuple[str, str], float]:
    """Weiss 2020 Supplementary Table 1: (iso3, osm class) -> km/h, with global defaults."""
    df = pd.read_csv(path, dtype={"iso3": str, "osm_class": str, "speed_kph": float})
    out = {
        (str(iso3), str(osm_class)): float(kph)
        for iso3, osm_class, kph in zip(
            df["iso3"].tolist(),
            df["osm_class"].tolist(),
            df["speed_kph"].tolist(),
            strict=True,
        )
    }
    if not any(k[0] == GLOBAL_DEFAULT_ISO3 for k in out):
        msg = "road table has no global-default rows (iso3 '-')"
        raise ValueError(msg)
    return out


def load_road_crosswalk(
    path: Path, road_kph: dict[tuple[str, str], float]
) -> dict[str, str]:
    """Overture road class -> Weiss OSM class; every target must have a global default."""
    df = pd.read_csv(path)
    out = {str(r.overture_class): str(r.weiss_osm_class) for r in df.itertuples()}
    _check_complete(set(out), ROAD_CLASSES, "road_class_crosswalk")
    for weiss_class in out.values():
        if (GLOBAL_DEFAULT_ISO3, weiss_class) not in road_kph:
            msg = f"crosswalk target {weiss_class!r} has no global default in the road table"
            raise ValueError(msg)
    return out


def _speed_or_none(value: object) -> float | None:
    if (
        value is None
        or (isinstance(value, float) and pd.isna(value))
        or str(value).strip() in {"", "mask"}
    ):
        return None
    return float(str(value))


def load_water_rule(path: Path) -> dict[str, WaterRule]:
    """Per Overture water class: motorised and walking speeds, or ignore; ocean masked when walking."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out: dict[str, WaterRule] = {}
    for r in df.itertuples():
        out[str(r.water_class)] = WaterRule(
            motorised_kph=_speed_or_none(r.motorised_speed_kph),
            walking_kph=_speed_or_none(r.walking_speed_kph),
            walking_masked=str(r.walking_speed_kph).strip() == "mask",
            role=str(r.role),
        )
    _check_complete(set(out), WATER_CLASSES, "water_rule")
    return out


def load_slope_rule(path: Path) -> SlopeRule:
    """Tobler and altitude constants from the YAML rule file."""
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if raw.get("apply_to") != "offroad_only":
        msg = f"slope rule apply_to={raw.get('apply_to')!r}; only 'offroad_only' is implemented"
        raise ValueError(msg)
    t, e = raw["tobler"], raw["elevation"]
    return SlopeRule(
        base_kph=float(t["base_kph"]),
        coefficient=float(t["coefficient"]),
        offset=float(t["offset"]),
        cap_kph=float(t["cap_kph"]),
        symmetric=bool(t["symmetric"]),
        elevation_scale=float(e["scale"]),
        elevation_rate=float(e["rate"]),
    )


def load_params(table_dir: Path | None = None) -> FrictionParams:
    """Load every table from *table_dir* (default: the shipped ``friction/tables``)."""
    d = tables_dir() if table_dir is None else table_dir
    road_kph = load_road_table(d / "road_speeds_weiss2020_st1.csv")
    return FrictionParams(
        land_cover_kph=load_land_cover(d / "land_cover_speeds_v0.csv"),
        road_kph=road_kph,
        road_crosswalk=load_road_crosswalk(d / "road_class_crosswalk_v0.csv", road_kph),
        water=load_water_rule(d / "water_rule_v0.csv"),
        slope=load_slope_rule(d / "slope_rule_v0.yaml"),
        table_dir=d,
    )


def speed_to_friction(speed_kph: float) -> float:
    """Minutes needed to travel one metre at *speed_kph* (the friction raster's unit)."""
    if speed_kph <= 0:
        msg = f"speed must be positive, got {speed_kph}"
        raise ValueError(msg)
    return MINUTES_PER_HOUR / (speed_kph * METRES_PER_KM)
