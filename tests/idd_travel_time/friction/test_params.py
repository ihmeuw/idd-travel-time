"""Parameter tables: completeness, provenance rules and unit conversion."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from idd_travel_time.friction import params as pr


def test_shipped_tables_load_and_are_complete() -> None:
    fp = pr.load_params()
    assert set(fp.land_cover_kph) == set(pr.LAND_COVER_SUBTYPES)
    assert set(fp.road_crosswalk) == set(pr.ROAD_CLASSES)
    assert set(fp.water) == set(pr.WATER_CLASSES)
    assert fp.slope.cap_kph == 5.0
    assert fp.slope.symmetric is True


def test_v0_land_cover_values_match_decisions() -> None:
    lc = pr.load_params().land_cover_kph
    assert lc["forest"] == pytest.approx(
        (1.62 + 3.24 + 3.24 + 4.00 + 3.24) / 5, abs=0.01
    )
    assert lc["shrub"] == pytest.approx(3.6)
    assert lc["moss"] == pytest.approx((3.0 + 1.62) / 2, abs=0.01)
    assert lc["mangrove"] == 2.0
    assert lc["urban"] == 5.0


def test_road_speed_country_then_global_default() -> None:
    fp = pr.load_params()
    assert fp.road_speed("UGA", "motorway") == 100.0  # Weiss 2020 ST1, Uganda row
    assert fp.road_speed("-", "motorway") == 105.0
    assert fp.road_speed(None, "unknown") == 30.0  # unknown -> Weiss generic "road"
    assert (
        fp.road_speed("UGA", "unknown") == 30.0
    )  # Uganda has no "road" row: global default
    assert fp.road_speed("ZZZ", "trunk") == 80.0  # unknown country -> global default


def test_water_rule_roles() -> None:
    w = pr.load_params().water
    assert w["river_water"].motorised_kph == 10.0
    assert w["river_water"].walking_kph == 1.0
    assert w["coastal_ocean"].motorised_kph == 19.0
    assert w["coastal_ocean"].walking_kph is None
    assert w["coastal_ocean"].walking_masked is True
    assert w["stream_water"].motorised_kph is None
    assert w["stream_water"].role == "ignore"


def test_speed_to_friction_minutes_per_metre() -> None:
    assert pr.speed_to_friction(60.0) == pytest.approx(0.001)  # Weiss 2018 example
    assert pr.speed_to_friction(5.0) == pytest.approx(0.012)
    with pytest.raises(ValueError, match="positive"):
        pr.speed_to_friction(0.0)


def test_incomplete_land_cover_table_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "lc.csv"
    p.write_text("land_cover_subtype,walking_speed_kph,source\ncrop,2.5,x\n")
    with pytest.raises(ValueError, match="missing"):
        pr.load_land_cover(p)


def test_crosswalk_target_without_default_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    lines = ["overture_class,weiss_osm_class,note"] + [
        f"{c},nosuch,x" for c in pr.ROAD_CLASSES
    ]
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="no global default"):
        pr.load_road_crosswalk(p, {("-", "motorway"): 105.0})


def test_slope_rule_requires_offroad_only(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(
        "tobler: {base_kph: 6, coefficient: -3.5, offset: 0.05, cap_kph: 5, symmetric: true}\n"
        "elevation: {scale: 1.016, rate: -0.0001072}\napply_to: everything\n"
    )
    with pytest.raises(ValueError, match="offroad_only"):
        pr.load_slope_rule(p)
    assert math.isfinite(pr.load_params().slope.elevation_rate)
