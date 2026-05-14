"""Tests for idd_travel_time.run_isochrone CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pytest
from click.testing import CliRunner
from shapely.geometry import Polygon

from idd_travel_time.models import TravelTimeRequest, TravelTimeResult
from idd_travel_time.run_isochrone import main

_SQUARE = Polygon([(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)])


def _fake_result(
    request: TravelTimeRequest, *, router: object = None
) -> TravelTimeResult:
    iso_gdf = gpd.GeoDataFrame(
        {
            "geometry": [_SQUARE],
            "duration_min": [request.duration_minutes],
            "mode": [request.mode.value],
            "time_context": [request.time_context.value],
        },
        crs="EPSG:4326",
    )
    return TravelTimeResult(
        request=request,
        isochrone_gdf=iso_gdf,
        network_gdf=None,
        poi_gdf=None,
        router_name="mock",
    )


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def output_file(tmp_path: Path) -> Path:
    return tmp_path / "out.html"


def _base_args(output_file: Path) -> list[str]:
    return [
        "--lat",
        "-1.2864",
        "--lon",
        "36.8172",
        "--mode",
        "walking",
        "--output",
        str(output_file),
    ]


class TestCliBasicWalking:
    def test_exits_zero(self, runner: CliRunner, output_file: Path) -> None:
        with (
            patch("idd_travel_time.run_isochrone.compute", side_effect=_fake_result),
            patch("idd_travel_time.run_isochrone.save_map", return_value=output_file),
        ):
            result = runner.invoke(main, _base_args(output_file))
        assert result.exit_code == 0, result.output

    def test_output_path_in_stdout(self, runner: CliRunner, output_file: Path) -> None:
        with (
            patch("idd_travel_time.run_isochrone.compute", side_effect=_fake_result),
            patch("idd_travel_time.run_isochrone.save_map", return_value=output_file),
        ):
            result = runner.invoke(main, _base_args(output_file))
        assert str(output_file) in result.output

    def test_missing_lat_fails(self, runner: CliRunner, output_file: Path) -> None:
        args = ["--lon", "36.8172", "--mode", "walking", "--output", str(output_file)]
        result = runner.invoke(main, args)
        assert result.exit_code != 0

    def test_missing_mode_fails(self, runner: CliRunner, output_file: Path) -> None:
        args = ["--lat", "-1.2864", "--lon", "36.8172", "--output", str(output_file)]
        result = runner.invoke(main, args)
        assert result.exit_code != 0

    def test_invalid_mode_fails(self, runner: CliRunner, output_file: Path) -> None:
        args = _base_args(output_file)
        args[args.index("walking")] = "teleportation"
        result = runner.invoke(main, args)
        assert result.exit_code != 0


class TestCliTransitMode:
    def test_transit_without_preset_fails(
        self, runner: CliRunner, output_file: Path
    ) -> None:
        args = [
            "--lat",
            "-1.2864",
            "--lon",
            "36.8172",
            "--mode",
            "transit",
            "--output",
            str(output_file),
        ]
        with (
            patch("idd_travel_time.run_isochrone.compute", side_effect=_fake_result),
            patch("idd_travel_time.run_isochrone.save_map", return_value=output_file),
        ):
            result = runner.invoke(main, args)
        assert result.exit_code != 0

    def test_transit_with_preset_succeeds(
        self, runner: CliRunner, output_file: Path
    ) -> None:
        args = [
            "--lat",
            "-1.2864",
            "--lon",
            "36.8172",
            "--mode",
            "transit",
            "--transit-preset",
            "informal_minibus",
            "--output",
            str(output_file),
        ]
        with (
            patch("idd_travel_time.run_isochrone.compute", side_effect=_fake_result),
            patch("idd_travel_time.run_isochrone.save_map", return_value=output_file),
        ):
            result = runner.invoke(main, args)
        assert result.exit_code == 0, result.output


class TestCliRouterSelection:
    def test_router_osmnx_flag(self, runner: CliRunner, output_file: Path) -> None:
        args = [*_base_args(output_file), "--router", "osmnx"]
        with (
            patch("idd_travel_time.run_isochrone.compute", side_effect=_fake_result),
            patch("idd_travel_time.run_isochrone.save_map", return_value=output_file),
        ):
            result = runner.invoke(main, args)
        assert result.exit_code == 0, result.output

    def test_invalid_router_fails(self, runner: CliRunner, output_file: Path) -> None:
        args = [*_base_args(output_file), "--router", "bicycle"]
        result = runner.invoke(main, args)
        assert result.exit_code != 0


class TestCliLayerSelection:
    def test_single_layer(self, runner: CliRunner, output_file: Path) -> None:
        args = [*_base_args(output_file), "--layers", "isochrone"]
        with (
            patch("idd_travel_time.run_isochrone.compute", side_effect=_fake_result),
            patch("idd_travel_time.run_isochrone.save_map", return_value=output_file),
        ):
            result = runner.invoke(main, args)
        assert result.exit_code == 0, result.output

    def test_multiple_layers(self, runner: CliRunner, output_file: Path) -> None:
        args = [
            *_base_args(output_file),
            "--layers",
            "isochrone",
            "--layers",
            "heatmap",
        ]
        with (
            patch("idd_travel_time.run_isochrone.compute", side_effect=_fake_result),
            patch("idd_travel_time.run_isochrone.save_map", return_value=output_file),
        ):
            result = runner.invoke(main, args)
        assert result.exit_code == 0, result.output
