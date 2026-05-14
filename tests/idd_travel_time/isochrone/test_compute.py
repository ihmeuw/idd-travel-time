"""Tests for idd_travel_time.isochrone.compute."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import geopandas as gpd
from shapely.geometry import Polygon

from idd_travel_time.constants import TransportMode
from idd_travel_time.isochrone.compute import _default_router, compute, compute_batch
from idd_travel_time.models import TravelTimeRequest, TravelTimeResult
from idd_travel_time.routers.base import Router

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SQUARE = Polygon([(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)])

WALKING_REQUEST = TravelTimeRequest(
    origin_lat=-1.286389,
    origin_lon=36.817223,
    mode=TransportMode.WALKING,
    duration_minutes=30,
)


def _make_fake_result(request: TravelTimeRequest, router_name: str) -> TravelTimeResult:
    iso_gdf = gpd.GeoDataFrame(geometry=[_SQUARE], crs="EPSG:4326")
    return TravelTimeResult(
        request=request,
        isochrone_gdf=iso_gdf,
        network_gdf=None,
        poi_gdf=None,
        router_name=router_name,
    )


def _make_mock_router(name: str = "mock") -> MagicMock:
    mock = MagicMock(spec=Router)
    mock.name = name
    mock.compute_isochrone.side_effect = lambda req: _make_fake_result(req, name)
    return mock


# ---------------------------------------------------------------------------
# _default_router
# ---------------------------------------------------------------------------


class TestDefaultRouter:
    def test_uses_osmnx_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ORS_API_KEY", raising=False)
        from idd_travel_time.routers.osmnx_router import OsmnxRouter

        router = _default_router()
        assert isinstance(router, OsmnxRouter)

    def test_uses_ors_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ORS_API_KEY", "fake-key")
        with patch("openrouteservice.Client"):
            from idd_travel_time.routers.openroute import OpenRouteServiceRouter

            router = _default_router()
            assert isinstance(router, OpenRouteServiceRouter)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------


class TestCompute:
    def test_calls_router_compute_isochrone(self) -> None:
        mock = _make_mock_router()
        compute(WALKING_REQUEST, router=mock)
        mock.compute_isochrone.assert_called_once_with(WALKING_REQUEST)

    def test_returns_travel_time_result(self) -> None:
        mock = _make_mock_router()
        result = compute(WALKING_REQUEST, router=mock)
        assert isinstance(result, TravelTimeResult)

    def test_result_router_name_matches(self) -> None:
        mock = _make_mock_router("test_backend")
        result = compute(WALKING_REQUEST, router=mock)
        assert result.router_name == "test_backend"

    def test_auto_router_selected_when_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ORS_API_KEY", raising=False)
        from idd_travel_time.routers.osmnx_router import OsmnxRouter

        with patch.object(
            OsmnxRouter,
            "compute_isochrone",
            return_value=_make_fake_result(WALKING_REQUEST, "osmnx"),
        ):
            result = compute(WALKING_REQUEST, router=None)
        assert result.router_name == "osmnx"

    def test_explicit_router_overrides_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORS_API_KEY", "fake-key")
        explicit_mock = _make_mock_router("explicit")
        result = compute(WALKING_REQUEST, router=explicit_mock)
        assert result.router_name == "explicit"


# ---------------------------------------------------------------------------
# compute_batch()
# ---------------------------------------------------------------------------


class TestComputeBatch:
    def test_returns_list_of_results(self) -> None:
        mock = _make_mock_router()
        requests = [WALKING_REQUEST, WALKING_REQUEST]
        results = compute_batch(requests, router=mock)
        assert len(results) == 2
        assert all(isinstance(r, TravelTimeResult) for r in results)

    def test_order_preserved(self) -> None:
        durations = [15, 30, 45]
        requests = [
            TravelTimeRequest(
                origin_lat=0.0,
                origin_lon=0.0,
                mode=TransportMode.WALKING,
                duration_minutes=d,
            )
            for d in durations
        ]
        mock = _make_mock_router()
        results = compute_batch(requests, router=mock)
        assert [r.request.duration_minutes for r in results] == durations

    def test_empty_list_returns_empty(self) -> None:
        mock = _make_mock_router()
        assert compute_batch([], router=mock) == []

    def test_single_router_instance_used(self) -> None:
        """All requests in a batch go through the same router object."""
        mock = _make_mock_router()
        compute_batch([WALKING_REQUEST, WALKING_REQUEST], router=mock)
        assert mock.compute_isochrone.call_count == 2
