"""Tests for compute_parallel in idd_travel_time.isochrone.compute."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from idd_travel_time.constants import TransportMode
from idd_travel_time.isochrone.compute import _worker, compute_parallel
from idd_travel_time.models import TravelTimeRequest, TravelTimeResult
from idd_travel_time.routers.base import Router

# ---------------------------------------------------------------------------
# Module-level router classes (must be at module level to be picklable)
# ---------------------------------------------------------------------------

_SQUARE = Polygon([(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)])


def _make_request(lat: float = 0.0, lon: float = 0.0, duration: int = 30) -> TravelTimeRequest:
    return TravelTimeRequest(
        origin_lat=lat, origin_lon=lon,
        mode=TransportMode.WALKING,
        duration_minutes=duration,
    )


def _make_result(request: TravelTimeRequest, router_name: str = "mock") -> TravelTimeResult:
    iso = gpd.GeoDataFrame(
        {"geometry": [_SQUARE], "duration_min": [request.duration_minutes],
         "mode": [request.mode.value], "time_context": [request.time_context.value]},
        crs="EPSG:4326",
    )
    return TravelTimeResult(
        request=request, isochrone_gdf=iso,
        network_gdf=None, poi_gdf=None, router_name=router_name,
    )


class MockRouter(Router):
    """Module-level router — always succeeds."""
    @property
    def name(self) -> str:
        return "mock"

    def compute_isochrone(self, request: TravelTimeRequest) -> TravelTimeResult:
        return _make_result(request)


class FailOnLatOneRouter(Router):
    """Module-level router — raises for requests with lat == 1.0."""
    @property
    def name(self) -> str:
        return "fail_on_lat_one"

    def compute_isochrone(self, request: TravelTimeRequest) -> TravelTimeResult:
        if round(request.origin_lat) == 1:
            raise RuntimeError(f"Injected failure for lat={request.origin_lat}")
        return _make_result(request)


class CountingRouter(Router):
    """Module-level router that records calls (thread-safe via GIL)."""
    calls: list[TravelTimeRequest] = []

    @property
    def name(self) -> str:
        return "counting"

    def compute_isochrone(self, request: TravelTimeRequest) -> TravelTimeResult:
        CountingRouter.calls.append(request)
        return _make_result(request)


def mock_factory() -> MockRouter:
    return MockRouter()


def fail_factory() -> FailOnLatOneRouter:
    return FailOnLatOneRouter()


def counting_factory() -> CountingRouter:
    return CountingRouter()


# ---------------------------------------------------------------------------
# _worker
# ---------------------------------------------------------------------------


class TestWorker:
    def test_calls_router_and_returns_result(self) -> None:
        req = _make_request()
        result = _worker(req, mock_factory)
        assert isinstance(result, TravelTimeResult)

    def test_propagates_router_exception(self) -> None:
        req = _make_request(lat=1.0)
        with pytest.raises(RuntimeError, match="Injected failure"):
            _worker(req, fail_factory)


# ---------------------------------------------------------------------------
# compute_parallel — basic correctness (use_threads=True for test isolation)
# ---------------------------------------------------------------------------


class TestComputeParallelBasic:
    def test_returns_list_same_length(self) -> None:
        reqs = [_make_request(lat=float(i)) for i in range(3)]
        results = compute_parallel(reqs, router_factory=mock_factory,
                                   max_workers=2, use_threads=True)
        assert len(results) == 3

    def test_all_results_are_travel_time_result(self) -> None:
        reqs = [_make_request(lat=float(i)) for i in range(3)]
        results = compute_parallel(reqs, router_factory=mock_factory,
                                   max_workers=2, use_threads=True)
        assert all(isinstance(r, TravelTimeResult) for r in results)

    def test_empty_requests_returns_empty(self) -> None:
        results = compute_parallel([], router_factory=mock_factory,
                                   max_workers=1, use_threads=True)
        assert results == []

    def test_single_request(self) -> None:
        reqs = [_make_request()]
        results = compute_parallel(reqs, router_factory=mock_factory,
                                   max_workers=1, use_threads=True)
        assert len(results) == 1
        assert isinstance(results[0], TravelTimeResult)

    def test_result_count_matches_request_count(self) -> None:
        reqs = [_make_request(lat=float(i)) for i in range(5)]
        results = compute_parallel(reqs, router_factory=mock_factory,
                                   max_workers=3, use_threads=True)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# compute_parallel — on_error behaviour
# ---------------------------------------------------------------------------


class TestComputeParallelErrors:
    def test_on_error_warn_inserts_none_for_failed(self) -> None:
        reqs = [_make_request(lat=float(i)) for i in range(3)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = compute_parallel(
                reqs, router_factory=fail_factory, max_workers=2,
                on_error="warn", use_threads=True,
            )
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 2
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)

    def test_on_error_raise_propagates(self) -> None:
        reqs = [_make_request(lat=float(i)) for i in range(3)]
        with pytest.raises(RuntimeError, match="Injected failure"):
            compute_parallel(
                reqs, router_factory=fail_factory, max_workers=2,
                on_error="raise", use_threads=True,
            )

    def test_invalid_on_error_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="on_error"):
            compute_parallel([], router_factory=mock_factory, on_error="ignore")


# ---------------------------------------------------------------------------
# compute_parallel — output_dir + skip_existing
# ---------------------------------------------------------------------------


class TestComputeParallelPersistence:
    def test_results_written_to_output_dir(self, tmp_path: Path) -> None:
        reqs = [_make_request(lat=float(i)) for i in range(3)]
        compute_parallel(reqs, router_factory=mock_factory, max_workers=2,
                         output_dir=tmp_path, use_threads=True)
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(subdirs) == 3

    def test_skip_existing_loads_from_disk(self, tmp_path: Path) -> None:
        reqs = [_make_request(lat=float(i)) for i in range(3)]

        # First run — compute and persist all three
        compute_parallel(reqs, router_factory=mock_factory, max_workers=2,
                         output_dir=tmp_path, use_threads=True)

        # Reset call log, second run — all should be skipped
        CountingRouter.calls.clear()
        results = compute_parallel(
            reqs, router_factory=counting_factory, max_workers=2,
            output_dir=tmp_path, skip_existing=True, use_threads=True,
        )
        assert CountingRouter.calls == [], "skip_existing=True should not recompute"
        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_skip_existing_false_always_recomputes(self, tmp_path: Path) -> None:
        reqs = [_make_request()]
        compute_parallel(reqs, router_factory=mock_factory,
                         output_dir=tmp_path, use_threads=True)
        CountingRouter.calls.clear()
        compute_parallel(
            reqs, router_factory=counting_factory,
            output_dir=tmp_path, skip_existing=False, use_threads=True,
        )
        assert len(CountingRouter.calls) == 1

    def test_no_output_dir_does_not_write(self, tmp_path: Path) -> None:
        reqs = [_make_request(lat=float(i)) for i in range(2)]
        compute_parallel(reqs, router_factory=mock_factory, use_threads=True)
        assert list(tmp_path.iterdir()) == []
