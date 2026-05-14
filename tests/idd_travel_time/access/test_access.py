"""Tests for access backends, ensemble, metrics, and benchmark."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from idd_travel_time.access.surface import BoundingBox, TravelTimeSurface
from idd_travel_time.access.surface_router import SurfaceRouter
from idd_travel_time.access.ensemble import EnsembleSurfaceRouter
from idd_travel_time.access.metrics import (
    benchmark_table,
    compare_surfaces,
    residual_geodataframe,
    score_at_points,
)
from idd_travel_time.access.benchmark import (
    BenchmarkResult,
    make_synthetic_ground_truth,
    run_benchmark,
)
from idd_travel_time.constants import TransportMode

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BBOX = BoundingBox(west=36.80, south=-1.35, east=36.90, north=-1.25)
_DEST = (-1.30, 36.85)
_RES = 0.01
_MODE = TransportMode.WALKING


def _make_surface(
    value: float = 20.0,
    nrows: int = 10,
    ncols: int = 10,
    backend: str = "mock",
    nan_fraction: float = 0.0,
) -> TravelTimeSurface:
    rng = np.random.default_rng(42)
    data = np.full((nrows, ncols), value, dtype=np.float32)
    data += rng.normal(0, 2.0, data.shape).astype(np.float32)
    data = np.clip(data, 0.0, None)
    if nan_fraction > 0:
        n = int(nrows * ncols * nan_fraction)
        data.ravel()[:n] = np.nan
    return TravelTimeSurface.from_arrays(
        data=data,
        west=_BBOX.west, north=_BBOX.north,
        resolution_deg=_RES,
        backend=backend,
        destination=_DEST,
    )


def _make_gt(surface: TravelTimeSurface, n: int = 50) -> gpd.GeoDataFrame:
    return make_synthetic_ground_truth(surface, n_points=n, noise_std_min=1.0, random_seed=0)


# ---------------------------------------------------------------------------
# Mock SurfaceRouter for testing
# ---------------------------------------------------------------------------

class ConstantRouter(SurfaceRouter):
    """Returns a surface with a constant travel time + small noise."""
    def __init__(self, value: float, name: str = "constant") -> None:
        self._value = value
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute_surface(self, destination, bbox, resolution_deg, mode, *, transit_profile=None):
        return _make_surface(value=self._value, backend=self._name)


# ---------------------------------------------------------------------------
# SurfaceRouter ABC
# ---------------------------------------------------------------------------


def test_surface_router_is_abstract() -> None:
    with pytest.raises(TypeError):
        SurfaceRouter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# OsmnxSurfaceRouter (mocked — no network calls)
# ---------------------------------------------------------------------------


class TestOsmnxSurfaceRouterMocked:
    """Tests that verify the OsmnxSurfaceRouter logic without Overpass calls."""

    def _make_fake_graph(self):
        """Minimal fake graph with 4 nodes in a small area."""
        import networkx as nx
        from shapely.geometry import Point

        G = nx.MultiDiGraph()
        G.graph["crs"] = "EPSG:4326"
        coords = [
            (36.83, -1.28), (36.87, -1.28),
            (36.83, -1.32), (36.87, -1.32),
        ]
        for i, (x, y) in enumerate(coords):
            G.add_node(i, x=x, y=y)
        G.add_edge(0, 1, length=400.0)
        G.add_edge(1, 2, length=400.0)
        G.add_edge(2, 3, length=400.0)
        G.add_edge(3, 0, length=400.0)
        G.add_edge(0, 2, length=560.0)
        G.add_edge(1, 3, length=560.0)
        return G

    def test_returns_travel_time_surface(self) -> None:
        from idd_travel_time.access.backends.osmnx_surface import OsmnxSurfaceRouter
        import networkx as nx

        fake_graph = self._make_fake_graph()
        fake_proj = fake_graph.copy()
        fake_proj.graph["crs"] = "EPSG:32637"
        # Use same coords in fake projected graph
        for n in fake_proj.nodes:
            fake_proj.nodes[n]["x"] = fake_graph.nodes[n]["x"] * 111000
            fake_proj.nodes[n]["y"] = fake_graph.nodes[n]["y"] * 111000

        router = OsmnxSurfaceRouter()

        with patch("idd_travel_time.access.backends.osmnx_surface.ox") as mock_ox:
            mock_ox.graph_from_bbox.return_value = fake_graph
            mock_ox.project_graph.return_value = fake_proj
            mock_ox.nearest_nodes.return_value = 0

            surface = router.compute_surface(_DEST, _BBOX, _RES, _MODE)

        assert isinstance(surface, TravelTimeSurface)
        assert surface.backend == "osmnx"
        assert surface.shape[0] > 0
        assert surface.shape[1] > 0

    def test_destination_stored(self) -> None:
        from idd_travel_time.access.backends.osmnx_surface import OsmnxSurfaceRouter

        fake_graph = self._make_fake_graph()
        fake_proj = fake_graph.copy()
        fake_proj.graph["crs"] = "EPSG:32637"
        for n in fake_proj.nodes:
            fake_proj.nodes[n]["x"] = fake_graph.nodes[n]["x"] * 111000
            fake_proj.nodes[n]["y"] = fake_graph.nodes[n]["y"] * 111000

        router = OsmnxSurfaceRouter()
        with patch("idd_travel_time.access.backends.osmnx_surface.ox") as mock_ox:
            mock_ox.graph_from_bbox.return_value = fake_graph
            mock_ox.project_graph.return_value = fake_proj
            mock_ox.nearest_nodes.return_value = 0
            surface = router.compute_surface(_DEST, _BBOX, _RES, _MODE)

        assert surface.destination == pytest.approx(_DEST)

    def test_uses_graph_cache_when_provided(self) -> None:
        from idd_travel_time.access.backends.osmnx_surface import OsmnxSurfaceRouter
        from idd_travel_time.io import GraphCache

        mock_cache = MagicMock(spec=GraphCache)
        fake_graph = self._make_fake_graph()
        fake_proj = fake_graph.copy()
        fake_proj.graph["crs"] = "EPSG:32637"
        for n in fake_proj.nodes:
            fake_proj.nodes[n]["x"] = fake_graph.nodes[n]["x"] * 111000
            fake_proj.nodes[n]["y"] = fake_graph.nodes[n]["y"] * 111000
        mock_cache.get_or_download.return_value = fake_graph

        router = OsmnxSurfaceRouter(graph_cache=mock_cache)
        with patch("idd_travel_time.access.backends.osmnx_surface.ox") as mock_ox:
            mock_ox.project_graph.return_value = fake_proj
            mock_ox.nearest_nodes.return_value = 0
            router.compute_surface(_DEST, _BBOX, _RES, _MODE)

        mock_cache.get_or_download.assert_called_once()


# ---------------------------------------------------------------------------
# FrictionSurfaceRouter
# ---------------------------------------------------------------------------


class TestFrictionSurfaceRouter:
    def test_requires_friction_or_speed(self) -> None:
        from idd_travel_time.access.backends.friction_surface import FrictionSurfaceRouter
        with pytest.raises(ValueError, match="friction_path"):
            FrictionSurfaceRouter()

    def test_uniform_speed_override(self) -> None:
        from idd_travel_time.access.backends.friction_surface import FrictionSurfaceRouter
        router = FrictionSurfaceRouter(speed_kph_override=5.0)
        assert router.name == "friction"
        surface = router.compute_surface(_DEST, _BBOX, _RES, _MODE)
        assert isinstance(surface, TravelTimeSurface)
        assert surface.backend == "friction"

    def test_uniform_surface_non_negative(self) -> None:
        from idd_travel_time.access.backends.friction_surface import FrictionSurfaceRouter
        router = FrictionSurfaceRouter(speed_kph_override=5.0)
        surface = router.compute_surface(_DEST, _BBOX, _RES, _MODE)
        valid = surface.data[~np.isnan(surface.data)]
        assert (valid >= 0).all()

    def test_destination_has_zero_travel_time(self) -> None:
        from idd_travel_time.access.backends.friction_surface import FrictionSurfaceRouter
        router = FrictionSurfaceRouter(speed_kph_override=5.0)
        surface = router.compute_surface(_DEST, _BBOX, _RES, _MODE)
        # Pixel at destination should have near-zero travel time
        dest_lat, dest_lon = _DEST
        col = int((dest_lon - surface.bbox.west) / surface.resolution_deg)
        row = int((surface.bbox.north - dest_lat) / surface.resolution_deg)
        if 0 <= row < surface.nrows and 0 <= col < surface.ncols:
            val = surface.data[row, col]
            if not np.isnan(val):
                assert val == pytest.approx(0.0, abs=1.0)

    def test_travel_time_increases_with_distance(self) -> None:
        from idd_travel_time.access.backends.friction_surface import FrictionSurfaceRouter
        router = FrictionSurfaceRouter(speed_kph_override=5.0)
        surface = router.compute_surface(_DEST, _BBOX, _RES, _MODE)
        # Mean travel time over pixels far from destination > near destination
        dest_lat, dest_lon = _DEST
        lats, lons = surface.pixel_coords()
        dist = np.sqrt((lats - dest_lat) ** 2 + (lons - dest_lon) ** 2)
        valid = ~np.isnan(surface.data)
        near = valid & (dist < 0.02)
        far = valid & (dist > 0.04)
        if near.any() and far.any():
            assert surface.data[near].mean() < surface.data[far].mean()

    def test_missing_geotiff_raises(self) -> None:
        from idd_travel_time.access.backends.friction_surface import FrictionSurfaceRouter
        router = FrictionSurfaceRouter(friction_path="/nonexistent/file.tif")
        with pytest.raises(FileNotFoundError):
            router.compute_surface(_DEST, _BBOX, _RES, _MODE)


# ---------------------------------------------------------------------------
# EnsembleSurfaceRouter
# ---------------------------------------------------------------------------


class TestEnsembleSurfaceRouter:
    def test_requires_at_least_two_routers(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            EnsembleSurfaceRouter([ConstantRouter(20.0)])

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            EnsembleSurfaceRouter(
                [ConstantRouter(20.0), ConstantRouter(30.0)],
                strategy="unknown",
            )

    def test_invalid_nan_policy(self) -> None:
        with pytest.raises(ValueError, match="nan_policy"):
            EnsembleSurfaceRouter(
                [ConstantRouter(20.0), ConstantRouter(30.0)],
                nan_policy="drop",
            )

    def test_weights_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="weights length"):
            EnsembleSurfaceRouter(
                [ConstantRouter(20.0), ConstantRouter(30.0)],
                weights=[0.5, 0.3, 0.2],
            )

    def test_negative_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="weights must be >= 0"):
            EnsembleSurfaceRouter(
                [ConstantRouter(20.0), ConstantRouter(30.0)],
                weights=[-0.5, 1.5],
            )

    def test_mean_ensemble(self) -> None:
        r1, r2 = ConstantRouter(20.0, "r1"), ConstantRouter(40.0, "r2")
        ens = EnsembleSurfaceRouter([r1, r2], strategy="mean")
        s = ens.compute_surface(_DEST, _BBOX, _RES, _MODE)
        valid = s.data[~np.isnan(s.data)]
        assert valid.mean() == pytest.approx(30.0, abs=3.0)

    def test_weighted_mean(self) -> None:
        r1, r2 = ConstantRouter(0.0, "r1"), ConstantRouter(40.0, "r2")
        # Weight: r1=0.25, r2=0.75 → expected ≈ 30
        ens = EnsembleSurfaceRouter([r1, r2], strategy="mean", weights=[0.25, 0.75])
        s = ens.compute_surface(_DEST, _BBOX, _RES, _MODE)
        valid = s.data[~np.isnan(s.data)]
        assert valid.mean() == pytest.approx(30.0, abs=4.0)

    def test_min_ensemble(self) -> None:
        r1, r2 = ConstantRouter(20.0, "r1"), ConstantRouter(40.0, "r2")
        ens = EnsembleSurfaceRouter([r1, r2], strategy="min")
        s = ens.compute_surface(_DEST, _BBOX, _RES, _MODE)
        valid = s.data[~np.isnan(s.data)]
        assert valid.mean() < 25.0  # closer to 20

    def test_max_ensemble(self) -> None:
        r1, r2 = ConstantRouter(20.0, "r1"), ConstantRouter(40.0, "r2")
        ens = EnsembleSurfaceRouter([r1, r2], strategy="max")
        s = ens.compute_surface(_DEST, _BBOX, _RES, _MODE)
        valid = s.data[~np.isnan(s.data)]
        assert valid.mean() > 35.0  # closer to 40

    def test_median_ensemble(self) -> None:
        r1 = ConstantRouter(10.0, "r1")
        r2 = ConstantRouter(30.0, "r2")
        r3 = ConstantRouter(50.0, "r3")
        ens = EnsembleSurfaceRouter([r1, r2, r3], strategy="median")
        s = ens.compute_surface(_DEST, _BBOX, _RES, _MODE)
        valid = s.data[~np.isnan(s.data)]
        assert valid.mean() == pytest.approx(30.0, abs=5.0)

    def test_compute_all_surfaces_returns_dict(self) -> None:
        r1, r2 = ConstantRouter(20.0, "r1"), ConstantRouter(40.0, "r2")
        ens = EnsembleSurfaceRouter([r1, r2], strategy="mean")
        all_s = ens.compute_all_surfaces(_DEST, _BBOX, _RES, _MODE)
        assert set(all_s.keys()) == {"r1", "r2"}

    def test_compute_uncertainty(self) -> None:
        r1, r2 = ConstantRouter(10.0, "r1"), ConstantRouter(50.0, "r2")
        ens = EnsembleSurfaceRouter([r1, r2])
        unc = ens.compute_uncertainty(_DEST, _BBOX, _RES, _MODE)
        valid = unc.data[~np.isnan(unc.data)]
        assert valid.mean() > 0.0  # should have some spread

    def test_name_reflects_strategy_and_backends(self) -> None:
        r1, r2 = ConstantRouter(20.0, "alpha"), ConstantRouter(30.0, "beta")
        ens = EnsembleSurfaceRouter([r1, r2], strategy="median")
        assert "median" in ens.name
        assert "alpha" in ens.name
        assert "beta" in ens.name

    def test_nan_policy_propagate(self) -> None:
        """With propagate, any NaN in any backend → NaN in output."""
        class NaNRouter(SurfaceRouter):
            @property
            def name(self): return "nan_router"
            def compute_surface(self, dest, bbox, res, mode, *, transit_profile=None):
                s = _make_surface(20.0, nan_fraction=0.5, backend="nan_router")
                return s

        r1 = ConstantRouter(20.0, "r1")
        r2 = NaNRouter()
        ens = EnsembleSurfaceRouter([r1, r2], strategy="mean", nan_policy="propagate")
        s = ens.compute_surface(_DEST, _BBOX, _RES, _MODE)
        assert np.isnan(s.data).any()

    def test_nan_policy_ignore(self) -> None:
        """With ignore, NaN backends are skipped at each pixel."""
        class NaNRouter(SurfaceRouter):
            @property
            def name(self): return "nan_router"
            def compute_surface(self, dest, bbox, res, mode, *, transit_profile=None):
                s = _make_surface(20.0, nan_fraction=1.0, backend="nan_router")
                return s

        r1 = ConstantRouter(20.0, "r1")
        r2 = NaNRouter()
        ens = EnsembleSurfaceRouter([r1, r2], strategy="mean", nan_policy="ignore")
        s = ens.compute_surface(_DEST, _BBOX, _RES, _MODE)
        # r1 has no NaNs so output should be mostly valid
        assert not np.all(np.isnan(s.data))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestScoreAtPoints:
    def test_perfect_predictions_zero_mae(self) -> None:
        s = _make_surface(20.0, nan_fraction=0.0)
        gt = _make_gt(s, n=30)
        gt["travel_time_min"] = _sample(s, gt)  # perfect predictions
        scores = score_at_points(s, gt)
        assert scores["mae"] == pytest.approx(0.0, abs=0.1)

    def test_keys_present(self) -> None:
        s = _make_surface(20.0, nan_fraction=0.0)
        gt = _make_gt(s, n=30)
        scores = score_at_points(s, gt)
        for k in ("n", "mae", "rmse", "bias", "r2", "pearson_r",
                  "coverage_error_30min", "coverage_error_60min"):
            assert k in scores

    def test_positive_bias_when_over_predicted(self) -> None:
        s = _make_surface(50.0, nan_fraction=0.0)
        gt = _make_gt(s, n=30)
        gt["travel_time_min"] = 20.0  # obs is 20, pred is ~50
        scores = score_at_points(s, gt)
        assert scores["bias"] > 0.0

    def test_empty_gt_returns_nan(self) -> None:
        s = _make_surface()
        gt = _make_gt(s, n=5)
        gt = gt.iloc[0:0]  # empty
        scores = score_at_points(s, gt)
        assert scores["n"] == 0
        assert np.isnan(scores["mae"])

    def test_split_filtering(self) -> None:
        s = _make_surface(nan_fraction=0.0)
        gt = _make_gt(s, n=50)
        gt["split"] = ["train"] * 30 + ["test"] * 20
        train_scores = score_at_points(s, gt, split="train")
        test_scores = score_at_points(s, gt, split="test")
        assert train_scores["n"] == 30
        assert test_scores["n"] == 20


class TestCompareSurfaces:
    def test_identical_surfaces_zero_diff(self) -> None:
        s = _make_surface(20.0)
        result = compare_surfaces(s, s)
        assert result["mean_diff"] == pytest.approx(0.0, abs=0.01)
        assert result["rmsd"] == pytest.approx(0.0, abs=0.01)

    def test_keys_present(self) -> None:
        a, b = _make_surface(20.0, backend="a"), _make_surface(30.0, backend="b")
        result = compare_surfaces(a, b)
        for k in ("n_pixels", "mean_diff", "rmsd", "pearson_r",
                  "agreement_30min", "agreement_60min"):
            assert k in result

    def test_mean_diff_sign(self) -> None:
        a = _make_surface(30.0, nan_fraction=0.0)
        b = _make_surface(20.0, nan_fraction=0.0)
        result = compare_surfaces(a, b)
        assert result["mean_diff"] > 0  # a - b > 0


class TestBenchmarkTable:
    def test_returns_dataframe(self) -> None:
        import pandas as pd
        s = _make_surface(nan_fraction=0.0)
        gt = _make_gt(s)
        table = benchmark_table({"s": s}, gt)
        assert isinstance(table, pd.DataFrame)

    def test_rows_per_backend_and_split(self) -> None:
        s = _make_surface(nan_fraction=0.0)
        gt = _make_gt(s, n=40)
        gt["split"] = ["train"] * 20 + ["test"] * 20
        table = benchmark_table({"s": s}, gt, splits=["train", "test"])
        assert len(table) == 2  # 1 backend × 2 splits


class TestResidualGeoDataFrame:
    def test_returns_geodataframe(self) -> None:
        s = _make_surface(nan_fraction=0.0)
        gt = _make_gt(s)
        result = residual_geodataframe(s, gt)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_columns_present(self) -> None:
        s = _make_surface(nan_fraction=0.0)
        gt = _make_gt(s)
        result = residual_geodataframe(s, gt)
        for col in ("predicted_min", "residual_min", "abs_residual_min", "backend"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_basic_run_no_gt(self) -> None:
        r = ConstantRouter(25.0, "c")
        result = run_benchmark({"c": r}, _DEST, _BBOX, _RES, _MODE)
        assert isinstance(result, BenchmarkResult)
        assert "c" in result.surfaces
        assert "c" in result.timing

    def test_with_synthetic_gt(self) -> None:
        r = ConstantRouter(25.0, "c")
        ref = _make_surface(25.0, nan_fraction=0.0)
        gt = _make_gt(ref, n=40)
        result = run_benchmark({"c": r}, _DEST, _BBOX, _RES, _MODE, ground_truth=gt)
        assert not result.scores.empty
        assert result.ground_truth_with_preds is not None

    def test_pairwise_comparison(self) -> None:
        r1, r2 = ConstantRouter(20.0, "r1"), ConstantRouter(40.0, "r2")
        result = run_benchmark({"r1": r1, "r2": r2}, _DEST, _BBOX, _RES, _MODE)
        assert "r1_vs_r2" in result.pairwise

    def test_summary_string(self) -> None:
        r = ConstantRouter(25.0, "c")
        ref = _make_surface(25.0, nan_fraction=0.0)
        gt = _make_gt(ref, n=30)
        result = run_benchmark({"c": r}, _DEST, _BBOX, _RES, _MODE, ground_truth=gt)
        s = result.summary()
        assert "c" in s

    def test_failed_backend_inserts_warning(self) -> None:
        import warnings

        class FailRouter(SurfaceRouter):
            @property
            def name(self): return "fail"
            def compute_surface(self, *a, **kw):
                raise RuntimeError("simulated failure")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_benchmark(
                {"fail": FailRouter(), "ok": ConstantRouter(20.0)},
                _DEST, _BBOX, _RES, _MODE,
            )
        assert any("fail" in str(w.message) for w in caught)
        assert "fail" not in result.surfaces
        assert "ok" in result.surfaces


class TestMakeSyntheticGroundTruth:
    def test_returns_geodataframe(self) -> None:
        s = _make_surface(nan_fraction=0.0)
        gt = make_synthetic_ground_truth(s, n_points=20)
        assert isinstance(gt, gpd.GeoDataFrame)

    def test_correct_count(self) -> None:
        s = _make_surface(nan_fraction=0.0)
        gt = make_synthetic_ground_truth(s, n_points=20)
        assert len(gt) == 20

    def test_all_non_negative(self) -> None:
        s = _make_surface(nan_fraction=0.0)
        gt = make_synthetic_ground_truth(s, n_points=30)
        assert (gt["travel_time_min"] >= 0).all()

    def test_all_nan_surface_raises(self) -> None:
        data = np.full((5, 5), np.nan, dtype=np.float32)
        s = TravelTimeSurface.from_arrays(
            data=data, west=0.0, north=0.0,
            resolution_deg=0.01, backend="test", destination=(0.0, 0.0)
        )
        with pytest.raises(ValueError, match="no valid pixels"):
            make_synthetic_ground_truth(s)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sample(surface: TravelTimeSurface, gdf: gpd.GeoDataFrame) -> np.ndarray:
    from idd_travel_time.access.metrics import _sample_surface_at_points
    return _sample_surface_at_points(surface, gdf)
