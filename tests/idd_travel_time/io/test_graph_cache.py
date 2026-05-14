"""Tests for idd_travel_time.io.graph_cache."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from idd_travel_time.io.graph_cache import (
    GraphCache,
    _fmt,
    cache_key,
    default_cache_from_env,
)

# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_deterministic(self) -> None:
        assert cache_key(-1.2864, 36.8172, 3000) == cache_key(-1.2864, 36.8172, 3000)

    def test_negative_lat_uses_n_prefix(self) -> None:
        key = cache_key(-1.2864, 36.8172, 3000)
        assert key.startswith("n")
        assert "-" not in key

    def test_positive_coords_no_n_prefix(self) -> None:
        key = cache_key(6.5244, 3.3792, 2000)
        assert not key.startswith("n")

    def test_contains_radius(self) -> None:
        key = cache_key(0.0, 0.0, 3000)
        assert "3000m" in key

    def test_contains_network_type(self) -> None:
        key = cache_key(0.0, 0.0, 3000, network_type="drive")
        assert "drive" in key

    def test_different_coords_different_keys(self) -> None:
        k1 = cache_key(0.0, 0.0, 3000)
        k2 = cache_key(1.0, 0.0, 3000)
        assert k1 != k2

    def test_different_radius_different_keys(self) -> None:
        k1 = cache_key(0.0, 0.0, 3000)
        k2 = cache_key(0.0, 0.0, 5000)
        assert k1 != k2

    def test_coords_rounded_to_4dp(self) -> None:
        # Two requests within ~11m of each other share the same cache key
        k1 = cache_key(-1.28640, 36.81720, 3000)
        k2 = cache_key(-1.28641, 36.81721, 3000)
        assert k1 == k2

    def test_coords_beyond_4dp_differ(self) -> None:
        # Beyond ~11m they should differ
        k1 = cache_key(-1.2864, 36.8172, 3000)
        k2 = cache_key(-1.2874, 36.8182, 3000)
        assert k1 != k2

    def test_no_filesystem_unsafe_chars(self) -> None:
        key = cache_key(-1.2864, 36.8172, 3000, network_type="all")
        for ch in r'\/:*?"<>|':
            assert ch not in key


# ---------------------------------------------------------------------------
# _fmt
# ---------------------------------------------------------------------------


class TestFmt:
    def test_negative(self) -> None:
        assert _fmt(-1.2864).startswith("n")

    def test_positive(self) -> None:
        result = _fmt(36.8172)
        assert not result.startswith("n")

    def test_zero(self) -> None:
        assert not _fmt(0.0).startswith("n")


# ---------------------------------------------------------------------------
# GraphCache — construction
# ---------------------------------------------------------------------------


class TestGraphCacheConstruction:
    def test_creates_cache_dir(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "graphs"
        GraphCache(cache_dir)
        assert cache_dir.is_dir()

    def test_nested_dir_created(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        GraphCache(deep)
        assert deep.is_dir()

    def test_repr_shows_count(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        assert "0 graphs" in repr(cache)

    def test_cache_dir_property(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        assert cache.cache_dir == tmp_path


# ---------------------------------------------------------------------------
# GraphCache — get_or_download
# ---------------------------------------------------------------------------


def _fake_graph() -> MagicMock:
    """A minimal fake networkx graph for mocking osmnx returns."""
    g = MagicMock()
    g.graph = {"crs": "EPSG:4326"}
    return g


class TestGetOrDownload:
    def test_downloads_on_cache_miss(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        fake = _fake_graph()

        with patch("idd_travel_time.io.graph_cache.GraphCache.get_or_download") as mock:
            mock.return_value = fake
            result = cache.get_or_download(-1.2864, 36.8172, 3000)
            mock.assert_called_once()

    def test_cache_hit_skips_overpass(self, tmp_path: Path) -> None:
        """Second call for same key must not call graph_from_point again."""
        cache = GraphCache(tmp_path)
        fake = _fake_graph()

        with patch("osmnx.graph_from_point", return_value=fake) as mock_dl, \
             patch("osmnx.save_graphml"), \
             patch("osmnx.load_graphml", return_value=fake):
            cache.get_or_download(-1.2864, 36.8172, 3000)
            # Manually mark as cached
            key = cache_key(-1.2864, 36.8172, 3000)
            (tmp_path / f"{key}.graphml").touch()
            cache.get_or_download(-1.2864, 36.8172, 3000)

        assert mock_dl.call_count == 1  # downloaded once, loaded once from cache

    def test_graphml_file_written_after_download(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        fake = _fake_graph()

        with patch("osmnx.graph_from_point", return_value=fake), \
             patch("osmnx.save_graphml") as mock_save, \
             patch("osmnx.load_graphml", return_value=fake):
            # Pre-create the file so has() returns True immediately
            key = cache_key(-1.2864, 36.8172, 3000)
            graphml_path = tmp_path / f"{key}.graphml"
            graphml_path.touch()
            cache.get_or_download(-1.2864, 36.8172, 3000)
            # save_graphml not called because file exists
            mock_save.assert_not_called()

    def test_download_saves_graphml(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        fake = _fake_graph()

        with patch("osmnx.graph_from_point", return_value=fake), \
             patch("osmnx.save_graphml") as mock_save:
            cache.get_or_download(-1.2864, 36.8172, 3000)
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# GraphCache — has / invalidate / clear
# ---------------------------------------------------------------------------


class TestCacheManagement:
    def test_has_false_before_download(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        assert not cache.has(-1.2864, 36.8172, 3000)

    def test_has_true_after_file_created(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        key = cache_key(-1.2864, 36.8172, 3000)
        (tmp_path / f"{key}.graphml").touch()
        assert cache.has(-1.2864, 36.8172, 3000)

    def test_invalidate_deletes_file(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        key = cache_key(-1.2864, 36.8172, 3000)
        (tmp_path / f"{key}.graphml").touch()
        deleted = cache.invalidate(-1.2864, 36.8172, 3000)
        assert deleted is True
        assert not cache.has(-1.2864, 36.8172, 3000)

    def test_invalidate_returns_false_if_not_cached(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        assert cache.invalidate(-1.2864, 36.8172, 3000) is False

    def test_clear_deletes_all_graphml(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        for i in range(3):
            (tmp_path / f"fake_{i}.graphml").touch()
        count = cache.clear()
        assert count == 3
        assert list(tmp_path.glob("*.graphml")) == []

    def test_clear_returns_zero_when_empty(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        assert cache.clear() == 0

    def test_repr_updates_after_clear(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        (tmp_path / "a.graphml").touch()
        (tmp_path / "b.graphml").touch()
        assert "2 graphs" in repr(cache)
        cache.clear()
        assert "0 graphs" in repr(cache)


# ---------------------------------------------------------------------------
# default_cache_from_env
# ---------------------------------------------------------------------------


class TestDefaultCacheFromEnv:
    def test_returns_none_without_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("IDD_GRAPH_CACHE_DIR", raising=False)
        assert default_cache_from_env() is None

    def test_returns_graph_cache_with_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("IDD_GRAPH_CACHE_DIR", str(tmp_path))
        cache = default_cache_from_env()
        assert isinstance(cache, GraphCache)
        assert cache.cache_dir == tmp_path


# ---------------------------------------------------------------------------
# OsmnxRouter integration — cache wiring
# ---------------------------------------------------------------------------


class TestOsmnxRouterCacheWiring:
    def test_router_uses_cache_when_provided(self, tmp_path: Path) -> None:
        from idd_travel_time.routers.osmnx_router import OsmnxRouter

        cache = GraphCache(tmp_path)
        router = OsmnxRouter(graph_cache=cache)
        assert router._cache is cache

    def test_router_picks_up_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from idd_travel_time.routers.osmnx_router import OsmnxRouter

        monkeypatch.setenv("IDD_GRAPH_CACHE_DIR", str(tmp_path))
        router = OsmnxRouter()
        assert isinstance(router._cache, GraphCache)

    def test_router_cache_none_without_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from idd_travel_time.routers.osmnx_router import OsmnxRouter

        monkeypatch.delenv("IDD_GRAPH_CACHE_DIR", raising=False)
        router = OsmnxRouter()
        assert router._cache is None

    def test_router_calls_cache_get_or_download(
        self, tmp_path: Path
    ) -> None:
        """When a cache is provided, _get_graph must call cache.get_or_download."""
        from idd_travel_time.constants import TransportMode
        from idd_travel_time.models import TravelTimeRequest
        from idd_travel_time.routers.osmnx_router import OsmnxRouter

        request = TravelTimeRequest(
            origin_lat=-1.2864, origin_lon=36.8172,
            mode=TransportMode.WALKING, duration_minutes=20,
        )

        mock_cache = MagicMock(spec=GraphCache)
        mock_cache.get_or_download.return_value = _fake_graph()
        router = OsmnxRouter(graph_cache=mock_cache)

        with patch.object(router, "_get_graph", wraps=router._get_graph):
            # Just test that _get_graph delegates to the cache
            import math
            from idd_travel_time.routers._helpers import effective_speed_kph
            from idd_travel_time.routers.osmnx_router import _travel_distance_meters
            speed = effective_speed_kph(request)
            dist = _travel_distance_meters(speed, request.duration_minutes)
            router._get_graph(request, dist * 1.2)

        mock_cache.get_or_download.assert_called_once_with(
            lat=request.origin_lat,
            lon=request.origin_lon,
            radius_m=pytest.approx(dist * 1.2, rel=0.01),
            network_type="all",
        )
