"""Tests for idd_travel_time.locations.geocode."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from idd_travel_time.locations.geocode import (
    GeocodeCache,
    GeocodingError,
    _cache_key,
    geocode,
    geocode_many,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_nominatim(lat: float = -1.3006, lon: float = 36.8073) -> MagicMock:
    """Return a mock requests.get that simulates a successful Nominatim hit."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = [
        {"lat": str(lat), "lon": str(lon), "display_name": "Test"}
    ]
    return resp


def _mock_nominatim_empty() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = []
    return resp


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_lowercases_place(self) -> None:
        assert _cache_key("Nairobi", None, "en") == _cache_key("nairobi", None, "en")

    def test_includes_country_hint(self) -> None:
        k1 = _cache_key("Lagos", "NG", "en")
        k2 = _cache_key("Lagos", None, "en")
        assert k1 != k2

    def test_includes_language(self) -> None:
        k1 = _cache_key("Nairobi", None, "en")
        k2 = _cache_key("Nairobi", None, "fr")
        assert k1 != k2

    def test_strips_whitespace(self) -> None:
        assert _cache_key("  Nairobi  ", None, "en") == _cache_key(
            "nairobi", None, "en"
        )


# ---------------------------------------------------------------------------
# GeocodeCache
# ---------------------------------------------------------------------------


class TestGeocodeCache:
    def test_miss_returns_none(self) -> None:
        cache = GeocodeCache()
        assert cache.get("unknown") is None

    def test_set_then_get(self) -> None:
        cache = GeocodeCache()
        cache.set("key", (-1.3, 36.8))
        assert cache.get("key") == (-1.3, 36.8)

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        cache = GeocodeCache(path)
        cache.set("key", (1.0, 2.0))
        assert path.exists()

    def test_loads_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"key": [1.0, 2.0]}))
        cache = GeocodeCache(path)
        assert cache.get("key") == (1.0, 2.0)

    def test_corrupted_disk_starts_fresh(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text("{bad json{{")
        cache = GeocodeCache(path)
        assert cache.get("anything") is None

    def test_disk_write_is_atomic(self, tmp_path: Path) -> None:
        """Writes go through a .tmp file then replace — no partial reads."""
        path = tmp_path / "cache.json"
        cache = GeocodeCache(path)
        cache.set("k1", (1.0, 2.0))
        cache.set("k2", (3.0, 4.0))
        # No .tmp file left over
        assert not (tmp_path / "cache.tmp").exists()
        loaded = json.loads(path.read_text())
        assert "k1" in loaded
        assert "k2" in loaded


# ---------------------------------------------------------------------------
# geocode
# ---------------------------------------------------------------------------


class TestGeocode:
    def test_returns_lat_lon_tuple(self) -> None:
        with patch("requests.get", return_value=_mock_nominatim(-1.3006, 36.8073)):
            result = geocode("Kenyatta National Hospital")
        assert result == pytest.approx((-1.3006, 36.8073))

    def test_returns_floats(self) -> None:
        with patch("requests.get", return_value=_mock_nominatim()):
            lat, lon = geocode("anywhere")
        assert isinstance(lat, float)
        assert isinstance(lon, float)

    def test_empty_results_raises_geocoding_error(self) -> None:
        with patch("requests.get", return_value=_mock_nominatim_empty()):
            with pytest.raises(GeocodingError, match="No geocoding results"):
                geocode("xyzzy this place does not exist")

    def test_network_error_raises_geocoding_error(self) -> None:
        import requests as req

        from idd_travel_time.locations.geocode import GeocodeCache

        fresh = GeocodeCache()
        with (
            patch("idd_travel_time.locations.geocode._get_cache", return_value=fresh),
            patch("requests.get", side_effect=req.RequestException("timeout")),
        ):
            with pytest.raises(GeocodingError, match="Nominatim request failed"):
                geocode("anywhere_nw_error_unique")

    def test_country_hint_passed_to_api(self) -> None:
        mock_get = MagicMock(return_value=_mock_nominatim())
        with patch("requests.get", mock_get):
            geocode("Lagos", country_hint="NG")
        call_params = mock_get.call_args[1]["params"]
        assert call_params.get("countrycodes") == "ng"

    def test_no_country_hint_omits_param(self) -> None:
        mock_get = MagicMock(return_value=_mock_nominatim())
        with patch("requests.get", mock_get):
            geocode("Nairobi")
        call_params = mock_get.call_args[1]["params"]
        assert "countrycodes" not in call_params

    def test_cache_hit_skips_request(self) -> None:
        """Second call for the same place must not hit the network."""
        from idd_travel_time.locations.geocode import GeocodeCache

        fresh_cache = GeocodeCache()
        mock_get = MagicMock(return_value=_mock_nominatim())
        with (
            patch(
                "idd_travel_time.locations.geocode._get_cache", return_value=fresh_cache
            ),
            patch("requests.get", mock_get),
        ):
            geocode("CacheTestPlace_unique_9999x")
            geocode("CacheTestPlace_unique_9999x")
        assert mock_get.call_count == 1

    def test_error_message_contains_place_name(self) -> None:
        with patch("requests.get", return_value=_mock_nominatim_empty()):
            with pytest.raises(GeocodingError, match="ZZZNonexistentPlaceZZZ"):
                geocode("ZZZNonexistentPlaceZZZ")


# ---------------------------------------------------------------------------
# geocode_many
# ---------------------------------------------------------------------------


class TestGeocodeMany:
    def test_returns_list_same_length(self) -> None:
        with patch("requests.get", return_value=_mock_nominatim()):
            results = geocode_many(["Nairobi", "Lagos", "Kampala"])
        assert len(results) == 3

    def test_all_coords_on_success(self) -> None:
        with patch("requests.get", return_value=_mock_nominatim(-1.286, 36.817)):
            results = geocode_many(["A", "B"])
        assert all(r is not None for r in results)

    def test_empty_list_returns_empty(self) -> None:
        assert geocode_many([]) == []

    def test_on_error_warn_inserts_none(self) -> None:
        responses = [_mock_nominatim(), _mock_nominatim_empty(), _mock_nominatim()]
        with patch("requests.get", side_effect=responses):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                results = geocode_many(
                    ["good", "bad_place_zzz", "also_good"],
                    on_error="warn",
                )
        assert results[1] is None
        assert results[0] is not None
        assert results[2] is not None
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_on_error_raise_propagates(self) -> None:
        import requests as req

        def _side_effect(url, params=None, headers=None, timeout=None):
            if params and "bad" in params.get("q", ""):
                msg = "simulated network error"
                raise req.RequestException(msg)
            return _mock_nominatim()

        with patch("requests.get", side_effect=_side_effect):
            with pytest.raises(GeocodingError):
                geocode_many(["good", "bad_xyzzy_place"], on_error="raise")

    def test_invalid_on_error_raises(self) -> None:
        with pytest.raises(ValueError, match="on_error"):
            geocode_many([], on_error="ignore")


# ---------------------------------------------------------------------------
# TravelTimeRequest.from_place
# ---------------------------------------------------------------------------


class TestFromPlace:
    def test_resolves_lat_lon(self) -> None:
        from idd_travel_time.constants import TransportMode
        from idd_travel_time.models import TravelTimeRequest

        with patch("requests.get", return_value=_mock_nominatim(-1.3006, 36.8073)):
            req = TravelTimeRequest.from_place(
                "Kenyatta National Hospital",
                TransportMode.WALKING,
            )
        assert req.origin_lat == pytest.approx(-1.3006)
        assert req.origin_lon == pytest.approx(36.8073)

    def test_mode_set_correctly(self) -> None:
        from idd_travel_time.constants import TransportMode
        from idd_travel_time.models import TravelTimeRequest

        with patch("requests.get", return_value=_mock_nominatim()):
            req = TravelTimeRequest.from_place("anywhere", TransportMode.CYCLING)
        assert req.mode is TransportMode.CYCLING

    def test_default_duration(self) -> None:
        from idd_travel_time.constants import TransportMode
        from idd_travel_time.models import TravelTimeRequest

        with patch("requests.get", return_value=_mock_nominatim()):
            req = TravelTimeRequest.from_place("anywhere", TransportMode.WALKING)
        assert req.duration_minutes == 30

    def test_custom_duration(self) -> None:
        from idd_travel_time.constants import TransportMode
        from idd_travel_time.models import TravelTimeRequest

        with patch("requests.get", return_value=_mock_nominatim()):
            req = TravelTimeRequest.from_place(
                "anywhere", TransportMode.WALKING, duration_minutes=45
            )
        assert req.duration_minutes == 45

    def test_geocoding_error_propagates(self) -> None:
        from idd_travel_time.constants import TransportMode
        from idd_travel_time.models import TravelTimeRequest

        with patch("requests.get", return_value=_mock_nominatim_empty()):
            with pytest.raises(GeocodingError):
                TravelTimeRequest.from_place("xyzzy", TransportMode.WALKING)

    def test_country_hint_forwarded(self) -> None:
        from idd_travel_time.constants import TransportMode
        from idd_travel_time.models import TravelTimeRequest

        mock_get = MagicMock(return_value=_mock_nominatim())
        with patch("requests.get", mock_get):
            TravelTimeRequest.from_place(
                "Lagos Island", TransportMode.WALKING, country_hint="NG"
            )
        call_params = mock_get.call_args[1]["params"]
        assert call_params.get("countrycodes") == "ng"


# ---------------------------------------------------------------------------
# CLI --place flag
# ---------------------------------------------------------------------------


class TestCliPlace:
    def test_place_flag_accepted(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from idd_travel_time.run_isochrone import main

        out = tmp_path / "map.html"

        def _fake_geocode(place, *, country_hint=None, language="en"):
            return (-1.3006, 36.8073)

        def _fake_compute(request, *, router=None):
            import geopandas as gpd
            from shapely.geometry import Polygon

            from idd_travel_time.models import TravelTimeResult

            sq = Polygon([(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)])
            iso = gpd.GeoDataFrame(
                {
                    "geometry": [sq],
                    "duration_min": [30],
                    "mode": ["walking"],
                    "time_context": ["off_peak"],
                },
                crs="EPSG:4326",
            )
            return TravelTimeResult(
                request=request,
                isochrone_gdf=iso,
                network_gdf=None,
                poi_gdf=None,
                router_name="mock",
            )

        runner = CliRunner()
        with (
            patch("idd_travel_time.locations.geocode.geocode", _fake_geocode),
            patch("idd_travel_time.run_isochrone.compute", _fake_compute),
            patch("idd_travel_time.run_isochrone.save_map", return_value=out),
        ):
            result = runner.invoke(
                main,
                [
                    "--place",
                    "Kenyatta National Hospital",
                    "--mode",
                    "walking",
                    "--output",
                    str(out),
                ],
            )
        assert result.exit_code == 0, result.output

    def test_place_and_lat_mutually_exclusive(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from idd_travel_time.run_isochrone import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--place",
                "Nairobi",
                "--lat",
                "-1.2864",
                "--lon",
                "36.8172",
                "--mode",
                "walking",
                "--output",
                str(tmp_path / "map.html"),
            ],
        )
        assert result.exit_code != 0

    def test_neither_place_nor_lat_fails(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from idd_travel_time.run_isochrone import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--mode",
                "walking",
                "--output",
                str(tmp_path / "map.html"),
            ],
        )
        assert result.exit_code != 0
