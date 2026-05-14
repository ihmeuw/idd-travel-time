"""Place-name and address geocoding via OpenStreetMap Nominatim.

Converts human-readable location strings to WGS-84 (lat, lon) coordinates.
Uses the public Nominatim API — no key required, rate-limited to 1 req/s.

Geocoding cache
---------------
Results are cached in memory by default (process lifetime). Set
``IDD_GEOCODE_CACHE_DIR`` to persist them to disk across sessions, which
is essential for batch runs over many named locations.

    from idd_travel_time.locations import geocode, geocode_many

    lat, lon = geocode("Kenyatta National Hospital, Nairobi")
    coords   = geocode_many(["Lagos Island", "Abuja", "Kampala"])

Rate limiting
-------------
Nominatim's usage policy requires a maximum of 1 request per second and a
meaningful User-Agent header. Both are enforced here. For large batches
(>100 unique locations) consider hosting a local Nominatim instance.

Public API
----------
geocode(place, *, country_hint=None, language="en") -> tuple[float, float]
geocode_many(places, *, country_hint=None, language="en") -> list[tuple[float, float] | None]
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from threading import Lock
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "idd-travel-time/1.0 (https://github.com/ihmeuw/idd-travel-time)"
_MIN_REQUEST_INTERVAL = 1.05  # seconds — Nominatim policy: max 1 req/s
_REQUEST_TIMEOUT = 10  # seconds

# ---------------------------------------------------------------------------
# Rate-limit state (module-level, shared across all callers in the process)
# ---------------------------------------------------------------------------

_last_request_time: float = 0.0
_rate_lock = Lock()


# ---------------------------------------------------------------------------
# Geocoding cache
# ---------------------------------------------------------------------------


class GeocodeCache:
    """Two-level cache: in-memory dict + optional on-disk JSON file.

    The disk file is a flat JSON object mapping normalised query strings
    to ``[lat, lon]`` pairs. Simple and grep-able.
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        self._mem: dict[str, tuple[float, float]] = {}
        self._path = cache_path
        if cache_path is not None:
            self._load_disk()

    def get(self, key: str) -> tuple[float, float] | None:
        return self._mem.get(key)

    def set(self, key: str, value: tuple[float, float]) -> None:
        self._mem[key] = value
        if self._path is not None:
            self._flush_disk()

    def _load_disk(self) -> None:
        assert self._path is not None  # noqa: S101  # narrowed by caller
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._mem = {k: tuple(v) for k, v in raw.items()}
            except Exception:  # noqa: BLE001
                warnings.warn(
                    f"Geocode cache at {self._path} is unreadable; starting fresh.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _flush_disk(self) -> None:
        assert self._path is not None  # noqa: S101  # narrowed by caller
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: list(v) for k, v in self._mem.items()}, indent=2))
        tmp.replace(self._path)


# Module-level cache instance — replaced by _get_cache() on first call.
_cache: GeocodeCache | None = None
_cache_lock = Lock()


def _get_cache() -> GeocodeCache:
    """Return (or lazily create) the module-level cache."""
    global _cache  # noqa: PLW0603
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is None:
            cache_dir = os.environ.get("IDD_GEOCODE_CACHE_DIR")
            path = Path(cache_dir) / "geocode_cache.json" if cache_dir else None
            _cache = GeocodeCache(path)
    return _cache


def _cache_key(place: str, country_hint: str | None, language: str) -> str:
    parts = [place.strip().lower()]
    if country_hint:
        parts.append(country_hint.lower())
    parts.append(language)
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Rate-limited HTTP
# ---------------------------------------------------------------------------


def _rate_limited_get(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Make a Nominatim request, honouring the 1 req/s rate limit."""
    global _last_request_time  # noqa: PLW0603
    with _rate_lock:
        elapsed = time.monotonic() - _last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        resp = requests.get(
            url,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
        )
        _last_request_time = time.monotonic()

    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def geocode(
    place: str,
    *,
    country_hint: str | None = None,
    language: str = "en",
) -> tuple[float, float]:
    """Resolve a place name or address to (lat, lon).

    Parameters
    ----------
    place:
        Free-text place name or address, e.g. ``"Kenyatta National Hospital,
        Nairobi"`` or ``"Lagos Island, Nigeria"``.
    country_hint:
        ISO 3166-1 alpha-2 country code to bias results, e.g. ``"NG"``.
        Does not restrict — results outside the country may still appear.
    language:
        Preferred language for result names (passed as ``accept-language``).

    Returns
    -------
    tuple[float, float]
        ``(lat, lon)`` of the best match.

    Raises
    ------
    GeocodingError
        If Nominatim returns no results or the request fails.
    """
    cache = _get_cache()
    key = _cache_key(place, country_hint, language)

    cached = cache.get(key)
    if cached is not None:
        return cached

    params: dict[str, Any] = {
        "q": place,
        "format": "json",
        "limit": 1,
        "accept-language": language,
    }
    if country_hint:
        params["countrycodes"] = country_hint.lower()

    try:
        results = _rate_limited_get(_NOMINATIM_URL, params)
    except requests.RequestException as exc:
        msg = f"Nominatim request failed for {place!r}: {exc}"
        raise GeocodingError(msg) from exc

    if not results:
        hint = f" (country_hint={country_hint!r})" if country_hint else ""
        msg = (
            f"No geocoding results for {place!r}{hint}. "
            "Try a more specific query or add a country_hint."
        )
        raise GeocodingError(msg)

    best = results[0]
    coords: tuple[float, float] = (float(best["lat"]), float(best["lon"]))
    cache.set(key, coords)
    return coords


def geocode_many(
    places: list[str],
    *,
    country_hint: str | None = None,
    language: str = "en",
    on_error: str = "warn",
) -> list[tuple[float, float] | None]:
    """Geocode a list of place names.

    Parameters
    ----------
    places:
        List of place names or addresses.
    country_hint:
        ISO 3166-1 alpha-2 code to bias all queries.
    language:
        Preferred language for results.
    on_error:
        ``"warn"`` — insert ``None`` for failed lookups (default).
        ``"raise"`` — re-raise the first ``GeocodingError``.

    Returns
    -------
    list[tuple[float, float] | None]
        Coordinates in the same order as *places*. Failed lookups produce
        ``None`` when *on_error="warn"*.
    """
    if on_error not in ("warn", "raise"):
        msg = f"on_error must be 'warn' or 'raise', got {on_error!r}"
        raise ValueError(msg)

    results: list[tuple[float, float] | None] = []
    for place in places:
        try:
            coords = geocode(place, country_hint=country_hint, language=language)
            results.append(coords)
        except GeocodingError as exc:
            if on_error == "raise":
                raise
            warnings.warn(str(exc), UserWarning, stacklevel=2)
            results.append(None)
    return results


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GeocodingError(RuntimeError):
    """Raised when a place name cannot be resolved to coordinates."""
