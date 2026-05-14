"""Persistent osmnx graph cache.

Caches street graphs downloaded from the Overpass API to disk as GraphML
files, keyed by (lat, lon, radius_m, network_type). A cache hit avoids a
round-trip to Overpass, which is the dominant latency in OsmnxRouter.

Cache key scheme
----------------
``{lat}_{lon}_{radius_m}m_{network_type}.graphml``

Coordinates are rounded to 4 decimal places (~11 m precision) — close enough
that two requests for "Nairobi city centre" with slightly different lat/lon
fractions share the same cached graph, while requests in genuinely different
areas do not collide.

Usage
-----
Direct use::

    from idd_travel_time.io.graph_cache import GraphCache

    cache = GraphCache("/path/to/cache_dir")
    graph = cache.get_or_download(lat, lon, radius_m=3000)

Integrated into OsmnxRouter::

    from idd_travel_time.io.graph_cache import GraphCache
    from idd_travel_time.routers import OsmnxRouter

    router = OsmnxRouter(graph_cache=GraphCache("/path/to/cache_dir"))
    result = compute(request, router=router)

Environment variable shortcut
------------------------------
Set ``IDD_GRAPH_CACHE_DIR`` to activate caching automatically in
``OsmnxRouter`` without passing a ``GraphCache`` instance explicitly.

Thread safety
-------------
Each ``get_or_download`` call locks on the specific cache key before writing,
so concurrent workers for different origins can proceed in parallel while
duplicate requests for the same origin block behind a single download.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

# Key locks: one threading.Lock per cache key, created on first access.
_key_locks: dict[str, threading.Lock] = {}
_key_locks_mutex = threading.Lock()

_COORD_PRECISION = 4   # decimal places for lat/lon rounding in cache key
_DEFAULT_NETWORK_TYPE = "all"


class GraphCache:
    """Persistent on-disk cache for osmnx street graphs.

    Parameters
    ----------
    cache_dir:
        Directory where GraphML files are stored. Created if it does not exist.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_download(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        network_type: str = _DEFAULT_NETWORK_TYPE,
    ) -> nx.MultiDiGraph:
        """Return a cached graph, downloading from Overpass if not cached.

        Parameters
        ----------
        lat, lon:
            Centre of the graph download area (WGS-84).
        radius_m:
            Radius in metres around (lat, lon) to include.
        network_type:
            osmnx network type (``"all"``, ``"drive"``, ``"walk"``, etc.).

        Returns
        -------
        networkx.MultiDiGraph
        """
        import osmnx as ox  # noqa: PLC0415

        key = cache_key(lat, lon, radius_m, network_type)
        path = self._path_for_key(key)

        lock = _get_key_lock(key)
        with lock:
            if path.exists():
                return ox.load_graphml(path)

            graph = ox.graph_from_point(
                (lat, lon),
                dist=round(radius_m),
                network_type=network_type,
            )
            ox.save_graphml(graph, path)
            return graph

    def has(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        network_type: str = _DEFAULT_NETWORK_TYPE,
    ) -> bool:
        """Return True if a cached graph exists for these parameters."""
        key = cache_key(lat, lon, radius_m, network_type)
        return self._path_for_key(key).exists()

    def invalidate(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        network_type: str = _DEFAULT_NETWORK_TYPE,
    ) -> bool:
        """Delete a cached graph. Returns True if the file existed.

        Use when OSM data for an area has changed and the cached graph is stale.
        """
        key = cache_key(lat, lon, radius_m, network_type)
        path = self._path_for_key(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Delete all cached graphs. Returns the number of files deleted."""
        count = 0
        for f in self._dir.glob("*.graphml"):
            f.unlink()
            count += 1
        return count

    @property
    def cache_dir(self) -> Path:
        return self._dir

    def __repr__(self) -> str:
        n = len(list(self._dir.glob("*.graphml")))
        return f"GraphCache({self._dir}, {n} graphs cached)"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _path_for_key(self, key: str) -> Path:
        return self._dir / f"{key}.graphml"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def cache_key(
    lat: float,
    lon: float,
    radius_m: float,
    network_type: str = _DEFAULT_NETWORK_TYPE,
) -> str:
    """Build a deterministic, filesystem-safe cache key.

    Coordinates are rounded to ``_COORD_PRECISION`` decimal places.
    Negative values use 'n' prefix to avoid filesystem issues.
    """
    lat_r = round(lat, _COORD_PRECISION)
    lon_r = round(lon, _COORD_PRECISION)
    lat_s = _fmt(lat_r)
    lon_s = _fmt(lon_r)
    radius_s = f"{round(radius_m)}m"
    raw = f"{lat_s}_{lon_s}_{radius_s}_{network_type}"

    # Hash long network_type strings to keep filenames short.
    if len(raw) > 120:
        suffix = hashlib.sha1(raw.encode()).hexdigest()[:12]
        raw = f"{lat_s}_{lon_s}_{radius_s}_{suffix}"

    return raw


def default_cache_from_env() -> GraphCache | None:
    """Return a GraphCache if IDD_GRAPH_CACHE_DIR is set, else None."""
    import os  # noqa: PLC0415

    cache_dir = os.environ.get("IDD_GRAPH_CACHE_DIR")
    if cache_dir:
        return GraphCache(cache_dir)
    return None


def _fmt(value: float) -> str:
    """Format a coordinate value for use in a cache key."""
    s = f"{abs(value):.{_COORD_PRECISION}f}"
    return f"n{s}" if value < 0 else s


def _get_key_lock(key: str) -> threading.Lock:
    """Get (or create) the per-key lock for *key*."""
    with _key_locks_mutex:
        if key not in _key_locks:
            _key_locks[key] = threading.Lock()
        return _key_locks[key]
