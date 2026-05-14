"""GeoParquet serialisation for TravelTimeResult.

Layout on disk
--------------
Each result occupies one directory::

    {output_dir}/
    └── {result_id}/
        ├── meta.json          request provenance + router_name + schema version
        ├── isochrone.parquet  always present
        ├── network.parquet    omitted when network_gdf is None
        └── poi.parquet        omitted when poi_gdf is None

``result_id`` is derived from the request parameters so the same request always
maps to the same directory name. This makes the store idempotent — re-running
over the same origin/mode/duration simply overwrites the existing files.

Result ID scheme
----------------
``{lat}_{lon}_{mode}_{duration}min_{time_context}``

Coordinates are rounded to 6 decimal places (~0.1 m precision) and negative
signs replaced with ``n`` to keep the path safe on all filesystems.

    -1.286389, 36.817223, walking, 30, off_peak
    → n1.286389_36.817223_walking_30min_off_peak

Public API
----------
write_result(result, output_dir)  -> Path   (the result directory)
read_result(result_dir)           -> TravelTimeResult
result_id(request)                -> str
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd

if TYPE_CHECKING:
    from idd_travel_time.models import TravelTimeRequest, TravelTimeResult

_SCHEMA_VERSION = "1"
_META_FILE = "meta.json"
_ISOCHRONE_FILE = "isochrone.parquet"
_NETWORK_FILE = "network.parquet"
_POI_FILE = "poi.parquet"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def result_id(request: TravelTimeRequest) -> str:
    """Deterministic directory name for a given request.

    Safe for use as a filesystem path on Linux, macOS, and Windows.
    """
    lat = _fmt_coord(request.origin_lat)
    lon = _fmt_coord(request.origin_lon)
    mode = request.mode.value
    duration = request.duration_minutes
    context = request.time_context.value

    parts = [lat, lon, mode, f"{duration}min", context]

    if request.transit_profile is not None:
        # Include transit profile parameters so two requests that differ only
        # in their transit assumptions get distinct result directories.
        lo, hi = request.transit_profile.typical_speed_kph
        parts.append(f"transit_{lo:.1f}_{hi:.1f}kph")

    return "_".join(parts)


def write_result(
    result: TravelTimeResult,
    output_dir: str | Path,
) -> Path:
    """Serialise *result* to a GeoParquet result directory.

    Writes atomically: all files go to a temp directory first, then the whole
    directory is moved into place. A partial write never appears in
    *output_dir*.

    Parameters
    ----------
    result:
        Completed TravelTimeResult.
    output_dir:
        Parent directory. Created if it does not exist.

    Returns
    -------
    Path
        The result directory (``output_dir / result_id``).
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rid = result_id(result.request)
    final_dir = out_root / rid

    # Write to a temp dir inside output_dir, then rename atomically.
    with tempfile.TemporaryDirectory(dir=out_root, prefix=f".tmp_{rid}_") as tmp:
        tmp_dir = Path(tmp)

        # meta.json
        meta = _build_meta(result)
        (tmp_dir / _META_FILE).write_text(json.dumps(meta, indent=2))

        # GeoDataFrames
        _write_gdf(result.isochrone_gdf, tmp_dir / _ISOCHRONE_FILE)
        if result.network_gdf is not None:
            _write_gdf(result.network_gdf, tmp_dir / _NETWORK_FILE)
        if result.poi_gdf is not None:
            _write_gdf(result.poi_gdf, tmp_dir / _POI_FILE)

        # Atomic rename — remove existing result dir if present
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(tmp_dir, final_dir)

    return final_dir


def read_result(result_dir: str | Path) -> TravelTimeResult:
    """Deserialise a TravelTimeResult from a result directory.

    Parameters
    ----------
    result_dir:
        Path produced by ``write_result()``.

    Returns
    -------
    TravelTimeResult
        Fully reconstructed result including the original request.

    Raises
    ------
    FileNotFoundError
        If *result_dir* or ``meta.json`` / ``isochrone.parquet`` are missing.
    ValueError
        If the schema version in ``meta.json`` is not supported.
    """
    d = Path(result_dir)
    if not d.is_dir():
        msg = f"Result directory not found: {d}"
        raise FileNotFoundError(msg)

    meta_path = d / _META_FILE
    if not meta_path.exists():
        msg = f"meta.json not found in {d}"
        raise FileNotFoundError(msg)

    meta = json.loads(meta_path.read_text())
    _check_schema_version(meta, d)

    request = _request_from_meta(meta)
    isochrone_gdf = _read_gdf(d / _ISOCHRONE_FILE)

    network_path = d / _NETWORK_FILE
    network_gdf = _read_gdf(network_path) if network_path.exists() else None

    poi_path = d / _POI_FILE
    poi_gdf = _read_gdf(poi_path) if poi_path.exists() else None

    # Import here to avoid circular import at module load.
    from idd_travel_time.models import TravelTimeResult

    return TravelTimeResult(
        request=request,
        isochrone_gdf=isochrone_gdf,
        network_gdf=network_gdf,
        poi_gdf=poi_gdf,
        router_name=meta["router_name"],
    )


# ---------------------------------------------------------------------------
# Private helpers — serialisation
# ---------------------------------------------------------------------------


def _fmt_coord(value: float) -> str:
    """Format a coordinate for use in a result_id path component.

    Rounds to 6 decimal places; replaces '-' with 'n' for FS safety.
    """
    formatted = f"{value:.6f}"
    return formatted.replace("-", "n")


def _build_meta(result: TravelTimeResult) -> dict[str, Any]:
    """Serialise request + router_name to a JSON-compatible dict."""
    req = result.request
    meta: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "router_name": result.router_name,
        "request": {
            "origin_lat": req.origin_lat,
            "origin_lon": req.origin_lon,
            "mode": req.mode.value,
            "duration_minutes": req.duration_minutes,
            "time_context": req.time_context.value,
            "viz_layers": [v.value for v in req.viz_layers],
            "transit_profile": None,
        },
    }
    if req.transit_profile is not None:
        tp = req.transit_profile
        meta["request"]["transit_profile"] = {
            "typical_speed_kph": list(tp.typical_speed_kph),
            "coverage": tp.coverage.value,
            "time_context": tp.time_context.value,
            "mode_name": tp.mode_name,
        }
    return meta


def _write_gdf(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write a GeoDataFrame to GeoParquet atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".parquet", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        gdf.to_parquet(tmp_path)
        shutil.move(str(tmp_path), str(path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_gdf(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        msg = f"Parquet file not found: {path}"
        raise FileNotFoundError(msg)
    return gpd.read_parquet(path)


# ---------------------------------------------------------------------------
# Private helpers — deserialisation
# ---------------------------------------------------------------------------


def _check_schema_version(meta: dict[str, Any], result_dir: Path) -> None:
    version = meta.get("schema_version")
    if version != _SCHEMA_VERSION:
        msg = (
            f"Unsupported schema version {version!r} in {result_dir}. "
            f"Expected {_SCHEMA_VERSION!r}. "
            "Re-run compute() to regenerate this result."
        )
        raise ValueError(msg)


def _request_from_meta(meta: dict[str, Any]) -> TravelTimeRequest:
    """Reconstruct a TravelTimeRequest from the meta.json request block."""
    from idd_travel_time.constants import (
        TimeContext,
        TransitCoverage,
        TransportMode,
        VizLayer,
    )
    from idd_travel_time.models import (
        TransitProfile,
        TravelTimeRequest,
    )

    r = meta["request"]

    transit_profile = None
    if r["transit_profile"] is not None:
        tp = r["transit_profile"]
        transit_profile = TransitProfile(
            typical_speed_kph=tuple(tp["typical_speed_kph"]),
            coverage=TransitCoverage(tp["coverage"]),
            time_context=TimeContext(tp["time_context"]),
            mode_name=tp["mode_name"],
        )

    return TravelTimeRequest(
        origin_lat=r["origin_lat"],
        origin_lon=r["origin_lon"],
        mode=TransportMode(r["mode"]),
        duration_minutes=r["duration_minutes"],
        time_context=TimeContext(r["time_context"]),
        viz_layers=[VizLayer(v) for v in r["viz_layers"]],
        transit_profile=transit_profile,
    )
