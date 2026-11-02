"""Compose the speed layers of one block into friction rasters and write them.

Layer precedence (DECISIONS 2026-10-21, flagged for Bobby): land cover is the
base; water replaces it where present; a road replaces both. Within a layer,
overlapping features resolve to the fastest speed. The walking surface puts
every road at 5 km/h and masks ocean; the motorised surface uses per-country
road speeds and boat speeds on water.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import rasterio
import yaml
from idd_tools.jobmon.atomic_io import AtomicRegistry, atomic_write, subtask_skip

from idd_travel_time.friction import inputs as io_
from idd_travel_time.friction import rasters
from idd_travel_time.friction.frame import block_grid, load_tile_index
from idd_travel_time.friction.params import (
    MINUTES_PER_HOUR,
    WALKING_ROAD_SPEED_KPH,
    FrictionParams,
)
from idd_travel_time.friction.slope import (
    elevation_factor,
    slope_degrees,
    tobler_factor,
)

if TYPE_CHECKING:
    import geopandas as gpd
    from numpy.typing import NDArray

    from idd_travel_time.friction.frame import BlockGrid

MODES: tuple[str, ...] = ("motorised", "walking")
LONLAT_PAD_DEG = (
    0.02  # read inputs slightly beyond the block so edge features rasterise
)
RASTER_PROFILE: dict[str, Any] = {
    "driver": "GTiff",
    "dtype": "float32",
    "count": 1,
    "nodata": np.nan,
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "compress": "ZSTD",
    "predictor": 3,
    "bigtiff": "IF_SAFER",
}


@dataclass
class BlockInputs:
    """Vector and raster inputs of one block, in their native CRS."""

    land_cover: gpd.GeoDataFrame
    roads: dict[str, gpd.GeoDataFrame]
    water: dict[str, gpd.GeoDataFrame]
    countries: gpd.GeoDataFrame
    elevation: NDArray[np.float32]
    elevation_missing_tiles: list[str] = field(default_factory=list)
    elevation_sources: dict[str, int] = field(default_factory=dict)


@dataclass
class BlockResult:
    """Outputs of one block build."""

    block_key: str
    outputs: dict[str, Path]
    stats: dict[str, Any]
    skipped: bool = False  # every output was already on disk (atomic, so complete)


def speed_kph_to_friction(speed: NDArray[np.float32]) -> NDArray[np.float32]:
    """Minutes per metre from km/h, elementwise; NaN stays NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(MINUTES_PER_HOUR / (speed * 1000.0), dtype=np.float32)


def read_block_inputs(grid: BlockGrid, paths: io_.InputPaths) -> BlockInputs:
    """Read every input for *grid*, scoped to its padded lon/lat bounds."""
    bounds = grid.lonlat_bounds(pad_deg=LONLAT_PAD_DEG)
    tiles = io_.resolve_elevation_tiles(paths, bounds)
    return BlockInputs(
        land_cover=io_.read_land_cover(paths, bounds),
        roads=io_.read_roads(paths, bounds),
        water=io_.read_water(paths, bounds),
        countries=io_.read_countries(paths, bounds),
        elevation=io_.elevation_on_grid(grid, tiles.files),
        elevation_missing_tiles=tiles.missing,
        elevation_sources=tiles.sources,
    )


def offroad_speed(
    grid: BlockGrid, data: BlockInputs, params: FrictionParams, stats: dict[str, Any]
) -> NDArray[np.float32]:
    """Land-cover walking speed times the slope and altitude multipliers."""
    overlap = np.zeros(grid.shape, dtype=np.uint8)
    lc = rasters.land_cover_speed(grid, data.land_cover, params, count=overlap)
    dx, dy = grid.ground_lengths()
    slope = slope_degrees(data.elevation, dx, dy)
    factor = tobler_factor(slope, params.slope) * elevation_factor(
        data.elevation, params.slope
    )
    stats["cells_land_cover"] = int(np.isfinite(lc).sum())
    stats["cells_land_cover_overlap"] = int((overlap >= 2).sum())  # noqa: PLR2004
    stats["cells_elevation_missing"] = int(np.isnan(data.elevation).sum())
    stats["elevation_missing_tiles"] = data.elevation_missing_tiles
    stats["elevation_source_tiles"] = dict(data.elevation_sources)
    return np.asarray(lc * factor, dtype=np.float32)


def compose_mode(
    grid: BlockGrid,
    data: BlockInputs,
    params: FrictionParams,
    offroad: NDArray[np.float32],
    mode: str,
    stats: dict[str, Any],
) -> NDArray[np.float32]:
    """Speed (km/h) per cell for one mode, applying layer precedence."""
    if mode not in MODES:
        msg = f"unknown mode {mode!r}; expected one of {MODES}"
        raise ValueError(msg)
    speed = offroad.copy()
    water, mask = rasters.water_speed(grid, data.water, params, mode=mode)
    has_water = np.isfinite(water)
    speed[has_water] = water[has_water]
    country_idx, iso3s = rasters.country_raster(grid, data.countries)
    road = rasters.road_speed(
        grid,
        data.roads,
        country_idx,
        iso3s,
        params,
        walking_kph=WALKING_ROAD_SPEED_KPH if mode == "walking" else None,
    )
    has_road = np.isfinite(road)
    speed[has_road] = road[has_road]
    speed[mask] = np.nan
    stats[f"{mode}_cells_water"] = int(has_water.sum())
    stats[f"{mode}_cells_road"] = int(has_road.sum())
    stats[f"{mode}_cells_masked"] = int(mask.sum())
    stats[f"{mode}_cells_nodata"] = int(np.isnan(speed).sum())
    stats[f"{mode}_cells_no_country"] = (
        int((country_idx == 0).sum()) if mode == "motorised" else None
    )
    return speed


def _clear_stale_tmp(path: Path) -> None:
    """Remove a ``<path>.tmp`` left by a killed writer so idd-tools ``atomic_write`` can start."""
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        tmp.unlink()


def write_raster(
    path: Path,
    array: NDArray[np.float32],
    grid: BlockGrid,
    registry: AtomicRegistry | None = None,
) -> None:
    """GeoTIFF through idd-tools ``atomic_write`` (tempfile + rename), registered in the task sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        **RASTER_PROFILE,
        "height": grid.shape[0],
        "width": grid.shape[1],
        "crs": grid.crs,
        "transform": grid.transform,
    }

    def write(tmp: Path) -> None:
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(array, 1)
        with rasterio.open(tmp) as check:
            if check.shape != grid.shape or check.count != 1:
                msg = f"written raster {tmp} has shape {check.shape}, expected {grid.shape}"
                raise RuntimeError(msg)

    _clear_stale_tmp(path)
    atomic_write(path, write)
    if registry is not None:
        registry.register(path)


def write_json(
    path: Path, payload: dict[str, Any], registry: AtomicRegistry | None = None
) -> None:
    """JSON through ``atomic_write``, registered like the rasters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _clear_stale_tmp(path)
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(payload, indent=1)))
    if registry is not None:
        registry.register(path)


def block_outputs(
    out_dir: Path, block_key: str, modes: tuple[str, ...]
) -> dict[str, Path]:
    """Every file a block build produces: one raster per mode plus its stats json."""
    out = {mode: out_dir / mode / f"{block_key}.tif" for mode in modes}
    out["stats"] = out_dir / "stats" / f"{block_key}.json"
    return out


def block_done(out_dir: Path, block_key: str, modes: tuple[str, ...] = MODES) -> bool:
    """True when every output is on disk; ``subtask_skip`` is atomic-aware, so present means complete."""
    return all(
        subtask_skip(p) for p in block_outputs(out_dir, block_key, modes).values()
    )


def use_system_memory_pool() -> str:
    """Route arrow allocations through the system allocator; returns the pool backend name.

    Arrow's default pool (jemalloc) retains tens of GB after the parquet reads
    on 56-core compute nodes: the Amsterdam block showed 45 GB in sacct with
    the default pool and 16 GB with the system pool, at similar wall time
    (probe tests 2026-10-21). The retained memory counts against the task's
    cgroup, so it sets the Slurm ask; the process's own anonymous peak is
    about 11 GB either way.
    """
    pa.set_memory_pool(pa.system_memory_pool())
    if os.environ.get("ARROW_DEFAULT_MEMORY_POOL") != "system":
        # The Python-side switch above is not enough for the C++ scanner; the
        # console script sets the variable before pyarrow loads (cli.py).
        warnings.warn(
            "ARROW_DEFAULT_MEMORY_POOL is not 'system'; expect ~45 GB cgroup usage per block",
            stacklevel=2,
        )
    return str(pa.default_memory_pool().backend_name)


def build_block(
    block_key: str,
    params: FrictionParams,
    paths: io_.InputPaths,
    out_dir: Path,
    *,
    resolution_m: float,
    modes: tuple[str, ...] = MODES,
    registry: AtomicRegistry | None = None,
) -> BlockResult:
    """Build and write the friction rasters of one block for each mode.

    Done work is skipped (``subtask_skip`` on every output), so a retried or
    resumed task redoes nothing that is already complete on disk.
    """
    t0 = time.time()
    outputs_all = block_outputs(out_dir, block_key, modes)
    if all(subtask_skip(p) for p in outputs_all.values()):
        saved = json.loads(outputs_all["stats"].read_text())
        rasters_only = {m: outputs_all[m] for m in modes}
        return BlockResult(
            block_key=block_key, outputs=rasters_only, stats=saved, skipped=True
        )
    use_system_memory_pool()
    tile_index = load_tile_index(paths.tile_index_root, int(resolution_m))
    grid = block_grid(tile_index, block_key, resolution_m)
    data = read_block_inputs(grid, paths)
    stats: dict[str, Any] = {
        "block_key": block_key,
        "cells": int(np.prod(grid.shape)),
        "clipped_edge_block": grid.clipped,
        "seconds_read": round(time.time() - t0, 1),
    }
    offroad = offroad_speed(grid, data, params, stats)
    outputs: dict[str, Path] = {}
    for mode in modes:
        speed = compose_mode(grid, data, params, offroad, mode, stats)
        out = outputs_all[mode]
        write_raster(out, speed_kph_to_friction(speed), grid, registry=registry)
        outputs[mode] = out
        del speed
    stats["seconds_total"] = round(time.time() - t0, 1)
    write_json(outputs_all["stats"], stats, registry=registry)
    return BlockResult(block_key=block_key, outputs=outputs, stats=stats)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    cmd = ["git", "-C", str(repo), *args]
    return subprocess.run(  # noqa: S603  # fixed argv, no shell; repo path from the package location
        cmd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_state(repo: Path) -> dict[str, Any]:
    try:
        commit = _git(repo, "rev-parse", "HEAD")
        dirty = _git(repo, "status", "--short") != ""
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def write_manifest(
    out_dir: Path,
    params: FrictionParams,
    paths: io_.InputPaths,
    *,
    resolution_m: float,
    modes: tuple[str, ...],
    label: str,
    repo_root: Path,
) -> Path:
    """Record everything a build depends on; one manifest per version directory."""
    tables = {
        p.name: _sha256(p) for p in sorted(params.table_dir.iterdir()) if p.is_file()
    }
    info_path = paths.tile_index_root / f"{int(resolution_m)}m" / "tile_index_info.yaml"
    manifest = {
        "label": label,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "resolution_m": resolution_m,
        "frame": yaml.safe_load(info_path.read_text()) if info_path.exists() else None,
        "modes": list(modes),
        "units": "minutes per metre, float32, NaN = no data (ocean on the walking surface, or no source layer)",
        "layer_precedence": "land cover (fastest subtype on overlap) < water (replaces) < roads (replaces); fastest road on overlap",
        "walking_road_speed_kph": WALKING_ROAD_SPEED_KPH,
        "inputs": {
            "overture_roads_dir": str(paths.roads_dir),
            "overture_water_dir": str(paths.water_dir),
            "overture_land_cover_dir": str(paths.land_cover_dir),
            "overture_divisions_dir": str(paths.divisions_dir),
            "fabdem_dir": str(paths.fabdem_dir) if paths.fabdem_dir else None,
            "elevation_dir": str(paths.elevation_dir),
            "elevation_fallback_dir": str(paths.elevation_fallback_dir)
            if paths.elevation_fallback_dir
            else None,
            "elevation_precedence": list(io_.ELEVATION_SOURCE_ORDER),
            "tile_index_root": str(paths.tile_index_root),
        },
        "tables": {"dir": str(params.table_dir), "sha256": tables},
        "builder": {"package": "idd_travel_time.friction", **_git_state(repo_root)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "manifest.yaml"
    out.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return out
