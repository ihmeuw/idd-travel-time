"""Per-block input volume from parquet footers: the bytes a block's bounding box makes the reader decode.

The Overture layers are read by bounding box (``inputs.read_geoparquet_bbox``).
pyarrow prunes row groups by the ``bbox`` column statistics in each file's
footer and decodes every surviving row group whole, so the bytes of the row
groups a block touches are what the block pays to read, whatever the row
filter keeps afterwards. Water row groups span most of a hemisphere (ocean
polygons), and that volume explains a block's process peak: leave-one-out R2
0.92 over the 130 blocks measured by probes 648131 and 648454 (2026-11-03),
against 0.39 for the output cell counts. The footers of all 113 files read in
about a second; nothing else is touched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from idd_travel_time.friction.build import LONLAT_PAD_DEG
from idd_travel_time.friction.frame import block_grid

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Any

    from idd_travel_time.friction.inputs import InputPaths

LAYERS = ("water", "roads", "land_cover")
VOLUME_COLS = [f"rg_mb_{layer}" for layer in LAYERS]
_BBOX_FIELDS = ("bbox.xmin", "bbox.ymin", "bbox.xmax", "bbox.ymax")


def _layer_dirs(paths: InputPaths) -> dict[str, Path]:
    return {
        "water": paths.water_dir,
        "roads": paths.roads_dir,
        "land_cover": paths.land_cover_dir,
    }


def row_group_table(paths: InputPaths) -> pd.DataFrame:
    """One row per row group of the water, roads and land-cover files: layer, file, extent, bytes, rows.

    Reads footers only. A file without ``bbox`` min/max statistics on every
    row group is an error: the reader could not prune it either, and the
    volume would be wrong rather than missing.
    """
    rows = []
    for layer, directory in _layer_dirs(paths).items():
        files = sorted(directory.glob("*.parquet"))
        if not files:
            msg = f"no parquet files for layer {layer!r} under {directory}"
            raise FileNotFoundError(msg)
        for f in files:
            md = pq.ParquetFile(f).metadata
            names = [md.schema.column(i).path for i in range(md.num_columns)]
            try:
                cols = {n: names.index(n) for n in _BBOX_FIELDS}
            except ValueError as e:
                msg = (
                    f"{f} has no bbox struct column; cannot size reads by bounding box"
                )
                raise ValueError(msg) from e
            for g in range(md.num_row_groups):
                rg = md.row_group(g)
                st = {n: rg.column(i).statistics for n, i in cols.items()}
                if any(s is None or not s.has_min_max for s in st.values()):
                    msg = f"{f} row group {g} lacks bbox statistics"
                    raise ValueError(msg)
                rows.append(
                    {
                        "layer": layer,
                        "file": f.name,
                        "row_group": g,
                        "bytes": int(rg.total_byte_size),
                        "rows": int(rg.num_rows),
                        "xmin": float(st["bbox.xmin"].min),
                        "ymin": float(st["bbox.ymin"].min),
                        "xmax": float(st["bbox.xmax"].max),
                        "ymax": float(st["bbox.ymax"].max),
                    }
                )
    return pd.DataFrame(rows)


def padded_bounds(
    block_keys: Sequence[str], tile_index: Any, resolution_m: float
) -> dict[str, tuple[float, float, float, float]]:
    """The lon/lat bounds each block reads with, padded exactly as the worker pads them."""
    return {
        k: block_grid(tile_index, k, resolution_m).lonlat_bounds(pad_deg=LONLAT_PAD_DEG)
        for k in block_keys
    }


def block_volumes(
    bounds_by_key: Mapping[str, tuple[float, float, float, float]],
    row_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Per block: MB of the row groups of each layer whose extent intersects the block's bounds.

    Columns ``rg_mb_water``, ``rg_mb_roads``, ``rg_mb_land_cover``, indexed
    by block key. A block that touches no row group of a layer gets 0.
    """
    xmin = row_groups["xmin"].to_numpy(float)
    ymin = row_groups["ymin"].to_numpy(float)
    xmax = row_groups["xmax"].to_numpy(float)
    ymax = row_groups["ymax"].to_numpy(float)
    mb = row_groups["bytes"].to_numpy(float) / 1e6
    layer_masks = {layer: (row_groups["layer"] == layer).to_numpy() for layer in LAYERS}
    out = {}
    for key, (lon0, lat0, lon1, lat1) in bounds_by_key.items():
        hit = (xmin < lon1) & (xmax > lon0) & (ymin < lat1) & (ymax > lat0)
        out[key] = {
            f"rg_mb_{layer}": float(mb[hit & mask].sum())
            for layer, mask in layer_masks.items()
        }
    frame = pd.DataFrame.from_dict(out, orient="index", columns=VOLUME_COLS)
    frame.index.name = "block_key"
    return frame.astype(float)


def volumes_for(
    block_keys: Sequence[str], paths: InputPaths, tile_index: Any, resolution_m: float
) -> pd.DataFrame:
    """``block_volumes`` for *block_keys* straight from the layers' footers."""
    table = row_group_table(paths)
    return block_volumes(padded_bounds(block_keys, tile_index, resolution_m), table)


def _unused(_: Any) -> None:  # keeps numpy imported for type narrowing in callers
    np.asarray(0)
