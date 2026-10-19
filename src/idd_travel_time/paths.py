"""Per-machine absolute roots and the output-node layout.

Two-tier path system (STANDARDS §Path safety): nothing committed holds an
absolute path. ``paths.yaml`` at the repo root — gitignored, copied from
``paths.yaml.example`` — holds this machine's roots and is read at call time,
never at import time, so importing the package needs no ``paths.yaml``.

Output management (STANDARDS §Output management): the project output root
holds one directory per *node* (versioned component). Code writes only to a
node's ``working/`` slot, or to ``scratch/<label>/`` for throwaway runs;
snapshots are created with ``idd-versions <node> freeze`` and never by code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from idd_tools.versions import scratch_dir as _scratch_dir
from idd_tools.versions import working_dir as _working_dir

_PATHS_FILE = Path(__file__).resolve().parents[2] / "paths.yaml"

# Nodes under ``project_output_root``.
NODE_MAPS = "maps"  # rendered maps: HTML (folium), PNG / PDF (static)
NODE_RESULTS = "results"  # TravelTimeResult parquet directories, surfaces, GeoTIFFs


def load_paths(paths_file: Path | None = None) -> dict[str, Any]:
    """Read ``paths.yaml`` (or *paths_file*) into a dict of absolute roots."""
    path = _PATHS_FILE if paths_file is None else paths_file
    if not path.exists():
        msg = f"{path.name} missing. Run: cp paths.yaml.example paths.yaml; then edit."
        raise SystemExit(msg)
    with path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def output_root(paths_file: Path | None = None) -> Path:
    """``project_output_root`` from ``paths.yaml``, user-expanded."""
    return Path(load_paths(paths_file)["project_output_root"]).expanduser()


def node_dir(node: str, paths_file: Path | None = None) -> Path:
    """``<project_output_root>/<node>`` — pass to ``idd-versions`` and the launcher flags."""
    return output_root(paths_file) / node


def working_dir(
    node: str, *, clean: bool = False, paths_file: Path | None = None
) -> Path:
    """``<node>/working/``, created if absent: where a run writes (see idd_tools.versions)."""
    return Path(_working_dir(node_dir(node, paths_file), clean=clean))


def scratch_dir(node: str, label: str, paths_file: Path | None = None) -> Path:
    """``<node>/scratch/<label>/``: a throwaway run that is never frozen."""
    return Path(_scratch_dir(node_dir(node, paths_file), label))
