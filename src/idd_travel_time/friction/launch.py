"""jobmon fan-out of ``build-friction block`` over the land blocks, to the idd-tools contract.

Blocks are packed into roughly ``TARGET_MIN``-minute tasks within memory tiers
(``pack_to_target``); the packed manifest is saved and each task is submitted
by reference (``--manifest <path> --task-id <id>``, the blessed
``examples/jobmon/bundled_manifest`` transport); every task is priced by two
``ResourcePredictor`` fits on a completed run's audit joined to its per-block
stats, so a 7 s edge block and a 40 GiB city block no longer share one ask.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from idd_tools.jobmon import CellSet
from idd_tools.jobmon.manifest import Task, TaskManifest
from idd_tools.jobmon.partition import inflate_cells, pack_to_target
from idd_tools.jobmon.resources import (
    AsymmetricClipPolicy,
    LinearModel,
    ResourcePredictor,
    resources_from_predictor,
)
from idd_tools.jobmon.submit import TaskTemplateSpec

from idd_travel_time.friction import inputs as io_
from idd_travel_time.friction.frame import land_block_keys, load_tile_index

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

TEMPLATE = "friction_block"
MANIFEST_NAME = "task_manifest.json"
GUI = "https://jobmon-gui.cluster.ihme.washington.edu/#/workflow/{workflow_id}"
WORLD_BOUNDS = (-180.0, -90.0, 180.0, 90.0)
TARGET_MIN = 10.0  # packed wall time per task; clears the >=5 min sizing floor
RUNTIME_FLOOR_MIN = 6.0
MEMORY_FLOOR_GIB = 4.0
MEM_MAX_GIB = 60.0  # per-task ceilings for max_resource_check
WALL_MAX_MIN = 120.0
MIN_TRAINING_TASKS = 10
SAFETY_BUFFER = (
    0.35  # ask = 1.35 x (prediction + worst under-residual): actual lands ~74% of asked
)
COST_BINS = 5  # quantile bins of per-block seconds inside a memory tier, for packing
MEM_TIERS: tuple[tuple[float, str], ...] = (
    (12.0, "low"),
    (24.0, "mid"),
    (float("inf"), "high"),
)
TIER_ORDER = {name: i for i, (_, name) in enumerate(MEM_TIERS)}
STAT_COLS = (
    "seconds_total",
    "cells",
    "cells_land_cover",
    "motorised_cells_road",
    "motorised_cells_water",
)
FEATURE_COLS_RUNTIME = ["est_seconds"]
FEATURE_COLS_MEMORY = ["max_task_rss_gib"]


def land_blocks(paths: io_.InputPaths, resolution_m: float) -> list[str]:
    """Block keys that intersect any country polygon (ocean-only blocks are skipped)."""
    tile_index = load_tile_index(paths.tile_index_root, int(resolution_m))
    countries = io_.read_countries(paths, WORLD_BOUNDS)
    return land_block_keys(tile_index, countries)


# ----------------------------------------------------------------------------
# Evidence: a completed run's per-block stats and its sacct audit
# ----------------------------------------------------------------------------


def mem_tier(max_rss_gib: float) -> str:
    """Memory tier of a block from the peak RSS of the task that built it."""
    for edge, name in MEM_TIERS:
        if max_rss_gib <= edge:
            return name
    return MEM_TIERS[-1][1]


def block_stats(evidence_dir: Path) -> pd.DataFrame:
    """Per-block stats of a completed run (``stats/<block>.json``), indexed by block key."""
    rows = []
    for p in sorted((evidence_dir / "stats").glob("*.json")):
        d = json.loads(p.read_text())
        rows.append({"block_key": d["block_key"], **{c: d.get(c) for c in STAT_COLS}})
    if not rows:
        msg = f"no block stats under {evidence_dir / 'stats'}"
        raise FileNotFoundError(msg)
    return pd.DataFrame(rows).set_index("block_key").astype(float)


def task_audit(evidence_dir: Path) -> pd.DataFrame:
    """Completed tasks of a run's ``audit.csv`` with their block keys unpacked."""
    audit = pd.read_csv(evidence_dir / "audit.csv")
    audit = audit[audit["state"].astype(str).str.startswith("COMPLETED")].copy()
    audit["block_keys"] = audit["task_args_json"].map(
        lambda s: [k for k in json.loads(s)["block_keys"].split(",") if k]
    )
    return audit


def block_evidence(evidence_dir: Path) -> pd.DataFrame:
    """Per block: its stats, the peak RSS of the task that built it, and its memory tier."""
    stats = block_stats(evidence_dir)
    audit = task_audit(evidence_dir)
    rss: dict[str, float] = {}
    for keys, m in zip(audit["block_keys"], audit["max_rss_gib"], strict=True):
        if pd.isna(m):
            continue
        for k in keys:
            rss[k] = max(rss.get(k, 0.0), float(m))
    stats["task_max_rss_gib"] = pd.Series(rss, dtype=float)
    stats["mem_tier"] = stats["task_max_rss_gib"].map(
        lambda v: mem_tier(v) if pd.notna(v) else "mid"
    )
    return stats


def task_features(block_keys: Sequence[str], evidence: pd.DataFrame) -> dict[str, Any]:
    """Predictor features of a task from its blocks; blocks without evidence take the medians."""
    rows = evidence.reindex(list(block_keys))[list(STAT_COLS)]
    rows = rows.fillna(evidence[list(STAT_COLS)].median())
    return {
        "n_blocks": len(block_keys),
        "est_seconds": float(rows["seconds_total"].sum()),
        "max_cells_land_cover": float(rows["cells_land_cover"].max()),
        "max_cells_road": float(rows["motorised_cells_road"].max()),
        "max_cells_water": float(rows["motorised_cells_water"].max()),
        # the peak RSS of the task that last built each block; a task needs its largest block's
        "max_task_rss_gib": float(
            evidence.reindex(list(block_keys))["task_max_rss_gib"]
            .fillna(float(evidence["task_max_rss_gib"].median()))
            .max()
        ),
    }


def training_frame(evidence_dir: Path) -> pd.DataFrame:
    """One row per audited task: features recomputed from its blocks' stats, elapsed minutes, peak GiB."""
    evidence = block_evidence(evidence_dir)
    audit = task_audit(evidence_dir)
    rows = [
        {
            "task_id": t,
            **task_features(keys, evidence),
            "elapsed_min": float(e) / 60.0,
            "max_rss_gib": float(m),
        }
        for t, keys, e, m in zip(
            audit["task_id"],
            audit["block_keys"],
            audit["elapsed_seconds"],
            audit["max_rss_gib"],
            strict=True,
        )
        if pd.notna(e) and pd.notna(m) and e > 0 and m > 0
    ]
    return pd.DataFrame(rows)


def fit_predictors(
    frame: pd.DataFrame, *, buffer: float = SAFETY_BUFFER
) -> tuple[ResourcePredictor, ResourcePredictor]:
    """Runtime (minutes) and memory (GiB) predictors: linear in real units with the asymmetric clip.

    Runtime is predicted from the sum of the blocks' measured seconds; memory
    from the peak RSS of the tasks that last built the blocks (on the training
    run itself that feature is the target, so the fit is near-identity and the
    policy's margin does the sizing). Cell counts were tried first and explain
    too little of the memory: the clip's worst under-residual then pinned every
    ask at the 120 GiB cap (dry run 2026-10-23). Real units, not log space, for
    the same reason: one small task's log residual would scale every ask.
    """
    if len(frame) < MIN_TRAINING_TASKS:
        msg = f"need >= {MIN_TRAINING_TASKS} audited tasks to fit the predictors, have {len(frame)}"
        raise ValueError(msg)
    runtime = ResourcePredictor(
        LinearModel(),
        AsymmetricClipPolicy(buffer=buffer, floor=RUNTIME_FLOOR_MIN),
        target_transform="identity",
    ).fit(frame, feature_cols=FEATURE_COLS_RUNTIME, target_col="elapsed_min")
    memory = ResourcePredictor(
        LinearModel(),
        AsymmetricClipPolicy(buffer=buffer, floor=MEMORY_FLOOR_GIB),
        target_transform="identity",
    ).fit(frame, feature_cols=FEATURE_COLS_MEMORY, target_col="max_rss_gib")
    return runtime, memory


# ----------------------------------------------------------------------------
# Packing and the manifest
# ----------------------------------------------------------------------------


def block_keys_of(task: Task) -> list[str]:
    """The block keys bundled into a packed task (worker side, via ``inflate_cells``)."""
    return [str(c["block"]) for c in inflate_cells(task.task_args)]


def pack_blocks(
    block_keys: Sequence[str],
    evidence: pd.DataFrame,
    *,
    workflow_name: str,
    target_min: float = TARGET_MIN,
) -> TaskManifest:
    """Pack blocks into ~``target_min``-minute tasks, features attached.

    Cost tiers are (memory tier x quantile bin of measured seconds), so a task
    holds blocks alike in both; the bin median prices the packing, while the
    task's own features (sum of its blocks' seconds, max cell counts) price
    the ask, so packing only needs to be roughly right.
    """
    ev = evidence.reindex(list(block_keys))
    tiers = ev["mem_tier"].fillna("mid")
    secs = ev["seconds_total"].fillna(float(evidence["seconds_total"].median()))
    edges = secs.quantile([i / COST_BINS for i in range(1, COST_BINS)]).to_numpy()
    cost_bin = pd.Series(
        np.searchsorted(edges, secs.to_numpy(), side="left"), index=secs.index
    )
    labels = [f"{t}-{b}" for t, b in zip(tiers, cost_bin, strict=True)]
    order = sorted(
        range(len(block_keys)),
        key=lambda i: (TIER_ORDER[tiers.iloc[i]], int(cost_bin.iloc[i]), block_keys[i]),
    )
    cellset = CellSet(
        axes=["tier", "block"],
        cells=[{"tier": labels[i], "block": block_keys[i]} for i in order],
    )
    bin_median = secs.groupby(pd.Series(labels, index=secs.index)).median().to_dict()
    overall = max(float(secs.median()), 1.0)

    def marginal(group: dict[str, Any]) -> float:
        return max(float(bin_median.get(group["tier"], overall)), 1.0)

    packed = pack_to_target(
        cellset,
        group_by=["tier"],
        marginal_s=marginal,
        load_s=0.0,
        target_s=target_min * 60.0,
        workflow_name=workflow_name,
        task_template=TEMPLATE,
    )
    tasks = []
    for t in packed.manifest.tasks:
        keys = block_keys_of(t)
        tasks.append(
            Task(
                index=t.index,
                task_id=f"blocks:{keys[0]}+{len(keys)}",
                task_template=TEMPLATE,
                task_args=t.task_args,
                task_features=task_features(keys, evidence),
                depends_on=t.depends_on,
                shared_axes=t.shared_axes,
            )
        )
    return TaskManifest(workflow_name=workflow_name, tasks=tasks)


def reference_manifest(
    manifest: TaskManifest, manifest_path: Path, out_dir: Path
) -> TaskManifest:
    """The manifest as submitted: each task carries only the manifest path, its id and the slot.

    jobmon formats every task arg into the command string, so the cell list
    stays in the saved manifest and the worker inflates its own bundle.
    """
    return TaskManifest(
        workflow_name=manifest.workflow_name,
        tasks=[
            Task(
                index=t.index,
                task_id=t.task_id,
                task_template=t.task_template,
                task_args={
                    "manifest": str(manifest_path),
                    "task_id": t.task_id,
                    "out_dir": str(out_dir),
                },
                task_features=t.task_features,
                depends_on=t.depends_on,
                shared_axes=t.shared_axes,
            )
            for t in manifest.tasks
        ],
    )


def template_spec(script: Path | None = None) -> TaskTemplateSpec:
    """The worker command: the console script under ``.venv/bin`` by absolute path, bundle by reference."""
    exe = (Path(sys.prefix) / "bin" / "build-friction") if script is None else script
    return TaskTemplateSpec(
        command_template=f"{exe} block --manifest {{manifest}} --task-id {{task_id}} --out-dir {{out_dir}}",
        node_args=["task_id"],
        op_args=["manifest", "out_dir"],
    )


def log_paths(task: Task, out_dir: Path) -> tuple[str, str]:
    """Concrete per-task stdout/stderr paths (jobmon rejects %-placeholders)."""
    key = re.sub(r"[^A-Za-z0-9_.+-]", "_", task.task_id)
    log_dir = out_dir / "logs"
    return str(log_dir / f"{key}.out"), str(log_dir / f"{key}.err")


def resources_callable(
    manifest: TaskManifest,
    runtime: ResourcePredictor,
    memory: ResourcePredictor,
    *,
    cores: int,
) -> Callable[[Any], dict[str, str | int]]:
    """Per-task ``resources`` for ``submit_with_manifest`` from the two predictors."""
    resources: Callable[[Any], dict[str, str | int]] = resources_from_predictor(
        manifest,
        memory_predictor=memory,
        runtime_predictor=runtime,
        static={"cores": cores},
    )
    return resources


def predicted_asks(
    manifest: TaskManifest, runtime: ResourcePredictor, memory: ResourcePredictor
) -> pd.DataFrame:
    """Numeric asks per task (minutes, GiB) for reporting before submission."""
    feats = pd.DataFrame([t.task_features for t in manifest.tasks])
    return pd.DataFrame(
        {
            "task_id": [t.task_id for t in manifest.tasks],
            "n_blocks": feats["n_blocks"].to_numpy(),
            "runtime_min": runtime.predict(feats).to_numpy(),
            "memory_gib": memory.predict(feats).to_numpy(),
        }
    )


def summarise_audit(audit: Any) -> dict[str, Any]:  # pandas frame from idd_tools
    """Elapsed, memory and efficiency-zone summary of an audit frame."""
    out: dict[str, Any] = {"n": len(audit)}
    for col, name in (("elapsed_seconds", "elapsed_s"), ("max_rss_gib", "max_rss_gib")):
        if col in audit:
            s = audit[col].dropna()
            if len(s):
                out[name] = {"median": float(s.median()), "max": float(s.max())}
    for col in (
        "efficiency_zone_runtime",
        "efficiency_zone_mem",
        "efficiency_zone_overall",
    ):
        if col in audit:
            out[col] = {str(k): int(v) for k, v in audit[col].value_counts().items()}
    return out
