"""jobmon fan-out of ``build-friction block`` over the land blocks, to the idd-tools contract.

Blocks are packed into roughly ``TARGET_MIN``-minute tasks within memory tiers
(``pack_to_target``); the packed manifest is saved and each task is submitted
by reference (``--manifest <path> --task-id <id>``, the blessed
``examples/jobmon/bundled_manifest`` transport); every task is priced by two
``ResourcePredictor`` fits whose features are properties of its blocks (cell
counts by layer, latitude), never a previous run's timings (DECISIONS
2026-11-03): block-level ``LinearModel`` fits give each block a predicted
build time and process peak, a task takes the sum and the max over its
blocks, and a probed slot's own measures calibrate the task-level asks.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from idd_tools.jobmon import CellSet, cost_by_feature
from idd_tools.jobmon.atomic_io import consolidate_atomic_registry
from idd_tools.jobmon.manifest import Task, TaskManifest
from idd_tools.jobmon.partition import inflate_cells, pack_to_target
from idd_tools.jobmon.resource_audit import feed_audit_to_predictor
from idd_tools.jobmon.resources import (
    AsymmetricClipPolicy,
    LinearModel,
    ResourcePredictor,
    resources_from_predictor,
)
from idd_tools.jobmon.scope import stratified_scope
from idd_tools.jobmon.submit import TaskTemplateSpec

from idd_travel_time.friction import inputs as io_
from idd_travel_time.friction.frame import block_grid, land_block_keys, load_tile_index
from idd_travel_time.friction.volume import VOLUME_COLS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

TEMPLATE = "friction_block"
MANIFEST_NAME = (
    "task_manifest.json"  # the latest packing; each launch also keeps a stamped copy
)
GUI = "https://jobmon-gui.cluster.ihme.washington.edu/#/workflow/{workflow_id}"
WORLD_BOUNDS = (-180.0, -90.0, 180.0, 90.0)
TARGET_MIN = 10.0  # packed wall time per task; clears the >=5 min sizing floor
RUNTIME_FLOOR_MIN = 6.0
MEMORY_FLOOR_GIB = 4.0
MEM_MAX_GIB = 60.0  # per-task ceilings for max_resource_check
WALL_MAX_MIN = 120.0
MIN_TRAINING_TASKS = 10
MIN_BLOCK_PEAKS = 20  # measured process peaks needed to fit the block memory model
# ask = (1 + buffer) x (prediction + worst under-residual). Bobby's target is
# the mean used/asked near 0.65 with a few reruns; a timed-out task resumes on
# retry at 1.5x, but jobmon does not scale memory on an OOM (probe 648454), so
# memory carries the larger buffer.
RUNTIME_BUFFER = 0.15
MEMORY_BUFFER = 0.25
COST_BINS = (
    5  # quantile bins of predicted block seconds inside a memory tier, for packing
)
# Memory tiers on the PREDICTED process peak of a block (GiB); over all 2,418
# blocks the volume model predicts 4.8-16.9 GiB, median 12.2 (2026-11-03).
MEM_TIERS: tuple[tuple[float, str], ...] = (
    (10.0, "low"),
    (13.0, "mid"),
    (float("inf"), "high"),
)
TIER_ORDER = {name: i for i, (_, name) in enumerate(MEM_TIERS)}
STAT_COLS = (
    "seconds_total",
    "cells",
    "cells_land_cover",
    "cells_land_cover_overlap",
    "motorised_cells_road",
    "motorised_cells_water",
    "cells_elevation_missing",
)
# Block-level covariates: block properties only, never a prior run's timings
# (DECISIONS 2026-11-03). Memory is the volume of parquet row groups the
# block's bounding box makes the reader decode (``volume.py``): leave-one-out
# R2 0.93 over 130 measured blocks, worst under-prediction 2 GiB. Runtime adds
# the output cell counts (R2 0.56 on the same blocks).
RUNTIME_BLOCK_FEATURES = [
    "cells_land_cover",
    "motorised_cells_road",
    "motorised_cells_water",
    "cells_elevation_missing",
    *VOLUME_COLS,
]
MEMORY_BLOCK_FEATURES = [*VOLUME_COLS]
FEATURE_COLS_RUNTIME = [
    "pred_seconds"
]  # sum of the block model's seconds over the task
FEATURE_COLS_MEMORY = [
    "max_pred_peak_gib"
]  # blocks run one at a time: the largest block sets the task
PROBE_BY = "mem_tier"  # the task feature the probe is stratified on (idd-tools stratified_scope)
PROBE_SEED = 0
# The REPORT line TaskProgressLogger.report() writes; parse_probe_logs keeps the
# maximum over a whole tree, parse_task_logs one line per file. A bundle task
# writes one line per block child, so the per-task peak needs every line.
REPORT_RE = re.compile(
    r"\[(?P<tag>[^\]]+)\] REPORT: total (?P<seconds>[\d.]+)s, peak RSS (?P<gib>[\d.]+) GiB"
)
BUNDLE_OVERHEAD_MIN = (
    0.5  # parent wall time beyond its blocks' build times (probe 648131: 0.3-0.5)
)


def land_blocks(paths: io_.InputPaths, resolution_m: float) -> list[str]:
    """Block keys that intersect any country polygon (ocean-only blocks are skipped)."""
    tile_index = load_tile_index(paths.tile_index_root, int(resolution_m))
    countries = io_.read_countries(paths, WORLD_BOUNDS)
    return land_block_keys(tile_index, countries)


# ----------------------------------------------------------------------------
# Evidence: per-block properties, measured peaks, and the block-level models
# ----------------------------------------------------------------------------


def mem_tier(pred_peak_gib: float) -> str:
    """Memory tier of a block from its predicted process peak."""
    for edge, name in MEM_TIERS:
        if pred_peak_gib <= edge:
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


def abs_latitudes(
    block_keys: Sequence[str], tile_index: Any, resolution_m: float
) -> pd.Series:
    """Absolute latitude of each block's centre, from the frame (degrees)."""
    out = {}
    for k in block_keys:
        _lon0, lat0, _lon1, lat1 = block_grid(
            tile_index, k, resolution_m
        ).lonlat_bounds()
        out[k] = abs((lat0 + lat1) / 2.0)
    return pd.Series(out, dtype=float, name="abs_lat")


def block_peaks(slot: Path) -> pd.Series:
    """Per block: the process peak RSS (GiB) measured when the block was built in *slot*.

    From each block's stats (``peak_rss_gib``, written by the instrumented
    worker), else from ``block_peaks.csv`` beside the stats (probe 648131's
    REPORT lines mapped to blocks by log order and verified against each
    block's own timing). A stats value overrides the csv for the same block.
    """
    peaks: dict[str, float] = {}
    csv = slot / "block_peaks.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        peaks.update(
            dict(zip(df["block_key"], df["peak_rss_gib"].astype(float), strict=True))
        )
    stats_dir = slot / "stats"
    if stats_dir.is_dir():
        for p in sorted(stats_dir.glob("*.json")):
            d = json.loads(p.read_text())
            if d.get("peak_rss_gib") is not None:
                peaks[d["block_key"]] = float(d["peak_rss_gib"])
    return pd.Series(peaks, dtype=float, name="peak_rss_gib")


@dataclass
class BlockModels:
    """The two block-level fits: build seconds and process peak from block content."""

    runtime: LinearModel
    memory: LinearModel
    n_runtime: int
    n_memory: int


def fit_block_models(stats: pd.DataFrame, peaks: pd.Series) -> BlockModels:
    """Fit seconds ~ content on every block of the evidence run, peak ~ content on the measured blocks."""
    rt = stats.dropna(subset=[*RUNTIME_BLOCK_FEATURES, "seconds_total"])
    runtime = LinearModel().fit(rt[RUNTIME_BLOCK_FEATURES], rt["seconds_total"])
    mem = stats.join(peaks.rename("peak_rss_gib"), how="inner").dropna(
        subset=[*MEMORY_BLOCK_FEATURES, "peak_rss_gib"]
    )
    if len(mem) < MIN_BLOCK_PEAKS:
        msg = f"need >= {MIN_BLOCK_PEAKS} blocks with a measured process peak, have {len(mem)}"
        raise ValueError(msg)
    memory = LinearModel().fit(mem[MEMORY_BLOCK_FEATURES], mem["peak_rss_gib"])
    return BlockModels(runtime, memory, len(rt), len(mem))


def block_evidence(
    evidence_dir: Path,
    *,
    latitudes: pd.Series,
    peaks: pd.Series,
    volumes: pd.DataFrame,
) -> pd.DataFrame:
    """Per block: its stats, latitude, input volumes, predicted seconds and peak, and its memory tier.

    *latitudes* (``abs_latitudes``) and *volumes* (``volume.volumes_for``) must
    cover every block in the evidence stats; *peaks* are the measured process
    peaks the memory model is fitted on (``block_peaks`` of a probed slot).
    """
    stats = block_stats(evidence_dir)
    stats["abs_lat"] = latitudes.reindex(stats.index).astype(float)
    if stats["abs_lat"].isna().any():
        missing = stats.index[stats["abs_lat"].isna()].tolist()
        msg = f"latitude missing for {len(missing)} blocks, e.g. {missing[:3]}"
        raise ValueError(msg)
    vols = volumes.reindex(stats.index)[VOLUME_COLS].astype(float)
    if vols.isna().any().any():
        missing = stats.index[vols.isna().any(axis=1)].tolist()
        msg = f"input volume missing for {len(missing)} blocks, e.g. {missing[:3]}"
        raise ValueError(msg)
    stats[VOLUME_COLS] = vols
    models = fit_block_models(stats, peaks)
    rt_feats = stats[RUNTIME_BLOCK_FEATURES].fillna(
        stats[RUNTIME_BLOCK_FEATURES].median()
    )
    stats["pred_seconds"] = np.maximum(models.runtime.predict(rt_feats).to_numpy(), 1.0)
    mem_feats = stats[MEMORY_BLOCK_FEATURES].fillna(
        stats[MEMORY_BLOCK_FEATURES].median()
    )
    stats["pred_peak_gib"] = np.maximum(
        models.memory.predict(mem_feats).to_numpy(), 1.0
    )
    stats["mem_tier"] = stats["pred_peak_gib"].map(mem_tier)
    return stats


def task_features(block_keys: Sequence[str], evidence: pd.DataFrame) -> dict[str, Any]:
    """Predictor features of a task from its blocks; blocks without evidence take the medians.

    ``pred_seconds`` sums the blocks' predicted build seconds, ``max_pred_peak_gib``
    is the largest predicted process peak among them (blocks run one at a time),
    and ``mem_tier`` the heaviest tier: the covariate the probe is stratified on.
    """
    rows = evidence.reindex(list(block_keys))
    pred_s = rows["pred_seconds"].fillna(float(evidence["pred_seconds"].median()))
    pred_p = rows["pred_peak_gib"].fillna(float(evidence["pred_peak_gib"].median()))
    tiers = rows["mem_tier"].fillna("mid")
    return {
        "n_blocks": len(block_keys),
        PROBE_BY: max((str(t) for t in tiers), key=lambda t: TIER_ORDER[t]),
        "pred_seconds": float(pred_s.sum()),
        "max_pred_peak_gib": float(pred_p.max()),
    }


def _predictors(
    runtime_buffer: float, memory_buffer: float
) -> tuple[ResourcePredictor, ResourcePredictor]:
    """Unfitted (runtime, memory) predictors: linear in real units, asymmetric clip, floors."""
    runtime = ResourcePredictor(
        LinearModel(),
        AsymmetricClipPolicy(buffer=runtime_buffer, floor=RUNTIME_FLOOR_MIN),
        target_transform="identity",
    )
    memory = ResourcePredictor(
        LinearModel(),
        AsymmetricClipPolicy(buffer=memory_buffer, floor=MEMORY_FLOOR_GIB),
        target_transform="identity",
    )
    return runtime, memory


def fit_predictors(
    frame: pd.DataFrame,
    *,
    runtime_buffer: float = RUNTIME_BUFFER,
    memory_buffer: float = MEMORY_BUFFER,
) -> tuple[ResourcePredictor, ResourcePredictor]:
    """Task-level runtime (minutes) and memory (GiB) predictors on a probed slot's measures.

    The features are the block models' sums and maxima (``task_features``);
    the targets are the slot's measured compute minutes and REPORT peaks
    (``slot_training_frame``). The fit is near-identity and the clip policy
    does the sizing. Fitted through ``feed_audit_to_predictor``, which refuses
    fewer than ``MIN_TRAINING_TASKS`` rows.
    """
    runtime, memory = _predictors(runtime_buffer, memory_buffer)
    feed_audit_to_predictor(
        frame,
        runtime,
        feature_cols=FEATURE_COLS_RUNTIME,
        target_col="elapsed_min",
        min_tasks=MIN_TRAINING_TASKS,
    )
    feed_audit_to_predictor(
        frame,
        memory,
        feature_cols=FEATURE_COLS_MEMORY,
        target_col="max_rss_gib",
        min_tasks=MIN_TRAINING_TASKS,
    )
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

    Cost tiers are (memory tier x quantile bin of predicted seconds), so a task
    holds blocks alike in both; the bin median prices the packing, while the
    task's own features (sum of predicted seconds, max predicted peak) price
    the ask, so packing only needs to be roughly right.
    """
    ev = evidence.reindex(list(block_keys))
    tiers = ev["mem_tier"].fillna("mid")
    secs = ev["pred_seconds"].fillna(float(evidence["pred_seconds"].median()))
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
    """Concrete per-task stdout/stderr paths (jobmon rejects %-placeholders).

    jobmon makes each path a directory with one subdirectory per Slurm attempt.
    """
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


# ----------------------------------------------------------------------------
# The probe: stratified on the memory tier, measured per task from its own slot
# ----------------------------------------------------------------------------


def stratified_probe(
    manifest: TaskManifest, n_per_tier: int, *, seed: int = PROBE_SEED
) -> TaskManifest:
    """The probe: ``n_per_tier`` tasks from every memory tier, sampled within the tier.

    idd-tools' flat ``probe_only`` takes the first N tasks of the manifest, and
    ``pack_blocks`` orders tasks low tier first and cheapest cost bin first, so
    a flat probe would measure only the lightest, widest bundles. Stratifying
    on ``mem_tier`` (``stratified_scope``) measures every tier; sampling within
    a tier avoids its cheapest bin, which comes first in manifest order.
    """
    return stratified_scope(manifest, n_per_tier, by=PROBE_BY, random=True, seed=seed)


def tasks_per_tier(manifest: TaskManifest) -> dict[str, int]:
    """Task count per ``mem_tier``, in tier order."""
    counts = Counter(str(t.task_features[PROBE_BY]) for t in manifest.tasks)
    return {tier: counts[tier] for _, tier in MEM_TIERS if tier in counts}


def report_peaks(log_dir: Path) -> pd.Series:
    """Per task: the largest process peak RSS (GiB) over every REPORT line under *log_dir*.

    A bundle task writes one REPORT line per block child plus one for the
    parent, all tagged with the task id, in one subdirectory per Slurm attempt.
    The child's peak is the block's own resident set; sacct's MaxRSS is the
    cgroup's peak, which fills with file cache to whatever was asked
    (DEAD_ENDS 2026-10-21).
    """
    peaks: dict[str, float] = {}
    if not log_dir.is_dir():
        return pd.Series(peaks, dtype=float, name="report_peak_gib")
    for path in log_dir.rglob("*"):
        if not path.is_file():
            continue
        for m in REPORT_RE.finditer(path.read_text(errors="replace")):
            peaks[m["tag"]] = max(peaks.get(m["tag"], 0.0), float(m["gib"]))
    return pd.Series(peaks, dtype=float, name="report_peak_gib")


def atomic_registry(slot: Path) -> pd.DataFrame:
    """The slot's consolidated atomic registry (one row per registered file), without writing to a snapshot.

    ``consolidate_atomic_registry`` folds the ``.atomic`` sidecars into
    ``.atomic_registry.parquet`` and writes it; a frozen snapshot is read-only,
    so there the parquet the audit left before the freeze is read as is. Empty
    when the slot has no sidecars.
    """
    if not (slot / ".atomic").is_dir():
        return pd.DataFrame(
            columns=["task_id", "task_template", "path", "folder", "registered_at"]
        )
    parquet = slot / ".atomic_registry.parquet"
    if os.access(slot, os.W_OK):
        return pd.read_parquet(consolidate_atomic_registry(slot))
    if parquet.exists():
        return pd.read_parquet(parquet)
    msg = f"{slot} is read-only and holds no consolidated registry; consolidate it before freezing"
    raise PermissionError(msg)


def registered_blocks(slot: Path) -> dict[str, list[str]]:
    """Per task: the block keys whose stats file it registered in its ``.atomic`` sidecar.

    The sidecar is the record of what a task actually built, across every
    attempt (always-append), so it survives the slot's manifest being
    re-packed by a later launch; the manifest only says what was planned.
    """
    registry = atomic_registry(slot)
    out: dict[str, list[str]] = {}
    for tid, path in zip(registry["task_id"], registry["path"], strict=True):
        p = Path(str(path))
        if p.parent.name == "stats" and p.suffix == ".json":
            out.setdefault(str(tid), []).append(p.stem)
    return {tid: sorted(set(keys)) for tid, keys in out.items()}


def block_compute_minutes(slot: Path) -> pd.Series:
    """Per task: the sum of the ``seconds_total`` of the blocks it registered, in minutes.

    A block's ``seconds_total`` is complete whichever attempt built it, so the
    sum is the task's compute time uncensored by a timeout; sacct's elapsed
    time is the last attempt's only. A task whose registered stats file is
    missing from disk is left out.
    """
    out: dict[str, float] = {}
    for tid, keys in registered_blocks(slot).items():
        secs = []
        for k in keys:
            p = slot / "stats" / f"{k}.json"
            if not p.exists():
                break
            secs.append(float(json.loads(p.read_text())["seconds_total"]))
        else:
            out[tid] = sum(secs) / 60.0
    return pd.Series(out, dtype=float, name="compute_min")


def slot_audits(slot: Path, workflows: Sequence[int] | None = None) -> pd.DataFrame:
    """Every audit row saved in the slot (``audit.csv`` and ``audit_<workflow>.csv``), one per task and workflow.

    The audit command overwrites ``audit.csv`` with the latest workflow and
    keeps a copy per workflow id, so a slot probed more than once holds all of
    them. *workflows* restricts the rows to those workflow ids.
    """
    # audit.csv and audit_<workflow>.csv only; the pipe-separated
    # audit_attempts_<workflow>.csv beside them are Slurm attempt tables.
    files = [slot / "audit.csv", *sorted(slot.glob("audit_[0-9]*.csv"))]
    frames = [pd.read_csv(p) for p in files if p.exists()]
    if not frames:
        return pd.DataFrame()
    audit = pd.concat(frames, ignore_index=True)
    if "workflow_id" in audit:
        audit = audit.drop_duplicates(subset=["workflow_id", "task_id"], keep="last")
        if workflows is not None:
            audit = audit[audit["workflow_id"].isin(list(workflows))]
    return audit.reset_index(drop=True)


def slot_training_frame(
    slot: Path, evidence: pd.DataFrame, *, workflows: Sequence[int] | None = None
) -> pd.DataFrame:
    """One row per completed, fully measured task of a probed slot: features recomputed, targets measured.

    Features come from ``task_features`` on the current *evidence* (not from
    the features saved at submission, which may predate the model) over the
    blocks the task registered in its sidecar (``registered_blocks``); the
    targets are the task's block compute time plus ``BUNDLE_OVERHEAD_MIN``
    (``elapsed_min``) and its REPORT peak (``max_rss_gib``). sacct's values
    stay as ``wall_min`` and ``sacct_rss_gib``, the asks it ran under as
    ``asked_mem_gib`` and ``asked_runtime_min``. Tasks without both measures
    are left out.
    """
    audit = slot_audits(slot, workflows)
    if audit.empty:
        return pd.DataFrame()
    done = audit[audit["state"].astype(str).str.startswith("COMPLETED")]
    keys_of = registered_blocks(slot)
    compute = block_compute_minutes(slot)
    peaks = report_peaks(slot / "logs")
    rows = []
    for row in done.itertuples(index=False):
        tid = str(row.task_id)
        if tid not in keys_of or tid not in compute.index or tid not in peaks.index:
            continue
        minutes = float(compute[tid]) + BUNDLE_OVERHEAD_MIN
        rows.append(
            {
                "task_id": tid,
                **task_features(keys_of[tid], evidence),
                "elapsed_min": minutes,
                "elapsed_seconds": minutes * 60.0,
                "max_rss_gib": float(peaks[tid]),
                "wall_min": float(row.elapsed_seconds) / 60.0,
                "sacct_rss_gib": float(row.max_rss_gib),
                "asked_mem_gib": float(getattr(row, "req_mem_gib", np.nan)),
                "asked_runtime_min": float(getattr(row, "timelimit_minutes", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def refit_table(
    frame: pd.DataFrame, runtime: ResourcePredictor, memory: ResourcePredictor
) -> pd.DataFrame:
    """Per task: its tier, the ask it ran under, what it used, the used/asked ratio, and the refit's ask."""
    return pd.DataFrame(
        {
            "task_id": frame["task_id"],
            PROBE_BY: frame.get(PROBE_BY),
            "asked_mem_gib": frame["asked_mem_gib"],
            "max_rss_gib": frame["max_rss_gib"],
            "mem_used_over_asked": frame["max_rss_gib"] / frame["asked_mem_gib"],
            "refit_mem_gib": memory.predict(frame).to_numpy(),
            "asked_runtime_min": frame["asked_runtime_min"],
            "elapsed_min": frame["elapsed_min"],
            "time_used_over_asked": frame["elapsed_min"] / frame["asked_runtime_min"],
            "refit_runtime_min": runtime.predict(frame).to_numpy(),
        }
    )


def tier_cost_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Observed memory and runtime per ``mem_tier`` (idd-tools ``cost_by_feature``)."""
    renamed = frame.rename(
        columns={"max_rss_gib": "jobmon_mem_gib", "elapsed_seconds": "jobmon_runtime_s"}
    )
    return cost_by_feature(renamed, by=PROBE_BY)


def registered_outputs(out_dir: Path) -> pd.Series:
    """Registered files per task, from the slot's ``.atomic`` sidecars folded by idd-tools.

    ``consolidate_atomic_registry`` writes ``<out_dir>/.atomic_registry.parquet``
    (one row per registered file); this counts rows per ``task_id``. Empty when
    no task has registered anything yet.
    """
    registry = atomic_registry(out_dir)
    if registry.empty:
        return pd.Series(dtype=int, name="n_files")
    return registry.groupby("task_id")["path"].count().rename("n_files")


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
