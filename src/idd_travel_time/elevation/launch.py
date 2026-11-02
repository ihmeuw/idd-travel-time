"""jobmon fan-out of ``fetch-fabdem zip`` over the FABDEM archives, sized per task.

Archives are packed into tasks that clear the >=5 min sizing floor
(STANDARDS §Task bundling): big archives run alone, small ones share a task.
Each task's runtime ask comes from a :class:`ResourcePredictor` fitted on
measured runs (elapsed against bytes fetched) with the asymmetric clip
policy, so the ask tracks the archive size instead of one static number;
memory comes from the workers' own ``REPORT`` lines. The audit after a run
labels every task over / well / under so the sizing is checked, not assumed.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from idd_tools.jobmon.manifest import Task, TaskManifest
from idd_tools.jobmon.resources import (
    AsymmetricClipPolicy,
    LinearModel,
    ResourcePredictor,
    gib_to_jobmon,
    minutes_to_jobmon,
)
from idd_tools.jobmon.submit import TaskTemplateSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TEMPLATE = "fabdem_zip"
GUI = "https://jobmon-gui.cluster.ihme.washington.edu/#/workflow/"
TARGET_BYTES = (
    1_200_000_000  # ~8 min at the measured 2.7 MB/s: clears the 5 min floor with margin
)
RUNTIME_FLOOR_MIN = 6.0  # never ask below this: the floor plus bind overhead
MEASURED_MB_PER_S = 2.4  # per task, 4 tiles in flight with 16 MB reads (probe 643361); fallback when no model fits
MIN_TRAINING_TASKS = 3
SPREAD_RATIO = 1.5  # largest / smallest fetched_mb in training; below it the line has no slope information
MAX_ARCHIVES = 40  # per task: near-empty archives cost ~10 s each regardless of size
OVERHEAD_MIN = 0.15  # per archive: listing the remote directory and the first requests


def remaining_sizes(
    index: Mapping[str, Sequence[str]], sizes: Mapping[str, int], tiles_dir: Path
) -> dict[str, int]:
    """Bytes still to fetch per archive: its size scaled by the share of its tiles not yet on disk.

    A tile counts as present when it exists with a non-zero size and no
    ``.tmp`` companion (the idd-tools done convention). Archives with
    nothing left are omitted, so they get no task at all.
    """
    out: dict[str, int] = {}
    for name, tiles in index.items():
        if name not in sizes or not tiles:
            continue
        present = sum(
            1
            for t in tiles
            if (tiles_dir / t).exists()
            and (tiles_dir / t).stat().st_size > 0
            and not (tiles_dir / (t + ".tmp")).exists()
        )
        left = len(tiles) - present
        if left > 0:
            out[name] = max(1, round(sizes[name] * left / len(tiles)))
    return out


def bundle_archives(
    sizes: Mapping[str, int],
    *,
    target_bytes: int = TARGET_BYTES,
    max_archives: int = MAX_ARCHIVES,
) -> list[list[str]]:
    """Pack archives into tasks of about ``target_bytes``: largest first, each alone if it exceeds the target.

    Greedy and deterministic: archives in descending size; an archive that
    would push the open bundle past the target, or past ``max_archives``
    (per-archive overhead dominates for near-empty archives), closes it.
    Returns bundles in descending order of their total bytes, so a probe of
    the first N measures the heaviest tasks.
    """
    by_size = sorted(sizes, key=lambda n: sizes[n], reverse=True)
    bundles: list[list[str]] = []
    current: list[str] = []
    total = 0
    for name in by_size:
        if current and (
            total + sizes[name] > target_bytes or len(current) >= max_archives
        ):
            bundles.append(current)
            current, total = [], 0
        current.append(name)
        total += sizes[name]
    if current:
        bundles.append(current)
    return sorted(bundles, key=lambda b: sum(sizes[n] for n in b), reverse=True)


def bundle_task_id(bundle: Sequence[str]) -> str:
    """``zip:<label>`` for one archive, ``zip:<first label>+<n>`` for a bundle.

    The registry sidecar is ``.atomic/<task_id>.json``; an id built from all
    40 names of a bundle is ~660 bytes and fails NAME_MAX (255) on open,
    which is how workflow 647766 lost its three bundle tasks.
    """
    labels = [n.replace("_FABDEM_V1-2.zip", "").removesuffix(".zip") for n in bundle]
    if len(labels) == 1:
        return f"zip:{labels[0]}"
    return f"zip:{labels[0]}+{len(labels)}"


def zip_manifest(
    bundles: Sequence[Sequence[str]],
    sizes: Mapping[str, int],
    fabdem_dir: Path,
    workflow_name: str,
    *,
    workers: int = 4,
    read_mb: int = 16,
) -> TaskManifest:
    """One task per bundle; the archive names travel comma-joined, the bytes as a feature."""
    tasks = [
        Task(
            index=i,
            task_id=bundle_task_id(bundle),
            task_template=TEMPLATE,
            task_args={
                "zip_names": ",".join(bundle),
                "fabdem_dir": str(fabdem_dir),
                "workers": workers,
                "read_mb": read_mb,
                "task_id": bundle_task_id(bundle),
            },
            task_features={
                "n_archives": len(bundle),
                "total_mb": sum(sizes[n] for n in bundle) / 1e6,
            },
            shared_axes=[],
        )
        for i, bundle in enumerate(bundles)
    ]
    return TaskManifest(workflow_name=workflow_name, tasks=tasks)


def template_spec(script: Path | None = None) -> TaskTemplateSpec:
    """The worker command: the console script under ``.venv/bin`` by absolute path."""
    exe = (Path(sys.prefix) / "bin" / "fetch-fabdem") if script is None else script
    return TaskTemplateSpec(
        command_template=(
            f"{exe} zip --zip-names {{zip_names}} --fabdem-dir {{fabdem_dir}} "
            f"--workers {{workers}} --read-mb {{read_mb}} --task-id {{task_id}}"
        ),
        node_args=["zip_names"],
        task_args=["fabdem_dir", "workers", "read_mb", "task_id"],
    )


def log_paths(task: Task, fabdem_dir: Path) -> tuple[str, str]:
    """Concrete per-task stdout/stderr paths (jobmon rejects %-placeholders)."""
    name = str(task.task_id).replace("zip:", "")[:120]
    log_dir = fabdem_dir / "logs"
    return str(log_dir / f"{name}.out"), str(log_dir / f"{name}.err")


def save_audit(audit: pd.DataFrame, audits_dir: Path) -> Path:
    """Keep every run's audit as ``audits/<workflow_id>.csv`` so the runtime model can train on all of them."""
    audits_dir.mkdir(parents=True, exist_ok=True)
    wf = (
        str(audit["workflow_id"].dropna().iloc[0])
        if "workflow_id" in audit and audit["workflow_id"].notna().any()
        else "unknown"
    )
    out = audits_dir / f"{wf}.csv"
    audit.to_csv(out, index=False)
    return out


def load_audits(audits_dir: Path, latest: pd.DataFrame | None = None) -> pd.DataFrame:
    """All saved audits plus the latest frame, one row per (workflow, task), latest attempt kept."""
    frames = (
        [pd.read_csv(p) for p in sorted(audits_dir.glob("*.csv"))]
        if audits_dir.exists()
        else []
    )
    if latest is not None and len(latest):
        frames.append(latest)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keys = [k for k in ("workflow_id", "task_id") if k in df]
    return df.drop_duplicates(subset=keys, keep="last") if keys else df


def _fetched_mb(rec: Mapping[str, Any]) -> float:
    return sum(int(m["size"]) for m in rec["members"] if m["status"] == "fetched") / 1e6


def training_frame(
    audit: pd.DataFrame, records: Sequence[Mapping[str, Any]], *, slack_s: float = 120.0
) -> pd.DataFrame:
    """Completed tasks with elapsed minutes and the bytes they fetched, for the runtime fit.

    A task is joined to the attempt records of its own archives whose finish
    stamp falls inside the task's Slurm start-end window (plus ``slack_s``),
    so a later run that found everything present cannot overwrite the
    evidence. Tasks with no stamped record in their window are left out.
    """
    stamped = [r for r in records if r.get("finished")]
    rows = []
    for _, r in audit.iterrows():
        if not str(r.get("state", "")).startswith("COMPLETED") or pd.isna(
            r.get("elapsed_seconds")
        ):
            continue
        names = [
            n
            for n in str(r.get("zip_names") or r.get("zip_name") or "").split(",")
            if n
        ]
        try:
            t0 = datetime.fromisoformat(str(r["start"])) - timedelta(seconds=slack_s)
            t1 = datetime.fromisoformat(str(r["end"])) + timedelta(seconds=slack_s)
        except (KeyError, ValueError, TypeError):
            continue
        matched = [
            rec
            for rec in stamped
            if rec["zip"] in names
            and t0 <= datetime.fromisoformat(str(rec["finished"])) <= t1
        ]
        if not matched:
            continue
        rows.append(
            {
                "task_id": r["task_id"],
                "fetched_mb": sum(_fetched_mb(rec) for rec in matched),
                "elapsed_min": float(r["elapsed_seconds"]) / 60.0,
            }
        )
    return pd.DataFrame(rows)


def fit_runtime_predictor(
    frame: pd.DataFrame, *, buffer: float = 0.25
) -> ResourcePredictor:
    """Elapsed minutes against megabytes fetched: linear model, asymmetric clip, fit in real units.

    Real units rather than log space because the intercept (archive listing,
    bind) matters for small tasks and zero-fetch tasks are legitimate points.
    """
    if len(frame) < 2:  # noqa: PLR2004
        msg = (
            f"need at least 2 completed tasks to fit a runtime model, have {len(frame)}"
        )
        raise ValueError(msg)
    pred = ResourcePredictor(
        LinearModel(),
        AsymmetricClipPolicy(buffer=buffer, floor=RUNTIME_FLOOR_MIN),
        target_transform="identity",
    )
    return pred.fit(frame, feature_cols=["fetched_mb"], target_col="elapsed_min")


def runtime_asks(pred: ResourcePredictor, manifest: TaskManifest) -> dict[str, float]:
    """Predicted safe runtime in minutes per task, from its total megabytes."""
    frame = pd.DataFrame(
        {
            "task_id": [t.task_id for t in manifest.tasks],
            "fetched_mb": [
                float(str(t.task_features["total_mb"])) for t in manifest.tasks
            ],
        }
    )
    asks = pred.predict(frame)
    return {
        tid: float(max(RUNTIME_FLOOR_MIN, a))
        for tid, a in zip(frame["task_id"], asks, strict=True)
    }


def measured_rate(frame: pd.DataFrame) -> float:
    """Median MB/s over audited tasks; elapsed includes per-archive overhead, so the rate errs low."""
    if frame.empty:
        return MEASURED_MB_PER_S
    ok = frame[(frame["fetched_mb"] > 0) & (frame["elapsed_min"] > 0)]
    if ok.empty:
        return MEASURED_MB_PER_S
    return float((ok["fetched_mb"] / (ok["elapsed_min"] * 60.0)).median())


def model_usable(
    frame: pd.DataFrame,
    *,
    min_tasks: int = MIN_TRAINING_TASKS,
    spread_ratio: float = SPREAD_RATIO,
) -> bool:
    """A linear fit needs tasks of unequal size: same-size probes pin the clip at 3x the largest task."""
    if len(frame) < min_tasks:
        return False
    mb = frame["fetched_mb"]
    return float(mb.max()) / max(float(mb.min()), 1.0) >= spread_ratio


def rate_asks(
    manifest: TaskManifest,
    *,
    mb_per_s: float = MEASURED_MB_PER_S,
    margin: float = 1.5,
    overhead_min: float = OVERHEAD_MIN,
) -> dict[str, float]:
    """Runtime asks from the measured per-task rate alone, for a probe before any model can be fitted.

    minutes = total_mb / rate x margin + overhead per archive, floored at the sizing floor.
    """
    out = {}
    for t in manifest.tasks:
        mb = float(str(t.task_features["total_mb"]))
        n = int(str(t.task_features["n_archives"]))
        out[t.task_id] = max(
            RUNTIME_FLOOR_MIN, mb / mb_per_s / 60.0 * margin + overhead_min * n
        )
    return out


def memory_to_jobmon(gib: float) -> str:
    """Jobmon memory string: whole GiB at or above 1 GiB, whole MiB below it.

    ``gib_to_jobmon`` rounds everything up to 1G; this mirror needs ~0.5 GiB
    per task and Slurm accepts ``NNNM``.
    """
    if gib >= 1.0:
        return str(gib_to_jobmon(gib))  # str(): pre-commit's mypy has no idd-tools
    return f"{max(1, math.ceil(gib * 1024))}M"


def resources_for(asks: Mapping[str, float], memory_gib: float, cores: int) -> Any:
    """The per-task ``resources`` callable for ``submit_with_manifest``."""
    mem = memory_to_jobmon(memory_gib)

    def resources(task: Task) -> dict[str, str | int]:
        return {
            "memory": mem,
            "runtime": minutes_to_jobmon(asks[task.task_id]),
            "cores": cores,
        }

    return resources


def summarise_audit(audit: Any) -> dict[str, Any]:  # pandas frame from idd_tools
    """Elapsed and memory summary of an audit frame for sizing the next run."""
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
            out[col] = audit[col].value_counts(dropna=False).to_dict()
    return out
