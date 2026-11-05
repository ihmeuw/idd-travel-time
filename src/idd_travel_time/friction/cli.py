"""``build-friction``: build friction-surface blocks into the friction node's slot.

The friction node follows the idd-tools versioning contract (STANDARDS
§Output management; idd-tools docs/versioning.md). ``launch`` takes
``versions_options``: a probe writes to ``scratch/<label>/``; the real run
writes ``working/`` and a snapshot exists only through ``idd-versions freeze``
after looking, or on success when launched with ``--current``.

Worker and launcher follow the repo's jobmon contract (CLAUDE.md): the worker
runs inside ``AtomicRegistry`` with ``subtask_skip`` resume and ``REPORT``
lines; the launcher packs blocks to the sizing floor, prices every task from
measured evidence, runs ``max_resource_check``, probes, registers and prints
the GUI URL.
"""

from __future__ import annotations

import os

# Must precede any import that loads pyarrow: the variable selects the C++
# library's default memory pool at load time, which the parquet scanner uses.
# With jemalloc (arrow's default) a block build is charged ~45 GB to its
# cgroup on compute nodes; with the system allocator ~16 GB. Setting the pool
# from Python after import does not reach the scanner (probe 642719).
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import sys

# Line-buffer stdout: the launcher runs detached with stdout in a file, and
# idd-jobmon-launch --wait-for-bind reads the "Jobmon GUI:" line from that
# file; block buffering kept it invisible until exit (2026-10-22).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import click
from idd_tools.jobmon.atomic_io import AtomicRegistry
from idd_tools.jobmon.manifest import TaskManifest
from idd_tools.jobmon.task_logger import TaskProgressLogger
from idd_tools.versions import (
    VersionsOptions,
    finish_if_requested,
    resolve_target,
    versions_options,
)

from idd_travel_time.friction import launch as ln
from idd_travel_time.friction.build import (
    MODES,
    block_done,
    build_block,
    write_manifest,
)
from idd_travel_time.friction.frame import load_tile_index
from idd_travel_time.friction.inputs import load_input_paths
from idd_travel_time.friction.params import load_params
from idd_travel_time.friction.volume import volumes_for
from idd_travel_time.paths import load_paths

REPO_ROOT = Path(__file__).resolve().parents[3]


def friction_node() -> Path:
    """The friction node directory from ``paths.yaml`` (``friction_dir``)."""
    return Path(load_paths()["friction_dir"])


def _task_blocks(manifest_path: Path, task_id: str) -> list[str]:
    """The block keys of one task, read back from the saved manifest (bundle by reference)."""
    manifest = TaskManifest.load(manifest_path)
    task = next((t for t in manifest.tasks if t.task_id == task_id), None)
    if task is None:
        msg = f"task_id {task_id!r} not found in {manifest_path}"
        raise click.ClickException(msg)
    return ln.block_keys_of(task)


@click.group()
def main() -> None:
    """Build friction surfaces (minutes per metre) on the equal-area block frame."""


@main.command("block")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    required=True,
    help="The slot the launcher resolved (working/ or scratch/<label>/); workers never choose it.",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=None,
    help="Saved task manifest; with --task-id, the task's block bundle is read from it.",
)
@click.option(
    "--task-id", default=None, help="This task's id (from the jobmon template)."
)
@click.option(
    "--block-key",
    "block_keys",
    multiple=True,
    help="Block keys instead of a manifest, e.g. B-0072X-0019Y; repeat or comma-separate.",
)
@click.option("--resolution", type=float, default=40.0, show_default=True)
@click.option("--modes", default=",".join(MODES), show_default=True)
def block(
    out_dir: Path,
    manifest_path: Path | None,
    task_id: str | None,
    block_keys: tuple[str, ...],
    resolution: float,
    modes: str,
) -> None:
    """Build the blocks of one task into the slot (the jobmon worker command).

    A bundle runs each block in its own subprocess so the task's peak memory
    is one block's, not the sum; each subprocess emits its own REPORT line, so
    the task's peak RSS is the largest block's. Done blocks are skipped
    through ``subtask_skip`` on every output.
    """
    if manifest_path is not None:
        if task_id is None:
            msg = "--task-id is required with --manifest"
            raise click.ClickException(msg)
        keys = _task_blocks(manifest_path, task_id)
    else:
        keys = [k for item in block_keys for k in item.split(",") if k]
        if not keys:
            msg = "give --manifest with --task-id, or at least one --block-key"
            raise click.ClickException(msg)
        task_id = task_id or f"local:{keys[0]}"
    mode_tuple = tuple(modes.split(","))
    log = TaskProgressLogger(tag=task_id)
    done = [k for k in keys if block_done(out_dir, k, mode_tuple)]
    log.entry(total=len(keys), already_done=len(done))

    if len(keys) > 1:
        script = Path(sys.prefix) / "bin" / "build-friction"
        for key in keys:
            if key in done:
                continue
            started = time.monotonic()
            cmd = [
                str(script),
                "block",
                "--out-dir",
                str(out_dir),
                "--block-key",
                key,
                "--task-id",
                task_id,
                "--resolution",
                str(resolution),
                "--modes",
                modes,
            ]
            subprocess.run(cmd, check=True)  # noqa: S603  # fixed argv from our own options
            log.subtask_done(key, elapsed=time.monotonic() - started)
        log.report()
        return

    key = keys[0]
    started = time.monotonic()
    with AtomicRegistry(
        out_dir, task_id=task_id, task_template=ln.TEMPLATE
    ) as registry:
        result = build_block(
            key,
            load_params(),
            load_input_paths(load_paths()),
            out_dir,
            resolution_m=resolution,
            modes=mode_tuple,
            registry=registry,
        )
    log.subtask_done(key, elapsed=time.monotonic() - started)
    if result.skipped:
        click.echo(f"{key}: already complete, skipped")
    else:
        click.echo(
            f"{key}: {result.stats['seconds_total']} s; elevation tiles {result.stats.get('elevation_source_tiles')}; "
            + ", ".join(f"{m}={p}" for m, p in result.outputs.items())
        )
    log.report()


@main.command("launch")
@click.option(
    "--jobmon-tag",
    "version_tag",
    required=True,
    help="Run tag in .jobmon_versions.toml (comparable-run key for resource sizing).",
)
@click.option("--cores", type=int, default=2, show_default=True)
@click.option("--resolution", type=float, default=40.0, show_default=True)
@click.option(
    "--blocks-file",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=None,
    help="Block keys, one per line. Default: every land block of the frame, listed into the slot.",
)
@click.option(
    "--evidence-dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=None,
    help="A completed run (audit.csv + stats/) to size from. Default: the node's current snapshot.",
)
@click.option(
    "--refit-from",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    required=True,
    help=(
        "A probed slot (audit.csv, task_manifest.json, stats/, logs/, block_peaks.csv): its "
        "measured block peaks fit the block memory model, and its per-task compute times and "
        "REPORT peaks calibrate the asks; --evidence-dir supplies the per-block content features."
    ),
)
@click.option(
    "--refit-workflow",
    "refit_workflows",
    type=int,
    multiple=True,
    help="Calibrate the asks on these workflow ids of the probed slot only (repeatable). Default: all its audits.",
)
@click.option(
    "--target-min",
    type=float,
    default=ln.TARGET_MIN,
    show_default=True,
    help="Packed wall time per task, from the evidence's per-block seconds.",
)
@click.option(
    "--probe-n",
    type=int,
    default=2,
    show_default=True,
    help="Probe: this many tasks from every memory tier (stratified on mem_tier), with --scratch LABEL.",
)
@click.option(
    "--probe-seed",
    type=int,
    default=ln.PROBE_SEED,
    show_default=True,
    help="Seed for sampling the probe tasks within each tier.",
)
@click.option(
    "--no-probe",
    is_flag=True,
    help="Submit every task (after a probe has been audited).",
)
@click.option("--concurrency", type=int, default=200, show_default=True)
@click.option("--project", default="proj_rapidresponse", show_default=True)
@click.option("--queue", default="all.q", show_default=True)
@click.option("--dry-run", is_flag=True, help="Pack, size and report; submit nothing.")
@versions_options
def launch(  # noqa: PLR0913
    *,
    version_tag: str,
    cores: int,
    resolution: float,
    blocks_file: Path | None,
    evidence_dir: Path | None,
    refit_from: Path,
    refit_workflows: tuple[int, ...],
    target_min: float,
    probe_n: int,
    probe_seed: int,
    no_probe: bool,
    concurrency: int,
    project: str,
    queue: str,
    dry_run: bool,
    versions: VersionsOptions,
) -> None:
    """Submit the block builds as a jobmon workflow into the node's slot.

    ``--scratch LABEL`` for a probe; nothing for ``working/``; ``--current
    --description "..."`` to freeze and promote on success. The slot is never
    cleaned here: every task writes only its own blocks and skips done ones.
    """
    from idd_tools.jobmon.resource_audit import max_resource_check
    from idd_tools.jobmon.submit import (
        submit_with_manifest,  # jobmon extra only on the cluster
    )
    from idd_tools.jobmon.version_tag import register_run

    node = friction_node()
    target = resolve_target(node, versions, clean=False)
    params = load_params()
    inputs = load_input_paths(load_paths())
    write_manifest(
        target,
        params,
        inputs,
        resolution_m=resolution,
        modes=MODES,
        label=target.name,
        repo_root=REPO_ROOT,
    )
    if blocks_file is None:
        keys = ln.land_blocks(inputs, resolution)
        blocks_file = target / "blocks.txt"
        blocks_file.write_text("\n".join(keys) + "\n")
    keys = [k.strip() for k in blocks_file.read_text().splitlines() if k.strip()]

    evidence_dir = evidence_dir or (node / "current").resolve()
    evidence = _evidence(evidence_dir, refit_from, inputs, resolution)
    frame = ln.slot_training_frame(
        refit_from, evidence, workflows=refit_workflows or None
    )
    sized_from = (
        f"block models on {len(evidence)} blocks of {evidence_dir} with "
        f"{len(ln.block_peaks(refit_from))} measured peaks; asks calibrated on "
        f"{len(frame)} probed tasks of {refit_from}"
        + (f" (workflows {list(refit_workflows)})" if refit_workflows else "")
    )
    runtime, memory = ln.fit_predictors(frame)
    manifest = ln.pack_blocks(
        keys,
        evidence,
        workflow_name=f"friction_{node.name}_{target.name}",
        target_min=target_min,
    )
    # A stamped file per launch: a running workflow's workers read the path they
    # were given, so a later launch into the same slot must not overwrite it.
    manifest_path = target / f"task_manifest_{time.strftime('%Y%m%dT%H%M%S')}.json"
    manifest.save(manifest_path)
    manifest.save(target / ln.MANIFEST_NAME)  # the latest packing, for a reader
    asks = ln.predicted_asks(manifest, runtime, memory)
    tiers = evidence.reindex(keys)["mem_tier"].fillna("mid").value_counts().to_dict()
    click.echo(
        f"{len(manifest.tasks)} tasks over {len(keys)} blocks (tiers {tiers}; target {target_min:g} min/task); "
        f"sized from {sized_from}: "
        f"runtime min {asks['runtime_min'].min():.0f} / median {asks['runtime_min'].median():.0f} / "
        f"max {asks['runtime_min'].max():.0f} min, sum {asks['runtime_min'].sum() / 60:,.1f} task-hours; "
        f"memory min {asks['memory_gib'].min():.1f} / median {asks['memory_gib'].median():.1f} / "
        f"max {asks['memory_gib'].max():.1f} GiB; {cores} cores"
    )
    violations = max_resource_check(
        {
            ln.TEMPLATE: {
                "memory": f"{asks['memory_gib'].max():.0f}G",
                "runtime": f"{asks['runtime_min'].max():.0f}m",
                "cores": cores,
            }
        },
        mem_max_gib=ln.MEM_MAX_GIB,
        wall_max_minutes=ln.WALL_MAX_MIN,
    )
    if violations:
        msg = f"resource ceilings exceeded: {violations}"
        raise click.ClickException(msg)
    submitted = (
        manifest
        if no_probe
        else ln.stratified_probe(manifest, probe_n, seed=probe_seed)
    )
    if not no_probe:
        probe_asks = asks[asks["task_id"].isin([t.task_id for t in submitted.tasks])]
        click.echo(
            f"probe: {len(submitted.tasks)} tasks, {probe_n} per memory tier "
            f"{ln.tasks_per_tier(submitted)} (seed {probe_seed}); asks "
            f"{probe_asks['runtime_min'].min():.0f}-{probe_asks['runtime_min'].max():.0f} min, "
            f"{probe_asks['memory_gib'].min():.1f}-{probe_asks['memory_gib'].max():.1f} GiB"
        )
        for row in probe_asks.itertuples(index=False):
            click.echo(
                f"  {row.task_id}: {row.n_blocks} blocks, {row.runtime_min:.0f} min, {row.memory_gib:.1f} GiB"
            )
    if dry_run:
        click.echo("dry run: nothing submitted")
        return

    submit = ln.reference_manifest(submitted, manifest_path, target)
    result = submit_with_manifest(
        submit,
        output_dir=target,
        templates={ln.TEMPLATE: ln.template_spec()},
        resources=ln.resources_callable(manifest, runtime, memory, cores=cores),
        # The probe is the stratified subset built above (one sample per memory
        # tier), not idd-tools' flat first-N slice, which would take the
        # lightest tier only given pack_blocks' ordering.
        probe_only=None,
        concurrency_limit=concurrency,
        log_path_fn=lambda task: ln.log_paths(task, target),
        project=project,
        queue=queue,
        # Default scale-up on retry stays: a block whose prior peak sat at the old 40G cap
        # may need more memory than its ask, and only a scaled retry recovers that.
    )
    register_run(
        target,
        version_tag,
        result.workflow_id,
        note=(
            f"probe={'none' if no_probe else f'{probe_n} per tier, seed {probe_seed}'}; "
            f"slot={target.name}; evidence={evidence_dir}; refit={refit_from}"
        ),
        registry_path=REPO_ROOT / ".jobmon_versions.toml",
    )
    click.echo(
        f"workflow {result.workflow_id} status {result.status}: {result.n_tasks_submitted} tasks -> {target}"
    )
    click.echo("GUI " + ln.GUI.format(workflow_id=result.workflow_id))
    if result.status != "D":
        msg = f"workflow {result.workflow_id} ended with status {result.status}; nothing frozen"
        raise click.ClickException(msg)
    finish_if_requested(node, versions)


def _evidence(
    evidence_dir: Path, slot: Path, inputs: Any, resolution: float
) -> Any:  # pandas frame
    """Block evidence: content stats of *evidence_dir*, latitudes from the frame, peaks measured in *slot*."""
    tile_index = load_tile_index(inputs.tile_index_root, int(resolution))
    keys = ln.block_stats(evidence_dir).index.tolist()
    return ln.block_evidence(
        evidence_dir,
        latitudes=ln.abs_latitudes(keys, tile_index, resolution),
        peaks=ln.block_peaks(slot),
        volumes=volumes_for(keys, inputs, tile_index, resolution),
    )


@main.command("audit")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    required=True,
)
@click.option(
    "--register-tag",
    default=None,
    help="Register every completed task as a probe measurement under this tag.",
)
@click.option(
    "--evidence-dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=None,
    help="The run whose block stats supply the content features. Default: the node's current snapshot.",
)
@click.option("--resolution", type=float, default=40.0, show_default=True)
def audit(
    out_dir: Path,
    register_tag: str | None,
    evidence_dir: Path | None,
    resolution: float,
) -> None:
    """Join the slot's last run record to sacct: zones, sizing floor, REPORT asks, probe registration."""
    from idd_tools.jobmon.resource_audit import (
        audit_run,
        policy_metrics,
        sizing_floor_report,
    )
    from idd_tools.jobmon.resources import parse_probe_logs

    frame = audit_run(out_dir, last_only=True)
    click.echo(json.dumps(ln.summarise_audit(frame), indent=1))
    click.echo(json.dumps(sizing_floor_report(frame), indent=1))
    overall = policy_metrics(frame)["overall"]
    click.echo(f"allocation zones (overall): {overall.to_dict()}")
    logs = out_dir / "logs"
    if logs.exists():
        click.echo(
            f"worker REPORT lines (max elapsed, max peak RSS, x2): {parse_probe_logs(logs)}"
        )
    out = out_dir / "audit.csv"
    frame.to_csv(out, index=False)
    wf = (
        frame["workflow_id"].dropna().unique().tolist()
        if "workflow_id" in frame
        else []
    )
    if len(wf) == 1:
        frame.to_csv(
            out_dir / f"audit_{int(wf[0])}.csv", index=False
        )  # kept per workflow
    click.echo(f"full audit -> {out}")
    evidence_dir = evidence_dir or (friction_node() / "current").resolve()
    evidence = _evidence(
        evidence_dir, out_dir, load_input_paths(load_paths()), resolution
    )
    training = ln.slot_training_frame(out_dir, evidence)
    if len(training) and ln.PROBE_BY in training:
        click.echo(
            "cost by memory tier:\n"
            + ln.tier_cost_curve(training).to_string(index=False)
        )
    if len(training) >= ln.MIN_TRAINING_TASKS:
        refit_runtime, refit_memory = ln.fit_predictors(training)
        click.echo(
            "refit on this slot's measures (features recomputed from block content; elapsed = block "
            "compute + overhead, max_rss = REPORT peak; asked = the ask it ran under; refit = what a "
            "fit on this slot would ask):\n"
            + ln.refit_table(training, refit_runtime, refit_memory)
            .round(1)
            .to_string(index=False)
        )
    else:
        click.echo(
            f"refit skipped: {len(training)} completed tasks, need {ln.MIN_TRAINING_TASKS}"
        )
    registered = ln.registered_outputs(out_dir)
    click.echo(
        f"registered outputs per task (.atomic sidecars, {int(registered.sum())} files): "
        f"{registered.to_dict()}"
    )
    if register_tag:
        from idd_tools.jobmon.run_registry import register_probe

        done = frame[frame["state"].astype(str).str.startswith("COMPLETED")]
        n = 0
        for row in done.itertuples(index=False):
            feats = (
                json.loads(str(row.task_features_json))
                if row.task_features_json
                else {}
            )
            tier = feats.get(ln.PROBE_BY)
            register_probe(
                out_dir,
                register_tag,
                int(row.workflow_id),
                n_cells=int(feats.get("n_blocks", 1)),
                # the tier travels with the measurement so the history can be fitted per tier
                shared_axes=[ln.PROBE_BY] if tier is not None else [],
                axis_values={ln.PROBE_BY: str(tier)} if tier is not None else None,
                elapsed_s=float(row.elapsed_seconds),
                max_rss_gib=float(row.max_rss_gib),
                note=str(row.task_id),
                registry_path=REPO_ROOT / ".jobmon_versions.toml",
            )
            n += 1
        click.echo(f"registered {n} task measurements under {register_tag}")


@main.command("verify")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    required=True,
)
def verify(out_dir: Path) -> None:
    """Every block in the slot's blocks.txt has both rasters and its stats json; list the rest."""
    keys = [
        k.strip()
        for k in (out_dir / "blocks.txt").read_text().splitlines()
        if k.strip()
    ]
    missing = [k for k in keys if not block_done(out_dir, k, MODES)]
    (out_dir / "missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    click.echo(
        f"{len(keys) - len(missing)} of {len(keys)} blocks complete; {len(missing)} missing -> {out_dir / 'missing.txt'}"
    )


if __name__ == "__main__":
    main()
