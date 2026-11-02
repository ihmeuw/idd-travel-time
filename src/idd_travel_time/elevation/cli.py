"""``fetch-fabdem``: mirror FABDEM V1-2 tiles through a jobmon fan-out.

Commands: ``zip`` (one archive, the worker), ``launch`` (probe or full
workflow), ``audit`` (asked-vs-actual from the run record), ``verify``
(completeness against the host's tile index).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pandas as pd
from idd_tools.jobmon.resources import minutes_to_jobmon

from idd_travel_time.elevation import fabdem as fb
from idd_travel_time.elevation import launch as ln
from idd_travel_time.paths import load_paths

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[3]


def default_fabdem_dir() -> Path:
    """The mirror root from paths.yaml (``fabdem_dir``)."""
    return Path(load_paths()["fabdem_dir"]).expanduser()


@click.group()
def main() -> None:
    """Mirror FABDEM V1-2 (bare-earth Copernicus GLO-30) from the Bristol data host."""


@main.command("zip")
@click.option(
    "--zip-names",
    required=True,
    help="Comma-separated archive names (one task may carry several).",
)
@click.option(
    "--fabdem-dir", type=click.Path(path_type=Path, file_okay=False), default=None
)
@click.option(
    "--workers",
    type=int,
    default=fb.WORKERS,
    show_default=True,
    help="Tiles in flight per archive.",
)
@click.option(
    "--read-mb",
    type=int,
    default=fb.READ_SIZE >> 20,
    show_default=True,
    help="Bytes per ranged request, MB.",
)
@click.option(
    "--task-id",
    default=None,
    help="jobmon task id; names the atomic sidecar. Default: derived from the archives.",
)
def zip_cmd(
    zip_names: str,
    fabdem_dir: Path | None,
    workers: int,
    read_mb: int,
    task_id: str | None,
) -> None:
    """Fetch every tile of the named archives into <fabdem-dir>/tiles (present tiles are skipped).

    Emits the idd-tools ``REPORT`` line (elapsed, peak RSS) that
    ``parse_probe_logs`` and the resource sizing read.
    """
    from idd_tools.jobmon import TaskProgressLogger
    from idd_tools.jobmon.atomic_io import AtomicRegistry

    root = fabdem_dir or default_fabdem_dir()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    names = [n for n in zip_names.split(",") if n]
    tag = "+".join(
        n.replace("_FABDEM_V1-2.zip", "").removesuffix(".zip") for n in names
    )
    log = TaskProgressLogger(tag=tag)
    log.entry(total=len(names))
    failed = 0

    def make_progress(
        started: float, got: dict[str, int]
    ) -> Callable[[fb.MemberOutcome, int, int], None]:
        def progress(o: fb.MemberOutcome, done: int, total: int) -> None:
            got["bytes"] += o.size if o.status == "fetched" else 0
            el = max(time.monotonic() - started, 1e-9)
            click.echo(
                f"[{done}/{total}] {o.status:7s} {o.name} {o.size / 1e6:6.1f} MB | "
                f"{got['bytes'] / 1e6:,.0f} MB in {el:,.0f} s = {got['bytes'] / 1e6 / el:.2f} MB/s"
                + (f" | {o.error}" if o.error else "")
            )

        return progress

    with AtomicRegistry(
        root, task_id=task_id or f"zip:{tag}", task_template=ln.TEMPLATE
    ) as registry:
        for zip_name in names:
            started = time.monotonic()
            progress = make_progress(started, {"bytes": 0})
            click.echo(
                f"{zip_name}: start {time.strftime('%Y-%m-%d %H:%M:%S')} workers={workers} read_mb={read_mb}"
            )
            outcomes = fb.fetch_zip(
                zip_name,
                root / "tiles",
                workers=workers,
                read_size=read_mb << 20,
                progress=progress,
                registry=registry,
            )
            record = fb.write_record(root / "records", zip_name, outcomes)
            counts = {
                s: sum(o.status == s for o in outcomes)
                for s in ("fetched", "present", "failed")
            }
            fetched_mb = sum(o.size for o in outcomes if o.status == "fetched") / 1e6
            click.echo(
                f"{zip_name}: {counts} {fetched_mb:,.0f} MB in {time.monotonic() - started:,.0f} s -> {record}"
            )
            for o in outcomes:
                if o.status == "failed":
                    click.echo(f"  FAILED {o.name}: {o.error}")
            log.subtask_done(zip_name, elapsed=time.monotonic() - started)
            failed += counts["failed"]
    log.report()
    if failed:
        msg = f"{failed} tile(s) failed"
        raise click.ClickException(msg)


@main.command("launch")
@click.option(
    "--jobmon-tag",
    "version_tag",
    required=True,
    help="Run tag in .jobmon_versions.toml.",
)
@click.option(
    "--memory-gib",
    type=float,
    default=1.0,
    show_default=True,
    help="Per-task memory ask (from the workers' REPORT peak RSS).",
)
@click.option("--cores", type=int, default=1, show_default=True)
@click.option(
    "--fabdem-dir", type=click.Path(path_type=Path, file_okay=False), default=None
)
@click.option(
    "--workers",
    type=int,
    default=fb.WORKERS,
    show_default=True,
    help="Tiles in flight per archive task.",
)
@click.option(
    "--read-mb",
    type=int,
    default=fb.READ_SIZE >> 20,
    show_default=True,
    help="Bytes per ranged request, MB.",
)
@click.option(
    "--target-gb",
    type=float,
    default=ln.TARGET_BYTES / 1e9,
    show_default=True,
    help="Bytes per task the packer aims for.",
)
@click.option(
    "--probe-n",
    type=int,
    default=2,
    show_default=True,
    help="Submit only this many tasks.",
)
@click.option(
    "--no-probe",
    is_flag=True,
    help="Submit every task (after a probe has been audited).",
)
@click.option(
    "--concurrency",
    type=int,
    default=16,
    show_default=True,
    help="Tasks at once; connections = concurrency x workers (64 tested).",
)
@click.option("--project", default="proj_rapidresponse", show_default=True)
@click.option("--queue", default="all.q", show_default=True)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the per-task asks and stop before submitting.",
)
def launch(  # noqa: PLR0913
    *,
    version_tag: str,
    memory_gib: float,
    cores: int,
    fabdem_dir: Path | None,
    workers: int,
    read_mb: int,
    target_gb: float,
    probe_n: int,
    no_probe: bool,
    concurrency: int,
    project: str,
    queue: str,
    dry_run: bool,
) -> None:
    """Submit the archives as a jobmon workflow, one task per bundle, each sized from measured runs.

    The runtime model is fitted on every completed task in this mirror's
    run records (elapsed against bytes fetched); the asks are printed
    before submission and checked against per-task ceilings.
    """
    from idd_tools.jobmon.resource_audit import audit_run, max_resource_check
    from idd_tools.jobmon.submit import (
        submit_with_manifest,
    )  # jobmon extra, cluster only
    from idd_tools.jobmon.version_tag import register_run

    root = fabdem_dir or default_fabdem_dir()
    index = fb.load_tile_index(root / fb.INDEX_NAME)
    sizes = fb.load_zip_sizes(root / fb.SIZES_NAME)
    missing = {n for n in index if n not in sizes}
    if missing:
        msg = f"{len(missing)} archives have no size in {fb.SIZES_NAME}; refusing to size blind"
        raise click.ClickException(msg)
    records = fb.read_records(root / "records")
    todo = ln.remaining_sizes(
        index, sizes, root / "tiles"
    )  # bytes still missing per archive; done ones drop out
    done = {n for n in sizes if n not in todo}
    bundles = ln.bundle_archives(todo, target_bytes=int(target_gb * 1e9))
    manifest = ln.zip_manifest(
        bundles,
        sizes,
        root,
        workflow_name=f"fabdem_{root.name}",
        workers=workers,
        read_mb=read_mb,
    )
    frame = ln.training_frame(
        ln.load_audits(root / "audits", audit_run(root, last_only=True)), records
    )
    if ln.model_usable(frame):
        asks = ln.runtime_asks(ln.fit_runtime_predictor(frame), manifest)
        basis = f"runtime model fitted on {len(frame)} completed tasks (elapsed vs MB fetched, asymmetric clip +25%)"
    else:
        rate = ln.measured_rate(frame)
        span = (
            f"{frame['fetched_mb'].min():,.0f}-{frame['fetched_mb'].max():,.0f} MB"
            if len(frame)
            else "none"
        )
        asks = ln.rate_asks(manifest, mb_per_s=rate)
        basis = (
            f"measured rate {rate:.2f} MB/s (median of {len(frame)} audited tasks spanning {span}; "
            f"a model needs >= {ln.MIN_TRAINING_TASKS} tasks spanning >= {ln.SPREAD_RATIO}x) "
            "x 1.5 margin + 9 s per archive"
        )
    s = pd.Series(asks)
    click.echo(
        f"{len(bundles)} tasks over {len(todo)} archives still to mirror ({len(done)} done; target {target_gb:.1f} GB/task); {basis}; "
        f"asks min {s.min():.0f} / median {s.median():.0f} / max {s.max():.0f} min, sum {s.sum() / 60:,.1f} task-hours; "
        f"memory {ln.memory_to_jobmon(memory_gib)} x {cores} core"
    )
    violations = max_resource_check(
        {
            ln.TEMPLATE: {
                "memory": ln.memory_to_jobmon(memory_gib),
                "runtime": minutes_to_jobmon(s.max()),
                "cores": cores,
            }
        },
        mem_max_gib=4.0,
        wall_max_minutes=90.0,
    )
    if violations:
        msg = f"asks exceed ceilings: {violations}"
        raise click.ClickException(msg)
    if dry_run:
        return
    (root / "logs").mkdir(parents=True, exist_ok=True)
    result = submit_with_manifest(
        manifest,
        output_dir=root,
        templates={ln.TEMPLATE: ln.template_spec()},
        resources=ln.resources_for(asks, memory_gib, cores),
        probe_only=None if no_probe else probe_n,
        concurrency_limit=concurrency,
        log_path_fn=lambda task: ln.log_paths(task, root),
        project=project,
        queue=queue,
    )
    register_run(
        root,
        version_tag,
        result.workflow_id,
        note=f"probe_n={None if no_probe else probe_n}; tasks={len(bundles)}; workers={workers}; read_mb={read_mb}; concurrency={concurrency}; memory_gib={memory_gib}",
        registry_path=REPO_ROOT / ".jobmon_versions.toml",
    )
    click.echo(
        f"workflow {result.workflow_id} status {result.status}: {result.n_tasks_submitted} tasks -> {root}\n"
        f"GUI {ln.GUI}{result.workflow_id}"
    )
    if result.status != "D":
        msg = f"workflow {result.workflow_id} ended with status {result.status}"
        raise click.ClickException(msg)


@main.command("audit")
@click.option(
    "--fabdem-dir", type=click.Path(path_type=Path, file_okay=False), default=None
)
@click.option(
    "--register-tag",
    default=None,
    help="Also append every completed task to the probe history under this tag.",
)
def audit(fabdem_dir: Path | None, register_tag: str | None) -> None:
    """Join the last run record to sacct; print elapsed, memory, allocation zones, and the workers' REPORT asks."""
    from idd_tools.jobmon.resource_audit import (
        audit_run,
        policy_metrics,
        sizing_floor_report,
    )
    from idd_tools.jobmon.resources import parse_probe_logs

    root = fabdem_dir or default_fabdem_dir()
    frame = audit_run(root, last_only=True)
    click.echo(json.dumps(ln.summarise_audit(frame), indent=1))
    click.echo(json.dumps(sizing_floor_report(frame), indent=1))
    metrics = policy_metrics(frame)
    click.echo("allocation zones (overall): " + metrics["overall"].to_dict().__repr__())
    try:
        click.echo(
            "worker REPORT lines (max elapsed, max peak RSS, x2): "
            + json.dumps(parse_probe_logs(root / "logs"))
        )
    except ValueError as exc:
        click.echo(f"no REPORT lines parsed: {exc}")
    out = root / "audit.csv"
    frame.to_csv(out, index=False)
    kept = ln.save_audit(frame, root / "audits")
    click.echo(
        f"full audit -> {out}; kept per workflow under {kept.parent} ({len(list(kept.parent.glob('*.csv')))} runs)"
    )
    if register_tag:
        from idd_tools.jobmon.run_registry import register_probe

        done = frame[frame["state"].astype(str).str.startswith("COMPLETED")]
        for _, r in done.iterrows():
            register_probe(
                root,
                register_tag,
                int(r["workflow_id"]),
                n_cells=int(str(r["n_archives"]))
                if "n_archives" in frame and str(r["n_archives"]).isdigit()
                else 1,
                shared_axes=[],
                elapsed_s=float(r["elapsed_seconds"]),
                max_rss_gib=float(r["max_rss_gib"]),
                note=str(r["task_id"]),
                registry_path=REPO_ROOT / ".jobmon_versions.toml",
            )
        click.echo(f"registered {len(done)} task measurements under {register_tag}")


@main.command("sizes")
@click.option(
    "--fabdem-dir", type=click.Path(path_type=Path, file_okay=False), default=None
)
def sizes(fabdem_dir: Path | None) -> None:
    """Re-measure archives recorded as 404 stubs under their host names (the index misnames 103)."""
    root = fabdem_dir or default_fabdem_dir()
    index = fb.load_tile_index(root / fb.INDEX_NAME)
    sizes, absent = fb.refresh_zip_sizes(root / fb.SIZES_NAME, index)
    total = sum(sizes.values()) / 1e9
    click.echo(
        f"{len(sizes)} archives sized ({total:,.1f} GB); {len(absent)} absent on the host: {absent}"
    )


@main.command("status")
@click.option(
    "--fabdem-dir", type=click.Path(path_type=Path, file_okay=False), default=None
)
def status(fabdem_dir: Path | None) -> None:
    """Progress of the mirror: tiles and bytes on disk, archives recorded, and each live task log's last line."""
    root = fabdem_dir or default_fabdem_dir()
    index = fb.load_tile_index(root / fb.INDEX_NAME)
    total = sum(len(v) for v in index.values())
    tiles = list((root / "tiles").glob("*.tif"))
    size = sum(p.stat().st_size for p in tiles)
    records = sorted((root / "records").glob("*.json"))
    click.echo(
        f"tiles {len(tiles):,} of {total:,} ({size / 1e9:,.1f} GB); archives recorded {len(records)} of {len(index)}"
    )
    partial = list((root / "tiles").glob("*.tif.tmp"))
    if partial:
        click.echo(f"{len(partial)} tile(s) in flight")
    stale = list((root / "tiles").glob("*.tmp-*"))
    if stale:
        click.echo(
            f"{len(stale)} stale temp file(s) from runs before the atomic registry (old naming); safe to delete"
        )
    logs = [p for p in (root / "logs").glob("*.out/*/*") if p.is_file()]
    for log in sorted(logs, key=lambda p: p.stat().st_mtime)[-8:]:
        lines = log.read_text().splitlines()
        if lines:
            click.echo(
                f"  {log.parents[1].name.replace('_FABDEM_V1-2.out', '')}: {lines[-1]}"
            )


@main.command("verify")
@click.option(
    "--fabdem-dir", type=click.Path(path_type=Path, file_okay=False), default=None
)
def verify(fabdem_dir: Path | None) -> None:
    """List tiles from the host's index that are missing or empty under <fabdem-dir>/tiles."""
    root = fabdem_dir or default_fabdem_dir()
    index = fb.load_tile_index(root / fb.INDEX_NAME)
    missing = fb.missing_tiles(index, root / "tiles")
    total = sum(len(v) for v in index.values())
    out = root / "missing.txt"
    out.write_text("\n".join(missing) + ("\n" if missing else ""))
    click.echo(
        f"{total - len(missing):,} of {total:,} tiles present; {len(missing):,} missing -> {out}"
    )
    if missing:
        msg = "mirror incomplete"
        raise click.ClickException(msg)


if __name__ == "__main__":
    main()
