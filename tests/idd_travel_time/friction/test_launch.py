"""Launcher pieces that need no cluster: evidence, block models, predictors, packing, the probe, slot measures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from idd_travel_time.friction import launch as ln
from idd_travel_time.friction.volume import VOLUME_COLS


def _num(value: object) -> float:
    """A task feature as a number (FeatureValue also admits str, bool and None)."""
    assert isinstance(value, int | float)
    return float(value)


def _write_evidence(root: Path, n_blocks: int = 24, seed: int = 0) -> list[str]:
    """A fake completed run: per-block stats whose seconds grow with land cover and roads."""
    rng = np.random.default_rng(seed)
    keys = [f"B-{i:04d}X-0001Y" for i in range(n_blocks)]
    (root / "stats").mkdir(parents=True)
    for i, k in enumerate(keys):
        s: dict[str, Any] = {
            "block_key": k,
            "cells": 67108864,
            "seconds_total": 20.0 + 25.0 * i + rng.normal(0, 3),  # 20 s .. ~600 s
            "cells_land_cover": 1e6 * (i + 1),
            "cells_land_cover_overlap": 1e5 * (i + 1),
            "motorised_cells_road": 1e4 * (i + 1),
            "motorised_cells_water": 5e3 * (i % 5),
            "cells_elevation_missing": 1e3 * (i % 3),
        }
        (root / "stats" / f"{k}.json").write_text(json.dumps(s))
    return keys


def _index(key: str) -> int:
    return int(key[2:6])


def _latitudes(keys: list[str]) -> pd.Series:
    return pd.Series({k: 5.0 + 2.0 * _index(k) for k in keys}, dtype=float)


def _volumes(keys: list[str]) -> pd.DataFrame:
    """Row-group MB per layer, growing with the block index."""
    return pd.DataFrame(
        {
            "rg_mb_water": [3000.0 + 500.0 * _index(k) for k in keys],
            "rg_mb_roads": [1000.0 + 800.0 * _index(k) for k in keys],
            "rg_mb_land_cover": [100.0 + 10.0 * _index(k) for k in keys],
        },
        index=keys,
    )


def _peaks(keys: list[str], seed: int = 1) -> pd.Series:
    """Measured process peaks following the water volume, with noise."""
    rng = np.random.default_rng(seed)
    v = _volumes(keys)
    return pd.Series(
        {k: 2.0 + 0.0008 * v.loc[k, "rg_mb_water"] + rng.normal(0, 0.3) for k in keys}
    )


def _evidence(root: Path) -> tuple[list[str], pd.DataFrame]:
    keys = _write_evidence(root)
    ev = ln.block_evidence(
        root, latitudes=_latitudes(keys), peaks=_peaks(keys), volumes=_volumes(keys)
    )
    return keys, ev


def _training_frame(n: int = 12) -> pd.DataFrame:
    """A fake probed-slot frame: features from the block models, targets measured."""
    rows = []
    for i in range(n):
        tier = ["low", "mid", "high"][i % 3]
        pred_peak = {"low": 7.0, "mid": 10.0, "high": 13.0}[tier] + 0.1 * i
        pred_s = 500.0 + 10 * i
        minutes = pred_s / 60.0 * 1.3 + 0.5
        rows.append(
            {
                "task_id": f"blocks:B-{i:04d}X-0001Y+3",
                "n_blocks": 3,
                ln.PROBE_BY: tier,
                "pred_seconds": pred_s,
                "max_pred_peak_gib": pred_peak,
                "elapsed_min": minutes,
                "elapsed_seconds": minutes * 60.0,
                "max_rss_gib": 0.95 * pred_peak,
                "wall_min": minutes,
                "sacct_rss_gib": 1.35 * pred_peak + 2,
                "asked_mem_gib": 1.35 * pred_peak + 2,
                "asked_runtime_min": 15.0,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Evidence and the block models
# ----------------------------------------------------------------------------


def test_mem_tier_edges() -> None:
    assert ln.mem_tier(5.0) == "low"
    assert ln.mem_tier(10.0) == "low"
    assert ln.mem_tier(10.1) == "mid"
    assert ln.mem_tier(13.0) == "mid"
    assert ln.mem_tier(16.0) == "high"


def test_block_stats_reads_every_stat_col(tmp_path: Path) -> None:
    keys = _write_evidence(tmp_path)
    st = ln.block_stats(tmp_path)
    assert list(st.index) == keys
    assert set(ln.STAT_COLS) <= set(st.columns)
    with pytest.raises(FileNotFoundError):
        ln.block_stats(tmp_path / "empty")


def test_abs_latitudes_uses_the_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    class Grid:
        def __init__(self, key: str) -> None:
            self.key = key

        def lonlat_bounds(self) -> tuple[float, float, float, float]:
            i = _index(self.key)
            return (0.0, -10.0 * i - 1.0, 1.0, -10.0 * i + 1.0)

    monkeypatch.setattr(ln, "block_grid", lambda _idx, key, _res: Grid(key))
    lat = ln.abs_latitudes(["B-0001X-0000Y", "B-0003X-0000Y"], object(), 40.0)
    assert lat.to_dict() == {"B-0001X-0000Y": 10.0, "B-0003X-0000Y": 30.0}


def test_block_peaks_from_csv_and_stats(tmp_path: Path) -> None:
    slot = tmp_path / "slot"
    (slot / "stats").mkdir(parents=True)
    pd.DataFrame({"block_key": ["A", "B"], "peak_rss_gib": [5.0, 6.0]}).to_csv(
        slot / "block_peaks.csv", index=False
    )
    (slot / "stats" / "B.json").write_text(
        json.dumps({"block_key": "B", "peak_rss_gib": 7.5})
    )
    (slot / "stats" / "C.json").write_text(
        json.dumps({"block_key": "C"})
    )  # no peak recorded
    peaks = ln.block_peaks(slot)
    assert peaks.to_dict() == {"A": 5.0, "B": 7.5}  # the stats value wins for B
    assert ln.block_peaks(tmp_path / "nothing").empty


def test_block_evidence_predicts_seconds_and_peak_from_content(tmp_path: Path) -> None:
    keys, ev = _evidence(tmp_path)
    assert {
        "abs_lat",
        "pred_seconds",
        "pred_peak_gib",
        "mem_tier",
        *VOLUME_COLS,
    } <= set(ev.columns)
    # both grow with land cover in the fixture
    assert ev.loc[keys[-1], "pred_seconds"] > ev.loc[keys[0], "pred_seconds"]
    assert ev.loc[keys[-1], "pred_peak_gib"] > ev.loc[keys[0], "pred_peak_gib"]
    assert (ev["pred_seconds"] >= 1.0).all()
    assert ev["mem_tier"].isin(["low", "mid", "high"]).all()
    assert ev["mem_tier"].nunique() >= 2


def test_block_evidence_refuses_missing_inputs(tmp_path: Path) -> None:
    keys = _write_evidence(tmp_path)
    lat, peaks, vols = _latitudes(keys), _peaks(keys), _volumes(keys)
    with pytest.raises(ValueError, match="latitude missing"):
        ln.block_evidence(tmp_path, latitudes=lat.iloc[:-1], peaks=peaks, volumes=vols)
    with pytest.raises(ValueError, match="input volume missing"):
        ln.block_evidence(tmp_path, latitudes=lat, peaks=peaks, volumes=vols.iloc[:-1])
    with pytest.raises(ValueError, match="measured process peak"):
        ln.block_evidence(tmp_path, latitudes=lat, peaks=peaks.iloc[:5], volumes=vols)


def test_task_features_sums_seconds_takes_max_peak_and_heaviest_tier(
    tmp_path: Path,
) -> None:
    keys, ev = _evidence(tmp_path)
    f = ln.task_features(keys[:2], ev)
    assert f["n_blocks"] == 2
    assert f["pred_seconds"] == pytest.approx(
        float(ev.loc[keys[:2], "pred_seconds"].sum())
    )
    assert f["max_pred_peak_gib"] == pytest.approx(
        float(ev.loc[keys[:2], "pred_peak_gib"].max())
    )
    low = [k for k in keys if ev.loc[k, "mem_tier"] == "low"]
    heavy = [k for k in keys if ev.loc[k, "mem_tier"] != "low"]
    assert low
    assert heavy
    assert (
        ln.task_features([low[0], heavy[-1]], ev)[ln.PROBE_BY]
        == ev.loc[heavy[-1], "mem_tier"]
    )
    unknown = ln.task_features(["B-9999X-9999Y"], ev)
    assert unknown["pred_seconds"] == pytest.approx(float(ev["pred_seconds"].median()))
    assert unknown[ln.PROBE_BY] == "mid"


def test_fit_predictors_scale_with_work_and_need_ten_tasks() -> None:
    frame = _training_frame()
    runtime, memory = ln.fit_predictors(frame)
    small, big = frame.iloc[[0]], frame.iloc[[-1]]
    assert runtime.predict(big).iloc[0] > runtime.predict(small).iloc[0]
    assert memory.predict(big).iloc[0] > memory.predict(small).iloc[0]
    assert runtime.predict(small).iloc[0] >= ln.RUNTIME_FLOOR_MIN
    assert memory.predict(small).iloc[0] >= ln.MEMORY_FLOOR_GIB
    # the clip covers every training actual
    assert (runtime.predict(frame).to_numpy() >= frame["elapsed_min"].to_numpy()).all()
    assert (memory.predict(frame).to_numpy() >= frame["max_rss_gib"].to_numpy()).all()
    with pytest.raises(ValueError, match="need 10"):
        ln.fit_predictors(frame.iloc[:5])


# ----------------------------------------------------------------------------
# Packing and the manifest
# ----------------------------------------------------------------------------


def test_pack_blocks_covers_every_block_once_with_features(tmp_path: Path) -> None:
    keys, ev = _evidence(tmp_path)
    m = ln.pack_blocks(keys, ev, workflow_name="wf", target_min=20.0)
    packed = [k for t in m.tasks for k in ln.block_keys_of(t)]
    assert sorted(packed) == sorted(keys)
    assert len({t.task_id for t in m.tasks}) == len(m.tasks)
    assert all(t.task_template == ln.TEMPLATE for t in m.tasks)
    for t in m.tasks:
        assert t.task_features["n_blocks"] == len(ln.block_keys_of(t))
        assert _num(t.task_features["pred_seconds"]) > 0
        assert str(t.task_features[ln.PROBE_BY]) in ("low", "mid", "high")
    # 24 blocks at ~20..600 s packed to 20 min: more than one task, fewer than one per block
    assert 1 < len(m.tasks) < len(keys)
    assert max(_num(t.task_features["pred_seconds"]) for t in m.tasks) < 2 * 20.0 * 60.0


def test_reference_manifest_carries_only_path_id_and_slot(tmp_path: Path) -> None:
    keys, ev = _evidence(tmp_path)
    m = ln.pack_blocks(keys, ev, workflow_name="wf")
    path = tmp_path / ln.MANIFEST_NAME
    m.save(path)
    ref = ln.reference_manifest(m, path, tmp_path / "working")
    assert [t.task_id for t in ref.tasks] == [t.task_id for t in m.tasks]
    assert ref.tasks[0].task_args == {
        "manifest": str(path),
        "task_id": m.tasks[0].task_id,
        "out_dir": str(tmp_path / "working"),
    }
    assert ref.tasks[0].task_features == m.tasks[0].task_features
    from idd_tools.jobmon.manifest import TaskManifest

    loaded = TaskManifest.load(path)
    assert ln.block_keys_of(loaded.tasks[0]) == ln.block_keys_of(m.tasks[0])


def test_predicted_asks_one_row_per_task(tmp_path: Path) -> None:
    keys, ev = _evidence(tmp_path)
    m = ln.pack_blocks(keys, ev, workflow_name="wf", target_min=5.0)
    runtime, memory = ln.fit_predictors(_training_frame())
    asks = ln.predicted_asks(m, runtime, memory)
    assert len(asks) == len(m.tasks)
    assert (asks["runtime_min"] >= ln.RUNTIME_FLOOR_MIN).all()
    assert (asks["memory_gib"] >= ln.MEMORY_FLOOR_GIB).all()
    assert asks["memory_gib"].min() < asks["memory_gib"].max()  # not one static ask


def test_template_spec_uses_absolute_script_and_by_reference_args() -> None:
    spec = ln.template_spec(Path("/x/.venv/bin/build-friction"))
    assert spec.command_template.startswith("/x/.venv/bin/build-friction block ")
    assert "--manifest {manifest}" in spec.command_template
    assert "--task-id {task_id}" in spec.command_template
    assert "--out-dir {out_dir}" in spec.command_template
    assert spec.node_args == ["task_id"]
    assert sorted(spec.op_args) == ["manifest", "out_dir"]


def test_template_spec_default_is_the_venv_console_script() -> None:
    import sys

    spec = ln.template_spec()
    assert spec.command_template.startswith(f"{sys.prefix}/bin/build-friction block ")


def test_log_paths_are_concrete(tmp_path: Path) -> None:
    keys, ev = _evidence(tmp_path)
    m = ln.pack_blocks(keys, ev, workflow_name="wf")
    out, err = ln.log_paths(m.tasks[0], tmp_path)
    assert "%" not in out
    assert "%" not in err
    assert out.startswith(str(tmp_path / "logs"))
    assert out.endswith(".out")


def test_summarise_audit_reports_zones() -> None:
    audit = pd.DataFrame(
        {
            "elapsed_seconds": [100.0, 300.0, np.nan],
            "max_rss_gib": [1.0, 2.0, np.nan],
            "efficiency_zone_runtime": ["well", "over", "well"],
        }
    )
    s = ln.summarise_audit(audit)
    assert s["n"] == 3
    assert s["elapsed_s"] == {"median": 200.0, "max": 300.0}
    assert s["efficiency_zone_runtime"] == {"well": 2, "over": 1}


# ----------------------------------------------------------------------------
# The stratified probe and a probed slot's own measures
# ----------------------------------------------------------------------------


def test_stratified_probe_takes_n_per_tier_and_keeps_ids(tmp_path: Path) -> None:
    keys, ev = _evidence(tmp_path)
    m = ln.pack_blocks(keys, ev, workflow_name="wf", target_min=5.0)
    tiers = ln.tasks_per_tier(m)
    assert len(tiers) >= 2
    probe = ln.stratified_probe(m, 1)
    assert ln.tasks_per_tier(probe) == dict.fromkeys(tiers, 1)
    assert {t.task_id for t in probe.tasks} <= {t.task_id for t in m.tasks}
    assert [t.index for t in probe.tasks] == list(range(len(probe.tasks)))
    assert ln.stratified_probe(m, 1).tasks == probe.tasks  # seeded, so reproducible
    first_tier = str(m.tasks[0].task_features[ln.PROBE_BY])
    assert any(str(t.task_features[ln.PROBE_BY]) != first_tier for t in probe.tasks)


def _write_slot(root: Path) -> tuple[Path, Any, pd.DataFrame]:
    """A fake probed slot: packed manifest, stats for the first task's blocks, logs, audit.csv."""
    keys, ev = _evidence(root / "evidence")
    m = ln.pack_blocks(keys, ev, workflow_name="wf", target_min=5.0)
    slot = root / "slot"
    (slot / "stats").mkdir(parents=True)
    m.save(slot / ln.MANIFEST_NAME)
    from idd_tools.jobmon.atomic_io import AtomicRegistry, atomic_write

    first, second = m.tasks[0], m.tasks[1]
    with AtomicRegistry(slot, task_id=first.task_id, task_template=ln.TEMPLATE) as r:
        for k in ln.block_keys_of(first):
            path = slot / "stats" / f"{k}.json"
            payload = {"block_key": k, "seconds_total": 60.0, "peak_rss_gib": 6.5}
            atomic_write(path, lambda tmp, p=payload: tmp.write_text(json.dumps(p)))
            r.register(path)
    with AtomicRegistry(slot, task_id=second.task_id, task_template=ln.TEMPLATE) as r:
        for k in ln.block_keys_of(
            second
        ):  # registered; one stats file then goes missing
            path = slot / "stats" / f"{k}.json"
            payload = {"block_key": k, "seconds_total": 60.0}
            atomic_write(path, lambda tmp, p=payload: tmp.write_text(json.dumps(p)))
            r.register(path)
    (slot / "stats" / f"{ln.block_keys_of(second)[-1]}.json").unlink()
    for task, attempts in ((first, ("111", "222")), (second, ("333",))):
        for i, jobid in enumerate(attempts):
            d = Path(ln.log_paths(task, slot)[0]) / jobid
            d.mkdir(parents=True)
            peak = 3.0 + i  # the retry shows the larger peak
            (d / f"x.o{jobid}").write_text(
                f"[{task.task_id}] REPORT: total 10.0s, peak RSS {peak:.2f} GiB, 1 subtasks (avg 10.0s)\n"
                f"[{task.task_id}] REPORT: total 12.0s, peak RSS {peak - 1:.2f} GiB, 1 subtasks (avg 12.0s)\n"
            )
    rows = [
        {
            "task_id": t.task_id,
            "workflow_id": 7,
            "state": "COMPLETED",
            "task_features_json": json.dumps(t.task_features),
            "elapsed_seconds": 600.0,
            "max_rss_gib": 20.0,
            "req_mem_gib": 24.0,
            "timelimit_minutes": 15.0,
        }
        for t in (first, second)
    ]
    pd.DataFrame(rows).to_csv(slot / "audit.csv", index=False)
    pd.DataFrame(rows).to_csv(
        slot / "audit_7.csv", index=False
    )  # the per-workflow copy
    (slot / "audit_attempts_7.csv").write_text(
        "JobID|JobName|State\n1_1|friction_block-1|COMPLETED\n"
    )
    return slot, m, ev


def test_report_peaks_takes_the_max_per_task_over_attempts(tmp_path: Path) -> None:
    slot, m, _ev = _write_slot(tmp_path)
    peaks = ln.report_peaks(slot / "logs")
    assert peaks[m.tasks[0].task_id] == pytest.approx(
        4.0
    )  # second attempt's larger peak
    assert peaks[m.tasks[1].task_id] == pytest.approx(3.0)
    assert ln.report_peaks(tmp_path / "none").empty


def test_registered_blocks_come_from_the_sidecars(tmp_path: Path) -> None:
    slot, m, _ev = _write_slot(tmp_path)
    reg = ln.registered_blocks(slot)
    # a frozen (read-only) slot is read from the parquet the consolidation left
    slot.chmod(0o555)  # read-only, as a frozen snapshot is
    try:
        assert ln.registered_blocks(slot) == reg
    finally:
        slot.chmod(0o755)
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / ".atomic").mkdir()
    frozen.chmod(0o555)
    try:
        with pytest.raises(PermissionError, match="read-only"):
            ln.registered_blocks(frozen)
    finally:
        frozen.chmod(0o755)
    assert reg[m.tasks[0].task_id] == sorted(ln.block_keys_of(m.tasks[0]))
    assert reg[m.tasks[1].task_id] == sorted(ln.block_keys_of(m.tasks[1]))
    assert ln.registered_blocks(tmp_path / "none") == {}


def test_block_compute_minutes_needs_every_registered_stats_file(
    tmp_path: Path,
) -> None:
    slot, m, _ev = _write_slot(tmp_path)
    compute = ln.block_compute_minutes(slot)
    n = len(ln.block_keys_of(m.tasks[0]))
    assert compute[m.tasks[0].task_id] == pytest.approx(n * 60.0 / 60.0)
    assert m.tasks[1].task_id not in compute  # one registered stats file is missing


def test_block_peaks_of_a_slot_come_from_its_stats(tmp_path: Path) -> None:
    slot, m, _ev = _write_slot(tmp_path)
    peaks = ln.block_peaks(slot)
    assert set(peaks.index) == set(ln.block_keys_of(m.tasks[0]))
    assert (peaks == 6.5).all()


def test_slot_audits_dedupes_the_per_workflow_copies_and_filters(
    tmp_path: Path,
) -> None:
    slot, _m, _ev = _write_slot(tmp_path)
    audits = ln.slot_audits(slot)
    assert len(audits) == 2  # audit.csv and audit_7.csv hold the same two rows
    assert audits["workflow_id"].notna().all()  # the attempts table was not read
    assert len(ln.slot_audits(slot, workflows=[7])) == 2
    assert ln.slot_audits(slot, workflows=[8]).empty
    assert ln.slot_audits(tmp_path / "none").empty


def test_slot_training_frame_recomputes_features_and_uses_slot_measures(
    tmp_path: Path,
) -> None:
    slot, m, ev = _write_slot(tmp_path)
    frame = ln.slot_training_frame(slot, ev)
    assert list(frame["task_id"]) == [
        m.tasks[0].task_id
    ]  # the second task lacks a compute measure
    assert ln.slot_training_frame(slot, ev, workflows=[8]).empty
    row = frame.iloc[0]
    n = len(ln.block_keys_of(m.tasks[0]))
    assert row["elapsed_min"] == pytest.approx(n + ln.BUNDLE_OVERHEAD_MIN)
    assert row["max_rss_gib"] == pytest.approx(4.0)
    assert row["wall_min"] == pytest.approx(10.0)
    assert row["sacct_rss_gib"] == pytest.approx(20.0)
    assert row["asked_mem_gib"] == pytest.approx(24.0)
    expected = ln.task_features(ln.block_keys_of(m.tasks[0]), ev)
    assert row["pred_seconds"] == pytest.approx(expected["pred_seconds"])
    assert row["max_pred_peak_gib"] == pytest.approx(expected["max_pred_peak_gib"])
    assert row[ln.PROBE_BY] == expected[ln.PROBE_BY]


def test_refit_table_reports_used_over_asked() -> None:
    frame = _training_frame()
    runtime, memory = ln.fit_predictors(frame)
    table = ln.refit_table(frame, runtime, memory)
    assert len(table) == 12
    assert table["mem_used_over_asked"].between(0, 1).all()
    assert table["time_used_over_asked"].iloc[0] == pytest.approx(
        frame["elapsed_min"].iloc[0] / 15.0
    )
    assert (table["refit_mem_gib"] >= ln.MEMORY_FLOOR_GIB).all()


def test_tier_cost_curve_one_row_per_tier() -> None:
    curve = ln.tier_cost_curve(_training_frame()).set_index(ln.PROBE_BY)
    assert sorted(curve.index) == ["high", "low", "mid"]
    assert (curve["num_tasks"] == 4).all()
    assert curve.loc["high", "max_mem_gib"] > curve.loc["low", "max_mem_gib"]


def test_registered_outputs_counts_sidecar_files(tmp_path: Path) -> None:
    from idd_tools.jobmon.atomic_io import AtomicRegistry, atomic_write

    with AtomicRegistry(tmp_path, task_id="blocks:a+2", task_template=ln.TEMPLATE) as r:
        for name in ("x.tif", "y.tif", "z.json"):
            path = tmp_path / name
            atomic_write(path, lambda tmp: tmp.write_text("1"))
            r.register(path)
    assert ln.registered_outputs(tmp_path).to_dict() == {"blocks:a+2": 3}
    assert ln.registered_outputs(tmp_path / "nothing").empty
