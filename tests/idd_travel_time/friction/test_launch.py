"""Launcher pieces that need no cluster: evidence, predictors, packing, the by-reference manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from idd_travel_time.friction import launch as ln


def _num(value: object) -> float:
    """A task feature as a number (FeatureValue also admits str, bool and None)."""
    assert isinstance(value, int | float)
    return float(value)


def _write_evidence(root: Path, n_blocks: int = 24, seed: int = 0) -> list[str]:
    """A fake completed run: per-block stats plus a bundle-of-2 audit with linear costs."""
    rng = np.random.default_rng(seed)
    keys = [f"B-{i:04d}X-0001Y" for i in range(n_blocks)]
    (root / "stats").mkdir(parents=True)
    stats: dict[str, dict[str, Any]] = {}
    for i, k in enumerate(keys):
        s: dict[str, Any] = {
            "block_key": k,
            "cells": 67108864,
            "seconds_total": 20.0 + 25.0 * i,  # 20 s .. ~600 s
            "cells_land_cover": 1e6 * (i + 1),
            "motorised_cells_road": 1e4 * (i + 1),
            "motorised_cells_water": 5e3 * (i % 5),
        }
        stats[k] = s
        (root / "stats" / f"{k}.json").write_text(json.dumps(s))
    rows = []
    for j in range(0, n_blocks, 2):
        pair = keys[j : j + 2]
        est = sum(stats[k]["seconds_total"] for k in pair)
        rows.append(
            {
                "task_id": "block:" + "+".join(pair),
                "state": "COMPLETED",
                "task_args_json": json.dumps({"block_keys": ",".join(pair)}),
                "task_features_json": json.dumps({"n_blocks": 2}),
                "elapsed_seconds": 30.0 + 1.1 * est + rng.normal(0, 5),
                "max_rss_gib": 2.0 + 1.5 * (j // 2) + rng.normal(0, 0.2),
            }
        )
    rows.append({**rows[-1], "task_id": "block:failed", "state": "FAILED"})
    pd.DataFrame(rows).to_csv(root / "audit.csv", index=False)
    return keys


def test_mem_tier_edges() -> None:
    assert ln.mem_tier(1.0) == "low"
    assert ln.mem_tier(12.0) == "low"
    assert ln.mem_tier(12.1) == "mid"
    assert ln.mem_tier(24.0) == "mid"
    assert ln.mem_tier(39.0) == "high"


def test_block_evidence_joins_stats_and_task_memory(tmp_path: Path) -> None:
    keys = _write_evidence(tmp_path)
    ev = ln.block_evidence(tmp_path)
    assert list(ev.index) == keys
    assert set(ln.STAT_COLS) <= set(ev.columns)
    # both blocks of a bundle inherit the bundle's peak RSS, and the failed row is ignored
    assert ev.loc[keys[0], "task_max_rss_gib"] == ev.loc[keys[1], "task_max_rss_gib"]
    assert ev["mem_tier"].isin(["low", "mid", "high"]).all()


def test_task_features_sums_seconds_and_fills_unknown_blocks(tmp_path: Path) -> None:
    keys = _write_evidence(tmp_path)
    ev = ln.block_evidence(tmp_path)
    f = ln.task_features(keys[:2], ev)
    assert f["n_blocks"] == 2
    assert f["est_seconds"] == pytest.approx(20.0 + 45.0)
    assert f["max_cells_land_cover"] == pytest.approx(2e6)
    assert f["max_task_rss_gib"] == pytest.approx(
        float(ev.loc[keys[0], "task_max_rss_gib"])
    )
    unknown = ln.task_features(["B-9999X-9999Y"], ev)
    assert unknown["est_seconds"] == pytest.approx(float(ev["seconds_total"].median()))


def test_training_frame_and_predictors_scale_with_work(tmp_path: Path) -> None:
    _write_evidence(tmp_path)
    frame = ln.training_frame(tmp_path)
    assert len(frame) == 12  # the FAILED row is excluded
    runtime, memory = ln.fit_predictors(frame)
    small = pd.DataFrame(
        [ln.task_features(["B-0000X-0001Y"], ln.block_evidence(tmp_path))]
    )
    big = pd.DataFrame(
        [ln.task_features(["B-0023X-0001Y"], ln.block_evidence(tmp_path))]
    )
    assert runtime.predict(big).iloc[0] > runtime.predict(small).iloc[0]
    assert runtime.predict(small).iloc[0] >= ln.RUNTIME_FLOOR_MIN
    assert memory.predict(small).iloc[0] >= ln.MEMORY_FLOOR_GIB
    with pytest.raises(ValueError, match="audited tasks"):
        ln.fit_predictors(frame.iloc[:3])


def test_pack_blocks_covers_every_block_once_with_features(tmp_path: Path) -> None:
    keys = _write_evidence(tmp_path)
    ev = ln.block_evidence(tmp_path)
    m = ln.pack_blocks(keys, ev, workflow_name="wf", target_min=20.0)
    packed = [k for t in m.tasks for k in ln.block_keys_of(t)]
    assert sorted(packed) == sorted(keys)
    assert len({t.task_id for t in m.tasks}) == len(m.tasks)
    assert all(t.task_template == ln.TEMPLATE for t in m.tasks)
    for t in m.tasks:
        assert t.task_features["n_blocks"] == len(ln.block_keys_of(t))
        assert _num(t.task_features["est_seconds"]) > 0
    # 24 blocks at 20..600 s packed to 20 min: more than one task, fewer than one per block
    assert 1 < len(m.tasks) < len(keys)
    # no task is packed far past the target by the bin-median pricing
    assert max(_num(t.task_features["est_seconds"]) for t in m.tasks) < 2 * 20.0 * 60.0


def test_reference_manifest_carries_only_path_id_and_slot(tmp_path: Path) -> None:
    keys = _write_evidence(tmp_path)
    ev = ln.block_evidence(tmp_path)
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
    # the worker reads its bundle back from the saved manifest
    from idd_tools.jobmon.manifest import TaskManifest

    loaded = TaskManifest.load(path)
    assert ln.block_keys_of(loaded.tasks[0]) == ln.block_keys_of(m.tasks[0])


def test_predicted_asks_one_row_per_task(tmp_path: Path) -> None:
    keys = _write_evidence(tmp_path)
    ev = ln.block_evidence(tmp_path)
    m = ln.pack_blocks(keys, ev, workflow_name="wf")
    runtime, memory = ln.fit_predictors(ln.training_frame(tmp_path))
    asks = ln.predicted_asks(m, runtime, memory)
    assert len(asks) == len(m.tasks)
    assert (asks["runtime_min"] >= ln.RUNTIME_FLOOR_MIN).all()
    assert (asks["memory_gib"] >= ln.MEMORY_FLOOR_GIB).all()
    assert asks["runtime_min"].min() < asks["runtime_min"].max()  # not one static ask


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
    keys = _write_evidence(tmp_path)
    m = ln.pack_blocks(keys, ln.block_evidence(tmp_path), workflow_name="wf")
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
