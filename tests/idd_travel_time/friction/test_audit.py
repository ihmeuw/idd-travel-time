"""Resource usage against asks, assembled from a probed slot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idd_travel_time.friction import audit as au
from idd_travel_time.friction import launch as ln
from tests.idd_travel_time.friction.test_launch import _write_slot


def test_task_usage_joins_asks_peaks_and_compute(tmp_path: Path) -> None:
    slot, m, _ev = _write_slot(tmp_path)
    usage = au.task_usage(slot)
    assert list(usage.columns) == au.USAGE_COLS
    assert len(usage) == 2
    first = usage.set_index("task_id").loc[m.tasks[0].task_id]
    assert first["workflow"] == 7
    assert first["asked_gib"] == 24.0
    assert first["peak_gib"] == pytest.approx(4.0)  # the larger attempt's REPORT peak
    assert first["sacct_gib"] == 20.0
    assert first["wall_min"] == pytest.approx(10.0)
    assert first["compute_min"] == pytest.approx(len(ln.block_keys_of(m.tasks[0])))
    assert first["mem_used_over_asked"] == pytest.approx(4.0 / 24.0)
    assert first["time_used_over_asked"] == pytest.approx(10.0 / 15.0)
    second = usage.set_index("task_id").loc[m.tasks[1].task_id]
    assert np.isnan(second["compute_min"])  # one registered stats file missing
    assert au.task_usage(tmp_path / "empty").empty


def test_attempts_by_state_finds_the_header_wherever_it_is(tmp_path: Path) -> None:
    slot = tmp_path / "slot"
    slot.mkdir()
    (slot / "audit_attempts_7.csv").write_text(
        "1_1|friction_block-1|TIMEOUT\n2_1|friction_block-2|COMPLETED\n"
        "JobID|JobName|State\n3_1|other_job-1|COMPLETED\n"
    )
    (slot / "audit_attempts_8.csv").write_text(
        "JobID|JobName|State\n4_1|friction_block-1|COMPLETED\n"
    )
    table = au.attempts_by_state(slot)
    assert list(table.index) == [7, 8]
    assert table.loc[7, "attempts"] == 2  # the other job is not counted
    assert table.loc[7, "TIMEOUT"] == 1
    assert table.loc[8, "COMPLETED"] == 1
    assert table.loc[8, "TIMEOUT"] == 0
    assert au.attempts_by_state(tmp_path).empty


def _usage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "workflow": [1, 1, 2],
            "task_id": ["a", "b", "c"],
            "tier": ["low", "high", "low"],
            "n_blocks": [3, 3, 3],
            "state": ["COMPLETED"] * 3,
            "asked_gib": [10.0, 20.0, 10.0],
            "peak_gib": [5.0, 15.0, 8.0],
            "sacct_gib": [10.0, 20.0, 10.0],
            "asked_min": [20.0, 20.0, 20.0],
            "wall_min": [10.0, 15.0, 5.0],
            "compute_min": [9.0, 14.0, 4.0],
            "mem_used_over_asked": [0.5, 0.75, 0.8],
            "time_used_over_asked": [0.5, 0.75, 0.25],
        }
    )


def test_usage_summary_per_workflow() -> None:
    s = au.usage_summary(_usage())
    assert list(s.index) == [1, 2]
    assert s.loc[1, "tasks"] == 2
    assert s.loc[1, "mem_used_over_asked_mean"] == pytest.approx(0.625)
    assert s.loc[1, "time_max"] == pytest.approx(0.75)
    assert s.loc[1, "asked_hours"] == pytest.approx(40.0 / 60)
    assert s.loc[1, "used_hours"] == pytest.approx(25.0 / 60)
    assert s.loc[1, "asked_gib_hours"] == pytest.approx((10 * 20 + 20 * 20) / 60)
    assert s.loc[1, "peak_gib_hours"] == pytest.approx((5 * 10 + 15 * 15) / 60)


def test_usage_by_tier_groups_workflow_and_tier() -> None:
    t = au.usage_by_tier(_usage())
    assert t.loc[(1, "low"), "tasks"] == 1
    assert t.loc[(1, "high"), "mem_used_over_asked"] == pytest.approx(0.75)
    assert t.loc[(2, "low"), "time_used_over_asked"] == pytest.approx(0.25)


# ----------------------------------------------------------------------------
# Zones and plots
# ----------------------------------------------------------------------------


def test_zone_of_uses_idd_tools_thresholds() -> None:
    from idd_tools.jobmon.resource_audit import EfficiencyBins

    r = pd.Series([0.1, 0.5, 0.79, 0.8, 1.2, np.nan])
    z = au.zone_of(r)
    assert list(z[:5]) == ["over", "well", "well", "under", "under"]
    assert pd.isna(z.iloc[5])
    assert list(au.zone_of(r, EfficiencyBins(over_max=0.2, under_min=0.9))[:5]) == [
        "over",
        "well",
        "well",
        "well",
        "under",
    ]


def test_classify_and_zone_counts() -> None:
    usage = _usage()
    c = au.classify(usage)
    assert list(c["mem_zone"]) == ["well", "well", "under"]
    assert list(c["time_zone"]) == ["well", "well", "over"]
    assert list(c["zone"]) == [
        "well",
        "well",
        "under",
    ]  # under on either axis dominates
    counts = au.zone_counts(c)
    assert counts.loc[1, "tasks"] == 2
    assert counts.loc[1, "zone_well"] == 2
    assert counts.loc[2, "time_over"] == 1
    assert counts.loc[2, "mem_under"] == 1
    assert counts.loc[2, "zone_under"] == 1
    assert counts.loc[2, "zone_over"] == 0


def test_plots_return_figures_with_legends() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    c = au.classify(_usage())
    f1 = au.plot_usage(c)
    assert len(f1.axes) == 1
    assert f1.axes[0].get_legend() is not None
    f2 = au.plot_usage_by_tier(c)
    assert len(f2.axes) == 2  # the tiers present: low and high
    assert f2.axes[-1].get_legend() is not None
    f3 = au.plot_ratios(c)
    assert len(f3.axes) == 1
    assert f3.axes[0].get_xlim()[1] >= 1.1
    assert f3.axes[0].get_legend() is not None
    for f in (f1, f2, f3):
        mpl.pyplot.close(f)
