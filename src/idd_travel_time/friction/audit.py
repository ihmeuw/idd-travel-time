"""Resource usage against asks for the jobmon runs of a slot: per task, per workflow, per tier.

The measures a sizing audit needs and idd-tools does not yet assemble (inbox
note 2026-11-05): a task's memory *use* is the largest process peak of its
block children (REPORT lines over every attempt), not sacct's MaxRSS, which
is the cgroup peak and fills with file cache to the ask; its runtime use is
the last attempt's wall time and, uncensored by a timeout, the build time of
the blocks it registered. Everything here reads what ``build-friction
audit`` and the launcher leave in the slot: ``audit*.csv``, ``logs/``,
``stats/``, ``.atomic/`` sidecars and ``audit_attempts_<workflow>.csv``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from idd_tools.jobmon.resource_audit import EfficiencyBins

from idd_travel_time.friction import launch as ln

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

USAGE_COLS = [
    "workflow",
    "task_id",
    "tier",
    "n_blocks",
    "state",
    "asked_gib",
    "peak_gib",
    "sacct_gib",
    "asked_min",
    "wall_min",
    "compute_min",
    "mem_used_over_asked",
    "time_used_over_asked",
]


def task_usage(slot: Path) -> pd.DataFrame:
    """One row per task and workflow audited in *slot*: asks, measured use, and used/asked ratios.

    Memory use is the task's REPORT peak (``launch.report_peaks``), runtime
    use its wall time (sacct) and its block compute time
    (``launch.block_compute_minutes``). Tasks without a REPORT peak keep NaN.
    """
    audits = ln.slot_audits(slot)
    if audits.empty:
        return pd.DataFrame(columns=USAGE_COLS)
    peaks = ln.report_peaks(slot / "logs")
    compute = ln.block_compute_minutes(slot)
    rows = []
    for r in audits.itertuples(index=False):
        raw = r.task_features_json
        feats = json.loads(raw) if isinstance(raw, str) and raw else {}
        tid = str(r.task_id)
        rows.append(
            {
                "workflow": int(r.workflow_id),
                "task_id": tid,
                "tier": feats.get(ln.PROBE_BY, "?"),
                "n_blocks": feats.get("n_blocks"),
                "state": r.state,
                "asked_gib": float(r.req_mem_gib),
                "peak_gib": float(peaks.get(tid, np.nan)),
                "sacct_gib": float(r.max_rss_gib),
                "asked_min": float(r.timelimit_minutes),
                "wall_min": float(r.elapsed_seconds) / 60.0,
                "compute_min": float(compute.get(tid, np.nan)),
            }
        )
    df = pd.DataFrame(rows)
    df["mem_used_over_asked"] = df["peak_gib"] / df["asked_gib"]
    df["time_used_over_asked"] = df["wall_min"] / df["asked_min"]
    return df[USAGE_COLS]


def attempts_by_state(slot: Path, task_prefix: str = ln.TEMPLATE) -> pd.DataFrame:
    """Every Slurm attempt saved beside the audits, counted by state per workflow.

    Reads ``audit_attempts_<workflow>.csv`` (``sacct --parsable2`` rows);
    the header line is located by name because some of these files were
    sorted after sacct wrote them. Only jobs whose name starts with
    *task_prefix* count.
    """
    out = []
    for path in sorted(slot.glob("audit_attempts_*.csv")):
        wf = int(path.stem.rsplit("_", 1)[1])
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        header = next((line for line in lines if line.startswith("JobID")), None)
        if header is None:
            continue
        rows = [line.split("|") for line in lines if line != header]
        table = pd.DataFrame(rows, columns=header.split("|"))
        table = table[table["JobName"].astype(str).str.startswith(task_prefix)]
        counts = table["State"].value_counts()
        out.append(
            {"workflow": wf, "attempts": len(table)}
            | {str(s): int(n) for s, n in counts.items()}
        )
    if not out:
        return pd.DataFrame(columns=["attempts"]).rename_axis("workflow")
    return pd.DataFrame(out).fillna(0).astype(int).set_index("workflow").sort_index()


def usage_summary(usage: pd.DataFrame) -> pd.DataFrame:
    """Per workflow: mean, min and max used/asked on both axes, and asked versus used hours.

    ``asked_hours`` sums the limits, ``used_hours`` the wall times;
    ``asked_gib_hours`` and ``peak_gib_hours`` weight memory by time.
    """
    g = usage.groupby("workflow")
    return pd.DataFrame(
        {
            "tasks": g.size(),
            "mem_used_over_asked_mean": g["mem_used_over_asked"].mean(),
            "mem_min": g["mem_used_over_asked"].min(),
            "mem_max": g["mem_used_over_asked"].max(),
            "time_used_over_asked_mean": g["time_used_over_asked"].mean(),
            "time_min": g["time_used_over_asked"].min(),
            "time_max": g["time_used_over_asked"].max(),
            "asked_hours": g["asked_min"].sum() / 60.0,
            "used_hours": g["wall_min"].sum() / 60.0,
            "asked_gib_hours": (usage["asked_gib"] * usage["asked_min"])
            .groupby(usage["workflow"])
            .sum()
            / 60.0,
            "peak_gib_hours": (usage["peak_gib"] * usage["wall_min"])
            .groupby(usage["workflow"])
            .sum()
            / 60.0,
        }
    )


def usage_by_tier(usage: pd.DataFrame) -> pd.DataFrame:
    """Per workflow and memory tier: mean asks, mean use and mean used/asked on both axes."""
    return usage.groupby(["workflow", "tier"]).agg(
        tasks=("task_id", "size"),
        asked_gib=("asked_gib", "mean"),
        peak_gib=("peak_gib", "mean"),
        mem_used_over_asked=("mem_used_over_asked", "mean"),
        asked_min=("asked_min", "mean"),
        wall_min=("wall_min", "mean"),
        compute_min=("compute_min", "mean"),
        time_used_over_asked=("time_used_over_asked", "mean"),
    )


# ----------------------------------------------------------------------------
# Zones: over / well / under on the measured ratios, and the three plots
# ----------------------------------------------------------------------------

ZONES = ("over", "well", "under")
# Status colours of the dataviz reference palette (good / warning / critical);
# a zone is never colour alone: each has a marker and a label.
ZONE_COLOR = {"well": "#0ca30c", "over": "#fab219", "under": "#d03b3b"}
ZONE_MARKER = {"well": "o", "over": "v", "under": "^"}
# Categorical slots 1-3 of the same palette, in their fixed order.
TIER_COLOR = {"low": "#2a78d6", "mid": "#eb6834", "high": "#1baf7a"}
TIER_MARKER = {"low": "o", "mid": "s", "high": "D"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#6b6a63", "#e1e0d9"
MARKER_PT2 = 48  # about an 8 px dot
TIERS = tuple(TIER_COLOR)


def zone_of(ratio: pd.Series, bins: EfficiencyBins | None = None) -> pd.Series:
    """``over`` / ``well`` / ``under`` per idd-tools' thresholds on a used/asked ratio; NaN stays NaN."""
    b = bins or EfficiencyBins()
    values = np.select(
        [ratio < b.over_max, ratio < b.under_min], ["over", "well"], default="under"
    )
    out = pd.Series(values, index=ratio.index, dtype="object")
    out[ratio.isna()] = np.nan
    return out


def classify(usage: pd.DataFrame, bins: EfficiencyBins | None = None) -> pd.DataFrame:
    """Add ``mem_zone``, ``time_zone`` and the combined ``zone`` to a usage frame.

    The combined zone is ``under`` when either axis is under (the risky case),
    ``well`` when both are well, and ``over`` otherwise.
    """
    out = usage.copy()
    out["mem_zone"] = zone_of(out["mem_used_over_asked"], bins)
    out["time_zone"] = zone_of(out["time_used_over_asked"], bins)
    either_under = (out["mem_zone"] == "under") | (out["time_zone"] == "under")
    both_well = (out["mem_zone"] == "well") & (out["time_zone"] == "well")
    out["zone"] = np.where(either_under, "under", np.where(both_well, "well", "over"))
    return out


def zone_counts(classified: pd.DataFrame) -> pd.DataFrame:
    """Per workflow: task counts per zone on time, on memory, and combined.

    ``time_*`` and ``mem_*`` count one axis each. ``zone_*`` counts the
    combined zone: ``zone_under`` = under on either axis, ``zone_well`` = well
    on both, ``zone_over`` = the rest (over on at least one axis, under on
    none).
    """
    parts = []
    for col, prefix in (("time_zone", "time"), ("mem_zone", "mem"), ("zone", "zone")):
        ct = pd.crosstab(classified["workflow"], classified[col]).reindex(
            columns=ZONES, fill_value=0
        )
        ct.columns = [f"{prefix}_{z}" for z in ct.columns]
        parts.append(ct)
    out = pd.concat(parts, axis=1)
    out.insert(0, "tasks", classified.groupby("workflow").size())
    return out


COMBINED_LABELS = {
    "over": "over on at least one axis, under on none",
    "well": "correct on both axes",
    "under": "under on either axis",
}


def _zone_labels(bins: EfficiencyBins | None) -> dict[str, str]:
    """Per-axis zone labels with the thresholds spelled out."""
    b = bins or EfficiencyBins()
    return {
        "over": f"over-resourced (< {b.over_max:.0%} used)",
        "well": f"correctly resourced ({b.over_max:.0%} to {b.under_min:.0%})",
        "under": f"under-resourced (> {b.under_min:.0%} used)",
    }


def _style_axes(ax: Any, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(visible=True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)


def _shade(
    ax: Any,
    x_edges: tuple[float, float],
    y_edges: tuple[float, float],
    xmax: float,
    ymax: float,
) -> None:
    """Shade the plane by combined zone: under where either axis is past its upper edge, well in the middle box, over elsewhere."""
    (x_lo, x_hi), (y_lo, y_hi) = x_edges, y_edges
    ax.axvspan(0, xmax, color=ZONE_COLOR["over"], alpha=0.10, lw=0)
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (x_lo, y_lo),
            x_hi - x_lo,
            y_hi - y_lo,
            color=ZONE_COLOR["well"],
            alpha=0.16,
            lw=0,
        )
    )
    ax.axvspan(x_hi, xmax, color=ZONE_COLOR["under"], alpha=0.12, lw=0)
    ax.axhspan(y_hi, ymax, color=ZONE_COLOR["under"], alpha=0.12, lw=0)


def plot_usage(classified: pd.DataFrame, bins: EfficiencyBins | None = None) -> Any:
    """Plot 1: every task, x = wall minutes, y = process peak GiB; colour = memory zone, marker = time zone."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = _zone_labels(bins)
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    for mz in ZONES:
        for tz in ZONES:
            sel = classified[
                (classified["mem_zone"] == mz) & (classified["time_zone"] == tz)
            ]
            if sel.empty:
                continue
            ax.scatter(
                sel["wall_min"],
                sel["peak_gib"],
                s=MARKER_PT2,
                c=ZONE_COLOR[mz],
                marker=ZONE_MARKER[tz],
                edgecolors=SURFACE,
                linewidths=1,
                alpha=0.9,
            )
    _style_axes(ax, "wall time used (min)", "process peak memory used (GiB)")
    ax.set_title(
        f"Usage of {len(classified)} tasks; colour: memory zone, marker: time zone",
        color=INK,
        loc="left",
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=ZONE_COLOR[z],
            markersize=8,
            label=f"memory {labels[z]}",
        )
        for z in ZONES
    ]
    handles += [
        Line2D(
            [],
            [],
            marker=ZONE_MARKER[z],
            color="none",
            markerfacecolor=MUTED,
            markersize=8,
            label=f"time {labels[z]}",
        )
        for z in ZONES
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_usage_by_tier(
    classified: pd.DataFrame, bins: EfficiencyBins | None = None
) -> Any:
    """Plot 2: one panel per tier, x = wall minutes, y = peak GiB; the plane shaded by zone at the tier's median asks.

    Asks differ per task, so the shaded bands are drawn at ``over_max`` and
    ``under_min`` times the tier's *median* ask on each axis; a point's own
    zone (its marker) is exact.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    b = bins or EfficiencyBins()
    tiers = [t for t in TIERS if t in set(classified["tier"])]
    fig, axes = plt.subplots(
        1,
        len(tiers),
        figsize=(5 * len(tiers), 5),
        sharex=True,
        sharey=True,
        facecolor=SURFACE,
        squeeze=False,
    )
    xmax = float(classified["wall_min"].max()) * 1.1
    ymax = float(classified["peak_gib"].max()) * 1.1
    for ax, tier in zip(axes[0], tiers, strict=True):
        sel = classified[classified["tier"] == tier]
        ask_min, ask_gib = (
            float(sel["asked_min"].median()),
            float(sel["asked_gib"].median()),
        )
        _shade(
            ax,
            (b.over_max * ask_min, b.under_min * ask_min),
            (b.over_max * ask_gib, b.under_min * ask_gib),
            xmax,
            ymax,
        )
        for z in ZONES:
            s = sel[sel["zone"] == z]
            if s.empty:
                continue
            ax.scatter(
                s["wall_min"],
                s["peak_gib"],
                s=MARKER_PT2,
                c=ZONE_COLOR[z],
                marker=ZONE_MARKER[z],
                edgecolors=SURFACE,
                linewidths=1,
                alpha=0.9,
            )
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        _style_axes(ax, "wall time used (min)", "process peak memory used (GiB)")
        ax.set_title(
            f"{tier} tier: {len(sel)} tasks; median ask {ask_min:.0f} min, {ask_gib:.0f} GiB",
            color=INK,
            loc="left",
            fontsize=10,
        )
    handles = [
        Line2D(
            [],
            [],
            marker=ZONE_MARKER[z],
            color="none",
            markerfacecolor=ZONE_COLOR[z],
            markersize=8,
            label=COMBINED_LABELS[z],
        )
        for z in ZONES
    ]
    axes[0][-1].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=9,
        title="combined zone (shading: at the tier's median ask)",
        title_fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_ratios(classified: pd.DataFrame, bins: EfficiencyBins | None = None) -> Any:
    """Plot 3: every task, x = time used/asked, y = memory used/asked; colour and marker by tier; plane shaded by zone."""
    import matplotlib.pyplot as plt

    b = bins or EfficiencyBins()
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    xmax = max(1.1, float(classified["time_used_over_asked"].max()) * 1.1)
    ymax = max(1.1, float(classified["mem_used_over_asked"].max()) * 1.1)
    _shade(ax, (b.over_max, b.under_min), (b.over_max, b.under_min), xmax, ymax)
    for tier in TIERS:
        sel = classified[classified["tier"] == tier]
        if sel.empty:
            continue
        ax.scatter(
            sel["time_used_over_asked"],
            sel["mem_used_over_asked"],
            s=MARKER_PT2,
            c=TIER_COLOR[tier],
            marker=TIER_MARKER[tier],
            edgecolors=SURFACE,
            linewidths=1,
            alpha=0.9,
            label=f"{tier} tier ({len(sel)})",
        )
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=":")
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    _style_axes(ax, "time used / asked", "memory used / asked")
    ax.set_title(f"Used over asked, {len(classified)} tasks", color=INK, loc="left")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    return fig
