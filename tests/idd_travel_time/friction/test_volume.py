"""Per-block input volume from parquet footers."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from idd_travel_time.friction import volume as vol
from idd_travel_time.friction.build import LONLAT_PAD_DEG
from idd_travel_time.friction.inputs import InputPaths


def _write_layer(
    directory: Path, name: str, boxes: list[tuple[float, float, float, float]]
) -> None:
    """One parquet file with one row group per box, so each row group's bbox statistics are the box."""
    directory.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": [str(i) for i in range(len(boxes))],
            "bbox": [
                {"xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]} for b in boxes
            ],
            "geometry": [b"x" * 1000 for _ in boxes],
        }
    )
    pq.write_table(table, directory / f"{name}.parquet", row_group_size=1)


def _paths(root: Path) -> InputPaths:
    return InputPaths(
        roads_dir=root / "roads",
        water_dir=root / "water",
        land_cover_dir=root / "lc",
        divisions_dir=root / "div",
        elevation_dir=root / "elev",
        elevation_fallback_dir=None,
        tile_index_root=root / "idx",
    )


def test_row_group_table_reads_extent_and_bytes_per_row_group(tmp_path: Path) -> None:
    _write_layer(tmp_path / "water", "ocean", [(-180, -90, 180, 90), (0, 0, 1, 1)])
    _write_layer(tmp_path / "roads", "primary", [(10, 10, 11, 11)])
    _write_layer(tmp_path / "lc", "a", [(0, 0, 2, 2)])
    _write_layer(tmp_path / "lc", "b", [(5, 5, 6, 6)])
    rg = vol.row_group_table(_paths(tmp_path))
    assert len(rg) == 5
    assert rg.groupby("layer").size().to_dict() == {
        "land_cover": 2,
        "roads": 1,
        "water": 2,
    }
    ocean = rg[(rg.layer == "water") & (rg.row_group == 0)].iloc[0]
    assert (ocean.xmin, ocean.ymin, ocean.xmax, ocean.ymax) == (-180, -90, 180, 90)
    assert ocean.bytes > 1000
    assert ocean.rows == 1


def test_row_group_table_refuses_a_layer_without_bbox(tmp_path: Path) -> None:
    _write_layer(tmp_path / "water", "ocean", [(0, 0, 1, 1)])
    _write_layer(tmp_path / "lc", "a", [(0, 0, 1, 1)])
    (tmp_path / "roads").mkdir()
    pq.write_table(
        pa.table({"id": ["1"], "geometry": [b"x"]}), tmp_path / "roads" / "r.parquet"
    )
    with pytest.raises(ValueError, match="no bbox"):
        vol.row_group_table(_paths(tmp_path))
    (tmp_path / "roads" / "r.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        vol.row_group_table(_paths(tmp_path))


def test_block_volumes_sum_the_row_groups_a_block_touches(tmp_path: Path) -> None:
    _write_layer(tmp_path / "water", "ocean", [(-180, -90, 180, 90), (0, 0, 1, 1)])
    _write_layer(tmp_path / "roads", "primary", [(10, 10, 11, 11)])
    _write_layer(tmp_path / "lc", "a", [(0, 0, 2, 2)])
    rg = vol.row_group_table(_paths(tmp_path))
    per_rg = float(rg.bytes.iloc[0]) / 1e6
    out = vol.block_volumes(
        {"A": (0.5, 0.5, 1.5, 1.5), "B": (50.0, 50.0, 51.0, 51.0)}, rg
    )
    assert list(out.columns) == vol.VOLUME_COLS
    assert out.loc["A", "rg_mb_water"] == pytest.approx(
        2 * per_rg
    )  # the ocean group and the small one
    assert out.loc["A", "rg_mb_land_cover"] == pytest.approx(per_rg)
    assert out.loc["A", "rg_mb_roads"] == 0.0
    assert out.loc["B", "rg_mb_water"] == pytest.approx(
        per_rg
    )  # the hemisphere-wide group only
    assert out.loc["B", "rg_mb_land_cover"] == 0.0


def test_padded_bounds_uses_the_worker_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    class Grid:
        def lonlat_bounds(
            self, pad_deg: float = 0.0
        ) -> tuple[float, float, float, float]:
            return (-pad_deg, -pad_deg, 1 + pad_deg, 1 + pad_deg)

    monkeypatch.setattr(vol, "block_grid", lambda _idx, _key, _res: Grid())
    b = vol.padded_bounds(["K"], object(), 40.0)["K"]
    assert b == (
        -LONLAT_PAD_DEG,
        -LONLAT_PAD_DEG,
        1 + LONLAT_PAD_DEG,
        1 + LONLAT_PAD_DEG,
    )
