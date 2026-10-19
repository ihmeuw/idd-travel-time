"""Tests for idd_travel_time.paths — the two-tier path layer and node slots."""

from __future__ import annotations

from pathlib import Path

import pytest

from idd_travel_time import paths


@pytest.fixture
def paths_file(tmp_path: Path) -> Path:
    root = tmp_path / "out"
    f = tmp_path / "paths.yaml"
    f.write_text(f"project_output_root: {root}\n")
    return f


class TestLoadPaths:
    def test_reads_yaml(self, paths_file: Path, tmp_path: Path) -> None:
        assert paths.load_paths(paths_file) == {
            "project_output_root": str(tmp_path / "out")
        }

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match=r"paths\.yaml\.example"):
            paths.load_paths(tmp_path / "nope.yaml")


class TestNodeLayout:
    def test_output_root(self, paths_file: Path, tmp_path: Path) -> None:
        assert paths.output_root(paths_file) == tmp_path / "out"

    def test_output_root_expands_user(self, tmp_path: Path) -> None:
        f = tmp_path / "paths.yaml"
        f.write_text("project_output_root: ~/somewhere\n")
        assert paths.output_root(f) == Path("~/somewhere").expanduser()

    def test_node_dir(self, paths_file: Path, tmp_path: Path) -> None:
        assert paths.node_dir(paths.NODE_MAPS, paths_file) == tmp_path / "out" / "maps"

    def test_working_dir_is_created_slot(
        self, paths_file: Path, tmp_path: Path
    ) -> None:
        slot = paths.working_dir(paths.NODE_RESULTS, paths_file=paths_file)
        assert slot == tmp_path / "out" / "results" / "working"
        assert slot.is_dir()

    def test_scratch_dir(self, paths_file: Path, tmp_path: Path) -> None:
        d = paths.scratch_dir(paths.NODE_MAPS, "probe", paths_file)
        assert d == tmp_path / "out" / "maps" / "scratch" / "probe"
