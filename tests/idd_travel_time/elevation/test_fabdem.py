"""Offline tests for the FABDEM mirror: a local HTTP server that honours byte ranges."""

from __future__ import annotations

import http.server
import json
import threading
import zipfile
from pathlib import Path

import pytest

from idd_travel_time.elevation import fabdem as fb
from idd_travel_time.elevation import launch as ln


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus the Range support the mirror relies on."""

    def log_message(self, *args: object) -> None:  # quiet
        pass

    def do_HEAD(self) -> None:
        path = Path(self.directory) / self.path.lstrip("/")
        self.send_response(200)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()

    def do_GET(self) -> None:
        path = Path(self.directory) / self.path.lstrip("/")
        data = path.read_bytes()
        rng = self.headers.get("Range")
        if rng:
            start, end = rng.replace("bytes=", "").split("-")
            s, e = int(start), int(end) if end else len(data) - 1
            body = data[s : e + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {s}-{e}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def served_zip(tmp_path: Path):
    """A zip with two tiles and a text member, served on a local port with Range support."""
    site = tmp_path / "site"
    site.mkdir()
    tiles = {
        "N01E001_FABDEM_V1-2.tif": b"x" * 5000,
        "N01E002_FABDEM_V1-2.tif": b"y" * 12345,
    }
    with zipfile.ZipFile(
        site / "N00E000-N10E010_FABDEM_V1-2.zip", "w", zipfile.ZIP_DEFLATED
    ) as zf:
        for name, data in tiles.items():
            zf.writestr(name, data)
        zf.writestr("readme.txt", "not a tile")
    handler = lambda *a, **k: _RangeHandler(*a, directory=str(site), **k)  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_address[1]}/"
    yield host, tiles
    server.shutdown()


def test_http_range_file_seek_and_read(served_zip):
    host, _ = served_zip
    f = fb.HttpRangeFile(host + "N00E000-N10E010_FABDEM_V1-2.zip")
    assert f.size > 0
    f.seek(-22, 2)
    tail = f.read(22)
    assert tail[:4] == b"PK\x05\x06"  # end-of-central-directory signature
    assert f.requests == 1


def test_fetch_zip_fetches_then_skips(served_zip, tmp_path: Path):
    host, tiles = served_zip
    out = tmp_path / "tiles"
    first = fb.fetch_zip("N00E000-N10E010_FABDEM_V1-2.zip", out, host=host)
    assert {o.name: o.status for o in first} == dict.fromkeys(tiles, "fetched")
    for name, data in tiles.items():
        assert (out / name).read_bytes() == data
    assert not list(out.glob("*.tmp-*"))
    second = fb.fetch_zip("N00E000-N10E010_FABDEM_V1-2.zip", out, host=host)
    assert {o.status for o in second} == {"present"}


def test_fetch_zip_parallel_workers_and_small_reads(served_zip, tmp_path: Path):
    host, tiles = served_zip
    out = tmp_path / "tiles"
    got = fb.fetch_zip(
        "N00E000-N10E010_FABDEM_V1-2.zip", out, host=host, workers=3, read_size=1024
    )
    assert {o.status for o in got} == {"fetched"}
    for name, data in tiles.items():
        assert (out / name).read_bytes() == data


def test_fetch_zip_reports_progress(served_zip, tmp_path: Path):
    host, tiles = served_zip
    seen: list[tuple[str, int, int]] = []
    fb.fetch_zip(
        "N00E000-N10E010_FABDEM_V1-2.zip",
        tmp_path / "tiles",
        host=host,
        progress=lambda o, done, total: seen.append((o.status, done, total)),
    )
    assert sorted(seen) == [("fetched", 1, len(tiles)), ("fetched", 2, len(tiles))]


def test_fetch_zip_recovers_stale_tmp_and_registers(served_zip, tmp_path: Path):
    from idd_tools.jobmon.atomic_io import AtomicRegistry

    host, tiles = served_zip
    out = tmp_path / "tiles"
    out.mkdir()
    stale = out / "N01E001_FABDEM_V1-2.tif.tmp"
    stale.write_bytes(b"partial")
    (out / "N01E001_FABDEM_V1-2.tif").write_bytes(
        b"x" * 5000
    )  # right size but a .tmp companion: not done
    with AtomicRegistry(
        tmp_path, task_id="zip:test", task_template="fabdem_zip"
    ) as reg:
        got = fb.fetch_zip(
            "N00E000-N10E010_FABDEM_V1-2.zip", out, host=host, registry=reg
        )
    assert {o.name: o.status for o in got} == dict.fromkeys(tiles, "fetched")
    assert not stale.exists()
    sidecar = tmp_path / ".atomic" / "zip:test.json"
    assert sidecar.exists()
    names = [Path(e["path"] if isinstance(e, dict) else e).name for e in reg.entries]
    assert sorted(names) == sorted(tiles)
    again = fb.fetch_zip("N00E000-N10E010_FABDEM_V1-2.zip", out, host=host)
    assert {o.status for o in again} == {"present"}


def test_fetch_zip_member_subset(served_zip, tmp_path: Path):
    host, _ = served_zip
    out = tmp_path / "tiles"
    got = fb.fetch_zip(
        "N00E000-N10E010_FABDEM_V1-2.zip",
        out,
        members=["N01E002_FABDEM_V1-2.tif"],
        host=host,
    )
    assert [o.name for o in got] == ["N01E002_FABDEM_V1-2.tif"]


def test_write_record_and_missing_tiles(tmp_path: Path):
    outcomes = [
        fb.MemberOutcome("a.tif", 3, "fetched"),
        fb.MemberOutcome("b.tif", 4, "failed", "boom"),
    ]
    rec = fb.write_record(tmp_path / "records", "Z.zip", outcomes)
    assert json.loads(rec.read_text())["members"][1]["error"] == "boom"
    recs = fb.read_records(tmp_path / "records")
    assert [r["zip"] for r in recs] == ["Z.zip"]
    assert recs[0]["finished"]
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "a.tif").write_bytes(b"abc")
    (tiles / "c.tif").write_bytes(b"")
    index = {"Z.zip": ["a.tif", "b.tif", "c.tif"]}
    assert fb.missing_tiles(index, tiles) == ["b.tif", "c.tif"]


def test_archive_tile_name_normalises_three_digit_latitudes():
    assert fb.archive_tile_name("N079W106_FABDEM_V1-2.tif") == "N79W106_FABDEM_V1-2.tif"
    assert fb.archive_tile_name("S001W161_FABDEM_V1-2.tif") == "S01W161_FABDEM_V1-2.tif"
    assert fb.archive_tile_name("N00E010_FABDEM_V1-2.tif") == "N00E010_FABDEM_V1-2.tif"
    with pytest.raises(ValueError, match="unexpected tile name"):
        fb.archive_tile_name("junk.tif")


def test_load_tile_index_and_sizes(tmp_path: Path):
    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "tile_name": "N01E002",
                    "file_name": "N001E002_FABDEM_V1-2.tif",
                    "zipfile_name": "Z1.zip",
                },
                "geometry": None,
            },
            {
                "type": "Feature",
                "properties": {
                    "tile_name": "N01E001",
                    "file_name": "N01E001_FABDEM_V1-2.tif",
                    "zipfile_name": "Z1.zip",
                },
                "geometry": None,
            },
            {
                "type": "Feature",
                "properties": {
                    "tile_name": "S05E100",
                    "file_name": "S05E100_FABDEM_V1-2.tif",
                    "zipfile_name": "Z2.zip",
                },
                "geometry": None,
            },
        ],
    }
    p = tmp_path / fb.INDEX_NAME
    p.write_text(json.dumps(gj))
    index = fb.load_tile_index(p)
    assert index == {
        "Z1.zip": ["N01E001_FABDEM_V1-2.tif", "N01E002_FABDEM_V1-2.tif"],
        "Z2.zip": ["S05E100_FABDEM_V1-2.tif"],
    }
    s = tmp_path / fb.SIZES_NAME
    s.write_text(json.dumps({"zips": {"Z1.zip": 10, "Z2.zip": None, "Z3.zip": 30}}))
    assert fb.load_zip_sizes(s) == {"Z1.zip": 10, "Z3.zip": 30}


def test_ordered_zips_puts_largest_then_median_first():
    index: dict[str, list[str]] = {f"Z{i}.zip": [] for i in range(5)}
    sizes = {"Z0.zip": 5, "Z1.zip": 50, "Z2.zip": 20, "Z3.zip": 40, "Z4.zip": 10}
    order = fb.ordered_zips(index, sizes)
    assert order[0] == "Z1.zip"
    assert order[1] == "Z2.zip"
    assert sorted(order) == sorted(index)
    assert fb.ordered_zips(index, None) == list(index)


def test_remaining_sizes_scales_by_missing_tiles(tmp_path: Path):
    index = {
        "A.zip": ["a1.tif", "a2.tif", "a3.tif", "a4.tif"],
        "B.zip": ["b1.tif"],
        "C.zip": ["c1.tif"],
    }
    sizes = {"A.zip": 4_000, "B.zip": 500, "C.zip": 700}
    (tmp_path / "a1.tif").write_bytes(b"x")
    (tmp_path / "a2.tif").write_bytes(b"x")
    (tmp_path / "a3.tif").write_bytes(b"x")
    (tmp_path / "a3.tif.tmp").write_bytes(b"partial")  # a3 is not done
    (tmp_path / "b1.tif").write_bytes(b"x")
    (tmp_path / "c1.tif").write_bytes(b"")  # empty: not done
    left = ln.remaining_sizes(index, sizes, tmp_path)
    assert left == {"A.zip": 2_000, "C.zip": 700}


def test_bundle_archives_caps_the_number_of_tiny_archives():
    sizes = {f"t{i}.zip": 1 for i in range(100)}
    bundles = ln.bundle_archives(sizes, target_bytes=10_000, max_archives=40)
    assert [len(b) for b in bundles] == [40, 40, 20]


def test_done_archives_uses_latest_attempt():
    recs: list[dict[str, object]] = [
        {
            "zip": "A.zip",
            "finished": "2026-01-01T00:00:00",
            "members": [{"status": "failed"}],
        },
        {
            "zip": "A.zip",
            "finished": "2026-01-02T00:00:00",
            "members": [{"status": "present"}, {"status": "fetched"}],
        },
        {
            "zip": "B.zip",
            "finished": "2026-01-02T00:00:00",
            "members": [{"status": "fetched"}, {"status": "failed"}],
        },
        {"zip": "C.zip", "finished": "2026-01-02T00:00:00", "members": []},
    ]
    assert fb.done_archives(recs) == {"A.zip"}


def test_bundle_archives_packs_small_ones_and_isolates_big_ones():
    sizes = {"big.zip": 2_000, "a.zip": 400, "b.zip": 400, "c.zip": 400, "d.zip": 50}
    bundles = ln.bundle_archives(sizes, target_bytes=1_000)
    assert bundles[0] == ["big.zip"]
    assert sorted(n for b in bundles for n in b) == sorted(sizes)
    for b in bundles[1:]:
        assert sum(sizes[n] for n in b) <= 1_000
    assert [sum(sizes[n] for n in b) for b in bundles] == sorted(
        (sum(sizes[n] for n in b) for b in bundles), reverse=True
    )


def test_manifest_template_and_logs(tmp_path: Path):
    sizes = {"A_FABDEM_V1-2.zip": 3_000_000, "B_FABDEM_V1-2.zip": 1_000_000}
    manifest = ln.zip_manifest(
        [["A_FABDEM_V1-2.zip", "B_FABDEM_V1-2.zip"]],
        sizes,
        tmp_path,
        workflow_name="wf",
    )
    assert [t.task_id for t in manifest.tasks] == ["zip:A+2"]
    assert manifest.tasks[0].task_args == {
        "zip_names": "A_FABDEM_V1-2.zip,B_FABDEM_V1-2.zip",
        "fabdem_dir": str(tmp_path),
        "workers": 4,
        "read_mb": 16,
        "task_id": "zip:A+2",
    }
    assert manifest.tasks[0].task_features == {"n_archives": 2, "total_mb": 4.0}
    spec = ln.template_spec(Path("/venv/bin/fetch-fabdem"))
    assert spec.command_template == (
        "/venv/bin/fetch-fabdem zip --zip-names {zip_names} --fabdem-dir {fabdem_dir}"
        " --workers {workers} --read-mb {read_mb} --task-id {task_id}"
    )
    assert spec.node_args == ["zip_names"]
    out, err = ln.log_paths(manifest.tasks[0], tmp_path)
    assert out.endswith("logs/A+2.out")
    assert err.endswith("logs/A+2.err")


def test_runtime_model_sizes_tasks_from_bytes(tmp_path: Path):
    import pandas as pd

    def rec(name, finished, size, status):
        return {
            "zip": name,
            "finished": finished,
            "members": [{"name": "t", "size": size, "status": status, "error": ""}],
        }

    records = [
        rec("X.zip", "2026-01-01T10:10:00", 2_000_000_000, "fetched"),
        rec("Y.zip", "2026-01-01T10:05:00", 1_000_000_000, "fetched"),
        rec("Z.zip", "2026-01-01T10:00:30", 5, "present"),
        rec(
            "X.zip", "2026-01-02T10:00:00", 5, "present"
        ),  # a later present-only attempt: must not count
    ]
    audit = pd.DataFrame(
        {
            "task_id": ["zip:X", "zip:Y", "zip:Z", "zip:W", "zip:V"],
            "zip_names": ["X.zip", "Y.zip", "Z.zip", "W.zip", "V.zip"],
            "state": ["COMPLETED", "COMPLETED", "COMPLETED", "FAILED", "COMPLETED"],
            "elapsed_seconds": [1200.0, 600.0, 30.0, 10.0, 50.0],
            "start": ["2026-01-01T09:50:00"] * 5,
            "end": ["2026-01-01T10:12:00"] * 5,
        }
    )
    frame = ln.training_frame(audit, records)
    assert sorted(frame["task_id"]) == ["zip:X", "zip:Y", "zip:Z"]
    assert frame.set_index("task_id").loc["zip:X", "fetched_mb"] == 2_000.0
    pred = ln.fit_runtime_predictor(frame)
    sizes = {"P.zip": 3_000_000_000, "Q.zip": 100_000_000}
    manifest = ln.zip_manifest(
        [["P.zip"], ["Q.zip"]], sizes, tmp_path, workflow_name="wf"
    )
    asks = ln.runtime_asks(pred, manifest)
    assert asks["zip:P"] > asks["zip:Q"]
    assert asks["zip:Q"] >= ln.RUNTIME_FLOOR_MIN
    assert asks["zip:P"] > 30.0  # 3 GB at ~0.6 min per 100 MB
    assert asks["zip:P"] < 90.0  # with the policy margin
    res = ln.resources_for(asks, 1.0, 1)(manifest.tasks[0])
    assert res["memory"] == "1G"
    assert res["cores"] == 1
    assert str(res["runtime"]).endswith(("m", "h"))


def test_save_and_load_audits_accumulate_runs(tmp_path: Path):
    import pandas as pd

    a = pd.DataFrame(
        {
            "workflow_id": [1, 1],
            "task_id": ["zip:A", "zip:B"],
            "state": ["COMPLETED"] * 2,
            "elapsed_seconds": [10.0, 20.0],
        }
    )
    b = pd.DataFrame(
        {
            "workflow_id": [2, 2],
            "task_id": ["zip:A", "zip:C"],
            "state": ["COMPLETED"] * 2,
            "elapsed_seconds": [11.0, 30.0],
        }
    )
    ln.save_audit(a, tmp_path / "audits")
    ln.save_audit(b, tmp_path / "audits")
    assert sorted(p.name for p in (tmp_path / "audits").glob("*.csv")) == [
        "1.csv",
        "2.csv",
    ]
    allf = ln.load_audits(tmp_path / "audits")
    assert len(allf) == 4  # same task id under two workflows is two measurements
    again = ln.load_audits(tmp_path / "audits", b)
    assert (
        len(again) == 4
    )  # the latest frame duplicates the saved one and is deduplicated
    assert ln.load_audits(tmp_path / "nothing").empty


def test_rate_asks_fallback(tmp_path: Path):
    sizes = {"P.zip": 2_880_000_000, "Q.zip": 10_000_000}
    manifest = ln.zip_manifest(
        [["P.zip"], ["Q.zip"]], sizes, tmp_path, workflow_name="wf"
    )
    asks = ln.rate_asks(manifest, mb_per_s=2.4, margin=1.5, overhead_min=1.0)
    assert abs(asks["zip:P"] - (2880 / 2.4 / 60 * 1.5 + 1)) < 1e-6
    assert (
        ln.rate_asks(manifest)["zip:P"] < asks["zip:P"]
    )  # the default overhead is smaller
    assert asks["zip:Q"] == ln.RUNTIME_FLOOR_MIN


def test_model_usable_needs_count_and_spread():
    import pandas as pd

    same = pd.DataFrame(
        {
            "fetched_mb": [2130.0, 2133.0, 2131.0, 2123.0],
            "elapsed_min": [15.4, 15.6, 14.9, 17.1],
        }
    )
    assert not ln.model_usable(same)  # four tasks, no size spread
    spread = pd.DataFrame(
        {"fetched_mb": [300.0, 1200.0, 2100.0], "elapsed_min": [3.0, 9.0, 15.0]}
    )
    assert ln.model_usable(spread)
    assert not ln.model_usable(spread.iloc[:2])  # below the task count


def test_measured_rate_is_median_mb_per_s():
    import pandas as pd

    frame = pd.DataFrame(
        {"fetched_mb": [1200.0, 600.0, 0.0], "elapsed_min": [10.0, 10.0, 1.0]}
    )
    assert ln.measured_rate(frame) == pytest.approx(
        (2.0 + 1.0) / 2
    )  # zero-fetch row excluded
    assert ln.measured_rate(pd.DataFrame()) == ln.MEASURED_MB_PER_S


def test_host_zip_name_fixes_the_three_index_misnamings():
    assert (
        fb.host_zip_name("S20E020-S-10E030_FABDEM_V1-2.zip")
        == "S20E020-S10E030_FABDEM_V1-2.zip"
    )
    assert (
        fb.host_zip_name("S60W060-S-50W050_FABDEM_V1-2.zip")
        == "S60W060-S50W050_FABDEM_V1-2.zip"
    )
    assert (
        fb.host_zip_name("N10W010-N20W000_FABDEM_V1-2.zip")
        == "N10W010-N20E000_FABDEM_V1-2.zip"
    )
    assert (
        fb.host_zip_name("N00E170-N10E180_FABDEM_V1-2.zip")
        == "N00E170-N10W180_FABDEM_V1-2.zip"
    )
    assert (
        fb.host_zip_name("N40W110-N50W100_FABDEM_V1-2.zip")
        == "N40W110-N50W100_FABDEM_V1-2.zip"
    )


def test_refresh_zip_sizes_remeasures_stubs_under_host_names(tmp_path: Path):
    import json

    path = tmp_path / "zip_sizes.json"
    path.write_text(
        json.dumps(
            {
                "zips": {
                    "A.zip": 1000,
                    "S20E020-S-10E030_FABDEM_V1-2.zip": 162,
                    "N00E170-N10E180_FABDEM_V1-2.zip": 162,
                },
                "n_tiles": 3,
            }
        )
    )
    seen = []

    def head(url):
        seen.append(url)
        return 5000 if "S10E030" in url else None

    sizes, absent = fb.refresh_zip_sizes(
        path,
        [
            "A.zip",
            "S20E020-S-10E030_FABDEM_V1-2.zip",
            "N00E170-N10E180_FABDEM_V1-2.zip",
        ],
        host="https://h/",
        head=head,
    )
    assert sizes == {
        "A.zip": 1000,
        "S20E020-S-10E030_FABDEM_V1-2.zip": 5000,
    }  # key stays the index name
    assert absent == ["N00E170-N10E180_FABDEM_V1-2.zip"]
    assert seen == [
        "https://h/S20E020-S10E030_FABDEM_V1-2.zip",
        "https://h/N00E170-N10W180_FABDEM_V1-2.zip",
    ]
    saved = json.loads(path.read_text())
    assert saved["zips"] == sizes
    assert saved["n_tiles"] == 3
    assert not path.with_name("zip_sizes.json.tmp").exists()


def test_memory_to_jobmon_uses_mib_below_one_gib() -> None:
    assert ln.memory_to_jobmon(0.5859375) == "600M"
    assert ln.memory_to_jobmon(0.586) == "601M"
    assert ln.memory_to_jobmon(1.0) == "1G"
    assert ln.memory_to_jobmon(2.3) == "3G"


def test_bundle_task_id_stays_short_for_wide_bundles() -> None:
    one = ["N20E130-N30E140_FABDEM_V1-2.zip"]
    assert ln.bundle_task_id(one) == "zip:N20E130-N30E140"
    wide = [f"S{i:02d}W010-S-{i - 10:02d}W000_FABDEM_V1-2.zip" for i in range(10, 50)]
    tid = ln.bundle_task_id(wide)
    assert tid == "zip:S10W010-S-00W000+40"
    assert len(f"{tid}.json".encode()) < 255
