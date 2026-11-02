"""Mirror FABDEM V1-2 one-degree tiles from the University of Bristol data host.

FABDEM (Hawker et al. 2022) is Copernicus GLO-30 with building and forest
heights removed; licence CC BY-NC-SA 4.0. The host publishes the product as
356 ten-degree zip archives whose members are compressed GeoTIFF tiles. A
member is fetched on its own by HTTP byte ranges: the archive's central
directory first, then only that member's bytes. Nothing is unpacked locally,
so a run resumes per tile: a tile already on disk at the size the archive
records for it is skipped.

Why not download the zips: 263 GB of archives would have to be deleted again
afterwards, and shared storage is never cleaned up from code.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

from idd_tools.jobmon.atomic_io import AtomicRegistry, atomic_write, subtask_skip

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

HOST = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/"
INDEX_NAME = "FABDEM_v1-2_tiles.geojson"
SIZES_NAME = "zip_sizes.json"
USER_AGENT = "Mozilla/5.0 (idd-travel-time FABDEM mirror)"
CHUNK = 1 << 22
READ_SIZE = (
    16 << 20
)  # bytes per ranged GET: the host's per-connection rate is fixed, so fewer round trips is faster
WORKERS = 4  # tiles in flight per archive, each on its own connection (the host's limit is per connection)
HTTP_PARTIAL = 206


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests, which is what ``zipfile`` needs.

    Each ``read`` is one ranged GET; transient failures are retried with
    exponential back-off because the host drops long connections.
    """

    def __init__(self, url: str, *, timeout: float = 120.0, retries: int = 5) -> None:
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.pos = 0
        self.requests = 0
        if not url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            msg = (
                f"refusing URL {url!r}: https only (plain http only for loopback tests)"
            )
            raise ValueError(msg)
        head = urllib.request.Request(  # noqa: S310
            url, method="HEAD", headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(head, timeout=timeout) as r:  # noqa: S310
            self.size = int(r.headers["Content-Length"])
        probe = urllib.request.Request(  # noqa: S310
            url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"}
        )
        with urllib.request.urlopen(probe, timeout=timeout) as r:  # noqa: S310
            if r.status != HTTP_PARTIAL:
                msg = f"{url}: host does not honour range requests (status {r.status})"
                raise OSError(msg)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(  # noqa: S310
            self.url,
            headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"},
        )
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310
                    self.requests += 1
                    return bytes(r.read())
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                OSError,
            ) as exc:
                last = exc
                time.sleep(2**attempt)
        msg = f"{self.url}: range {start}-{end} failed after {self.retries} attempts: {last}"
        raise OSError(msg)

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        if self.pos >= self.size:
            return 0
        n = min(len(b), self.size - self.pos)
        data = self._get(self.pos, self.pos + n - 1)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url: str, *, buffer_size: int = READ_SIZE) -> zipfile.ZipFile:
    """A ``ZipFile`` over the remote archive; only the directory is read at open."""
    return zipfile.ZipFile(
        io.BufferedReader(HttpRangeFile(url), buffer_size=buffer_size)
    )


STUB_BYTES = (
    162  # what a HEAD on a misnamed archive recorded: the host's 404 body, not a zip
)
_SIGNED_HEMISPHERE = re.compile(r"([NS])-(\d)")


def host_zip_name(index_name: str) -> str:
    """The archive's name on data.bris.ac.uk, from the tile index's ``zipfile_name``.

    The index misnames 103 of its 356 archives (2026-10-23, workflow 643472
    ended E on them): a signed southern latitude (``S20E020-S-10E030``), the
    zero meridian as ``W000``, and the antimeridian as ``E180``; the host uses
    ``S10``, ``E000`` and ``W180``. Every other name is unchanged.
    """
    return (
        _SIGNED_HEMISPHERE.sub(r"\1\2", index_name)
        .replace("W000", "E000")
        .replace("E180", "W180")
    )


def head_size(url: str, *, timeout: float = 60.0) -> int | None:
    """Content-Length of ``url`` from a HEAD request; ``None`` when the host has no such file."""
    req = urllib.request.Request(url, method="HEAD")  # noqa: S310  # https host only
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310  # https host only
            return int(r.headers.get("Content-Length") or 0)
    except urllib.error.HTTPError as exc:
        if exc.code == HTTPStatus.NOT_FOUND:
            return None
        raise


def refresh_zip_sizes(
    path: Path,
    names: Iterable[str],
    *,
    host: str = HOST,
    head: Callable[[str], int | None] = head_size,
) -> tuple[dict[str, int], list[str]]:
    """Re-measure every archive recorded at or below ``STUB_BYTES`` (or missing) under its host name.

    Keys stay the index names; values are the host file's bytes. Archives the
    host still lacks are dropped from the file and returned so the launcher can
    report them instead of sizing blind. The file is rewritten atomically.
    """
    sizes = load_zip_sizes(path) if path.exists() else {}
    absent: list[str] = []
    for name in names:
        if sizes.get(name, 0) > STUB_BYTES:
            continue
        size = head(host + host_zip_name(name))
        if size is None or size <= STUB_BYTES:
            sizes.pop(name, None)
            absent.append(name)
        else:
            sizes[name] = size
    payload = {"zips": dict(sorted(sizes.items())), "n_tiles": None}
    with contextlib.suppress(OSError, ValueError):
        payload["n_tiles"] = json.loads(path.read_text()).get("n_tiles")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    atomic_write(path, lambda t: t.write_text(json.dumps(payload, indent=1)))
    return sizes, absent


_TILE_NAME = re.compile(r"^([NS])(\d{2,3})([EW])(\d{3})_FABDEM_V1-2\.tif$")


def archive_tile_name(index_name: str) -> str:
    """The host's index writes latitudes with three digits (``N079W106``); the archives use two (``N79W106``)."""
    m = _TILE_NAME.match(index_name)
    if not m:
        msg = f"unexpected tile name in the index: {index_name!r}"
        raise ValueError(msg)
    ns, lat, ew, lon = m.groups()
    return f"{ns}{int(lat):02d}{ew}{int(lon):03d}_FABDEM_V1-2.tif"


def load_tile_index(path: Path) -> dict[str, list[str]]:
    """Archive name -> sorted member tile names (archive naming), from the host's tile index GeoJSON."""
    data = json.loads(Path(path).read_text())
    out: dict[str, list[str]] = {}
    for feature in data["features"]:
        props = feature["properties"]
        out.setdefault(props["zipfile_name"], []).append(
            archive_tile_name(props["file_name"])
        )
    return {k: sorted(v) for k, v in sorted(out.items())}


def load_zip_sizes(path: Path) -> dict[str, int]:
    """Archive name -> size in bytes (from the host's response headers)."""
    data = json.loads(Path(path).read_text())
    return {k: int(v) for k, v in data["zips"].items() if v}


@dataclass(frozen=True)
class MemberOutcome:
    """What happened to one tile of one archive."""

    name: str
    size: int
    status: str  # fetched | present | failed
    error: str = ""


def _fetch_member(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    dest: Path,
    *,
    retries: int,
    registry: AtomicRegistry | None = None,
) -> MemberOutcome:
    """Stream one member into ``dest`` through idd-tools ``atomic_write``; register it on success.

    A lingering ``<dest>.tmp`` is a write this archive's task started and
    never finished (only one task ever writes a given tile), so it is
    removed first: that is the package's recover-then-skip contract.
    """
    stale = dest.with_name(dest.name + ".tmp")
    stale.unlink(missing_ok=True)

    def write(tmp: Path) -> None:
        with zf.open(info) as src, tmp.open("wb") as dst:
            while chunk := src.read(CHUNK):
                dst.write(chunk)
            # sync, then drop the file's pages from the cache: otherwise every byte written stays
            # charged to the task's cgroup and sacct reports the memory cap instead of the process
            dst.flush()
            os.fsync(dst.fileno())
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if tmp.stat().st_size != info.file_size:
            msg = f"size {tmp.stat().st_size} != {info.file_size}"
            raise OSError(msg)  # the zip reader has already checked the CRC

    last = ""
    for attempt in range(retries):
        try:
            atomic_write(dest, write)
        except (OSError, zipfile.BadZipFile) as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
            continue
        if registry is not None:
            registry.register(dest)
        return MemberOutcome(dest.name, info.file_size, "fetched")
    return MemberOutcome(dest.name, info.file_size, "failed", last)


def fetch_zip(
    zip_name: str,
    tiles_dir: Path,
    *,
    members: Iterable[str] | None = None,
    host: str = HOST,
    retries: int = 3,
    workers: int = WORKERS,
    read_size: int = READ_SIZE,
    progress: Callable[[MemberOutcome, int, int], None] | None = None,
    registry: AtomicRegistry | None = None,
) -> list[MemberOutcome]:
    """Fetch every ``.tif`` member of one archive into ``tiles_dir``; present tiles are skipped.

    ``workers`` tiles are fetched at once, each thread on its own connection
    to the archive (a ``ZipInfo`` carries the member's offset, so any
    ``ZipFile`` over the same bytes can read it). ``progress`` is called as
    each tile finishes with (outcome, done_so_far, total_to_fetch). A tile
    is skipped when idd-tools ``subtask_skip`` says it is done (present, no
    ``.tmp`` companion) and its size matches the archive's record; fetched
    tiles are registered in ``registry`` (the task's atomic sidecar).
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)
    wanted = None if members is None else set(members)
    url = host + host_zip_name(zip_name)  # the index misnames some archives
    listing = open_remote_zip(url, buffer_size=read_size)
    outcomes: list[MemberOutcome] = []
    todo: list[tuple[zipfile.ZipInfo, Path]] = []
    for info in listing.infolist():
        name = Path(info.filename).name
        if not name.endswith(".tif") or (wanted is not None and name not in wanted):
            continue
        dest = tiles_dir / name
        if subtask_skip(dest) and dest.stat().st_size == info.file_size:
            outcomes.append(MemberOutcome(name, info.file_size, "present"))
        else:
            todo.append((info, dest))
    local = threading.local()

    def one(item: tuple[zipfile.ZipInfo, Path]) -> MemberOutcome:
        if not hasattr(local, "zf"):
            local.zf = open_remote_zip(url, buffer_size=read_size)
        return _fetch_member(
            local.zf, item[0], item[1], retries=retries, registry=registry
        )

    if todo:
        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for done, outcome in enumerate(
                (
                    fut.result()
                    for fut in cf.as_completed(
                        [pool.submit(one, item) for item in todo]
                    )
                ),
                start=1,
            ):
                outcomes.append(outcome)
                if progress is not None:
                    progress(outcome, done, len(todo))
    return outcomes


def write_record(
    records_dir: Path, zip_name: str, outcomes: Sequence[MemberOutcome]
) -> Path:
    """One JSON record per archive per attempt, stamped with its finish time.

    Attempts are never overwritten: a later run that finds every tile present
    writes a second record with nothing fetched, and the runtime model joins
    each task to the record its own run produced by that stamp.
    """
    records_dir.mkdir(parents=True, exist_ok=True)
    finished = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = records_dir / f"{zip_name}.{finished.replace(':', '')}.json"
    out.write_text(
        json.dumps(
            {
                "zip": zip_name,
                "finished": finished,
                "members": [asdict(o) for o in outcomes],
            },
            indent=1,
        )
    )
    return out


def read_records(records_dir: Path) -> list[dict[str, object]]:
    """Every attempt record, oldest first; records without a stamp (first version) sort first."""
    recs = [json.loads(p.read_text()) for p in sorted(records_dir.glob("*.json"))]
    return sorted(recs, key=lambda r: str(r.get("finished", "")))


def done_archives(records: Iterable[Mapping[str, object]]) -> set[str]:
    """Archives whose latest attempt saw every member fetched or present, none failed."""
    latest: dict[str, Mapping[str, object]] = {}
    for rec in records:  # oldest first, so the last assignment wins
        latest[str(rec["zip"])] = rec
    out = set()
    for name, rec in latest.items():
        members = list(rec["members"])  # type: ignore[call-overload]
        if members and all(m["status"] in ("fetched", "present") for m in members):
            out.add(name)
    return out


def missing_tiles(index: dict[str, list[str]], tiles_dir: Path) -> list[str]:
    """Tiles named in the index that are absent or empty on disk."""
    return sorted(
        name
        for names in index.values()
        for name in names
        if not (tiles_dir / name).exists() or (tiles_dir / name).stat().st_size == 0
    )


def ordered_zips(
    index: dict[str, list[str]], sizes: dict[str, int] | None
) -> list[str]:
    """Archives ordered for a run: the largest first, then the median-sized one, then the rest by size.

    A probe of the first N tasks then measures the worst case and a typical case.
    """
    names = list(index)
    if not sizes:
        return names
    by_size = sorted(names, key=lambda n: sizes.get(n, 0), reverse=True)
    if len(by_size) < 3:  # noqa: PLR2004
        return by_size
    median = by_size[len(by_size) // 2]
    rest = [n for n in by_size if n not in (by_size[0], median)]
    return [by_size[0], median, *rest]
