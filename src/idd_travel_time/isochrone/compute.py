"""Isochrone computation — thin dispatch layer.

This module is the single call site callers use to compute isochrones.
It owns:
- Router selection (default: OsmnxRouter when no ORS key is set,
  OpenRouteServiceRouter when ORS_API_KEY is present)
- The public compute() function
- Batch helpers for multi-origin runs

Callers that want explicit router control can pass one in:

    from idd_travel_time.isochrone import compute
    from idd_travel_time.routers import OsmnxRouter

    result = compute(request, router=OsmnxRouter())
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from idd_travel_time.routers.base import Router

if TYPE_CHECKING:
    from idd_travel_time.models import TravelTimeRequest, TravelTimeResult


def _default_router() -> Router:
    """Select the best available router for the current environment.

    Priority:
    1. OpenRouteServiceRouter if ORS_API_KEY is set.
    2. OsmnxRouter otherwise (local, no key required).
    """
    if os.environ.get("ORS_API_KEY"):
        from idd_travel_time.routers.openroute import OpenRouteServiceRouter  # noqa: PLC0415
        return OpenRouteServiceRouter()

    from idd_travel_time.routers.osmnx_router import OsmnxRouter  # noqa: PLC0415
    return OsmnxRouter()


def compute(
    request: TravelTimeRequest,
    *,
    router: Router | None = None,
) -> TravelTimeResult:
    """Compute a travel-time isochrone for *request*.

    Parameters
    ----------
    request:
        Fully populated TravelTimeRequest.
    router:
        Routing backend to use. If ``None``, selects automatically:
        ORS if ``ORS_API_KEY`` is set, osmnx otherwise.

    Returns
    -------
    TravelTimeResult
        Isochrone polygon plus optional network/POI GeoDataFrames.

    Examples
    --------
    >>> from idd_travel_time.isochrone import compute
    >>> from idd_travel_time.models import TravelTimeRequest
    >>> from idd_travel_time.constants import TransportMode
    >>>
    >>> req = TravelTimeRequest(
    ...     origin_lat=-1.286389,
    ...     origin_lon=36.817223,
    ...     mode=TransportMode.WALKING,
    ...     duration_minutes=30,
    ... )
    >>> result = compute(req)                          # auto-selects router
    >>> result = compute(req, router=OsmnxRouter())    # explicit router
    """
    active_router = router if router is not None else _default_router()
    return active_router.compute_isochrone(request)


def compute_batch(
    requests: list[TravelTimeRequest],
    *,
    router: Router | None = None,
) -> list[TravelTimeResult]:
    """Compute isochrones for multiple requests sequentially.

    All requests share the same router instance. For parallel / cluster
    execution, callers should manage concurrency externally.

    Parameters
    ----------
    requests:
        List of TravelTimeRequest objects.
    router:
        Routing backend. Auto-selected if ``None``.

    Returns
    -------
    list[TravelTimeResult]
        Results in the same order as *requests*.
    """
    active_router = router if router is not None else _default_router()
    return [active_router.compute_isochrone(req) for req in requests]


def compute_parallel(
    requests: list[TravelTimeRequest],
    *,
    router_factory: object | None = None,
    max_workers: int | None = None,
    output_dir: str | Path | None = None,
    skip_existing: bool = True,
    on_error: str = "warn",
    use_threads: bool = False,
) -> list[TravelTimeResult | None]:
    """Compute isochrones for multiple requests in parallel.

    Uses :class:`concurrent.futures.ProcessPoolExecutor` so each worker
    is a separate process — safe for CPU-bound routing work and avoids
    Python GIL contention.

    Each worker instantiates its own router via *router_factory* (a
    zero-argument callable). This is necessary because router objects
    (e.g. osmnx graphs, HTTP sessions) are not safely shareable across
    processes.

    Parameters
    ----------
    requests:
        List of TravelTimeRequest objects.
    router_factory:
        Zero-argument callable returning a :class:`~idd_travel_time.routers.base.Router`.
        Defaults to ``_default_router``. Called once per worker process.
    max_workers:
        Number of worker processes. Defaults to ``os.cpu_count()``.
    output_dir:
        If set, each result is written to disk via
        :func:`~idd_travel_time.io.write_result` immediately after
        computation. Useful for long batch runs — results are safe even
        if a later worker fails.
    skip_existing:
        When *output_dir* is set, skip requests whose result directory
        already exists. Allows resuming an interrupted batch run.
    on_error:
        How to handle per-request failures:
        ``"warn"`` — log a warning and insert ``None`` in the results list.
        ``"raise"`` — re-raise the first exception encountered.

    Returns
    -------
    list[TravelTimeResult | None]
        Results in the same order as *requests*. Failed requests produce
        ``None`` (when *on_error="warn"*).

    Examples
    --------
    >>> from idd_travel_time.isochrone import compute_parallel
    >>> from idd_travel_time.routers import OsmnxRouter
    >>> from idd_travel_time.io import GraphCache
    >>>
    >>> cache = GraphCache("/data/graph_cache")
    >>> results = compute_parallel(
    ...     requests,
    ...     router_factory=lambda: OsmnxRouter(graph_cache=cache),
    ...     max_workers=4,
    ...     output_dir="/data/results",
    ...     skip_existing=True,
    ... )
    """
    import logging  # noqa: PLC0415
    import warnings  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    from idd_travel_time.io import result_id, write_result  # noqa: PLC0415

    logger = logging.getLogger(__name__)

    if on_error not in ("warn", "raise"):
        msg = f"on_error must be 'warn' or 'raise', got {on_error!r}"
        raise ValueError(msg)

    out_root = _Path(output_dir) if output_dir is not None else None
    factory = router_factory if router_factory is not None else _default_router

    # Build work list, optionally skipping already-computed results.
    work: list[tuple[int, TravelTimeRequest]] = []
    results: list[TravelTimeResult | None] = [None] * len(requests)

    for i, req in enumerate(requests):
        if out_root is not None and skip_existing:
            expected = out_root / result_id(req)
            if expected.is_dir() and (expected / "meta.json").exists():
                # Load existing result rather than recomputing.
                try:
                    from idd_travel_time.io import read_result  # noqa: PLC0415
                    results[i] = read_result(expected)
                    continue
                except Exception:  # noqa: BLE001
                    pass  # corrupted — recompute
        work.append((i, req))

    if not work:
        return results

    futures = {}
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with executor_cls(max_workers=max_workers) as pool:
        for i, req in work:
            fut = pool.submit(_worker, req, factory)
            futures[fut] = i

        for fut in as_completed(futures):
            i = futures[fut]
            try:
                result = fut.result()
                results[i] = result
                if out_root is not None:
                    write_result(result, out_root)
            except Exception as exc:  # noqa: BLE001
                if on_error == "raise":
                    raise
                warnings.warn(
                    f"Request {i} failed: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    return results


def _worker(
    request: TravelTimeRequest,
    router_factory: object,
) -> TravelTimeResult:
    """Top-level function executed in each worker process.

    Must be importable at module level (not a lambda or nested function)
    for pickling under ProcessPoolExecutor.
    """
    router = router_factory()  # type: ignore[call-arg]
    return router.compute_isochrone(request)
