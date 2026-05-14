# Decisions log

<!-- Append-only. Never delete or overwrite entries. -->

## 2026-05-06: Tiling deferred — single-city focus for proof-of-principle
**Decision:** No tile-aware surface computation in the v0.1 release; single-city
bbox is the supported unit.
**Why:** Proof-of-principle phase. Cross-tile border-load handling adds
complexity (8-neighbour reads, graph stitching) that doesn't change the
methodology being validated. Defer until a real multi-city target appears.
**Revisit if:** A consumer requests country- or region-scale surfaces, or
single-bbox memory/runtime becomes the bottleneck.

---

<!-- Entries below pre-date the 2026-05-14 schema migration.
     Future entries use canonical schema:
     Decision / Why / Revisit if. -->

## 2026-05-06: No GTFS / no schedule-exact transit
**Context:** Library must work globally, including cities with no formal transit data (Abuja,
Iquitos). Schedule-exact tools (r5py, OpenTripPlanner) require GTFS feeds.
**Decision:** Use TransitProfile speed-assumption model instead of schedule-exact routing.
**Rationale:** Global coverage > precision. Typical-speed assumptions are sufficient for
isochrone planning at the scale IDD cares about.
**Alternatives rejected:** r5py — requires GTFS, not available for most LMICs.

## 2026-05-06: Router is an abstract base class
**Context:** ORS is free but rate-limited; Google Maps is more reliable but paid. We want
to swap backends without changing call sites.
**Decision:** Abstract `Router` base class with a single `compute_isochrone(request)` method.
ORS is default; osmnx is local fallback; Google is a paid stub.
**Rationale:** One-line swap at call time; no architectural change needed when upgrading.
**Alternatives rejected:** Hardcoding ORS — makes paid upgrade painful.

## 2026-05-06: isochrone/ is a subpackage, not a module
**Context:** Anticipate growth: batch computation, caching, multi-origin runs.
**Decision:** `isochrone/` subpackage with `compute.py` entry point.
**Rationale:** Easier to extend than a flat module.
**Alternatives rejected:** Single `isochrone.py` — hard to split later.

## 2026-05-06: No numbered pipeline stages
**Context:** This is a library + CLI, not a sequential data pipeline.
**Decision:** Flat subpackage structure (profiles/, routers/, isochrone/, viz/).
**Rationale:** Numbered stages imply sequential dependency; library components are orthogonal.
**Alternatives rejected:** Numbered stages — wrong metaphor for a library.

## 2026-05-06: ORS transit range scaling via speed ratio
**Context:** ORS has no transit profile. For TRANSIT mode we use driving-car as the
road-network proxy, but we need the polygon to cover the correct real-world area.
**Decision:** Scale the ORS range_seconds by (TransitProfile.midpoint_speed / 30 kph).
30 kph is ORS's implicit urban driving assumption.
**Rationale:** Simple, transparent, matches the "typical speed" philosophy of the project.
**Alternatives rejected:** Passing a custom speed to ORS speed_limit — not supported in
the isochrones endpoint; would require Directions API workaround.

## 2026-05-06: osmnx polygon via node-buffer union
**Context:** osmnx can return a convex hull but it over-approximates. Alpha shapes are
fragile and require tuning a parameter.
**Decision:** Buffer each reachable node by ~50m in UTM, then union all buffers.
**Rationale:** Naturally follows the road network topology; no tuning parameter; robust
to sparse node distributions; fast.
**Alternatives rejected:** Convex hull — too coarse. Shapely alphashape — fragile,
requires tuning alpha; not in standard Shapely.

## 2026-05-06: _default_router() auto-selects ORS > osmnx
**Context:** Users shouldn't need to think about router selection for basic use.
**Decision:** If ORS_API_KEY is set, use ORS; otherwise fall back to osmnx (no key required).
**Rationale:** ORS gives better polygons via its native isochrone API; osmnx is always available.
**Alternatives rejected:** Always requiring explicit router — too much friction for notebooks.

## 2026-05-06: Heatmap derived from isochrone polygon boundary points
**Context:** A true density heatmap would need the full road network node set.
**Decision:** Sample N points uniformly along the polygon exterior and feed to folium HeatMap.
**Rationale:** Always available (no network_gdf required), visually communicates the reachable
boundary shape, no extra data fetching.
**Alternatives rejected:** Node-based heatmap — requires network_gdf, which is optional.

## 2026-05-06: GeoParquet result layout — one directory per result
**Context:** Need persistent results that survive across sessions and can be
shared between collaborators and cluster workers.
**Decision:** One directory per result: meta.json (request provenance + schema
version) + isochrone.parquet + optional network.parquet + optional poi.parquet.
Directory name = result_id(request) — deterministic from request parameters so
re-runs are idempotent overwrites, not accumulating duplicates.
**Rationale:** Directory-per-result is easy to list, inspect, and delete
individually. GeoParquet preserves CRS and geometry types natively. JSON sidecar
keeps the full request round-trippable without baking it into GDF columns.
**Alternatives rejected:** Single flat parquet per result — can't store multiple
GDFs cleanly. SQLite/PostGIS — too heavy for proof-of-principle.

## 2026-05-06: Graph cache key rounds to 4 decimal places
**Context:** Two requests for "Nairobi city centre" may have slightly different
lat/lon fractions. We want them to share a cached graph.
**Decision:** Round coordinates to 4 decimal places (~11 m) for the cache key.
**Rationale:** 11 m precision is fine for a graph download radius of kilometres.
Avoids spurious cache misses from floating-point noise.
**Alternatives rejected:** Exact coordinates — too many spurious misses.
Grid-snapping — more complex, same effect.

## 2026-05-06: Graph cache uses per-key threading.Lock
**Context:** Parallel batch workers for the same origin must not trigger
duplicate Overpass downloads.
**Decision:** One threading.Lock per cache key, created on first access.
Concurrent requests for different origins proceed in parallel; duplicate
requests for the same origin serialise behind one download.
**Rationale:** Minimal overhead; solves the duplicate-download problem without
a process-level lock that would serialise all workers.
**Alternatives rejected:** Global lock — serialises all workers. No lock — race
condition produces duplicate downloads and potentially corrupt GraphML files.

## 2026-05-06: compute_parallel accepts use_threads flag
**Context:** ProcessPoolExecutor can't pickle locally-defined classes (test mocks,
lambdas). ORS is I/O-bound so threads are more efficient than processes for it.
**Decision:** Add use_threads=False default. False = ProcessPoolExecutor (CPU-bound
osmnx); True = ThreadPoolExecutor (I/O-bound ORS, and test isolation).
**Rationale:** One flag, zero architectural change, covers both real and test use cases.
**Alternatives rejected:** Always threads — loses GIL bypass for osmnx. Separate
functions — more surface area for no benefit.

## 2026-05-06: Static viz uses contextily as optional dependency
**Context:** contextily adds ~50MB of tile-fetching machinery. Not all users need it.
**Decision:** Optional import in _add_basemap; silently falls back to grey background.
**Rationale:** Graceful degradation > hard dependency. Basemap is cosmetic, not functional.
**Alternatives rejected:** Required dependency — too heavy for cluster nodes without internet.

## 2026-05-06: Geocoding via Nominatim (no API key)
**Context:** Users specify locations by name ("Kenyatta National Hospital"), not
coordinates. Nominatim is free, global, and requires no key.
**Decision:** Nominatim via requests; rate-limited to 1 req/s (usage policy);
results cached in-memory by default, optionally on-disk via IDD_GEOCODE_CACHE_DIR.
**Rationale:** Zero friction for proof-of-principle runs. Disk cache makes batch
use practical.
**Alternatives rejected:** Google Places API — requires key, adds cost. geopy —
adds dependency for thin wrapper over the same endpoint.

## 2026-05-06: TravelTimeRequest.from_place() classmethod
**Context:** Most callers think in place names, not coordinates.
**Decision:** from_place(name, mode, *, ...) classmethod that geocodes then
constructs the request. Raises GeocodingError on failure.
**Rationale:** Single call instead of two. Keeps the main constructor clean
(still requires lat/lon directly — no implicit geocoding side effects).
**Alternatives rejected:** Implicit geocoding in __post_init__ if a string is
passed — too magic, breaks type annotations.

## 2026-05-14: TravelTimeSurface uses float32 + Affine + JSON sidecar
**Context:** Need a raster format that is self-describing, round-trips cleanly,
and is readable by R, QGIS, and Python without a custom reader.
**Decision:** GeoTIFF (float32, LZW compressed, nodata=-9999) + JSON sidecar
for backend/destination/metadata provenance.
**Rationale:** GeoTIFF is universal; sidecar avoids TIFF tag complexity.
**Alternatives rejected:** NetCDF — heavier dependency; zarr — overkill for
single-surface outputs.

## 2026-05-14: OsmnxSurfaceRouter uses KD-tree nearest-node assignment
**Context:** After Dijkstra we have travel times at sparse network nodes.
Need to interpolate to a dense regular grid (40–100 m).
**Decision:** scipy cKDTree over node (lat,lon) — each grid cell gets the
travel time of its nearest street node.
**Rationale:** Fast (O(n log n) build, O(1) query); preserves discontinuities
at barriers (rivers, walls) better than smooth interpolation; correct for road
networks where travel time is fundamentally node-based.
**Alternatives rejected:** IDW interpolation — artificially smooths across
impassable barriers. Rasterising edges directly — much more complex.

## 2026-05-14: FrictionSurfaceRouter uses MCP_Geometric not MCP
**Context:** Plain MCP sums cell costs without accounting for move distance.
Diagonal moves (√2 longer) would be undercosted.
**Decision:** MCP_Geometric with sampling=(dy_m, dx_m) in metres so the
cumulative cost is in minutes and diagonal moves are correctly weighted.
**Rationale:** Matches the MAP/Weiss/Watmough methodology exactly.
**Alternatives rejected:** Plain MCP — systematically underestimates diagonal
paths, producing artifacts at 45° in uniform-friction areas.

## 2026-05-14: EnsembleSurfaceRouter nan_policy defaults to "propagate"
**Context:** With "ignore", a backend that returns all-NaN (failed download,
out-of-extent) silently contributes nothing, making the ensemble look better
than it is.
**Decision:** Default "propagate" — NaN in any backend makes the pixel NaN.
"ignore" available explicitly when complementary partial coverage is intended.
**Rationale:** Fail loudly. A NaN region in the ensemble is a signal that
something is wrong with a backend, not a value to silently skip.
