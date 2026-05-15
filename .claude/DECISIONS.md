# Decisions log

<!-- Append-only. Never delete or overwrite entries. -->

## 2026-05-14: Canonical outputs go to /mnt/team/idd/pub/idd-travel-time/, not /tmp or source tree
**Decision:** Library examples and notebooks write to
`/mnt/team/idd/pub/idd-travel-time/{maps,results}/`. Never `/tmp`, never the
source tree. `.gitignore` no longer hides outputs/data — they shouldn't be
in the source tree to begin with.
**Why:** STANDARDS §Output management was being silently violated in six
places left over from Claude Project (3 CLI docstring examples, 3 notebook
paths). Gitignoring `outputs/` and `*.parquet` in the repo documented the
wrong pattern.
**Revisit if:** A genuine use case requires writing outputs from the
library to a location other than `/mnt/team/idd/pub/<project>/` — at which
point update STANDARDS, not just this repo.

## 2026-05-14: ipykernel is a required dep for every Python idd-* repo
**Decision:** `[project.dependencies]` must include
`ipykernel (>=7.0.0,<8.0.0)` for every Python idd-* repo. Promoted to
canonical: `~/.claude/STANDARDS.md` §Environment and the scaffold-repo
pyproject template.
**Why:** Without it the conda env isn't selectable as a Jupyter kernel —
notebooks fail with "kernel not found" or silently run against base,
defeating the one-env-per-repo discipline.
**Revisit if:** A repo legitimately has no notebooks — note the deviation
in that repo's DECISIONS.md.

## 2026-05-14: ORS smoke test passed; OSMnx/ORS isochrone shape divergence is real
**Decision:** With an ORS API key wired up via `.env`,
`OpenRouteServiceRouter` is integration-verified. For KNH 30-min walking,
ORS returns a single 11.3 km² Polygon containing the origin; OSMnx returns
a 5.5 km² MultiPolygon that does *not* contain the origin.
**Why:** ORS uses its native isochrone API (proper alpha-shape on the
road network); OSMnx in this library uses 50 m node-buffer-union,
producing under-approximated polygons with disconnected pockets. The ~2×
area gap and origin-containment difference are substantive, not bugs.
**Revisit if:** Users surface confusion about which is "right" — at that
point the library should pick a default and explain the trade-off, or
expose both via `viz.compare`.

## 2026-05-14: Backends require live-data smoke tests, not just mocked unit tests
**Decision:** Every router / surface-router backend (Osmnx, ORS, MAP friction,
ensemble) must have a live-data smoke test exercised before any "this works"
claim. Mocked unit tests are necessary but not sufficient.
**Why:** During the Nairobi smoke test (notebook 10) on 2026-05-14, three
real bugs surfaced that all 348 mocked tests had passed cleanly:
(1) `ox.graph_from_bbox` 1.x → 2.x signature change (bbox-tuple, not
north/south/east/west kwargs); (2) `ox.nearest_nodes` receiving WGS-84
coordinates against a UTM-projected graph (silent wrong-answer with
travel times ~3× too high); (3) missing `scikit-learn` dep that osmnx
2.x now requires for haversine `nearest_nodes`. None of these were
catchable by mocking — they only emerge when the backend actually talks
to the real upstream library / live API.
**Revisit if:** Library scope shifts so that all backends are mock-only
(unlikely — backends only exist to talk to live external sources).

## 2026-05-14: Notebook 10 (Nairobi walking) is the proof-of-principle gate
**Decision:** `notebooks/10_travel_time_surface.ipynb` is the canonical
live-data smoke test for `OsmnxSurfaceRouter` and, by extension, the
access layer's Dijkstra + KD-tree + raster pipeline.
**Why:** A real bbox, a real destination (Kenyatta National Hospital), a
real Overpass download, and concrete sanity-checkable outputs (min near 0,
max ~ half-diagonal × meander / walking-speed). Anything wrong with edge
weights, projection handling, or grid assignment shows up as obviously
wrong numbers.
**Revisit if:** A faster or cheaper integration test replaces it, or the
access layer is rewritten such that this notebook no longer covers the
relevant code paths.

## 2026-05-14: ORS smoke test parked — no API key
**Decision:** `OpenRouteServiceRouter` is not currently verifiable against
live ORS — `ORS_API_KEY` is not set in this environment.
**Why:** ORS requires a free-tier key (500 req/day at openrouteservice.org)
or paid tier. No key currently provisioned for this repo / user.
**Revisit if:** A key is obtained, or ORS is removed as a supported
backend. Until then, treat `OpenRouteServiceRouter` as "implemented per
spec but not integration-verified."

## 2026-05-14: Friction-GeoTIFF smoke test parked — no source friction raster
**Decision:** `FrictionSurfaceRouter`'s GeoTIFF path is not currently
exercisable against a real raster. The `speed_kph_override` path is
verified via the Nairobi ensemble run.
**Why:** The IHME cluster's `/snfs1/.../access2/mean/2019_06_10/synoptic/`
holds the Weiss 2018 *travel-time-to-cities* product (output, in minutes),
not the source friction raster (input, in min/m). No MAP friction raster
is currently available on-cluster or in the project tree. The library is
designed to consume a friction raster; consuming an accessibility raster
would be a different operation.
**Revisit if:** A MAP friction raster is downloaded to a known path, or
the team builds its own friction surface from OSM + speed assumptions
(planned).

## 2026-05-14: Origin point not guaranteed inside isochrone polygon
**Decision:** Known limitation: `OsmnxRouter` builds the isochrone polygon
by buffering each reachable road node by `_NODE_BUFFER_DEG ≈ 50 m` and
unioning. If the origin point sits in a 50-m gap between road nodes
(e.g. inside a building), the literal origin lat/lon may fall outside
the returned polygon even though the origin is the start of the
isochrone.
**Why:** The polygon represents the road-network-reachable subset, not
"everything within X metres of road." Surfaced during the 2026-05-14
Nairobi smoke test: KNH (-1.3006, 36.8073) → origin-in-polygon == False
despite a sensible 5.5 km² 30-min walking polygon centred at (-1.298,
36.808).
**Revisit if:** Users start filing this as a bug, or the polygon shape
matters for downstream area calculations. Cheap fix: snap-or-buffer the
origin into the polygon as a post-processing step. Real fix: switch to
alpha-shape with origin included, or use a finer node buffer.

## 2026-05-14: Strict ruff `ALL` keeps research-friendly per-file ignores
**Decision:** `[tool.ruff.lint] select = ["ALL"]` is the policy, but
tests/** carries a broad per-file-ignores list (PLR2004, SLF001, FBT,
SIM117, N806, E741, PD901, RUF003, ERA001, S101, ARG, S105) and the CLI
entry point ignores PLR0913 (click decorators). Notebooks are excluded
from ruff entirely (vignettes are illustrative, not production-strict).
Three "style debate" rules are dropped globally (TC001, TC003, RUF001).
**Why:** Tests, CLIs, and vignettes legitimately violate rules that make
sense in src/. Dropping `select = ALL` for the whole repo loses checks
the src code benefits from; per-file ignores keep the strictness where
it matters and relax it where the rule is genuinely wrong for the
context.
**Revisit if:** A linting rule causes recurring noise across many src
files (consider promoting it to the global ignore list) or a rule
currently ignored is being routinely violated in a way that masks real
bugs.

## 2026-05-14: Mypy strict + targeted overrides for tests and CLI
**Decision:** `strict = true` globally with two `[[tool.mypy.overrides]]`
blocks: `tests.*` relaxes `disallow_untyped_defs / _calls / _decorators
/ incomplete_defs`; `idd_travel_time.run_isochrone` relaxes
`disallow_untyped_decorators` only (click).
**Why:** Tests use mocks, fixtures, and `**kwargs` patterns that don't
type cleanly under strict. The CLI is decorated with `@click.command`
and `@click.option`, whose returned wrappers don't preserve the
underlying function signature — strict-typed-decorators noise without
real value.
**Revisit if:** Click ships properly-preserving decorator type stubs
(would let us drop the CLI override), or test code adopts a more
type-friendly style (would let us drop the tests override).

---


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
