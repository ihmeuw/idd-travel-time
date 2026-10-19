# idd-travel-time

Travel-time isochrones from arbitrary origin points — globally, including cities
with informal transit and no GTFS coverage.

---

## What it does

Given an origin location, a transport mode, a time-of-day context, and a duration,
`idd-travel-time` produces a map showing every place reachable from that origin
within the specified time budget:

- **Isochrone polygon** — shaded reachable area
- **Road network** — reachable edges
- **POI scatter** — named places within the isochrone
- **Heatmap** — reachability density

It is designed to work in any city worldwide, including cities where no GTFS feed
exists and transit operates informally (Lagos danfo, Nairobi matatu, Kampala
boda-boda, Jakarta bajaj/ojek).

---

## Install

```bash
pip install idd-travel-time
```

Or in development mode:

```bash
git clone https://github.com/ihmeuw/idd-travel-time
cd idd-travel-time
uv venv --python 3.12        # .venv on the uv-managed CPython
uv sync --all-extras         # runtime + dev + basemaps extra
cp paths.yaml.example paths.yaml   # gitignored; set project_output_root for this machine
```

`idd-tools` is a git dependency (`ssh://git@github.com/ihmeuw/idd-tools.git`,
pinned to `main` in `uv.lock`), so `uv sync` needs read access to that repo.

### Where outputs go

`project_output_root` (from `paths.yaml`) holds one directory per *node*:
`maps/` (rendered maps) and `results/` (result parquet directories, surfaces,
GeoTIFFs). Code writes only to a node's `working/` slot, obtained from
`idd_travel_time.paths.working_dir("maps")` / `working_dir("results")`, or to
`scratch/<label>/` for throwaway runs. A snapshot is made only by hand:

```bash
idd-versions <project_output_root>/maps freeze "why this snapshot exists"
```

(`idd-versions` comes with idd-tools; `status`, `promote`, `label`, `gc` are
the other subcommands.)

---

## Quickstart

```python
from idd_travel_time.constants import TransportMode
from idd_travel_time.models import TravelTimeRequest
from idd_travel_time.isochrone import compute
from idd_travel_time import viz

# 30-minute walking isochrone from Kenyatta National Hospital, Nairobi
request = TravelTimeRequest(
    origin_lat=-1.3006,
    origin_lon=36.8073,
    mode=TransportMode.WALKING,
    duration_minutes=30,
)

result = compute(request)        # auto-selects router
m = viz.render(result)           # returns a folium.Map
viz.save(result, "map.html")     # save to HTML
```

---

## Transport modes

| Mode | Notes |
|---|---|
| `walking` | Pedestrian network |
| `cycling` | Cycleable roads and paths |
| `driving` | Road network, speed scales with time context |
| `transit` | Requires a `TransitProfile` — see below |

---

## Transit model

Transit is **not schedule-exact**. Instead of GTFS, it uses *typical-speed
assumptions* parameterised by time-of-day context. This is intentional — it
enables global coverage including cities where no schedule data exists.

### Named presets

```python
from idd_travel_time.models import TransitProfile

# Use a preset as-is
profile = TransitProfile.from_preset("informal_minibus")

# Override one parameter
profile = TransitProfile.from_preset(
    "informal_minibus",
    typical_speed_kph=(10.0, 18.0),   # rush-hour Lagos
)

# Fully custom
from idd_travel_time.constants import TransitCoverage, TimeContext
profile = TransitProfile(
    typical_speed_kph=(20.0, 30.0),
    coverage=TransitCoverage.ROAD_NETWORK,
    time_context=TimeContext.OFF_PEAK,
    mode_name="Electric tuk-tuk pilot",
)
```

| Preset | Typical use | Speed range (kph) | Coverage |
|---|---|---|---|
| `formal_bus` | Developed-world fixed-route bus | 15–30 | fixed routes |
| `informal_minibus` | Sub-Saharan Africa / SE Asia danfo/matatu | 15–25 | road network |
| `mototaxi` | Latin America / West Africa okada/boda-boda | 20–35 | road network |
| `metro` | Cities with subway or light rail | 30–60 | fixed routes |
| `tuk_tuk` | South / SE Asia auto-rickshaw/bajaj | 15–25 | road network |

---

## Routing backends

Backends are plug-and-play via an abstract `Router` interface. Swapping from
free to paid is a one-line change.

```python
from idd_travel_time.routers import OsmnxRouter, OpenRouteServiceRouter
from idd_travel_time.isochrone import compute

# Local, no API key (default when ORS_API_KEY is not set)
result = compute(request, router=OsmnxRouter())

# ORS free tier (set ORS_API_KEY env var)
result = compute(request, router=OpenRouteServiceRouter())

# Auto-select: ORS if ORS_API_KEY is set, else osmnx
result = compute(request)
```

| Backend | Key required | Notes |
|---|---|---|
| `OsmnxRouter` | No | Downloads graph from Overpass; good for batch/cluster |
| `OpenRouteServiceRouter` | Yes (`ORS_API_KEY`) | REST API; 500 req/day free tier |
| `GoogleMapsRouter` | Yes (`GOOGLE_MAPS_API_KEY`) | Stub — not yet implemented |

---

## Batch computation

```python
from idd_travel_time.isochrone import compute_batch
from idd_travel_time.routers import OsmnxRouter

results = compute_batch(requests, router=OsmnxRouter())
```

All requests share a single router instance. For parallel execution, manage
concurrency externally.

---

## CLI

```bash
# Walking isochrone — no --output: written to <maps node>/working/<result_id>.html
run-isochrone \
    --lat -1.3006 --lon 36.8073 \
    --mode walking \
    --duration 30

# Same, as a throwaway run under <maps node>/scratch/probe/
run-isochrone --lat -1.3006 --lon 36.8073 --mode walking --scratch probe

# Informal transit, rush hour, specific layers
run-isochrone \
    --lat 6.4541 --lon 3.3947 \
    --mode transit \
    --transit-preset informal_minibus \
    --time-context rush_hour \
    --duration 45 \
    --layers isochrone heatmap \
    --output lagos_transit.html
```

```
Options:
  --lat FLOAT                     Origin latitude (WGS-84)  [required]
  --lon FLOAT                     Origin longitude (WGS-84)  [required]
  --mode [walking|cycling|driving|transit]  [required]
  --duration INTEGER              Duration in minutes  [default: 30]
  --time-context [rush_hour|off_peak|weekend|night]  [default: off_peak]
  --transit-preset TEXT           Required when --mode=transit
  --router [auto|ors|osmnx]       [default: auto]
  --layers TEXT                   Repeat for multiple layers  [default: all]
  --output PATH                   Output HTML file  [default: <maps node>/working/<result_id>.html]
  --scratch LABEL                 Write to <maps node>/scratch/LABEL/ instead (never frozen)
  --freeze / --current            On success, freeze working/ (and promote with --current);
                                  needs --description
  --label, --tag, --allow-dirty, --clean   idd-versions launcher flags (idd-tools)
```

The node flags apply only to the default location; with `--output` the map is
written exactly there and nothing is frozen.

---

## Visualisation layers

```python
from idd_travel_time.constants import VizLayer
from idd_travel_time import viz

# All available layers
m = viz.render(result)

# Subset
m = viz.render(result, layers=[VizLayer.ISOCHRONE, VizLayer.HEATMAP])

# Save
viz.save(result, "map.html", layers=[VizLayer.ISOCHRONE])
```

| Layer | Description | Always available? |
|---|---|---|
| `ISOCHRONE` | Reachable-area polygon | Yes |
| `NETWORK` | Reachable road edges | When router produces one |
| `POI_SCATTER` | Named places within isochrone | When router fetches them |
| `HEATMAP` | Reachability density heatmap | Yes (derived from polygon) |

---

## Notebooks

Vignettes live in `notebooks/vignettes/`; `notebooks/scratch/` is for
exploration and is never authoritative.

| Notebook | Description |
|---|---|
| `notebooks/vignettes/01_walkthrough.ipynb` | Basic usage: walking isochrone, all four layers, save to HTML |
| `notebooks/vignettes/02_global_cities.ipynb` | Informal transit in Lagos, Nairobi, Kampala, Jakarta; batch computation; off-peak vs rush-hour comparison |

---

## Development

Local dev uses a project `.venv` with a uv-managed interpreter — no conda
anywhere in the Python story. uv owns the interpreter and every package
(including installing idd-travel-time itself editable — no `PYTHONPATH=src`).
One-time setup:

```bash
uv venv --python 3.12        # .venv on the uv-managed CPython
uv sync --all-extras         # runtime + dev + basemaps extra
.venv/bin/pre-commit install
```

Run code via `.venv/bin/python` by absolute path (Slurm scripts do exactly
this — no activation of any kind); `source .venv/bin/activate` is optional
for interactive shells. Tests: `.venv/bin/python -m pytest`.

### What has not been decided yet

- ORS API key management (env var vs config file)
- Caching strategy for repeat queries on the same origin/duration
- Batch/cluster entry point design (single origin vs. many origins in parallel)
- Output file format for cluster runs (GeoParquet? GeoJSON?)
- Non-folium viz backends (matplotlib static maps)
- Schedule-exact transit routing via r5py (deferred — requires GTFS)

---

## Repo structure

```
src/idd_travel_time/
├── constants.py            enumerations, column names, numeric defaults
├── paths.py                paths.yaml loader; node slots working_dir()/scratch_dir() via idd-tools
├── models.py               TravelTimeRequest, TravelTimeResult, TransitProfile
├── profiles/
│   └── presets.py          five named TransitProfile presets + registry
├── routers/
│   ├── base.py             abstract Router interface
│   ├── openroute.py        ORS backend
│   ├── osmnx_router.py     local Overpass fallback
│   └── google.py           paid stub
├── isochrone/
│   └── compute.py          compute() and compute_batch() — dispatch layer
├── viz/
│   └── folium_viz.py       render() and save() — four-layer folium map
└── run_isochrone.py        CLI entry point (run-isochrone)
```

---

## Parallel batch computation

```python
from idd_travel_time.isochrone import compute_parallel
from idd_travel_time.routers import OsmnxRouter
from idd_travel_time.io import GraphCache

cache = GraphCache("/data/graph_cache")

results = compute_parallel(
    requests,
    router_factory=lambda: OsmnxRouter(graph_cache=cache),
    max_workers=4,
    output_dir="/data/results",   # write each result to disk as it completes
    skip_existing=True,            # resume interrupted runs automatically
    on_error="warn",               # insert None for failures, continue
)
```

`compute_parallel` uses `ProcessPoolExecutor` by default (bypasses the GIL,
safe for CPU-bound routing). Pass `use_threads=True` for I/O-bound backends
like ORS. Each worker gets its own router instance from `router_factory`.

---

## GeoParquet persistence

```python
from idd_travel_time.io import write_result, read_result, result_id

# Save
result_dir = write_result(result, "/data/results")
# → /data/results/n1.286389_36.817223_walking_30min_off_peak/

# Convenience method on TravelTimeResult
result_dir = result.to_parquet("/data/results")

# Load
result = read_result(result_dir)
result = TravelTimeResult.from_parquet(result_dir)

# The result_id is deterministic — re-running the same request overwrites
result_id(request)   # e.g. "n1.286389_36.817223_walking_30min_off_peak"
```

Layout on disk:

```
{result_id}/
├── meta.json          request provenance + schema version
├── isochrone.parquet  always present
├── network.parquet    present when router produced network edges
└── poi.parquet        present when router fetched POIs
```

---

## Graph caching

Caches osmnx Overpass downloads to disk so repeat runs are instant:

```python
from idd_travel_time.io import GraphCache
from idd_travel_time.routers import OsmnxRouter

cache = GraphCache("/data/graph_cache")
router = OsmnxRouter(graph_cache=cache)

# Or via environment variable (auto-picked up by OsmnxRouter):
# export IDD_GRAPH_CACHE_DIR=/data/graph_cache
```

Cache management:

```python
cache.has(lat, lon, radius_m)        # True if cached
cache.invalidate(lat, lon, radius_m) # delete one entry
cache.clear()                        # delete all entries
```

---

## Static maps (matplotlib)

For reports, papers, or batch thumbnail generation:

```python
from idd_travel_time import viz

# Returns a matplotlib Figure
fig = viz.render_static(result)
fig = viz.render_static(result, layers=[VizLayer.ISOCHRONE, VizLayer.HEATMAP])
fig = viz.render_static(result, figsize=(8, 8), dpi=200)

# Save directly to PNG / PDF / SVG
viz.save_static(result, "map.png")
viz.save_static(result, "map.pdf", dpi=300)
```

Requires `contextily` for tiled basemaps (optional — degrades gracefully to
a grey background if not installed). It ships as the `basemaps` extra:

```bash
pip install "idd-travel-time[basemaps]"   # consumers
uv sync --all-extras                       # development .venv
```

---

## Location lookup (geocoding)

Resolve place names and addresses to coordinates without leaving the library:

```python
from idd_travel_time.locations import geocode, geocode_many
from idd_travel_time.models import TravelTimeRequest
from idd_travel_time.constants import TransportMode

# Single place
lat, lon = geocode("Kenyatta National Hospital, Nairobi")
lat, lon = geocode("Lagos Island", country_hint="NG")  # bias by country

# Multiple places — None for failed lookups (on_error="warn")
coords = geocode_many(
    ["Lagos Island", "Abuja Central", "Kano, Nigeria"],
    country_hint="NG",
    on_error="warn",
)

# Build a request directly from a place name
req = TravelTimeRequest.from_place(
    "Tafawa Balewa Square, Lagos",
    TransportMode.TRANSIT,
    country_hint="NG",
    transit_profile=TransitProfile.from_preset("informal_minibus"),
    duration_minutes=30,
)
```

**Cache:** results are cached in memory by default. Set `IDD_GEOCODE_CACHE_DIR`
to persist to disk across sessions — essential for batch runs:

```bash
export IDD_GEOCODE_CACHE_DIR=/data/geocode_cache
```

**CLI — place name instead of coordinates:**

```bash
run-isochrone \
    --place "Kenyatta National Hospital, Nairobi" \
    --mode walking --duration 20 \
    --output knh_walking.html

run-isochrone \
    --place "Lagos Island" --country-hint NG \
    --mode transit --transit-preset informal_minibus \
    --output lagos_transit.html
```

> Uses OpenStreetMap Nominatim (free, no key, 1 req/s rate limit).
> For >100 unique locations consider hosting a local Nominatim instance.

---

## Travel-time surfaces (access layer)

A **travel-time surface** answers: *"From every point in this area, how long
does it take to reach this facility?"* The output is a raster where each pixel
value is travel time in minutes.

```python
from idd_travel_time.access import (
    BoundingBox, OsmnxSurfaceRouter, FrictionSurfaceRouter,
    EnsembleSurfaceRouter, TravelTimeSurface,
    run_benchmark, make_synthetic_ground_truth,
)
from idd_travel_time.constants import TransportMode

# Define the area and destination
bbox = BoundingBox(west=36.72, south=-1.37, east=36.90, north=-1.22)
destination = (-1.3006, 36.8073)   # Kenyatta National Hospital, Nairobi

# Compute a surface
router = OsmnxSurfaceRouter()
surface = router.compute_surface(
    destination=destination,
    bbox=bbox,
    resolution_deg=0.01,    # ~1.1 km; use 0.001 for ~111 m
    mode=TransportMode.WALKING,
)

print(surface)
# TravelTimeSurface(15×18 px, 0.0100°/px, backend='osmnx', mean=24.3 min, nan=3%)

# Threshold: who is within 30 minutes?
within_30 = surface.threshold(30.0)          # bool array
print(f"{surface.coverage_at(30):.1%} of area within 30 min")

# Save to GeoTIFF
surface.to_geotiff("nairobi_walking.tif")

# Reload
surface2 = TravelTimeSurface.from_geotiff("nairobi_walking.tif")
```

### Surface routing backends

| Backend | Algorithm | Best for |
|---|---|---|
| `OsmnxSurfaceRouter` | Dijkstra on OSM street graph → KD-tree interpolation | Urban areas, road-network fidelity |
| `FrictionSurfaceRouter` | `skimage.MCP_Geometric` on friction raster | Off-road walking, global/rural coverage |

Both implement `SurfaceRouter` — swapping is one line.

### Ensemble and uncertainty

```python
ens = EnsembleSurfaceRouter(
    [OsmnxSurfaceRouter(), FrictionSurfaceRouter(speed_kph_override=5.0)],
    strategy="mean",           # or "median", "min", "max"
    weights=[0.6, 0.4],
    nan_policy="propagate",    # or "ignore"
)
surface = ens.compute_surface(destination, bbox, 0.01, TransportMode.WALKING)
uncertainty = ens.compute_uncertainty(destination, bbox, 0.01, TransportMode.WALKING)
```

### Benchmarking against ground truth

```python
from idd_travel_time.access import run_benchmark, make_synthetic_ground_truth

# Generate synthetic GT (or supply real GPS-tracked observations)
ref = OsmnxSurfaceRouter().compute_surface(destination, bbox, 0.01, TransportMode.WALKING)
ground_truth = make_synthetic_ground_truth(ref, n_points=200, noise_std_min=3.0)

result = run_benchmark(
    routers={"osmnx": OsmnxSurfaceRouter(),
             "friction": FrictionSurfaceRouter(speed_kph_override=5.0)},
    destination=destination,
    bbox=bbox,
    resolution_deg=0.01,
    mode=TransportMode.WALKING,
    ground_truth=ground_truth,
    test_fraction=0.3,
)
print(result.summary())
# result.scores     — MAE, RMSE, R², coverage_error per backend × split
# result.surfaces   — dict of computed surfaces
# result.pairwise   — pixel-wise comparison between all backend pairs
```

### Notebooks

| Notebook | Description |
|---|---|
| `10_travel_time_surface.ipynb` | Single destination → raster, threshold, GeoTIFF, folium map |
| `11_surface_backends.ipynb` | OsmnxSurfaceRouter vs FrictionSurfaceRouter, writing custom backends |
| `12_ensemble_surfaces.ipynb` | Weighted ensembles, uncertainty maps, NaN policy |
| `13_surface_metrics.ipynb` | Ground truth comparison, in-sample/OOS metrics, residual maps |
