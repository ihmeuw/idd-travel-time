# Project status
Updated: 2026-05-14

## Goals
- Library for travel-time isochrones and accessibility surfaces in LMICs.
- Geographic access to health facilities at 40–100 m resolution, globally.
- First-class support for informal-transit cities (no GTFS required).

## Recent steps
- 2026-05-06: Scaffold + isochrone layer + IO/cache + tests.
- 2026-05-14 (morning): Canonical-scaffold alignment.
- 2026-05-14 (afternoon): Env stood up (conda + poetry), pushed to github.com/ihmeuw/idd-travel-time, 348 tests verified at 94% coverage.
- 2026-05-14 (afternoon): Repaired osmnx 2.x compat surfaced by Nairobi smoke test (`graph_from_bbox` signature, WGS-84/UTM `nearest_nodes` bug, scikit-learn dep) — `3fa12dd`.
- 2026-05-14 (afternoon): Whole-repo strict ruff/mypy cleanup with research-friendly per-file ignores — `9c0c1b8`.
- 2026-05-14 (afternoon): All routing backends live-smoke-tested (Osmnx isochrone, Osmnx surface, ORS isochrone, Friction-uniform); only FrictionSurfaceRouter GeoTIFF path remains parked.
- 2026-05-14 (afternoon): Promoted `ipykernel` and Poetry-non-interactive workaround to canonical STANDARDS.
- 2026-05-14 (afternoon, late): Audited Claude-Project repo against STANDARDS §Output management — fixed six `/tmp` example paths and removed the `outputs/` / `data/` / `*.parquet` / `*.csv` lines from `.gitignore`; created canonical `/mnt/team/idd/pub/idd-travel-time/{maps,results}/`. — `9482fd3`.

## Next steps
1. Build a top-down STANDARDS-compliance audit checklist for inherited Claude-Project repos (output paths, scratch use, source-tree-output, naming, embedded examples). Apply to other idd-* repos.
2. `viz.compare([result_a, result_b], labels=...)` helper — overlay multiple results on one folium map with shared bounds, so notebook 01's walking-vs-driving comparison is legible.
3. Real friction-GeoTIFF smoke test — download MAP global friction raster (or wait until team builds its own from OSM + speed assumptions).
4. `nearest_facility_surface()` — multi-source MCP/Dijkstra, O(grid).
5. Snap-or-buffer origin into OsmnxRouter's polygon — small UX fix for the known origin-not-in-polygon limitation.
6. Run end-to-end benchmark (notebook 13) with real / GPS ground truth.

## Parking lot
- Linked side-by-side ipyleaflet panels for true mode comparison (option B; full feature, not today).
- ORS Collaborative-tier upgrade — academic, 5× rate-limit headroom.
- Tile-aware surface computation — 8-neighbour border load.
- Population raster integration (WorldPop).
- SLURM / cluster batch entry point.
- r5py integration for formal-transit cities (deferred — requires GTFS).
- Multi-source `FrictionSurfaceRouter` via MCP `find_costs(starts=[…])`.
- Upstream Poetry interpreter-scan bug (workaround documented; could file upstream).
