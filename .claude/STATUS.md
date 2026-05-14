# Project status
Updated: 2026-05-14

## Goals
- Library for travel-time isochrones and accessibility surfaces in LMICs.
- Geographic access to health facilities at 40–100 m resolution, globally.
- First-class support for informal-transit cities (no GTFS required).

## Recent steps
- 2026-05-06: Repo scaffold — pyproject, CI, pre-commit, src layout.
- 2026-05-06: Isochrone layer — models, transit profiles, routers (OSMnx / ORS / Google stub), `compute_batch`, `compute_parallel`.
- 2026-05-06: IO + cache — GeoParquet result persistence, graph cache with per-key locking, geocoding with optional disk cache.
- 2026-05-14: Access layer — `TravelTimeSurface`, OSMnx + friction-surface backends, `EnsembleSurfaceRouter`, benchmark/metrics modules.
- 2026-05-14: Notebooks 01–13 and 348 tests (94% coverage claimed; not yet Bobby-verified).
- 2026-05-14: Canonical-scaffold alignment — `setup_claude.sh --init`, settings/.gitignore/pre-commit patched, schema migration on `.claude/*.md`.

## Next steps
1. Smoke-test notebook 10 on a real city — the proof-of-principle gate.
2. Run end-to-end benchmark (notebook 13) with real or GPS ground truth.
3. Validate `FrictionSurfaceRouter` against a real MAP friction GeoTIFF.
4. Stand up the conda env and run pytest to verify the 348-test / 94%-coverage claim.
5. Implement `nearest_facility_surface()` — multi-source MCP/Dijkstra, O(grid) not O(N × grid).

## Parking lot
- Tile-aware surface computation — 8-neighbour border load.
- Population raster integration (WorldPop).
- SLURM / cluster batch entry point.
- r5py integration for formal-transit cities (deferred — requires GTFS).
- Output format for cluster runs (GeoParquet? GeoTIFF? Both?).
- Tile border width tuning (depends on actual tile sizes chosen).
- Ground-truth source decision (GPS surveys / OSRM / DHS clusters).
- Multi-source `FrictionSurfaceRouter` via MCP `find_costs(starts=[…])` — enables nearest-facility surfaces.
