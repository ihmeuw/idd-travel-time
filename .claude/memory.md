# Session memory
Updated: 2026-05-14

## Current task
Packaging final repo for GitHub push and Claude Project setup.

## Context / why
idd-travel-time is an IHME research library for travel-time isochrones and
accessibility surfaces in LMICs. Goal: map geographic access to health facilities
at 40–100 m resolution globally, supporting informal transit cities.

## Where we are
Code complete with 348 tests (94% coverage). Two main layers:
1. isochrone/ — polygon isochrones (OsmnxRouter, ORS, parallel batch)
2. access/ — raster travel-time surfaces (OsmnxSurfaceRouter via Dijkstra,
   FrictionSurfaceRouter via MCP_Geometric, EnsembleSurfaceRouter)

Not yet validated on live data — notebook 10 needs a real-city smoke test.

## Next steps
1. Smoke test notebook 10 on a real city
2. nearest_facility_surface() — multi-source, O(grid)
3. Tile-aware computation — 8-neighbour border load
4. Population raster integration (WorldPop)
5. SLURM cluster batch entry point

## Resume prompt
idd-travel-time (ihmeuw/idd-travel-time) is a Python library for travel-time
isochrones and accessibility surfaces. It has isochrone/ (polygon isochrones)
and access/ (raster surfaces). SurfaceRouter is the abstract interface;
OsmnxSurfaceRouter uses Dijkstra + KD-tree; FrictionSurfaceRouter uses
skimage MCP_Geometric. EnsembleSurfaceRouter combines backends. 348 tests,
94% coverage. Next task is nearest_facility_surface() using multi-source MCP.
See .claude/STATUS.md and .claude/DECISIONS.md for full context.
