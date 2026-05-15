# Session memory
Updated: 2026-05-14 17:55

## Current task
Session wrap. Today's session took the Claude-Project-generated repo from
"untested + no env" to "all backends live-data verified, strict-pre-commit
clean, canonical-aligned, on GitHub, output paths fixed to canonical."

## Context / why
The repo had been generated entirely by Claude Project — never run, no
env, no commits, no live data. Goal: validate the library actually works
against real upstream APIs, bring the codebase to the canonical bar, and
capture the lessons (especially the one about mocked tests missing real
API drift, and the one about not auditing pre-existing code against
STANDARDS).

## Where we are
- 7 commits on main, all pushed to ihmeuw/idd-travel-time.
- Conda env `idd-travel-time` (Python 3.12) stands up cleanly. ipykernel
  in main deps; scikit-learn was added for osmnx 2.x haversine snap.
- All four routing backends live-data verified for the KNH 30-min Nairobi
  query: OsmnxRouter (5.5 km² MultiPolygon), OsmnxSurfaceRouter (270 valid
  pixels, sensible 14–243 min spread), OpenRouteServiceRouter (11.3 km²
  Polygon), FrictionSurfaceRouter via speed-override (exercised in the
  surface smoke test). Only FrictionSurfaceRouter's GeoTIFF path is parked
  — IHME cluster has the accessibility *output* (Weiss 2018), not the
  source friction raster.
- pre-commit clean across whole repo: ruff (with notebooks/ excluded),
  ruff-format, mypy strict (with tests.* and run_isochrone overrides),
  nbstripout, four pre-commit-hooks checks. 348 pytest tests pass at 94%.
- Canonical STANDARDS updated: §Environment now codifies conda+poetry
  pairing, the non-interactive poetry invocation workaround, ipykernel as
  a required main dep. scaffold-repo template aligned.
- Output paths fixed: examples and notebooks write to
  `/mnt/team/idd/pub/idd-travel-time/{maps,results}/`; `.gitignore` no
  longer documents the anti-pattern of expecting outputs in the source
  tree.
- `.env` exists locally with ORS_API_KEY; gitignored.

## Lessons (worth carrying forward)
- When given a Claude-Project-generated repo: do a top-down
  STANDARDS-compliance audit (§Output management, §I/O conventions,
  §Pre-commit, §Environment) as an explicit early step. Bug-fix and
  lint-cleanup passes do *not* catch values-and-conventions violations.
  Six `/tmp` paths sat in plain sight through three reads of the same
  file this session; only caught when Bobby noticed. Logged in
  `~/.claude/projects/-mnt-share-homes-bcreiner-repos-idd-travel-time/corrections/2026-05-14.md`.
- Mocked tests don't substitute for live smoke tests. 348 green mocked
  tests passed cleanly while three osmnx-2.x API-shape bugs sat in the
  code paths.

## Next steps
1. Build a top-down STANDARDS-compliance audit checklist for inherited
   Claude-Project repos. Apply to the other idd-* repos.
2. `viz.compare([result_a, result_b], labels=...)` helper — make notebook
   01's mode-vs-mode comparison eyeball-able.
3. Real friction-GeoTIFF smoke test — download MAP's global friction
   raster or build our own.
4. `nearest_facility_surface()` — multi-source MCP/Dijkstra design.
5. Snap-or-buffer origin into OsmnxRouter's polygon (small UX fix).
6. Apply for ORS Collaborative tier — academic / UW email should qualify;
   5× the daily rate limit.

## Resume prompt
idd-travel-time is at github.com/ihmeuw/idd-travel-time (7 commits on
main), strict-pre-commit-clean, 348 tests / 94% coverage. All four
routing backends are live-data verified for KNH Nairobi queries; only
FrictionSurfaceRouter's GeoTIFF path is parked (no source friction
raster yet). Conda env `idd-travel-time` (Python 3.12) — deps include
scikit-learn and ipykernel. ORS_API_KEY in .env (gitignored). Output
paths are canonical (`/mnt/team/idd/pub/idd-travel-time/{maps,results}/`).
Key session findings: (a) OSMnx and ORS isochrones differ ~2× in area
for the same query because of node-buffer-union vs native alpha-shape;
recorded in DECISIONS. (b) Six `/tmp` violations slipped past my reads
of the same files because I treated docstring/notebook examples as
"background" — explicit STANDARDS-compliance audit is now in the next-
steps list. Next likely task: implement `viz.compare()` so notebooks
can show mode-vs-mode comparisons legibly. See .claude/STATUS.md and
.claude/DECISIONS.md for the full picture.
