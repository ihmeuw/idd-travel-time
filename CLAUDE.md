# idd-travel-time — repo rules

## Environment scheme: venv-only (declared 2026-08-09)

This repo runs the **pure-uv scheme**: a project `.venv` at the repo root,
uv-managed interpreter, uv-managed dependencies. One scheme per repo, never
mixed.

- **Expected:** `./.venv` exists and is the only environment for this repo.
  Its interpreter is a uv-managed standalone CPython (`uv python install`),
  never a conda python. All code runs via `.venv/bin/python` — by absolute
  path in Slurm task commands, no activation required.
- **Forbidden:** conda hosting of any kind for this repo — no conda env named
  after the repo, no `environment.yml`, no `UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX`
  idiom. Those belong to the retired Poetry/conda schemes.
- **Session start:** do NOT activate a conda env (this overrides the global
  "activate the matching conda environment" rule — there is no matching env).
  Use `.venv/bin/python` directly.
- **Sync idiom:** `uv sync --inexact` day-to-day; plain `uv sync` (exact) for
  clean rebuilds, vetted first with `uv sync --dry-run`. `--all-extras` to get
  the full dev surface including the `basemaps` extra (contextily).
- **Jobmon (since 2026-10-21):** the global 40 m friction-surface fan-out runs
  through jobmon via idd-tools. Declared as the optional `jobmon` extra
  (`idd-tools[jobmon]`) with the ihme `[[tool.uv.index]]` block and the
  `jobmon_installer_ihme` source pin, per STANDARDS §Environment. Cluster
  sync idiom: `uv sync --extra jobmon`; CI syncs without the extra. Before
  2026-10-21 this repo declared nothing jobmon-related.
- Interpreter downloads are deliberate only: the user-level uv policy is
  `python-downloads = "manual"` — a plain `uv venv`/`uv sync` must never
  trigger a CPython download.

## jobmon via idd-tools: the contract (installed 2026-10-23 by /jobmon-contract)

These sentences bind every session in this repo. Code already in the repo
is not a precedent that overrides them: if an existing worker or launcher
breaks a line below, it is a defect to report, not a pattern to copy.

- **Read first, every time.** Before writing or changing any launcher or
  worker, read the idd-tools README end to end and the docstrings of
  `idd_tools.jobmon.submit`, `atomic_io`, `task_logger`, `resources`,
  `resource_audit`, `run_registry`, `calibrate` and `partition`, and list in
  the plan which of their calls the code will use. Reading in a previous
  session does not count.
- **Every worker:** runs inside `AtomicRegistry(output_dir, task_id=...,
  task_template=...)`, writes outputs only through `atomic_write` /
  `atomic_write_parquet(..., registry=r)`, decides "already done" only with
  `subtask_skip`, takes `--task-id {task_id}` from the template, and emits
  `TaskProgressLogger.entry(...)` and `.report()` so a `REPORT:` line exists
  for sizing. A worker with its own temp-file writes or its own done-check
  is wrong even when it works.
- **Every launcher:** builds a `TaskManifest` and submits with
  `submit_with_manifest`; bundles work so every task clears the >=5 min
  sizing floor (`calibrate_bundle_size`, `pack_to_target` /
  `rectangular_partition`); sizes resources per task from measured evidence
  (probe -> `audit_run` -> `ResourcePredictor` or
  `fit_from_probe_history`), never one static ask across tasks of unequal
  size; runs `max_resource_check` before submitting; confirms with
  `probe_only` on the exact code that will run at scale; calls
  `register_run` and `register_probe`; prints the jobmon GUI URL.
- **After every run:** `audit_run` + `policy_metrics` + `sizing_floor_report`.
  The target is well-allocated (actual 50-80% of asked) on both memory and
  runtime. Over-asking is not free: it is a failure that costs this
  account's priority. An over- or under-allocated run is sized again before
  the next submission; "it finished" is not a pass.
- **Reruns resume, never repeat:** done work is skipped through the
  registry and `subtask_skip`; nothing that exists on disk is recomputed or
  refetched; a cancelled run loses nothing but the tasks in flight.
- **A change to the worker or launcher after the probe re-probes.** The
  full run goes out only on code a probe has run.
