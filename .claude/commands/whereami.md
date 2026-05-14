# /whereami — Reorientation

Before normal orientation, run two pre-flight checks: the proposed-rules
inbox and the weekly-scan state.

## Step 1: Check the proposed-rules inbox

```bash
ls ~/.claude/proposed_rules/ 2>/dev/null
```

If the directory contains one or more `.md` files (any name, typically dated
like `2026-05-10.md`):

> "There are N proposed rule(s) pending review from the weekly scan.
>  Review now, defer to end of session, or skip this session?"

- **Review now**: call `/formalize-rule review` and step through each draft
  (accept / edit / reject with reason). When finished, proceed to Step 2.
- **Defer**: do nothing. The drafts stay in the inbox and will surface again
  on the next `/whereami` invocation (this or any other repo).
- **Skip this session**: remember in-conversation that Bobby said skip; do
  not surface again until next session. Do not write a persistent marker.

If the directory is empty or does not exist, skip Step 1 silently and proceed.

## Step 2: Check the weekly-scan state

Read `~/.claude/last_scan_state.json` (created by `/formalize-rule` after every
scan). Compute the current ISO week via `date -u +"%G-W%V"`.

If the file does not exist, OR `last_scan_iso_week` is **not** the current
ISO week, prompt:

> "Weekly /formalize-rule scan hasn't fired yet this week (last scan:
>  <YYYY-MM-DD>, or never). Run now, defer, or skip this session?"

- **Run now**: invoke `/formalize-rule scan` (the **interactive** manual mode,
  not `--automatic`). After it completes, the skill writes the state file.
  Then proceed to Step 3.
- **Defer**: do nothing. Ask again on the next `/whereami` this session and
  future sessions until a scan runs.
- **Skip this session**: same in-conversation note as Step 1 — don't ask
  again this session, no persistent marker.

**Never auto-run the scan.** Bobby is always the trigger. The nag is the
mechanism; the scan is not automatic.

If `last_scan_iso_week` IS the current ISO week, skip Step 2 silently.

## Step 3: Orient

Read `.claude/memory.md` and `.claude/STATUS.md` silently, then give a single
focused paragraph covering:
- Which project and branch we are on
- What we were doing when we last left off
- Where exactly we stopped
- What the immediate next step is

Do not summarize the full STATUS.md. Do not list recent steps. Just orient —
one paragraph, enough to decide whether to continue or ask a clarifying
question before starting work.

If `.claude/memory.md` is blank or contains only the template, say so and run
the first-run initialization flow from CLAUDE.md instead.
