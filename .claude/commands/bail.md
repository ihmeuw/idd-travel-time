# /bail — Emergency session close

Trigger the "save and go" protocol from CLAUDE.md. I am leaving right now.

1. Ask at most 1–2 quick disambiguating questions where you genuinely cannot make a judgment call
2. Within 2 minutes, write draft files for everything that changed:
   - `.claude/STATUS_draft.md`
   - `.claude/DECISIONS_draft.md` (if decisions were made)
   - `.claude/DEAD_ENDS_draft.md` (if approaches were abandoned)
3. Overwrite `.claude/memory.md` with a populated snapshot of current state — fill in every section (current task, context, where we are, next steps, resume prompt) with real content. Do NOT reset memory.md to the blank template.
4. Each draft must include a Claude notes block at the top:

```
<!-- DRAFT — generated YYYY-MM-DD HH:MM without review -->
<!-- Claude notes:
  - What was happening when this was written
  - Anything uncertain or that needs Bobby's judgment
  - Why specific entries were categorized the way they were
  - Any open questions that should be resolved at next session start
-->
```

5. Self-check: explicitly account for each of the four files. For each of memory.md, STATUS_draft.md, DECISIONS_draft.md, DEAD_ENDS_draft.md, state either "written this run" or "intentionally skipped because <reason>" (e.g., no decisions made, no approaches abandoned). If any item is unaccounted for, redo it before reporting done.
6. Confirm: "Drafts written to .claude/. Safe to close."
