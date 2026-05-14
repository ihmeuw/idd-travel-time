# /wrap — Interactive session close

Trigger the "let's wrap up" protocol from CLAUDE.md:

1. Summarize what we did this session (3–5 bullets)
2. Propose updates to STATUS.md, DECISIONS.md, DEAD_ENDS.md — show diffs, wait for approval
3. Write approved updates to the actual files
4. Overwrite memory.md with a populated end-of-session snapshot — fill in every section (current task, context, where we are, next steps, resume prompt) with real content reflecting how the session ended. Do NOT reset memory.md to the blank template; "clean" means tidy and final, not empty.
5. Delete any _draft files if they exist
6. Prompt: "Want me to propose a research-ops update?" — only if something substantive changed that affects the cross-repo view
7. Self-check: explicitly account for each of the four files. For each of memory.md, STATUS.md, DECISIONS.md, DEAD_ENDS.md, state either "updated this run" or "intentionally skipped because <reason>". If any item is unaccounted for, redo it before reporting done.
