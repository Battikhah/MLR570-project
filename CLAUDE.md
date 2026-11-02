# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- `AGENTS.md` (imported above) is the single source of truth shared with Codex; edit rules there, not here.
- When dispatched by Orca, follow the worker preamble and the `orchestration` skill (`ask`, `worker_done`).
- Use web search/fetch only for public scholarly sources; record every source in `literature/sources.yaml`.
- PDFs: `pdftotext -layout` is available. Extract to a temp dir outside the repo, or to `literature/pdfs/` (git-ignored).
