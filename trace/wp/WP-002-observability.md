# WP-002: observability layer, event schema and the DuckDB query layer

Status: proposed. Depends on WP-001. Tier: adds `DuckDB` to the `scripts/` environment (package-manager, `pkg/` branch, owner approval) and a hook entry to `.claude/settings.json` (owner approval).

## Decision to record (as a `D-` record in this package)

Files in git stay the source of truth: trace records, reports, figures with their JSON sidecars, run manifests. Agents never write to a database. The harness writes events as JSONL. One builder loads records, events, GitHub and CI state and sidecars into a local DuckDB file that is regenerated and never committed, and everything that reads numbers (the digest, the state page, cross-run plots) reads them from there through SQL. Alternatives rejected: a server database (nothing to operate, no concurrent writers needed), SQLite (no Parquet or JSON reading, row-oriented), JSON only (aggregations would be reimplemented in every consumer).

## Deliverables

1. **Event schema.** `docs/observability.md` defines one JSONL event per line with `ts` (UTC ISO 8601), `source` (`harness`, `github`, `ci`, `run`), `event` (`session_start`, `session_end`, `tool_summary`, `pr_opened`, `review`, `merge`, `ci_result`, `run_recorded`), `session_id`, `platform` (`local-factory`, `cloud`, `workflow`), `wp` (work-package id when known), `pr`, `commit`, and a free `data` object (for `session_end`: input and output tokens, cost in USD if the harness reports it, tool-call counts by tool, PR URLs mentioned, outcome). Events are append-only; one file per session under an events directory named by the environment variable `PM_EVENTS_DIR` (the factory sets it inside the depot; cloud sessions set it to a directory under the working tree that is ignored by git).
2. **Harness hook.** `scripts/factory/emit-event.sh`: a `SessionStart` and `Stop` hook script that reads the hook JSON on stdin and appends the event; at `Stop` it summarizes the transcript the hook names (token usage, tool counts, PR URLs) so the transcript itself never has to be readable from inside a session. Wiring: the repository's `.claude/settings.json` for cloud sessions, the factory's own settings for local sessions (documented, not in the repository).
3. **Collectors.** `scripts/trace/collect.jl`: pulls pull requests, reviews, merges and check results for the repository with `gh pr list --json` and `gh pr view --json`, and CI test results from the `trace` and `test` jobs, into events of `source = github` and `source = ci`. It runs in local factory sessions and in CI only: cloud sessions cannot run `gh pr`, and `gh api` is denied by the repository settings, so they do not collect. Idempotent: re-running does not duplicate events (key on the GitHub ids).
4. **Builder.** `scripts/trace/observe.jl`: loads `trace/records/`, `trace/build/graph.json` and `coverage.json` from WP-001, every events file, run manifests and figure sidecars (`*.json` next to `*.png` under `scripts/runs/` and `reports/`) into `trace/build/observe.duckdb` with tables `nodes`, `edges`, `events`, `sessions`, `prs`, `checks`, `runs`, `figures` (path, hash, run id, work package, requirement ids, the sidecar scalars as columns). Writes `trace/build/state.json` with the aggregates the state page and the digest use: cost and tokens per work package and per session, review precision (findings confirmed versus refuted, from review events and the follow-up commits), merges per tier, stale frontier, coverage. Exports Parquet snapshots of all tables to a directory given on the command line, for history outside git.
5. **Overseer input.** `.claude/agents/overseer.md` reads `trace/build/state.json` instead of recomputing; the digest cites the numbers with the build timestamp.
6. **Queries.** `scripts/trace/queries.sql`: the named queries the digest and the page use, so that a human can run them in the DuckDB shell on the same file.

## Non-goals

The static state page itself (its data contract is `state.json`; the page is a later package), Managed Agents events, OpenTelemetry export, any change to engine code, storing images or reports in the database.

## Acceptance

- `julia +1.13 --project=scripts scripts/trace/observe.jl` builds the database from the records of WP-001 and a fixture events directory (committed under `test/fixtures/events/`) and writes `state.json`.
- `T-003 events from a session hook are well formed [R-nnn]`: run `emit-event.sh` on fixture hook inputs for `SessionStart` and `Stop` with a fixture transcript; assert the schema.
- `T-004 cost per work package aggregates [R-nnn]`: the fixture holds two sessions on two packages; assert the per-package totals in `state.json`.
- `T-005 figures table links sidecars to requirements [R-nnn]`: a fixture PNG and sidecar pair resolves to its run and requirement ids.
- The `trace` CI job also runs the observe builder on the fixtures.
- `docs/observability.md` is under two screens and names where events live on each platform.

## Budgets

CPU only, seconds. `DuckDB` is the single new dependency, in `scripts/` only.
