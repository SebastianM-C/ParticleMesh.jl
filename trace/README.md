# Traceability records

Requirements, decisions, interfaces, work packages, tests, runs and sessions as content-addressed records that link upstream by hash, so that a change upstream marks everything downstream stale. This file is the specification; `docs/plan/factory-and-observability.md` section 1 is the background.

## Records

One TOML file per record in `trace/records/`, named `<id>.toml`. An id is a prefix and a zero-padded number (`R-014`, `WP-001`), assigned in order with `next_id` (one past the highest in use). Records are never deleted (set `status = "superseded"`), so ids are never reused. Files hold top-level keys only, then `hash`, then `[upstream]` as the last table.

| Prefix | Required fields | Link fields |
|---|---|---|
| `R-` requirement | `title`, `status`, `source`, `statement`, `rationale` | `derived_from` (requirements) |
| `D-` decision | `title`, `status`, `source`, `decision`, `alternatives_rejected`, `rationale`, `consequences` | `derived_from` (requirements) |
| `I-` interface | `title`, `status`, `source`, `contract` | `derived_from` (decisions) |
| `WP-` work package | `title`, `status`, `source`, `scope`, `non_goals`, `acceptance`, `budgets` | `implements`, `satisfies`, `depends_on` |
| `T-` test | `title`, `status`, `checks`, `tolerance`, `testset` | `verifies` |
| `RUN-` run | `title`, `status`, `inputs`, `commit` | `verifies` |
| `S-` session | `agent`, `platform`, `start`, `end`, `outcome`; optional `pr`, `review_status`, `notes` | `work_packages` |

Status is `proposed`, `accepted` or `superseded` for R, D and I. Where the plan gives no rationale, the record says so. Transcribe, do not invent.

**Hash.** `hash = "sha256:<hex>"` is the SHA-256 of the canonical form of the parsed record with `hash` and `upstream` removed. The canonical form is JSON without whitespace: object keys sorted by code point, arrays in order, `\r\n` normalized to `\n` in strings, other control characters as `\u00XX`, integers in decimal, floats as Julia's shortest round-trip `repr`, dates as ISO 8601 strings. It is implemented once, in `canonical` and `record_hash` in `scripts/trace/Trace.jl`. Key order, quoting style and whitespace outside strings do not change the hash.

**Upstream.** `[upstream]` maps every id in the record's link fields to that record's hash when this record was written or last reviewed (`R-014 = "sha256:..."`).

**Staleness.** A record is *directly stale* when an upstream hash it recorded differs from that upstream's current hash. It is *stale through upstream* when a record it links to is stale. Directly stale records are the *stale frontier*: re-review them first. Re-reviewing a record means checking it against its upstream, editing it if needed, and restamping it with `--review`. Records downstream of it stay stale only if its content changed.

## Commands

All from the repository root, in the `scripts/` environment:

```
julia +1.13 --project=scripts scripts/trace/build.jl [--check]
julia +1.13 --project=scripts scripts/trace/stamp.jl [--review] (ID... | --all)
julia +1.13 --project=scripts scripts/trace/session.jl --agent A --platform P --wp WP-001,... \
    --start <commit> --end <commit> --outcome O [--pr N] [--review S] [--notes TEXT]
julia +1.13 --project=scripts scripts/trace/test/runtests.jl
```

- `build.jl` writes `trace/build/graph.json` (nodes with `trace_status` `ok`, `stale` or `hash-mismatch`; edges from upstream to downstream with the recorded hash), `trace/build/coverage.json` (per requirement: decisions, work packages, tests, runs) and `trace/build/summary.txt` (stale frontier, untraced tests, errors), and prints the summary. `trace/build/` is ignored by git. With `--check` it exits 1 on a hash mismatch, a broken or unrecorded link, a cycle or an untraced test. A stale accepted record is only a warning. CI runs it in the `trace` job.
- `stamp.jl` sets `hash` after an edit and adds `[upstream]` entries for new links at their current hashes. `--review` refreshes all entries.
- `session.jl` writes the next `S-` record and stamps it.

Workflow: edit or add records, run `stamp.jl` on them, run `build.jl --check`, and commit the records with the change that needs them. Requirement, decision and interface records are tier 2 (`.github/CODEOWNERS`). Work-package, test, run and session records are not.

## Test tags

A testset that verifies requirements starts its name with its `T-` id and ends with the requirement ids in brackets:

```julia
@testset "T-031 Esirkepov charge conservation [R-012, R-013]" begin
```

The builder scans the `.jl` files under `test/`, `lib/`, `scripts/` and `benchmark/`. The text between the id and the brackets must equal the record's `testset` field, and the bracketed ids must equal its `verifies` list. A testset with a `T-` id but no record, a `T-` record with no testset, a mismatch, or an id used twice is an untraced test. Testsets without a `T-` id (nested ones, for example) are ignored.
