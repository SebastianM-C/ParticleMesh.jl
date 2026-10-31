# WP-001: trace layer MVP

Status: proposed. Tier: the records are tier 0, the CI job makes the PR tier 2 (owner approval).

## Scope

The traceability layer of `docs/plan/factory-and-observability.md` section 1, enough for every later work package, test and run to be recorded: the record format, the builder, the first records transcribed from the plan, the test-tag convention and one CI gate. Until this lands, section 1 is the specification; after it, `trace/README.md` is.

## Deliverables

1. **Record format.** One TOML file per node under `trace/records/`, named `<PREFIX>-<nnn>.toml` with the prefixes of section 1 (`R-`, `D-`, `I-`, `WP-`, `T-`, `RUN-`, `S-`). Fields per node type as listed there, plus `hash` (SHA-256 of the file's canonical content with the `hash` and `upstream` tables removed; define the canonicalization in the README and implement it in one function) and `upstream`, a table `{id = hash}` recording the hash of every node the record derives from at the time it was written. Ids are the prefix plus a zero-padded number, assigned in order, never reused. Only Julia stdlibs (`TOML`, `SHA`) and `JSON` from the `scripts/` environment; no new dependency, so no package-manager round trip.
2. **Builder.** `scripts/trace/build.jl`, run as `julia +1.13 --project=scripts scripts/trace/build.jl`: reads all records, verifies every `hash`, marks a node stale when a recorded upstream hash no longer matches and propagates staleness downstream, parses test files for `@testset "T-nnn ... [R-nnn, ...]"` names, and writes `trace/build/graph.json` (nodes, edges, status, hashes), `trace/build/coverage.json` (requirements versus tests and runs) and a plain-text summary of the stale frontier and of untraced tests (a testset with a `T-` id that has no record, or a `T-` record whose testset does not exist). A `--check` flag exits non-zero on a hash mismatch or an untraced test, and prints stale accepted nodes as warnings.
3. **First records.** `R-` records for the requirements of `docs/plan/landscape-and-requirements.md` section 6; `D-001` to `D-014` for decisions A1 to A14 of `docs/plan/open-items.md`, each `derived_from` the requirements it serves, with the alternatives the plan text rejects; `I-` records, status proposed, for the interfaces named in `docs/plan/architecture-draft.md` section 0; `T-000` for the existing test; `WP-001` for this package. Transcribe, do not invent: when the plan gives no rationale, say so in the record.
4. **Test tags.** The convention `@testset "T-031 name [R-012, R-013]"` documented in `trace/README.md`; the existing test already follows it.
5. **CI gate.** A job `trace` in `.github/workflows/CI.yml` that runs the builder with `--check`. It fails on a hash mismatch or an untraced test, and only warns on stale accepted nodes for now.
6. **Session records.** The `S-` schema and `scripts/trace/session.jl`, which writes one `S-` record from arguments (agent, platform, work packages touched, start and end commit or PR, outcome). Wiring it into the harness (hooks, cost fields) is a later package.

## Non-goals

The viewer and the static state page (A13, later package), cost attribution per package, Managed Agents events, any change to the engine code.

## Acceptance

- `julia +1.13 --project=scripts scripts/trace/build.jl --check` passes on the first records and on the test suite.
- A test `T-001 trace staleness propagates [R-nnn]` (the requirement that covers traceability, from the transcription above): edit a copy of an `R-` record in a temporary directory, rebuild, and assert that the dependent `D-`, `WP-` and `T-` nodes are stale and unrelated nodes are not.
- A test `T-002 trace hashes are canonical [R-nnn]`: re-serializing a record with reordered keys yields the same hash.
- The `trace` CI job is green on the PR.
- `trace/README.md` describes the format, the commands and the tag convention in under two screens.

## Depends on

Decisions A6, A11, A13 (layout and environments); section 1 of the factory document as the specification; the requirement list of landscape section 6.

## Budgets

CPU only, seconds per build. No GPU, no network beyond the Julia package servers.
