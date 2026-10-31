# Trace: links from the goals to the work

Git holds the history, the diffs and the content addresses. This directory adds only the links between the owner's goals, the work packages and the tests, and one check over them. Background: `docs/plan/factory-and-observability.md` section 1.

## The three kinds of file

- **Goals** are `docs/goals/A<n>-<slug>.md`, first line `# A<n>. <title>`. The owner's (`docs/goals/README.md`).
- **Work packages** are `trace/wp/WP-<nnn>-<slug>.md`: TOML front matter, then the package in Markdown (scope, deliverables, non-goals, acceptance, budgets). Ids are assigned in order, one past the highest in this directory, and never reused.

  ```
  +++
  id = "WP-004"
  status = "proposed"            # proposed, active, done, parked, withdrawn
  goals = ["A2", "A8"]           # the goals it serves; at least one
  depends_on = ["WP-003"]        # packages it builds on
  adds_dependencies = ["HDF5"]   # optional: Julia packages its PR adds (F2)

  [checked]                      # per goal, the git blob id it was last checked against
  A2 = "<40 hex digits>"
  A8 = "<40 hex digits>"
  +++
  ```
- **Tests** carry their id and what they verify in the testset name, goals or packages: `@testset "T-031 Esirkepov charge conservation [A2, WP-004]"`. Test ids are unique across `test/`, `lib/`, `scripts/` and `benchmark/`; nested testsets without an id are ignored.

Requirements, interfaces and plans are prose in `docs/plan/` that cites goal ids; they are not linked records.

## Staleness and the cascade

A package is stale when a goal it serves has changed since it was checked against it: the `[checked]` blob id differs from `git hash-object` of the goal file now. Blob ids depend only on content, so they survive squash merges and shallow checkouts. Withdrawn packages are never stale.

When a goal changes, the planner (`plan-packages` in re-derive mode) reads `git diff` of the goal, and for each stale package confirms, edits, splits or withdraws it, then refreshes its `[checked]` entries; it also updates the prose in `docs/plan/` that cites the goal. That pull request merges on the automated review, and the digest lists the changes for the owner's veto.

## Commands

From the repository root, in the `scripts/` environment:

```
julia +1.13 --project=scripts scripts/trace/build.jl [--check] [--refresh WP-nnn ...]
julia +1.13 --project=scripts scripts/trace/test/runtests.jl
```

`build.jl` prints coverage (per goal: packages, tests), the stale packages and the errors. `--refresh` first records the current blob ids of the named packages' goals, after you have checked the packages against them. `--check` exits 1 on an error: a malformed file or front matter, an unknown goal, package or tag, a goal without a `[checked]` entry, a duplicate or untagged test id, or a forbidden engine dependency (ModelingToolkit, MethodOfLines, Molly or Checkpointing in the root or a `lib/` `Project.toml`). Stale packages are warnings. CI runs both commands.

## Digests

The overseer writes `trace/digests/<date>.md`.
