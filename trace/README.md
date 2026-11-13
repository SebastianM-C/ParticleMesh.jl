# Trace: links from the goals to the work

Git holds the history and the diffs. This directory adds only the links between the owner's goals, the work packages and the tests, and one check over them. Background: `docs/plan/factory-and-observability.md` section 1.

## The three kinds of file

- **Goals** are `docs/goals/A<n>-<slug>.md`, first line `# A<n>. <title>`. The owner's (`docs/goals/README.md`).
- **Work packages** are `trace/wp/WP-<nnn>-<slug>.md`: TOML front matter, a title line, then one or two sentences: what to build, and the external check that shows it works (theory, a conservation law, another code). Nothing else: no scope, deliverables, acceptance lists, reserved test ids or non-goals. The implementer reads the code, plans, and chooses the tests when the work starts, and the PR description says what was built and what the evidence is. A done package keeps its two sentences and names the report where its outcome lives. Ids are assigned in order and never reused: a new package takes one past the highest id ever used, deleted packages included (`git log --format= --name-only -- trace/wp` lists every file this directory has held).

  ```
  +++
  id = "WP-015"
  status = "proposed"            # proposed, active, done, parked
  goals = ["A2", "A8"]           # the goals it serves; at least one
  depends_on = ["WP-014"]        # packages it builds on
  adds_dependencies = ["SciMLBase"]  # optional: Julia packages its PR adds (F2)
  +++
  # WP-015: swappable relativistic pushers

  Boris, Vay, Higuera–Cary and exact-rotation Boris as device functions and as OrdinaryDiffEq
  algorithms, each measured against exact orbits.
  ```

  A package that is withdrawn is deleted (`git rm`) in the pull request that withdraws it, with the reason in the commit message; git keeps the file. A done package stays while tests or other records cite it, with `status = "done"`.
- **Tests** carry their id and what they verify in the testset name, goals or packages: `@testset "T-110 Esirkepov charge conservation [A2, WP-009]"`. Test ids are unique across `test/`, `lib/`, `scripts/` and `benchmark/`; nested testsets without an id are ignored.

Requirements, interfaces and plans are prose in `docs/plan/` that cites goal ids; they are not linked records.

## When a goal changes

The next session that picks up an open package citing the goal re-reads it against the goal and rewrites the sentences that no longer fit; at two sentences a package is re-read, not re-derived.

## Commands

From the repository root, in the `scripts/` environment:

```
julia +1.13 --project=scripts scripts/trace/build.jl [--check]
julia +1.13 --project=scripts scripts/trace/test/runtests.jl
```

`build.jl` prints coverage (per goal: packages, tests) and the errors. `--check` exits 1 on an error: a malformed file or front matter, an unknown front-matter key, an unknown goal, package or tag, a duplicate or untagged test id, or a forbidden engine dependency (ModelingToolkit, MethodOfLines, Molly or Checkpointing in the root or a `lib/` `Project.toml`). CI runs both commands. The check sees only the packages present, so it cannot catch a reused id or a citation of a deleted package outside test tags and `depends_on`; the next id comes from the git log as above.
