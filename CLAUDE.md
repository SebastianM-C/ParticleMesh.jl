# ParticleMesh.jl: instructions for every Claude session

ParticleMesh.jl is a differentiable particle–mesh simulation engine in Julia with pluggable physics modules. First module: electromagnetic particle-in-cell for laser-wakefield acceleration. Long term: a showcase and stress test for Julia and Reactant compiler technology. This file is the contract for every agent session, local or cloud.

## Orientation (read in this order)

1. `docs/plan/overview.md`: the picture, what is decided, what is open.
2. `docs/plan/open-items.md`: decisions A1 to A13 (binding), thin spots B, deferred items C.
3. `docs/plan/architecture-draft.md`: layers; section 0 for the engine/module split and the Reactant rules.
4. `docs/plan/factory-and-observability.md`: traceability records, risk tiers, how sessions are recorded.
5. The work package you were given, in `trace/`.

## Rules that are never relaxed

- This repository is public. Never commit hostnames, ssh aliases, IP addresses, machine user names, tokens, or the names of private packages or products. Describe hardware by GPU or CPU model ("RTX 4080 SUPER", "MI300X").
- Julia 1.13 is the target (`julia +1.13` with juliaup). CI runs 1.13; the next pre-release is allowed to fail.
- The engine works in normalized units with the laser frequency as reference: time in 1/ω₀, length in c/ω₀, fields in m_e c ω₀/e, density in units of the critical density. Unit conversion lives only in the user layer.
- The element type `T` is generic. No float literals in kernels (`zero(T)`, `T(π)`, `eps(T)`). Float32, Float64 and ForwardDiff duals must pass the CPU test matrix.
- Kernels are KernelAbstractions kernels. No scalar indexing outside kernels. Allocate with `similar` or `KernelAbstractions.allocate`. Static shapes: fixed-capacity particle buffers with an alive mask and a count; no `push!` or `resize!` inside a step; no host-side branching on device data.
- Deposition exists in two variants, atomics and sort plus segmented reduction, and the tests check that they agree.
- Engine code never references a specific physics; physics modules plug in through the interaction interfaces. The engine must not depend on ModelingToolkit, MethodOfLines, Molly or Checkpointing.jl.
- Parameters enter through particle weights and sources, never through particle positions.
- Reference mode is deterministic: counter-based random numbers, sort-based deposition, bitwise reproducibility tested.
- The Reactant CPU smoke test is part of CI. If it fails, fix the kernel; do not remove the test.
- Never fabricate citations, expected values or benchmark numbers; mark anything unverified as such.
- No comments in `Project.toml` files; explain compat choices in the commit message.
- Only the `package-manager` role changes a `Project.toml` or a manifest, on a `pkg/` branch; CI rejects such changes on any other branch. Need a package? Add it locally to keep working, leave the dependency files out of your PR, and post the request on the pinned issue "Package requests".
- Plots are made with CairoMakie from `scripts/`, never from package code; every plot is saved as PNG next to a JSON sidecar with the scalar values it shows.

## Layout

A Pkg workspace: the umbrella package at the root, with `test/`, `docs/`, `benchmark/` and the `lib/` subpackages as workspace members sharing the committed root `Manifest.toml` (`lib/` packages are reached through `[sources]` with relative paths, so every worktree is self-contained). `scripts/` is its own environment outside the workspace, with a committed manifest, for the user code that produces results: benchmark cases, validation and cross-code runs, campaigns, figures. `reports/` Markdown reports with their figures; `docs/plan/` design record; `trace/` traceability records and builder; `questions/`; `.claude/agents/` roles; `.claude/workflows/` saved workflows.

## Working here

- Work in a worktree: `git fetch`, then EnterWorktree (worktrees live under `.claude/worktrees/`, ignored by git, branched from `origin/main`). Instantiate with `julia +1.13 --project -e 'using Pkg; Pkg.instantiate()'`, run the CPU tests, open a PR. Every worktree pays one precompilation against the shared depot; keep dependency lists short.
- Tests: `julia +1.13 --project -e 'using Pkg; Pkg.test()'`; for a subpackage, `--project=lib/<Name>`. One `Pkg.instantiate()` at the root covers the whole workspace; `scripts/` instantiates separately. GPU tests run only with `PM_TEST_GPU=1` and through `scripts/gpu-lock.sh`, which serializes GPU use between agents. Never run GPU work from two worktrees at once. A sandboxed session has no GPU; where the machine provides a `gpu-runner` session (see that machine's own CLAUDE.md), request the run from it instead of trying CUDA locally.
- Test names carry trace tags, `@testset "T-031 Esirkepov charge conservation [R-012]"`, so the trace builder can link tests to requirements.
- Do not ask blocking questions. Write the assumption to `questions/<id>.md`, proceed, and mark the affected records.
- Decisions: A1 to A13 in `docs/plan/open-items.md` are the current decisions, stated as they stand; git is their history. To change one, edit its text in the PR that needs the change and say why in the PR description (tier 2). Never write "decided on <date>" in the plan and never keep superseded text next to the new one.
- Pull requests: tier 0 (tests, docs, mechanical changes) may auto-merge on green CI; tier 1 merges after the lazy-consensus window; tier 2 waits for explicit approval. Tiers are defined in `docs/plan/factory-and-observability.md`.
- Anything you post on GitHub (PR descriptions, issues, comments) states that it was written by Claude Code and names the model id. Commits carry the attribution trailer the harness provides. Replace your home directory by `~` in any path you post.
- Before benchmark numbers or scientific claims leave the repository (a README table, a paper, a talk), they get an independent review by a fresh reviewer session.
- Runs produce the standard plot set with scalar sidecars; record every run in `trace/` with its inputs hashed. Reports are Markdown in `reports/`, with the header `reports/README.md` describes and the figures committed next to them.

## Upstream issues and factory issues

- If you suspect a bug in a dependency or in Julia itself, do not open, comment on or react to anything in another repository. The factory's GitHub credentials reach only this repository, by design. Hand the suspicion to a fresh `upstream-reporter` subagent (`.claude/agents/upstream-reporter.md`): it reduces the problem to a minimal working example that does not depend on this repository, confirms it in a clean temporary environment, and posts it as one comment on this repository's pinned issue "Potential upstream issues" (`gh issue list --label upstream-triage --state open`). Meanwhile work around the suspected bug and write `questions/<id>.md` linking the comment. Humans and sessions outside the factory triage those comments and file upstream.
- Problems with the setup or the process rather than the code (sandbox denials, depots, the GPU runner, tooling, a rule in this file that cannot be followed, decisions that contradict each other) go as one comment per problem on the pinned issue "Factory issues" (`gh issue list --label factory --state open`): what you tried, the exact error, the workaround, the time lost. Then continue. Never retry outside the sandbox or look for another way in.
- A package you need goes as one comment on the pinned issue "Package requests" (`gh issue list --label pkg-request --state open`): package, version constraint if any, what for, which work package. The `package-manager` role answers with a PR.
- All three issues are inboxes. Never create, edit, close or pin issues; only comment.

## Background jobs and waiting

- Never wait for another process with a loop over `pgrep -f "<pattern>"`: the waiting shell's own command line contains the pattern, so the loop waits on itself (this has stalled a queued benchmark for twenty minutes before). Chain dependent work in one command (`cmd1; cmd2`, or `&` and `wait`), or wait on the exact PID (`while kill -0 <pid> 2>/dev/null; do sleep 10; done`). If a pattern is unavoidable, write it so it cannot match itself: `pgrep -f '[b]ench.jl'`.
- A GPU job requested from the runner is finished when its log file ends with a line starting `# gpu-run finished:`; wait for that line, not for a process.

## Julia notes

- Find a package's location with `julia +1.13 --startup=no --project -e "using X; Base.pkgdir(X)"`; package versions are in the `Manifest.toml` of the environment you are in (a manifest may be versioned per Julia release, `Manifest-v1.13.toml`).
- Test and docs environments use `[sources]` to reach the packages in this repository; do not `Pkg.develop` over them.
- Prefer `foreachindex`-style KernelAbstractions kernels; `Atomix` for atomics; `Adapt` for structs crossing to the device.
