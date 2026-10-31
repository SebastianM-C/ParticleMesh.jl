# ParticleMesh.jl

> **Experimental, and built by Claude.** This package is being developed largely by Claude
> (Anthropic's Claude Code, running as autonomous agents and saved workflows) under the direction
> and review of @SebastianM-C. It is also an experiment in how such a software
> factory should be organized. Expect gaps, churn and unverified claims until a release says
> otherwise; every number that leaves this repository is meant to pass an independent review first.

ParticleMesh.jl is a differentiable particle–mesh simulation engine in Julia with pluggable
physics modules. The engine owns what every particle–mesh method shares: particle sets, structured
grids, the particle↔mesh transfers (deposition and gather) with shape functions, mesh operators,
the time loop, diagnostics, parallel exchange, and the differentiation and backend machinery.
Physics modules supply what is deposited, which mesh equations are solved, which force acts on a
particle and which pusher integrates it.

The first module is an electromagnetic particle-in-cell model for laser-wakefield acceleration,
built to optimize laser and plasma parameters by gradient. An electrostatic module serves as the
proof that the engine/module split holds. Long term the project is a showcase and stress test for
Julia and Reactant compiler technology: the same KernelAbstractions kernels run natively on
NVIDIA and AMD GPUs and, through Reactant, as a traced, raised and sharded program.

## Status

Nothing runs yet. The design record in `docs/plan/` (start with `overview.md`) is the current
state of the project; the aims and decisions in `docs/goals/` are binding for anyone,
human or agent, who works here.

## Layout

| Path | Contents |
|---|---|
| `src/` | the umbrella package |
| `lib/` | subpackages (engine core, extensions, physics modules), resolved through `[sources]` |
| `docs/goals/` | the owner's aims and binding decisions, the root of everything else |
| `docs/plan/` | the design record derived from them: landscape, requirements, architecture, open items, process |
| `trace/` | work packages (linked to the goals they serve), the trace checker's specification, digests |
| `benchmark/` | regression microbenchmarks, a workspace member |
| `scripts/` | user code that produces results (benchmark cases, validation and cross-code runs, campaigns, figures); its own environment with a committed manifest |
| `reports/` | Markdown reports with their figures, one directory per report |
| `questions/` | assumptions agents made instead of asking |
| `.claude/` | agent roles, saved workflows and settings used by the Claude Code sessions |

## License

MIT, see `LICENSE`.
