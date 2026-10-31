# ParticleMesh: the overall picture and the remaining design decisions

Index of the design record and the zoom-out over the planning documents; the detail lives in the documents it points to.

## 1. The documents

| File | What it holds |
|---|---|
| `landscape-and-requirements.md` | what exists (Julia, JAX, production PIC, BO practice, reduced models), the MethodOfLines v1 measurements, requirements R1 to R10, risks, sources |
| `architecture-draft.md` | layers L0 to L4, core numerics, parallel layout, differentiation, symbolic front-end, hardware mapping, order of work |
| `sciml-reuse.md` | what to reuse from SciML per layer, with what was verified in the package sources |
| `open-items.md` | decisions taken (A1 to A14), thin spots (B1 to B11), deferred items, risks, the first cloud-session batch (G), what the implementation team receives (E) |
| `factory-and-observability.md` | traceability layer, compute topology, platform facts, visualization, budget pools, overseer, human interface, factory shape |
| `workflows-primer.md` | how saved Claude Code workflows work, the `plan-review` script, the factory's workflow list |
| `overview.md` | this file |

## 2. The picture in one pass

Science: differentiable laser-wakefield simulation in Julia, used to optimize a handful of laser and plasma parameters for electron energy, built up a model ladder: 1D wake ODE with a test electron, r-z quasi-static fluid plus envelope, 1D3V PIC, 2D Cartesian PIC with a moving window, then envelope 3D and full 3D on many GPUs. The objective and the gradient-versus-Bayesian question are deliberately still open.

Software: a plain-numeric core (SoA fields and particles, KernelAbstractions kernels, dimension-generic, element type chosen by the user, normalized units with ω₀ as reference), a parallel layer that grows from a no-op exchange to slab decomposition and MPI, a tangent-bundle forward-mode layer for about three parameters, an optional symbolic front-end that reuses the EDM components and MOL's staggered Yee update, and SciML around the loop (the `DiscreteProblem` harness, callbacks, ensembles, DiffEqGPU for test particles, LinearSolve, SciMLSensitivity and Optimization for the reduced models, ModelingToolkit for parameters and composition). Under A9 the package is a generic particle–mesh engine with pluggable physics modules (LWFA first, electrostatic PIC as the proof of the split, EDM and Molly-style interactions later) and has an explicit long-term role as a showcase and stress test for Julia and Reactant compiler technology; the engine follows rules that keep the Reactant path open from day one (`architecture-draft.md` §0).

Validation: classic 1D benchmarks with expected numbers, one matched 2D LWFA case run in ParticleMesh, Smilei and PIConGPU for correctness and throughput, conservation residuals, FP32-versus-FP64 drift, gradient checks against finite differences with deterministic deposition.

Factory: requirements, decisions, interfaces, work packages, tests, runs and agent sessions as content-addressed records with staleness propagation; implementation by saved Claude Code workflows with pinned roles, on cloud sessions for CPU-only packages and on the local 4080 for GPU packages; a cron-driven Agent SDK loop on the 4080 under an API key for nightly validation; an overseer (a scheduled task now, a Managed Agents deployment later) that reads artifacts and writes a daily digest; a human interface built on risk tiers, lazy consensus, no blocking questions, one daily batch and one weekly hour; plots with scalar sidecars as the inspection channel for humans and agents alike; telemetry and hooks feeding the same trace graph.

```
 you ──edit R-/D- records, answer questions, approve tier-2 PRs──┐
   ▲                                                                      ▼
 daily digest ◄── overseer (reads artifacts, no GPU) ◄── trace graph ◄── repository
 dashboard                                                 ▲   ▲   ▲        ▲
 steering session                                          │   │   │        │
                      cloud sessions / workflows (CPU) ────┘   │   └── GPU runs on the 4080 (on demand)
                      agent session records via hooks/OTel ────┘       run records, plots, sidecars
```

## 3. What is decided (pointers)

A1 normalized units, ω₀ reference, SI in the user layer · A2 baseline scheme direction (Yee, Boris, Esirkepov, quadratic shapes, antenna laser, immobile ions, quiet start; the written decision set with tests is still owed) · A3 parameters enter through weights and the antenna · A4 deterministic deposition as reference mode, counter-based RNG · A5 verify and benchmark against Smilei and PIConGPU · A6 public MIT repository, CPU CI plus a GPU runner · A8 element type generic, DoubleFloats nice-to-have · 2D Cartesian first, envelope instead of RZ, multi-GPU 3D eventual · ForwardDiff first with about three parameters · KA core, EDM reuse, MOL optional · no Checkpointing.jl · Claude Code workflows as the production line, Managed Agents as a bounded track · overseer as a scheduled task first · risk tiers and the daily batch · A9 generic particle–mesh framework, LWFA as the first physics module, Reactant a declared long-term target with a CPU smoke test from day one · A10 problem description separated from numerical implementation, the symbolic front-end as the mechanism, FEM/DG mesh backends admitted but unscheduled. · A11 Pkg workspace with committed manifests, `scripts/` as its own environment · A12 one package-manager role owns dependencies · A13 CairoMakie plots with sidecars, Markdown reports, static state page

## 4. Design decisions still open, by when they bite

### Before the cloud sessions start
1. **Repository topology and naming.** Closed: monorepo with `lib/` subpackages in a Pkg workspace (A11); Julia 1.13; no EDM dependency at first, direct dependencies on LaserTypes.jl and GPUDiagnostics.jl, RunManifests lifted out of EDM; `ParticleMesh.jl` at https://github.com/SebastianM-C/ParticleMesh.jl, agents in worktrees under `.claude/worktrees/` via EnterWorktree.
2. **Trace schema and tooling.** Record format (TOML or YAML), id scheme, hash algorithm and what exactly is hashed, how test tags are parsed from Julia testsets, how CI results and session records attach, Python or shell so it runs in any sandbox. The minimal builder must exist before the first session so every session is recorded.
3. **Role roster.** The actual subagent definitions: implementer, numerics reviewer, plot inspector, trace maintainer, overseer; model, effort, tool allowlists, permission rules per machine, budget per role.
4. **Cloud environment.** Setup script within the five-minute cache window, the Julia allowlist, the depot strategy for the SciML stack (SessionStart hook versus a prebuilt Docker depot image), environment variables, what secrets if any.
5. **The written A2 decision set and the A5 benchmark numbers**, committed, so sessions do not guess.
6. **Safety of unattended runs.** Unattended agents need a separate user account or container with no access to the owner's credentials, a repository pre-commit or CI check that rejects hostnames and private package names, and an explicit list of what an agent may never do. Unattended sessions now run under a separate configuration directory with the Bash sandbox on.

### Before the first GPU package
7. **Testing and CI design.** Test tiers (unit, conservation, benchmark, cross-code, gradient), runtime budgets per tier, which run on GitHub Actions and which on the GPU runner or the nightly loop, where expected values and cross-code reference outputs live (size, LFS or a self-hosted server), how failures become trace records.
8. **Artifact and data management.** Run record format (reuse EDM's manifests or generalize them), where plots and large outputs go, retention, the publishing path to the dashboard, content addressing of inputs.
9. **Dashboard.** Reuse the EDM dashboard pattern (static viewer plus builder) with the trace views added, or build new; hosting and access.
10. **Moving window specification** (B1), **differentiable diagnostics** (B3), **FP32 drift test** (B5), **symbolic spike criteria** (B6), **Reactant compatibility spike** (B9), **Molly interface study** (B10).
11. **The GPU loop.** The Agent SDK script: cadence, what it runs, how it reports, API key handling, budget caps, container versus bare metal, reuse of the EDM campaign recipes.
12. **Observability backend.** Where OTLP goes (a self-hosted Grafana stack, a hosted one, or a file collector at first), retention, how cost is attributed per work package.

### Before the first scientific result
13. **The objective and gradient-versus-Bayesian** (deferred by decision): which quantity, which injection, smoothing, how "informative gradient" is judged at each ladder level, and the role of BO as baseline or partner.
14. **Kill and pivot criteria.** What result at level 0 or level 2 would mean the gradient route is not worth the PIC effort, and what the pivot is (gradient-enhanced BO, surrogates). Define it now so the ladder's early levels are a real test.
15. **Deliverables and claims.** Which paper or talk the first result targets, what claims it makes, and therefore which validation evidence is mandatory; the factory experiment itself as a possible second write-up, which decides what process data to record from day one.
16. **Envelope model scoping** (B2) and **multi-GPU prerequisites** (B7).

### Process
17. **Governance of the plan.** Decision record format, who may change a requirement, how often the planning documents are reviewed (the `plan-review` workflow on a schedule), how the plan is versioned alongside the code.
18. **Cost model.** Per-package and monthly ceilings, alerts, and what happens when a pool runs out mid-package.
19. **The tier thresholds**: which paths are tier 2 (`.github/CODEOWNERS`) and when a package type moves between tiers, adjusted on evidence (A14).

## 5. Suggested order for closing them

Items 1 to 6 come first; they are the input the cloud sessions need. Items 7 to 12 can be written while the cloud sessions run, since they bite only when GPU work starts. Items 13 to 15 are deferred by decision (open-items D); the level-0 reduced model (one of the cloud packages) will produce the first data for it. Items 16 to 19 can wait for evidence.
