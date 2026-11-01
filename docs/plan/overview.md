# ParticleMesh: the overall picture and the remaining design decisions

The zoom-out over the planning documents; the detail lives in the documents it points to, and `README.md` is their index.

## 1. The picture in one pass

Science: differentiable laser-wakefield simulation in Julia, built up a model ladder to optimize a handful of laser and plasma parameters (A0; the levels in `landscape-and-requirements.md` §5).

Software: a generic particle–mesh engine with pluggable physics modules, the electromagnetic module with LWFA on it the only one planned (A9, `architecture-draft.md` §0.1); KernelAbstractions kernels, generic in the element type and in normalized units (A8, A1), that keep the Reactant path open (§0.4); SciML around the loop (`sciml-reuse.md`); the layers L0 to L4 in `architecture-draft.md`.

Validation: the 1D benchmarks and one matched 2D LWFA case against Smilei and PIConGPU (A5), conservation residuals and gradient checks to the tolerances of A7, FP32-versus-FP64 drift (open-items B5).

Factory: the owner sets the goals and vetoes derived work from the daily digest (A0); the links and staleness are in `trace/README.md`, and the compute split, the overseer, the human interface and the merge policy in `factory-and-observability.md` (§2, §3d, §3e, F4).

```
 you ──edit docs/goals, answer questions, veto from the digest───────────┐
   ▲                                                                      ▼
 daily digest ◄── overseer (reads artifacts, no GPU) ◄── coverage  ◄── repository
 dashboard                                                 ▲   ▲   ▲        ▲
 steering session                                          │   │   │        │
                      factory sessions / workflows ────────┘   │   └── GPU runs on the 4080 (on demand)
                      agent session records via hooks/OTel ────┘       run manifests, plots, sidecars
```

## 2. What is decided (pointers)

The full text is in `docs/goals/` (A0 to A10) and in `factory-and-observability.md` section 5 (F1 to F4).

A0 Aims · A1 Units and normalization · A2 Baseline numerical scheme · A3 How parameters enter the simulation · A4 Determinism policy · A5 Canonical benchmark cases with numbers · A6 Repository and environment · A7 Acceptance criteria and budgets per work package · A8 Element type is a user-level choice · A9 Scope: a generic particle–mesh framework · A10 Problem description separated from numerical implementation · factory settings: F1 Package environments · F2 Dependencies · F3 Results, plots and reports · F4 Merge policy.

Derived design choices without a goal id (2D Cartesian first with the envelope model instead of quasi-3D RZ, ForwardDiff first with about three parameters, the KernelAbstractions core, no Checkpointing.jl, saved workflows as the production line with Managed Agents as a bounded track) are in `architecture-draft.md` (constraints, §3), `open-items.md` D and `factory-and-observability.md` §4.

## 3. Design decisions still open, by when they bite

### Factory setup
1. **Budget per role.** The roles are defined in `.claude/agents/` (model, effort and tools in their frontmatter, per-run overrides in the workflows that start them); what each role may spend is open.
2. **Cloud environment.** An environment exists and has been through a few test rounds. Open: the network access list (the Julia allowlist), which still needs refining; the setup script within the five-minute cache window; the depot strategy for the SciML stack (SessionStart hook versus a prebuilt Docker depot image), environment variables, what secrets if any.
3. **The A5 expected values**, committed, so sessions do not guess: WP-012, WP-014 (Weibel) and WP-017 propose them, and the owner accepts them in A5.
4. **Safety of unattended runs.** Unattended sessions run under a separate configuration directory with the Bash sandbox on and the deny rules of `.claude/settings.json`; CI's leak scan rejects IP addresses, ssh URLs and private keys. Open: a separate user account or container with no access to the owner's credentials, and a check that also rejects hostnames and private package names.

### Before the first GPU package
5. **Testing and CI design.** Test tiers (unit, conservation, benchmark, cross-code, gradient), runtime budgets per tier, which run on GitHub Actions and which on the GPU runner or the nightly loop, where expected values and cross-code reference outputs live (size, LFS or a self-hosted server), how failures become trace records.
6. **Artifact and data management.** Run record format (reuse EDM's manifests or generalize them), where plots and large outputs go, retention, the publishing path to the dashboard, content addressing of inputs.
7. **Dashboard.** Reuse the EDM dashboard pattern (static viewer plus builder) with the trace views added, or build new; hosting and access.
8. **Moving window specification** (B1), **differentiable diagnostics** (B3), **FP32 drift test** (B5), **symbolic spike criteria** (B6), **Reactant compatibility spike** (B9), **Molly interface study** (B10).
9. **The GPU loop.** The Agent SDK script: cadence, what it runs, how it reports, API key handling, budget caps, container versus bare metal, reuse of the EDM campaign recipes.
10. **Observability backend.** Where OTLP goes (a self-hosted Grafana stack, a hosted one, or a file collector at first), retention, how cost is attributed per work package.

### Before the first scientific result
11. **The objective and gradient-versus-Bayesian** (deferred by decision): which quantity, which injection, smoothing, how "informative gradient" is judged at each ladder level, and the role of BO as baseline or partner.
12. **Kill and pivot criteria.** What result at level 0 or level 2 would mean the gradient route is not worth the PIC effort, and what the pivot is (gradient-enhanced BO, surrogates). Define it now so the ladder's early levels are a real test.
13. **Deliverables and claims.** Which paper or talk the first result targets, what claims it makes, and therefore which validation evidence is mandatory; the factory experiment itself as a possible second write-up, which decides what process data to record from day one.
14. **Envelope model scoping** (B2) and **multi-GPU prerequisites** (B7).

### Process
15. **Plan reviews.** How often the planning documents are reviewed (the `plan-review` workflow on a schedule).
16. **Cost model.** Per-package and monthly ceilings, alerts, and what happens when a pool runs out mid-package.
17. **The tier thresholds**: which paths are tier 2 (`.github/CODEOWNERS`) and when a package type moves between tiers, adjusted on evidence (F4).

## 4. Suggested order for closing them

Items 1 to 4 come first. Items 5 to 10 can be written while the CPU-side work runs, since they bite only when GPU work starts. Items 11 to 13 are deferred by decision (A0, open-items D); the level-0 reduced model will produce the first data for them. Items 14 to 17 can wait for evidence.
