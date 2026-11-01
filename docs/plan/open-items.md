# ParticleMesh: open items

What is still open in the plan: thin spots (B), what can wait (C), and the dependencies and risks to keep visible (D). The goals and decisions are in `docs/goals/`, the factory settings F1 to F4 in `factory-and-observability.md` section 5.

## B. Thin spots in the plan (fill in during planning or as the first spike of a phase)

B1. **Moving window mechanics.** Specify: advance by whole cells every k steps; fields shifted by an index offset (circular buffer) rather than memory moves; particle positions in window coordinates with a global offset; refill at the front sampling the global density profile; drop at the back. With slab decomposition this becomes a ring rotation of ranks. The hook the core must already have is a proposal in `architecture-draft.md` §0.7 (a static grid separate from a small dynamic grid state); the choice between a ring buffer and a memory shift is left to the window package and can be made without touching a kernel.

B2. **Envelope model is a sub-project, not a free cost reduction.** It needs an envelope equation solver (second order in time, with the Smilei/INF&RNO-style scheme), a modified pusher with the ponderomotive force and the |a|²-corrected γ, and a susceptibility deposition. Scope it as its own phase with its own validation (envelope versus full PIC in 2D).

B3. **Differentiable diagnostics.** Histograms and "cut-off energy = max" are not differentiable. The diagnostics module should offer soft binning or kernel density estimates, smooth maxima (log-sum-exp with a temperature) and quantiles. This is independent of the objective decision but should exist before it.

B4. **Reduced-model equations and references.** Level 0 (1D nonlinear quasi-static wake, test electron with dephasing) is covered by the Esarey–Schroeder–Leemans 2009 review; the bubble-regime scalings usually attributed to Lu et al. 2007 are not yet verified as a citation. The r-z quasi-static fluid model of level 1 has no pinned reference yet; verify one before writing the spec.

B5. **FP32 validity.** JAX-in-Cell reports deviations in 32-bit; the cell-plus-offset layout fixes positions but momenta and fields still drift. Plan an FP32-versus-FP64 drift test on a rented FP64-capable GPU as part of phase 2 validation and decide which quantities (energy sums, deposition accumulators) stay FP64.

B6. **Symbolic spike acceptance.** The claim that an MTK-generated force runs inside a KA kernel with duals needs a pass/fail test: generated function inlined into a GPU kernel, correct on CPU and GPU, `Dual` inputs accepted. DiffEqGPU does this already, so the risk is low, but the team should not discover the details mid-phase.

B8. **Machine availability.** Hardware: one local RTX 4080 SUPER and rented GPU VMs. GPU runs are launched on demand through the `gpu-runner` session that factory sessions message (see `CLAUDE.md`); unattended GPU work is deferred, with rented GPU VMs as the later option. Cloud means Anthropic-hosted agent sessions (CPU only, no Julia preinstalled) for CPU-side work, and rented GPU VMs when a second or larger GPU is needed; EDM's campaign backends (`orchestration/backends/` in ElectronDynamicsModels.jl: local, Hot Aisle, RunPod, Verda) can be borrowed for those. Plan phases 1 to 3 around one local GPU; treat anything larger as rented.

B9. **Reactant compatibility spike** (parked with Reactant, A9; it runs right after the smoke-test package once Reactant supports GPUCompiler 2 and KernelAbstractions 0.10). Compile one step of the electromagnetic module on an electrostatic case with no transverse fields (A9; KA kernels through `ReactantBackend`) with `@compile` on CPU; check raising (`raise=true`) of gather, both deposition variants, sort and compaction, and the Yee stencils; measure compile time; run `Enzyme.gradient` on the raised step with `raise_first=true`; find out whether `Dual`-valued locals inside a raised kernel survive (if not, forward mode on the Reactant path is Enzyme-MLIR forward, rule 8 of §0.4); check scatter-add determinism on the GPU. Outcomes become run records; failures become upstream issues with minimal reproducers. Include the harness: one `SimpleFunctionMap` solve around the step under `@compile`, reusing the existing SimpleDiffEq-on-Reactant work, and a note on OrdinaryDiffEq PR #4436's status, including whether it covers `FunctionMap` in `OrdinaryDiffEqFunctionMap` (unverified).

B10. **Molly interface study.** Reading Molly's `System`, interaction and simulator types and `src/interactions/ewald.jl` (PME with KA kernels): reference material for a later pairwise/P³M slot (§0.3), no prerequisite of any package; no code dependency on Molly. The engine owns no container or step (`architecture-draft.md` §0.7), so nothing waits on it.

B7. **Multi-GPU prerequisites.** Check CUDA-aware MPI and the device interconnect topology on the rented multi-GPU instance before the 3D phase; decide device-to-device copies versus MPI for the single-node case.

B11. **Unverified platform fact.** Whether workflow agents fire Claude Code hooks is not confirmed (`factory-and-observability.md` §3); check it before relying on hooks to attribute cost to work packages for workflow runs.

## C. Can wait (phase 4 and later)

Reactant as a production backend (2D on Reactant-CUDA, sharding, multi-host, TPU), reverse mode and checkpointing, 3D block decomposition and load balancing, PSATD or Lehe solver, ionization injection, quantum-corrected radiation reaction, MOL staggered backend beyond the spike, P³M and Molly interoperability (AtomsCalculators/AtomsBase), EDM symbolic forces inside raised kernels, FEM (MOL Ferrite backend) and DG (Trixi) Maxwell backends, PointNeighbors for the pairwise slot.

## D. Dependencies and risks to keep visible

- Checkpointing.jl: not a dependency (maintenance status unknown). Reverse mode over the PIC loop will use in-house checkpointing: periodic two-level checkpoints of the full state over the fixed-step loop with Enzyme per step is a few dozen lines; the binomial (Revolve) schedule is a published algorithm and can be implemented in-house with tests if memory forces it; Reactant's traced loops with checkpointing remain the alternative on that path.
- Reactant and Enzyme-on-CUDA move fast with open correctness bugs (Oct 2026); nothing in phases 1 to 3 may depend on them.
- MethodOfLines stays a dev dependency of the optional extension only, never of the core.
- Reactant raising covers a growing but incomplete class of kernels and the compile time of a many-kernel step is unmeasured; the CI smoke test (once Reactant is unparked, A9) and B9 keep this visible without letting phases 1 to 3 depend on it.
- OrdinaryDiffEq's Reactant path is a draft PR (#4436); the Reactant time loop uses SimpleDiffEq or a `@trace` loop until it lands. The MOL Ferrite backend has no public trace yet; do not plan around it.
- The `FunctionMap` harness and the tangent layout both multiply memory: with `save_everystep = false` the harness holds five copies of the state after `init` and six after `solve` (T-104: `u0`, `u`, `uprev`, the cache's unused `tmp`, and the saved start and end states), against the factor of two A7's memory budget assumes; a 2D case must be sized for the 4080 explicitly.
- The objective and the gradient-versus-Bayesian decision must land before the tangent layer is frozen (end of phase 2), because they fix which parameters are active and which diagnostic is reduced.
