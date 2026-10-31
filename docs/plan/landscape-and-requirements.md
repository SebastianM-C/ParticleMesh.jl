# ParticleMesh: landscape and requirements

Status: landscape survey; the requirements of §6 are settled by decisions A1 to A14 in `open-items.md`.

Goal as stated: differentiable LWFA simulations in Julia, used to optimize laser and plasma parameters for maximum electron energy. Question: build on MethodOfLines.jl v1, reuse something existing, or write a PIC core from scratch?

## 1. What a PIC code for LWFA actually is

A PIC time step is four kernels, only one of which is a PDE discretization:

| Stage | Operation | Math character | Differentiability |
|---|---|---|---|
| Field solve | Yee/FDTD or spectral (PSATD) Maxwell update | linear PDE on a staggered grid | linear, trivial |
| Gather | interpolate E, B from grid to each particle | shape-function weights × grid values | piecewise polynomial in position |
| Push | Boris / Vay / Higuera-Cary | smooth ODE step | smooth |
| Deposit | charge-conserving current (Esirkepov / Villasenor-Buneman) scatter-add to grid | shape-function weights, integer cell indices, atomics on GPU | piecewise polynomial; cell crossings are kinks for low-order shapes |

Plus bookkeeping that has no PDE analogue: moving window (shift arrays, inject fresh plasma at the front, drop particles at the back), density profiles and ramps, laser injection (antenna or initial condition), particle sorting/binning, binomial current filtering, diagnostics (spectra, cut-off energy, charge, emittance).

MethodOfLines can at most cover the first row. For a Yee scheme that row is ~50 lines of KernelAbstractions code. The hard engineering is rows 2–4 and the bookkeeping.

### Cost scale (2D and quasi-3D are the realistic targets)

- Resolution is set by the laser wavelength along z (λ₀/20–30) and the plasma wavelength transversely (λ_p/20). Time step by CFL; steps ~10⁴–10⁵ for mm-scale acceleration.
- 1D3V: trivial cost, but no self-focusing, no bubble, no transverse dynamics. Useful for numerics and gradient-quality studies, not for physics claims.
- 2D Cartesian: ~10⁶ cells, ~10⁷ macroparticles → single-GPU minutes to hours. Standard for method papers; quantitative energies differ from 3D.
- Quasi-3D (azimuthal modes, FBPIC/WarpX RZ): a few 10⁶ cells × 2–3 modes; single-GPU hours. This is what the LWFA community uses for parameter studies (Valenta et al. 2026 used quasi-3D + boosted frame).
- Full 3D: 10⁸–10⁹ cells, multi-GPU. Out of scope for a differentiable prototype.

A forward-mode derivative with p parameters multiplies memory and FLOPs by roughly p+1. Reverse mode needs the trajectory (state × steps) or checkpointing.

## 2. What the objective "electron energy" means, and whether it has a useful gradient

This is the central design question, more than any software choice.

- Self-injection (the Valenta et al. setting) is a threshold process: wave breaking, trapping near a separatrix, beam loading. The cut-off energy as a function of (a₀, w₀, τ, n₀) has jumps where injection switches on/off and is chaotic at the particle level. A pathwise AD gradient exists almost everywhere but may be dominated by noise and discontinuities. Irshad's 2025 thesis dismisses gradient methods for exactly this reason ("stochastic, non-differentiable objectives").
- Externally injected test bunch, or a tagged test-particle population in the wake: energy gain is a smooth function of the wake structure (amplitude, phase velocity, dephasing length, depletion). Gradients are informative. This is the objective to start with.
- Density-downramp injection: deterministic and more controllable than self-injection, still threshold-like.
- Ionization injection and QED processes are stochastic; they need score-function estimators and are out of scope initially.
- Chaos: the review by Joglekar et al. (2026) only cautions that gradients must "reflect the underlying continuous physics" (citing Metz et al. on exploding gradients in chaotic systems). Nobody has published gradient-quality studies for LWFA PIC. That is itself a publishable question: at which horizon and particle count does the pathwise gradient of bunch energy w.r.t. a₀ stop agreeing with finite differences?

Surrogate/intermediate objectives worth having in the loop: peak accelerating field E_z behind the driver, wake phase velocity, dephasing length, laser depletion length, energy gain of a test electron injected at a fixed phase. These are smooth and connect to the analytic scalings: the Esarey–Schroeder–Leemans 2009 Rev. Mod. Phys. review, the bubble-regime scalings usually attributed to Lu et al. 2007 (citation unverified, open-items B4), and the closed-form maximum-energy expressions Valenta et al. 2025 derive including diffraction and depletion.

## 3. Existing tools

### Julia

| Package | What it is | Physics | GPU | AD | Verdict for ParticleMesh |
|---|---|---|---|---|---|
| MethodOfLines.jl v1.6.0 (commit fcf63be) | symbolic finite-difference discretizer on MTK `PDESystem` | PDEs only; staggered Yee grids 1D–3D with E/H leapfrog groups since v1 | generated residual/RHS scalar-indexes the state; GPU not supported (see §4) | through MTK/SciMLSensitivity on CPU | Maxwell front-end at best; today not a core |
| ParticleInCell.jl 1.1.0 (JuliaPlasma, Dec 2025) | electrostatic PIC: Poisson solve, interpolation, Boris pusher, MPI halo code | electrostatic only, no FDTD | no | no | not reusable beyond ideas |
| GEMPIC.jl 0.4.2 (JuliaVlasov, Jul 2026) | geometric EM PIC, Hamiltonian splitting, B-spline FEM Maxwell, 1D1V/1D2V/2D3V | structure-preserving research code, periodic boxes | no | no | reference for conservation tests |
| JustPIC.jl | marker-in-cell advection for geodynamics (KernelAbstractions, CUDA/AMDGPU/Metal) | no field solve, no charge | yes | no | pattern reference for GPU particle↔grid interpolation, injection, cell lists |
| TestParticle.jl | test particles in prescribed fields | no self-consistency | partly | – | no |
| ElectronDynamicsModels.jl (the authors' package) | MTK electron models, GPU field evaluators, Liénard–Wiechert radiation, campaign/manifest infrastructure, CUDA+ROCm | test particles in analytic lasers (`lpwa.jl` uses an analytic plane-wave orbit) | yes | – | reuse: laser definitions (LaserTypes.jl), GPU backends, run manifests, precision studies |
| Reactant.jl 0.2.x | XLA/MLIR compilation of Julia arrays, Enzyme-MLIR AD; the Julia analogue of JAX | – | yes | yes | the "JAX-in-Cell in Julia" route; gather/scatter under reverse mode has open regressions as of Oct 2026 (#3589 gradient correctness, #3550 segfault, #3276 perf); compiles KernelAbstractions kernels through its `ReactantBackend` and raises them to StableHLO (AD only on raised kernels, not every kernel raisable); sharding via `Sharding.Mesh`/`NamedSharding` on the IFRT runtime; declared long-term target (architecture §0) |
| Molly.jl 0.23.3 (JuliaMolSim) | differentiable molecular dynamics: `System`, pairwise/specific/general interactions (AtomsCalculators), simulators, loggers, neighbour finders; Ewald and PME long-range electrostatics | MD, not plasma; PME is a particle–mesh method | yes (CUDA fast path, KA path for all backends) | Enzyme reverse and forward (`MollyEnzymeExt`); units unsupported under AD; no adjoint sensitivity | design reference for the container/interaction/simulator split; long-term P³M bridge (architecture §0.3) |
| Oceananigans.jl 0.116 with Reactant | ocean model with a `ReactantState` architecture: KA kernels run through `ReactantBackend`, grids sharded with `Sharding.Mesh`; DJ4Oceananigans (JuliaCon 2026) reports Enzyme+Reactant gradients on CPU/GPU/TPU | stencil code, no particles | yes | yes | the precedent for "KA kernels + Reactant + sharding"; shows what a particle code adds as a stress test |
| Trixi.jl 0.17.18 (0.11.1 inspected) | DG solver for hyperbolic conservation laws, `semidiscretize` → `ODEProblem`; a 1D Maxwell equation type in the depot copy; CPU scaling to 61,440 cores (arXiv:2607.25866); GPU through TrixiCUDA (CUDA-only, unregistered) | hyperbolic PDEs, MHD, 1D Maxwell | partial | through SciMLSensitivity on CPU | possible DG Maxwell backend for the mesh-solve slot long term (architecture §0.5); not for the first module |
| TrixiParticles.jl 0.5.3 + PointNeighbors.jl 0.6.7 | SPH multiphysics with KA kernels (NVIDIA, AMD, Apple); fixed-radius neighbour search with a GPU-compatible `FullGridCellList` (GPU support marked work in progress) | SPH fluids and structures | yes | no | precedent for a KA particle framework; neighbour-search candidate for the pairwise/P³M slot |
| Ferrite.jl 1.7.0 and an expected MOL Ferrite backend | FEM toolbox; the MOL backend is not public as of 2026-10-30 (no PR or issue found) | unstructured FEM Poisson/Maxwell | not verified | – | geometry-heavy electrostatic problems, not LWFA; a second mesh backend that proves the description/implementation split |
| FastMultipole.jl 2.3.0 (BYU FLOW Lab), ExaFMMt.jl 0.1.2 | fast multipole for the Laplace kernel; CPU-parallel expansions, CUDA for direct interactions only; ForwardDiff/ReverseDiff-compatible (README) | long-range sums without a mesh | partial | forward/reverse (ForwardDiff, ReverseDiff) | mesh-free option for a pairwise or gravity module's long-range slot; not needed for LWFA |
| OrdinaryDiffEq PR #4436 (draft, 2026-08-30, updated 2026-10-17) and SimpleDiffEq under Reactant (prior work by the authors) | explicit solves run the ordinary `solve!` loop inside Reactant compilation; untraceable options throw before tracing | – | – | – | the time-loop harness and the reduced models on the Reactant path (architecture §0.6) |
| Enzyme.jl + KernelAbstractions/CUDA.jl | reverse mode through GPU kernels | – | yes | yes | works in basic cases (KA #924 thread, Oct 2026); open bugs: `cis` in kernels (#3850), `fill!` on CuArray (#3807), tape-type mismatches (#3851); CUDA.jl PR #3150 for `autodiff` inside kernels still open |
| ForwardDiff Duals in KA kernels | forward mode | – | yes | yes | robust; the right first AD mode for ≤ ~20 parameters |
| DiffEqGPU.jl | GPU ODE ensembles | – | yes | yes | test-particle ensembles in a frozen wake |

### Outside Julia

- **JAX-in-Cell** (Ma, Jorge, Lu, Tran, Woolford, arXiv:2512.12160, Dec 2025): 1D3V relativistic EM PIC fully in JAX. Yee lattice, Boris or implicit Crank–Nicolson, quadratic 3-point spline shape, Villasenor–Buneman charge-conserving current, binomial filtering, periodic/reflective/absorbing BCs. AD demo is a forward-mode JVP (damped Newton on the two-stream growth rate vs drift velocity). A100 ≈ 100× CPU. No laser, no moving window, no multi-D. The closest prior art; a Julia 1D3V reproduction is a natural first validation target.
- **Differentiable particle–mesh outside plasma**: pmwd ("particle-mesh with derivatives", Li et al., ApJS 270 (2024), arXiv:2211.09958) is a differentiable cosmological PM N-body library in JAX; it takes gradients by the adjoint method with reverse time integration of the reversible dynamics, so memory does not grow with the number of steps (arXiv:2211.09815 for the method). JaxPM is an earlier JAX PM solver. The same deposit → FFT Poisson → gather structure as an electrostatic PIC step, so a PM gravity module would be a near-free demonstration of the engine's generality, and pmwd's reversible-adjoint route is a data point for reverse mode on the ladder.
- **ADEPT** (Joglekar): differentiable 1D Vlasov/fluid solvers in JAX, time-integrated with diffrax; includes a 1D electrostatic PIC entry.
- **Review**: Joglekar, Thomas, Milder, Miller, Palastro, Froula, "Differentiable Programming for Plasma Physics: From Diagnostics to Discovery and Design", arXiv:2603.11231 (Mar 2026). Reverse mode with gradient checkpointing at every step (O(√T) memory); lists "optimizing pulse shapes for laser wakefield acceleration" as future work. So as of March 2026 no differentiable LWFA simulation has been published.
- **Production PIC**: WarpX (GPU, Python, RZ modes, boosted frame), FBPIC (RZ spectral, GPU), Smilei and PIConGPU (both containerized), OSIRIS, EPOCH. None differentiable. They are the validation references and the Bayesian-optimization baseline.
- **Optimization practice in LWFA**: Bayesian optimization over full PIC runs, via **optimas** (Ax/BoTorch + libEnsemble). Valenta, Esirkepov, Ludwig, Wilks, Bulanov, PRAB 28, 094601 (2025), arXiv:2501.06069: 3D PIC + BO, maximize cut-off energy at fixed laser energy over density, duration, amplitude, spot size, channel radius; derives analytic max-energy expressions. Valenta et al., MLST 7, 025030 (2026), arXiv:2512.10728: quasi-3D boosted frame + BO, refined self-guiding matching conditions. BO of spectral pulse shaping with WarpX (Phys. Plasmas 33, 063101, arXiv:2512.09125; only abstract checked). Multi-objective multi-fidelity BO (Irshad et al., arXiv:2210.03484). These papers set the bar: tens of simulations per optimum with BO. Gradients pay off either when parameter count grows (pulse shaping: 10–100 coefficients) or when combined with BO as gradient-enhanced surrogates.
- **Reduced models**: Wake-T (Python/numba): quasi-static 2D r-z wake with laser envelope on its own grid, 1D cold-fluid model extended slice-wise to r-z, and a gridless quasi-static axisymmetric model (Ferran Pousa et al., CPC 328, 2026, arXiv:2603.16623). Seconds to minutes per run, used for fast loops and multi-fidelity BO. Not differentiable. These are PDE/ODE models, so they are where MethodOfLines and SciMLSensitivity fit naturally.

## 4. MethodOfLines v1: what the rewrite does and does not give

Verified against v1.6.0 (commit fcf63be, 2026-10-29) and by the experiments in §4.1.

- **Array form at the symbolic level.** The interior of every PDE is one symbolic equation over slices, e.g. `D(u[2:n-1]) ~ (u[1:n-2] .- 2u[2:n-1] .+ u[3:n]) ./ dx^2`; boundaries are one-element slices. The equation count no longer depends on resolution. Discretization time dropped 50–200× (Discourse announcement, Aug 2026).
- **DAE path.** `discretize` now returns a `DAEProblem` (residual form `D(u) - f ~ 0`) so that `mtkcompile` is skipped and the arrays survive on the `System`. Explicit RK methods need `ODEProblem(mtkcompile(sys))`, which scalarizes.
- **Codegen still scalarizes.** The DAE residual is generated one entry per unknown (comment in `src/dae_discretization.jl`; issue #691, closed Oct 2026 by rewording the docs, not by fixing codegen). Upstream tracking issue ModelingToolkit #5139 (Sep 2026, bot-generated, unreviewed, open) lists residual/Jacobian codegen over views as draft PRs. Discretize + first compile grow superlinearly: 4096 unknowns took 16 s + 26 s in #691's Kuramoto–Sivashinsky test. A 2D LWFA grid has 10⁶ unknowns.
- **GPU.** MOL's own `test/GPU/solution_wrap.jl` solves under `CUDA.@allowscalar` and says "The generated RHS writes element by element, so the solve itself needs scalar indexing; only the wrapping is checked without it." Only solution wrapping is GPU-safe.
- **Staggered grids.** v1 added Yee layouts up to 3D (`edge_aligned_var = [Ex => x, ..., Hz => (x, y)]`), PEC walls, periodic BCs, a `SplitODEProblem` for `SplitEuler`, and `staggered_groups`, which generates per-group leapfrog `update!` functions from an `ArrayMaker` with `fuse_broadcasts`. Symbolics 7.48 does ship `fuse_broadcasts`. Whether that path emits broadcasts over views (GPU-safe) or per-element code is tested below.
- **No particles.** Nothing in MOL handles Lagrangian particles, scatter-add, or moving windows, and MTK's symbolic layer is the wrong tool for them (10⁷ particles are a batch, not a system of named unknowns).

### 4.1 Experiments (scratch project, Julia 1.13.3, MTK 11.46.1, Symbolics 7.48.0, CUDA 6.4.2)

Three scratch scripts (not in the repository; results tabulated below), RTX 4080 SUPER, one warm process per run.

**Experiment 1: 1D heat equation through `discretize` (DAE path).** The symbolic system is one slice equation plus two one-element boundary equations for any N, as advertised.

| N | unknowns | `discretize` | first residual call (compile) | residual eval |
|---|---|---|---|---|
| 256 | 257 | 0.06 s | 0.48 s | 26 µs |
| 1024 | 1025 | 0.15 s | 1.22 s | 28 µs |
| 4096 | 4097 | 1.8 s | 7.1 s | 110 µs |

On `CuArray`s with `allowscalar(false)`, both the DAE residual and the `mtkcompile`-based ODE RHS throw "Scalar indexing is disallowed". So the whole `discretize`/`ODEProblem` family is CPU-only, and compile cost grows superlinearly.

**Experiment 2: 1D Maxwell (E, B) on a Yee-staggered periodic grid.** Symbolic output: 6 slice equations plus 2 ghost-cell equations, independent of N. `staggered_groups` builds the two leapfrog `update!` functions.

| N | states | `DAEProblem` build | `staggered_groups` build | first `update!` call | 100 leapfrog steps, CPU |
|---|---|---|---|---|---|
| 1024 | 2050 | 0.3 s | 0.1 s | 0.22 s | 0.2 ms |
| 4096 | 8194 | 3.6 s | 0.1 s | 0.07 s | 0.5 ms |
| 16384 | 32770 | 59.4 s | 0.3 s | 0.07 s | 1.6 ms |

Two different things are happening. The `DAEProblem` residual codegen is the O(N) expansion and grows roughly as N² in wall time (59 s at 16k states; a 2D grid with 10⁶ cells is out of reach). The leapfrog `update!` codegen, by contrast, is resolution-independent: constant compile time and broadcast-speed execution (about 0.5 ns per state per step on CPU). With default codegen flags, `update!`, the group `f`, and the `SplitODEProblem` `f1` all scalar-index on the GPU. Periodic ghost cells come out as "algebraic" states that `update!` never writes, so a leapfrog loop needs a manual ghost copy per step.

**Experiment 3: GPU leapfrog with `staggered_groups(prob; scalarize_single_element_regions = false)`.** Symbolics 7.48 documents this flag precisely for GPU arrays, and MOL forwards it.

The generated `update!` for one group is three fused broadcasts over views (the interior slice and the two one-element boundary regions), 249 expression nodes regardless of N, each written through `SymbolicUtils.Code.fused_materialize_noalias!`. Results on `CuArray`s with `allowscalar(false)`:

| Check | Result |
|---|---|
| `update!` for both groups on GPU | runs, no scalar indexing |
| 10 leapfrog steps GPU vs CPU | max difference 5.6e-16 |
| 2000 steps at N = 4096 (8194 states) | CPU 8.1 ms, GPU 25.0 ms (4.1 µs vs 12.5 µs per step) |
| MTK-generated group `f` on GPU | scalar indexing error |

The GPU time at this size is launch overhead (six tiny kernels per step); it would only win at 2D sizes. So the Yee leapfrog path of MOL is GPU-capable and resolution-independent, but with two caveats: `staggered_groups` takes a `DAEProblem`, whose residual codegen is the O(N²) build above (59 s at 16k states), although the groups themselves only need the `System` and `u0`, so a small MOL change would remove that; and the flag is not the default, so a one-line `scalarize_single_element_regions = false` is needed.

### 4.2 Verdict

Two different codegen paths exist and the answer differs for each.

- The general path (`discretize`, `DAEProblem`, `ODEProblem(mtkcompile(sys))`): array form only at the symbolic level; generated code is O(N) in size, scalar-indexed, CPU-only, with superlinear compile time. Not usable for a GPU field solver at PIC resolutions. Fixing it is upstream work (MTK #5139), worth tracking, not worth blocking on.
- The staggered Yee leapfrog path (`staggered_groups` + `scalarize_single_element_regions = false`): resolution-independent fused-broadcast code that runs on the GPU. This is a genuine Maxwell solver front-end. What it lacks: a constructor that skips the O(N²) `DAEProblem` residual build (small MOL PR), ghost-cell updates for periodic BCs inside the leapfrog, a moving window, current sources from particles, and any AD story on the GPU (Enzyme through fused broadcasts on CuArrays is untested here).

For ParticleMesh that means: MOL can provide the Maxwell update symbolically if we want MTK composability, but it does not shorten the PIC project, because the particle half, the window and the AD plumbing are the work. Where MOL fits without caveats: the reduced fluid/envelope models of §5 at CPU-scale N, with AD through SciMLSensitivity.

## 5. Proposed approach: a model ladder, with the PIC core written directly

Recommendation: write the PIC core as plain Julia with KernelAbstractions kernels (CUDA on the 4080 SUPER, ROCm on rented MI300X, both backends already exercised by EDM), use ForwardDiff dual numbers through the whole time loop as the first AD mode, and keep MTK/MOL for the reduced models and parameter plumbing. Reverse mode (Enzyme on kernels, or Reactant for a whole-array formulation) is phase 3, once a smooth objective exists and parameter counts justify it.

Why not Reactant first: it is the closest analogue to JAX-in-Cell, but multi-D deposition needs sorting plus scatter-add, exactly the ops with open reverse-mode regressions (Oct 2026). Forward mode with KA kernels has no such dependency and gives bitwise-reproducible gradients.

Model ladder (each level validates the next and already produces optimization results):

0. **1D nonlinear wake + test electron** (ODEs in ξ; Sprangle/Esarey quasi-static wake equation, Gaussian or arbitrary pulse shape). Cost: ms. Gradients via ForwardDiff/SciMLSensitivity. Compare optimum vs the Valenta 2025 analytic max-energy expressions. Answers "is the gradient of electron energy informative?" within days.
1. **2D r-z quasi-static fluid wake + paraxial laser envelope** (the Wake-T class of model; a PDE system where MOL applies at N ~ 10⁴). Adds self-focusing, diffraction, depletion. Cost: seconds. Still fully differentiable on CPU.
2. **1D3V EM PIC** (KA kernels, Duals, GPU). Validate against JAX-in-Cell's benchmarks (Landau damping, two-stream, Weibel) and against level 0 for the wake. Measure gradient noise vs particle count and horizon.
3. **2D Cartesian EM PIC** with moving window, density profile, laser antenna, immobile ions, injected test bunch, quadratic/cubic shapes, Esirkepov current, binomial filter. Validate against Smilei/PIConGPU runs (containerized). First real optimization: bunch energy vs (a₀, w₀, τ, focus, n₀, ramp length).
4. **Quasi-3D RZ modes**, Lehe/PSATD solver for numerical Cherenkov, reverse mode with checkpointing for pulse shaping, smoothed objectives for self-injection.

## 6. Requirements

Physics
- R1 Target geometry for the first results: 2D Cartesian first (cheaper, method paper); the envelope model instead of quasi-3D RZ (architecture §3).
- R2 Objective: externally injected bunch energy (smooth) first; self-injection cut-off energy only via smoothed surrogates later (the objective decision itself is deferred, open-items D).
- R3 Parameters: a₀, w₀, τ, focus position, chirp; n₀, ramp lengths, channel depth. Count ≤ 20 → forward mode suffices.
- R4 Injection: external bunch and density downramp in scope; ionization injection and QED out of scope for v1.

Numerics
- R5 Yee + Boris + Esirkepov, shape order ≥ 2 (gradient smoothness), moving window, binomial filtering; Lehe/PSATD later.
- R6 Precision: FP32 particles on the 4080 SUPER (FP64 is 1/64 rate on Ada), FP64 on rented datacenter GPUs; deposition must be reproducible across both. EDM's precision-floor work applies.
- R7 Validation suite: classic 1D benchmarks, 1D wake vs analytic, 2D LWFA vs Smilei, gradient vs finite differences at every level.

Software
- R8 Package layout: MTK-free GPU core; optional MTK/MOL front-end for the fluid models and for composing with EDM models; LaserTypes.jl for laser definitions; EDM-style run manifests.
- R9 AD: ForwardDiff Duals end-to-end in phase 1–3; Enzyme/Reactant reverse mode in phase 4 with per-step checkpointing.
- R10 Optimization layer: Optimization.jl with gradients; BO baseline (optimas or Julia BO) on the same objective for a fair comparison, since BO is what the field uses.
- R11 Modular physics (A9, architecture §0): an engine (particles, grids, transfers, loop, AD, backends) separate from physics modules (interactions, solvers, pushers, sources); the LWFA module first, an electrostatic module as the proof of the split; EDM as an external-field and force module later; interfaces that do not preclude pairwise, Molly-style interactions.
- R12 Reactant-compilable core (A9, architecture §0.4): static shapes, masks instead of resizing, both deposition variants, no scalar indexing outside kernels, a Reactant CPU smoke test in CI from the first package; a particle-workload stress-test suite across KA-CUDA, Reactant-CUDA and Reactant-CPU as a deliverable in its own right.
- R13 Problem description separated from numerical implementation (principle, A10): physics modules expressible as descriptions (MTK systems, `PDESystem`, equation types, source and boundary data) with the numerical choices (shape functions, Yee/FEM/DG, pusher, deposition variant, backend) made separately; hand-written kernels are the production path and generated code replaces them one slot at a time.

## 7. Risks

- Gradient pathologies (thresholds, chaos, particle noise) could make the LWFA objective useless for gradient descent beyond the fluid levels. Mitigation: the ladder; smoothed objectives; publish the gradient-quality study either way.
- Forward mode memory: p+1 copies of fields and particle buffers; at 2D scale with p=10 on 16 GB this limits resolution. Mitigation: chunk parameters, FP32, a rented GPU with more memory.
- Enzyme/Reactant GPU reverse mode is moving fast with open correctness bugs; do not design phase 1–3 around it.
- Dev effort: a validated 2D GPU PIC with moving window is months of work even without AD. The ladder keeps results flowing meanwhile.
- MethodOfLines as a core would import the codegen scaling problem; keep it to the reduced models unless MTK #5139 ships.

## 8. Sources

- MOL v1 announcement: https://discourse.julialang.org/t/ann-methodoflines-v1-symbolic-arrays-massive-compile-speedups/138974
- MOL v1.0.0 release: https://github.com/SciML/MethodOfLines.jl/releases/tag/v1.0.0
- MOL #691 (codegen scalarized): https://github.com/SciML/MethodOfLines.jl/issues/691
- MTK #5139 (whole-array codegen tracking): https://github.com/SciML/ModelingToolkit.jl/issues/5139
- JAX-in-Cell: https://arxiv.org/abs/2512.12160
- Differentiable programming for plasma physics (review): https://arxiv.org/abs/2603.11231
- ADEPT: https://github.com/physicistphil/adept
- Valenta et al. 2025, PRAB 28, 094601: https://arxiv.org/abs/2501.06069
- Valenta et al. 2026, MLST 7, 025030: https://arxiv.org/abs/2512.10728
- BO via spectral pulse shaping (abstract only): https://arxiv.org/abs/2512.09125
- Multi-objective multi-fidelity BO: https://arxiv.org/abs/2210.03484
- Irshad thesis 2025 (search result only): https://edoc.ub.uni-muenchen.de/36185/1/Irshad_Faran.pdf
- optimas: https://github.com/optimas-org/optimas
- Wake-T docs: https://wake-t.readthedocs.io/ ; gridless quasi-static model: https://arxiv.org/abs/2603.16623
- ParticleInCell.jl: https://github.com/JuliaPlasma/ParticleInCell.jl ; GEMPIC.jl: https://github.com/JuliaVlasov/GEMPIC.jl ; JustPIC.jl: https://github.com/JuliaGeodynamics/JustPIC.jl
- Enzyme/KA/CUDA status threads: https://github.com/JuliaGPU/KernelAbstractions.jl/issues/924 , https://github.com/EnzymeAD/Enzyme.jl/issues/3850 , https://github.com/EnzymeAD/Enzyme.jl/issues/3851 , https://github.com/EnzymeAD/Enzyme.jl/issues/3807 , https://github.com/JuliaGPU/CUDA.jl/pull/3150
- Reactant gather/scatter threads: https://github.com/EnzymeAD/Reactant.jl/issues/3589 , https://github.com/EnzymeAD/Reactant.jl/issues/3550 , https://github.com/EnzymeAD/Reactant.jl/issues/3276
- Reactant kernel compilation, raising and sharding: `docs/src/tutorials/kernels.md`, `docs/src/tutorials/raising.md` and `ext/ReactantKernelAbstractionsExt.jl` in Reactant 0.2.289 (rendered under https://enzymead.github.io/Reactant.jl/stable/tutorials/ ; sharding page: https://enzymead.github.io/Reactant.jl/stable/tutorials/sharding )
- Oceananigans Reactant extension: https://github.com/CliMA/Oceananigans.jl/blob/main/ext/OceananigansReactantExt/Architectures.jl ; DJ4Oceananigans (JuliaCon 2026): https://pretalx.com/juliacon-2026/talk/3RDRVU/ ; FOSDEM talk on Reactant with Oceananigans: https://fosdem.org/schedule/event/TYF8ZD-accelerating_scientific_code_on_ai_hardware_with_reactant_jl
- Molly.jl documentation: https://juliamolsim.github.io/Molly.jl/stable/documentation/ and https://juliamolsim.github.io/Molly.jl/stable/differentiable/ ; sources of 0.23.3 (`ext/MollyCUDAExt.jl`, `ext/MollyEnzymeExt.jl`, `src/interactions/ewald.jl`)
- pmwd: https://arxiv.org/abs/2211.09958 (adjoint method: https://arxiv.org/abs/2211.09815 ); JaxPM (early snapshots only): https://git.aquila-consortium.org/guilhem_lavaux/JaxPM ; FastMultipole.jl: https://github.com/byuflowlab/FastMultipole.jl ; ParticleMeshEwald.jl (Flatiron, unregistered, adjacent name): https://github.com/flatironinstitute/ParticleMeshEwald.jl
- OrdinaryDiffEq PR #4436 (Reactant shares the ordinary solver paths): https://github.com/SciML/OrdinaryDiffEq.jl/pull/4436 ; tracking issue for untraceable options: https://github.com/SciML/OrdinaryDiffEq.jl/issues/4657
- Trixi.jl scaling paper: https://arxiv.org/abs/2607.25866 ; TrixiParticles on GPUs (JuliaCon 2026): https://pretalx.com/juliacon-2026/talk/Y7LGHP/ ; PointNeighbors.jl: https://github.com/trixi-framework/PointNeighbors.jl ; TrixiParticles poster (Zenodo, March 2026, search result only): https://zenodo.org/records/18846554
- Background reading: Debus et al. 2019 "Circumventing the dephasing and depletion limits of LWFA"; Palastro et al. 2020 "Dephasingless LWFA"; Wenz & Karsch 2020 "Physics of LWFA" (CAS notes).

Unverified: arXiv:2608.08903 (energy-optimized scaling laws, Optimas), seen only in search results and not fetched.
