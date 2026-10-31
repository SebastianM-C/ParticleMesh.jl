# ParticleMesh: architecture draft

Companion to `landscape-and-requirements.md`. Status: design record; §0 and the decisions A1 to A14 in `open-items.md` are binding.

Constraints:
- keep a symbolic-numeric angle possible without blocking on it;
- start with Cartesian 2D, reach multi-GPU 3D eventually, quasi-3D RZ optional;
- about 3 parameters with ForwardDiff first; find out later where Enzyme and Reactant help;
- prototype on the local RTX 4080 SUPER, produce on rented datacenter GPUs (MI300X or NVIDIA instances, through EDM's campaign backends);
- objective design and gradient-versus-Bayesian are discussed separately (not covered here);
- long term (A9) a generic particle–mesh framework and a showcase and stress test for Julia and Reactant compiler technology, with modular physics, a path to EDM integration and possible connections to Molly.jl; LWFA is the first physics module, not the definition of the package (§0).

Prior work this builds on (ElectronDynamicsModels.jl): MTK component models compiled to eight ODEs, KernelAbstractions kernels portable across NVIDIA and AMD, electron sharding across eight datacenter GPUs at 98% kernel efficiency, the Landau–Lifshitz force generated symbolically for any field, the run-manifest provenance layer, and the outlook naming differentiability of the field kernels as the next step.

## 0. Scope: a generic particle–mesh framework

Scope (A9): long term the package is a showcase and stress test for Julia and Reactant compiler technology, the physics is modular so it can be extended and perhaps integrate with EDM, and connections to packages such as Molly.jl are worth exploring. All of it is long term. The near-term plan (2D LWFA, ForwardDiff with about three parameters, KernelAbstractions on the 4080) stands. What changes now is only what would be expensive to retrofit: the split between engine and physics modules, the rules that keep the Reactant path open, and the name.

### 0.1 Engine and physics modules

The engine owns what every particle–mesh method shares: particle sets (SoA, fixed capacity, alive mask, cell sorting and compaction), structured grids with staggering and guard cells, the two transfers particle→mesh (deposition) and mesh→particle (gather) with shape functions, mesh operators (stencils, FFT-based solves), the time-loop harness, diagnostics, checkpointing, the parallel exchange, and the differentiation and backend machinery. A physics module supplies the specific content: which quantities are deposited, which mesh equations are solved, which force acts on a particle and which pusher integrates it, which sources and boundaries exist.

| Module | Deposit | Mesh solve | Gather / force | Pusher | Status |
|---|---|---|---|---|---|
| Electromagnetic PIC for laser–plasma (LWFA) | current (Esirkepov), optionally charge | Yee FDTD, moving window, antenna, absorbing layers | E, B at the particle; Lorentz force | Boris | first module, phases 1–3 |
| Electrostatic PIC | charge | Poisson (FFT or LinearSolve) | E; electrostatic force | leapfrog / Boris | cheap second module: the JAX-in-Cell and ParticleInCell.jl benchmark set, and the CI-sized proof that the split is real |
| Test particles in prescribed fields | none | none (analytic fields from LaserTypes.jl / EDM evaluators) | MTK-generated forces incl. radiation reaction | Boris or per-particle RK | the EDM integration path (L3); first as the "external field" source inside the LWFA module |
| Envelope (ponderomotive) laser | susceptibility | envelope equation plus Yee for the wake | ponderomotive force, |a|²-corrected γ | modified Boris | phase 5 (B2) |
| Pairwise and P³M corrections | charge (long-range part) | Poisson / Ewald mesh part | mesh force plus short-range pairwise force | Verlet family | the Molly.jl bridge, long term (§0.3) |

Only the LWFA module has a schedule. The table exists so that the engine's interfaces are shaped by more than one module from the start, which is the cheap part; writing the electrostatic module early is also cheap and is the best test that the split holds.

Interface sketch. The main integration goal is SciML (A9); Molly compatibility is a nice-to-have, not a design driver (§0.3). So the container is a problem-like object that SciML can `solve` through the `DiscreteProblem`/`FunctionMap` harness (§0.6), parameters follow the SciML conventions (`p`, MTK parameter identities, `remake`), diagnostics are callbacks or saved states, and the reduced models of the ladder are ordinary SciML problems. Within that frame the vocabulary is close to Molly's, because both are particle codes: a container holding species, grids and fields; "interactions" as the pluggable unit in three kinds: mesh-mediated (deposit → solve → gather, the PIC kind), pairwise (short-range, needs a neighbour structure, the MD kind) and external (analytic fields and forces, the EDM kind); a `step!` that composes them; loggers for diagnostics; a simulator type that owns the time loop. Element type `T`, dimension `N` and the array type (hence the backend) are type parameters of the container, as decided in A8; unit handling stays in the user layer (A1), which also matches Molly's experience that units and differentiable simulation do not mix.

### 0.2 The compiler showcase and stress test

The same kernel source should run on three execution paths: (i) KernelAbstractions native on CUDA and ROCm, the production path of phases 1–3; (ii) Reactant, where the same KA kernels are traced, raised to StableHLO, fused with the surrounding array program, sharded across devices and differentiated by Enzyme-MLIR; (iii) Enzyme on the native path.

Verified mechanism (Reactant 0.2.289 sources; 0.2.303 registered): `ReactantKernelAbstractionsExt` defines `ReactantBackend <: KA.GPU`; `KA.get_backend` of a traced or concrete Reactant array returns it; a kernel launch inside `@compile` is captured through `Reactant.ka_with_reactant`; CUDA.jl must be loaded for kernel compilation even without an NVIDIA device; raising is automatic on TPU and opt-in (`raise=true`) elsewhere; AD of kernels inside Reactant works only on raised kernels (`raise=true, raise_first=true`); the docs state that not every class of kernel is raisable yet. Sharding needs the IFRT runtime preference, `Sharding.Mesh` plus `NamedSharding`, arrays created with `to_rarray(; sharding)`, communication inserted by the compiler, one compiled function per sharding, multi-host supported.

The precedent is Oceananigans: `OceananigansReactantExt` defines the `ReactantState` architecture whose `device` is Reactant's `ReactantBackend`, and its `Distributed(::ReactantState)` builds a `Sharding.Mesh` over `(:x, :y[, :z])`; DJ4Oceananigans (JuliaCon 2026) reports accurate Enzyme+Reactant gradients on several configurations and runs on CPU, GPU and TPU; the FOSDEM talk shows the model retargeted to thousands of TPUs. Oceananigans is a stencil code. A particle–mesh code stresses exactly what stencils do not:

- data-dependent indexing in gather and deposition (scatter with collisions: atomics on the native path, scatter-add or sort plus segmented reduction on the raised path);
- sorting, compaction and injection at fixed capacity (dynamic particle counts expressed through masks and a count scalar, no `push!` or `resize!` inside a traced step);
- the moving window (index shift of fields, refill at the front) and, under sharding, particle migration between shards next to halo exchange;
- mixed precision and compensated accumulation;
- reverse mode through scatter and gather over a long loop with checkpointing (the open Reactant regressions #3589, #3550, #3276 sit precisely here);
- compile time of a full step with a dozen kernels.

Deliverable of this angle: a particle-workload benchmark and correctness suite (gather, both deposition variants, sort and compaction, window shift, halo plus migration, one full step, one gradient) run on KA-CUDA, Reactant-CUDA and Reactant-CPU, with results recorded as run records in the trace graph and failures turned into upstream issues with minimal reproducers. It has value for the Julia ecosystem independent of the LWFA result, the way Oceananigans serves as Reactant's stencil benchmark.

### 0.3 Molly.jl: a nice-to-have compatibility target, not a design driver

Decision (A9): the main integration goal is SciML; Molly compatibility is a nice-to-have, kept in case other interesting physics comes up; do not design around it. What follows is therefore reference material and a list of things not to preclude.

Molly 0.23.3 (verified from its documentation and its sources): `System` holds `atoms`, `coords`, `velocities`, `boundary`, `pairwise_inters`, `specific_inter_lists`, `general_inters`, `neighbor_finder`, `loggers`, `constraints`; general interactions implement the AtomsCalculators `forces!` interface and `System` implements the AtomsBase `AbstractSystem` interface; simulators (`VelocityVerlet`, `Langevin`, ...) run through `simulate!`; the backend follows the array type (a CUDA fast path in `MollyCUDAExt`, a KernelAbstractions path for all backends); long-range electrostatics are the `Ewald` and `PME` general interactions, and PME is itself a particle–mesh method with KA kernels; reverse-mode differentiable simulation goes through `MollyEnzymeExt` with `autodiff(set_runtime_activity(Reverse), loss, ...)`, forward mode exists, memory is named as the limit (Checkpointing.jl suggested), loggers are ignored by AD, units are unsupported in differentiable runs, velocity Verlet in NVE gives exploding gradients, adjoint sensitivity is not available, and the stated golden rule is to check against finite differences.

To take now: the container / interaction / simulator / logger vocabulary, the array-type-selects-backend convention, and Molly's AD lessons (chunked losses, loggers outside the differentiated path, finite-difference checks as a gate) for L2. To explore later: (a) exposing a particle–mesh field solve as an AtomsCalculators general interaction so Molly can use it as a long-range solver, and conversely using Molly's pairwise interactions and neighbour finders as the short-range part of a P³M scheme here; (b) a shared benchmark on deposition and gather kernels, since PME and PIC deposition are the same operation; (c) whether a species here can present an AtomsBase-compatible view. None of it is scheduled; it is recorded so the interfaces do not preclude it.

### 0.4 Rules that keep the Reactant path open at no cost

1. Static shapes: fixed-capacity particle buffers, an alive mask and a count; injection and the moving window write into free slots; compaction is a stable partition within the same buffer. No data-dependent allocation inside `step!`.
2. The step is a pure function of (state, parameters) with its buffers passed explicitly: no globals, no host-side branching on device data (branch on static configuration only).
3. Both deposition variants are first-class and tested for equality: atomics (native fast path) and sort plus segmented reduction (deterministic, raisable). The raised path may also use scatter-add; its determinism is a measurement, not an assumption.
4. Scalar indexing never appears outside kernels; allocations go through `similar` or `KernelAbstractions.allocate`, so Reactant's traced `allocate` applies.
5. Reference kernels use only patterns the raiser handles: index arithmetic, loads and stores, scalar math; no shared-memory or warp-intrinsic tricks. Optimized native variants may exist behind the same interface.
6. Parameters are arrays or scalars of `T`, never baked into closures, so Enzyme-MLIR can see them.
7. A Reactant CPU smoke test (one electrostatic and one electromagnetic step compiled with `@compile` and compared with the native CPU result) runs in CI from the first work package. It needs no GPU, so the cloud sessions can run it.
8. The AD strategy sits behind an interface (`jvp(step!, state, params, seeds)`, later `vjp`): the native path implements it with the tangent-bundle layout (§4), the Reactant path with Enzyme-MLIR forward mode on the raised program. User code does not see the difference.

Long term and unscheduled: TPU runs, multi-host sharding, reverse mode over the full loop on Reactant, P³M and Molly interoperability, EDM symbolic forces inside raised kernels (B6 extended: can an MTK-generated force be raised?).

### 0.5 Problem description separated from numerical implementation

Rationale (A10) for keeping the symbolic connection: it enforces a separation of concerns between the description of the problem and its numerical implementation. That turns the symbolic front-end (L3) from a door kept open into the mechanism behind a design principle: a physics module should be expressible as a problem description (species with their equations of motion and forces as MTK systems; the mesh equations as a `PDESystem` or an equation type; sources and boundaries as data), and the numerical implementation (shape functions, Yee versus FEM versus DG, pusher, deposition variant, backend) should be a separate choice that leaves the description untouched. Hand-written kernels remain the production path; the principle fixes what the kernel interfaces must look like so that generated code can replace them one slot at a time.

Numerical backends this admits for the mesh-solve slot, with an assessment:

- Yee FDTD as hand-written KA kernels (phases 1 to 3) and MOL's `staggered_groups` fused update (verified GPU-safe; §5 and the landscape notes).
- FEM through an expected MOL Ferrite backend (nothing public as of 2026-10-30, so timing is unknown; Ferrite 1.7.0 is the current release). Assessment: an unstructured FEM Poisson or Maxwell solve is what geometry-heavy electrostatic problems need (electrodes, structures), not what LWFA needs (structured box, moving window, GPU throughput, charge-conserving deposition matched to the Yee stagger); GPU assembly in Ferrite could not be verified. Value: a second mesh backend that proves the slot and the description/implementation split, for the electrostatic module; no role in the LWFA module.
- DG through Trixi.jl: the inspected copy (0.11.1; 0.17.18 registered) carries a 1D Maxwell equation type; DG-PIC is an established method family, but charge conservation and the particle–mesh coupling on DG meshes are a research topic of their own, and Trixi's GPU support (TrixiCUDA, CUDA-only, unregistered; a multi-GPU prototype) is not at the level of the KA path. Value: a third mesh backend and the precedent of `semidiscretize` turning a mesh solver into an `ODEProblem` right-hand side; not for the first module.
- Fast multipole as a mesh-free alternative for the long-range slot: FastMultipole.jl 2.3.0 (BYU FLOW Lab; Laplace kernel, CPU-parallel expansions, CUDA only for the direct interactions, ForwardDiff- and ReverseDiff-compatible per its README) and ExaFMMt.jl 0.1.2 (wrapper). Not needed for LWFA; relevant only if a pairwise or gravity module wants long-range sums without a mesh.
- TrixiParticles.jl 0.5.3 (SPH with KA kernels on NVIDIA, AMD and Apple per the March 2026 poster and the JuliaCon 2026 talk) and PointNeighbors.jl 0.6.7 (`GridNeighborhoodSearch` with `FullGridCellList` marked GPU-compatible, GPU support still "work in progress"): precedents for a KA particle framework whose coupled particle systems form one `Semidiscretization`, and a candidate neighbour search for the pairwise and P³M slot (§0.3) to compare with the engine's own cell sorting. SPH itself is a different physics (fluids whose particles carry the state) and enters only if a module wants it.

### 0.6 Time integration on Reactant: the harness ties the ladder together

`sciml-reuse.md` §1 wraps the hand-written step in a `DiscreteProblem` with `FunctionMap` (OrdinaryDiffEq) or `SimpleFunctionMap` (SimpleDiffEq), and the reduced models of the ladder are ODE problems. Two facts make this harness the natural Reactant entry point: SimpleDiffEq already runs under Reactant (prior work, to be reused), and OrdinaryDiffEq PR #4436 "Share ordinary solver paths with Reactant" (draft, opened 2026-08-30, updated 2026-10-17, 33 files) makes explicit solves inside Reactant compilation run the ordinary `solve!` loop with step control, with untraceable options (implicit algorithms, callbacks, `saveat`, user `tstops`, ...) throwing before tracing and an `OrdinaryDiffEqCoreReactantExt` for the traced primitives. So on the Reactant path the PIC step runs under `SimpleFunctionMap` or a `@trace` loop now, and under the shared OrdinaryDiffEq loop once #4436 lands (whether it covers `FunctionMap` is part of the B9 spike). The level-0 and level-1 reduced models run on Reactant through the same solvers, so the whole ladder, not only the PIC core, sits on the compiler path. Diagnostics then live outside the traced solve, as saved states or as reductions inside the step; Molly's lesson that loggers are ignored by AD applies.

## 1. Layering

The rule: the hot loop is plain numeric kernels over typed arrays; everything symbolic generates code that plugs into those kernels, never the other way around. That is the EDM pattern applied to a grid code.

```
L4  Optimization / campaigns     Optimization.jl, BO baseline, run manifests, cloud (EDM backends)
L3  Symbolic front-end (opt.)    MTK particle models → force kernels; MOL staggered groups → Maxwell update;
                                 LaserTypes.jl / Symbolics expressions → laser & density kernels; parameter identities
L2  Differentiation              tangent-bundle layout (value + P tangents), Dual-number push, per-component atomics;
                                 Enzyme forward/reverse on kernels; Reactant path: same KA kernels traced, raised, sharded (smoke test from day one)
L1  Parallel layout              Grid{N} with guard cells, slab/block decomposition, halo + particle exchange,
                                 moving window as a global shift; single device → multi-device node → MPI
L0  Core numerics                SoA fields and particles, KernelAbstractions kernels: gather, push, deposit,
                                 Yee E/B, filter, inject, absorb; dims-generic via Val{N}; physics modules plug in via §0.1
```

L0 and L1 have no symbolic dependency and are the first deliverable. L3 is an optional package extension that reuses EDM. The engine/module split of §0.1 cuts across the layers: the engine is the L0–L2 machinery, a physics module is a set of L0 kernels (and, through L3, their generators) plugged into the interaction interfaces.

## 2. Core numerics (L0)

Data layout
- Fields: one `AbstractArray{T,N}` per component (Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, ρ optional), Yee offsets carried as type-level `Val` tuples, guard cells sized by the shape-function order. 2D is 3D with one collapsed axis (2D3V): all field and momentum components exist, variation only along the resolved axes. This is how Smilei and PIConGPU do it and keeps one kernel source for N = 1, 2, 3.
- Particles (per species): SoA buffers with `cell::Int32` per axis plus an in-cell offset `T` in [0, 1), momentum `T × 3`, weight, alive mask, optional tag. The cell+offset split is what keeps FP32 usable over a long moving window (PIConGPU layout); FP32 is required on the 4080 SUPER (FP64 at 1/64 rate), FP64 is available on rented datacenter GPUs (MI300X, NVIDIA data-center cards). `T` is a type parameter throughout and is chosen at the user level; nothing assumes Float64 or Float32. The same kernels must run with Float32, Float64, DoubleFloats (Double32 is the useful one on consumer GPUs, since it keeps FP32 throughput) and ForwardDiff duals; the deposition kernel selects its accumulation path by a trait on `T` (atomics for primitives and, component-wise, for duals; sorted segmented reduction for compensated types whose addition is not component-wise associative). See `open-items.md` A8.
- Fixed-capacity buffers with periodic compaction and cell sorting (supercells). Sorting also enables a deposition variant without atomics (segmented reduction), which matters for the Reactant path and for deterministic bitwise reproducibility.

Kernels (KernelAbstractions, `foreachindex` style as in EDM)
- `gather!`: tensor-product shape functions (order 2 or 3; order 1 only for tests, its kinks hurt gradients), per-component staggering.
- `push!`: Boris by default; the force term is a pluggable callable so that L3 can substitute an MTK-generated force (Lorentz + Landau–Lifshitz, later quantum corrections). For non-Lorentz forces a generic RK/implicit per-particle step replaces Boris.
- `deposit_current!`: Esirkepov in N-D (implement the 3D formula, specialize the collapsed axis), atomics on primitive types only.
- `yee_E!`, `yee_B!`: curl stencils over guard cells; Lehe or PSATD solver later for numerical Cherenkov.
- `filter!` (binomial), `absorb!` (PML or simple damping layers), `inject!` (plasma at the window front, laser antenna at a plane), `window_shift!`.
- Time loop: an explicit Julia loop with hooks for diagnostics and checkpoints. No ODE solver in the loop; DifferentialEquations only enters at the reduced-model levels.

Validation targets: Landau damping, two-stream, Weibel (JAX-in-Cell set), 1D wake against the analytic nonlinear wake, 2D LWFA against Smilei and PIConGPU (containerized setups exist), energy and charge conservation.

## 3. Parallel layout (L1): 2D now, multi-GPU 3D later

- `Grid{N}` carries a global extent, a local block, and guard regions. Single-device runs use the same code with a no-op exchange.
- Decomposition: start with a 1D slab decomposition along the propagation axis. It matches the moving window (ranks form a ring; the window advance is a whole-cell shift where the front rank injects and the back rank drops), keeps halo exchange one-dimensional, and already covers one node of eight GPUs. 3D block decomposition and load balancing (particles concentrate in the bubble and bunch) come after.
- Exchange primitive: `exchange_halos!(fields)` and `exchange_particles!(species)` as the only communication points. Implementations in order: none (single device); device-to-device copies between devices of one node driven by one Julia task per device (EDM's sharding pattern, generalized from "split the electrons" to "split the domain"); CUDA-aware MPI.jl for multi-node. ImplicitGlobalGrid.jl is a candidate for the structured-grid halo part (GPU-aware MPI, CUDA and AMDGPU); particle migration is ours.
- Vendor portability stays free through KernelAbstractions plus GPUDiagnostics.jl, so the MI300X cloud route remains open.
- Prototyping without a multi-GPU box: run the decomposition on the KA CPU backend with MPI on a many-core CPU, or several ranks on one GPU, before renting a multi-GPU instance.

### On skipping quasi-3D

RZ with azimuthal modes is a separate numerics path (cylindrical Maxwell, mode-decomposed deposition) that does not advance the multi-GPU 3D goal. The cost reducer that keeps Cartesian geometry and the same kernels is the laser envelope (ponderomotive guiding center) model: it removes the λ₀ resolution along the propagation axis, 10 to 30 times fewer cells, and the ponderomotive force is smoother than the resolved laser oscillation, which is good for gradients. Suggested path: 2D full PIC → 2D envelope → 3D envelope on one node → 3D full PIC for production checks. Validity (a₀ up to a few, pulse length much larger than λ₀) covers the LWFA regime of interest.

## 4. Differentiation (L2)

### Forward mode with about 3 parameters (phase 1)

- Tangent-bundle layout rather than `CuArray{Dual}`: every field and particle array exists as a value array plus `P` tangent arrays. Inside `push!` and `gather!` the thread loads value and tangents into a local `ForwardDiff.Dual{Tag,T,P}`, does the arithmetic, and stores the components back. `deposit_current!` adds value and each tangent with separate primitive atomics, since atomics on `Dual` structs are not available. `P = 0` recovers the primal code with no overhead.
- Cost: memory and FLOPs scale as `P + 1`. A 2D run at 2048 × 512 cells, 16 particles per cell, FP32, P = 3 needs roughly 2 GB for particles, which fits the 4080.
- Which parameters are active is a setup-time decision (a₀, w₀ or τ, n₀, focus position, ramp length). Through L3 these are symbolic parameter identities, so "active parameter" is a tag on the MTK parameter and the tangent seeding is generated, not hand-written.
- Noise and non-smoothness (cell crossings, injection events) are discussed with the objective; the layout above is neutral to that discussion.

### Where Enzyme helps

- Enzyme forward mode with `BatchDuplicated` shadows over the same SoA layout could replace the hand-rolled Dual load/store with less code and possibly better register use. Worth a phase 2 comparison on `push!`.
- Reverse mode becomes necessary when the parameter count grows (pulse shaping with 10 to 100 spectral coefficients, a spline- or network-parameterized density profile). Plan: `Enzyme.autodiff(Reverse, step!, ...)` per time step with a checkpointing schedule over the loop (store full state every k steps, recompute in between), custom rules for `deposit_current!` if Enzyme does not handle the atomics (its adjoint is a gather, easy to write by hand). Open Enzyme-on-CUDA bugs as of Oct 2026 (`cis` in kernels, `fill!`, tape types) are tracked in the landscape notes; none is a blocker for forward mode.
- Enzyme on MTK-generated scalar force functions is well-trodden; the L3 symbolic force is not a risk for AD.

### Where Reactant helps

- Reactant compiles whole Julia array programs to XLA with Enzyme-MLIR for both AD modes, supports `@trace` loops with automatic checkpointing, and has multi-device sharding. A PIC written as array operations (sort, segmented reduction for deposition, gather by indexing, stencil broadcasts) would get fusion, sharding and reverse mode in one go. That is the JAX-in-Cell route and the most direct bridge to the symbolic layer, because generated array code (including MOL's fused staggered update) can be traced.
- Risks as of Oct 2026: gather/scatter under reverse mode has open correctness and performance regressions, compile times, and control-flow restrictions. Since A9 Reactant is a declared long-term target, not an experiment, and the mechanism is verified (§0.2): the KA kernels are traced through `ReactantBackend` and raised, so no array rewrite is needed; the deposition kernel stays available in both atomic and sort-based forms so the raised path is a backend, not a rewrite; AD inside Reactant works only on raised kernels, so forward mode on that path is Enzyme-MLIR forward over the raised program, while the Dual layout above stays the native-path implementation behind the `jvp` interface (§0.4, rule 8). Whether `Dual`-valued locals inside a kernel survive raising is untested and part of the B9 spike. The first Reactant target is the 1D3V level, where parity with JAX-in-Cell is a day's work; Reactant on CPU needs no GPU, so a smoke test belongs in CI from the first package.

## 5. Symbolic front-end (L3): the door kept open

The principle this layer serves is stated in §0.5: the symbolic layer is how problem description and numerical implementation are kept apart, which is why it stays in the design even though hand-written kernels are the production path.

What EDM already gives, and how it maps onto PIC:

| EDM today | ParticleMesh use |
|---|---|
| `ChargedParticle(external_field, radiation_model)` composed from `ParticleDynamics`, `EMFieldDynamics`, forces | same components with a `GridField` input in place of `external_field`: E and B at the particle are kernel inputs; `mtkcompile` + `build_function` yield a device-callable force or RHS |
| Landau–Lifshitz generated for any field | radiation reaction in the pusher for free; quantum-corrected LL later as one more component |
| LaserTypes.jl fields, unit systems | laser antenna/initial fields and density profiles as symbolic expressions compiled to kernels; SI/natural units handled once |
| parameters via SymbolicIndexingInterface, tunables | the active-parameter set for the tangent layer |
| run manifests, campaigns, dashboard | unchanged |

On the grid side, MOL's `staggered_groups` with `scalarize_single_element_regions = false` produces resolution-independent fused-broadcast Yee updates that run on the GPU (measured 2026-10-30). It can be an alternative Maxwell backend behind the same `yee_E!`/`yee_B!` interface once (a) a constructor from the `System` avoids the O(N²) `DAEProblem` residual build (small MOL PR), (b) current sources and guard-cell exchange are added. ModelingToolkit #5139 would extend this to arbitrary array models; worth tracking, not worth waiting for.

What this buys: composable physics models, units, documentation value, and symbolic parameter identities for AD; what it costs: nothing on the hot loop, since the generated code is scalar per particle or fused per slice.

## 6. Hardware mapping

| Stage | Machine | Precision | Notes |
|---|---|---|---|
| L0/L1 development, 1D and 2D | local 4080, KA CPU backend for tests | FP32 (FP64 for CPU reference) | 16 GB limits 2D with P = 3 to ~10⁷ particles |
| 2D campaigns, envelope 2D/3D prototypes | local 4080, or a rented single-GPU instance when 16 GB is not enough | FP32 | consumer cards keep the FP64 caveat |
| 3D, multi-GPU, production | rented multi-GPU instance (MI300X via KA/ROCm, or NVIDIA) | FP64 available | slab decomposition over one instance's devices first, MPI later |

## 7. Suggested order of work

1. L0 core, dims-generic, CPU backend, 1D3V and 2D3V, benchmark suite (JAX-in-Cell set), bitwise-reproducible runs with manifests; the engine/module split with the electrostatic module as the second module; the Reactant CPU smoke test in CI (§0.4).
2. CUDA backend on the 4080; 2D LWFA with moving window, laser antenna, density ramp, test bunch; cross-check against Smilei.
3. Tangent layer with P = 3; gradient checks against finite differences (common seeds) at 1D and 2D.
4. L3 spike: EDM `ChargedParticle` with `GridField` → force kernel; MOL staggered Yee as alternative backend.
5. Envelope model in 2D, then 3D on one node with slab decomposition.
6. Enzyme forward comparison; the Reactant stress-test suite (§0.2) on CPU and on the 4080, then 2D on Reactant-CUDA; reverse mode with checkpointing for a many-parameter objective.

Open points, deferred by decision (open-items D): the objective and the gradient-versus-Bayesian question; both only touch L4 and the diagnostics hooks, not the layering above.

Which SciML pieces each layer reuses, with what was verified in the package sources, is in `sciml-reuse.md`.
