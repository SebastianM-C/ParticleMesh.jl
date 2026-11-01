# What ParticleMesh can reuse from SciML (numerical side)

Companion to `architecture-draft.md`. Verified against package sources (OrdinaryDiffEq as of 2026-09-26, SciMLSensitivity, SimpleDiffEq, DiffEqGPU) and package docs where noted.

Short version: the PIC step itself (Yee + Boris + Esirkepov) reuses nothing and should not be forced into an ODE solver. Everything around the step can be SciML: the solve/callback/ensemble harness, test-particle integration, implicit per-particle solves, Poisson and other linear solves, forward sensitivities, the reduced-model ladder end to end, the optimization layer, and the symbolic layer.

## 1. The time loop: wrap the hand-written step, do not integrate it

| Piece | Reuse | Verdict |
|---|---|---|
| OrdinaryDiffEq generic integrators (Tsit5, symplectic `VerletLeapfrog`, ...) on the full PIC state | no | the Yee/Boris leapfrog is a specific splitting with E, B and momentum staggered in time; a generic method breaks charge conservation and multiplies RHS evaluations |
| `DiscreteProblem` + `FunctionMap()` (OrdinaryDiffEq) or `SimpleFunctionMap` (SimpleDiffEq) around `step!` | yes, as the harness | verified in OrdinaryDiffEq's `lib/OrdinaryDiffEqFunctionMap`: the in-place form calls `f(u, uprev, p, t + dt)`, i.e. "write the next state from the previous one", with `scale_by_time = false`. That is a ping-pong buffer pair; the module's step sorts the particle buffers in place (`sort_by_cell!`, `architecture-draft.md` §0.7 item 9; compaction will be in place too, §0.4 rule 1), so the harness copies the previous state into the next before calling it. Overhead per step is microseconds |
| `RecursiveArrayTools.ArrayPartition` as the state container | yes | holds the six field arrays and the particle SoA buffers (including `Int32` cell indices) as one `AbstractVector`; works with GPU arrays. To verify early: integrator initialization with mixed element types, `saveat` copies |
| `init`/`step!` integrator interface | yes | interactive stepping, diagnostics between steps, natural place to hang checkpointing |
| `DiscreteCallback`, DiffEqCallbacks `PeriodicCallback`, `SavingCallback`, `PresetTimeCallback` | yes | output cadence, running reductions (energy, spectra, bunch moments), window advance as a callback if preferred |
| `remake(prob; p = [a₀ => ...])`, SymbolicIndexingInterface on the solution | yes | parameter variation by name; `sol[Ex]`-style access via a MOL-like solution wrapper or MTK `System` metadata |
| `EnsembleProblem` + `EnsembleThreads`/`EnsembleDistributed` | yes | parameter scans with one GPU per worker process, `reduction` for on-the-fly aggregation; the EDM manifest layer plugs on top |

Alternative to the wrapper: implement `PICProblem <: SciMLBase.AbstractDEProblem` with its own `solve` and a `AbstractTimeseriesSolution`, as MethodOfLines does with `PDETimeSeriesSolution`. Same ecosystem benefits, own loop; decide when the step is stable. Fixed-capacity particle buffers with an alive mask are required either way, since the state length cannot change.

Reactant and the harness: architecture §0.6 (SimpleDiffEq under Reactant now, the shared OrdinaryDiffEq loop once PR #4436 lands; the PR in the landscape table). The reduced models of §5 go through the same solvers.

## 2. Per-particle integration

| Piece | Reuse | Verdict |
|---|---|---|
| Boris pusher | hand-written | about 30 lines, with Vay, Higuera–Cary, exact-rotation Boris and quadratic-unitary behind the same pusher type (WP-015, WP-016, `pushers-literature.md`). Each pusher is one device function for the PIC kernel, and an extension on OrdinaryDiffEqCore wraps the same functions as OrdinaryDiffEq algorithms for single particles in prescribed fields, so test-particle runs and the pusher tests use `solve` with `saveat`, callbacks, ensembles and `Vern9` as the reference. TestParticle.jl (v0.26.0, checked 2026-10-31) is not a reference: its native Boris pushers (the `BorisPushers` subpackage) are nonrelativistic and require KernelAbstractions 0.9, and its relativistic path is the Lorentz-force ODE through OrdinaryDiffEq, which `scripts/pushers/` reproduces directly. Its test suite is a useful list of orbit cases |
| DiffEqGPU `EnsembleGPUKernel` with `GPUTsit5`, `GPUVern7/9`, `GPUEM`, Rosenbrock/Rodas/Kvaerno variants; backends CUDA, ROCm, Metal, oneAPI through KernelAbstractions (verified in `src`/`ext`) | yes for test-particle stages | frozen-wake test particles (ladder level 0), EDM-style ensembles, MTK-generated RHS supported ("Symbolic-Numeric GPU Acceleration with ModelingToolkit" tutorial). Too heavy for the in-loop push |
| SimpleDiffEq kernel-level steppers `GPUSimpleRK4`, `GPUSimpleTsit5`, `GPUSimpleVern7/9`, `LoopRK4` (verified exports) | yes, as inlined stepper code | for a non-Lorentz force (Landau–Lifshitz, later quantum corrections) where Boris does not apply; the batched kernel stepper EDM experimented with is this idea |
| `SimpleNonlinearSolve` (`SimpleNewtonRaphson` on StaticArrays) | yes | implicit pushers (Crank–Nicolson Picard as in JAX-in-Cell, implicit Vay), kernel-compatible; same trick as EDM's Newton light-cone kernel |
| StaticArrays (JuliaArrays) | yes | per-particle vectors inside kernels |

## 3. Field solvers and linear algebra

| Piece | Reuse | Verdict |
|---|---|---|
| Yee update | hand-written, or MOL `staggered_groups` fused-broadcast backend (GPU-safe with `scalarize_single_element_regions = false`, measured 2026-10-30) | optional symbolic backend behind the same interface |
| PSATD / spectral solver | AbstractFFTs + CUFFT/rocFFT (JuliaGPU, not SciML) | later, for numerical Cherenkov |
| `LinearSolve.jl` with Krylov methods on GPU arrays, `SciMLOperators.FunctionOperator` for matrix-free stencils | yes | Poisson solves for the initial field of the electrostatic cases and for the initial self-fields of an injected bunch, Lorenz-gauge and divergence checks (EDM's gauge and beam verification scripts already do this kind of thing), any implicit field step |
| `NonlinearSolve`, `SteadyStateDiffEq` | yes | matched spot size and density, quasi-static equilibria of the reduced models |

Same ecosystem, outside SciML proper: Trixi.jl (DG), the Ferrite backend MOL is expected to get (FEM), and TrixiParticles.jl with PointNeighbors.jl; their assessment as mesh-solve backends and neighbour search is architecture §0.5.

## 4. Gradients

| Piece | Reuse | Verdict |
|---|---|---|
| ForwardDiff `Dual` (JuliaDiff, used throughout SciML) | yes | the tangent layer of the architecture draft; no `PreallocationTools.DiffCache`: state and workspaces are built in the execution element type (`architecture-draft.md` §0.7) |
| `ADTypes` + `DifferentiationInterface` | yes | backend-agnostic gradient calls at the optimization layer (ForwardDiff, Enzyme, Mooncake, Reactant) |
| SciMLSensitivity AD pass-through algorithms (`ForwardDiffSensitivity`, `ReverseDiffAdjoint`, `TrackerAdjoint`, Zygote/Mooncake/Enzyme pass-through) | partly | their `_concrete_solve_adjoint` methods accept `AbstractDiscreteProblem` (verified in `concrete_solve.jl`), so a PIC wrapped as `DiscreteProblem` gets `solve`-level forward sensitivities without extra code. Reverse through a GPU time loop is not what they are built for |
| SciMLSensitivity continuous adjoints (`InterpolatingAdjoint`, `GaussAdjoint`, `QuadratureAdjoint`, checkpointing options) | reduced models only | ODE-only by construction; perfect for the wake ODE and the r-z fluid level |
| Checkpointing.jl (Argonne): `@ad_checkpoint Revolve(n) for i in 1:steps`, Periodic and Online schemes, Enzyme per iteration, GPU arrays stay on device | no (`open-items.md` D): in-house checkpointing instead; listed for reference | the piece SciMLSensitivity does not provide for discrete loops; the same schedules (periodic two-level, Revolve) are implemented in-house with tests; pairs with hand-written adjoint rules for deposition if needed |

## 5. The reduced-model ladder is fully SciML

- Level 0, 1D nonlinear wake plus test electron: OrdinaryDiffEq (`DynamicalODEProblem` and symplectic methods for long test-particle runs), `BoundaryValueDiffEq` if the wake is posed with conditions behind the pulse, SciMLSensitivity adjoints, `Optimization.jl`.
- Level 1, r-z quasi-static fluid plus envelope: MethodOfLines array form on the DAE path (CPU, N ~ 10⁴ fine), SciMLSensitivity, `Integrals.jl` for energy and spectral integrals.
- Shapes as parameters: `DataInterpolations.jl` splines for pulse temporal profile and density profile, differentiable in the control points and evaluable inside GPU kernels (EDM's `lpwa.jl` already does this on CUDA); `FindFirstFunctions` for the kernel-side search.
- Closures and missing physics later: `ModelingToolkitNeuralNets` and Lux inside MTK/MOL models (MOL has a batched-callable path for Lux networks), i.e. UDEs on the reduced levels trained against PIC.

## 6. Optimization and campaigns (inventory only; the method choice is the separate discussion)

- `Optimization.jl`: LBFGS (OptimizationOptimJL), NLopt, Ipopt/MadNLP via OptimizationMOI for constraints such as fixed laser energy, Optimisers (Adam) for many-parameter pulse shaping, multi-start.
- Black-box and Bayesian baselines without leaving Julia: `Surrogates.jl` (Kriging with EI/LCB/SRBF `surrogate_optimize`), OptimizationBBO, Evolutionary. Quality below Ax/BoTorch, sufficient for a baseline.
- Campaign driver: `EnsembleProblem` + `EnsembleDistributed` (one GPU per worker) with the EDM run manifests.

## 7. Symbolic layer

ModelingToolkit components and `mtkcompile`, `build_function` to device-callable scalar code, `MTKParameters` tunables as the vector the tangent layer differentiates, SymbolicIndexingInterface for named access, `remake` with symbolic maps, DynamicQuantities units; Symbolics codegen flags `fuse_broadcasts` and `scalarize_single_element_regions`; MOL staggered groups. Covered in the architecture draft, section 5.

Why the symbolic layer stays (A10): it enforces a separation of concerns between problem description and numerical implementation (architecture §0.5); the production kernels are hand-written, but their interfaces are shaped so that MTK-generated forces, MOL-generated mesh updates and later FEM or DG backends can replace them slot by slot.

## 8. Needed but outside SciML

KernelAbstractions (kernels), AcceleratedKernels (backend-agnostic `sort!`, `accumulate`, `reduce`: particle sorting, segmented-sum deposition, diagnostics; EDM uses it), Atomix (atomics in KA kernels), Adapt and GPUArrays, CUDA.jl/AMDGPU.jl, MPI.jl and ImplicitGlobalGrid.jl (halo exchange), AbstractFFTs/CUFFT.

## 9. Do not force

Generic ODE solvers on the PIC state; continuous adjoints on the PIC loop; MOL's general `discretize` path on the GPU; `DiffEqGPU.EnsembleGPUArray` for the loop (it is an array-of-ODEs abstraction, not a grid-plus-particles one); NeuralPDE (PINNs solve a different problem).
