+++
id = "WP-010"
status = "done"
goals = ["A0", "A2", "A3", "A4", "A5", "A7", "A8", "A9"]
depends_on = []
adds_dependencies = ["KernelAbstractions", "Atomix", "Adapt", "ForwardDiff"]

[checked]
A0 = "a8c098e1e3e8427f42c9d233ce646b15b1b66088"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A3 = "2e44fafc2ffe3634ff3dd6e84bb90825e76c585e"
A4 = "8c6d1910f937970d7f14e1a844ed3ff49f67802a"
A5 = "d8ba66a9b95c54fa5e30ede80993135d051424b6"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-010: spike, a 1D periodic electrostatic PIC with a checked gradient

The first runnable result and the bootstrap of the factory (A0: the first implementation of an area is a spike; contracts are written from working code). One session, sequential, in `lib/ParticleMeshCore`, which this package creates. It replaces the parked WP-004 to WP-009; their text is the proposal this spike tests, and `docs/plan/architecture-draft.md` §0.7 lists the decisions that bind and the proposals that do not. The spike ends with a write-up from which the next planning round cuts the packages that split engine from module and settle the interfaces.

## Scope

Everything a 1D periodic electrostatic particle-in-cell simulation needs, written as KernelAbstractions kernels on the CPU backend within the §0.4 rules, generic in `T`, plus the two physical benchmarks of the A5 1D set that are electrostatic (Landau damping, two-stream instability) and one finite-difference-checked gradient through the whole simulation. The electrostatic pieces live in a submodule of the same package so that the engine/module split is a refactor made from working code, not a guess made before it. They are not a physics module and do not become one: the electromagnetic module is the only physics module planned and the proof of the split (A9), and the spike's electrostatic step is the reference for the Landau and two-stream cases until the electromagnetic module reproduces them as edge cases with no transverse fields, as Smilei runs them.

## Deliverables

1. **Package.** `lib/ParticleMeshCore` with `Project.toml`, added to the root workspace and re-exported by `ParticleMesh`; direct dependencies KernelAbstractions `"0.10"`, Atomix, Adapt, with compat entries; ForwardDiff as a test dependency (the frontmatter `adds_dependencies` has no way to mark that, so it lists ForwardDiff next to the direct dependencies for the F2 check); whatever else the spike needs, added per F2 and listed in the PR. No Reactant, no SciML, no DoubleFloats.
2. **Engine pieces, 1D periodic.** A grid; a structure-of-arrays species with cell plus offset, three velocity or momentum components, weight and status, at fixed capacity; shape functions of order 1 and 2 with gather and scalar deposition; deposition in `Reference` mode (sort by cell, deterministic node-wise reduction) and `Fast` mode (atomics); fill and fold of periodic guards; a sort by cell; quiet-start loading with weights from a density callable with explicit parameters (A3) and thermal velocities from counter-based Philox4x32-10 draws (A4). Write kernels over `CartesianIndex{N}` where it costs nothing, but test only N = 1.
3. **Electrostatic step** (the `Electrostatic` submodule of the spike package, not a physics module, A9). The field from the deposited density by a discrete Gauss law on the periodic grid (a prefix sum of the zero-mean density is enough and makes the Gauss law exact; an FFT solve through FFTW.jl, which accepts only its own `Float32` and `Float64` element types and so cannot carry `Dual`, is not needed), a leapfrog or Boris push, and a step that is a pure function of state and parameters, composed in the order the module chooses; a plain `run!` loop.
4. **Physics.** Landau damping of a small-amplitude wave and the two-stream growth rate, each compared with linear theory for the chosen parameters, the measured and theoretical values recorded without a threshold (A7: measure, then propose) in `lib/ParticleMeshCore/test/tolerances.toml`, which this package creates, and the standard plots from `scripts/` with JSON sidecars (F3). The theory values are computed in the test from the dispersion relation, with the derivation referenced; nothing from memory.
5. **Gradient.** A `ForwardDiff` derivative of a smooth scalar diagnostic (the field energy at a fixed time, or a log-sum-exp maximum) with respect to a parameter that enters through the weights (the amplitude of the initial density perturbation), compared with a Float64 finite-difference scan in `Reference` mode with common seeds; the run is long enough that particles cross cells and the periodic seam. The floor is measured from the scan as A7 states (at least five step sizes, the spread of the three neighbouring difference quotients around its minimum), never computed from `eps` and the step: the spike found the formula floor `eps·|f|/h` four orders of magnitude below the real one, because the particle dynamics amplify rounding. Report the floor and the agreement.
6. **Write-up.** `reports/spike-electrostatic-1d.md`: what was built, which of the parked packages' contracts held, which broke and how (expected friction: `floor` and comparisons on `Dual` offsets, the sorted reduction's scratch and determinism, the ordering of the step), measured CPU time per particle-step and scratch per particle for both deposition modes, and a proposed list of the next packages. This is the input of the next planning round; the trace tooling is not touched.

## Non-goals

No 2D or 3D runs, no bounded axes, no moving window, no Esirkepov flux or Maxwell solve (Weibel, the third A5 case, waits for the electromagnetic module), no SciML harness, no Reactant (A9), no Double32, no GPU run, no Molly study, no source-text test for physics names. No contract is frozen by this package: names and signatures are the spike's, and the write-up proposes which to keep.

## Acceptance

Test ids T-130 to T-159, each named with its id and what it verifies, for example `@testset "T-130 partition of unity of the shape weights [A7, WP-010]"`.

- T-130: partition of unity of the shape weights for orders 1 and 2, every `T` of the matrix (Float32, Float64, `Dual{Float32,3}`), to a proposed `C·eps(T)`.
- T-131: gather reproduces constant and linear fields to roundoff, away from the periodic seam.
- T-132: deposition is the adjoint of gather, with fill and fold as the guard convention, to roundoff.
- T-133: fill and fold are adjoints: ⟨fill x, y⟩ = ⟨x, fold y⟩ to roundoff.
- T-134: the deposited total equals Σ wᵢqᵢ to a proposed `C·eps(T)·count`.
- T-135: `Reference` and `Fast` agree per node to a proposed bound (Float32 and Float64).
- T-136: `Reference` is bitwise identical across reruns and across `JULIA_NUM_THREADS` = 1 and many, for one step and for the whole Landau run (A4).
- T-137: the discrete Gauss law holds to roundoff after every step.
- T-138: free streaming at zero field is exact for a dyadic displacement and within `C·eps(T)·steps` otherwise; the periodic wrap conserves the count.
- T-139: the same seed gives bitwise identical loads; Philox known-answer vectors from a cited source, or `@test_broken` with `questions/philox-kat-vectors.md` naming what was unreachable.
- T-140: `Dual` weights give partials equal to the exact derivative of the deposited density, and the `Reference` deposition of the `Dual` weights' value parts is bitwise equal to the primal deposition of the same weights. The load itself is not bitwise across element types (A4: bitwise holds per element type and backend; the spike saw 1 ulp in the loader with `Dual` parameters), so the primal species is fed the value parts and the claim is about deposition.
- T-141: the Landau damping rate and T-142: the two-stream growth rate match linear theory, with the measured relative error recorded in `tolerances.toml` as a proposal and no threshold asserted in this package.
- T-143: the finite-difference gradient check of deliverable 5: the ForwardDiff derivative agrees with the central differences within a stated multiple of the measured floor (A7), the multiple measured and proposed like every other tolerance. The spike measured a ratio of 0.70 (relative difference 1.94e-9 against a floor of 2.78e-9) and asserts agreement within `C` times the floor, with `C = 1` recorded as a proposed entry in `lib/ParticleMeshCore/test/tolerances.toml` (`gradient.jl` reads it from there).
- T-144: allocation of one step after warm-up does not grow between two particle counts and two grid sizes.
- The trace check passes; CI is green; the write-up and the plots with sidecars are in the PR.

## Depends on

Nothing. The next packages depend on its write-up.

## Budgets

CPU timings and scratch recorded without thresholds (A7's throughput budget is on the GPU). Tier 2 (F4): the PR edits `/Project.toml` and `/lib/*/Project.toml`.
