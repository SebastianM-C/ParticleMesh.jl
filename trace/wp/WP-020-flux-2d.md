+++
id = "WP-020"
status = "proposed"
goals = ["A2", "A4", "A7", "A8", "A9"]
depends_on = ["WP-009", "WP-018"]
adds_dependencies = []

[checked]
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A4 = "0997b929ff6d0c55f79b6f562c66d6e86c8f8e24"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
A9 = "89b2103e556ff49abca35bdb96e72429b5d30845"
+++
# WP-020: the Esirkepov flux deposition at N = 2, with a Reference path that computes each particle's weights once

`docs/plan/architecture-draft.md` §0.8 item 2. The engine has every other N = 2 transfer (WP-018); the current deposition is the one missing piece between it and a 2D electromagnetic step. It is the hot kernel of the step, so this package also fixes the cost of the sorted path (#93) instead of porting it: in 1D the Reference `deposit_flux!` recomputes each particle's whole stencil for every face it touches and costs about 16 times `deposit!` (`reports/wp-009-flux.md`); in 2D the stencil is the product of two, so the same design would cost far more.

This is a spike in the A0 sense: the choices below are starting points, and the code may show better ones. Say so in the PR and update §0.8 item 2 and §0.7 contract 13 from the code.

## Scope

`lib/ParticleMeshCore` (src and test), `scripts/engine2d/` for the measurement, `reports/wp-020-flux-2d.md`, §0.7 contract 13 and §0.8 item 2 in `docs/plan/architecture-draft.md`, and `questions/`.

- `deposit_flux!` gains an N = 2 method: Jx and Jy from Esirkepov's decomposition of the old and new tensor-product weights, prefix sums per axis in increasing node order; Jz, the collapsed-axis component, from Esirkepov's out-of-plane weight (which carries the ⅓ ΔSx ΔSy term and is not the average of the old and new products), multiplied by the module's `quantity_z`. Verify that form against Esirkepov's paper and the Smilei and PIConGPU sources and cite what you read; what you could not read is marked unverified in the report (A5 requires the same scheme in all three codes).
- Per-axis flux guard width (`flux_guard_width` for N = 2).
- `Fast()`: atomics, one work item per particle.
- `Reference()`: deterministic and bitwise across reruns, thread counts and launch shapes (A4), with each particle's old and new weights computed once. The mechanism is the package's choice (for example, per-particle stencil contributions written to a fixed-size scratch buffer in the `Workspace` and reduced per face in sorted order, or a per-cell tile reduction); measure at least two before fixing one, as WP-018 did. If the new mechanism is also better at N = 1, the N = 1 method may switch to it, as long as T-110 to T-117 still pass; the change of N = 1 bits is then stated in the PR.
- Measured inputs from a review of `main` at `b714331` (CPU, Intel Core i9-14900K, 1 thread, 1D Landau tiled to 512 cells, 2^18 particles, Float32): the Reference flux costs 86 ns per particle against 16.5 ns for a serial scatter of the same `flux_stencil` in the same cell and slot order, so 5.2× of it is recomputation; the flux is 88 of the 144 ns of the Reference step. One candidate to measure is compute-once with a fixed-chunk two-level reduction: contributions into per-chunk partials over fixed slot chunks, then a per-node reduction in chunk order, which keeps the order independent of thread count and launch shape (A4).
- Time the mechanisms at the staggers the 2D layout deposits (Jx at (Center, Node), Jy at (Node, Center), Jz at (Node, Node); §0.8 item 5), not only at (Center, Center): at order 2 a Node axis needs `floor` and widens the per-node loop from 3 to 4 cells per axis, and WP-018 measured that gap at 4.7× on the CPU (121 against 26 ns per particle). Binning by the nearest node (`floor(x + 1/2)`) instead of by cell for even orders at Node staggers is a candidate that removes it.
- If the chosen mechanism also suits `deposit!`'s Reference path (the same per-node recomputation, `transfers.jl`), it may be applied there too; the PR states any N = 1 bits that change.
- Every kernel within §0.4 rules 1 to 6, no float literals, `workgroupsize = nothing` on every launch (contract 12).

## Deliverables

1. The N = 2 method and its tests, T-300 to T-307.
2. `scripts/engine2d/` measurement: ns per particle of Reference and Fast at N = 2 (order 1 and 2, Float32 and Float64, two grid sizes, 1 thread and the machine's thread count), against `deposit!` at N = 2 on the same particles, and the N = 1 Reference flux before and after if N = 1 changed. PNG plus JSON sidecar plus manifest (F3).
3. Report with the measurement, the mechanism choice and why, and the scheme sources read.
4. `tolerances.toml` entries for the new `C` multiples, `status = "proposed"`.
5. §0.7 contract 13 restated for N from the code; §0.8 item 2 deleted as done.

## Non-goals

No electromagnetic code (WP-022 and WP-023), no mesh solve (WP-021), no GPU run (requested from the GPU runner after merge, a follow-up), no `Project.toml` or Manifest change, no change of `pushers.jl`.

## Acceptance

- T-300 discrete continuity at N = 2 [A2, A7, WP-020]: for particles moving less than one cell per axis, crossing both seams and the corner, (ρⁿ⁺¹ − ρⁿ)/Δt + ∇·J = 0 at every node to C·eps(T) scaled by count and max|q|, with ρ from `deposit!` at the matching stagger, orders 1 and 2, Float32, Float64, `Dual{Float32,3}`.
- T-301 the Jz weight [A2, WP-020]: a particle moving only in z deposits Jz equal to the charge-weighted shape at its position; a particle moving in x, y and z deposits Σ Jz = q·vz·w to roundoff, and the per-node Jz equals a hand-evaluated Esirkepov out-of-plane weight in one test case.
- T-307 Jz and the in-plane components against hand-evaluated Esirkepov weights for a particle that moves in all three directions across a cell boundary, at N = 1 (Jy, Jz as the old/new average) and N = 2 [A2, WP-020]. Beyond T-301 (Jz at N = 2 in one hand-evaluated case), it covers N = 1 and the in-plane components across a cell boundary, and it must fail on a mutation that replaces the 1D Jy, Jz old/new average by the new-position weights.
- T-302 Reference and Fast agree per node within C·eps(T)·Σ|contributions| [A4, WP-020].
- T-303 Reference is bitwise across reruns, 1 and 4 threads, and two explicit `workgroupsize` values, per element type [A4, WP-020].
- T-304 reduction to 1D [A7, WP-020]: motion along x only with a y-constant load gives y-summed Jx equal to the 1D `deposit_flux!` to C·eps(T).
- T-305 `Dual` [A8, WP-020]: partials exact, primal bitwise equal to the Float32 run.
- T-306 allocation per call flat across particle counts and grid sizes after warm-up, as T-268 measures it [A7, A9, WP-020].
- T-110 to T-117 and T-260 to T-269 pass. `build.jl --check` passes; CPU CI green.
- The report states the Reference cost against Fast and against `deposit!` at N = 2. No threshold is asserted; the numbers are proposals.

## Depends on

WP-018 (N = 2 engine), WP-009 (the N = 1 flux and its tests). Runs in parallel with WP-021 and WP-022; WP-023 builds on all three. Test ids T-300 to T-309.
