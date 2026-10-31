+++
id = "WP-009"
status = "proposed"
goals = ["A2", "A3", "A4", "A7", "A8"]
depends_on = ["WP-011"]
adds_dependencies = []

[checked]
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A3 = "2e44fafc2ffe3634ff3dd6e84bb90825e76c585e"
A4 = "8c6d1910f937970d7f14e1a844ed3ff49f67802a"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "ca00019a5a518e1747bce0643e09c87eb11494b2"
+++
# WP-009: engine core, flux: charge-conserving deposition in 1D (Esirkepov)

Re-derived from the spike's write-up for N = 1, keeping this package's id and its reserved test range T-110 to T-129: it is the same interface, not a new one, written from the engine WP-011 leaves behind. Esirkepov's scheme is the source of A7's strongest conservation claim, Gauss's law to roundoff, and the piece most likely to break AD smoothness at cell crossings. A small engine-only package with an exact continuity identity and a crossing-smoothness check settles that risk before any Maxwell code exists, so a failure in WP-014's Gauss test can only come from the module. The extras must survive a sort between the push and the deposit, and the tests check that they do.

## Scope

The third engine transfer next to `gather!` and `deposit!`: `deposit_flux!` for N = 1. Jx comes from the differences of the shape weights between the old and the new position (Esirkepov's formula, which in 1D reduces to a prefix sum of the weight differences over the stencil). Jy and Jz come from the weights averaged over the old and new positions, times a module-supplied quantity. The function has a Reference mode (sorted, node-wise reduction) and a Fast mode (atomics), with the accumulation trait, and supports shapes of order 1 and 2. The caller declares the old cell and offset as Species extras, using WP-011's mechanism, and supplies the quantity as WP-011's callable. Like every engine launch it takes WP-011's launch keyword `workgroupsize`, with the backend default.

The engine knows no Maxwell equations. The package is proven by exact discrete continuity against the engine's own `deposit!`, held to roundoff, and by the smoothness of the flux through a cell crossing (A2's AD-smoothness rule). Gauss's law is WP-014's test and builds on this one. The N-D generalization (the 3D formula with collapsed axes) belongs to the first N-generic package, and this text and `architecture-draft.md` §2 say so. This package is the single owner of `deposit_flux!`. Its CUDA run belongs to WP-013 if this package merges before WP-013's runs; otherwise to the `follow-up` issue WP-013 opens for it.

## Deliverables

1. **`lib/ParticleMeshCore/src/flux.jl`**: `deposit_flux!(mode, Jx, Jy, Jz, species, shape, grid, ws; quantity_y, quantity_z, workgroupsize = nothing)`. Jx lies on the stagger dual to ρ's, so that discrete continuity uses the same shape and stagger pair as `deposit!`. The function reads the old cell and offset from `species.extras` and requires |Δ| < Δx per step, including cell crossings and the periodic seam; it documents that precondition and leaves larger displacements outside the contract. It is generic in `T`, follows the §0.4 rules, and its Reference path is deterministic and does not depend on `workgroupsize`.
2. **Guards**: the same guard convention as `deposit!`: deposit into the guards, then `fold_guards!`. The function shares the spike's `Workspace`.
3. **Test helper**: the staggered discrete divergence that matches the flux's staggering, in `lib/ParticleMeshCore/test/helpers.jl`, used for continuity tests only (the engine gets no divergence operator).
4. **Tests** in `lib/ParticleMeshCore/test/flux.jl` and their measured "proposed" entries in `tolerances.toml`.
5. **Plan text**: §0.7 lists `deposit_flux!` among the engine contracts, with the launch keyword. §2's Esirkepov entry reads "N = 1 now, N-D with the first N-generic package". The PR that implements this package sets its status to done and refreshes its `[checked]` entries (`build.jl --refresh WP-009`).

## Non-goals

- No N > 1 kernels or tests, and no 3D formula specialized down to 1D.
- No Maxwell solve, no Yee update, no pusher, no Gauss-law test of a field (WP-014), and no current filter (A2's binomial filter waits for the 2D LWFA module).
- No change to `Species`, `sort_by_cell!`, `deposit!`, the quantity callable or the launch keyword (owned by WP-011). No CUDA path and no GPU test here (WP-013 or its follow-up issue).
- No multi-cell moves per step: the precondition is documented, not handled.
- Parameters never enter through positions (A3): the flux is linear in the weights. No Reactant (A9).

## Acceptance

Test ids T-110 to T-129.

- T-110 discrete continuity [A2, A7, A8, WP-009]: for loaded species with random displacements |Δ| < Δx, including moves across a cell face and across the periodic seam, (ρ_new − ρ_old)/Δt + (Jx_{i+1/2} − Jx_{i−1/2})/Δx = 0 per node to C·eps(T)·max|ρ|/Δt, with ρ from `deposit!` using the same shape. Checked for orders 1 and 2, both modes, Float32, Float64 and `Dual{Float32,3}`, and also with a `sort_by_cell!` between storing the old position and depositing (the extras follow the particles).
- T-111 the continuity residual does not grow over N steps of a drift, checked at two values of N [A7, WP-009].
- T-112 transverse currents [A7, WP-009]: Σ Jy·Δx equals Σ q·w·u_y to C·eps(T)·count·max|q·w·u_y|. At zero displacement, Jy equals `deposit!` with quantity q·w·u_y to roundoff.
- T-113 Reference and Fast agree per face within a proposed bound for Jx, Jy and Jz, and Reference is bitwise identical across reruns, at `JULIA_NUM_THREADS` = 1 and 4, and for the default and two explicit values of `workgroupsize`, per element type [A4, A7, WP-009].
- T-114 `Dual` weights give the exact partials of Jx, Jy and Jz (the flux is linear in the weights) to C·eps of the duals' base type [A3, A8, WP-009].
- T-115 a particle at rest deposits exactly zero Jx [A7, WP-009].
- T-116 smoothness through a crossing [A2, A7, WP-009]: the ForwardDiff derivative of a smooth functional of Jx with respect to the displacement agrees with a check based on the `fd_scan` floor (A7's rule, in Float64 Reference with a common seed, as WP-011's helper enforces) for displacements whose end point crosses a cell face. Order 2 is expected to be C¹; for order 1 the measured agreement is recorded.
- An allocation check: one `deposit_flux!` after warm-up does not grow with particle count or grid size. `build.jl --check` passes, and CPU CI is green.

## Depends on

WP-011 (Species extras, the deposited quantity, the launch keyword, `fd_scan`, and the guard convention as contracts). The earlier dependencies on WP-005 and WP-006 are gone: those packages are withdrawn and their pieces exist in the engine. WP-014 builds on this package.

## Budgets

A7: continuity to roundoff, C·eps(T) scaled by the problem size per step, with C measured and proposed in `tolerances.toml`. CPU ns per particle of `deposit_flux!` in both modes, at 1 and 16 threads, and scratch per particle including the two extras, reported next to `deposit!`'s with no threshold. A7's "Gauss's law to roundoff with Esirkepov" is proven here as continuity and closed by WP-014's Gauss test. Tier 1.
