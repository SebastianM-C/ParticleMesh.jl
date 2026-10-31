+++
id = "WP-006"
status = "proposed"
goals = ["A2", "A3", "A4", "A7", "A8"]
depends_on = ["WP-004", "WP-005"]
adds_dependencies = []

[checked]
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A3 = "2e44fafc2ffe3634ff3dd6e84bb90825e76c585e"
A4 = "c85034796ed0c2c17c6629196b93ee07c4d8b2f2"
A7 = "b4cade448159d5276774626df7f94299f8621395"
A8 = "ca00019a5a518e1747bce0643e09c87eb11494b2"
+++
# WP-006: engine core, transfers: shapes, gather, deposition in both modes

The particle↔mesh transfers of `lib/ParticleMeshCore`. Contracts: `docs/plan/architecture-draft.md` §0.7 items 3, 4 (`fold_guards!`), 7, 8, 14 to 18; §0.4 rule 3. It can start once WP-005 has fixed the species type and accessors, and runs in parallel with the rest of WP-005.

## Scope

Shape functions as type-level configuration, `guard_width(shape)`, staggered weights, the device-level gather and the species-level `gather!`; scalar deposition `deposit!(mode, field, species, shape, quantity, grid, gstate, ws)` in `Reference()` (sorted segmented reduction, deterministic) and `Fast()` (atomics) modes; the accumulation trait; `fold_guards!` as the adjoint of `fill_guards!`. No physics: what is deposited is a module-supplied per-particle array or species constant.

## Deliverables

1. **Shapes.** `Shape{1}`, `Shape{2}` (the A2 default, quadratic), optionally `Shape{3}`; B-spline weights at any stagger; `guard_width(shape)`; the device-level gather at `(cell::NTuple{N,Int32}, offset::NTuple{N,T})` for module kernels to inline.
2. **`gather!(out, fields, species, shape, grid, gstate)`** into a workspace `NamedTuple` named like the field set.
3. **`deposit!`** in both modes with the signature above; `ws` holds the sort's segment offsets (WP-005's key helper) and the `DiffCache` scratch (item 17). It accumulates into its target. `accumulation(T)` on the target field's element type: `Atomic()` for primitive floats, `Sorted()` for every other type, including `Dual` and Double32; in `Fast()` mode a `Sorted()` type still deposits sorted. No ForwardDiff or DoubleFloats code and no extension.
4. **`fold_guards!(fields, grid, gstate)`**: adds periodic guard contributions into the owning interior cells, then zeroes those guards.
5. Tests in `lib/ParticleMeshCore/test/transfers.jl`, its table in `tolerances.toml`.

## Non-goals

No Esirkepov or any current deposition: that is the LWFA module's (`lib/Wakefields`), built on these weights (§0.7 item 14). No pusher, no field solve. No tangent-bundle layout (§0.7 item 9). No Reactant (A9).

## Acceptance

Test ids T-060 to T-079.

- T-060: partition of unity of the weights at every stagger, all shapes, every `T`, to a proposed `C·eps(T)`.
- T-061: gather reproduces constant and linear fields exactly (to roundoff) on every stagger. For a quadratic field and a degree-p shape with p ≥ 2, the gathered value is f + (p+1)h²f''/24; the test checks that known shift (B-splines reproduce degree 0 and 1 only).
- T-062: ⟨fill_guards! x, y⟩ = ⟨x, fold_guards! y⟩ to roundoff, on periodic axes, all N.
- T-063: deposition is the adjoint of gather: ⟨deposit(q), f⟩ = Σᵢ qᵢ gather(f)ᵢ to roundoff.
- T-064: the total deposited quantity equals Σ wᵢqᵢ to a proposed `C·eps(T)·count`.
- T-065: `Reference()` and `Fast()` agree per node to a proposed `C·eps(T)·contributions` bound, for Float32 and Float64.
- T-066: `Reference()` is bitwise identical across reruns and across `JULIA_NUM_THREADS` = 1 and many (A4).
- T-067: with `Dual{Float32,3}` weights, the value part equals the primal deposit bitwise in `Reference()` mode, and the partials equal the exact derivative, since deposition is linear in the weights (A3).
- T-068: deposition and gather run for every `T` of the A8 matrix (Double32 may be `@test_broken` with the missing function named, A8).
- T-069: allocation of `gather!` and `deposit!` does not grow with particle count or grid size, including under a `ForwardDiff.gradient` over weight parameters with a matching `DiffCache` chunk size (§0.7 items 13 and 17).

## Depends on

WP-004 (grid, field sets, `fill_guards!`, scaffolding) and WP-005 (species type and accessors, sort key helper and segment offsets).

## Budgets

CPU timings per particle for gather and both deposition modes recorded without a threshold; scratch bytes per particle reported. Proposed `C` values for T-060, T-064 and T-065 recorded in `tolerances.toml` marked "proposed", with the measured maxima at two or more sizes.
