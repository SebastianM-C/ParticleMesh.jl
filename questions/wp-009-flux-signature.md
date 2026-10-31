# WP-009: the signature of `deposit_flux!`

Work package: WP-009 (consumer: WP-014). Files: `lib/ParticleMeshCore/src/flux.jl`, `docs/plan/architecture-draft.md` §0.7 contract 13.

WP-009 names `deposit_flux!(mode, Jx, Jy, Jz, species, shape, grid, ws; quantity_y, quantity_z, workgroupsize = nothing)`. The implementation adds two arguments and one helper:

1. **`stagger`, positional after `shape`**, as in `deposit!`. The package asks Jx to lie on the stagger dual to ρ's "so that discrete continuity uses the same shape and stagger pair as `deposit!`", and `deposit!` takes the stagger explicitly; without it the flux would have to fix ρ's stagger itself (the shape's natural one). The electromagnetic module needs ρ, Ey and Ez at `Node()` with order 2, which is not the natural stagger of order 2.
2. **`dt`, a required keyword.** With Jx a current density, continuity `(ρ_new - ρ_old)/Δt + (Jx[j] - Jx[j-1])/Δx = 0` needs Δx/Δt. The alternative, a flux per step that the caller divides by Δt, leaves Jx in units of charge per area and makes every caller repeat the scaling.
3. **`flux_guard_width(shape, stagger)`**, exported: the old and new stencils together reach one node further than either one, so the flux needs one guard cell more than `guard_width(shape)` except for order 2 at `Center()`. `deposit_flux!` checks it and throws an `ArgumentError`.

Two conventions the package left open, stated in the docstring and §0.7: `Jx[j]` is the face half a cell to the right of ρ's node `j` (the convention of the electrostatic reference's `Ehalf`), and T-112's sums carry no Δx (`Σ_j Jy[j] = Σ q·w·u_y`), because the engine's weights are densities and `deposit!`'s total is `Σ q·w` (T-134); the package text writes `Σ Jy·Δx`.

Assumption: these are what the package meant; the WP-014 session was told and builds on them.
