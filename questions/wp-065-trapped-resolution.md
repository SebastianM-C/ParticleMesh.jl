# WP-065: open choices of the objective over every trapped electron

The owner chose the objective: every trapped electron of the first bucket, born or self-injected, at a fixed laser energy, over duration, waist, focus and density, with the delay fixed (`TCASE.free`: its derivative is the scatter of a rough J); the later buckets' charge is dark current reported beside it. These are the choices the package made under it, each with the assumption it proceeded on. Files: `scripts/optim/trapped2d_case.jl`, `scripts/optim/trapped2d.jl`.

1. **Which electron counts as trapped.** A plasma electron has no label, so the choice is by its state. Options: a smooth weight in the forward momentum u_x; a weight in γ; a test on the wake's phase (an electron ahead of the separatrix). Assumed: a C² step in u_x from 3 to 5 (`TCASE.rise`), since the wake at 0.072 n_c moves with γ_φ ≈ 3.7 and the plasma's quiver stays below u_x ≈ 1 in WP-064's clean runs. Each run also reports J with the edges at (2, 4) and (4, 6).
2. **The dopant under a density change.** Assumed: it stays at 1 % of the plasma's density (`TCASE.dopant_fraction`), so the density moves both.
3. **The refinement.** WP-064 refined only Δx (Δy = π/4 at every resolution). Assumed: Δx and Δy together (Δy = 2Δx, `ny = ppl/16`), with x-only runs at λ₀/32 to /64 beside them.
