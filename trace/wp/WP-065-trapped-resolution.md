+++
id = "WP-065"
status = "active"
goals = ["A0", "A1", "A3", "A4", "A7"]
depends_on = ["WP-052", "WP-064"]
adds_dependencies = []
+++
# WP-065: every trapped electron at a fixed laser energy, under refinement

A user-layer objective (`scripts/optim/`) over every trapped electron in the first bucket of WP-064's ionization case, born or injected from the plasma itself, chosen by a smooth weight in the forward momentum, with the later buckets' charge reported beside it, with a₀ tied to the pulse's duration and waist so that the laser energy is fixed and five free parameters (duration, waist, focus, delay, density). It is understood when the objective is the closed form on hand-placed electrons, its ForwardDiff gradient matches central differences, and runs at λ₀/16 to λ₀/64 show how far the objective and its gradient have converged and whether they stay smooth across the onset of the plasma's own injection. Outcome: T-890 to T-893 in `scripts/optim/test/runtests.jl` and `reports/wp-065-trapped-resolution.md`.
