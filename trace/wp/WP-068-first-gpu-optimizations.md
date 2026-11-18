+++
id = "WP-068"
status = "active"
goals = ["A0", "A3", "A7", "A8"]
depends_on = ["WP-065", "WP-067"]
adds_dependencies = []
+++
# WP-068: the first optimizations of the trapped-electron objective on the GPU

Maximize WP-065's objective over its four free parameters (duration, waist, focus, density) at the a₀ = 1.9 energy budget with L-BFGS-B from a few starts in the box duration 4 to 9, waist 6 to 14, focus 64 to 110 and density 0.05 to 0.10 n_c, at λ₀/32 (Δy = 2Δx) with the compensated gather (WP-067) in Float32 on the GPU, recording at every point J, its gradient, a₀ and the dark current; then again under a peak power cap, CETAL-PW's 1 PW at 25 J carried to the case by τ/λ_p, which at the fixed energy is a floor τ ≥ 2.45/√n on the duration; and under that cap with J counting the ionization's electrons alone, and with the plasma's own share capped smoothly. It is done when the optimum's J and gradient are re-checked at λ₀/48 and against Float64 central differences, and the paths, the optimum and the cost per point are in a report. Outcome: T-970 to T-977 in `scripts/optim/test/runtests.jl` and `reports/wp-068-first-gpu-optimizations.md`.
