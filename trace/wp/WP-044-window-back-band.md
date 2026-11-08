+++
id = "WP-044"
status = "active"
goals = ["A3", "A7"]
depends_on = ["WP-028", "WP-043"]
adds_dependencies = []
+++
# WP-044: observables past the moving window's back band

The moving window names the back columns whose fields its retirements make jump between neighbouring parameter values (`window_back_band`), and the WP-043 functional reads past them (#370). It works when the jitter of a long run lies inside that band and, past it, ForwardDiff agrees with the finite-difference floor of A7 at 600 steps, where reading the band fails. Outcome: T-590 and T-591 in `lib/ParticleMeshElectromagnetic/test/lwfa_gradient.jl`.
