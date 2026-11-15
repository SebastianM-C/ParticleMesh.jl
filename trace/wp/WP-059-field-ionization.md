+++
id = "WP-059"
status = "done"
goals = ["A0", "A1", "A2", "A3", "A4", "A7", "A8"]
depends_on = ["WP-023", "WP-028", "WP-055", "WP-056", "WP-057"]
adds_dependencies = []
+++
# WP-059: deterministic field ionization of a dopant, for ionization injection

Field ionization of a dopant in the 2D electromagnetic module without random numbers: macro-atoms on a fixed lattice whose bound populations follow the rate equations of an ADK tunnelling rate (the module's own generic kernel function, in normalized units) in the field at each atom, and which release their electrons by weight at fixed birth steps into pre-allocated slots, so that the dopant's density, its section and the rates enter through weights only (A3) and the charge born is a smooth function of the laser and target parameters. It works when the rate agrees with an independent code's coefficients, the population of one level in a constant field is the exact solution and a chain of levels converges to it at first order in Δt, Gauss's law holds to roundoff with the ions' charge counted (periodic, at an open edge and in the moving window, both deposition modes), Reference mode stays bitwise, and ForwardDiff through a small ionization-injection run agrees with the finite-difference floor of A7. Outcome: T-770 to T-785 in `lib/ParticleMeshElectromagnetic/test/ionization.jl` and `reports/wp-059-field-ionization.md`.
