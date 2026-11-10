+++
id = "WP-051"
status = "done"
goals = ["A0", "A1", "A3", "A7"]
depends_on = ["WP-045"]
adds_dependencies = []
+++
# WP-051: a converged 2D wake optimization

A resolution study in Δx, Δy and read-out time of WP-045's optimization, with an objective that does not depend on when it is read, that finds a discretization at which the optimum (τ, focus) stays put under refinement. It works when the converged duration is explained against 1D theory: the linear value 2/ω_p, the 1D nonlinear cold-fluid wake at the same a₀, and plane-wave and low-a₀ runs of the same code.
