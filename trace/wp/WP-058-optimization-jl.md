+++
id = "WP-058"
status = "active"
goals = ["A0", "A1", "A7"]
depends_on = ["WP-045", "WP-048", "WP-051", "WP-052"]
adds_dependencies = ["Optimization", "OptimizationLBFGSB", "ADTypes"]
+++
# WP-058: the optimization scripts on Optimization.jl

The cases of `scripts/optim/` become objectives on a scaled box that Optimization.jl's bound-constrained L-BFGS-B maximizes with ForwardDiff gradients, one run per point, in place of the hand-written `maximize.jl`, and the Bayesian baseline of WP-048 takes the same objective. It works when the WP-045 and WP-052 runs, with every check they assert, reach their reported optima again with the library solver.
