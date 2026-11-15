+++
id = "WP-058"
status = "done"
goals = ["A0", "A1", "A7"]
depends_on = ["WP-045", "WP-048", "WP-051", "WP-052"]
adds_dependencies = ["Optimization", "OptimizationBase", "OptimizationLBFGSB", "OptimizationMadNLP", "MadNLP", "SciMLBase", "TimerOutputs", "DifferentiationInterface"]
+++
# WP-058: the optimization scripts on Optimization.jl

The cases of `scripts/optim/` become plain Optimization.jl problems (scaled unknowns, `MaxSense`, `AutoForwardDiff`, a callback for the record) that its bound-constrained L-BFGS-B solves in place of the hand-written `maximize.jl`, and the Bayesian baseline of WP-048 becomes a solver of the same interface. It works when the WP-045 and WP-052 runs, with every check they assert, reach their reported optima again with the library solver. Outcome: `reports/wp-058-optimization-jl.md`.
