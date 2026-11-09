+++
id = "WP-048"
status = "active"
goals = ["A0", "A1", "A7"]
depends_on = ["WP-045"]
adds_dependencies = []
+++
# WP-048: a Bayesian-optimization baseline for the WP-045 problem

A Gaussian-process Bayesian optimizer with expected improvement (`scripts/optim/bayes.jl`) maximizes the WP-045 wake amplitude over the same box, and `scripts/optim/bo_compare.jl` counts the objective calls and CPU seconds it and the gradient-based BFGS of WP-045 need to come within 1 %, 0.1 % and 0.01 % of the best value either finds, from the same seeded starts. It works when the optimizer reaches the known minimum of an analytic test function and both methods agree on the optimum of the wake case.
