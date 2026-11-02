+++
id = "WP-008"
status = "done"
goals = ["A3", "A7", "A8", "A9"]
depends_on = ["WP-014"]
adds_dependencies = ["SciMLBase", "RecursiveArrayTools", "OrdinaryDiffEqFunctionMap"]
+++
# WP-008: the SciML harness: the electromagnetic step as a `DiscreteProblem`

Wrap the electromagnetic module's `step!` as a SciML `DiscreteProblem`, so callbacks, ensembles and `remake` over parameters work on it, with the same bits as the plain loop. Outcome: #131.
