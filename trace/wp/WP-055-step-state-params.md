+++
id = "WP-055"
status = "done"
goals = ["A3", "A4", "A0"]
depends_on = ["WP-008", "WP-014", "WP-024", "WP-026", "WP-028", "WP-047"]
adds_dependencies = []
+++
# WP-055: the electromagnetic step as a function of the state and the parameters

Make `initialize!` and `step!` of the electromagnetic module read every parameter in A3's sense (the lasers, the densities and the profile, the witness bunch, the seed) from `EMParams`, at N = 1 and N = 2, with the state keeping what a run evolves and the configuration it was sized for and the diagnostics an observer's, without changing a result. It works when a fixed set of short 1D and 2D runs gives the same bits before and after per element type and deposition mode, and one state handed dual parameters gives the bits of a state rebuilt from the dual problem. Outcome: the description of the pull request that adds this file.
