+++
id = "WP-064"
status = "active"
goals = ["A0", "A1", "A3", "A4", "A7"]
depends_on = ["WP-044", "WP-058", "WP-059"]
adds_dependencies = []
+++
# WP-064: an objective on the electrons born by field ionization

A user-layer objective (`scripts/optim/`) next to the witness bunch's: the kinetic energy carried by the electrons a doped section's field ionization has released, selected by their species and not by an energy cut, with a smooth position weight that is 1 over the first wake bucket and 0 before the window's back band, and the plasma electrons' own highest energy and trapped weight reported beside it as the guard against self-injection. It is fit to optimize on when one 2D run traps born electrons, the objective is the closed form on hand-placed electrons, and a scan in one parameter shows ForwardDiff's derivative within the measured finite-difference floor of A7 with no steps between neighbouring runs, at a stated cost per run. Outcome: T-830 to T-834 in `scripts/optim/test/runtests.jl` and `reports/wp-064-born-electron-objective.md`.
