+++
id = "WP-056"
status = "active"
goals = ["A4", "A3", "A0"]
depends_on = ["WP-055", "WP-008", "WP-028", "WP-034", "WP-047"]
adds_dependencies = []
+++
# WP-056: the closed state of the electromagnetic module

Close the state of the electromagnetic module: everything a run evolves (the particles, the fields, a PML's convolution fields, the moving window's counters, the witness bunch) is one list of arrays, which the SciML partition holds at N = 1 and N = 2, so that with the step index and the parameters a run can be saved after any step and continued in a newly built state, with no step treated specially and without changing a result. It works when a run restored at several steps (across a window shift, a refill and the witness injection, with both PMLs) continues with the bits of the uninterrupted run per element type, and the fixed set of short 1D and 2D runs of WP-055 gives the same bits before and after. Outcome: the description of the pull request that adds this file.
