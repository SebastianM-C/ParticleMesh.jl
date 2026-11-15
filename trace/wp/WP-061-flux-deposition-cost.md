+++
id = "WP-061"
status = "done"
goals = ["A7", "A8", "A4"]
depends_on = ["WP-020", "WP-037", "WP-045"]
adds_dependencies = []
+++
# WP-061: the 2D Reference flux deposition at the cost of its arithmetic

Make the 2D Reference `deposit_flux!` cost what its arithmetic costs for every element type, duals included (#442: 15 times the plain cost with two partials, 85% of a gradient run), without changing a bit of its result on the CPU. It works when the engine's recorded Reference digests and the before/after digests of the state and the histories of the module's runs are unchanged (the workspaces' digest holds the patch and changes with its layout), a test holds the stencil to code with no call left out of line, and the time per phase of the WP-045 run is measured before and after per element type on the CPU; a choice measured only on the CPU (the patch layout, the work-group of the patch kernel) is the CPU backend's alone. Outcome: `reports/wp-061-flux-deposition-cost.md`.
