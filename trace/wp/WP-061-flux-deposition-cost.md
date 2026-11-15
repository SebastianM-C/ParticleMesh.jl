+++
id = "WP-061"
status = "active"
goals = ["A7", "A8", "A4"]
depends_on = ["WP-020", "WP-037", "WP-045"]
adds_dependencies = []
+++
# WP-061: the 2D Reference flux deposition at the cost of its arithmetic

Make the 2D Reference `deposit_flux!` cost what its arithmetic costs for every element type, duals included (#442: 15 times the plain cost with two partials, 85% of a gradient run), without changing a bit of its result on the CPU. It works when the engine's recorded Reference digests and the before/after digests of the module's runs are unchanged, a test holds the stencil to code with no call left out of line, and the time per phase of the WP-045 run is measured before and after per element type on the CPU, with the patch layout measured on a GPU. Outcome: `reports/wp-061-flux-deposition-cost.md`.
