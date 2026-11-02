+++
id = "WP-018"
status = "done"
goals = ["A0", "A2", "A3", "A4", "A6", "A7", "A8", "A9"]
depends_on = ["WP-009", "WP-011", "WP-014"]
adds_dependencies = []
+++
# WP-018: the engine at N = 2 on the CPU

Port the engine's grid, shapes, gather, deposit, sort, advance and loading to N = 2, held to exact identities, with N = 1 unchanged bit for bit. Outcome: `reports/wp-018-engine-2d.md`.
