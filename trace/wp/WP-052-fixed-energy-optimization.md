+++
id = "WP-052"
status = "done"
goals = ["A0", "A1", "A3", "A7"]
depends_on = ["WP-045", "WP-047"]
adds_dependencies = []
+++
# WP-052: the 2D optimization at a fixed laser energy

A user-layer optimization (`scripts/optim/`) that maximizes the witness bunch's mean energy gain of a small 2D LWFA run over the pulse duration, the plasma density and the witness centre, with a₀ coupled to the duration so that the laser energy stays fixed, so that the optimum lies inside the box instead of at a bound. It works when the launched energy is the same for every duration, the gradient agrees with finite differences, the optimizer from two starts reaches the best of a grid scan with slices confirming an interior maximum, and the same problem at fixed a₀ runs to a bound. Outcome: `reports/wp-052-fixed-energy-optimization.md`.
