+++
id = "WP-033"
status = "active"
goals = ["A4", "A7", "A8"]
depends_on = ["WP-023"]
adds_dependencies = []
+++
# WP-033: a Maxwell–Jüttner quiet start

The engine's quiet start draws momenta from a Maxwell–Jüttner distribution, boosted for a drift, from Philox draws (A4), so every loaded γ is finite, and the electromagnetic module loads isotropic thermal species this way by default. It works when the rest-frame and boosted moments and distributions match quadrature from θ = 1e−5 to 100, and the 1D and 2D Landau rates agree with the relativistic linear theory of that load.
