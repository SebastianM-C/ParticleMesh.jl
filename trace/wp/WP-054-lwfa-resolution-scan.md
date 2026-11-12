+++
id = "WP-054"
status = "done"
goals = ["A5", "A7"]
depends_on = ["WP-053"]
adds_dependencies = []
+++
# WP-054: A5's first LWFA case on three grids

A5's first 2D laser-wakefield case run at λ₀/32, λ₀/48 and λ₀/64 in ParticleMesh, Smilei and PIConGPU, with everything physical unchanged and the outputs at the same times, to say how each code's own observables change with the grid and which code-to-code distances are discretization and shrink and which stay (A7 sets its LWFA threshold after this). It works when the three grids' runs of each code are compared on the lattice points they share, an observed order is reported only where the three values support one, and the distances of `reports/lwfa-2d-a5.md` are given at each resolution. Outcome: a report on the scan.
