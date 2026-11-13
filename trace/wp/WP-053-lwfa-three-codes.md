+++
id = "WP-053"
status = "active"
goals = ["A5", "A7"]
depends_on = ["WP-017", "WP-026", "WP-028"]
adds_dependencies = []
+++
# WP-053: A5's first LWFA case in three codes

A5's first 2D laser-wakefield case run in ParticleMesh, Smilei and PIConGPU from one case file, with one launched pulse and one region and unit for the electron spectra, and compared pair by pair: laser, wake and trapped electrons, the Smilei–PIConGPU distance next to ParticleMesh's distance to each (A7). It works when the three vacuum runs show the same pulse on the axis (carrier phase, centroid and amplitude, with the focus ahead of the injection plane and on it) and the plasma runs' differences are measured against that. Outcome: `reports/lwfa-2d-a5.md`.
