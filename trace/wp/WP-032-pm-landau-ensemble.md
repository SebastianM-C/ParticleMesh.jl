+++
id = "WP-032"
status = "done"
goals = ["A5", "A7"]
depends_on = ["WP-023", "WP-025"]
adds_dependencies = []
+++
# WP-032: ParticleMesh's oblique Landau ensemble over WP-025's 32 seeds

ParticleMesh's oblique 2D Landau ensemble over the 32 seeds of WP-025, run on the CPU box in requests of a few seeds each so that an eviction loses at most one, and compared with the committed Smilei and PIConGPU ensembles. It works when the three ensembles agree within their combined jackknife errors or the difference is explained, and ParticleMesh's and Smilei's ns per particle-step on the idle 16 cores are recorded. Outcome: `reports/wp-032-pm-landau-ensemble.md`.
