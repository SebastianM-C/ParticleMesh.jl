+++
id = "WP-031"
status = "active"
goals = ["A5", "A7"]
depends_on = ["WP-012", "WP-023"]
adds_dependencies = []
+++
# WP-031: are the common offsets from linear theory discretization error?

Scan Δx and Δt in Smilei on the CPU box for the cold cases whose rates both codes place below linear theory by the same amount, the 2D Weibel growth rate (−1.1 %, WP-023) and the 1D two-stream rate (−1.5 %, #168 and #226), and the noisy 1D Landau rate (#167) if the same scan can resolve it. The check is the offset's limit as Δx and Δt go to zero against the same fit applied to the exact linear solution of the run's initial state, with the tolerance that limit supports for A7 proposed in `questions/`.
