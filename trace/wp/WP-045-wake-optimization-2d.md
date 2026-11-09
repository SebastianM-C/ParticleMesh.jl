+++
id = "WP-045"
status = "done"
goals = ["A0", "A1", "A3", "A7"]
depends_on = ["WP-043", "WP-044"]
adds_dependencies = []
+++
# WP-045: the first 2D optimization: wake amplitude over pulse duration and focus

A user-layer optimization (`scripts/optim/`) that maximizes the RMS on-axis wake amplitude of a small 2D LWFA run over the pulse duration and the focus, at fixed a₀ and density, with ForwardDiff gradients and a box-constrained quasi-Newton maximizer. It works when the optimizer from two starts reaches the best cell of a brute-force grid scan of the same box, with its gradient checked against finite differences and slices through the optimum confirming a maximum, and its duration is compared with the 1D linear-theory value 2/ω_p.
