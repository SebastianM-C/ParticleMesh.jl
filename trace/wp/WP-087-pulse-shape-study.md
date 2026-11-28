+++
id = "WP-087"
status = "active"
goals = ["A0", "A3", "A7", "A8"]
depends_on = ["WP-065", "WP-066", "WP-068"]
adds_dependencies = []
+++
# WP-087: optimizing the pulse's shape at a fixed plasma and laser energy

Maximize WP-065's trapped-electron objective over the pulse's shape alone, the plasma and the laser energy fixed: the focus, a wavefront of aberration modes on a fixed super-Gaussian near field (Legendre modes P₃ and P₄ on the 2D antenna, then Zernike modes to radial order 4 on WP-085's 3D one) and the GDD and TOD, with L-BFGS-B from several random starts, since a mode odd under a mirror of the box that keeps the polarization has no gradient at an unaberrated pulse. It is understood when the energy is shown to be fixed at every point and the antenna launches it, the gradient matches central differences, and the starts' paths, optima and the shapes they reach are in a report, first in 2D and then in 3D. Outcome: T-1180 to T-1189 in `scripts/optim/test/runtests.jl` and `reports/wp-087-pulse-shape-study.md`.
