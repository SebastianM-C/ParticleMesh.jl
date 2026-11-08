+++
id = "WP-043"
status = "done"
goals = ["A0", "A3", "A7", "A8"]
depends_on = ["WP-026", "WP-028"]
adds_dependencies = []
+++
# WP-043: a gradient through a small laser-wakefield run

A ForwardDiff gradient of a smooth wake-amplitude functional (the soft maximum of the on-axis E_x at the end) through a small 2D LWFA run with the antenna, the moving window, the density ramp and a plasma, with respect to a₀, the duration, the focus, the density and the ramp length together. It works when each partial agrees with the finite-difference floor of A7 (`fd_scan`, Float64, Reference), and the `Dual{Float32,3}` partials agree with the Float64 ones within a measured tolerance. Outcome: T-580 to T-582 in `lib/ParticleMeshElectromagnetic/test/lwfa_gradient.jl`.
