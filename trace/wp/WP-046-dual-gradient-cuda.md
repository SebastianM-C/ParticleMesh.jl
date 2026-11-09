+++
id = "WP-046"
status = "done"
goals = ["A4", "A7", "A8"]
depends_on = ["WP-013", "WP-043", "WP-044"]
adds_dependencies = []
+++
# WP-046: the LWFA gradient with dual numbers on CUDA

Run WP-043's `Dual{Float32,3}` gradient through the 2D step with the antenna, the moving window and the density ramp on CUDA, fixing what breaks there, at WP-043's size and at a refined, longer case. It works when the CUDA partials agree with the CPU's within a measured tolerance, the Reference run is bitwise across reruns and launch shapes with the Fast mode's bits equal to it, and the partials of cases refined two and three times converge with resolution, with the time per step on both backends recorded. Outcome: T-610 to T-613 in `lib/ParticleMeshElectromagnetic/test/lwfa_gradient.jl` and `gpu.jl`, measured in the PR that added them.
