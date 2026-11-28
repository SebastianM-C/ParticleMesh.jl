+++
id = "WP-090"
status = "active"
goals = ["A5", "A7"]
depends_on = ["WP-082"]
adds_dependencies = []
+++
# WP-090: ParticleMesh against PIConGPU in 3D on one GPU, like for like

The time per step of ParticleMesh and PIConGPU on one GPU, in the same precision, algorithms and box, with start-up, compile and output timed apart: a periodic warm plasma at several sizes and particle counts per cell, giving the particle-push rate, and WP-082's 3D LWFA case with and without the dopant. Outcome: `scripts/bench3d/`, T-1200 to T-1203, and `reports/wp-090-gpu-benchmark-picongpu-3d.md`.
