+++
id = "WP-030"
status = "proposed"
goals = ["A5", "A6", "A7"]
depends_on = ["WP-017", "WP-019"]
adds_dependencies = []
+++
# WP-030: Smilei on the GPU

A GPU build of Smilei v5.1 (the nvhpc route of Smilei's own GPU install guide) as its own container family, `smilei-gpu`, next to the CPU image, which stays the reference. It works when it agrees with the CPU build within the seed spread on the 2D oblique Landau and Weibel cases, and its time per particle-step is recorded on the RTX 4080 SUPER next to PIConGPU's and ParticleMesh's (A5).
