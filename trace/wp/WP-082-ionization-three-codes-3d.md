+++
id = "WP-082"
status = "proposed"
goals = ["A5", "A7"]
depends_on = ["WP-075", "WP-081"]
adds_dependencies = []
+++
# WP-082: the ionization-injection case in 3D in three codes

WP-081's case in 3D, from the same driver with a third cell count, in ParticleMesh, Smilei and PIConGPU. It works when the codes agree on the pulse, the wake and the born electrons as closely as in 2D, and the 2D and 3D runs differ as the geometry predicts (a focused beam's on-axis field and the smaller trapped charge).
