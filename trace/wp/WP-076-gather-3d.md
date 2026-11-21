+++
id = "WP-076"
status = "done"
goals = ["A2", "A4", "A7", "A8"]
depends_on = ["WP-067", "WP-074", "WP-075"]
+++
# WP-076: the compensated gather in the 3D step

Let the 3D step take the `CompensatedGather` of WP-067, the particles (the push, the ionization's rates and births, the initial kick) reading a copy of E and B sharpened along x on every transverse line, so that the 2D runs keep their bits. It works when the 3D copy and the particles' gather match their closed-form transfers to roundoff whatever the transverse wavenumbers, a 3D run with it that is invariant along y or z is the 2D run with it to roundoff, Gauss's law and the Reference mode's bits across reruns, launch shapes and threads hold with it in the box and in the window, and the gradient through it agrees with finite differences in Float64 and `Dual{Float32,3}`.
