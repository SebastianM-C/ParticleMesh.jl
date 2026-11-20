+++
id = "WP-073"
status = "active"
goals = ["A2", "A3", "A4", "A7", "A8"]
depends_on = ["WP-026", "WP-042", "WP-072"]
+++
# WP-073: the laser antenna in 3D

Let a GaussianLaser with a two-coordinate center drive the 3D step through the total-field/scattered-field antenna, launching the round paraxial beam with its B to order k_⊥² at any polarization angle, and refuse the lasers that have no 3D form yet. It works when the launched beam is the scheme's angular-spectrum propagation of the incident field to the antenna's order, carries the incident Poynting flux and laser_energy, is the mirror of itself under y ↔ z, keeps Gauss's law and ∇·B off the planes, gives gradients in its parameters that agree with finite differences, and leaves the step bitwise across reruns, launch shapes and threads, in Float32, Float64 and `Dual{Float32,3}`.
