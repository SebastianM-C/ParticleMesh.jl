+++
id = "WP-074"
status = "active"
goals = ["A2", "A3", "A4", "A7", "A8"]
depends_on = ["WP-028", "WP-034", "WP-072", "WP-073"]
+++
# WP-074: the moving window in 3D

Let the 3D step run in a moving window along x, with the density ramp and the refill of fresh plasma at the injection column, the antennas left behind in the lab frame, and the PML on the window's back edge, generalizing the 2D window over N so that the 2D runs keep their bits. It works when a 3D window run that is invariant along y or z is the 2D window run to roundoff, a cold plasma stays uniform and neutral while the window renews it, Gauss's law holds across shifts with a laser driving the plasma, the refill is the static quiet start bit for bit, the Reference step is bitwise across reruns, launch shapes and threads, and the field energy is differentiable in the ramp length, in Float32, Float64 and `Dual{Float32,3}`.
