+++
id = "WP-072"
status = "active"
goals = ["A2", "A4", "A7", "A8"]
depends_on = ["WP-027", "WP-034", "WP-042"]
+++
# WP-072: open transverse boundaries in 3D

Let the 3D step open y, z or both, with the absorbing layer or the PML on each open axis and the particles that cross an open face retired, generalizing the 2D absorbers over the axis so that the 2D runs keep their bits. It works when a 3D run that does not vary along a periodic axis is the 2D open run to roundoff, Gauss's law holds off the layers with particles leaving through every face, the Reference step is bitwise across reruns, launch shapes and threads, and a pulse leaves through the faces and corners without late growth, in Float32, Float64 and `Dual{Float32,3}`.
