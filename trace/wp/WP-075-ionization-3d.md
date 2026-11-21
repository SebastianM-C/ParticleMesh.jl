+++
id = "WP-075"
status = "done"
goals = ["A2", "A3", "A4", "A7", "A8"]
depends_on = ["WP-059", "WP-063", "WP-072", "WP-074"]
+++
# WP-075: field ionization in the 3D step

Let the 3D step carry the dopant of WP-059 and WP-063, its lattice across the transverse cells of its rows of y and z, the rates of its levels' laws, its born electrons and its window refill and open faces, generalizing the 2D ionization over N so that the 2D runs keep their bits. It works when a 3D run with a dopant that is invariant along y or z is the 2D run to roundoff, the populations in a constant field follow the closed-form ADK and Tong–Lin rates, Gauss's law holds with the ions' charge in the box, at open faces and across window shifts, the Reference step is bitwise across reruns, launch shapes and threads, and the born charge is differentiable through the rates and weights in Float64 and `Dual{Float32,3}`.
