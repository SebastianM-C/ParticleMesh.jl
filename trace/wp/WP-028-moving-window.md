+++
id = "WP-028"
status = "proposed"
goals = ["A2", "A3", "A4", "A7", "A8"]
depends_on = ["WP-023"]
adds_dependencies = []
+++
# WP-028: the moving window

A window moving at c along the propagation axis that shifts fields and particles by whole cells and loads fresh plasma at its front from a longitudinal density profile (vacuum, ramp, plateau), with the profile entering through particle weights (A3) and fixed-capacity buffers (§0.4). It works when a uniform plasma stays uniform and neutral to roundoff in the window frame, a vacuum pulse is the same with and without the window, and Gauss's law holds across shifts.
