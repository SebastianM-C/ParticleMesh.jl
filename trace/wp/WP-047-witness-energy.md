+++
id = "WP-047"
status = "active"
goals = ["A0", "A3", "A7", "A8"]
depends_on = ["WP-028", "WP-043", "WP-044"]
adds_dependencies = []
+++
# WP-047: a witness bunch and its mean energy gain

A bunch of test electrons injected at fixed phases of the moving window, whose bunch parameters enter only through its averaging weights (A3), and its weighted mean energy gain as a differentiable observable of a 2D LWFA run, the objective of the 2D optimization. It works when a witness in a prescribed field gains the energy the field's work predicts, and when ForwardDiff through WP-043's case agrees with the finite-difference floor of A7 for the laser, plasma and bunch parameters.
