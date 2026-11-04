+++
id = "WP-027"
status = "done"
goals = ["A2", "A4", "A7", "A8", "A10"]
depends_on = ["WP-023"]
adds_dependencies = []
+++
# WP-027: open transverse boundaries

Open transverse boundaries in 2D: a simple absorbing layer for the fields first (A2), with its thickness and strength as case parameters, and particles that leave the box retired through the alive mask (§0.4). It works when the reflection of a pulse hitting the layer is measured against layer thickness and angle of incidence, and Gauss's law still holds to roundoff in the interior. Outcome: `reports/wp-027-open-boundaries.md`.
