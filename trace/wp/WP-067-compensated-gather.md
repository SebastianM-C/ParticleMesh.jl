+++
id = "WP-067"
status = "done"
goals = ["A2", "A7", "A8"]
depends_on = ["WP-059"]
+++
# WP-067: a gather compensated for the particle shape's loss of laser field

Let the particles (the push, the ionization rates, the births' kick) read a copy of E and B sharpened along the laser axis so that the shape's mean transfer at the laser's wavenumber is one, as a numerics option that leaves the mesh solve, the current and the diagnostics on the grid's fields. It works when the copy and the particles' gather match their closed-form transfers to roundoff, Gauss's law and the Reference mode's bits are untouched, and the gradient through it agrees with finite differences.
