+++
id = "WP-062"
status = "done"
goals = ["A4", "A7", "A8"]
depends_on = ["WP-055", "WP-061"]
+++
# WP-062: the recorder's cadence and selection

The history recorder records every n-th step and only the groups asked for (field energies, kinetic energy, mode amplitudes), and a step computes the kinetic-energy cache once, and only where the next step's observer reads it. What is recorded has the bits of the same entries of the recorder that records everything, and a run's evolved arrays are the same under every observer; the outcome is in `reports/wp-062-recorder-options.md`.
