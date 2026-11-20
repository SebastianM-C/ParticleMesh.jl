+++
id = "WP-050"
status = "done"
goals = ["A0", "A3", "A7", "A8"]
depends_on = ["WP-023"]
adds_dependencies = ["UnsafeAtomics"]
+++
# WP-050: a reverse-mode spike with Enzyme through the 2D step

Enzyme reverse mode through steps of the 2D electromagnetic module on the GPU, checked against ForwardDiff by the chain rule df/dα = ⟨∇f, dx/dα⟩ near roundoff, with its cost against the plain run and dual numbers and a report of what works and what blocks; outcome in `reports/wp-050-enzyme-spike.md`.
