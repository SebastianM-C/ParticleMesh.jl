+++
id = "WP-050"
status = "parked"
goals = ["A0", "A3", "A7", "A8"]
depends_on = ["WP-023"]
adds_dependencies = []
+++
# WP-050: a reverse-mode spike with Enzyme through the 2D step

Enzyme reverse mode through steps of the 2D electromagnetic module (Esirkepov current, both deposition modes), checked against ForwardDiff by the chain rule df/dα = ⟨∇f, dx/dα⟩ near roundoff, with a report of what works and what blocks. Parked until an Enzyme release accepts LLVM.jl 10 (KernelAbstractions 0.10 requires it), so that the check can move into the test suite; outcome in `reports/wp-050-enzyme-spike.md`.
