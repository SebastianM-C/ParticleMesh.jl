+++
id = "WP-080"
status = "done"
goals = ["A0", "A4", "A7", "A8"]
depends_on = ["WP-042", "WP-050", "WP-056", "WP-077"]
adds_dependencies = ["Checkpointing"]
+++
# WP-080: checkpointed reverse mode through the 2D and 3D step

Differentiate full-length runs of the 2D and 3D step in Enzyme's reverse mode through Checkpointing.jl's Revolve scheme, with checkpoints that hold only the evolved arrays, and give the 3D flux deposition its pullback and reverse rule. It works when the checkpointed gradient equals the plain reverse pass's and the duals' by the chain rule in 2D and 3D, its memory is bounded by the checkpoints and one step's tape, and its time and memory against the plain run, Dual{1}, Dual{4} and the plain reverse pass are measured at production size.
