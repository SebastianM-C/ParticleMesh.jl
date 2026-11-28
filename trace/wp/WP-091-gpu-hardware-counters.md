+++
id = "WP-091"
status = "active"
goals = ["A0", "A6", "A8"]
depends_on = ["WP-041", "WP-068", "WP-084"]
adds_dependencies = ["GPUDiagnostics"]
+++
# WP-091: what limits the step's kernels on the GPU, from hardware counters

Collect Nsight Compute counters through GPUDiagnostics 0.6 over a window of steady-state steps of the optimization's 2D and 3D trapped-beam cases, in Float32 and in the gradient runs' duals, on an RTX 5090, and say per kernel group what holds the step's time and what limits it (memory bandwidth, latency, occupancy, atomics, divergence). It is done when GPUDiagnostics' known-count probe passes on the card, the per-kernel table and figures of each case are in a report with their run manifests, and the report ranks the optimization targets.
