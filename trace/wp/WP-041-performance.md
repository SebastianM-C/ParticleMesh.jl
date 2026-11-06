+++
id = "WP-041"
status = "proposed"
goals = ["A2", "A4"]
depends_on = ["WP-009", "WP-018", "WP-020"]
+++
# WP-041: deposition, sort and Fast-mode throughput

Measure the 2D step per kernel at production sizes, then remove the measured bottlenecks: the Reference flux recomputing each particle's stencil per node, the full merge sort and gather every step, Fast mode's never-sorted global atomics, and cell binning where nearest-node binning reaches fewer cells. It works when Reference keeps its bitwise determinism and charge conservation (with digests re-recorded only where binning changes) and each change shows a measured speedup on the CPU and the RTX 4080 SUPER.
