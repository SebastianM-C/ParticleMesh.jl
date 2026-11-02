+++
id = "WP-013"
status = "done"
goals = ["A2", "A4", "A5", "A6", "A7", "A8", "A9"]
depends_on = ["WP-008", "WP-009", "WP-011", "WP-014", "WP-018"]
adds_dependencies = ["CUDACore"]
+++
# WP-013: the engine and the electromagnetic module on CUDA

Run the engine and the electromagnetic module on the RTX 4080 SUPER: the CPU identities hold, Reference mode is bitwise across reruns and launch shapes, and first throughput and memory are measured. Outcome: `reports/gpu-cuda.md`.
