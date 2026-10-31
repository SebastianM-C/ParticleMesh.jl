# A5. Canonical benchmark cases with numbers

The 1D set (Landau damping, two-stream growth rate, Weibel) with expected values; one 2D LWFA case fully specified (a₀, w₀, τ, n₀, ramp, box, resolution, particles per cell, steps, diagnostics) that is also run in Smilei and PIConGPU (containerized). Experience from earlier cross-code work: matched setups are the under-specified, labor-intensive part; do it once, in a versioned config.

Verification against both PIConGPU and Smilei covers correctness and performance. For the performance comparison to be fair the configs must match shape-function order, current deposition scheme (Esirkepov in all three), filtering, precision (PIConGPU defaults to FP32, Smilei to FP64), particles per cell and output cadence, and run on the same card; report particle-steps per second and cell-steps per second with the GPU model named, never the hostname.
