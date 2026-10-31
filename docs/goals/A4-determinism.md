# A4. Determinism policy

FP32 atomics make deposition order-dependent (relative run-to-run noise ~1e-7), which breaks finite-difference gradient checks at small steps. Decide: a sort-based deterministic deposition as the reference mode, atomics as the fast mode, bitwise reproducibility as a tested property of the reference mode, counter-based RNG (Philox-style) for particle loading so seeds mean the same thing on every backend.

Bitwise means: the same element type on the same backend gives the same bits across reruns and thread counts. It does not hold across element types: the value part of a dual run is not the Float64 run bit for bit, because the compiler may fuse operations differently on the two paths (observed as 1 ulp in the loader), so gradient checks compare within the measured floor of A7, never bit for bit, and agreement across backends is a measured tolerance.
