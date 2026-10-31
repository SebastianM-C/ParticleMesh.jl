# A4. Determinism policy

FP32 atomics make deposition order-dependent (relative run-to-run noise ~1e-7), which breaks finite-difference gradient checks at small steps. Decide: a sort-based deterministic deposition as the reference mode, atomics as the fast mode, bitwise reproducibility as a tested property of the reference mode, counter-based RNG (Philox-style) for particle loading so seeds mean the same thing on every backend.
