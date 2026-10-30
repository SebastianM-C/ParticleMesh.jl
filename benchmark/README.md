# Benchmarks

Regression microbenchmarks of the engine (BenchmarkTools `SUITE` in `benchmarks.jl`), a workspace member of the root project, so CI can run them against the shared manifest. The benchmark case definitions with expected values (the 1D set, the 2D laser-wakefield case matched to the reference codes) and the particle-workload stress suite live in `scripts/`, because they are user code that produces result artifacts. Results go to `benchmark/results/` (ignored) and are recorded as run records in `trace/`. Report GPU and CPU models, never machine names.
