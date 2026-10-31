# Run by T-172, T-173, T-174, T-177 and T-113 in a subprocess with a given thread count: prints the
# digests of `thread_digests()`, the one-thread Fast-mode check of T-174 and the flux digests of
# `flux_digests()`.
include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "digest.jl"))
include(joinpath(@__DIR__, "engine_cases.jl"))
include(joinpath(@__DIR__, "flux_cases.jl"))

println("THREADS ", Threads.nthreads())
foreach(l -> println("DIGEST ", l), thread_digests())
foreach(println, fast_spike_lines())
foreach(l -> println("FLUX ", l), flux_digests())
