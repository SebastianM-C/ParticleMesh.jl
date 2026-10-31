# Run by T-172, T-173, T-174 and T-177 in a subprocess with a given thread count: prints the
# digests of `thread_digests()` and the one-thread Fast-mode check of T-174.
include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "digest.jl"))
include(joinpath(@__DIR__, "engine_cases.jl"))

println("THREADS ", Threads.nthreads())
foreach(l -> println("DIGEST ", l), thread_digests())
foreach(println, fast_spike_lines())
