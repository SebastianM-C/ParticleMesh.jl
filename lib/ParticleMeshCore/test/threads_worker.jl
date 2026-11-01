# Run by T-172, T-173, T-174, T-177, T-113, T-260, T-266, T-303 and T-312 in a subprocess with a given
# thread count: prints the digests of `thread_digests()`, the one-thread Fast-mode check of T-174,
# the flux digests of `flux_digests()`, the N = 1 engine digests of `n1_lines()`, the 2D digests of
# `engine2d_digests()`, the 2D flux digests of `flux2d_digests()` and the periodic-solve digests of `spectral_digests()`.
include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "digest.jl"))
include(joinpath(@__DIR__, "engine_cases.jl"))
include(joinpath(@__DIR__, "flux_cases.jl"))
include(joinpath(@__DIR__, "n1_digests.jl"))
include(joinpath(@__DIR__, "engine2d_cases.jl"))
include(joinpath(@__DIR__, "flux2d_cases.jl"))
include(joinpath(@__DIR__, "spectral_cases.jl"))

println("THREADS ", Threads.nthreads())
foreach(l -> println("DIGEST ", l), thread_digests())
foreach(println, fast_spike_lines())
foreach(l -> println("FLUX ", l), flux_digests())
foreach(l -> println("N1 ", l), n1_lines())
foreach(l -> println("E2D ", l), engine2d_digests())
foreach(l -> println("F2D ", l), flux2d_digests())
foreach(l -> println("SPECTRAL ", l), spectral_digests())
