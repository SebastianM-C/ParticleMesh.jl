# Run by T-136 in a subprocess with a given thread count: prints the digests of one step and of
# the whole Landau run (the configuration of T-141) in Reference mode.
using ParticleMeshCore, KernelAbstractions
include(joinpath(@__DIR__, "es_reference.jl"))
using .ESReference
include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "digest.jl"))

st1, p1, _ = landau_case(Float32; n = 64, ppc = 512, mode = Reference(), tmax_ωp = 1)
initialize!(st1, p1)
step!(st1, p1, 1)
one = state_digest(st1)

st, p, info = landau_case(Float64; LANDAU_KW...)
initialize!(st, p)
run!(st, p, info.nsteps)
println("DIGEST ", Threads.nthreads(), " ", one, " ", state_digest(st))
