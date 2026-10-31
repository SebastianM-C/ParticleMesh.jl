# Subprocess worker of T-218: short Reference runs of the three cases for every element type,
# printing one bitwise state digest per run. Run by determinism.jl at 1 and 4 threads.

using Test
include("setup.jl")
include("em_helpers.jl")
include("determinism_cases.jl")

for (name, T, digest) in determinism_digests()
    println("DIGEST ", Threads.nthreads(), " ", name, " ", type_label(T), " ", digest)
end
