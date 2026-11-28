# Subprocess worker of T-1253: the Reference runs of the 3D step with the filter and the sort from
# the last order (nd_features_cases.jl) for every element type, printing one bitwise state digest
# per run. Started by nd_features.jl at 1 and 4 threads, beside its in-process runs.

using Test
include("setup.jl")
include("em_helpers.jl")
include("em2d_cases.jl")
include("boundaries_cases.jl")
include("window_cases.jl")
include("em3d_cases.jl")
include("nd_features_cases.jl")

for (label, T, digest) in features3d_digests()
    println("ND3DF ", Threads.nthreads(), " ", label, " ", type_label(T), " ", digest)
end
