# Subprocess worker of T-327: the WP-022 solver runs at N = 1 and N = 2 for every element type,
# printing one bitwise digest per run. Run by yee_nd.jl at 1 and 4 threads.

using Test
include("setup.jl")
include("em_helpers.jl")
include("yee_nd_cases.jl")

for (N, T, digest) in yee_nd_digests()
    println("YEEND ", Threads.nthreads(), " ", N, " ", type_label(T), " ", digest)
end
