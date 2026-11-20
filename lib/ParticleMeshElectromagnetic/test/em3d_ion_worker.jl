# Subprocess worker of T-1045 and T-1054: the Reference runs of `ion3d_digests` (em3d_cases.jl) for
# every element type, without and with the compensated gather, printing one bitwise state digest per
# run. Started by em3d_ionization.jl at 1 and 4 threads, beside its in-process runs.

using Test
include("setup.jl")
include("em_helpers.jl")
include("em2d_cases.jl")
include("window_cases.jl")
include("ionization_cases.jl")
include("em3d_cases.jl")

for (label, T, digest) in ion3d_digests()
    println("EM3DI ", Threads.nthreads(), " ", label, " ", type_label(T), " ", digest)
end
for (label, T, digest) in ion3d_digests(; gather = CompensatedGather())   # T-1054
    println("EM3DG ", Threads.nthreads(), " ", label, " ", type_label(T), " ", digest)
end
