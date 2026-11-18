# Subprocess worker of T-952: the Reference runs of the 3D plasma (em3d_cases.jl) for every element
# type and shape, printing one bitwise state digest per run. Started by em3d_step.jl at 1 and 4
# threads, beside its in-process runs.

using Test
include("setup.jl")
include("em_helpers.jl")
include("em2d_cases.jl")
include("em3d_cases.jl")

for (label, T, digest) in em3d_step_digests()
    println("EM3DJ ", Threads.nthreads(), " ", label, " ", type_label(T), " ", digest)
end
