# Subprocess worker of T-952 and T-1014: the Reference runs of the periodic 3D plasma and of the
# open one (em3d_cases.jl) for every element type and shape or absorber, printing one bitwise
# state digest per run. Started by em3d_step.jl at 1 and 4 threads, beside its in-process runs.

using Test
include("setup.jl")
include("em_helpers.jl")
include("em2d_cases.jl")
include("em3d_cases.jl")

for (label, T, digest) in em3d_step_digests()
    println("EM3DJ ", Threads.nthreads(), " ", label, " ", type_label(T), " ", digest)
end
for (label, T, digest) in open3d_step_digests()
    println("EM3DO ", Threads.nthreads(), " ", label, " ", type_label(T), " ", digest)
end
