# Subprocess worker of T-952, T-1014, T-1027 and T-1034: the Reference runs of the periodic 3D plasma,
# the open one, the 3D antenna and the 3D window (em3d_cases.jl) for every element type and shape, absorber or case, printing one bitwise
# state digest per run. Started by em3d_step.jl at 1 and 4 threads, beside its in-process runs.

using Test
include("setup.jl")
include("em_helpers.jl")
include("em2d_cases.jl")
include("window_cases.jl")
include("em3d_cases.jl")

for (label, T, digest) in em3d_step_digests()
    println("EM3DJ ", Threads.nthreads(), " ", label, " ", type_label(T), " ", digest)
end
for (label, T, digest) in open3d_step_digests()
    println("EM3DO ", Threads.nthreads(), " ", label, " ", type_label(T), " ", digest)
end
for (label, T, digest) in antenna3d_step_digests()
    println("EM3DA ", Threads.nthreads(), " ", label, " ", type_label(T), " ", digest)
end
for (label, T, digest) in window3d_step_digests()
    println("EM3DW ", Threads.nthreads(), " ", label, " ", type_label(T), " ", digest)
end
