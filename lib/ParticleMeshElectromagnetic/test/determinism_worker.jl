# Subprocess worker of T-218, T-355, T-363 and T-434: short Reference runs of the three cases, of
# the 2D plasma (em2d_cases.jl) and of the 2D plasma with an open axis (boundaries_cases.jl) for
# every element type, printing one bitwise state digest per run. Run once per suite at 1 and 4
# threads (`worker_output`).

using Test
include("setup.jl")
include("em_helpers.jl")
include("determinism_cases.jl")
include("em2d_cases.jl")
include("boundaries_cases.jl")

for (name, T, digest) in determinism_digests()
    println("DIGEST ", Threads.nthreads(), " ", name, " ", type_label(T), " ", digest)
end
for (T, digest) in em2d_digests()
    println("EM2D ", Threads.nthreads(), " ", type_label(T), " ", digest)
end
for (T, digest) in open_step_digests()
    println("EM2DO ", Threads.nthreads(), " ", type_label(T), " ", digest)
end
for (T, digest) in em2d_step_digests()
    println("EM2DJ ", Threads.nthreads(), " ", type_label(T), " ", digest)
end
