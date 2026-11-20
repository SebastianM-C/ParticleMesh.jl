# WP-083: run only T-1140 of gpu.jl (and T-1141) on CUDA. From the repository root of this branch:
#     PM_TEST_GPU=1 julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/wp083/t1140_gpu.jl
using Test
const TD = joinpath(pwd(), "lib", "ParticleMeshElectromagnetic", "test")
include(joinpath(TD, "setup.jl"))
include(joinpath(TD, "antenna_enzyme.jl"))
GPU_BACKEND === nothing && error("no functional GPU: ", GPU_SWITCH.reason)
CUDACore.allowscalar(false)
const GPU = GPU_BACKEND
println("WP-083 GPU test on ", CUDACore.name(CUDACore.device()))
function find_testsets(ex, ids, out = Any[])
    if ex isa Expr
        if ex.head == :macrocall && string(ex.args[1]) == "@testset" && length(ex.args) >= 3 && any(id -> occursin(id, string(ex.args[3])), ids)
            push!(out, ex)
        else
            foreach(a -> find_testsets(a, ids, out), ex.args)
        end
    end
    return out
end
sets = find_testsets(Meta.parseall(read(joinpath(TD, "gpu.jl"), String)), ("T-1140",))
length(sets) == 1 || error("expected T-1140 in gpu.jl, found $(length(sets))")
@testset "WP-083 on the GPU" begin
    Core.eval(Main, sets[1])
end
println("\nMeasured err/(eps(T)·scale):")
for id in sort(collect(keys(MEASURED)))
    println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
end
