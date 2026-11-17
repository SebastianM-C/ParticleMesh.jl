# The GPU testsets of the electromagnetic module that run the 2D flux deposition, taken from its
# test/gpu.jl, after the CPU test files that define their bodies and store the CPU runs they
# compare with (the suite's own setup, in the suite's order):
#
#   PM_TEST_GPU=1 julia +1.13 -t 8 --project=<repo>/lib/ParticleMeshElectromagnetic/test em_gpu_tests.jl <repo>
#
# With `cpu` as a second argument only the CPU files run (a dry run of the file order).
using Test
const ROOT = ARGS[1]
const DIR = joinpath(ROOT, "lib", "ParticleMeshElectromagnetic", "test")
const DRY = length(ARGS) >= 2 && ARGS[2] == "cpu"
# `PM_EM_SET=sort` adds what the step's sort from the last order touches (WP-041): the window's
# allocation, the restart and the field ionization on the GPU, and the PR's own CPU tests.
const SET = get(ENV, "PM_EM_SET", "flux")
const FILES = ["determinism.jl", "em2d.jl", "em2d_step.jl", "filter.jl", "window.jl", "lwfa_gradient.jl", "witness.jl",
               (SET == "sort" ? ["gradient.jl", "harness.jl", "params.jl", "restart.jl", "ionization.jl", "resort_step.jl"] : String[])...]
const IDS = ["T-380", "T-381", "T-446", "T-560", "T-561", "T-645", "T-613", "T-611", (SET == "sort" ? ["T-562", "T-747", "T-780"] : String[])...]

"Evaluate `code` as if it stood in the suite's file `name`, so that its relative includes resolve there."
as_file(code, name) = task_local_storage(() -> include_string(Main, code, joinpath(DIR, name)), :SOURCE_PATH, joinpath(DIR, name))

# runtests.jl up to its list of files: the setup and the case files
as_file(first(split(read(joinpath(DIR, "runtests.jl"), String), "# The suite's files in their order")), "runtests.jl")
const GPU_LINES = split(read(joinpath(DIR, "gpu.jl"), String), '\n')
as_file(join(GPU_LINES[1:(findfirst(startswith("if GPU_BACKEND === nothing"), GPU_LINES) - 1)], '\n'), "gpu.jl")
if !DRY
    GPU_BACKEND === nothing && error("no GPU: " * gpu_skip_line(GPU_SWITCH))
    CUDACore.allowscalar(false)
    @eval const GPU = GPU_BACKEND
    println("GPU tests on ", CUDACore.name(CUDACore.device()), ": ", GPU_SWITCH.reason)
end

"The source of the testset `id` of gpu.jl: from its `@testset` line to the `end` at the same indentation."
function testset_source(id)
    a = findfirst(startswith("    @testset \"$id "), GPU_LINES)
    b = a + findfirst(==("    end"), GPU_LINES[(a + 1):end])
    return join(GPU_LINES[a:b], '\n')
end

@testset "the module's GPU test sets ($SET)" begin
    try
        for f in FILES
            t = time()
            Base.include(Main, joinpath(DIR, f))
            println("Time of $f: $(round(time() - t; digits = 1)) s")
            flush(stdout)
        end
        for id in (DRY ? () : IDS)
            t = time()
            as_file(testset_source(id), "gpu.jl")
            println("Time of $id: $(round(time() - t; digits = 1)) s")
            flush(stdout)
        end
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
