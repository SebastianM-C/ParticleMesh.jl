# WP-013: T-199, the GPU switch (gpu_switch.jl), on the CPU.

"Run gpu_switch.jl in a fresh process with `PM_TEST_GPU` set to `env` (or unset) and no visible CUDA device."
function switch_subprocess(env)
    code = """
    include($(repr(joinpath(@__DIR__, "gpu_switch.jl"))))
    println("FUNCTIONAL ", CUDACore.functional())
    sw = gpu_switch()
    println(sw.run ? "RUN" : gpu_skip_line(sw))
    """
    vars = Dict{String,Union{String,Nothing}}("CUDA_VISIBLE_DEVICES" => "", "PM_TEST_GPU" => env)
    cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) -e $code`, vars)
    out = IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = devnull))
    return (; ok = success(p), lines = split(String(take!(out)), '\n'; keepempty = false))
end

@testset "T-199 the GPU switch [A6, WP-013]" begin
    # The decision, for every combination of the variable and the GPU.
    @test gpu_switch("", false).run == false
    @test gpu_switch("", true).run == true
    @test gpu_switch("0", true).run == false && gpu_switch("0", false).run == false
    @test gpu_switch("1", true).run == true
    @test_throws ErrorException gpu_switch("1", false)
    @test_throws ErrorException gpu_switch("yes", true)
    @test startswith(gpu_skip_line(gpu_switch("", false)), "GPU tests (WP-013) not run: ")
    # The suite's own path, in a process with no visible CUDA device: unset or 0, one line and a
    # clean exit; 1, a failure. Should CUDA still report itself functional there, the unset case
    # runs the GPU tests instead, as it must on a machine with a GPU.
    unset = switch_subprocess(nothing)
    @test unset.ok
    functional = "FUNCTIONAL true" in unset.lines
    if functional
        @test "RUN" in unset.lines
    else
        @test count(startswith("GPU tests (WP-013) not run: "), unset.lines) == 1
        @test !switch_subprocess("1").ok
    end
    off = switch_subprocess("0")
    @test off.ok && "GPU tests (WP-013) not run: PM_TEST_GPU=0" in off.lines
end
