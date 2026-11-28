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

@testset "T-1210 the GPU testsets can wait for their turn in a machine's GPU queue (PM_GPU_HOLD) [A6]" begin
    @test gpu_hold("") === nothing
    released = joinpath(mktempdir(), "released")
    # A queue's hold mode: answers once it has the GPU, keeps it until its input closes.
    p = gpu_hold("echo granted; cat >/dev/null; touch $(released)")
    @test p isa Base.Process && p in GPU_HOLD
    sleep(0.2)
    @test process_running(p) && !isfile(released)
    close(p.in)
    wait(p)
    @test success(p) && isfile(released)
    filter!(!=(p), GPU_HOLD)
    # Anything but "granted" is refused, so the GPU testsets never run without their turn.
    @test_throws ErrorException gpu_hold("echo busy")
    @test_throws ErrorException gpu_hold("exit 1")
end

@testset "T-1211 the machine GPU queue runs one job at a time, in priority order, with the hold among them [A6]" begin
    queue = normpath(joinpath(@__DIR__, "..", "..", "..", "scripts", "gpu-queue.sh"))
    q = mktempdir()
    order = joinpath(q, "order.txt")
    env = ("GPU_QUEUE_DIR" => q, "PM_GPU_LOCK" => joinpath(q, "lock"), "GPU_QUEUE_POLL" => "0.1")
    gq(args...) = addenv(`bash $queue $args`, env...)
    job(name) = "echo start $name >> $order; sleep 0.3; echo end $name >> $order"
    withenv(env...) do
        # A job already running outside the queue books the card first; the rest wait for it.
        outside = run(`sleep 1.5`; wait = false)
        run(gq("--pid", string(getpid(outside)), "-p", "0", "-t", "outside"); wait = false)
        sleep(0.3)
        for (tag, prio) in (("A", 5), ("B", 5), ("C", 1), ("D", 9))
            read(gq("-d", "-t", tag, "-p", string(prio), "--", "sh", "-c", job(tag)), String)
            sleep(0.05)
        end
        read(gq("-d", "-t", "X", "--", "sh", "-c", "exit 3"), String)
        hold = gpu_hold("bash $queue --hold -t suite")    # waits behind X, ahead of D
        open(io -> println(io, "start suite"), order, "a")
        sleep(0.3)
        open(io -> println(io, "end suite"), order, "a")
        close(hold.in)
        wait(hold)
        filter!(!=(hold), GPU_HOLD)
        deadline = time() + 30
        while !(isfile(order) && count(==("end D"), readlines(order)) == 1) && time() < deadline
            sleep(0.1)
        end
        @test readlines(order) == ["start C", "end C", "start A", "end A", "start B", "end B",
                                   "start suite", "end suite", "start D", "end D"]
        logs = readdir(joinpath(q, "logs"); join = true)
        xlog = only(filter(contains("/X-"), logs))
        @test startswith(last(readlines(xlog)), "# gpu-run finished: exit 3 after ")
        @test success(outside)
        @test contains(read(gq("--status"), String), "waiting, in the order they will run:\n  (none)")
    end
end
