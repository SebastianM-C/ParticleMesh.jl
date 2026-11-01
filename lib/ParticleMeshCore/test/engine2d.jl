# WP-018, the engine at N = 2 on the CPU backend: T-260 to T-269.

@testset "T-260 the port keeps N = 1 bit for bit [A0, A4, WP-018]" begin
    # The N = 1 engine digests recorded before the port (n1_digests.jl), in process and in the
    # subprocesses at 1 and 4 threads. Atomic Fast() bits are an A4 claim on one thread only.
    nonatomic(lines) = filter(l -> !startswith(l, "depositfast|"), lines)
    inprocess = n1_lines()
    @test length(N1_RECORDED) == 81
    @test nonatomic(inprocess) == nonatomic(N1_RECORDED)
    Threads.nthreads() == 1 && @test inprocess == N1_RECORDED
    n1_worker(nt) = [chopprefix(l, "N1 ") for l in worker_lines(nt) if startswith(l, "N1 ")]
    @test n1_worker(1) == N1_RECORDED
    @test nonatomic(n1_worker(4)) == nonatomic(N1_RECORDED)
    # T-136's subprocess digests (one Float32 step, the whole T-141 run) at 1 and 4 threads;
    # `Base.hash` is fixed within one Julia version only, as in T-170.
    results = T136_RESULTS[]
    if results === nothing
        @info "T-260: T-136 did not run first; its recorded digests are not rechecked here"
    else
        for r in results
            @test r.one == T136_RECORDED.one skip = VERSION >= v"1.14-"
            @test r.full == T136_RECORDED.full skip = VERSION >= v"1.14-"
        end
    end
end
