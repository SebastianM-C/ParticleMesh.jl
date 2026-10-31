# T-220: the allocation of one step, after warm-up, does not grow with the particle count or the
# grid size (measured at two sizes, not asserted to be zero bytes). At more than one thread the
# Reference sort allocates per particle in AcceleratedKernels' threaded CPU sample sort (#82,
# the open item of the engine's T-144); CI runs one thread.

function step_bytes(name, mode; ppc, cells)
    st, p, prob, _ = case_state(name, Float64; mode, ppc, cells, t_end = 1)
    step!(st, p, 1)
    step!(st, p, 2)
    return @allocated step!(st, p, 3)
end

@testset "T-220 allocation per step does not grow with particle count or grid size [A7, WP-014]" begin
    for name in ("landau", "weibel"), mode in (Reference(), Fast())
        small = step_bytes(name, mode; ppc = 16, cells = 64)
        more_particles = step_bytes(name, mode; ppc = 256, cells = 64)
        larger_grid = step_bytes(name, mode; ppc = 16, cells = 512)
        @info "T-220 bytes per step" name mode small more_particles larger_grid threads = Threads.nthreads()
        if Threads.nthreads() == 1 || mode isa Fast
            @test more_particles <= small
            @test larger_grid <= small
        end
    end
end

@testset "T-221 step! refuses a step outside the histories it was allocated for [A7, WP-014]" begin
    st, p, prob, _ = case_state("weibel", Float64; ppc = 4, t_end = 1)
    @test_throws ArgumentError step!(st, p, prob.nsteps + 1)
    @test_throws ArgumentError step!(st, p, 0)
    step!(st, p, prob.nsteps)
end
