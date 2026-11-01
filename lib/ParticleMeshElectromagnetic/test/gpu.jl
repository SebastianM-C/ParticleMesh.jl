# WP-013, the module on CUDA: T-195 to T-198. Included last by runtests.jl, after physics.jl has
# stored the CPU runs that T-197 compares against; the testsets run only when the GPU switch says
# so (the engine's gpu_switch.jl), otherwise the suite prints one line.

"Run the shared bodies `(id, body)` on `be`, one nested testset per CPU test id."
function rerun_module_on(be, bodies)
    for (id, body) in bodies
        @testset "$id on $(backend_label(be))" begin
            body(be)
        end
    end
end

"Relative difference of two histories: max |a − b| over max |b|."
history_rel(a, b) = maximum(abs, a .- b) / maximum(abs, b)

if GPU_BACKEND === nothing
    println(gpu_skip_line(GPU_SWITCH))
else
    CUDACore.allowscalar(false)
    const GPU = GPU_BACKEND
    println("GPU tests (WP-013) on ", CUDACore.name(CUDACore.device()), ": ", GPU_SWITCH.reason)

    @testset "T-195 the module on CUDA passes its CPU tests with the CPU's tolerances [A2, A5, A7, A8, WP-013]" begin
        rerun_module_on(GPU, (("T-210", body_T210), ("T-211", body_T211), ("T-212", body_T212), ("T-214", body_T214),
                              ("T-215", body_T215), ("T-216", body_T216), ("T-213", body_T213), ("T-217", body_T217),
                              ("T-219", body_T219)))
    end

    @testset "T-196 the module's Reference step on CUDA is bitwise across reruns and launch shapes [A4, A9, WP-013]" begin
        first = determinism_digests(; be = GPU)
        @test determinism_digests(; be = GPU) == first
        @test determinism_digests(; be = GPU) == first
        for wgs in (64, 256)
            @test determinism_digests(; workgroupsize = wgs, be = GPU) == first
        end
        # the harness's solve equals initialize! plus run! bit for bit on CUDA (T-100)
        for n in (nothing, 40)
            r = harness_equals_loop(Float64; n, be = GPU)
            @test r.ok && r.t == r.n
            @test r.state && r.hist && r.u0 && r.again
        end
    end

    @testset "T-197 CPU and CUDA Reference runs differ by a measured relative amount [A4, A7, WP-013]" begin
        for name in ("landau", "two-stream", "weibel")
            haskey(PHYSICS, name) && haskey(PHYSICS, physics_key(name, GPU)) || continue
            c, g = PHYSICS[name], PHYSICS[physics_key(name, GPU)]
            rate = abs(g.f.gamma - c.f.gamma) / abs(c.f.gamma)
            W = name == "weibel" ? :WBz : :WEx
            hist = history_rel(getfield(g.h, W), getfield(c.h, W))
            total = history_rel(total_energy_history(g.h), total_energy_history(c.h))
            @info "T-197 CPU–CUDA" name rate hist total
            d = get!(MEASURED, "T-197 CPU-CUDA relative", Dict{String,Float64}())
            d["$name rate"], d["$name $W history"], d["$name total energy"] = rate, hist, total
            @test isfinite(rate) && isfinite(hist) && isfinite(total)
        end
        # cumsum_periodic! alone: both backends run one association order, so the difference is
        # the backends' arithmetic only (on one tile and on three).
        for T in (Float32, Float64), n in (64, 2 * ParticleMeshCore.SCAN_TILE + 37)
            grid = PeriodicGrid(n, T(3 // 8), 2)
            a = test_field(T, grid, j -> T(sin(0.37j) + cos(1.1j) / 5 + 3 / 10))
            oc = cumsum_periodic!(field(backend, T, grid), a, grid)
            og = on(GPU, cumsum_periodic!, field(backend, T, grid), a, grid)
            r = maximum(abs, oc .- og) / (eps(T) * n * maximum(abs, a) * cellsize(grid))
            record!("T-197 cumsum_periodic! CPU-CUDA n=$n", T, r, GPU)
            @info "T-197 cumsum_periodic! CPU–CUDA" T n bitwise = same_bits(oc, og) ratio = r
            @test isfinite(r)
        end
    end

    @testset "T-198 module allocation per step on CUDA does not grow with particle count or grid size [A7, WP-013]" begin
        for name in ("landau", "weibel"), mode in (Reference(), Fast())
            small = device_step_bytes(GPU, name, mode; ppc = 16, cells = 64)
            more_particles = device_step_bytes(GPU, name, mode; ppc = 256, cells = 64)
            larger_grid = device_step_bytes(GPU, name, mode; ppc = 16, cells = 512)
            @info "T-198 bytes per step on CUDA" name mode small more_particles larger_grid
            @test more_particles.dev <= small.dev && larger_grid.dev <= small.dev
            @test more_particles.host <= small.host + 1024 && larger_grid.host <= small.host + 1024
        end
    end
end
