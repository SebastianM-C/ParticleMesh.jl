# WP-013, the module on CUDA: T-195 to T-197 and T-280; WP-023's 2D step: T-380, T-381; WP-029's filter: T-446; WP-028's window: T-560 to T-562; WP-047's witness bunch: T-645; WP-046's dual gradient: T-611 to T-613; WP-055's parameters: T-736; WP-056's restart: T-747; WP-057's field modifiers and one step: T-758; WP-059's field ionization: T-780; WP-062's recorder options: T-813; WP-063's level types: T-827; WP-066's shaped antenna: T-909; WP-067's compensated gather: T-969; WP-070's Galerkin gather: T-999; WP-042's 3D step: T-958; WP-072's open 3D axes: T-1019; WP-073's 3D antenna: T-1029; WP-074's 3D window: T-1039; WP-075's 3D ionization: T-1049; WP-076's 3D compensated gather: T-1059; WP-079's M4 solver: T-1098, T-1099. Included last by runtests.jl, after physics.jl has
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
                              ("T-219", body_T219), ("T-420", body_T420), ("T-522", body_T522),
                              ("T-480", body_T480), ("T-481", body_T481)))
    end

    @testset "T-380 the 2D step on CUDA passes T-360, T-361, T-364 and T-365 with the CPU's tolerances [A2, A5, A7, A8, WP-023]" begin
        rerun_module_on(GPU, (("T-360", body_T360), ("T-361", body_T361), ("T-364", body_T364), ("T-365", body_T365)))
    end

    @testset "T-381 the 2D step with current on CUDA is bitwise across reruns and launch shapes, and its rates differ from the CPU's by less than a measured bound [A4, A7, WP-023, WP-037]" begin
        first = em2d_step_digests(; be = GPU)
        @test em2d_step_digests(; be = GPU) == first
        for wgs in (64, 256)
            @test em2d_step_digests(; workgroupsize = wgs, be = GPU) == first
        end
        # CPU and CUDA: the 2D physics runs differ by a measured relative amount (as T-197)
        for name in ("landau-oblique-2d", "weibel-2d")
            # both runs must exist (T-364, T-365 and T-380 store them), or the comparison would compare nothing
            @test haskey(PHYSICS_2D, name) && haskey(PHYSICS_2D, physics_key(name, GPU))
            haskey(PHYSICS_2D, name) && haskey(PHYSICS_2D, physics_key(name, GPU)) || continue
            c, g = PHYSICS_2D[name], PHYSICS_2D[physics_key(name, GPU)]
            rate = abs(g.f.gamma - c.f.gamma) / abs(c.f.gamma)
            @info "T-381 CPU–CUDA 2D" name rate
            get!(MEASURED, "T-381 CPU-CUDA relative rate", Dict{String,Float64}())[name] = rate
            @test isfinite(rate)
            @test rate <= emC("T-381", "rate")
        end
    end

    @testset "T-958 the 3D step on CUDA passes T-950, T-951, T-953 and T-954 with the CPU's tolerances, and its Reference step is bitwise across reruns and launch shapes [A2, A4, A5, A7, A8, WP-042]" begin
        rerun_module_on(GPU, (("T-950", body_T950), ("T-951", body_T951), ("T-953", body_T953), ("T-954", body_T954)))
        first = em3d_step_digests(; be = GPU)
        @test em3d_step_digests(; be = GPU) == first
        for wgs in (64, 256)
            @test em3d_step_digests(; workgroupsize = wgs, be = GPU) == first
        end
    end

    @testset "T-1019 the open 3D step on CUDA passes T-1010 and T-1011 with the CPU's tolerances, and its Reference step is bitwise across reruns and launch shapes [A2, A4, A5, A7, A8, WP-072]" begin
        rerun_module_on(GPU, (("T-1010", body_T1010), ("T-1011", body_T1011)))
        first = open3d_step_digests(; be = GPU)
        @test open3d_step_digests(; be = GPU) == first
        for wgs in (64, 256)
            @test open3d_step_digests(; workgroupsize = wgs, be = GPU) == first
        end
    end

    @testset "T-1029 the 3D antenna on CUDA passes T-1024 and T-1026 with the CPU's tolerances, and its Reference step is bitwise across reruns and launch shapes [A2, A4, A5, A8, WP-073]" begin
        rerun_module_on(GPU, (("T-1024", body_T1024), ("T-1026", body_T1026)))
        first = antenna3d_step_digests(; be = GPU)
        @test antenna3d_step_digests(; be = GPU) == first
        for wgs in (64, 256)
            @test antenna3d_step_digests(; workgroupsize = wgs, be = GPU) == first
        end
    end

    # WP-079: the M4 solver. T-1085's plane wave (its main beams run on the CPU only), T-1087's shaped
    # beam and T-1091's bitwise Reference step and Fast–Reference agreement on CUDA, and the M4
    # window run of T-561 on both backends, differing by roundoff.
    @testset "T-1098 the M4 2D step on CUDA passes T-1085, T-1087 and T-1091 with the CPU's tolerances, and differs from the CPU's by roundoff [A2, A4, A7, A8, WP-079]" begin
        rerun_module_on(GPU, (("T-1085", body_T1085), ("T-1087", body_T1087), ("T-1091", body_T1091)))
        prob = window_lwfa(Float64; nsteps = 120)
        g, c = window_state(prob; be = GPU, solver = M4()), window_state(prob; solver = M4())
        run!(g, params(prob), prob.nsteps)
        run!(c, params(prob), prob.nsteps)
        names = (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        m = maximum(n -> maximum(abs, getfield(c.fields, n)), names)
        d = maximum(n -> maximum(abs, Array(getfield(g.fields, n)) .- getfield(c.fields, n)), names)
        @info "T-1098 CPU–CUDA M4 window fields" rel = d / m
        record!("T-1098", Float64, d / (m * eps(Float64)), GPU)
        @test d <= emC("T-561") * eps(Float64) * m
    end

    @testset "T-1099 the M4 3D step on CUDA passes T-1094, T-1096, T-1097 and T-1100 with the CPU's tolerances [A2, A4, A7, A8, WP-079]" begin
        rerun_module_on(GPU, (("T-1094", body_T1094), ("T-1096", body_T1096), ("T-1097", body_T1097), ("T-1100", body_T1100)))
    end

    @testset "T-1039 the 3D window on CUDA passes T-1030 to T-1033, T-1035, T-1036 and T-1038 with the CPU's tolerances, and its Reference step is bitwise across reruns and launch shapes [A2, A3, A4, A7, A8, WP-074]" begin
        rerun_module_on(GPU, (("T-1030", body_T1030), ("T-1031", body_T1031), ("T-1032", body_T1032), ("T-1033", body_T1033),
                              ("T-1035", body_T1035), ("T-1036", body_T1036), ("T-1038", body_T1038)))
        first = window3d_step_digests(; be = GPU)
        @test window3d_step_digests(; be = GPU) == first
        for wgs in (64, 256)
            @test window3d_step_digests(; workgroupsize = wgs, be = GPU) == first
        end
    end

    @testset "T-1049 the 3D ionization on CUDA passes T-1040 to T-1045 with the CPU's tolerances, its Reference step bitwise across reruns and launch shapes (T-1045's body) [A2, A3, A4, A7, A8, WP-075]" begin
        rerun_module_on(GPU, (("T-1040", body_T1040), ("T-1041", body_T1041), ("T-1042", body_T1042), ("T-1043", body_T1043),
                              ("T-1044", body_T1044), ("T-1045", body_T1045)))
    end

    @testset "T-1059 the 3D compensated gather on CUDA passes T-1050 to T-1054 with the CPU's tolerances, its Reference step bitwise across reruns and launch shapes (T-1054's body) [A2, A4, A7, A8, WP-076]" begin
        rerun_module_on(GPU, (("T-1050", body_T1050), ("T-1051", body_T1051), ("T-1052", body_T1052), ("T-1053", body_T1053),
                              ("T-1054", body_T1054)))
    end

    @testset "T-446 the binomial filter on CUDA passes T-440 to T-442 with the CPU's tolerances, and the filtered Reference step is bitwise across reruns and launch shapes [A2, A4, A7, WP-029]" begin
        rerun_module_on(GPU, (("T-440", body_T440), ("T-441", body_T441), ("T-442", body_T442)))
        first = filter_digests(; be = GPU)
        @test filter_digests(; be = GPU) == first
        for wgs in (64, 256)
            @test filter_digests(; workgroupsize = wgs, be = GPU) == first
        end
    end

    @testset "T-560 the moving window on CUDA passes T-470 to T-476, T-478's filter and T-479 with the CPU's tolerances [A2, A3, A4, A7, A8, WP-028]" begin
        rerun_module_on(GPU, (("T-470", body_T470), ("T-471", body_T471), ("T-472", body_T472), ("T-473", body_T473), ("T-474", body_T474),
                              ("T-475", body_T475), ("T-476", body_T476), ("T-478", body_T478), ("T-479", body_T479)))
    end

    # The same laser in the same plasma on both backends, 120 steps through about 50 shifts (Float64,
    # Reference): a backend's bits are its own (A4), so the fields differ, by roundoff.
    @testset "T-561 CPU and CUDA window runs differ by roundoff in the fields [A4, A7, WP-028]" begin
        prob = window_lwfa(Float64; nsteps = 120)
        g, c = window_state(prob; be = GPU), window_state(prob)
        run!(g, params(prob), prob.nsteps)
        run!(c, params(prob), prob.nsteps)
        names = (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        m = maximum(n -> maximum(abs, getfield(c.fields, n)), names)
        d = maximum(n -> maximum(abs, Array(getfield(g.fields, n)) .- getfield(c.fields, n)), names)
        @info "T-561 CPU–CUDA window fields" rel = d / m
        record!("T-561", Float64, d / (m * eps(Float64)), GPU)
        @test d <= emC("T-561") * eps(Float64) * m
    end

    @testset "T-562 window allocation per step on CUDA does not grow with particle count or grid size [A7, WP-028]" begin
        small = device_step_bytes_window(GPU; ppc = (2, 2), cells = (24, 6))
        more_particles = device_step_bytes_window(GPU; ppc = (6, 6), cells = (24, 6))
        larger_grid = device_step_bytes_window(GPU; ppc = (2, 2), cells = (72, 18))
        @info "T-562 bytes per three window steps on CUDA" small more_particles larger_grid
        @test more_particles.dev <= small.dev
        @test larger_grid.dev <= small.dev
        # Host bytes, three steps, as T-280: constant up to 1 KiB a step, plus 1 KiB per doubling of
        # the capacity for the merge sort's bookkeeping. Both larger cases have nine times the capacity.
        slack = 3 * 1024 * (1 + log2(9) + 1)
        @test more_particles.host <= small.host + slack
        @test larger_grid.host <= small.host + slack
    end

    @testset "T-645 the witness bunch on CUDA passes T-640 and T-641 with the CPU's tolerances [A3, A7, A8, WP-047]" begin
        rerun_module_on(GPU, (("T-640", body_T640), ("T-641", body_T641)))
    end

    # WP-046: WP-043's gradient with duals on CUDA.
    @testset "T-613 the dual gradient case on CUDA passes T-610 [A4, A8, WP-046]" begin
        rerun_module_on(GPU, (("T-610", body_T610),))
    end

    # The same Dual{Float32,3} run on both backends (Reference): a backend's bits are its own (A4),
    # and over 350 steps of particle dynamics the Float32 difference grows past roundoff, so the
    # bound is T-582's Float32-against-Float64 scale, measured and proposed (A7).
    @testset "T-611 Dual{Float32,3} partials of the WP-043 case on CUDA against the CPU's [A4, A7, A8, WP-046]" begin
        c = T582_S[] === nothing ? lwfa_wake_amplitude(lwfa_dual_x0()) : T582_S[]
        g = lwfa_wake_amplitude(lwfa_dual_x0(); be = GPU)
        @test g isa D3
        rel = maximum(abs.(components(g) .- components(c)) ./ abs.(components(c)))
        @info "T-611 CPU–CUDA dual wake amplitude" cpu = components(c) cuda = components(g) rel
        get!(MEASURED, "T-611 CPU-CUDA rel", Dict{String,Float64}())["Dual{Float32,3}"] = rel
        @test all(isfinite, components(g)) && all(!iszero, components(g))
        @test rel <= emC("T-611")
    end

    # The longer, refined cases the GPU is for: Δx, Δy and Δt divided by r = 2 and 3 (r² times the
    # cells and particles), 600·r steps (WP-044's horizon), read past the back band. The functional
    # here is the log-mean-exp, the soft maximum less log(n)/β: the soft maximum grows with the
    # number n of columns read, which refining multiplies (the PR's physics breaker). It reads the
    # window past the same physical width at every resolution, the back band of the coarse case
    # (`axis_Ex_past`), not past the band's fixed number of cells, whose width shrinks as 1/r: the
    # three runs then read the same stretch of wake, and the successive differences between
    # resolutions are the discretization error of one functional (measured on the CPU,
    # Dual{Float32,3}: density's partial −0.1876, −0.2278, −0.2373). The runs are
    # timed on both backends (r = 1 and 2), after one untimed coarse run per backend that compiles
    # every kernel, the window's included (they start at step 199).
    @testset "T-612 the refined and longer dual gradient case on CUDA converges with resolution [A7, A8, WP-046]" begin
        xd = lwfa_dual_x0()
        coarse = first(lwfa_gradient_run(xd, Reference(); nsteps = 1))
        width = window_back_band(coarse) * Float64(value_of(cellsize(coarse.grid)[1]))
        function timed(be, refine)
            nsteps = refine * LWFA_LONG_STEPS
            t = @elapsed begin
                st, _ = lwfa_gradient_run(xd, Reference(); be, refine, nsteps)
                KernelAbstractions.synchronize(be)
            end
            v = axis_Ex_past(st, width)
            @test length(v) == ncells(st.grid)[1] - refine * window_back_band(coarse)    # the coarse band's width at every r
            @test all(iszero, ParticleMeshElectromagnetic.window_lost(st))
            return (lme = soft_maximum(v, LWFA_GRADIENT_BETA) - log(length(v)) / LWFA_GRADIENT_BETA, step = t / nsteps)
        end
        m = get!(MEASURED, "T-612", Dict{String,Float64}())
        runs = Dict{Any,Any}()
        for be in (GPU, CPU())
            timed(be, 1)                                                        # compile
            for r in (be === GPU ? (1, 2, 3) : (1, 2))
                runs[(be, r)] = timed(be, r)
                m["s per step, r = $r, $(backend_label(be))"] = runs[(be, r)].step
            end
        end
        c = map(r -> components(runs[(GPU, r)].lme), (1, 2, 3))
        d12, d23 = abs.(c[2] .- c[1]), abs.(c[3] .- c[2])
        ratio = maximum(d23 ./ d12)
        monotone = all((c[2] .- c[1]) .* (c[3] .- c[2]) .> 0)    # an oscillation would pass the ratio
        cpu_rel = maximum(abs.(c[2] .- components(runs[(CPU(), 2)].lme)) ./ abs.(c[2]))
        @info "T-612 resolution convergence, CUDA" c d12 d23 ratio monotone cpu_rel times = m
        m["successive difference ratio"], m["r = 2 CPU-CUDA rel"] = ratio, cpu_rel
        @test all(x -> all(isfinite, x) && all(!iszero, x), c)
        @test monotone
        @test ratio <= emC("T-612")
        @test cpu_rel <= emC("T-611")      # the refined, longer run on CUDA is the CPU's too
    end

    @testset "T-736 the parameters handed to a state on CUDA: T-730, T-731 and T-734 with the CPU's assertions [A3, A4, A8, WP-055]" begin
        rerun_module_on(GPU, (("T-730", body_T730), ("T-731", body_T731), ("T-734", body_T734)))
    end

    @testset "T-813 the recorder's cadence and selection on CUDA: T-810, T-811 and T-814 with the CPU's assertions [A4, A8, WP-062]" begin
        rerun_module_on(GPU, (("T-810", body_T810), ("T-811", body_T811), ("T-814", body_T814)))
    end

    @testset "T-747 the restart on CUDA: T-740, T-741, T-742 and T-744 with the CPU's assertions [A4, A8, WP-056]" begin
        rerun_module_on(GPU, (("T-740", body_T740), ("T-741", body_T741), ("T-742", body_T742), ("T-744", body_T744)))
    end

    # of T-756, the part that holds the kernels against the 1D kernels of before WP-057 on the device
    # (its other part holds them against host arithmetic bit for bit, which holds on the CPU only, A4)
    @testset "T-758 the field modifiers and the one step on CUDA: T-750, T-751, T-753, T-756's kernels, T-757 and T-759 with the CPU's assertions [A4, A8, WP-057]" begin
        rerun_module_on(GPU, (("T-750", body_T750), ("T-751", body_T751), ("T-753", body_T753), ("T-756 kernels", body_T756_kernels), ("T-757", body_T757),
                              ("T-757 window", body_T757_window), ("T-759", body_T759)))
    end

    # T-771's host momenta (`push_momentum` on the host against the device's) and T-775's digests across
    # reruns and launch shapes are among the bodies; T-779's dual objectives are compared with the CPU's
    @testset "T-780 the field ionization on CUDA: T-771 to T-777, T-784 and T-785 with the CPU's assertions, and the dual objectives of T-779 against the CPU's [A2, A4, A7, A8, WP-059]" begin
        rerun_module_on(GPU, (("T-771", body_T771), ("T-772", body_T772), ("T-773", body_T773), ("T-774", body_T774), ("T-775", body_T775),
                              ("T-776", body_T776), ("T-777", body_T777), ("T-784", body_T784), ("T-785", body_T785)))
        xd = (mkdual(ION_X0[1], 1, 0, 0), mkdual(ION_X0[2], 0, 1, 0), mkdual(ION_X0[3], 0, 0, 1), mkdual(ION_X0[4], 0, 0, 0))
        c, g = ion_objectives(xd), ion_objectives(xd; be = GPU)
        rel = maximum(o -> maximum(abs.(components(g[o]) .- components(c[o])) ./ abs.(components(c[o]))), 1:2)
        @info "T-780 the dual objectives, CUDA against the CPU" cpu = c cuda = g rel
        get!(MEASURED, "T-780 CPU-CUDA rel", Dict{String,Float64}())["Dual{Float32,3}"] = rel
        @test all(o -> all(isfinite, components(g[o])) && all(!iszero, components(g[o])), 1:2)
        @test rel <= emC("T-780")
    end

    # a tuple of levels of three types in one kernel, one of them defined in the tests; the suppressed rate's partials; the zero-suppression bits
    @testset "T-827 the levels' rate laws on CUDA: T-822, T-823 and T-824 with the CPU's assertions [A2, A4, A8, WP-063]" begin
        rerun_module_on(GPU, (("T-822", body_T822), ("T-823", body_T823), ("T-824", body_T824)))
    end

    @testset "T-969 the compensated gather on CUDA: T-960, T-961 and T-962 with the CPU's assertions [A2, A4, A8, WP-067]" begin
        rerun_module_on(GPU, (("T-960", body_T960), ("T-961", body_T961), ("T-962", body_T962)))
    end

    @testset "T-999 the Galerkin gather on CUDA: T-993's kernels, T-996 and T-997 with the CPU's assertions [A2, A4, A8, WP-070]" begin
        rerun_module_on(GPU, (("T-993", body_T993), ("T-996", body_T996), ("T-997", body_T997)))
    end

    @testset "T-909 the shaped antenna on CUDA passes T-906 with the CPU's tolerances [A2, A7, A8, WP-066]" begin
        rerun_module_on(GPU, (("T-906", body_T906),))
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

    @testset "T-197 CPU and CUDA Reference runs differ by less than a measured bound [A4, A7, WP-013, WP-037]" begin
        for name in ("landau", "two-stream", "weibel")
            # both runs must exist (physics.jl and T-195 store them), or the comparison would compare nothing (#184)
            @test haskey(PHYSICS, name) && haskey(PHYSICS, physics_key(name, GPU))
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
            @test rate <= emC("T-197", "rate") && hist <= emC("T-197", "history") && total <= emC("T-197", "total")
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
            @test r <= emC("T-197", "C_cumsum")
        end
    end

    @testset "T-280 module allocation per step on CUDA does not grow with particle count or grid size [A7, WP-013]" begin
        for name in ("landau", "weibel"), mode in (Reference(), Fast())
            small = device_step_bytes(GPU, name, mode; ppc = 16, cells = 64)
            more_particles = device_step_bytes(GPU, name, mode; ppc = 256, cells = 64)
            larger_grid = device_step_bytes(GPU, name, mode; ppc = 16, cells = 512)
            @info "T-280 bytes per step on CUDA" name mode small more_particles larger_grid
            @test more_particles.dev <= small.dev && larger_grid.dev <= small.dev
            # Host bytes: constant up to 1 KiB, plus, per sorted species, AcceleratedKernels' GPU
            # merge sort's bookkeeping, one merge-pass launch per doubling of the capacity (the
            # engine's T-198 measures it on its own): at most 1 KiB per doubling.
            sortslack(b, factor) = b.sorted * 1024 * (log2(factor) + 1)
            @test more_particles.host <= small.host + 1024 + sortslack(more_particles, 16)
            @test larger_grid.host <= small.host + 1024 + sortslack(larger_grid, 8)
        end
    end
end
