# Tests of WP-090's benchmark generator, timers and readers (scripts/bench3d/bench3d.jl), run by
# scripts/lwfa/test/runtests.jl after the LWFA tests (the benchmark builds on ionization.jl).

include(joinpath(@__DIR__, "..", "bench3d.jl"))

const BENCH_CASE_FILE = joinpath(@__DIR__, "..", "cases", "gpu-3d.toml")

param_number(text, key) = parse(Float64, match(Regex("\\b$key = ([-0-9.e]+)"), text).captures[1])

@testset "T-1200 the uniform benchmark is one plasma in both codes: the box, Δt, density, lattice, temperature, element type and algorithms of ParticleMesh's problem are PIConGPU's deck [A1, A5, WP-090]" begin
    case = TOML.parsefile(BENCH_CASE_FILE)
    U = reference_units(Float64(case["lambda0_um"]) * 1e-6)
    for u in uniform_entries(case)
        @test u.dt == 0.95 / sqrt(3 / u.dx^2) && u.dx == 2π / 32
        prob = uniform_problem(u)
        T = u.float == "Float32" ? Float32 : Float64
        @test prob.cells == u.cells && prob.dt == T(u.dt) && prob.nsteps == u.steps && prob.boundaries == (:periodic, :periodic, :periodic)
        @test prob.length == ntuple(k -> T(u.cells[k] * u.dx), 3) && isempty(prob.lasers) && prob.window === nothing
        e = only(prob.species)
        @test e.ppc == u.ppc && e.density == T(u.density) && e.thermal == ntuple(_ -> T(u.vth), 3) && iszero(e.amplitude)
        @test e.charge == -one(T) && e.mass == one(T) && all(iszero, e.drift)
        f = picongpu_uniform_param_files(u)
        @test occursin("#define SIMDIM DIM3", f["dimension.param"])
        @test occursin("precisionPIConGPU = $(u.float == "Float32" ? "precision32Bit" : "precision64Bit");", f["precision.param"])
        for k in ("Sqrt", "Exp", "Trigonometric")
            @test occursin("precision$k = precisionPIConGPU;", f["precision.param"])
        end
        sim = f["simulation.param"]
        @test param_number(sim, "DELTA_T_SI") == u.dt * U.time
        @test all(param_number(sim, "CELL_$(w)_SI") == u.dx * U.length for w in ("WIDTH", "HEIGHT", "DEPTH"))
        @test param_number(sim, "BASE_DENSITY_SI") == U.density
        @test param_number(f["density.param"], "DENSITY_e") == u.density
        @test occursin("mCT::Int<$(u.ppc[1]), $(u.ppc[2]), $(u.ppc[3])>", f["particle.param"])
        @test occursin("TYPICAL_PARTICLES_PER_CELL = $(prod(u.ppc))u;", sim)
        # PIConGPU's temperature is mₑv_th² in keV: 1.28 keV at v_th = 0.05 c
        @test param_number(f["particle.param"], "temperature") ≈ u.vth^2 * 510.99895 rtol = 1e-8
        @test occursin("particles::shapes::TSC", f["species.param"]) && occursin("currentSolver::Esirkepov<UsedParticleShape>", f["species.param"])
        @test occursin("particles::pusher::Boris", f["species.param"]) && occursin("maxwellSolver::Yee", f["fieldSolver.param"])
        @test occursin("FileOutputParticles = MakeSeq_t<>", f["fileOutput.param"])
        args = picongpu_uniform_arguments(u)
        @test occursin("-d 1 1 1 ", args) && occursin("-g $(u.cells[1]) $(u.cells[2]) $(u.cells[3]) ", args) && occursin("-s $(u.steps) ", args)
        @test occursin("--periodic 1 1 1 ", args) && occursin("--currentInterpolation none", args) && !occursin("openPMD", args)
        @test occursin("--fields_energy.period $(u.steps)", args) && occursin("--e_energy.period $(u.steps)", args)
    end
    # ParticleMesh's numerics: the same algorithms, no filter
    for m in ("reference", "fast")
        n = bench_numerics(mode_named(m))
        @test n.shape === Quadratic && n.pusher isa Boris && n.solver isa Yee && n.current isa Esirkepov && n.filter == BinomialFilter(0)
        @test n.mode isa (m == "reference" ? Reference : Fast) && n.gather === nothing
    end
    bad = deepcopy(case)
    bad["uniform"][1]["cells"] = [100, 128, 128]
    @test_throws ErrorException uniform_entries(bad)               # not a multiple of PIConGPU's supercell
    # prepare: one directory per entry and mode, PIConGPU's with its params, build.sh and run.sh
    root = prepare_bench(BENCH_CASE_FILE, "t1200-$(getpid())")
    try
        names = readlines(joinpath(root, "runs.txt"))
        @test length(names) == 3 * length(case["uniform"]) + 3 * length(case["lwfa"]["kinds"])
        @test "pm-c128-p8-fast" in names && "picongpu-c256-p8" in names && "pm-lwfa-plasma-reference" in names && "picongpu-lwfa-plasma" in names
        run_sh = read(joinpath(root, "picongpu-c128-p1", "run.sh"), String)
        @test occursin("mpiexec -n 1 setup/bin/picongpu -d 1 1 1 -g 128 128 128", run_sh) && occursin("PIC_SEED=1", run_sh)
        @test occursin("pic-build", read(joinpath(root, "picongpu-c128-p1", "build.sh"), String)) && !occursin("pic-build", run_sh)
        @test TOML.parsefile(joinpath(root, "pm-c128-p8-f64-reference", "entry.toml"))["float"] == "Float64"
        @test isfile(joinpath(root, "picongpu-lwfa-plasma", "params", "precision.param"))
    finally
        rm(root; recursive = true)
    end
end

@testset "T-1201 WP-082's LWFA deck in Float32 and without outputs: precision32Bit and otherwise WP-082's files, the same grid, ranks and window [A5, WP-082, WP-090]" begin
    lcase = TOML.parsefile(joinpath(REPO, "scripts", "lwfa", "cases", "ionization-3d.toml"))
    d = ion_derived(lcase)
    for kind in ("plasma", "plasma-nodopant", "vacuum")
        f64 = picongpu_ion_param_files(lcase; kind, seed = 1)
        f32 = picongpu_ion_param_files(lcase; kind, seed = 1, precision = "Float32")
        @test occursin("precision64Bit", f64["precision.param"]) && occursin("precision32Bit", f32["precision.param"])
        @test keys(f64) == keys(f32) && all(f64[n] == f32[n] for n in keys(f64) if n != "precision.param")
    end
    p = picongpu_parameters(picongpu_ion_lab(lcase); plasma = true, precision = "Float32")
    a = picongpu_ion_arguments(lcase, p)
    b = picongpu_bench_lwfa_arguments(a, d.steps)
    @test !occursin("openPMD", b) && !occursin("energyHistogram", b) && occursin("--fields_energy.period $(d.steps)", b)
    @test count("--fields_energy.period", b) == 1
    for keep in ("-d 1 2 1", "-r 2", "-g 128 1104 128", "-s $(d.steps)", "--periodic 0 0 0", "-m", "--windowMovePoint", "--fieldAbsorber pml",
                 "--currentInterpolation none", "--autoAdjustGrid 0")
        @test occursin(keep, b)
    end
    # the words kept are WP-082's, in order
    kept = split(a)
    @test all(w -> w in kept, split(b)[1:(end - 2)])
end

@testset "T-1202 PIConGPU's log: its durations, initialization and loop times, progress points, and the steady time per step from timing points [A5, WP-090]" begin
    @test pic_duration("6sec 188msec") == 6.188
    @test pic_duration("1min  1sec 539msec") ≈ 61.539
    @test pic_duration("1h  2min  0sec 5msec") ≈ 3720.005
    log = """
    PIConGPUVerbose PHYSICS(1) | macro particles per device: 9043968
    initialization time:  3sec 474msec = 3.474 sec
      0 % =        0 | time elapsed:             2sec 521msec | avg time per step:   0msec
      4 % =       85 | time elapsed:             6sec 188msec | avg time per step:  43msec
      9 % =      170 | time elapsed:            10sec 380msec | avg time per step:  41msec
     69 % =     1190 | time elapsed:            57sec 443msec | avg time per step:  44msec
     99 % =     1700 | time elapsed:       1min 25sec 817msec | avg time per step:  52msec
    calculation  simulation time:  1min 26sec 856msec = 86.856 sec
    full simulation time:  1min 31sec  73msec = 91.073 sec
    """
    r = read_picongpu_log(log)
    @test r.init == 3.474 && r.loop == 86.856 && r.full == 91.073
    @test r.marks == [(0, 2.521), (85, 6.188), (170, 10.38), (1190, 57.443), (1700, 85.817)]
    s = steady_step(r.marks, 170)
    @test s.from == 170 && s.to == 1700 && s.seconds ≈ (85.817 - 10.38) / 1530
    @test steady_step(r.marks, 100).from == 170                     # the first point at or after the warm-up
    @test_throws ErrorException steady_step(r.marks, 1701)          # no point after it
    @test_throws ErrorException steady_step(r.marks, 1700)          # one point is no interval
    @test read_picongpu_log("no timing here").marks == Tuple{Int,Float64}[] && read_picongpu_log("").init === nothing
end

@testset "T-1203 ParticleMesh's timer on the CPU: the particle count, the timing points every block, the steady interval, and a plasma at the loaded temperature [A5, A7, WP-090]" begin
    case = TOML.parsefile(BENCH_CASE_FILE)
    case["uniform_defaults"]["steps"], case["uniform_defaults"]["warmup"], case["uniform_defaults"]["block"] = 40, 20, 10
    case["uniform"] = [Dict{String,Any}("name" => "c16-p8", "cells" => [16, 16, 16], "ppc" => [2, 2, 2])]
    mktempdir() do dir
        open(io -> TOML.print(io, case), joinpath(dir, "case.toml"), "w")
        open(io -> TOML.print(io, merge(case["uniform"][1], Dict("kind" => "uniform", "mode" => "reference"))), joinpath(dir, "entry.toml"), "w")
        out = run_pm_uniform(dir)
        @test out["particles"] == 16^3 * 8 && out["computed_cells"] == 16^3 && out["steps"] == 40
        @test [m[1] for m in out["marks"]] == [1, 10, 20, 30, 40]
        @test out["marks"][1][2] == 0 && issorted([m[2] for m in out["marks"]])
        s = out["steady"]
        @test s["from"] == 20 && s["to"] == 40
        @test s["seconds_per_step"] ≈ (out["marks"][end][2] - out["marks"][3][2]) / 20
        @test s["ns_per_particle_step"] ≈ 1e9 * s["seconds_per_step"] / out["particles"]
        # the loaded kinetic energy is the temperature's, 3/2·v_th² per unit mass and weight (θ = v_th² ≪ 1; 32768
        # particles with three components drawn, a relative sampling error of about √(2/(3·32768)) = 0.45 %)
        V = (16 * case["uniform_defaults"]["dx"])^3
        K0 = 1.5 * case["uniform_defaults"]["thermal_velocity"]^2 * case["uniform_defaults"]["density"] * V
        @test isapprox(out["energy_start"]["K"], K0; rtol = 0.03)
        @test out["energy_start"]["WE"] < 1e-12 * K0 && out["energy_start"]["WB"] == 0     # the initial solve leaves roundoff
        # 40 steps of a resolved warm plasma: the total energy holds to well under a percent, and the field takes some
        e1 = out["energy_end"]
        @test 0 < e1["WE"] < 0.05 * K0
        @test isapprox(e1["K"] + e1["WE"] + e1["WB"], out["energy_start"]["K"]; rtol = 0.01)
        @test isfile(joinpath(dir, "timing.json")) && isfile(joinpath(dir, "manifest.toml"))
        # what compare reads back from a PIConGPU directory and this one
        pic = joinpath(dir, "runs", "picongpu-c16-p8")
        mkpath(pic)
        cp(joinpath(dir, "case.toml"), joinpath(pic, "case.toml"))
        open(io -> TOML.print(io, merge(case["uniform"][1], Dict("kind" => "uniform", "ranks" => 1, "float" => "Float32"))), joinpath(pic, "entry.toml"), "w")
        write(joinpath(pic, "picongpu.log"), """
        initialization time:  1sec 0msec = 1.0 sec
          0 % =        0 | time elapsed:             0sec 100msec | avg time per step:   0msec
         50 % =       20 | time elapsed:             0sec 300msec | avg time per step:   0msec
         95 % =       38 | time elapsed:             0sec 480msec | avg time per step:   0msec
        calculation  simulation time:  0sec 500msec = 0.5 sec
        full simulation time:  1sec 600msec = 1.6 sec
        """)
        U = reference_units(0.8e-6)
        J = U.energy * U.density * U.length^3
        write(joinpath(pic, "fields_energy.dat"), "#step total Bx By Bz Ex Ey Ez\n0 0 0 0 0 0 0 0\n40 $(6J) 1 0 $(J) $(2J) $(3J) 0\n")
        write(joinpath(pic, "e_energy_all.dat"), "#step Ekin_Joule E_Joule\n0 $(K0 * J) 0\n40 $(K0 * J) 0\n")
        write(joinpath(pic, "container.txt"), "# compile start: 2026-11-28T08:00:00.000000000Z\n# compile end: 2026-11-28T08:01:10.500000000Z\n")
        r = read_picongpu_bench(pic)
        @test r["steady"]["seconds_per_step"] ≈ 0.18 / 18 && r["particles"] == 16^3 * 8 && r["compile_seconds"] == 70.5
        @test r["energy_end"]["WE"] ≈ 5 && r["energy_end"]["WB"] ≈ (1 + J) / J && r["energy_start"]["K"] ≈ K0
        cp(dir * "/timing.json", mkpath(joinpath(dir, "runs", "pm-c16-p8-reference")) * "/timing.json")
        c = compare_bench(joinpath(dir, "runs"), joinpath(dir, "cmp"))
        g = c["groups"]["uniform/c16-p8"]
        @test g["picongpu_over_pm"]["pm-reference"] ≈ r["steady"]["seconds_per_step"] / s["seconds_per_step"]
        t = bench_table(JSON.parsefile(joinpath(dir, "cmp", "comparison.json")))
        @test occursin("| c16-p8 | Float32 | 32768 | PIConGPU 0.8.0 | 10.0 |", t) && occursin("ParticleMesh, Reference", t)
        @test isfile(joinpath(dir, "cmp", "steps.png")) && isfile(joinpath(dir, "cmp", "steps.json"))
    end
end
