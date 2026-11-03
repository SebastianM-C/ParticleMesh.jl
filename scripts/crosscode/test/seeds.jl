# Tests of the oblique seed extension (WP-025). Included by runtests.jl after picongpu.jl, which
# defines CC and loads SmileiDeck and PIConGPUDeck.

const REPORTS = normpath(joinpath(CC, "..", "..", "reports"))

@testset "T-390 the 32-seed oblique case extends WP-017's four-seed case by seeds alone [A5, A7, WP-025]" begin
    c = load_case("landau-oblique-2d")
    old = TOML.parsefile(joinpath(REPORTS, "crosscode-picongpu", "smilei-landau-oblique-2d-bc8fd6af", "case.toml"))
    seeds, oldseeds = case_seeds(c), collect(Int, old["numerics"]["seeds"])
    @test oldseeds == [20261031, 20262031, 20263031, 20264031]
    @test length(seeds) == 32 && allunique(seeds)
    @test seeds[1:4] == oldseeds                       # WP-017's seeds stay in the ensemble, first
    @test minimum(diff(sort(seeds))) > prod(SmileiDeck.smilei_parameters(c).number_of_patches)
    # apart from the seed list, the case is the one WP-017's runs used: grid, time, numerics,
    # species, diagnostics and every observable, the ensemble rule included
    strip_seeds(x) = (y = delete!(copy(x), "_path"); y["numerics"] = delete!(copy(y["numerics"]), "seeds"); y)
    @test strip_seeds(c) == strip_seeds(old)
    @test c["observables"]["damping_rate_mode_ensemble"] == old["observables"]["damping_rate_mode_ensemble"]
    # the generators run every seed: one namelist per seed in Smilei, one entry per seed in PIConGPU
    p = PIConGPUDeck.picongpu_parameters(c)
    @test p.seeds == seeds
    @test count(s -> occursin("random_seed = $s,", namelist(c; seed = s)), seeds) == 32
end

# A fake command for the run scripts: records its call in calls.txt and sleeps `secs` seconds.
function fake_command(bin, name, body)
    f = joinpath(bin, name)
    write(f, "#!/usr/bin/env bash\n" * body * "\n")
    chmod(f, 0o755)
end

@testset "T-391 a seeded run.sh skips finished seeds and starts none after its budget [A5, WP-025]" begin
    seeds = [11, 22, 33]
    runjob(dir, bin, env...) = read(addenv(`bash $(joinpath(dir, "run.sh"))`, "PM_RUN_DIR" => dir,
        "PATH" => bin * ":" * ENV["PATH"], env...), String)
    # Smilei: seed 11 finished in an earlier job; budget 2 s, each run 3 s
    mktempdir() do dir
        bin = mkpath(joinpath(dir, "bin"))
        fake_command(bin, "smilei", "echo \"\$(basename \"\$PWD\")\" >> \"\$PM_RUN_DIR/calls.txt\"; sleep 3")
        for s in seeds
            mkpath(joinpath(dir, "seed-$s"))
        end
        write(joinpath(dir, "seed-11", "container.txt"), "# start: x\n# end: y\n")
        p = SmileiDeck.smilei_parameters(load_case("landau-oblique-2d"))
        write(joinpath(dir, "run.sh"), SmileiDeck.run_script(p; seeds, budget = 2))
        out1 = runjob(dir, bin)
        @test readlines(joinpath(dir, "calls.txt")) == ["seed-22"]
        @test occursin("# seeds remaining after the budget: 33", out1)
        @test count(l -> startswith(l, "# end:"), readlines(joinpath(dir, "seed-22", "container.txt"))) == 1
        out2 = runjob(dir, bin)                         # the next job runs what is left
        @test readlines(joinpath(dir, "calls.txt")) == ["seed-22", "seed-33"]
        @test occursin("# all seeds done", out2)
        runjob(dir, bin)                                # and a third one runs nothing
        @test length(readlines(joinpath(dir, "calls.txt"))) == 2
        # without a budget every pending seed runs in one job
        write(joinpath(dir, "run.sh"), SmileiDeck.run_script(p; seeds))
        rm(joinpath(dir, "seed-33", "container.txt"))
        rm(joinpath(dir, "seed-22", "container.txt"))
        runjob(dir, bin)
        @test readlines(joinpath(dir, "calls.txt")) == ["seed-22", "seed-33", "seed-22", "seed-33"]
    end
    # PIConGPU: one compile per job that has work, the end lines in the top container.txt
    mktempdir() do dir
        bin = mkpath(joinpath(dir, "bin"))
        fake_command(bin, "pm-versions", "echo 'fake = true'")
        fake_command(bin, "pic-create", "mkdir -p \"\$2/include/picongpu/param\"")
        fake_command(bin, "pic-build", "mkdir -p bin; echo built >> \"\$PM_RUN_DIR/calls.txt\"")
        fake_command(bin, "mpiexec", "echo \"seed \$PIC_SEED in \$(basename \"\$PWD\")\" >> \"\$PM_RUN_DIR/calls.txt\"; sleep 3")
        mkpath(joinpath(dir, "params"))
        write(joinpath(dir, "params", "x.param"), "")
        for s in seeds
            mkpath(joinpath(dir, "seed-$s"))
        end
        write(joinpath(dir, "container.txt"), "# run seed-11 seed 11 start: x\n# run seed-11 seed 11 end: y\n")
        p = merge(PIConGPUDeck.picongpu_parameters(load_case("landau-oblique-2d")), (; seeds))
        write(joinpath(dir, "run.sh"), PIConGPUDeck.run_script(p; budget = 2))
        env = ("PIC_EXAMPLES" => dir, "PICONGPU_TAG" => "t", "PICONGPU_COMMIT" => "c", "PIC_BACKEND" => "b")
        out1 = runjob(dir, bin, env...)
        @test readlines(joinpath(dir, "calls.txt")) == ["built", "seed 22 in seed-22"]
        @test occursin("# seeds remaining after the budget: 33", out1)
        out2 = runjob(dir, bin, env...)
        @test readlines(joinpath(dir, "calls.txt")) == ["built", "seed 22 in seed-22", "built", "seed 33 in seed-33"]
        @test occursin("# all seeds done", out2)
        out3 = runjob(dir, bin, env...)                 # nothing left: no compile
        @test length(readlines(joinpath(dir, "calls.txt"))) == 4 && occursin("# all seeds done", out3)
        lines = readlines(joinpath(dir, "container.txt"))
        for s in seeds                                  # each seed started and ended once
            @test count(==("# run seed-$s seed $s end"), [l[1:findfirst(':', l)-1] for l in lines if startswith(l, "# run ")]) == 1
        end
        @test count(l -> startswith(l, "# compile start:"), lines) == 2   # appended, not overwritten
    end
    # a killed seed (a start line, no end) runs again; a finished one after it does not; the
    # budget lists every seed it did not start, in order, and only those
    for code in (:smilei, :picongpu)
        mktempdir() do dir
            five = [11, 22, 33, 44, 55]
            bin = mkpath(joinpath(dir, "bin"))
            for s in five
                mkpath(joinpath(dir, "seed-$s"))
            end
            if code === :smilei
                fake_command(bin, "smilei", "echo \"\$(basename \"\$PWD\")\" >> \"\$PM_RUN_DIR/calls.txt\"; sleep 3")
                write(joinpath(dir, "seed-11", "container.txt"), "# start: x\n")
                write(joinpath(dir, "seed-44", "container.txt"), "# start: x\n# end: y\n")
                write(joinpath(dir, "run.sh"), SmileiDeck.run_script(SmileiDeck.smilei_parameters(load_case("landau-oblique-2d")); seeds = five, budget = 2))
                env = ()
            else
                fake_command(bin, "pm-versions", "echo 'fake = true'")
                fake_command(bin, "pic-create", "mkdir -p \"\$2/include/picongpu/param\"")
                fake_command(bin, "pic-build", "mkdir -p bin")
                fake_command(bin, "mpiexec", "echo \"seed-\$PIC_SEED\" >> \"\$PM_RUN_DIR/calls.txt\"; sleep 3")
                mkpath(joinpath(dir, "params")); write(joinpath(dir, "params", "x.param"), "")
                write(joinpath(dir, "container.txt"), "# run seed-11 seed 11 start: x\n# run seed-44 seed 44 start: x\n# run seed-44 seed 44 end: y\n")
                p = merge(PIConGPUDeck.picongpu_parameters(load_case("landau-oblique-2d")), (; seeds = five))
                write(joinpath(dir, "run.sh"), PIConGPUDeck.run_script(p; budget = 2))
                env = ("PIC_EXAMPLES" => dir, "PICONGPU_TAG" => "t", "PICONGPU_COMMIT" => "c", "PIC_BACKEND" => "b")
            end
            out = runjob(dir, bin, env...)
            @test readlines(joinpath(dir, "calls.txt")) == ["seed-11"]
            @test occursin("# seeds remaining after the budget: 22 33 55\n", out)
        end
    end
    # prepare_run passes the budget into run.sh in both codes, and leaves it empty without one
    mktempdir() do runs
        for (prep, name) in ((SmileiDeck.prepare_run, "s"), (PIConGPUDeck.prepare_run, "p"))
            @test occursin("\nbudget=5400\n", read(joinpath(prep("landau-oblique-2d"; tag = name * "1", runs, budget = 5400), "run.sh"), String))
            @test occursin("\nbudget=\n", read(joinpath(prep("landau-oblique-2d"; tag = name * "2", runs), "run.sh"), String))
        end
    end
    # the command line: flags anywhere, positional arguments in order
    @test SmileiDeck.cli_options(["case", "--budget", "60", "tag"], ("--budget",)) == (["case", "tag"], Dict("--budget" => "60"))
    @test SmileiDeck.cli_options(["--precision", "Float32", "case", "--budget", "7"], ("--precision", "--budget")) ==
          (["case"], Dict("--precision" => "Float32", "--budget" => "7"))
    @test_throws ErrorException SmileiDeck.cli_options(["case", "--budget"], ("--budget",))
end

include(joinpath(CC, "subensembles.jl"))
include(joinpath(CC, "landau_relativistic.jl"))
@isdefined(Theory) || include(joinpath(CC, "theory.jl"))
using .Theory: landau_root, landau_rate_relativistic, thermal_load
using JSON
using Random: Xoshiro

@testset "T-392 sub-ensembles are the ensemble rule on disjoint groups of seeds [A7, WP-025]" begin
    # Synthetic seed coefficients: a damped standing wave plus independent noise per seed, on
    # the oblique case's dump grid; run b damps faster than run a.
    c = load_case("landau-oblique-2d")
    dt = c["time"]["dt"] * c["diagnostics"]["fields_every"]
    t = collect(0:dt:c["time"]["t_end"])
    rng = Xoshiro(392)
    wave(γ) = 3e-4 .* exp.(-γ .* t) .* cos.(1.4 .* t)
    noisy(γ) = [wave(γ) .+ 2e-6 .* complex.(randn(rng, length(t)), randn(rng, length(t))) for _ in 1:8]
    ca, cb = noisy(0.15), noisy(0.17)
    o = c["observables"]["damping_rate_mode_ensemble"]
    mktempdir() do root
        dirs = map(("a", "b"), (ca, cb)) do n, cs
            d = mkpath(joinpath(root, n))
            cp(c["_path"], joinpath(d, "case.toml"))
            open(io -> JSON.print(io, Dict("t" => t, "seeds" => collect(1:8), "re" => [real.(x) for x in cs],
                "im" => [imag.(x) for x in cs])), joinpath(d, "seed-coefficients.json"), "w")
            d
        end
        tt, ss, cc = seed_coefficients(dirs[1])
        @test tt == t && ss == 1:8 && cc == ca                       # the coefficients round-trip exactly
        r = subensembles(dirs..., joinpath(root, "out"))
        rule(cs) = ensemble_damping(t, cs; fit_start = o["fit_start"], floor_factor = o["floor_factor"])
        @test r["g4"]["groups"] == 2 && r["g8"]["groups"] == 1
        @test r["g4"]["a"]["fits"][2]["gamma"] == rule(ca[5:8]).gamma   # group 2 holds seeds 5 to 8
        @test r["g4"]["a"]["fits"][1]["indices"] == 1:4
        @test r["g8"]["b"]["fits"][1]["gamma"] == rule(cb).gamma
        @test r["g8"]["difference_b_minus_a"][1] == rule(cb).gamma - rule(ca).gamma
        @test r["g4"]["a"]["sd_gamma"] ≈ abs(rule(ca[1:4]).gamma - rule(ca[5:8]).gamma) / sqrt(2)
        for (cs, γ) in ((ca, 0.15), (cb, 0.17))         # the rule sees each rate within 3 jackknife errors
            e = rule(cs)
            @test abs(e.gamma - γ) < 3 * e.jackknife.stderr
        end
        # the window of all the maxima the rule took is the rule's own fit
        fm = r["first_maxima"]["a"]
        na = length(rule(ca).used)
        @test r["first_maxima"]["rule_maxima"] == [na, length(rule(cb).used)]
        @test only(x for x in fm if x["maxima"] == na)["gamma"] ≈ rule(ca).gamma rtol = 1e-12
        @test length(fm) == max(na, length(rule(cb).used)) - 2
        @test first(fm)["maxima"] == 3
        @test all(c -> c["difference_b_minus_a"] == r["first_maxima"]["b"][c["maxima"] - 2]["gamma"] - fm[c["maxima"] - 2]["gamma"],
                  r["first_maxima"]["common"])
        # the group mean of the jackknife errors is the mean over the groups
        g4 = r["g4"]["a"]
        @test g4["mean_jackknife_stderr"] ≈ mean(f["jackknife_stderr"] for f in g4["fits"]) rtol = 1e-14
        # resampling: every bootstrap sample and every leave-one-out fit counted once
        rs = r["resampling"]["a"]
        @test sum(values(rs["bootstrap_maxima_used"])) == rs["bootstrap_samples"]
        @test sum(values(rs["jackknife_maxima_used"])) == 8
        @test rs["valid_splits"] == rs["random_splits"] && rs["gamma"] == rule(ca).gamma
        @test r["resampling"]["bootstrap_difference_sd"] ≈ hypot(rs["bootstrap_gamma_sd"], r["resampling"]["b"]["bootstrap_gamma_sd"])
        @test isfile(joinpath(root, "out", "subensembles.png")) && isfile(joinpath(root, "out", "subensembles.json"))
    end
end

@testset "T-393 the fixed-window jackknife is the rule's when every leave-one-out fit takes the same maxima [A7, WP-025]" begin
    c = load_case("landau-oblique-2d")
    dt = c["time"]["dt"] * c["diagnostics"]["fields_every"]
    t = collect(0:dt:c["time"]["t_end"])
    # noise-free seeds that differ in their rate: one stop for every subset, a non-zero jackknife
    cs = [complex.(3e-4 .* exp.(-(0.15 + 0.002 * (j - 4.5)) .* t) .* cos.(1.4 .* t)) for j in 1:8]
    e = ensemble_damping(t, cs; fit_start = 3.0, floor_factor = 10.0)
    loo = [damping_from_peaks(t, ensemble_energy(cs[setdiff(1:8, j)]); fit_start = 3.0, floor_factor = 10.0).used for j in 1:8]
    @test all(==(e.used), loo)                         # the condition: one stop for all leave-one-out fits
    w = first_maxima_fits(t, cs, e.used)
    @test w[end]["maxima"] == length(e.used)
    @test w[end]["gamma"] == e.gamma
    @test w[end]["jackknife_stderr"] ≈ e.jackknife.stderr rtol = 1e-12
    @test e.jackknife.stderr > 1e-4                    # not the trivial 0 = 0
end

@testset "T-394 Landau damping with a relativistic push: the non-relativistic limit, the loads, the T scaling [A5, A7, WP-025]" begin
    T, kλ = 0.0025, 0.5
    k = kλ / sqrt(T)
    root = landau_root(kλ)
    nr = landau_rate_relativistic(k, T; relativistic = false)
    @test abs(nr.gamma + imag(root)) < 1e-4          # the time-domain solution finds the dispersion root
    @test abs(nr.omega - real(root)) < 1e-3
    # the loads: normalized, and their mean kinetic energy ⟨γ − 1⟩ (3/2 T exactly for the
    # classical energy; the series 3T/2 (1 − 5T/4 + 35T²/8) for the Maxwellian in momentum)
    ps = range(1e-9, 14 * sqrt(T); length = 200_001)
    # (for the Maxwellian in velocity, ⟨γ − 1⟩ = ⟨v²/2 + 3v⁴/8 + 5v⁶/16⟩ = 3T/2 (1 + 15T/4 + 175T²/8))
    for (kind, ke) in ((:momentum_maxwellian, 1.5T * (1 - 1.25T + 4.375T^2)), (:classical_energy, 1.5T),
                       (:velocity_maxwellian, 1.5T * (1 + 3.75T + 21.875T^2)))
        f = thermal_load(kind, T)
        w = 4π .* ps .^ 2 .* f.(ps) .* step(ps)
        @test sum(w) ≈ 1 rtol = 1e-6
        @test sum(w .* (sqrt.(1 .+ ps .^ 2) .- 1)) ≈ ke rtol = 1e-5
    end
    pc = landau_rate_relativistic(k, T; load = :momentum_maxwellian)
    sm = landau_rate_relativistic(k, T; load = :classical_energy)
    pm = landau_rate_relativistic(k, T; load = :velocity_maxwellian)
    @test pc.gamma < sm.gamma < pm.gamma < nr.gamma   # slower resonant electrons, less damping
    # a relativistic correction: the deficit scales with T at fixed kλ_D (here within 15 %)
    T4 = T / 4
    k4 = kλ / sqrt(T4)
    d(T_, k_) = landau_rate_relativistic(k_, T_; relativistic = false).gamma - landau_rate_relativistic(k_, T_).gamma
    @test d(T, k) / d(T4, k4) ≈ 4 rtol = 0.15
end

@testset "T-395 the committed 32-seed records are what their code computes from the committed coefficients [A4, A7, WP-025]" begin
    R = joinpath(REPORTS, "crosscode-picongpu")
    runs = [joinpath(R, "smilei-landau-oblique-2d-431f4192"), joinpath(R, "picongpu-landau-oblique-2d-fp64-431f4192")]
    c = load_case("landau-oblique-2d")
    o = c["observables"]["damping_rate_mode_ensemble"]
    for r in runs
        t, s, cs = seed_coefficients(r)
        @test s == case_seeds(c)                       # the case's seeds, in its order
        e = ensemble_damping(t, cs; fit_start = o["fit_start"], floor_factor = o["floor_factor"])
        ob = JSON.parsefile(joinpath(r, "observables.json"))
        @test e.gamma == ob["measured"] && e.jackknife.stderr == ob["measured_stderr"] && length(e.used) == ob["maxima_used"]
    end
    mktempdir() do out
        # the sub-ensembles and the theory comparison regenerate bit for bit
        r = subensembles(runs..., joinpath(out, "sub"))
        @test JSON.parsefile(joinpath(out, "sub", "subensembles.json")) ==
              JSON.parsefile(joinpath(R, "subensembles-landau-oblique-2d-431f4192", "subensembles.json"))
        compare_theory(runs, joinpath(out, "th"))
        @test JSON.parsefile(joinpath(out, "th", "landau-relativistic.json")) ==
              JSON.parsefile(joinpath(R, "landau-relativistic-431f4192", "landau-relativistic.json"))
        # the guards: other seeds, another case
        mkpath(joinpath(out, "b2"))
        cp(joinpath(runs[2], "case.toml"), joinpath(out, "b2", "case.toml"))
        t, s, cs = seed_coefficients(runs[2])
        write_seed_coefficients(joinpath(out, "b2"), t, reverse(s), cs; observable = "x")
        @test_throws ErrorException subensembles(runs[1], joinpath(out, "b2"), joinpath(out, "x"))
        write(joinpath(out, "b2", "case.toml"), replace(read(joinpath(runs[2], "case.toml"), String), "floor_factor = 10.0" => "floor_factor = 11.0"; count = 1))
        write_seed_coefficients(joinpath(out, "b2"), t, s, cs; observable = "x")
        @test_throws ErrorException subensembles(runs[1], joinpath(out, "b2"), joinpath(out, "x"))
        # the coefficients round-trip exactly, real and imaginary parts in place
        w = [complex.(randn(Xoshiro(395), 5), randn(Xoshiro(396), 5)) for _ in 1:3]
        write_seed_coefficients(out, collect(0.0:4.0), [1, 2, 3], w; observable = "x")
        @test seed_coefficients(out) == (collect(0.0:4.0), [1, 2, 3], w)
        # wall times from an appended container.txt: a killed start is superseded by the next one
        f = joinpath(out, "container.txt")
        write(f, """
            # compile start: 2026-11-02T10:00:00.000Z
            # compile end: 2026-11-02T10:01:40.000Z
            # run seed-1 seed 1 start: 2026-11-02T10:01:40.000Z
            # compile start: 2026-11-02T11:00:00.000Z
            # compile end: 2026-11-02T11:01:00.000Z
            # run seed-1 seed 1 start: 2026-11-02T11:01:00.000Z
            # run seed-1 seed 1 end: 2026-11-02T11:06:30.000Z
            """)
        @test stamp_interval(f, "# run seed-1 seed 1 start:", "# run seed-1 seed 1 end:") == 330.0
        @test stamp_intervals(f, "# compile start:", "# compile end:") == [100.0, 60.0]
    end
end

include(joinpath(CC, "seed_count.jl"))
include(joinpath(CC, "ensemble_report.jl"))
using .EnsembleReport: EnsembleReport

@testset "T-396 the seed-count design and the plot labels [A7, WP-025]" begin
    four = joinpath(REPORTS, "crosscode-picongpu", "smilei-landau-oblique-2d-bc8fd6af")
    # calibrate's M: the per-seed noise is the run's whatever M is; M = the run's own is the default
    a, b = calibrate(four; R = 40, M = 16), calibrate(four; R = 40)
    @test a["noise_per_seed"] == b["noise_per_seed"] && a["M"] == 16 && b["M"] == 4
    @test calibrate(four; R = 40, M = 4)["gammas"] == b["gammas"]
    @test 0.3 < a["sd_gamma"] / b["sd_gamma"] < 0.75   # about 1/2 for four times the seeds
    # difference_spread: root sum of squares per M, the jackknife scaled as √(M₀/M)
    mktempdir() do d
        ra, rb = mkpath(joinpath(d, "a")), mkpath(joinpath(d, "b"))
        for (r, jk) in ((ra, 0.006), (rb, 0.008))
            open(io -> JSON.print(io, Dict("measured_stderr" => jk, "M" => 4)), joinpath(r, "observables.json"), "w")
        end
        res = [Dict{String,Any}("run" => n, "M" => M, "sd_gamma" => sd / sqrt(M)) for (n, sd) in (("a", 3.0), ("b", 4.0)) for M in (4, 16)]
        ds = difference_spread(res, [ra, rb], [4, 16])
        @test ds["sd_difference_synthetic"] ≈ [2.5, 1.25]
        @test ds["sd_difference_measured_jackknife_scaled"] ≈ [0.01, 0.005]
        @test_throws ErrorException difference_spread(res, [ra, ra], [4, 16])
    end
    # the 2D ensemble's label names its projection; 1D keeps Ex
    @test EnsembleReport.signal_label(load_case("landau-oblique-2d")) == ("Ê", "(E·k̂) of mode [2, 1]")
    @test EnsembleReport.signal_label(load_case("landau")) == ("Ê₁", "Ex")
end
