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
end
