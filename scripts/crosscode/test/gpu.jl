# Tests of the Smilei GPU build's recipe, generator, load check and analysis (WP-030). Included
# by runtests.jl, which defines CC and loads SmileiDeck, SmileiRead and Observables.
using HDF5, JSON, TOML
using SHA: sha256
using Random: Xoshiro, randperm, randn

# compare.jl and loadcheck.jl include the readers and the analysis again; each goes into a
# module of its own so that Main's bindings stay those of runtests.jl.
module GPUCompare
include(joinpath(Main.CC, "compare.jl"))
end
module GPULoadCheck
include(joinpath(Main.CC, "loadcheck.jl"))
end
using .GPULoadCheck.LoadCheck: read_track, compare_tracks, compare_loads

@testset "T-400 the smilei-gpu recipe is pinned and builds the CPU image's Smilei [A6, WP-030]" begin
    recipe = read(joinpath(CC, "containers", "smilei-gpu.Containerfile"), String)
    cpu = read(joinpath(CC, "containers", "smilei.Containerfile"), String)
    froms = [l for l in split(recipe, '\n') if startswith(strip(l), "FROM")]
    @test length(froms) == 1 && occursin(r"@sha256:[0-9a-f]{64}\b", only(froms))
    # every source fetched by git is cloned at a tag and checked against a full commit id
    for name in ("SMILEI", "HDF5")
        tag = match(Regex("ARG $(name)_TAG=(\\S+)"), recipe)
        commit = match(Regex("ARG $(name)_COMMIT=(\\S+)"), recipe)
        @test tag !== nothing && commit !== nothing && occursin(r"^[0-9a-f]{40}$", commit.captures[1])
        @test occursin("--branch \"\${$(name)_TAG}\"", recipe)
        @test occursin("rev-parse HEAD)\" = \"\${$(name)_COMMIT}\"", recipe)
    end
    @test count("git clone", recipe) == 2
    @test !occursin(r"\b(master|main|develop)\b", recipe)
    # the same Smilei release and commit as the CPU reference image
    for key in ("SMILEI_TAG", "SMILEI_COMMIT")
        @test match(Regex("ARG $key=(\\S+)"), recipe).captures == match(Regex("ARG $key=(\\S+)"), cpu).captures
    end
    # the SDK release is fixed by package version and checked against the installed compilers
    series = match(r"ARG NVHPC_SERIES=(\d+)-(\d+)\n", recipe)
    release = match(r"ARG NVHPC_RELEASE=(\d+)\.(\d+)\n", recipe)
    @test series !== nothing && release !== nothing && series.captures == release.captures
    @test occursin("\"nvhpc-\${NVHPC_SERIES}=\${NVHPC_RELEASE}\"", recipe)
    @test occursin("grep -q \"nvc++ \${NVHPC_RELEASE}\"", recipe)
    @test occursin("grep -q \"release \${CUDA_RELEASE}\"", recipe)
    # the guide's build configuration, both fastmath variants, the RTX 4080 SUPER's arch
    @test occursin("config=\"gpu_nvidia\"", recipe)
    @test occursin(r"for variant in fastmath nofastmath", recipe)
    @test match(r"ARG PM_SM=(\d+)", recipe).captures[1] == "89"
    env = read(joinpath(CC, "containers", "smilei-gpu", "nvidia_env.sh"), String)
    @test occursin("-gpu=\${PM_GPU_CC}\${PM_FASTMATH}", env)
    @test occursin("-arch=sm_\${PM_SM:?}", env)
    @test !occursin("ccnative", replace(env, r"#[^\n]*" => ""))
    for f in ("nvidia_env.sh", "acc-probe.cpp", "pm-versions.sh")
        @test occursin("COPY smilei-gpu/$f ", recipe)
    end
end


# The lines of a namelist other than the one setting `key`.
without_key(text, key) = filter(l -> !occursin(Regex("^\\s*$key\\s*="), l), split(text, '\n'))
python_parses(text) = (f = tempname() * ".py"; write(f, text);
    success(`python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" $f`))

@testset "T-401 the GPU namelist differs from the CPU one only in the build and the layout [A5, WP-030]" begin
    @test Sys.which("python3") !== nothing
    for name in ("landau-oblique-2d", "weibel-2d")
        c = load_case(name)
        cpu = namelist(c)
        gpu = namelist(c; gpu = true)
        @test !occursin("gpu_computing", cpu)
        @test count("gpu_computing", gpu) == 1 && nl_value(gpu, "gpu_computing") == "True"
        @test nl_value(gpu, "number_of_patches") == "[1, 1]"
        # apart from the gpu line, only the patch line differs; with the CPU's layout nothing does
        lc, lg = split(cpu, '\n'), without_key(gpu, "gpu_computing")
        @test length(lc) == length(lg)
        diffs = [(a, b) for (a, b) in zip(lc, lg) if a != b]
        @test length(diffs) == 1 && occursin("number_of_patches", diffs[1][1])
        pc = smilei_parameters(c).number_of_patches
        @test without_key(namelist(c; gpu = true, patches = pc), "gpu_computing") == lc
        p = smilei_parameters(c; gpu = true)
        @test p.threads == 1 && p.gpu_computing && p.number_of_patches == (1, 1)
        @test smilei_parameters(c).threads == prod(pc) && !smilei_parameters(c).gpu_computing
        @test python_parses(gpu)
    end
    # the patch rule: per axis a power of 2 dividing the cells, more than 5 cells per patch
    c = load_case("landau-oblique-2d")    # 128 × 128 cells
    @test smilei_parameters(c; patches = (16, 8)).number_of_patches == (16, 8)
    @test_throws ErrorException smilei_parameters(c; patches = (3, 1))
    @test_throws ErrorException smilei_parameters(c; patches = (32, 1))   # 4 cells per patch
    @test_throws ErrorException smilei_parameters(c; patches = (4,))
    # schema 1 has no GPU path (Smilei's GPU port covers 2D and 3D Cartesian)
    @test_throws ErrorException namelist(load_case("landau"); gpu = true)
    @test_throws ErrorException smilei_parameters(load_case("landau"); patches = (2,))
    @test_throws ErrorException prepare_run("landau-oblique-2d"; runs = mktempdir(), fastmath = false)
end

# Stand-ins for the image's programs: `smilei` (and `smilei-nofastmath`) record their call and
# write what Smilei writes into the working directory (its namelist dump smilei.py and the
# time-loop line), `nvidia-smi` answers the queries run.sh makes, `pm-versions` one line.
function stub_bin()
    stub = mktempdir()
    calls = joinpath(stub, "calls")
    for b in ("smilei", "smilei-nofastmath")
        write(joinpath(stub, b), """
        #!/usr/bin/env bash
        echo "\$(basename "\$0") \$* \$(pwd)" >> $calls
        printf '# GENERAL DEFINITIONS FOR SMILEI\\n' > smilei.py
        printf 'Time_in_time_loop\\t12.5\\t99%% coverage\\n'
        """)
    end
    write(joinpath(stub, "nvidia-smi"), """
    #!/usr/bin/env bash
    case "\$*" in
      *name,driver_version*) echo "NVIDIA GeForce RTX 4080 SUPER, 595.104.02" ;;
      *compute-apps*) ;;
      *) echo "0 %, 1 MiB" ;;
    esac
    """)
    write(joinpath(stub, "pm-versions"), "#!/usr/bin/env bash\necho 'smilei_tag = \"v5.1\"'\n")
    foreach(f -> chmod(joinpath(stub, f), 0o755), ("smilei", "smilei-nofastmath", "nvidia-smi", "pm-versions"))
    return stub, calls
end

run_sh(dir, stub) = run(setenv(`bash $(joinpath(dir, "run.sh"))`, "PATH" => stub * ":" * ENV["PATH"], "PM_RUN_DIR" => dir))

@testset "T-402 run.sh keeps the generated namelist, records the build and the GPU, and resumes [A5, F3, WP-030, #256]" begin
    stub, calls = stub_bin()
    runs = mktempdir()
    case = load_case("landau-oblique-2d")
    seeds = case_seeds(case)
    dirs = Dict{Any,String}()
    for (gpu, fastmath, bin, layout) in ((true, true, "smilei", "1x1"), (true, false, "smilei-nofastmath", "1x1"),
                                         (false, true, "smilei", "4x4"))
        dir = prepare_run("landau-oblique-2d"; runs, gpu, fastmath)
        dirs[(gpu, fastmath)] = dir
        m = TOML.parsefile(joinpath(dir, "manifest.toml"))
        @test m["build"] == (gpu ? "gpu" : "cpu") && m["patches"] == parse.(Int, split(layout, "x"))
        @test m["image"]["recipe"] == "scripts/crosscode/containers/$(gpu ? "smilei-gpu" : "smilei").Containerfile"
        run_sh(dir, stub)
        for s in seeds
            sd = joinpath(dir, "seed-$s")
            # Smilei's dump went to smilei.py; the input the manifest hashed is intact (#256)
            @test "sha256:" * bytes2hex(open(sha256, joinpath(sd, "namelist.py"))) == m["inputs"]["seed-$s/namelist.py"]
            @test startswith(read(joinpath(sd, "smilei.py"), String), "# GENERAL DEFINITIONS")
            ct = read(joinpath(sd, "container.txt"), String)
            @test occursin("# build: $(gpu ? "gpu" : "cpu")\n", ct) && occursin("# patches: $layout\n", ct)
            @test occursin("# seed: $s\n", ct) && occursin("\n# start: ", ct) && occursin("\n# end: ", ct)
            h = GPUCompare.hardware(sd)
            @test h["build"] == (gpu ? "gpu" : "cpu") && h["patches"] == layout
            @test h["threads"] == (gpu ? 1 : 16)
            if gpu
                @test h["gpu"] == "NVIDIA GeForce RTX 4080 SUPER, 595.104.02"
                @test h["fastmath"] == (fastmath ? "yes" : "no")
                @test h["gpu_before"] == "0 %, 1 MiB" && h["gpu_after"] == "0 %, 1 MiB" && h["gpu_processes_before"] == ""
            else
                @test !haskey(h, "gpu") && !haskey(h, "fastmath")
            end
            @test GPUCompare.smilei_build(sd) == (gpu ? "gpu" : "cpu")
            @test GPUCompare.smilei_time_loop(sd) == 12.5
        end
        @test isfile(joinpath(dir, "versions.toml")) == gpu
        made = readlines(calls)
        @test count(l -> startswith(l, "$bin namelist.py ") && occursin(dir, l), made) == length(seeds)
    end
    # a GPU directory run again skips the finished seeds and runs the one whose end is missing
    dir = dirs[(true, true)]
    n = countlines(calls)
    run_sh(dir, stub)
    @test countlines(calls) == n
    f = joinpath(dir, "seed-$(seeds[2])", "container.txt")
    write(f, replace(read(f, String), r"# end: [^\n]*\n" => ""))
    run_sh(dir, stub)
    @test countlines(calls) == n + 1 && endswith(readlines(calls)[end], "seed-$(seeds[2])")
    # a run without the time-loop line reports none
    sd = joinpath(dir, "seed-$(seeds[1])")
    write(joinpath(sd, "smilei.log"), "no timers\n")
    @test GPUCompare.smilei_time_loop(sd) === nothing
end

@testset "T-403 the load-check run: one step, the t = 0 corner of every mobile species [A5, WP-030]" begin
    for name in ("landau-oblique-2d", "weibel-2d")
        c = load_case(name)
        s = first(case_seeds(c))
        p = smilei_parameters(c)
        for gpu in (false, true)
            full = namelist(c; seed = s, gpu)
            lc = namelist(c; seed = s, gpu, load_check = true)
            @test python_parses(lc)
            @test nl_number(lc, "number_of_timesteps") == 1
            @test floor(Int, nl_number(lc, "simulation_time") / nl_number(lc, "timestep")) == 1
            # the case's namelist up to its diagnostics, apart from the step count and the cadences
            head(x) = split(x[1:findfirst("DiagScalar(", x).start-1], '\n')
            hf, hl = head(full), head(lc)
            @test length(hf) == length(hl)
            changed = [strip(split(a, "=")[1]) for (a, b) in zip(hf, hl) if a != b]
            @test sort(changed) == ["number_of_timesteps", "print_every", "simulation_time"]
            # every diagnostic at t = 0 only: the cadences exceed the one step
            every = [parse(Int, m[1]) for m in eachmatch(r"every = (\d+)", lc[findfirst("DiagScalar(", lc).start:end])]
            @test every[1] == 1 && all(>(1), every[2:end])
            mobile = [x.name for x in p.species if x.time_frozen == 0]
            tracked = [m[1] for m in eachmatch(r"DiagTrackParticles\(\s*species = \"([^\"]+)\"", lc)]
            @test tracked == mobile
            @test count("filter = pm_load_check_corner", lc) == length(mobile)
            corner = parse.(Float64, [m[1] for m in eachmatch(r"particles\.[xy] < ([0-9.e-]+)", lc)])
            @test corner == collect(p.grid_length ./ 8)
            @test count("attributes = [\"x\", \"y\", \"px\", \"py\", \"pz\", \"w\"]", lc) == length(mobile)
            @test (count("gpu_computing", lc) == 1) == gpu
        end
    end
    runs = mktempdir()
    c = load_case("landau-oblique-2d")
    h = bytes2hex(open(sha256, c["_path"]))[1:8]
    for (gpu, layout, tag) in ((true, nothing, "smilei-gpu-loadcheck-landau-oblique-2d-p1x1-$h"),
                               (true, (4, 4), "smilei-gpu-loadcheck-landau-oblique-2d-p4x4-$h"),
                               (false, nothing, "smilei-loadcheck-landau-oblique-2d-p4x4-$h"))
        dir = prepare_run("landau-oblique-2d"; runs, gpu, patches = layout, load_check = true)
        @test basename(dir) == tag
        @test !isdir(joinpath(dir, "seed-$(first(case_seeds(c)))"))   # one run, the first seed
        m = TOML.parsefile(joinpath(dir, "manifest.toml"))
        @test m["load_check"] == true && m["seeds"] == [first(case_seeds(c))]
        @test read(joinpath(dir, "namelist.py"), String) ==
              namelist(c; seed = first(case_seeds(c)), gpu, patches = layout, load_check = true)
        @test occursin("namelist.py > smilei.log", read(joinpath(dir, "run.sh"), String))
    end
end

# A TrackParticlesDisordered file in DiagnosticTrack.cpp's layout.
function write_track(file, t; iteration = 0, species = "electron")
    h5open(file, "w") do f
        g = create_group(f, "data/" * lpad(string(iteration), 10, '0') * "/particles/" * species)
        for (k, (grp, comp)) in (("x", ("position", "x")), ("y", ("position", "y")), ("px", ("momentum", "x")),
                                 ("py", ("momentum", "y")), ("pz", ("momentum", "z")))
            haskey(g, grp) || create_group(g, grp)
            g[grp][comp] = t[k]
        end
        g["weight"] = t["w"]
    end
end

function write_loadcheck_run(dir, t; rho_shift = 0.0, ukin = 0.5, build = "cpu")
    mkpath(dir)
    write_track(joinpath(dir, "TrackParticlesDisordered_electron.h5"), t)
    write(joinpath(dir, "scalars.txt"), "# 1 time\n# 2 Utot\n# 3 Ukin\n#\n# time Utot Ukin\n0.0 1.0 $ukin\n0.1 1.0 0.4\n")
    h5open(joinpath(dir, "Fields0.h5"), "w") do f
        g = create_group(f, "data/0000000000")
        attributes(g)["time"] = 0.0
        attributes(g)["dt"] = 0.1
        for (name, stag) in (("Ex", [0.5, 0.0]), ("Rho", [0.0, 0.0]))
            v = [sin(i + 2j) + (name == "Rho" ? rho_shift : 0.0) for i in 0:4, j in 0:3]
            g[name] = permutedims(v)
            a = attributes(g[name])
            a["gridSpacing"] = [0.25, 0.5]; a["gridGlobalOffset"] = [0.0, 0.0]; a["position"] = stag
            a["axisLabels"] = ["x", "y"]; a["dataOrder"] = "C"
        end
    end
    open(io -> TOML.print(io, Dict("build" => build, "patches" => [4, 4], "load_check" => true)), joinpath(dir, "manifest.toml"), "w")
end

@testset "T-404 the load check matches particles by position and sees one ulp [A4, A5, WP-030]" begin
    rng = Xoshiro(404)
    n = 200
    # a regular lattice: distinct positions, random momenta
    xs = [(i + 0.5) * 0.01 for i in 0:19 for j in 0:9]
    ys = [(j + 0.5) * 0.02 for i in 0:19 for j in 0:9]
    t = Dict("x" => xs, "y" => ys, "px" => randn(rng, n), "py" => randn(rng, n), "pz" => randn(rng, n), "w" => fill(0.25, n))
    perm = randperm(rng, n)
    shuffled = Dict(k => v[perm] for (k, v) in t)
    r = compare_tracks(t, shuffled)
    @test r["identical"] && r["same_positions"] && r["particles"] == [n, n]
    @test all(r[k]["differing"] == 0 && r[k]["max_abs_difference"] == 0 for k in ("px", "py", "pz", "w"))
    # one ulp in one momentum is seen, and only there
    bumped = deepcopy(shuffled)
    bumped["py"][17] = nextfloat(bumped["py"][17])
    r = compare_tracks(t, bumped)
    @test !r["identical"] && r["py"]["differing"] == 1 && r["px"]["differing"] == 0 && r["pz"]["differing"] == 0
    @test r["py"]["max_abs_difference"] == eps(shuffled["py"][17])
    # a dropped component of the momentum (pz read as py) is seen in every particle that differs
    swapped = merge(shuffled, Dict("pz" => shuffled["py"]))
    @test compare_tracks(t, swapped)["pz"]["differing"] == n
    # another count, or other positions, is not identical
    fewer = Dict(k => v[1:end-1] for (k, v) in t)
    @test !compare_tracks(t, fewer)["identical"]
    moved = merge(t, Dict("x" => t["x"] .+ 1e-3))
    r = compare_tracks(t, moved)
    @test !r["same_positions"] && r["max_abs_position_difference"] ≈ 1e-3
    @test_throws ErrorException compare_tracks(merge(t, Dict("x" => fill(0.5, n), "y" => fill(0.5, n))), t)
    # the reader returns what was written; compare_loads on two run directories
    a, b, c = mktempdir(), mktempdir(), mktempdir()
    write_loadcheck_run(a, t)
    write_loadcheck_run(b, shuffled; build = "gpu")
    write_loadcheck_run(c, bumped; rho_shift = 1e-12, ukin = nextfloat(0.5))
    rt = read_track(joinpath(a, "TrackParticlesDisordered_electron.h5"))
    @test all(rt[k] == t[k] for k in keys(t))
    out = mktempdir()
    r = compare_loads(a, b, out)
    @test r["identical"] && r["a"]["build"] == "cpu" && r["b"]["build"] == "gpu"
    @test r["fields_t0"]["Rho"]["bitwise_equal"] && r["fields_t0"]["Ex"]["max_abs_difference"] == 0
    @test r["scalars_first_output"]["Ukin"]["difference"] == 0
    @test isfile(joinpath(out, "loadcheck.json"))
    r = compare_loads(a, c, out)
    @test !r["identical"] && r["species"]["electron"]["py"]["differing"] == 1
    @test !r["fields_t0"]["Rho"]["bitwise_equal"]
    @test r["fields_t0"]["Rho"]["max_abs_difference"] ≈ 1e-12 rtol = 1e-3
    @test r["fields_t0"]["Ex"]["bitwise_equal"]
    @test r["scalars_first_output"]["Ukin"]["difference"] == eps(0.5)
end

@testset "T-405 GPU runs carry their own label; paired seeds only with one layout [A5, A7, WP-030]" begin
    runs = mktempdir()
    cpu = prepare_run("landau-oblique-2d"; runs)
    g11 = prepare_run("landau-oblique-2d"; runs, gpu = true)
    g11n = prepare_run("landau-oblique-2d"; runs, gpu = true, fastmath = false)
    g44 = prepare_run("landau-oblique-2d"; runs, gpu = true, patches = (4, 4))
    @test GPUCompare.run_label(cpu) == "smilei-fp64"
    @test GPUCompare.run_label(g11) == "smilei-gpu-p1x1-fp64"
    @test GPUCompare.run_label(g11n) == "smilei-gpu-p1x1-nofastmath-fp64"
    @test GPUCompare.run_label(g44) == "smilei-gpu-p4x4-fp64"
    @test GPUCompare.run_label(prepare_run("landau-oblique-2d"; runs, patches = (4, 4))) == "smilei-fp64"   # the default layout
    @test GPUCompare.run_label(prepare_run("landau-oblique-2d"; runs, patches = (8, 2))) == "smilei-p8x2-fp64"
    @test GPUCompare.paired_loads(cpu, g44) && GPUCompare.paired_loads(g11, g11n)
    @test !GPUCompare.paired_loads(cpu, g11) && !GPUCompare.paired_loads(g44, g11)
    # a CPU run prepared before WP-030 has no layout in its manifest: the generator's rule (4 × 4)
    old = joinpath(runs, "old-cpu")
    cp(cpu, old)
    m = TOML.parsefile(joinpath(old, "manifest.toml"))
    delete!(m, "patches"); delete!(m, "build")
    open(io -> TOML.print(io, m), joinpath(old, "manifest.toml"), "w")
    @test GPUCompare.run_label(old) == "smilei-fp64"
    @test GPUCompare.paired_loads(old, g44) && !GPUCompare.paired_loads(old, g11)
    # other seeds are not paired
    m2 = TOML.parsefile(joinpath(g44, "manifest.toml"))
    m2["seeds"] = reverse(m2["seeds"])
    other = joinpath(runs, "other-seeds"); cp(g44, other)
    open(io -> TOML.print(io, m2), joinpath(other, "manifest.toml"), "w")
    @test !GPUCompare.paired_loads(cpu, other)
    # the paired statistics: differences b − a over the seeds both measured
    st = GPUCompare.paired_stats([1.0, 2.0, 3.0, nothing], [1.5, 2.5, 2.0, 4.0], 2.0)
    @test st["n"] == 3 && st["differences"] == [0.5, 0.5, -1.0]
    @test st["mean"] == 0.0 && st["std"] ≈ sqrt(0.75) && st["stderr_of_mean"] ≈ 0.5
    @test st["max_abs"] == 1.0 && st["mean_in_units_of_reference_seed_spread"] == 0.0
    @test GPUCompare.paired_stats([nothing], [1.0], 1.0)["n"] == 0
end
