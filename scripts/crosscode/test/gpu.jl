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

@testset "T-430 the smilei-gpu recipe is pinned and builds the CPU image's Smilei [A6, WP-030]" begin
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
    # every nvc++ -gpu= names the bundled toolkit (without a driver nvc++ would ask for another)
    @test occursin("-gpu=\${PM_GPU_CC},cuda\${PM_CUDA}\${PM_FASTMATH}", env)
    @test occursin("-gpu=\${PM_GPU_CC:?},cuda\${PM_CUDA:?} ", env)
    @test count("PM_CUDA=\${CUDA_RELEASE}", recipe) == 2
    @test occursin("-gpu=\${cc},cuda\${CUDA_RELEASE}", recipe) && occursin("test -s /opt/smilei/GPU_CC", recipe)
    @test occursin("-arch=sm_\${PM_SM:?}", env)
    @test !occursin("ccnative", replace(env, r"#[^\n]*" => ""))
    # the SDK's HPC-X Open MPI is not relocated: OPAL_PREFIX names its tree, and the build starts
    # MPI once as Smilei does (thread level multiple), so a broken MPI stops the build
    @test occursin("OPAL_PREFIX=/opt/nvhpc-mpi", recipe) && occursin("ln -s \"\$prefix\" /opt/nvhpc-mpi", recipe)
    @test occursin("test -f \"\$prefix/share/openmpi/help-mpi-runtime.txt\"", recipe)
    @test occursin("LD_LIBRARY_PATH=/opt/hdf5/lib:/opt/nvhpc-mpi/lib:/opt/nvhpc-ucx/lib:", recipe)
    @test occursin("MPI_THREAD_MULTIPLE", read(joinpath(CC, "containers", "smilei-gpu", "mpi-probe.c"), String))
    @test occursin("timeout 120 env \$cfg", recipe) && occursin("OMPI_MCA_ess_singleton_isolated=1", recipe)
    @test occursin("> /opt/smilei/mpi_env.sh", recipe) && occursin("test -s /opt/smilei/mpi_env.sh", recipe)
    @test !occursin("OMPI_MCA_pml=", split(recipe, "RUN NV=")[1])   # the pml is the probe's, not a fixed ENV
    # the variants get the right flags and names, the probe prefers cc89, the MPI prefix is the
    # library directory's parent, the probe and the ldd check stop the build
    @test occursin("PM_FASTMATH=\$([ \$variant = fastmath ] && echo ,fastmath || true)", recipe)
    @test occursin("name=\$([ \$variant = fastmath ] && echo smilei || echo smilei-nofastmath)", recipe)
    @test occursin("for cc in cc89 cc86; do", recipe)
    @test occursin("prefix=\$(readlink -f \"\$libdir/..\")", recipe)
    @test occursin("OMPI_MCA_mca_base_component_show_load_errors=1 /tmp/mpi-probe; then", recipe)
    @test occursin("return (size == 1 && provided == MPI_THREAD_MULTIPLE) ? 0 : 1;", read(joinpath(CC, "containers", "smilei-gpu", "mpi-probe.c"), String))
    @test occursin("if [ -n \"\$missing\" ]; then echo \"\$b: \$missing\"; exit 1; fi;", recipe)
    for f in ("nvidia_env.sh", "acc-probe.cpp", "pm-versions.sh", "mpi-probe.c")
        @test occursin("COPY smilei-gpu/$f ", recipe)
    end
end


# The lines of a namelist other than the one setting `key`.
without_key(text, key) = filter(l -> !occursin(Regex("^\\s*$key\\s*="), l), split(text, '\n'))
python_parses(text) = (f = tempname() * ".py"; write(f, text);
    success(`python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" $f`))

@testset "T-431 the GPU namelist differs from the CPU one only in the build and the layout [A5, WP-030]" begin
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
    # the rule at its edges, on cell counts where they show: 96 cells in 16 patches have 6 cells
    # each (accepted), 80 cells in 16 have 5 (refused), 96 in 3 or 6 are not powers of 2
    @test SmileiDeck.check_patches((16, 1), (96, 96)) == (16, 1)
    @test_throws ErrorException SmileiDeck.check_patches((16, 1), (80, 80))
    @test_throws ErrorException SmileiDeck.check_patches((3, 1), (96, 96))
    @test_throws ErrorException SmileiDeck.check_patches((6, 1), (96, 96))
    # a GPU run is one rank without threads whatever its layout
    @test smilei_parameters(c; gpu = true, patches = (4, 4)).threads == 1
    # schema 1 has no GPU path (Smilei's GPU port covers 2D and 3D Cartesian)
    @test_throws ErrorException namelist(load_case("landau"); gpu = true)
    @test_throws ErrorException smilei_parameters(load_case("landau"); patches = (2,))
    @test_throws ErrorException SmileiDeck.prepare_run("landau-oblique-2d"; runs = mktempdir(), fastmath = false)
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
        echo "pml=\${OMPI_MCA_pml:-unset}"
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

function run_sh(dir, stub)
    mpi = joinpath(stub, "mpi_env.sh")
    isfile(mpi) || write(mpi, "export OMPI_MCA_pml=ucx\nexport UCX_TLS=self,sm\n")
    run(setenv(`bash $(joinpath(dir, "run.sh"))`, "PATH" => stub * ":" * ENV["PATH"], "PM_RUN_DIR" => dir, "PM_MPI_ENV" => mpi))
end

@testset "T-432 run.sh keeps the generated namelist, records the build and the GPU, and resumes [A5, WP-030]" begin
    stub, calls = stub_bin()
    runs = mktempdir()
    case = load_case("landau-oblique-2d")
    seeds = case_seeds(case)
    dirs = Dict{Any,String}()
    for (gpu, fastmath, bin, layout) in ((true, true, "smilei", "1x1"), (true, false, "smilei-nofastmath", "1x1"),
                                         (false, true, "smilei", "4x4"))
        dir = SmileiDeck.prepare_run("landau-oblique-2d"; runs, gpu, fastmath)
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
                @test occursin("# mpi: OMPI_MCA_pml=ucx UCX_TLS=self,sm\n", ct)   # the image's verified MPI setting
                @test h["mpi"] == "OMPI_MCA_pml=ucx UCX_TLS=self,sm"
                @test occursin("pml=ucx", read(joinpath(sd, "smilei.log"), String))   # the image's MPI setting reached Smilei
                @test h["gpu_before"] == "0 %, 1 MiB" && h["gpu_after"] == "0 %, 1 MiB" && h["gpu_processes_before"] == ""
            else
                @test !haskey(h, "gpu") && !haskey(h, "fastmath")
            end
            @test GPUCompare.smilei_build(sd) == (gpu ? "gpu" : "cpu")
            @test GPUCompare.smilei_time_loop(sd) == 12.5
            gpu || @test occursin("pml=unset", read(joinpath(sd, "smilei.log"), String))
        end
        @test isfile(joinpath(dir, "versions.toml")) == gpu
        # the toolchain record is hashed with the outputs; the seeds' time loops and the build go
        # into the records the analysis writes
        @test haskey(GPUCompare.top_outputs(dir), "versions.toml") == gpu
        @test GPUCompare.smilei_seed_timing(dir, seeds)["time_loop_seconds_per_seed"] == fill(12.5, length(seeds))
        sd1 = joinpath(dir, "seed-$(first(seeds))")
        perf = GPUCompare.smilei_performance!(Dict{String,Any}(), sd1, 10^9)
        @test perf["time_loop_seconds"] == 12.5 && perf["ns_per_mobile_particle_step_time_loop"] == 12.5 &&
              perf["build"] == (gpu ? "gpu" : "cpu")
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

@testset "T-433 the load-check run: one step, the t = 0 corner of every mobile species [A5, WP-030]" begin
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
            @test occursin(r"\n    return \(particles\.x < [0-9.e-]+\) \* \(particles\.y < [0-9.e-]+\)\n", lc)
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
        dir = SmileiDeck.prepare_run("landau-oblique-2d"; runs, gpu, patches = layout, load_check = true)
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
        HDF5.attributes(g)["time"] = 0.0
        HDF5.attributes(g)["dt"] = 0.1
        for (name, stag) in (("Ex", [0.5, 0.0]), ("Rho", [0.0, 0.0]))
            v = [sin(i + 2j) + (name == "Rho" ? rho_shift : 0.0) for i in 0:4, j in 0:3]
            g[name] = permutedims(v)
            a = HDF5.attributes(g[name])
            a["gridSpacing"] = [0.25, 0.5]; a["gridGlobalOffset"] = [0.0, 0.0]; a["position"] = stag
            a["axisLabels"] = ["x", "y"]; a["dataOrder"] = "C"
        end
    end
    open(io -> TOML.print(io, Dict("build" => build, "patches" => [4, 4], "load_check" => true)), joinpath(dir, "manifest.toml"), "w")
end

@testset "T-434 the load check matches particles by position and sees one ulp [A4, A5, WP-030]" begin
    rng = Xoshiro(404)
    n = 200
    # a regular lattice: distinct positions, random momenta
    xs = [(i + 0.5) * 0.01 for i in 0:19 for j in 0:9]
    ys = [(j + 0.5) * 0.02 for i in 0:19 for j in 0:9]
    t = Dict("x" => xs, "y" => ys, "px" => randn(rng, n), "py" => randn(rng, n), "pz" => randn(rng, n), "w" => fill(0.25, n))
    perm = randperm(rng, n)
    shuffled = Dict(k => v[perm] for (k, v) in t)
    r = compare_tracks(t, shuffled)
    @test r["identical"] && r["same_positions"] && r["positions_match"] && r["particles"] == [n, n]
    @test all(r[k]["differing"] == 0 && r[k]["max_abs_difference"] == 0 for k in ("px", "py", "pz", "w"))
    # one ulp in one momentum is seen, and only there
    bumped = deepcopy(shuffled)
    bumped["py"][17] = nextfloat(bumped["py"][17])
    r = compare_tracks(t, bumped)
    @test !r["identical"] && r["py"]["differing"] == 1 && r["px"]["differing"] == 0 && r["pz"]["differing"] == 0
    @test r["py"]["max_abs_difference"] == eps(shuffled["py"][17])
    @test r["py"]["max_relative_to_max_abs"] == eps(shuffled["py"][17]) / maximum(abs, t["py"])
    # one ulp in one weight is seen too
    wb = deepcopy(shuffled)
    wb["w"][3] = nextfloat(wb["w"][3])
    r = compare_tracks(t, wb)
    @test !r["identical"] && r["w"]["differing"] == 1
    # an exactly representable lattice, (k + ½)/64, on which rounding x/spacing would split points
    ex = Dict("x" => [(i + 0.5) / 64 for i in 0:15 for j in 0:15], "y" => [(j + 0.5) / 64 for i in 0:15 for j in 0:15])
    for k in ("px", "py", "pz"); ex[k] = randn(rng, 256); end
    ex["w"] = fill(1.0, 256)
    @test compare_tracks(ex, ex)["identical"]
    exb = deepcopy(ex); exb["x"] = nextfloat.(exb["x"]); exb["y"] = prevfloat.(exb["y"])
    pr = randperm(rng, 256)
    r = compare_tracks(ex, Dict(k => v[pr] for (k, v) in exb))
    @test r["positions_match"] && r["positions"]["x"]["max_ulps"] == 1 && r["px"]["differing"] == 0
    # a dropped component of the momentum (pz read as py) is seen in every particle that differs
    swapped = merge(shuffled, Dict("pz" => shuffled["py"]))
    @test compare_tracks(t, swapped)["pz"]["differing"] == n
    # another count, or other positions, is not identical
    fewer = Dict(k => v[1:end-1] for (k, v) in t)
    @test !compare_tracks(t, fewer)["identical"]
    # one ulp in a position (the GPU build's host code rounds so): matched, reported, not identical
    nudged = deepcopy(t)
    nudged["x"][5] = nextfloat(nudged["x"][5])
    r = compare_tracks(t, nudged)
    @test r["positions_match"] && !r["same_positions"] && !r["identical"]
    @test r["positions"]["x"]["differing"] == 1 && r["positions"]["x"]["max_ulps"] == 1 && r["positions"]["y"]["differing"] == 0
    @test r["px"]["differing"] == 0
    # a shift below half the lattice spacing (0.01 in x) still matches, one above it does not
    moved = merge(t, Dict("x" => t["x"] .+ 1e-3))
    r = compare_tracks(t, moved)
    @test !r["same_positions"] && r["positions_match"] && r["max_abs_position_difference"] ≈ 1e-3
    @test r["match_tolerance"] ≈ 0.005
    @test !compare_tracks(t, merge(t, Dict("x" => t["x"] .+ 0.006)))["positions_match"]
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
    # two species: the run is identical only when every species is
    d2a, d2b = mktempdir(), mktempdir()
    write_loadcheck_run(d2a, t); write_loadcheck_run(d2b, t)
    write_track(joinpath(d2a, "TrackParticlesDisordered_beam.h5"), t; species = "beam")
    write_track(joinpath(d2b, "TrackParticlesDisordered_beam.h5"), bumped; species = "beam")
    r = compare_loads(d2a, d2b, out)
    @test r["species"]["electron"]["identical"] && !r["species"]["beam"]["identical"] && !r["identical"]
end

@testset "T-435 GPU runs carry their own label; paired seeds only with one layout [A5, A7, WP-030]" begin
    runs = mktempdir()
    cpu = SmileiDeck.prepare_run("landau-oblique-2d"; runs)
    g11 = SmileiDeck.prepare_run("landau-oblique-2d"; runs, gpu = true)
    g11n = SmileiDeck.prepare_run("landau-oblique-2d"; runs, gpu = true, fastmath = false)
    g44 = SmileiDeck.prepare_run("landau-oblique-2d"; runs, gpu = true, patches = (4, 4))
    @test GPUCompare.run_label(cpu) == "smilei-fp64"
    @test GPUCompare.run_label(g11) == "smilei-gpu-p1x1-fp64"
    @test GPUCompare.run_label(g11n) == "smilei-gpu-p1x1-nofastmath-fp64"
    @test GPUCompare.run_label(g44) == "smilei-gpu-p4x4-fp64"
    @test GPUCompare.run_label(SmileiDeck.prepare_run("landau-oblique-2d"; runs, patches = (4, 4))) == "smilei-fp64"   # the default layout
    @test GPUCompare.run_label(SmileiDeck.prepare_run("landau-oblique-2d"; runs, patches = (8, 2))) == "smilei-p8x2-fp64"
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
    # a sub-ensemble of the first n seeds: its manifest lists them, the analysis takes them from
    # there, and it pairs with a full run of the same layout over the seeds both ran
    case = load_case("landau-oblique-2d")
    sub = SmileiDeck.prepare_run("landau-oblique-2d"; runs, gpu = true, patches = (4, 4), first_seeds = 4)
    @test endswith(basename(sub), "-p4x4-s4-" * bytes2hex(open(sha256, case["_path"]))[1:8])
    @test TOML.parsefile(joinpath(sub, "manifest.toml"))["seeds"] == case_seeds(case)[1:4]
    @test sort(filter(startswith("seed-"), readdir(sub))) == sort(["seed-$x" for x in case_seeds(case)[1:4]])
    @test occursin("for seed in $(join(case_seeds(case)[1:4], ' ')); do", read(joinpath(sub, "run.sh"), String))
    @test GPUCompare.run_seeds(sub, TOML.parsefile(joinpath(sub, "case.toml"))) == case_seeds(case)[1:4]
    @test GPUCompare.run_seeds(cpu, TOML.parsefile(joinpath(cpu, "case.toml"))) == case_seeds(case)
    @test GPUCompare.paired_loads(cpu, sub) && !GPUCompare.paired_loads(g11, sub)
    @test_throws ErrorException SmileiDeck.prepare_run("landau-oblique-2d"; runs, gpu = true, first_seeds = 33)
    @test_throws ErrorException SmileiDeck.prepare_run("landau-oblique-2d"; runs, gpu = true, first_seeds = 0)
    all32 = SmileiDeck.prepare_run("landau-oblique-2d"; runs, gpu = true, first_seeds = 32)
    @test TOML.parsefile(joinpath(all32, "manifest.toml"))["seeds"] == case_seeds(case)
    @test_throws ErrorException SmileiDeck.prepare_run("weibel-2d"; runs, gpu = true, first_seeds = 1)   # not seeded
    # seeds the case does not have are refused, and runs that share no seed are not paired
    m2 = TOML.parsefile(joinpath(g44, "manifest.toml"))
    m2["seeds"] = [12345]
    other = joinpath(runs, "other-seeds"); cp(g44, other)
    open(io -> TOML.print(io, m2), joinpath(other, "manifest.toml"), "w")
    @test !GPUCompare.paired_loads(cpu, other)
    @test_throws ErrorException GPUCompare.run_seeds(other, TOML.parsefile(joinpath(other, "case.toml")))
    # the paired statistics: differences b − a over the seeds both measured
    st = GPUCompare.paired_stats([1.0, 2.0, 3.0, nothing], [1.5, 2.5, 2.0, 4.0], 2.0)
    @test st["n"] == 3 && st["differences"] == [0.5, 0.5, -1.0]
    @test st["mean"] == 0.0 && st["std"] ≈ sqrt(0.75) && st["stderr_of_mean"] ≈ 0.5
    @test st["max_abs"] == 1.0 && st["mean_in_units_of_reference_seed_spread"] == 0.0
    @test GPUCompare.paired_stats([nothing], [1.0], 1.0)["n"] == 0
    st = GPUCompare.paired_stats([1.0, 2.0], [1.5, 2.5], 2.0)
    @test st["mean"] == 0.5 && st["mean_in_units_of_reference_seed_spread"] == 0.25
    # pairing over the common seeds, whatever their order and whichever run has more
    seeds_, xa, xb = GPUCompare.paired_values([5, 7, 9, 11], [50, 70, 90, 110], [9, 5], [91, 51])
    @test seeds_ == [5, 9] && xa == [50, 90] && xb == [51, 91]
    # the drift of a series against the reference
    sd = GPUCompare.series_drift([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 4.0, 8.0], [1.0, 2.0 + 2e-8, 4.0, 8.0 + 8e-5])
    @test sd["max_abs_difference_relative_to_max"] ≈ 1e-5 && sd["at_t"] == 3.0
    @test sd["max_relative_difference"] ≈ 1e-5 && sd["first_t_relative_above_1e-10"] == 1.0
    @test sd["first_t_relative_above_1e-6"] == 3.0 && sd["first_t_relative_above_1e-2"] === nothing
    # a series that is exactly 0 where the other holds roundoff: the Inf ratio is not a drift
    sz = GPUCompare.series_drift([0.0, 1.0], [0.0, 1.0], [1e-30, 1.0])
    @test sz["first_t_relative_above_1e-2"] === nothing && sz["max_relative_difference"] == 0.0
    # the paired drift of the mode coefficients, from both runs' seed-coefficients.json
    ca, cb = mktempdir(), mktempdir()
    t = collect(0.0:0.5:2.0)
    c1 = [1.0 + 0im, 2.0 + 0im, 0.5im, -1.0 + 0im, 0.25 + 0im]
    c2 = [2.0 + 0im, 4.0 + 0im, 1.0im, -2.0 + 0im, 0.5 + 0im]
    wr(d, cs) = open(io -> JSON.print(io, Dict("t" => t, "seeds" => [7, 9], "re" => [real.(c) for c in cs], "im" => [imag.(c) for c in cs])),
                     joinpath(d, "seed-coefficients.json"), "w")
    wr(ca, [c1, c2])
    d1 = copy(c1); d1[3] += 1e-7im     # 1e-7 / max|c1| = 5e-8 at t = 1.0
    d2 = copy(c2); d2[5] += 4e-11      # 4e-11 / 4 = 1e-11 at t = 2.0, below 1e-10
    wr(cb, [d1, d2])
    pd = GPUCompare.paired_coefficient_difference(ca, cb, [7, 9])
    @test pd["per_seed"]["7"]["max_relative_to_max"] ≈ 5e-8 && pd["per_seed"]["7"]["at_t"] == 1.0
    @test pd["per_seed"]["7"]["first_t_above_1e-10"] == 1.0 && pd["per_seed"]["7"]["first_t_above_1e-6"] === nothing
    @test pd["per_seed"]["9"]["max_relative_to_max"] ≈ 1e-11 rtol = 1e-4
    @test pd["per_seed"]["9"]["first_t_above_1e-10"] === nothing
    @test pd["max_over_seeds"] ≈ 5e-8
end

module GPUThroughput
include(joinpath(Main.CC, "throughput.jl"))
end

@testset "T-436 time per mobile particle-step from the run records [A5, WP-030]" begin
    stub, _ = stub_bin()
    runs = mktempdir()
    case = load_case("landau-oblique-2d")
    # the mobile particle-steps by hand: 128 × 128 cells × 32 × 32 electrons, round(25/0.0073733) steps
    steps = round(Int, case["time"]["t_end"] / case["time"]["dt"])
    ps = 128 * 128 * 32 * 32 * steps
    sub = SmileiDeck.prepare_run("landau-oblique-2d"; runs, gpu = true, first_seeds = 3)
    run_sh(sub, stub)
    # the stub's time loop is 12.5 s; give the seeds known wall times
    walls = [100.0, 150.0, 120.0]
    for (s, w) in zip(case_seeds(case)[1:3], walls)
        f = joinpath(sub, "seed-$s", "container.txt")
        lines = filter(l -> !startswith(l, "# start:") && !startswith(l, "# end:"), readlines(f))
        write(f, join(lines, "\n") * "\n# start: 2026-11-03T10:00:00.000000000Z\n# end: 2026-11-03T10:" *
                 lpad(string(Int(w) ÷ 60), 2, '0') * ":" * lpad(string(Int(w) % 60), 2, '0') * ".000000000Z\n")
    end
    r = GPUThroughput.run_throughput(sub)
    @test r["mobile_particle_steps_per_run"] == ps
    @test r["wall_ns_per_mobile_particle_step"]["median"] ≈ 1e9 * 120 / ps
    @test r["wall_ns_per_mobile_particle_step"]["min"] ≈ 1e9 * 100 / ps && r["wall_ns_per_mobile_particle_step"]["max"] ≈ 1e9 * 150 / ps
    @test r["wall_ns_per_mobile_particle_step"]["n"] == 3
    @test r["time_loop_ns_per_mobile_particle_step"]["median"] ≈ 1e9 * 12.5 / ps
    @test r["build"] == "gpu"
    @test r["code"] == "smilei"
    @test r["hardware"]["gpu"] == "NVIDIA GeForce RTX 4080 SUPER, 595.104.02"
    # Smilei's own step count, when its log has it, sets the particle-steps
    for s in case_seeds(case)[1:3]
        open(joinpath(sub, "seed-$s", "smilei.log"), "a") do io
            println(io, " Time-Loop started: number of time-steps n_time = 3390")
        end
    end
    @test GPUThroughput.run_throughput(sub)["mobile_particle_steps_per_run"] == 128 * 128 * 32 * 32 * 3390
    # a run without its wall time is refused
    f = joinpath(sub, "seed-$(case_seeds(case)[2])", "container.txt")
    write(f, replace(read(f, String), r"# end: [^\n]*\n" => ""))
    @test_throws ErrorException GPUThroughput.run_throughput(sub)
end
