# Tests of the cross-code tooling: julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl
using Test
using HDF5
using TOML

const CC = dirname(@__DIR__)
include(joinpath(CC, "smilei.jl"))
include(joinpath(CC, "read.jl"))
include(joinpath(CC, "theory.jl"))
include(joinpath(CC, "analysis.jl"))
using .SmileiDeck, .SmileiRead, .Theory, .Observables

# The value of `key = <value>` in a generated namelist (first occurrence after `from`).
function nl_value(text, key; from = 1)
    m = match(Regex("\\b$key\\s*=\\s*([^\\n]*?),?\\s*\\n"), text, from)
    m === nothing && error("$key not in namelist")
    return m.captures[1]
end
nl_number(text, key; from = 1) = parse(Float64, strip(nl_value(text, key; from), ['[', ']', ' ']))

@testset "T-010 unit conversion to Smilei [A1, WP-003]" begin
    # Hand-computed from cases/landau.toml: L = 2π/10, 64 cells, so Δx = π/320; dt as given;
    # steps = round(25 / 0.0093266) = round(2680.505) = 2681; v_th = 0.05 c gives
    # T = 0.05² = 0.0025 m_e c²; the 1 % perturbation of n = 1 n_c is the absolute amplitude
    # 0.01 over one wavelength; 64 cells give 8 patches of 8 cells.
    landau = load_case("landau")
    p = smilei_parameters(landau)
    @test p.cell_length ≈ π / 320 rtol = 1e-15
    @test p.number_of_cells == 64
    @test p.grid_length ≈ 2π / 10 rtol = 1e-15
    @test p.timestep == 0.0093266
    @test p.number_of_timesteps == 2681
    @test p.number_of_patches == 8
    e, i = p.species
    @test e.density == 1.0 && e.charge == -1.0 && e.mass == 1.0
    @test e.temperature ≈ 0.0025 rtol = 1e-15
    @test e.momentum == "maxwell-juettner"
    @test e.cosine.base == 1.0
    @test e.cosine.xamplitude ≈ 0.01 rtol = 1e-15
    @test e.cosine.xnumber == 1
    @test e.cosine.xlength ≈ 2π / 10 rtol = 1e-15
    @test e.time_frozen == 0.0
    @test i.momentum == "cold" && i.cosine === nothing && i.mass == 1836.0
    @test i.time_frozen > p.simulation_time

    # Hand-computed from cases/two-stream.toml: Δx = 1.0260398641294912/128 = 0.008015936438511650;
    # steps = round(40 / 0.0076151) = round(5252.72) = 5253; each beam 0.5 n_c with 0.1 %
    # perturbation: absolute amplitude 0.0005; drifts ∓0.1 c; 128 cells give 16 patches.
    ts = load_case("two-stream")
    q = smilei_parameters(ts)
    @test q.cell_length ≈ 0.00801593643851165 rtol = 1e-14
    @test q.number_of_timesteps == 5253
    @test q.number_of_patches == 16
    b1, b2, ion = q.species
    @test b1.density == 0.5
    @test b1.cosine.xamplitude ≈ 0.0005 rtol = 1e-15
    @test b1.mean_velocity == -0.1 && b2.mean_velocity == 0.1
    @test b1.momentum == "cold" && b1.temperature == 0.0

    # A density and a mass other than 1 exercise the scalings: n = 0.25 n_c with α = 0.1
    # gives the absolute amplitude 0.025; m = 4 m_e at v_th = 0.01 c gives T = 4 × 1e-4 m_e c².
    synthetic = deepcopy(landau)
    s = synthetic["species"][1]
    s["density"] = 0.25
    s["mass"] = 4.0
    s["thermal_velocity"] = 0.01
    s["perturbation"] = Dict("amplitude" => 0.1, "mode" => 3)
    r = smilei_parameters(synthetic).species[1]
    @test r.cosine.xamplitude ≈ 0.025 rtol = 1e-15
    @test r.cosine.xnumber == 3
    @test r.temperature ≈ 4e-4 rtol = 1e-14

    # The namelist carries the same numbers.
    text = namelist(landau)
    @test nl_number(text, "cell_length") ≈ π / 320 rtol = 1e-15
    @test nl_number(text, "timestep") == 0.0093266
    @test nl_number(text, "number_of_timesteps") == 2681
    m = match(r"cosine\(([^,]+), xamplitude=([^,]+), xvacuum=0\., xlength=([^,]+), xphi=0\., xnumber=(\d+)\)", text)
    @test m !== nothing
    @test parse(Float64, m[1]) == 1.0
    @test parse(Float64, m[2]) ≈ 0.01 rtol = 1e-15
    @test parse(Float64, m[3]) ≈ 2π / 10 rtol = 1e-15
    @test parse(Int, m[4]) == 1
    @test nl_number(text, "temperature") ≈ 0.0025 rtol = 1e-15
    @test occursin("geometry = \"1Dcartesian\"", text)
    @test occursin("interpolation_order = 2", text)
    @test count("Species(", text) == 2
end

@testset "T-012 Smilei reader on a synthetic file [WP-003]" begin
    dir = mktempdir()
    # scalars.txt as DiagnosticScalar::write formats it (precision 10).
    w(x) = lpad(string(round(x; sigdigits = 11)), 20)
    open(joinpath(dir, "scalars.txt"), "w") do io
        println(io, "# 1 time\n# 2 Utot\n# 3 Uelm\n# 4 Uelm_Ex\n#")
        println(io, "#", lpad("time", 19), lpad("Utot", 20), lpad("Uelm", 20), lpad("Uelm_Ex", 20))
        for (t, a, b, c) in ((0.0, 1.5, 2.5e-7, 2.5e-7), (0.04, 1.5000001, 2.4e-7, 2.4e-7))
            println(io, w(t), w(a), w(b), w(c))
        end
    end
    sc = read_scalars(dir)
    @test sc.time == [0.0, 0.04]
    @test sc.columns["Utot"] == [1.5, 1.5000001]
    @test sc.columns["Uelm_Ex"] == [2.5e-7, 2.4e-7]
    @test Set(keys(sc.columns)) == Set(["time", "Utot", "Uelm", "Uelm_Ex"])

    # Fields0.h5 in the layout of DiagnosticFields: /data/<10-digit iteration>/<field>, with the
    # openPMD attributes time and dt on the iteration and gridSpacing, gridGlobalOffset and
    # position on the dataset; 4 cells give 5 points.
    dx, dt = 0.25, 0.2
    ex(it) = [sin(2π * (j + 0.5) / 4) * it for j in 0:4]
    rho(it) = [1.0 + 0.1 * j + it for j in 0:4]
    h5open(joinpath(dir, "Fields0.h5"), "w") do f
        attributes(f)["name"] = "DiagFields"
        for it in (0, 10, 20)
            g = create_group(f, "data/" * lpad(string(it), 10, '0'))
            attributes(g)["time"] = it * dt
            attributes(g)["dt"] = dt
            for (name, v, s) in (("Ex", ex(it), 0.5), ("Rho", rho(it), 0.0))
                g[name] = v
                a = attributes(g[name])
                a["gridSpacing"] = [dx]
                a["gridGlobalOffset"] = [0.0]
                a["position"] = [s]
            end
        end
    end
    fl = read_fields(dir)
    @test fl.iterations == [0, 10, 20]
    @test fl.time ≈ [0.0, 2.0, 4.0]
    @test sort(collect(keys(fl.fields))) == ["Ex", "Rho"]
    @test size(fl.fields["Ex"]) == (5, 3)
    @test fl.fields["Ex"][:, 2] == ex(10)
    @test fl.fields["Rho"][:, 3] == rho(20)
    @test fl.x["Rho"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    @test fl.x["Ex"] == [0.125, 0.375, 0.625, 0.875, 1.125]
    only_ex = read_fields(dir; names = ["Ex"])
    @test collect(keys(only_ex.fields)) == ["Ex"]
    # the mode-1 amplitude of the synthetic Ex at iteration 10 is 10
    @test mode_amplitude(fl.fields["Ex"], 1)[2] ≈ 10 rtol = 1e-12
end

@testset "T-013 container recipe is pinned [WP-003]" begin
    recipe = read(joinpath(CC, "containers", "smilei.Containerfile"), String)
    froms = [l for l in split(recipe, '\n') if startswith(strip(l), "FROM")]
    @test !isempty(froms)
    @test all(l -> occursin(r"@sha256:[0-9a-f]{64}\b", l), froms)
    tag = match(r"ARG SMILEI_TAG=(\S+)", recipe)
    commit = match(r"ARG SMILEI_COMMIT=(\S+)", recipe)
    @test tag !== nothing && occursin(r"^v\d+(\.\d+)*$", tag.captures[1])
    @test commit !== nothing && occursin(r"^[0-9a-f]{40}$", commit.captures[1])
    # fetched at the tag and checked against the commit; no branch names
    @test occursin("--branch \"\${SMILEI_TAG}\"", recipe)
    @test occursin("rev-parse HEAD)\" = \"\${SMILEI_COMMIT}\"", recipe)
    @test !occursin(r"\b(master|main|develop)\b", recipe)
end

@testset "T-160 linear-theory values of the 1D cases [A5, WP-003]" begin
    landau = load_case("landau")
    obs = landau["observables"]["damping_rate"]
    e = landau["species"][1]
    kλ = (2π / landau["grid"]["length"]) * e["thermal_velocity"] / sqrt(e["density"])
    @test kλ ≈ 0.5 rtol = 1e-14
    ω = landau_root(kλ)
    @test abs(Theory.landau_dispersion(ω, kλ)) < 1e-12
    @test -imag(ω) ≈ obs["theory"] rtol = 1e-12
    @test real(ω) ≈ obs["theory_frequency"] rtol = 1e-12
    # toward small k: close to the Bohm–Gross frequency √(1 + 3k²λ²), weakly damped
    ω_small = landau_root(0.2)
    @test abs(Theory.landau_dispersion(ω_small, 0.2)) < 1e-10
    @test real(ω_small) ≈ sqrt(1 + 3 * 0.2^2) rtol = 1e-2
    @test 0 < -imag(ω_small) < 1e-3 * -imag(ω)

    ts = load_case("two-stream")
    obs = ts["observables"]["growth_rate"]
    b = ts["species"][2]
    v0 = b["drift"]
    kv0 = (2π / ts["grid"]["length"]) * v0 / sqrt(2b["density"])
    @test kv0 ≈ sqrt(3 / 8) rtol = 1e-14
    @test two_stream_cold_growth(kv0) ≈ obs["theory_nonrelativistic"] rtol = 1e-12
    @test obs["theory_nonrelativistic"] ≈ 1 / (2sqrt(2)) rtol = 1e-12
    @test two_stream_cold_growth(kv0; gamma0 = 1 / sqrt(1 - v0^2)) ≈ obs["theory"] rtol = 1e-12
    # the γ returned solves the dispersion relation: ω = iγ
    γ = obs["theory"]; γ0 = 1 / sqrt(1 - v0^2)
    D = 1 - (1 / (2γ0^3)) * (1 / (im * γ - kv0)^2 + 1 / (im * γ + kv0)^2)
    @test abs(D) < 1e-12
    @test two_stream_cold_growth(1.5) == 0
end

@testset "T-161 rate fits recover known rates on synthetic signals [WP-003]" begin
    γ, ω = 0.15, 1.4
    t = collect(0:0.01:30)
    U = exp.(-2γ .* t) .* cos.(ω .* t) .^ 2 .+ 1e-9
    r = damping_from_peaks(t, U; fit_start = 1.0, floor_factor = 10.0)
    @test r.gamma ≈ γ rtol = 1e-3
    @test r.omega ≈ ω rtol = 1e-2
    @test all(>(10 * r.floor), r.U_fit)

    g = 0.35
    U2 = @. min(1e-8 * exp(2g * t), 1e-2) * (1 + 1e-3 * sin(7t))
    r2 = growth_from_window(t, U2; window = (1e-3, 3e-2))
    @test r2.gamma ≈ g rtol = 1e-2
    @test all(u -> 1e-3 * r2.U_max <= u <= 3e-2 * r2.U_max, r2.U_fit)
    # a later nonlinear bounce higher than the first saturation does not move the window
    sat = 10.0
    U3 = @. ifelse(t < sat, 1e-8 * exp(2g * t),
        1e-8 * exp(2g * sat) * (0.6 + 0.4 * cos(1.5 * (t - sat))) * (1 + 0.08 * (t - sat)))
    @test argmax(U3) > findfirst(>(sat + 4), t)          # the global maximum is a later bounce
    r3 = growth_from_window(t, U3; window = (1e-3, 3e-2))
    @test r3.t_max ≈ sat atol = 0.5
    @test r3.gamma ≈ g rtol = 1e-2
    # a first local maximum below 10 % of the global maximum (a noise wiggle) is skipped: the anchor
    # falls through to the next maximum
    U4 = @. 0.05 * exp(-((t - 5) / 0.5)^2) + exp(-((t - 20) / 3)^2) + 1e-6
    @test any(i -> isapprox(t[i], 5; atol = 0.02), local_maxima(U4))    # the early wiggle is a local maximum ...
    @test U4[findfirst(>=(4.9), t)] < 0.1 * maximum(U4)                  # ... below the threshold
    r4 = growth_from_window(t, U4; window = (1e-3, 3e-2))
    @test r4.t_max ≈ 20 atol = 0.05
    @test r4.U_max ≈ maximum(U4)

    f = [3cos(2π * 2 * j / 16 + 0.3) + 0.5 for j in 0:16]   # 16 points plus the periodic duplicate
    @test mode_amplitude(reshape(f, :, 1), 2)[1] ≈ 3 rtol = 1e-12
end

# WP-012: seed ensembles, the ParticleMesh column through the case files, analysis repairs.
include(joinpath(CC, "ensemble.jl"))
include(joinpath(CC, "manifest.jl"))
using .Ensemble, .RunManifest
using Random: Xoshiro, randn, rand
using Statistics: mean, std, var
using SHA: sha256

# Narrowband noise like the thermal Langmuir fluctuations of a PIC run: K complex sinusoids with
# random frequencies in [1, 2) ω_p, random phases and Gaussian amplitudes, mean energy σ².
function narrowband_noise(rng, t, σ; K = 24)
    ωk = 1 .+ rand(rng, K)
    φ = 2π .* rand(rng, K)
    a = randn(rng, ComplexF64, K) .* (σ / sqrt(K))
    return [sum(a .* cis.(ωk .* ti .+ φ)) for ti in t]
end

@testset "T-180 ensemble fit recovers a damping rate; the floor falls as 1/M [A5, A7, WP-012]" begin
    γ, ω = 0.1534, 1.4157
    t = collect(0:0.0373:50)
    signal = exp.(-γ .* t) .* cos.(ω .* t) .* cis(0.4)    # a standing wave's mode coefficient
    # without noise the ensemble of identical seeds is the signal itself
    e0 = ensemble_damping(t, [signal, signal, signal]; fit_start = 3.0, floor_factor = 10.0)
    @test e0.ok
    @test e0.gamma ≈ γ rtol = 1e-3
    # 60 realizations of M = 8 seeds with independent noise, σ² = 3e-3 of the initial energy:
    # the jackknife standard error over seeds covers the error like a standard error should
    # (about 68 % within 1σ), and no realization is off by more than 4σ
    M, R, σ = 8, 60, sqrt(3e-3)
    pulls = Float64[]
    for r in 1:R
        rng = Xoshiro(1000r + M)
        e = ensemble_damping(t, [signal .+ narrowband_noise(rng, t, σ) for _ in 1:M]; fit_start = 3.0, floor_factor = 10.0)
        @test e.ok && e.M == M && length(e.used) >= 3
        @test isfinite(e.jackknife.stderr) && e.jackknife.stderr > 0
        push!(pulls, (e.gamma - γ) / e.jackknife.stderr)
    end
    @test mean(abs.(pulls) .<= 1) >= 0.5
    @test maximum(abs, pulls) <= 4
    # the fitted floor (median of the last 20 % of |⟨Ê⟩|²) at M seeds against one seed: the ratio
    # is M within three standard errors from the spread over realizations
    for Mk in (4, 16)
        f1, fM = Float64[], Float64[]
        for r in 1:R
            rng = Xoshiro(7919r + Mk)
            cs = [signal .+ narrowband_noise(rng, t, σ) for _ in 1:Mk]
            push!(fM, ensemble_damping(t, cs; fit_start = 3.0, floor_factor = 10.0).floor)
            push!(f1, ensemble_damping(t, cs[1:1]; fit_start = 3.0, floor_factor = 10.0).floor)
        end
        ratio = mean(f1) / mean(fM)
        δ = ratio * sqrt(var(f1) / (R * mean(f1)^2) + var(fM) / (R * mean(fM)^2))
        @test abs(ratio - Mk) <= 3δ
        @test δ < Mk / 4                                   # the check has power
    end
    # the complex coefficient keeps the phase that mode_amplitude drops
    f = [3cos(2π * j / 16 + 0.3) for j in 0:16]
    c = only(mode_coefficient(reshape(f, :, 1), 1))
    @test abs(c) ≈ 3 rtol = 1e-12
    @test angle(c) ≈ 0.3 rtol = 1e-12
    @test abs(c) ≈ only(mode_amplitude(reshape(f, :, 1), 1)) rtol = 1e-14
end

# The value of the `random_seed = ` line of a namelist, and the namelist without that line.
seed_line(text) = parse(Int, match(r"\n\s*random_seed = (\d+),\n", text)[1])
without_seed(text) = replace(text, r"\n\s*random_seed = \d+,\n" => "\n")

@testset "T-181 one namelist per seed, identical apart from the seed [A5, WP-003, WP-012]" begin
    landau = load_case("landau")
    seeds = case_seeds(landau)
    @test length(seeds) >= 5 && allunique(seeds)
    # Smilei seeds each patch with random_seed + patch index (Patch.cpp), so seeds closer than the
    # patch count would share streams; the case spaces them further
    np = smilei_parameters(landau).number_of_patches
    @test minimum(diff(sort(seeds))) > np
    texts = [namelist(landau; seed = s) for s in seeds]
    @test [seed_line(x) for x in texts] == seeds
    @test length(unique(bytes2hex.(sha256.(without_seed.(texts))))) == 1
    @test all(x -> count("random_seed", x) == 1, texts)
    # a case without the list keeps its single random_seed (schema 1 stays backward compatible)
    ts = load_case("two-stream")
    @test case_seeds(ts) == [ts["numerics"]["random_seed"]]
    @test namelist(ts) == namelist(ts; seed = ts["numerics"]["random_seed"])
    # prepare_run writes one directory per seed with that namelist, and one run.sh that runs them all
    runs = mktempdir()
    dir = prepare_run("landau"; tag = "t181", runs)
    for s in seeds
        @test read(joinpath(dir, "seed-$s", "smilei.py"), String) == namelist(landau; seed = s)
        @test isfile(joinpath(dir, "seed-$s", "case.toml"))
    end
    sh = read(joinpath(dir, "run.sh"), String)
    @test occursin("for seed in $(join(seeds, ' ')); do", sh) && occursin(raw"cd \"$top/seed-$seed\"", sh)
    m = TOML.parsefile(joinpath(dir, "manifest.toml"))
    @test m["seeds"] == seeds
    @test all(s -> haskey(m["inputs"], "seed-$s/smilei.py"), seeds)
end

@testset "T-183 seed aggregation: mean and spread, refusal of a failed seed [A7, WP-012]" begin
    rng = Xoshiro(183)
    μ, σ, M = 0.15, 0.02, 400
    fits = [(; gamma = μ + σ * randn(rng), ok = true, reason = "") for _ in 1:M]
    a = aggregate_seeds(1:M, fits)
    @test a.M == M
    @test abs(a.mean - μ) <= 3σ / sqrt(M)
    @test abs(a.std - σ) <= 3σ / sqrt(2(M - 1))          # standard error of a sample std
    @test a.stderr ≈ a.std / sqrt(M)
    # one failed seed refuses the whole aggregation and is named
    bad = copy(fits)
    bad[7] = (; gamma = NaN, ok = false, reason = "fewer than three maxima in the ensemble fit")
    err = try
        aggregate_seeds(101:100+M, bad); nothing
    catch e
        e
    end
    @test err isa SeedRefused
    @test err.seeds == [107]
    @test occursin("seed 107", sprint(showerror, err))
    # damping_from_peaks results (no `ok` field) fail when γ is NaN: a pure-noise series
    t = collect(0:0.05:25)
    noisy = abs2.(narrowband_noise(Xoshiro(1), t, 1.0))
    r = damping_from_peaks(t, noisy; fit_start = 3.0, floor_factor = 10.0)
    @test !isfinite(r.gamma)
    @test_throws SeedRefused aggregate_seeds([1, 2], [r, (; gamma = 0.1, t_fit = [1.0, 2.0, 3.0])])
end

@testset "T-184 the manifest stores the bare image id [A5, WP-012]" begin
    id = "7030c120c62ce14dbae971cce76a3973f0a031ef63632a6c4bf4fa20e3937b4c"
    dir = mktempdir()
    log = joinpath(dir, "runner.log")
    write(log, """
    # gpu-run 2026-10-31T17:58:05Z
    # worktree: <worktree> (HEAD 1b01ce7)
    # command: smilei scripts/runs/smilei-landau-9a47a107
    # image: localhost/particlemesh-smilei:e4f0148b8952fc87 id $id
    # network: off, gpu: no, timeout: 7200s
    # gpu-run finished: exit 0 after 51s; image: $id; log: <log>
    """)
    img = image_from_header(log)
    @test img.id == id
    @test img.name == "localhost/particlemesh-smilei:e4f0148b8952fc87"
    # run-smilei.sh's image.txt writes "# image: <name> <id>" without the word id
    txt = joinpath(dir, "image.txt")
    write(txt, "# image: particlemesh-smilei:e4f0148b8952fc87 sha256:$id\n# podman run finished: exit 0\n")
    @test image_from_header(nothing, txt).id == id
    @test image_from_header(joinpath(dir, "absent.log")).id == "unknown"
    write_manifest(dir; extra = Dict{String,Any}("image" => Dict{String,Any}("id" => img.id, "name" => img.name)))
    m = read_manifest(dir)
    @test m["image"]["id"] == id
    @test occursin(r"^[0-9a-f]{64}$", m["image"]["id"])
end

@testset "T-185 the two-stream resolution pair differs only in resolution [A5, WP-012]" begin
    a, b = load_case("two-stream"), load_case("two-stream-256")
    @test b["grid"]["cells"] == 2a["grid"]["cells"]
    @test b["time"]["dt"] ≈ a["time"]["dt"] / 2 rtol = 1e-12
    # the output cadence in time is the same, so both fits see the same sampling
    for k in ("scalars_every", "fields_every")
        @test b["diagnostics"][k] == 2a["diagnostics"][k]
    end
    strip_res(c) = (c = deepcopy(c); for k in ("name", "description", "_path"); delete!(c, k); end;
        delete!(c["grid"], "cells"); delete!(c["time"], "dt");
        delete!(c["diagnostics"], "scalars_every"); delete!(c["diagnostics"], "fields_every"); c)
    @test strip_res(a) == strip_res(b)
    pb = smilei_parameters(b)
    @test pb.number_of_timesteps == round(Int, b["time"]["t_end"] / b["time"]["dt"])
    @test pb.timestep < pb.cell_length
end

@testset "T-186 a fit with too few points is refused, not reported [A5, WP-012]" begin
    t = collect(0:0.24:40)                                 # two-stream field dumps every 32 steps
    g = 0.35
    U = @. min(1e-8 * exp(2g * t), 1e-2)
    r = growth_from_window(t, U; window = (1e-3, 3e-2))
    c = checked_fit(r; min_points = 5)
    @test c.usable && c.gamma == r.gamma && c.points == length(r.t_fit)
    @test c.points >= 5
    # a window that the cadence crosses in four samples
    tc = collect(0:1.6:40)
    rc = growth_from_window(tc, @. min(1e-8 * exp(2g * tc), 1e-2); window = (1e-3, 3e-2))
    cc = checked_fit(rc; min_points = 5)
    @test !cc.usable && cc.gamma === nothing && cc.points < 5
    @test occursin("points", cc.flag)
    # a failed fit (NaN) is refused too
    @test !checked_fit((; gamma = NaN, gamma_stderr = NaN, t_fit = Float64[])).usable
end

# The ParticleMesh column (scripts/es1d/crosscode.jl), in a module of its own so that its
# includes of the cross-code modules do not clash with the ones above.
module PMCross
include(joinpath(dirname(dirname(@__DIR__)), "es1d", "crosscode.jl"))
end

bits(x) = reinterpret(UInt64, Array(x))

@testset "T-182 a seed means one ParticleMesh load and one set of observables [A4, WP-012]" begin
    case = PMCross.load_case("landau")
    small = deepcopy(case)                             # the case at 64 particles per cell and 2/ω₀
    small["species"][1]["particles_per_cell"] = 64
    small["time"]["t_end"] = 2.0
    s1, s2 = PMCross.case_seeds(case)[1:2]
    a = only(PMCross.build(small; seed = s1)[1].species)
    b = only(PMCross.build(small; seed = s1)[1].species)
    c = only(PMCross.build(small; seed = s2)[1].species)
    @test Array(a.count) == Array(b.count) == [64 * 64]
    @test Array(a.cell[1]) == Array(b.cell[1])
    @test bits(a.offset[1]) == bits(b.offset[1]) && bits(a.w) == bits(b.w)
    @test all(d -> bits(a.u[d]) == bits(b.u[d]), 1:3)
    # another seed: the same lattice positions and weights (seed-independent), other velocities
    @test Array(a.cell[1]) == Array(c.cell[1]) && bits(a.offset[1]) == bits(c.offset[1]) && bits(a.w) == bits(c.w)
    for d in 1:3
        ua, uc = Array(a.u[d]), Array(c.u[d])
        @test count(ua .!= uc) == length(ua)
        @test abs(std(uc) / std(ua) - 1) < 0.05        # both Maxwellian at v_th, 4096 draws
    end
    # the default seed of a seeded case is its first one
    @test bits(only(PMCross.build(small)[1].species).u[1]) == bits(a.u[1])
    # rerun through the run path: bitwise identical series and identical observables
    r1 = PMCross.simulate(small; seed = s1, diagnostics_every = 40)
    r2 = PMCross.simulate(small; seed = s1, diagnostics_every = 40)
    r3 = PMCross.simulate(small; seed = s2)
    @test bits(r1.U) == bits(r2.U)
    @test reinterpret(UInt64, r1.C) == reinterpret(UInt64, r2.C)
    @test isequal(PMCross.observe(small, r1.tU, r1.U, abs2.(r1.C))[2], PMCross.observe(small, r2.tU, r2.U, abs2.(r2.C))[2])
    # chunked (with diagnostics) and unchunked runs give the same bits
    @test bits(r3.U) != bits(r1.U)
    r1b = PMCross.simulate(small; seed = s1)
    @test bits(r1b.U) == bits(r1.U)
    @test length(r1.tC) == length(1:small["diagnostics"]["fields_every"]:r1.info.nsteps)
end
