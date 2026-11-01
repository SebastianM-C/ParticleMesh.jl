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

@testset "T-253 the PIConGPU recipe is pinned [A6, WP-017]" begin
    recipe = read(joinpath(CC, "containers", "picongpu.Containerfile"), String)
    froms = [l for l in split(recipe, '\n') if startswith(strip(l), "FROM")]
    @test !isempty(froms)
    @test all(l -> occursin(r"@sha256:[0-9a-f]{64}\b", l), froms)
    # every source fetched by git is cloned at a tag and checked against a full commit id
    for name in ("PICONGPU", "OPENPMD")
        tag = match(Regex("ARG $(name)_TAG=(\\S+)"), recipe)
        commit = match(Regex("ARG $(name)_COMMIT=(\\S+)"), recipe)
        @test tag !== nothing && occursin(r"^v?\d+(\.\d+)+$", tag.captures[1])
        @test commit !== nothing && occursin(r"^[0-9a-f]{40}$", commit.captures[1])
        @test occursin("--branch \"\${$(name)_TAG}\"", recipe)
        @test occursin("rev-parse HEAD)\" = \"\${$(name)_COMMIT}\"", recipe)
    end
    @test count("git clone", recipe) == 2
    @test !occursin(r"\b(master|main|develop)\b", recipe)
    # the CUDA series is fixed and checked against the installed nvcc
    series = match(r"ARG CUDA_SERIES=(\d+)-(\d+)\n", recipe)
    release = match(r"ARG CUDA_RELEASE=(\d+)\.(\d+)\n", recipe)
    @test series !== nothing && release !== nothing && series.captures == release.captures
    @test occursin("grep -q \"release \${CUDA_RELEASE}\"", recipe)
    @test isfile(joinpath(CC, "containers", "picongpu", "pm-versions.sh"))
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
    # on the runs' own time grid (t ≤ 25, fields every 4 steps) and noise (per seed 2.5e-3 of the
    # initial energy), where the floor window still holds signal: the jackknife still covers
    tr = collect(0:4*0.0093266:25)
    sr = exp.(-γ .* tr) .* cos.(ω .* tr)
    pr = Float64[]
    for r in 1:R
        rng = Xoshiro(4242r)
        e = ensemble_damping(tr, [sr .+ narrowband_noise(rng, tr, sqrt(2.5e-3)) for _ in 1:M]; fit_start = 3.0, floor_factor = 10.0)
        e.ok && isfinite(e.jackknife.stderr) && push!(pr, (e.gamma - γ) / e.jackknife.stderr)
    end
    @test length(pr) >= R - 2
    @test mean(abs.(pr) .<= 1) >= 0.5
    @test mean(abs.(pr) .<= 2) >= 0.85
    # the fitted floor (median of the last 20 % of |⟨Ê⟩|²) at M seeds against one seed, on a grid
    # long enough (t ≤ 50) for the floor window to hold noise only: the ratio is M within three
    # standard errors from the spread over realizations
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

@testset "T-187 the seed scatter measures the noise of the mean without the signal [A5, WP-012]" begin
    t = collect(0:0.0373:50)
    signal = 0.3 .* exp.(-0.02 .* t) .* cos.(1.4 .* t)        # a large coherent part that must not enter
    M, R, σ = 8, 40, sqrt(3e-3)
    est = Float64[]
    for r in 1:R
        rng = Xoshiro(31r)
        cs = [signal .+ narrowband_noise(rng, t, σ) for _ in 1:M]
        sc = seed_scatter(cs, eachindex(t))
        push!(est, sc.noise_of_mean)
        @test sc.noise_per_seed ≈ M * sc.noise_of_mean
    end
    # the noise of each seed has mean energy σ² (over amplitudes and phases), so that of the mean is σ²/M
    @test abs(mean(est) / (σ^2 / M) - 1) <= 3 * std(est) / sqrt(R) / (σ^2 / M) + 0.05
    # noise shared by all seeds cancels in the scatter like the signal: the scatter is no independence test
    shared = narrowband_noise(Xoshiro(5), t, σ)
    @test seed_scatter([signal .+ shared for _ in 1:M], eachindex(t)).noise_of_mean ≈ 0 atol = 1e-30
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
    @test isequal(PMCross.observe(small, r1.tU, r1.U, r1.tC, abs2.(r1.C))[2], PMCross.observe(small, r2.tU, r2.U, r2.tC, abs2.(r2.C))[2])
    # chunked (with diagnostics) and unchunked runs give the same bits
    @test bits(r3.U) != bits(r1.U)
    r1b = PMCross.simulate(small; seed = s1)
    @test bits(r1b.U) == bits(r1.U)
    @test length(r1.tC) == length(1:small["diagnostics"]["fields_every"]:r1.info.nsteps)
end

@testset "T-188 the ParticleMesh mode fit runs on the field cadence [A5, WP-012]" begin
    # two-stream dumps fields every 32 steps and scalars every 4: each fit must use its own times
    ts = PMCross.load_case("two-stream")
    small = deepcopy(ts)
    for s in small["species"]
        s["mobile"] && (s["particles_per_cell"] = 100)
    end
    sim = PMCross.simulate(small)
    @test length(sim.tC) == length(sim.C) && length(sim.tU) == length(sim.U)
    @test sim.tC[2] - sim.tC[1] ≈ 32 * small["time"]["dt"] rtol = 1e-12
    @test_throws ErrorException PMCross.observe(small, sim.tU, sim.U, sim.tU, abs2.(sim.C))
    r, res = PMCross.observe(small, sim.tU, sim.U, sim.tC, abs2.(sim.C))
    @test res["measured_from_mode_points"] >= 5
    @test res["measured_from_mode"] ≈ res["measured"] rtol = 0.05     # two estimates of one growth rate
end

# ---------------------------------------------------------------------------------------------
# WP-019: case schema 2, the 2D Smilei generator and reader, the 2D theory and the 2D mode fit.
# ---------------------------------------------------------------------------------------------

using SHA: sha256

# WP-014's 1D Weibel relation, included read-only in a module of its own (the file includes
# scripts/crosscode/theory.jl itself and expects REPO).
module EM1DTheory
const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO, "lib", "ParticleMeshElectromagnetic", "test", "theory.jl"))
end

const NAMELIST_HASHES = TOML.parsefile(joinpath(@__DIR__, "namelist-hashes.toml"))

"A case file with `edit!` applied, written to a temporary file and loaded through `load_case`."
function load_variant(name, edit!)
    c = TOML.parsefile(joinpath(CC, "cases", name * ".toml"))
    edit!(c)
    path = joinpath(mktempdir(), name * ".toml")
    open(io -> TOML.print(io, c), path, "w")
    return load_case(path)
end

"SplitMix64 finalizer: a counter-based hash of 64 bits."
function splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end
"Deterministic uniform noise in [-1/2, 1/2) from indices 0 ≤ i, j, n < 2²¹ (counter-based, no RNG state)."
pseudo_noise(i, j, n) = splitmix64(UInt64(i) | UInt64(j) << 21 | UInt64(n) << 42) / 2.0^64 - 0.5

@testset "T-270 schema 2 keeps schema 1 [A1, A5, WP-019]" begin
    # Schema-1 files load to the same values as before: load_case adds only the path.
    for n in ("landau", "two-stream", "weibel")
        raw = TOML.parsefile(joinpath(CC, "cases", n * ".toml"))
        c = load_case(n)
        @test c["schema"] == 1
        @test delete!(copy(c), "_path") == raw
    end
    # The namelists of the schema-1 cases are byte-identical to those of the generator before
    # the schema-2 change (hashes committed before it, test/namelist-hashes.toml).
    for n in ("landau", "two-stream")
        @test "sha256:" * bytes2hex(sha256(namelist(load_case(n)))) == NAMELIST_HASHES[n]
    end
    # A schema-1 file means a 1D grid; its per-axis view.
    v = case_view(load_case("landau"))
    @test v.ndims == 1
    @test v.lengths == (0.6283185307179586,) && v.cells == (64,)
    e = v.species[1]
    @test e.particles_per_cell == (50000,) && e.drift == (0.0, 0.0, 0.0)          # per seed (WP-012)
    @test e.mode == (1,) && e.amplitude == 0.01
    @test v.species[2].mode === nothing
    @test case_view(load_case("two-stream")).species[1].drift == (-0.1, 0.0, 0.0)
    w = case_view(load_case("weibel")).species
    @test w[1].drift == (0.0, 0.3, 0.0) && w[2].drift == (0.0, -0.3, 0.0)   # transverse_drift = [v_y, v_z]
    @test w[2].amplitude == -0.001

    # Both 2D cases parse to their per-axis grid, mode vector, particles per cell and drift.
    lo = case_view(load_case("landau-oblique-2d"))
    @test lo.ndims == 2 && lo.cells == (128, 128)
    @test all(L -> isapprox(L, 2π * sqrt(5) / 10; rtol = 1e-15), lo.lengths)
    @test lo.species[1].particles_per_cell == (32, 32)
    @test lo.species[1].mode == (2, 1) && lo.species[1].amplitude == 0.01
    @test lo.species[1].drift == (0.0, 0.0, 0.0)
    @test lo.species[2].particles_per_cell == (1, 1) && lo.species[2].mode === nothing
    wb = case_view(load_case("weibel-2d"))
    @test wb.ndims == 2 && wb.cells == (256, 256)
    @test all(L -> isapprox(L, π * sqrt(5); rtol = 1e-15), wb.lengths)
    bp, bm = wb.species[1], wb.species[2]
    @test bp.particles_per_cell == (8, 8) && bm.particles_per_cell == (8, 8)
    @test bp.drift == (0.0, 0.0, 0.3) && bm.drift == (0.0, 0.0, -0.3)      # beams along z
    @test bp.mode == (2, 1) && bm.mode == (2, 1)
    @test bp.amplitude == 0.001 && bm.amplitude == -0.001

    # Validation.
    pert(c) = c["species"][1]["perturbation"]
    @test_throws ErrorException load_variant("weibel-2d", c -> pert(c)["mode"] = [1.5, 1])
    @test_throws ErrorException load_variant("weibel-2d", c -> pert(c)["mode"] = [2])
    @test_throws ErrorException load_variant("weibel-2d", c -> c["grid"]["boundaries"] = ["periodic", "open"])
    @test_throws ErrorException load_variant("weibel-2d", c -> c["grid"]["boundaries"] = "periodic")
    # a box in which the stated mode is not periodic
    @test_throws ErrorException load_variant("weibel-2d", c -> c["grid"]["length"][2] *= 1.1)
    @test_throws ErrorException load_variant("weibel-2d", c -> c["wavevector"]["angle"] += 0.01)
    # a species perturbed along another mode than the stated wavevector
    @test_throws ErrorException load_variant("weibel-2d", c -> pert(c)["mode"] = [1, 2])
    @test_throws ErrorException load_variant("weibel-2d", c -> c["species"][1]["particles_per_cell"] = 64)
    @test_throws ErrorException load_variant("weibel-2d", c -> c["species"][1]["drift"] = 0.3)
    @test_throws ErrorException load_variant("weibel-2d", c -> c["species"][1]["drift"] = [0.0, 0.0, 1.0])
    @test_throws ErrorException load_variant("weibel-2d", c -> c["species"][1]["transverse_drift"] = [0.3, 0.0])
    @test_throws ErrorException load_variant("weibel-2d", c -> c["schema"] = 3)
    @test load_variant("weibel-2d", identity)["name"] == "weibel-2d"   # the round trip itself loads
end

@testset "T-271 2D unit conversion and namelist [A1, A5, WP-019]" begin
    # Hand-computed from cases/landau-oblique-2d.toml: L = 2π√5/10 per axis, 128 cells, so
    # Δx = Δy = 2π√5/1280 = 0.0109762730172511; Δt = 0.95/√(2/Δ²) = 0.95Δ/√2 = 0.0073733272285205;
    # steps = round(25/Δt) = round(3390.60) = 3391; v_th = 0.05 c gives T = 0.0025 m_e c²; 32 × 32
    # electrons per cell; 128 cells per axis give 4 × 4 patches of 32 × 32 cells.
    c = load_case("landau-oblique-2d")
    p = smilei_parameters(c)
    Δ = 2π * sqrt(5) / 1280
    @test p.geometry == "2Dcartesian"
    @test all(d -> isapprox(d, Δ; rtol = 1e-15), p.cell_length)
    @test p.number_of_cells == (128, 128)
    @test p.timestep == c["time"]["dt"]
    @test p.timestep ≈ 0.95 * Δ / sqrt(2) rtol = 1e-14
    @test p.number_of_timesteps == 3391
    @test p.number_of_patches == (4, 4) && p.threads == 16
    e, ion = p.species
    @test e.density == 1.0
    @test e.temperature ≈ 0.0025 rtol = 1e-15
    @test e.momentum == "maxwell-juettner"
    @test e.regular_number == (32, 32) && e.particles_per_cell == 1024
    @test e.mean_velocity == (0.0, 0.0, 0.0)
    @test e.profile.base == 1.0
    @test e.profile.amplitude ≈ 0.01 rtol = 1e-15
    @test e.profile.mode == (2, 1)
    @test all(L -> isapprox(L, 2π * sqrt(5) / 10; rtol = 1e-15), e.profile.lengths)
    @test e.time_frozen == 0.0
    @test ion.profile === nothing && ion.regular_number == (1, 1) && ion.time_frozen > p.simulation_time

    # Hand-computed from cases/weibel-2d.toml: L = π√5 per axis, 256 cells, Δ = π√5/256 =
    # 0.0274406825431278; Δt = 0.95Δ/√2 = 0.0184333180713014; steps = round(40/Δt) =
    # round(2170.0) = 2170; each beam 0.5 n_c with the relative amplitude ±0.001, so the
    # absolute amplitude ±0.0005; drifts ±0.3 c along z; 8 × 8 per cell; 4 × 4 patches.
    w = load_case("weibel-2d")
    q = smilei_parameters(w)
    Δw = π * sqrt(5) / 256
    @test all(d -> isapprox(d, Δw; rtol = 1e-15), q.cell_length)
    @test q.number_of_cells == (256, 256)
    @test q.timestep ≈ 0.95 * Δw / sqrt(2) rtol = 1e-14
    @test q.number_of_timesteps == 2170
    @test q.number_of_patches == (4, 4)
    b1, b2, _ = q.species
    @test b1.density == 0.5
    @test b1.profile.amplitude ≈ 0.0005 rtol = 1e-15
    @test b2.profile.amplitude ≈ -0.0005 rtol = 1e-15
    @test b1.mean_velocity == (0.0, 0.0, 0.3) && b2.mean_velocity == (0.0, 0.0, -0.3)
    @test b1.momentum == "cold" && b1.regular_number == (8, 8) && b1.particles_per_cell == 64

    # Both case files: Δt = 0.95 of the 2D Courant limit 1/√(1/Δx² + 1/Δy²), to roundoff.
    for r in (p, q)
        dx, dy = r.cell_length
        @test r.timestep ≈ 0.95 / sqrt(1 / dx^2 + 1 / dy^2) rtol = 4eps()
    end

    # The patch rule on a box that is not square: powers of 2 per axis dividing the cells, more
    # than 2·oversize + 1 = 5 cells per patch at order 2 (Smilei Params.cpp).
    for cells in ((64, 32), (128, 128), (256, 64), (48, 40), (16, 8))
        r = smilei_parameters(load_variant("landau-oblique-2d", c -> begin
            c["grid"]["cells"] = collect(cells)
            c["time"]["dt"] = 0.95 / sqrt(sum(abs2, cells ./ c["grid"]["length"]))
        end))
        for d in 1:2
            np_ = r.number_of_patches[d]
            @test ispow2(np_) && cells[d] % np_ == 0 && cells[d] ÷ np_ > 5
        end
        @test r.threads == prod(r.number_of_patches) <= 16
    end

    # The namelist carries the same numbers.
    for (case, pp) in ((c, p), (w, q))
        text = namelist(case)
        @test occursin("geometry = \"2Dcartesian\"", text)
        @test occursin("interpolation_order = 2", text)
        @test occursin("cell_length = [$(pp.cell_length[1]), $(pp.cell_length[2])]", text)
        @test occursin("number_of_cells = [$(pp.number_of_cells[1]), $(pp.number_of_cells[2])]", text)
        @test nl_number(text, "timestep") == pp.timestep
        @test nl_number(text, "number_of_timesteps") == pp.number_of_timesteps
        # Smilei's step count, n_time = (int)(simulation_time/timestep), is the case's
        @test trunc(Int, nl_number(text, "simulation_time") / nl_number(text, "timestep")) == pp.number_of_timesteps
    end
    # the reason: n·Δt/Δt truncates below n for the Landau step (Smilei ran 3390 of 3391 steps)
    @test trunc(Int, 3391 * p.timestep / p.timestep) == 3390
    for (case, pp) in ((c, p), (w, q))
        text = namelist(case)
        m = match(r"number_of_patches = \[(\d+), (\d+)\]", text)
        @test m !== nothing
        for d in 1:2
            np_ = parse(Int, m[d])
            @test ispow2(np_) && pp.number_of_cells[d] % np_ == 0 && pp.number_of_cells[d] ÷ np_ > 5
        end
        @test occursin("EM_boundary_conditions = [[\"periodic\"], [\"periodic\"]]", text)
        @test count("boundary_conditions = [[\"periodic\"], [\"periodic\"]]", text) ==
              length(pp.species) + 1                                  # every species and Main
        @test count("Species(", text) == length(pp.species)
        @test occursin("import numpy as np", text)
    end
    text = namelist(w)
    @test occursin("mean_velocity = [0.0, 0.0, 0.3]", text)
    @test occursin("mean_velocity = [0.0, 0.0, -0.3]", text)
    @test count("regular_number = [8, 8]", text) == 2
    @test count("particles_per_cell = 64,", text) == 2
    @test occursin("regular_number = [32, 32]", namelist(c))
end

# The density profiles of a generated 2D namelist: name => (base, amplitude, mx, Lx, my, Ly), or
# a constant density. Parses the form `smilei.jl` writes.
function nl_profiles(text)
    out = Dict{String,Any}()
    for m in eachmatch(r"def density_(\w+)\(x, y\):\n    return ([^\n]+)\n", text)
        e = match(r"^(\S+) \+ (\S+) \* np\.cos\(2\.0 \* np\.pi \* \((\S+) \* x / (\S+) \+ (\S+) \* y / (\S+)\)\)$", m[2])
        e === nothing && error("unexpected profile $(m[2])")
        out[m[1]] = parse.(Float64, Tuple(e.captures))
    end
    return out
end

@testset "T-272 the oblique profile [A5, WP-019]" begin
    for name in ("landau-oblique-2d", "weibel-2d")
        c = load_case(name)
        v = case_view(c)
        prof = nl_profiles(namelist(c))
        (Lx, Ly), (Nx, Ny) = v.lengths, v.cells
        xs = ((0:Nx-1) .+ 0.5) .* (Lx / Nx)     # Smilei samples profiles at cell centres
        ys = ((0:Ny-1) .+ 0.5) .* (Ly / Ny)
        for s in v.species
            s.mode === nothing && continue
            @test haskey(prof, s.name)
            base, amp, mx, lx, my, ly = prof[s.name]
            @test (mx, my) == Float64.(s.mode) && lx == Lx && ly == Ly
            f(x, y) = base + amp * cos(2.0 * π * (mx * x / lx + my * y / ly))
            ref(x, y) = s.density * (1 + s.amplitude * cos(2π * (s.mode[1] * x / Lx + s.mode[2] * y / Ly)))
            @test maximum(abs(f(x, y) - ref(x, y)) for x in xs, y in ys) <= 4eps() * s.density
        end
        # the namelist passes the function, not a separable cosine
        @test !occursin("cosine(", namelist(c))
    end
    # The opposite-phase beams of weibel-2d give a uniform net density to roundoff.
    c = load_case("weibel-2d")
    v = case_view(c)
    prof = nl_profiles(namelist(c))
    (Lx, Ly), (Nx, Ny) = v.lengths, v.cells
    n(p, x, y) = p[1] + p[2] * cos(2.0 * π * (p[3] * x / p[4] + p[5] * y / p[6]))
    total = [n(prof["beam_plus"], x, y) + n(prof["beam_minus"], x, y)
             for x in ((0:Nx-1) .+ 0.5) .* (Lx / Nx), y in ((0:Ny-1) .+ 0.5) .* (Ly / Ny)]
    @test maximum(abs.(total .- 1.0)) <= 4eps()
end

@testset "T-273 theory of the 2D cases [A5, A7, WP-019]" begin
    # Oblique Landau: |k| and the rate equal those of landau.toml.
    landau = load_case("landau")
    k1 = 2π / landau["grid"]["length"]
    e1 = landau["species"][1]
    c = load_case("landau-oblique-2d")
    v = case_view(c)
    wv = mode_wavevector(v.species[1].mode, v.lengths)
    @test wv.k ≈ k1 rtol = 1e-14
    @test wv.angle ≈ atan(1, 2) rtol = 1e-14                    # mode [2, 1] in a square box, not 45°
    @test all(wv.kvec .* v.lengths ./ 2π .≈ (2, 1))             # periodic in the box
    @test c["wavevector"]["magnitude"] ≈ wv.k rtol = 1e-14
    @test c["wavevector"]["angle"] ≈ wv.angle rtol = 1e-14
    e2 = c["species"][1]
    @test e2["thermal_velocity"] == e1["thermal_velocity"] && e2["density"] == e1["density"]
    kλ = wv.k * e2["thermal_velocity"] / sqrt(e2["density"])
    ω = landau_root(kλ)
    obs = c["observables"]
    @test -imag(ω) ≈ landau["observables"]["damping_rate"]["theory"] rtol = 1e-13
    @test real(ω) ≈ landau["observables"]["damping_rate"]["theory_frequency"] rtol = 1e-13
    for key in ("damping_rate_mode", "damping_rate_energy")
        @test obs[key]["theory"] ≈ -imag(ω) rtol = 1e-14
        @test obs[key]["theory_frequency"] ≈ real(ω) rtol = 1e-14
    end
    # The Ex/Ey energy ratio of a longitudinal wave on the Yee grid, and its continuum limit.
    p = smilei_parameters(c)
    ratio = yee_energy_ratio(wv.kvec, p.cell_length)
    @test obs["energy_ratio"]["theory"] ≈ ratio rtol = 1e-14
    @test obs["energy_ratio"]["theory_continuum"] ≈ (wv.kvec[1] / wv.kvec[2])^2 rtol = 1e-14
    @test ratio < (wv.kvec[1] / wv.kvec[2])^2                    # sin(a)/a falls with a: kx Δx > ky Δy
    # continuum limit: the ratio approaches (kx/ky)² as Δ → 0
    @test yee_energy_ratio(wv.kvec, p.cell_length ./ 1000) ≈ 4 rtol = 1e-6

    # 2D Weibel: a root of the cold-fluid relation in the theory.jl header, equal to WP-014's 1D
    # relation and to weibel.toml's value at the same |k|, v₀ and ω_p.
    w = load_case("weibel-2d")
    wv2 = case_view(w)
    kw = mode_wavevector(wv2.species[1].mode, wv2.lengths)
    @test kw.k ≈ 2 rtol = 1e-14
    @test kw.angle ≈ atan(1, 2) rtol = 1e-14
    @test all(kw.kvec .* wv2.lengths ./ 2π .≈ (2, 1))
    @test w["wavevector"]["magnitude"] ≈ kw.k rtol = 1e-14
    v0 = wv2.species[1].drift[3]
    n0 = sum(s.density for s in wv2.species if s.name != "ion")
    γ = weibel_cold_growth_2d(kw.k, v0; ωp2 = n0)
    @test abs(weibel_dispersion_2d(im * γ, kw.k, v0; ωp2 = n0)) < 1e-12
    @test γ ≈ EM1DTheory.weibel_cold_growth(kw.k, v0; ωp2 = n0) rtol = 1e-14
    weibel1d = load_case("weibel")
    @test weibel1d["species"][1]["transverse_drift"][1] == v0     # the same v₀ and n₀ as the 1D case
    @test 2π / weibel1d["grid"]["length"] ≈ kw.k rtol = 1e-14
    @test γ ≈ weibel1d["observables"]["growth_rate"]["theory"] rtol = 1e-13
    for key in ("growth_rate_mode", "growth_rate_energy")
        @test w["observables"][key]["theory"] ≈ γ rtol = 1e-14
    end
    # the rotation about k maps the 1D case onto the 2D one at every |k|, v₀ and ω_p
    for k in (0.3, 1.0, 2.0, 7.5), u in (0.05, 0.3, 0.8), ωp2 in (0.5, 1.0)
        @test weibel_cold_growth_2d(k, u; ωp2) ≈ EM1DTheory.weibel_cold_growth(k, u; ωp2) rtol = 1e-14
    end
end

@testset "T-274 the 2D mode fit [A7, WP-019]" begin
    # A non-square periodic box with Smilei's 2D layout: (Nx + 1) × (Ny + 1) points per field,
    # the last row and column repeating the first, Yee positions as `read_fields` gives them
    # (a dual axis starts at −½ cell: Ex at (i − ½, j), Ey at (i, j − ½), Bx at (i, j − ½),
    # By at (i − ½, j), in cells).
    Nx, Ny, Lx, Ly = 32, 24, 1.3, 0.7          # k_xΔx ≠ k_yΔy, so the Yee and continuum forms differ
    Δx, Δy = Lx / Nx, Ly / Ny
    mode = (2, 1)
    wv = mode_wavevector(mode, (Lx, Ly))
    kx, ky = wv.kvec
    k̂ = wv.kvec ./ wv.k
    pos(sx, sy) = (((0:Nx) .- sx) .* Δx, ((0:Ny) .- sy) .* Δy)
    pEx, pEy = pos(0.5, 0.0), pos(0.0, 0.5)
    pBx, pBy = pos(0.0, 0.5), pos(0.5, 0.0)
    sample(f, p, nt) = [f(x, y, n) for x in p[1], y in p[2], n in 1:nt]
    phase(x, y) = kx * x + ky * y

    # A damped longitudinal wave: φ = A(t) cos(k·r) on the nodes, E = −∇φ by Yee differences,
    # which gives Ex = A (2/Δx) sin(kxΔx/2) sin(k·r) at the Ex points (and likewise Ey), with
    # A(t) = A₀ e^{−γt} cos(ωt), and noise on every point that sets the floor of the projected
    # mode near t = 50, so that the last 20 % of the series (the floor estimate) is noise.
    γ, ω, A0, σ = 0.15, 1.4, 1e-3, 2e-4
    t = collect(0:0.05:60)
    nt = length(t)
    A(n) = A0 * exp(-γ * t[n]) * cos(ω * t[n])
    sx, sy = 2sin(kx * Δx / 2) / Δx, 2sin(ky * Δy / 2) / Δy
    Ex = sample((x, y, n) -> A(n) * sx * sin(phase(x, y)), pEx, nt)
    Ey = sample((x, y, n) -> A(n) * sy * sin(phase(x, y)), pEy, nt)
    # the noise, then the periodic duplicate restored
    for F in (Ex, Ey), n in 1:nt, j in 1:Ny, i in 1:Nx
        F[i, j, n] += σ * pseudo_noise(i, j + (F === Ey ? 100 : 0), n)
    end
    for F in (Ex, Ey)
        F[Nx+1, :, :] .= F[1, :, :]
        F[:, Ny+1, :] .= F[:, 1, :]
    end
    c = project_mode(Ex, Ey, pEx, pEy, mode, (Lx, Ly); component = :parallel)
    @test length(c) == nt
    # the noiseless amplitude of E·k̂ at t = 0 is A₀ (k̂x sx + k̂y sy)
    @test abs(c[1]) ≈ A0 * (k̂[1] * sx + k̂[2] * sy) rtol = 1e-2
    r = damping_from_peaks(t, abs2.(c); fit_start = 1.0, floor_factor = 10.0)
    @test isfinite(r.gamma) && length(r.t_fit) >= 5
    @test abs(r.gamma - γ) <= 3 * r.gamma_stderr
    @test r.gamma ≈ γ rtol = 2e-2
    # The frequency from the mean spacing of the maxima is not checked: the noise can split one
    # maximum into two a few outputs apart, which leaves the rate intact (both lie on the line)
    # but shortens one spacing (here ω ≈ 1.56 for 1.4).

    # A growing transverse field B ⊥ k with noise in its amplitude (∇·B = 0 on the Yee grid is not
    # needed for the projection):
    # Bx = −k̂y b(t) sin(k·r), By = k̂x b(t) sin(k·r), b = b₀ e^{γt} until saturation.
    g = 0.26
    b(n) = 1e-6 * exp(g * min(t[n], 20.0))
    Bx = sample((x, y, n) -> -k̂[2] * b(n) * sin(phase(x, y)) * (1 + 1e-3 * pseudo_noise(n, 1, 2)), pBx, nt)
    By = sample((x, y, n) -> k̂[1] * b(n) * sin(phase(x, y)) * (1 + 1e-3 * pseudo_noise(n, 1, 2)), pBy, nt)
    cp = project_mode(Bx, By, pBx, pBy, mode, (Lx, Ly); component = :perpendicular)
    rg = growth_from_window(t, abs2.(cp); window = (1e-4, 1e-2))
    @test isfinite(rg.gamma) && length(rg.t_fit) >= 10
    @test abs(rg.gamma - g) <= 3 * rg.gamma_stderr + 1e-12
    @test rg.gamma ≈ g rtol = 1e-3
    # B ⊥ k has no component along k̂ (to roundoff, staggered positions included)
    cpar = project_mode(Bx, By, pBx, pBy, mode, (Lx, Ly); component = :parallel)
    @test maximum(abs.(cpar) ./ abs.(cp)) < 1e-12

    # A field along a different mode projects to zero on (2, 1), to roundoff.
    for other in ((1, 2), (2, -1), (-2, 1), (3, 1), (0, 1))
        wo = mode_wavevector(other, (Lx, Ly))
        F = sample((x, y, n) -> cos(wo.kvec[1] * x + wo.kvec[2] * y + 0.3), pEx, 1)
        @test abs(only(mode_coefficient(F, mode, pEx, (Lx, Ly)))) < 1e-14
        @test abs(only(mode_coefficient(F, other, pEx, (Lx, Ly)))) ≈ 1 rtol = 1e-13
    end

    # The Ex/Ey field-energy ratio of a longitudinal wave on the Yee grid, computed as Smilei's
    # scalars do (½ ΔxΔy Σ E² over the distinct points), equals its Yee form to roundoff.
    φ(i, j) = cos(kx * i * Δx + ky * j * Δy + 0.4)
    ex = [(φ(i - 1, j) - φ(i, j)) / Δx for i in 1:Nx, j in 0:Ny-1]   # Smilei initE: (φ(i−1) − φ(i))/Δx
    ey = [(φ(i, j - 1) - φ(i, j)) / Δy for i in 0:Nx-1, j in 1:Ny]
    U(f) = 0.5 * Δx * Δy * sum(abs2, f)
    @test U(ex) / U(ey) ≈ yee_energy_ratio(wv.kvec, (Δx, Δy)) rtol = 1e-13
    @test !isapprox(U(ex) / U(ey), (kx / ky)^2; rtol = 1e-3)      # the grid form is not the continuum one
end

@testset "T-275 the 2D Smilei reader [WP-019]" begin
    # Fields0.h5 as DiagnosticFields2D writes it: per iteration one dataset per field of C-order
    # dimensions [Nx + 1, Ny + 1] (x slowest), with per-axis gridSpacing, gridGlobalOffset and
    # position (0.5 on a dual axis, whose first point Smilei puts at −Δ/2). HDF5.jl shows C-order
    # dimensions reversed, so the test writes the transpose.
    dir = mktempdir()
    Nx, Ny, Δx, Δy, dt = 4, 3, 0.25, 0.5, 0.2
    val(name, it) = [(name == "Ex" ? 1.0 : 2.0) * (i + 10j) + 1000it for i in 0:Nx, j in 0:Ny]
    stag = Dict("Ex" => [0.5, 0.0], "Ey" => [0.0, 0.5], "Rho" => [0.0, 0.0])
    h5open(joinpath(dir, "Fields0.h5"), "w") do f
        for it in (0, 8)
            g = create_group(f, "data/" * lpad(string(it), 10, '0'))
            attributes(g)["time"] = it * dt
            attributes(g)["dt"] = dt
            for name in ("Ex", "Ey", "Rho")
                g[name] = permutedims(val(name, it))
                a = attributes(g[name])
                a["gridSpacing"] = [Δx, Δy]
                a["gridGlobalOffset"] = [0.0, 0.0]
                a["position"] = stag[name]
                a["axisLabels"] = ["x", "y"]
                a["dataOrder"] = "C"
            end
        end
    end
    h5open(joinpath(dir, "Fields0.h5"), "r") do f
        @test size(f["data/0000000000/Ex"]) == (Ny + 1, Nx + 1)      # C order [Nx + 1, Ny + 1]
    end
    fl = read_fields(dir)
    @test fl.iterations == [0, 8]
    @test fl.time ≈ [0.0, 1.6]
    @test sort(collect(keys(fl.fields))) == ["Ex", "Ey", "Rho"]
    @test size(fl.fields["Ex"]) == (Nx + 1, Ny + 1, 2)               # points × points × outputs
    @test fl.fields["Ex"][:, :, 2] == val("Ex", 8)
    @test fl.fields["Ey"][:, :, 1] == val("Ey", 0)
    @test fl.fields["Ex"][3, 2, 1] == 2 + 10 * 1                     # (i, j) = (2, 1): x first
    @test fl.x["Ex"] == ([-0.125, 0.125, 0.375, 0.625, 0.875], [0.0, 0.5, 1.0, 1.5])
    @test fl.x["Ey"] == ([0.0, 0.25, 0.5, 0.75, 1.0], [-0.25, 0.25, 0.75, 1.25])
    @test fl.x["Rho"] == ([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.5, 1.0, 1.5])
    sub = read_fields(dir; names = ["Ey"])
    @test collect(keys(sub.fields)) == ["Ey"]
end
