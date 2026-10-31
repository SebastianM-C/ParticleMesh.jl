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

    f = [3cos(2π * 2 * j / 16 + 0.3) + 0.5 for j in 0:16]   # 16 points plus the periodic duplicate
    @test mode_amplitude(reshape(f, :, 1), 2)[1] ≈ 3 rtol = 1e-12
end
