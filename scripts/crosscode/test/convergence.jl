# WP-031: the linear cold-beam initial-value problem (theory.jl, `cold_ivp`) and the scan tooling
# of convergence.jl. Included by runtests.jl, after its own includes of the cross-code modules.

include(joinpath(CC, "convergence.jl"))
using .Convergence
using LinearAlgebra: eigen, eigvals
using JSON

@testset "T-400 the linear cold-beam systems have the dispersion roots as eigenvalues [A5, WP-031]" begin
    # Parameters away from the cases', so that nothing is tuned to them; two beams of unequal
    # density for the two-stream system (the case is symmetric).
    for (k, v0) in ((6.1237243569579, 0.1), (1.5, 0.2), (3.0, 0.45))
        γ0 = 1 / sqrt(1 - v0^2)
        beams = [(; q = -1.0, m = 1.0, n = 0.5, v = v0), (; q = -1.0, m = 1.0, n = 0.5, v = -v0)]
        λ = eigvals(two_stream_ivp_matrix(k, beams))
        γ = two_stream_cold_growth(k * v0; gamma0 = γ0)
        @test maximum(real, λ) ≈ γ rtol = 1e-12 atol = 1e-12
        # each nonzero eigenvalue −iω is a root of 1 = (ω_p²/2γ₀³)[(ω − kv₀)⁻² + (ω + kv₀)⁻²]
        D(ω) = 1 - (1 / (2γ0^3)) * (1 / (ω - k * v0)^2 + 1 / (ω + k * v0)^2)
        nz = [l for l in λ if abs(l) > 1e-8]
        @test length(nz) == 4
        @test all(abs(D(im * l)) < 1e-10 for l in nz)
        # Weibel: ±√X± of X² − A X − B = 0 and the electrostatic branch ω² = ω_p²/γ₀ (E_k, which the
        # Weibel relation leaves out because E_k = 0 on its branch), zero otherwise
        Λ = eigvals(weibel_ivp_matrix(k, beams))
        @test maximum(real, Λ) ≈ weibel_cold_growth_2d(k, v0) rtol = 1e-12
        nzw = [l for l in Λ if abs(l) > 1e-8]
        @test length(nzw) == 6
        es = [l for l in nzw if abs(abs2(l) - 1 / γ0) < 1e-10]
        @test length(es) == 2 && all(abs(real(l)) < 1e-12 for l in es)
        @test all(abs(weibel_dispersion_2d(im * l, k, v0)) < 1e-10 for l in nzw if !(l in es))
    end
    # unequal beams: the growing root of the general cold two-beam relation
    # 1 = Σₛ ω_ps²/(γₛ³ (ω − k vₛ)²), found here by a scan of the dispersion function itself
    k = 4.0
    beams = [(; q = -1.0, m = 1.0, n = 0.7, v = 0.12), (; q = -1.0, m = 1.0, n = 0.3, v = -0.08)]
    λ = eigvals(two_stream_ivp_matrix(k, beams))
    Dg(ω) = 1 - sum(b.n / ((1 - b.v^2)^(-1.5) * (ω - k * b.v)^2) for b in beams)
    @test all(abs(Dg(im * l)) < 1e-9 for l in λ if abs(l) > 1e-8)
    @test count(l -> real(l) > 1e-6, λ) == 1
end

@testset "T-401 the exact linear solution keeps Gauss's law and grows at the root [A5, WP-031]" begin
    for name in ("two-stream", "weibel-2d")
        p = cold_ivp(load_case(name))
        t = collect(0:0.25:40)
        sol = ivp_solution(p.M, p.y0, t)
        @test maximum(gauss_residual(sol, p.k, p.beams; kind = p.kind)) < 1e-13
        # a state violating Gauss's law keeps its violation: ik E − Σ q δn is constant
        y = copy(p.y0)
        iE = p.kind === :two_stream ? 2length(p.beams) + 1 : 3length(p.beams) + 1
        y[iE] += 1e-4
        s2 = ivp_solution(p.M, y, t)
        per = p.kind === :two_stream ? 2 : 3
        viol = [im * p.k * s2[iE, c] - sum(b.q * s2[per * (j - 1) + 1, c] for (j, b) in enumerate(p.beams)) for c in axes(s2, 2)]
        scale = maximum(abs(b.q * s2[per * (j - 1) + 1, c]) for (j, b) in enumerate(p.beams), c in axes(s2, 2))
        @test maximum(abs, viol .- viol[1]) < 1e-13 * scale
        @test abs(viol[1]) > 1e-4
        # on the growing eigenvector the field grows at exactly the root
        F = eigen(p.M)
        j = argmax(real.(F.values))
        g = ivp_solution(p.M, F.vectors[:, j], t)[p.index, :]
        @test linear_fit(t, log.(abs2.(g))).slope / 2 ≈ real(F.values[j]) rtol = 1e-10
    end
    # the initial states: two-stream, both beams perturbed alike, E(0) = α/k from Gauss's law
    # (q = −1, Σ δn = α n₀); Weibel, opposite perturbations, no net charge and no field
    ts = cold_ivp(load_case("two-stream"))
    @test abs(ts.y0[ts.index]) ≈ 1e-3 / ts.k rtol = 1e-14
    @test ts.y0[1] == ts.y0[3] == 0.5e-3 && ts.y0[2] == ts.y0[4] == 0
    @test ts.energy_scale == load_case("two-stream")["grid"]["length"] / 4
    wb = cold_ivp(load_case("weibel-2d"))
    @test wb.y0[1] == -wb.y0[4] == 0.5e-3
    @test all(iszero, wb.y0[7:9])
    @test wb.k ≈ 2 rtol = 1e-14
    @test wb.beams[1].v == 0.3 && wb.beams[2].v == -0.3
    # the energy of a field Re(A e^{ik·r}) over the box is |A|² V/4: check on the Smilei grid sum
    L, N = 1.3, 64
    A = 0.7 - 0.2im
    x = (0:N-1) .* (L / N)
    @test sum(abs2, real.(A .* cis.(2π / L .* x))) * (L / N) / 2 ≈ abs2(A) * L / 4 rtol = 1e-13
    @test_throws ErrorException cold_ivp(load_case("landau"))
end

@testset "T-402 a scan point changes only what it names [A5, WP-031]" begin
    ts = load_case("two-stream")
    # refine 2 of the 128-cell case is the committed 256-cell case of WP-012, namelist for namelist
    d2 = derive_case(ts; name = "n0256", refine = 2)
    strip_header(s) = join(split(s, '\n')[2:end], '\n')
    @test strip_header(namelist(d2)) == strip_header(namelist(load_case("two-stream-256")))
    @test strip_header(namelist(derive_case(ts; name = "n0128"))) == strip_header(namelist(ts))
    # the output times are those of the case at every point; the Courant number follows refine
    for (r, m) in ((1 // 2, 1), (8, 1), (1, 4), (4, 4))
        d = derive_case(ts; name = "p", refine = r, dt_divisor = m)
        for k in ("scalars_every", "fields_every")
            @test d["diagnostics"][k] * d["time"]["dt"] ≈ ts["diagnostics"][k] * ts["time"]["dt"] rtol = 1e-14
        end
        @test d["time"]["dt"] * d["grid"]["cells"] * m ≈ ts["time"]["dt"] * ts["grid"]["cells"] rtol = 1e-14
    end
    # everything else equal, apart from the named changes and the record of them
    d = derive_case(ts; name = "x", amplitude_factor = 0.1, ppc_factor = 1 // 10, seed = 7, fields = ["Ex"])
    @test [s["perturbation"]["amplitude"] for s in d["species"] if haskey(s, "perturbation")] ≈ [1e-4, 1e-4]
    @test [s["particles_per_cell"] for s in d["species"]] == [100, 100, 1]     # the immobile ions keep theirs
    @test d["numerics"]["random_seed"] == 7 && d["diagnostics"]["fields"] == ["Ex"]
    @test d["derived"]["from_sha256"] == bytes2hex(open(sha256, ts["_path"]))
    clean(c) = (c = deepcopy(c); foreach(k -> delete!(c, k), ("name", "description", "_path", "derived")); c)
    a, b = clean(ts), clean(d)
    for s in vcat(a["species"], b["species"])
        delete!(s, "particles_per_cell")
        haskey(s, "perturbation") && delete!(s["perturbation"], "amplitude")
    end
    delete!(a["numerics"], "random_seed"); delete!(b["numerics"], "random_seed")
    delete!(a["diagnostics"], "fields"); delete!(b["diagnostics"], "fields")
    @test a == b
    # 2D: cells and particles per cell per axis
    w = derive_case(load_case("weibel-2d"); name = "n128", refine = 1 // 2, ppc_factor = 2)
    @test w["grid"]["cells"] == [128, 128]
    @test w["species"][1]["particles_per_cell"] == [16, 16] && w["species"][3]["particles_per_cell"] == [1, 1]
    @test w["diagnostics"]["fields_every"] == 8 && w["diagnostics"]["scalars_every"] == 2
    # a product that is not an integer is refused
    @test_throws ErrorException derive_case(ts; name = "bad", refine = 3 // 256)
    @test_throws ErrorException derive_case(load_case("weibel-2d"); name = "bad", refine = 1 // 8)   # scalars_every 4/8
    # a derived file written as prepare writes it passes the generator's validation
    f = joinpath(mktempdir(), "w.toml")
    open(io -> TOML.print(io, w; sorted = true), f, "w")
    @test load_case(f)["grid"]["cells"] == [128, 128]
end

# A series that follows the exact linear solution up to t_sat and then turns over, so that the
# window rule finds a saturation peak; `distort` multiplies it by a known factor.
function synthetic_run(case, key, t; tsat, distort = t -> 1.0)
    U = ivp_series(case, key, t) .* distort.(t)
    i = findlast(<=(tsat), t)
    U[i+1:end] .= U[i] .* exp.(-(t[i+1:end] .- t[i]))
    return U
end

@testset "T-403 the residual from the linear solution is exact on synthetic runs [A7, WP-031]" begin
    for (name, key, tsat) in (("two-stream", "growth_rate", 24.9), ("weibel-2d", "growth_rate_mode", 32.1),
                              ("weibel-2d", "growth_rate_energy", 32.1))
        case = load_case(name)
        θ = case["observables"][key]["theory"]
        w = Tuple(case["observables"][key]["window"])
        t = collect(0:0.03:40)
        # the linear solution itself, and times a constant: the residual is zero, the offset is
        # the window's sampling of the transient (nonzero)
        for c in (1.0, 0.37)
            U = synthetic_run(case, key, t; tsat, distort = _ -> c)
            f = point_fits(case, key, t, U; window = w)
            @test isfinite(f.gamma)
            @test abs(f.gamma - f.gamma_ivp) < 1e-12 * θ
            @test abs(f.gamma / θ - 1) > 1e-3
            @test f.ratio[1] ≈ c rtol = 1e-12
        end
        # a growth e^{2δt} on top: the residual is δ exactly, at any sample offset of the window
        δ = -0.004θ
        U = synthetic_run(case, key, t; tsat, distort = s -> exp(2δ * s))
        f = point_fits(case, key, t, U; window = w)
        @test (f.gamma - f.gamma_ivp) ≈ δ rtol = 1e-9
        @test f.idx == findfirst(==(first(f.r.t_fit)), t):findfirst(==(last(f.r.t_fit)), t)
    end
    # the local rate of an exponential is its rate, of a product the sum
    t = collect(0:0.01:10)
    tc, g = local_rate(t, exp.(0.6 .* t))
    @test all(x -> isapprox(x, 0.3; rtol = 1e-12), g)
    @test first(tc) ≈ 1.0 && last(tc) ≈ 9.0
end

@testset "T-404 summarize reads a finished Smilei run into the scan summary [A5, WP-031]" begin
    # one finished 1D point in Smilei's layout (as T-012 writes it) and one that did not run
    root = mktempdir()
    dir = joinpath(root, "scan")
    mkpath(joinpath(dir, "summary", "cases"))
    case = derive_case(load_case("two-stream"); name = "tiny", refine = 1 // 4)   # 32 cells
    for p in ("tiny", "notrun")
        c = merge(case, Dict("name" => "two-stream-$p"))
        open(io -> TOML.print(io, c; sorted = true), joinpath(dir, "summary", "cases", p * ".toml"), "w")
        mkpath(joinpath(dir, p))
        open(io -> TOML.print(io, c; sorted = true), joinpath(dir, p, "case.toml"), "w")
        write(joinpath(dir, p, "manifest.toml"), "commit = \"test\"\n")
    end
    rd = joinpath(dir, "tiny")
    ts = [0.0, 0.5, 1.0]
    U = [1e-8, 2e-8, 4e-8]
    w(x) = lpad(string(round(x; sigdigits = 11)), 20)
    open(joinpath(rd, "scalars.txt"), "w") do io
        println(io, "# 1 time\n# 2 Utot\n# 3 Uelm_Ex\n#")
        println(io, "#", lpad("time", 19), lpad("Utot", 20), lpad("Uelm_Ex", 20))
        for i in 1:3
            println(io, w(ts[i]), w(1.5 + 1e-9i), w(U[i]))
        end
    end
    N, L = 32, case["grid"]["length"]
    A = [1e-4 * cis(0.3), 2e-4 * cis(0.5)]
    h5open(joinpath(rd, "Fields0.h5"), "w") do f
        for (n, it) in enumerate((0, 64))
            g = create_group(f, "data/" * lpad(string(it), 10, '0'))
            attributes(g)["time"] = it * case["time"]["dt"]
            attributes(g)["dt"] = case["time"]["dt"]
            v = [real(A[n] * cis(2π * j / N)) for j in 0:N]       # N + 1 points, the last repeats the first
            g["Ex"] = v
            a = attributes(g["Ex"])
            a["gridSpacing"] = [L / N]; a["gridGlobalOffset"] = [0.0]; a["position"] = [0.5]
        end
    end
    write(joinpath(rd, "container.txt"), "# threads: 2, cpus: 2\n# cpu model: Test CPU\n# start: 2026-11-03T10:00:00.000Z\n# end: 2026-11-03T10:00:07.500Z\n")
    done = summarize_scan(dir)
    @test done == ["tiny"]
    @test strip(read(joinpath(dir, "summary", "missing.txt"), String)) == "notrun"
    rec = JSON.parsefile(joinpath(dir, "summary", "tiny.json"))
    @test rec["wall_seconds"] == 7.5 && rec["cpu"] == "Test CPU" && rec["threads"] == "2, cpus: 2"
    @test Float64.(rec["series"]["growth_rate"]["U"]) == U
    c = complex.(Float64.(rec["series"]["mode"]["c_re"]), Float64.(rec["series"]["mode"]["c_im"]))
    @test c ≈ A rtol = 1e-12
    @test Float64.(rec["series"]["mode"]["U"]) ≈ abs2.(A) rtol = 1e-12
    @test rec["outputs"]["scalars.txt"] == "sha256:" * bytes2hex(open(sha256, joinpath(rd, "scalars.txt")))
    @test isfile(joinpath(dir, "summary", "tiny-manifest.toml"))
end

@testset "T-405 summarize projects the 2D dumps on B·ê of the case's mode [A5, WP-031]" begin
    # one finished 2D point: B = Re(A e^{ik·r}) ê with ê = ẑ × k̂, Bx and By on their Yee points
    # (dual along y and x, Smilei's first dual point at −Δ/2, T-275), dumped in C order
    root = mktempdir()
    dir = joinpath(root, "scan")
    mkpath(joinpath(dir, "summary", "cases"))
    case = derive_case(load_case("weibel-2d"); name = "tiny", refine = 1 // 4, fields = ["Bx", "By"])   # 64²
    open(io -> TOML.print(io, case; sorted = true), joinpath(dir, "summary", "cases", "tiny.toml"), "w")
    rd = joinpath(dir, "tiny")
    mkpath(rd)
    open(io -> TOML.print(io, case; sorted = true), joinpath(rd, "case.toml"), "w")
    write(joinpath(rd, "manifest.toml"), "commit = \"test\"\n")
    N = 64
    L = case["grid"]["length"]
    Δ = L ./ N
    kv = 2π .* (2, 1) ./ L
    k̂ = kv ./ sqrt(sum(abs2, kv))
    ê = (-k̂[2], k̂[1])
    A = [3e-5 * cis(-0.4), 7e-4 * cis(1.1)]
    pos(dual, d) = dual ? ((0:N) .- 0.5) .* Δ[d] : (0:N) .* Δ[d]
    field(n, comp, dx, dy) = [comp * real(A[n] * cis(kv[1] * x + kv[2] * y)) for x in pos(dx, 1), y in pos(dy, 2)]
    h5open(joinpath(rd, "Fields0.h5"), "w") do f
        for (n, it) in enumerate((0, 16))
            g = create_group(f, "data/" * lpad(string(it), 10, '0'))
            attributes(g)["time"] = it * case["time"]["dt"]
            attributes(g)["dt"] = case["time"]["dt"]
            for (name, comp, st) in (("Bx", ê[1], [0.0, 0.5]), ("By", ê[2], [0.5, 0.0]))
                g[name] = permutedims(field(n, comp, st[1] == 0.5, st[2] == 0.5))
                a = attributes(g[name])
                a["gridSpacing"] = collect(Δ); a["gridGlobalOffset"] = [0.0, 0.0]; a["position"] = st
                a["axisLabels"] = ["x", "y"]; a["dataOrder"] = "C"
            end
        end
    end
    w(x) = lpad(string(round(x; sigdigits = 11)), 20)
    open(joinpath(rd, "scalars.txt"), "w") do io
        println(io, "# 1 time\n# 2 Utot\n# 3 Uelm_Bx_m\n# 4 Uelm_By_m\n#")
        println(io, "#", lpad("time", 19), lpad("Utot", 20), lpad("Uelm_Bx_m", 20), lpad("Uelm_By_m", 20))
        println(io, w(0.0), w(2.4), w(1e-9), w(3e-9))
        println(io, w(0.1), w(2.4), w(2e-9), w(5e-9))
    end
    write(joinpath(rd, "container.txt"), "# start: 2026-11-03T10:00:00.000Z\n# end: 2026-11-03T10:00:01.000Z\n")
    @test summarize_scan(dir) == ["tiny"]
    rec = JSON.parsefile(joinpath(dir, "summary", "tiny.json"))
    s = rec["series"]["growth_rate_mode"]
    @test complex.(Float64.(s["c_re"]), Float64.(s["c_im"])) ≈ A rtol = 1e-10
    @test Float64.(rec["series"]["growth_rate_energy"]["U"]) ≈ [4e-9, 7e-9] rtol = 1e-10
    @test sort(collect(keys(rec["series"]))) == ["growth_rate_energy", "growth_rate_mode"]
end
