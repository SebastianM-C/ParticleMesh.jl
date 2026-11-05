# WP-031: the linear cold-beam initial-value problem (theory.jl, `cold_ivp`) and the scan tooling
# of convergence.jl. Included by runtests.jl, after its own includes of the cross-code modules.

include(joinpath(CC, "convergence.jl"))
using .Convergence
using LinearAlgebra: eigen, eigvals
using JSON

@testset "T-410 the linear cold-beam systems have the dispersion roots as eigenvalues [A5, WP-031]" begin
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

@testset "T-411 the exact linear solution keeps Gauss's law and grows at the root [A5, WP-031]" begin
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
        @test gauss_residual(s2, p.k, p.beams; kind = p.kind) ≈ abs.(viol) ./ scale rtol = 1e-12
        # the growing root of the case's system is the case's registered theory value (two beams, the ions immobile)
        c = load_case(name)
        θ = first(o["theory"] for (_, o) in c["observables"] if o["method"] == "exponential_window")
        @test maximum(real, eigvals(p.M)) ≈ θ rtol = 1e-12
        @test length(p.beams) == 2
        # the time origin: y(0) = y0, and y(t) equals the eigen-expansion Σ cⱼ e^{λⱼ t} vⱼ
        @test ivp_solution(p.M, p.y0, [0.0])[:, 1] == p.y0
        F0 = eigen(p.M)
        cj = F0.vectors \ p.y0
        for ti in (3.7, 21.0)
            @test ivp_solution(p.M, p.y0, [ti])[:, 1] ≈ F0.vectors * (cj .* exp.(F0.values .* ti)) rtol = 1e-8
        end
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
    @test wb.index == 9 && wb.energy_scale == prod(load_case("weibel-2d")["grid"]["length"]) / 4
    # ivp_series in the units of the runs: the box energy of the observed field, or |amplitude|²
    tsc = load_case("two-stream")
    L = tsc["grid"]["length"]
    @test ivp_series(tsc, "growth_rate", [0.0]) ≈ [L / 4 * (1e-3 / ts.k)^2] rtol = 1e-13
    @test ivp_series(tsc, "mode", [0.0]) ≈ [(1e-3 / ts.k)^2] rtol = 1e-13
    wc = load_case("weibel-2d")
    @test ivp_series(wc, "growth_rate_energy", [5.0]) ./ ivp_series(wc, "growth_rate_mode", [5.0]) ≈ [wb.energy_scale] rtol = 1e-13
    # an asymmetric Weibel load has net charge: E_k(0) from Gauss's law, kept along the solution
    asym = deepcopy(wc)
    asym["species"][2]["perturbation"]["amplitude"] = -0.0004
    pa = cold_ivp(asym)
    @test abs(pa.y0[7]) > 0
    @test maximum(gauss_residual(ivp_solution(pa.M, pa.y0, 0:1.0:30), pa.k, pa.beams; kind = :weibel)) < 1e-13
    # the energy of a field Re(A e^{ik·r}) over the box is |A|² V/4: check on the Smilei grid sum
    L, N = 1.3, 64
    A = 0.7 - 0.2im
    x = (0:N-1) .* (L / N)
    @test sum(abs2, real.(A .* cis.(2π / L .* x))) * (L / N) / 2 ≈ abs2(A) * L / 4 rtol = 1e-13
    @test_throws ErrorException cold_ivp(load_case("landau"))
    # the −Δt/2 load (#282): two-stream, half a kick qE⁰Δt/(2mγ³) on each beam's velocity and
    # nothing else; symmetric Weibel, no field at t = 0 and so no change; an unknown start refused
    tsh = cold_ivp(tsc; momenta_at = :t_minus_half)
    γb = 1 / sqrt(1 - 0.1^2)
    kick = -1 * ts.y0[ts.index] * tsc["time"]["dt"] / (2 * γb^3)
    @test tsh.y0[[2, 4]] ≈ [kick, kick] rtol = 1e-14
    @test tsh.y0[[1, 3, 5]] == ts.y0[[1, 3, 5]] && tsh.M == ts.M
    @test cold_ivp(wc; momenta_at = :t_minus_half).y0 == wb.y0
    pah = cold_ivp(asym; momenta_at = :t_minus_half)
    @test pah.y0[2] ≈ -pa.y0[7] * wc["time"]["dt"] / (2 * (1 / sqrt(1 - 0.09))) rtol = 1e-14
    @test pah.y0[5] ≈ -pa.y0[7] * wc["time"]["dt"] / (2 * (1 / sqrt(1 - 0.09))) rtol = 1e-14
    @test pah.y0[[3, 6]] == [0, 0] && pah.y0[[1, 4, 7, 8, 9]] == pa.y0[[1, 4, 7, 8, 9]]
    # every beam's own charge, mass and γ: beams that differ in all three
    ts2 = deepcopy(tsc)
    merge!(ts2["species"][2], Dict("charge" => -2.0, "mass" => 3.0, "drift" => 0.2))
    p2, p2h = cold_ivp(ts2), cold_ivp(ts2; momenta_at = :t_minus_half)
    E0 = p2.y0[p2.index]
    @test p2h.y0[2] ≈ -1 * E0 * tsc["time"]["dt"] / (2 * 1 * (1 - 0.1^2)^-1.5) rtol = 1e-14
    @test p2h.y0[4] ≈ -2 * E0 * tsc["time"]["dt"] / (2 * 3 * (1 - 0.2^2)^-1.5) rtol = 1e-14
    w2 = deepcopy(asym)
    merge!(w2["species"][2], Dict("charge" => -2.0, "mass" => 3.0, "drift" => [0.0, 0.0, -0.2]))
    q2, q2h = cold_ivp(w2), cold_ivp(w2; momenta_at = :t_minus_half)
    Ek = q2.y0[7]
    @test q2h.y0[2] ≈ -1 * Ek * wc["time"]["dt"] / (2 * 1 / sqrt(1 - 0.09)) rtol = 1e-14
    @test q2h.y0[5] ≈ -2 * Ek * wc["time"]["dt"] / (2 * 3 / sqrt(1 - 0.04)) rtol = 1e-14
    @test_throws ArgumentError cold_ivp(tsc; momenta_at = :t1)
end

@testset "T-412 a scan point changes only what it names [A5, WP-031]" begin
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
    rich = derive_case(ts; name = "r", refine = 1 // 2, dt_divisor = 4, amplitude_factor = 0.1, ppc_factor = 1 // 10, seed = 7)
    @test rich["derived"] == Dict{String,Any}("from" => "two-stream", "from_sha256" => d["derived"]["from_sha256"],
        "refine" => "1//2", "dt_divisor" => "4//1", "amplitude_factor" => 0.1, "ppc_factor" => "1//10",
        "seed" => 7, "fields" => "case")
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
    # the box request of a scan: one Smilei step per point, then the summarize step; only the
    # summary comes back; the cost-ledger category of scripts/cpu-box.sh
    scan = joinpath(CC, "scans", "two-stream-offset.toml")
    dir = prepare_scan(scan; runs = mktempdir())
    req = TOML.parsefile(joinpath(dir, "box-request.toml"))
    pts = TOML.parsefile(scan)["points"]
    @test req["category"] == "validation"
    @test [s["kind"] for s in req["steps"]] == [fill("smilei", length(pts)); "julia"]
    @test [basename(s["dir"]) for s in req["steps"][1:end-1]] == [p["name"] for p in pts]
    @test req["steps"][end]["args"][3] == "summarize"
    @test length(req["outputs"]) == 1 && endswith(only(req["outputs"]), "/summary")
    @test isfile(joinpath(dir, pts[1]["name"], "namelist.py")) && isfile(joinpath(dir, "summary", "scan.toml"))
    # both scan files: every point's prepared case is derive_case of its keys, its name says its
    # cells and its Courant number (nNNN = cells per axis, dtN = the case's 0.95 / N), and the
    # request keeps the scan's timeout
    for scanfile in ("two-stream-offset.toml", "weibel-2d-offset.toml")
        sp = joinpath(CC, "scans", scanfile)
        sc = TOML.parsefile(sp)
        base = load_case(sc["case"])
        dir = prepare_scan(sp; runs = mktempdir())
        @test TOML.parsefile(joinpath(dir, "box-request.toml"))["timeout"] == sc["timeout"]
        for p in sc["points"]
            c = TOML.parsefile(joinpath(dir, "summary", "cases", p["name"] * ".toml"))
            want = derive_case(base; Convergence.point_kwargs(p)...)
            delete!(want, "_path")
            @test c == want
            m = match(r"^n0*(\d+)", p["name"])
            @test first(vcat(c["grid"]["cells"])) == parse(Int, m[1])
            dtm = match(r"-dt(\d+)", p["name"])
            dx = first(vcat(c["grid"]["length"])) / first(vcat(c["grid"]["cells"]))
            courant = sc["case"] == "weibel-2d" ? c["time"]["dt"] * sqrt(2) / dx : c["time"]["dt"] / dx
            @test courant ≈ 0.95 / (dtm === nothing ? 1 : parse(Int, dtm[1])) rtol = 2e-4
        end
    end
end

# A series that follows the exact linear solution up to t_sat and then turns over, so that the
# window rule finds a saturation peak; `distort` multiplies it by a known factor.
function synthetic_run(case, key, t; tsat, distort = t -> 1.0)
    U = ivp_series(case, key, t) .* distort.(t)
    i = findlast(<=(tsat), t)
    U[i+1:end] .= U[i] .* exp.(-(t[i+1:end] .- t[i]))
    return U
end

@testset "T-413 the residual from the linear solution is exact on synthetic runs [A7, WP-031]" begin
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
            @test f.ratio[2] ≈ c rtol = 1e-12
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
    U0 = exp.(0.6 .* t)
    U0[501] = 0                                 # t = 5: every stencil holding it is skipped
    tz, gz = local_rate(t, U0)
    @test all(x -> abs(x - 5) > 1.0 - 1e-9, tz) && length(tz) == length(tc) - 201
end

@testset "T-414 summarize reads a finished Smilei run into the scan summary [A5, WP-031]" begin
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
            HDF5.attributes(g)["time"] = it * case["time"]["dt"]
            HDF5.attributes(g)["dt"] = case["time"]["dt"]
            v = [real(A[n] * cis(2π * j / N)) for j in 0:N]       # N + 1 points, the last repeats the first
            g["Ex"] = v
            a = HDF5.attributes(g["Ex"])
            a["gridSpacing"] = [L / N]; a["gridGlobalOffset"] = [0.0]; a["position"] = [0.5]
        end
    end
    write(joinpath(rd, "container.txt"), "# smilei version: v5.1 test\n# threads: 2, cpus: 2\n# cpu model: Test CPU\n# start: 2026-11-03T10:00:00.000Z\n# end: 2026-11-03T10:00:07.500Z\n")
    done = summarize_scan(dir)
    @test done == ["tiny"]
    @test strip(read(joinpath(dir, "summary", "missing.txt"), String)) == "notrun"
    rec = JSON.parsefile(joinpath(dir, "summary", "tiny.json"))
    @test rec["wall_seconds"] == 7.5 && rec["cpu"] == "Test CPU" && rec["threads"] == "2" && rec["smilei"] == "v5.1 test"
    @test Float64.(rec["energy"]["Utot"]) == [1.5 + 1e-9i for i in 1:3]
    @test rec["dt"] == case["time"]["dt"] && rec["steps"] == round(Int, case["time"]["t_end"] / case["time"]["dt"])
    @test sort(collect(keys(rec["outputs"]))) == ["Fields0.h5", "container.txt", "scalars.txt"]
    @test Float64.(rec["series"]["growth_rate"]["U"]) == U
    c = complex.(Float64.(rec["series"]["mode"]["c_re"]), Float64.(rec["series"]["mode"]["c_im"]))
    @test c ≈ A rtol = 1e-12
    @test Float64.(rec["series"]["mode"]["U"]) ≈ abs2.(A) rtol = 1e-12
    @test rec["outputs"]["scalars.txt"] == "sha256:" * bytes2hex(open(sha256, joinpath(rd, "scalars.txt")))
    @test isfile(joinpath(dir, "summary", "tiny-manifest.toml"))
end

@testset "T-415 summarize projects the 2D dumps on B·ê of the case's mode [A5, WP-031]" begin
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
            HDF5.attributes(g)["time"] = it * case["time"]["dt"]
            HDF5.attributes(g)["dt"] = case["time"]["dt"]
            for (name, comp, st) in (("Bx", ê[1], [0.0, 0.5]), ("By", ê[2], [0.5, 0.0]))
                g[name] = permutedims(field(n, comp, st[1] == 0.5, st[2] == 0.5))
                a = HDF5.attributes(g[name])
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
    @test Float64.(s["U"]) ≈ abs2.(A) rtol = 1e-10
    @test Float64.(rec["series"]["growth_rate_energy"]["U"]) ≈ [4e-9, 7e-9] rtol = 1e-10
    @test sort(collect(keys(rec["series"]))) == ["growth_rate_energy", "growth_rate_mode"]
end

# The cases and summaries of a synthetic scan in `dir`: each point's series is the linear solution
# times e^{2δ t} up to the saturation (synthetic_run), with δ = δ∞ + a Δx² + b Δt², so that the
# residual, its order and its limit are known.
function synthetic_scan(dir, base, points; δ∞, a, b, tsat = 24.9)
    mkpath(joinpath(dir, "cases"))
    write(joinpath(dir, "scan.toml"), "name = \"synthetic\"\ncase = \"two-stream\"\n")
    θ = base["observables"]["growth_rate"]["theory"]
    for (name, kw) in points
        c = derive_case(base; name, kw...)
        open(io -> TOML.print(io, c; sorted = true), joinpath(dir, "cases", name * ".toml"), "w")
        dt, dx = c["time"]["dt"], c["grid"]["length"] / c["grid"]["cells"]
        t = collect((0:c["diagnostics"]["scalars_every"]:round(Int, 40 / dt)) .* dt)
        δ = (δ∞ + a * dx^2 + b * dt^2) * θ
        U = synthetic_run(c, "growth_rate", t; tsat, distort = s -> exp(2δ * s))
        rec = Dict("point" => name, "derived" => c["derived"], "wall_seconds" => 1.0, "threads" => "1", "cpu" => "test",
            "steps" => 1, "energy" => Dict("t" => t, "Utot" => fill(1.0, length(t))),
            "series" => Dict("growth_rate" => Dict("t" => t, "U" => U, "c_re" => nothing, "c_im" => nothing)))
        open(io -> JSON.print(io, rec), joinpath(dir, name * ".json"), "w")
    end
end

@testset "T-416 the report's residuals, families, orders and code comparison [A7, WP-031]" begin
    R = ConvergenceReport
    # order and Richardson limit of e = 1 + 3h²
    h = [1.0, 0.5, 0.25]
    @test R.observed_order(h, 1 .+ 3 .* h .^ 2) ≈ [2.0] rtol = 1e-12
    @test R.richardson(h, 1 .+ 3 .* h .^ 2, 2) ≈ 1 rtol = 1e-12
    # the families of both scans, from their prepared cases
    for (scanfile, want) in (
            ("two-stream-offset.toml", ["dx" => ["n0064", "n0128", "n0256", "n0512", "n1024"],
                "dt[refine=1//1]" => ["n0128", "n0128-dt2", "n0128-dt4", "n0128-dt8"], "dt[refine=4//1]" => ["n0512", "n0512-dt4"]]),
            ("weibel-2d-offset.toml", ["dx" => ["n064", "n128", "n256", "n512"],
                "dt[refine=1//1]" => ["n256", "n256-dt2"], "dt[refine=1//2]" => ["n128", "n128-dt2", "n128-dt4", "n128-dt8"]]))
        dir = prepare_scan(joinpath(CC, "scans", scanfile); runs = mktempdir())
        pts = map(readdir(joinpath(dir, "summary", "cases"))) do f
            c = TOML.parsefile(joinpath(dir, "summary", "cases", f))
            (; name = splitext(f)[1], derived = c["derived"], dt = c["time"]["dt"],
               dx = first(vcat(c["grid"]["length"])) / first(vcat(c["grid"]["cells"])))
        end
        got = [first(f) => [p.name for p in last(f)] for f in R.families(pts)]
        @test sort(got; by = first) == sort(want; by = first)
    end
    # report on a synthetic scan: the residual of every point is its δ/θ, the offset γ/θ − 1, the
    # Δx residuals (Δt ∝ Δx along the family, so both terms are of order Δx²) converge at order 2 to δ∞
    base = load_case("two-stream")
    θ = base["observables"]["growth_rate"]["theory"]
    pts = ["n0064" => (; refine = 1 // 2), "n0128" => (;), "n0256" => (; refine = 2), "n0512" => (; refine = 4),
           "n0128-dt2" => (; dt_divisor = 2), "n0128-dt4" => (; dt_divisor = 4), "n0128-a10" => (; amplitude_factor = 10)]
    δ∞, a, b = -0.004, 2.0, 30.0
    sd = joinpath(mktempdir(), "summary")
    synthetic_scan(sd, base, pts; δ∞, a, b)
    out = joinpath(dirname(sd), "out")
    res = R.report(sd, out)
    for (name, _) in pts
        c = TOML.parsefile(joinpath(sd, "cases", name * ".toml"))
        dt, dx = c["time"]["dt"], c["grid"]["length"] / c["grid"]["cells"]
        e = res["points"][name]["observables"]["growth_rate"]
        @test e["residual"] ≈ δ∞ + a * dx^2 + b * dt^2 rtol = 1e-7
        rec = JSON.parsefile(joinpath(sd, name * ".json"))
        t, U = Float64.(rec["series"]["growth_rate"]["t"]), Float64.(rec["series"]["growth_rate"]["U"])
        f = point_fits(c, "growth_rate", t, U; window = Tuple(c["observables"]["growth_rate"]["window"]))
        @test e["offset"] ≈ f.gamma / θ - 1 rtol = 1e-12
        @test e["offset_ivp"] ≈ f.gamma_ivp / θ - 1 rtol = 1e-12
    end
    d = res["convergence"]["dx_growth_rate"]
    @test d["points"] == ["n0064", "n0128", "n0256", "n0512"]
    @test d["observed_order_residual"][1] ≈ 2 rtol = 1e-5
    @test d["richardson_order"] == 2
    @test d["richardson_residual"] ≈ δ∞ rtol = 1e-6
    @test res["convergence"]["dt[refine=1//1]_growth_rate"]["points"] == ["n0128", "n0128-dt2", "n0128-dt4"]
    # the limit over all plain points (#281) recovers δ∞ of δ∞ + a Δx² + b Δt² exactly, the
    # amplitude point left out; the model without Δt² misses it, and the spread says so
    L = res["limit"]["growth_rate"]
    @test L["model"] == "dx2+dt+dt2" && sort(L["points"]) == ["n0064", "n0128", "n0128-dt2", "n0128-dt4", "n0256", "n0512"]
    # every point refitted over the samples common to all their windows (one count for all)
    @test L["samples"] >= 3 && L["common_window"][1] <= L["common_window"][2]
    want = map(L["points"]) do name
        c = TOML.parsefile(joinpath(sd, "cases", name * ".toml"))
        δ∞ + a * (c["grid"]["length"] / c["grid"]["cells"])^2 + b * c["time"]["dt"]^2
    end
    @test L["residual_common"] ≈ want rtol = 1e-7
    @test L["r_inf"] ≈ δ∞ rtol = 1e-7
    @test L["fits"]["dx2+dt+dt2"]["max_abs_misfit"] < 1e-10
    @test abs(L["fits"]["dx2+dt"]["r_inf"] - δ∞) > 1e-7 && L["r_inf_spread"] >= abs(L["fits"]["dx2+dt"]["r_inf"] - L["r_inf"])
    # the residual against the −Δt/2 start is recorded per point and is the fit against cold_ivp's
    # :t_minus_half solution over the same samples
    for (name, _) in pts
        c = TOML.parsefile(joinpath(sd, "cases", name * ".toml"))
        rec = JSON.parsefile(joinpath(sd, name * ".json"))
        t, U = Float64.(rec["series"]["growth_rate"]["t"]), Float64.(rec["series"]["growth_rate"]["U"])
        fh = point_fits(c, "growth_rate", t, U; window = Tuple(c["observables"]["growth_rate"]["window"]), momenta_at = :t_minus_half)
        e = res["points"][name]["observables"]["growth_rate"]
        @test e["residual_minus_half_start"] ≈ (fh.gamma - fh.gamma_ivp) / θ rtol = 1e-12
        @test e["residual_minus_half_start"] != e["residual"]
        @test e["gamma_ivp_minus_half_start"] == fh.gamma_ivp
    end
    # the limit fit itself: the coefficients of δ∞ + aΔx² + bΔt² in the model r∞ + aΔx² + bΔt + cΔt²,
    # every variant fitted (six plain points), the misfit of the model without Δt², the spread as
    # defined, and the −Δt/2 limit fitted to its own residuals
    @test L["fits"]["dx2+dt+dt2"]["coefficients"] ≈ [δ∞, a, 0, b] atol = 1e-9
    @test sort(collect(keys(L["fits"]))) == sort(["dx2+dt+dt2", "dx2+dt", "dx2+dt+dt2, without the coarsest", "dx2+dt, without the coarsest"])
    @test L["fits"]["dx2+dt, without the coarsest"]["points"] == 5 && L["fits"]["dx2+dt+dt2"]["points"] == 6
    @test L["fits"]["dx2+dt"]["max_abs_misfit"] > 1e-6
    @test L["r_inf_spread"] == maximum(abs(f["r_inf"] - L["r_inf"]) for f in values(L["fits"]))
    pl = [TOML.parsefile(joinpath(sd, "cases", n * ".toml")) for n in L["points"]]
    dxs = [c["grid"]["length"] / c["grid"]["cells"] for c in pl]
    dts = [c["time"]["dt"] for c in pl]
    @test L["minus_half_start"]["r_inf"] == R.limit_fit(dxs, dts, Float64.(L["residual_common_minus_half_start"]))["r_inf"]
    @test L["minus_half_start"]["r_inf"] != L["r_inf"]
    # independently of limit_fit: least squares without the coarsest point by hand
    keep = dxs .< maximum(dxs)
    A = [ones(count(keep)) dxs[keep] .^ 2 dts[keep]]
    @test L["fits"]["dx2+dt, without the coarsest"]["r_inf"] ≈ (A \ Float64.(L["residual_common"])[keep])[1] rtol = 1e-12
    @test all(isfile, joinpath.(out, ["results.json", "manifest.toml", "offset-vs-resolution.png", "local-rate.png", "ratio-to-linear.png"]))
    # a plain point without a fit (here a flat series, which never enters the window) leaves the
    # observable out of the limit rather than fitting the others
    sdx = joinpath(mktempdir(), "summary")
    cp(sd, sdx)
    rec = JSON.parsefile(joinpath(sdx, "n0128-dt4.json"))
    rec["series"]["growth_rate"]["U"] = fill(1.0, length(rec["series"]["growth_rate"]["U"]))
    open(io -> JSON.print(io, rec), joinpath(sdx, "n0128-dt4.json"), "w")
    resx = R.report(sdx, joinpath(dirname(sdx), "out"))
    @test haskey(resx["points"]["n0128-dt4"]["observables"]["growth_rate"], "flag") && !haskey(resx["limit"], "growth_rate")
    # compare_codes: the same summary against itself, and against a copy grown by e^{2δt}
    cmp = R.compare_codes(sd, sd, joinpath(dirname(sd), "cmp1"))
    for (_, v) in cmp["points"]
        @test v["relative_difference_same_samples"] == 0 && v["ratio_window_start"] == 1 && v["ratio_window_end"] == 1
    end
    # the summary directories are recorded as the repository names them, never as absolute paths:
    # relative to the repository inside it, the directory's name outside it
    @test cmp["reference"] == "summary" && cmp["other"] == "summary"
    mktempdir(Convergence.ROOT) do inside
        sdi = joinpath(inside, "summary")
        cp(sd, sdi)
        c = R.compare_codes(abspath(sdi), sd, joinpath(inside, "cmp"))
        @test c["reference"] == relpath(sdi, Convergence.ROOT) && c["other"] == "summary"
        @test !any(isabspath, (c["reference"], c["other"]))
    end
    sd2 =joinpath(mktempdir(), "summary")
    synthetic_scan(sd2, base, pts; δ∞, a, b)
    δc = 1e-4
    for (name, _) in pts
        rec = JSON.parsefile(joinpath(sd2, name * ".json"))
        s = rec["series"]["growth_rate"]
        t = Float64.(s["t"])
        s["U"] = Float64.(s["U"]) .* exp.(2δc .* t)
        open(io -> JSON.print(io, rec), joinpath(sd2, name * ".json"), "w")
    end
    cmp = R.compare_codes(sd, sd2, joinpath(dirname(sd2), "cmp2"))
    for (_, v) in cmp["points"]
        @test v["relative_difference_same_samples"] ≈ δc / v["gamma_reference"] rtol = 1e-8
        @test v["ratio_t0"] == 1
    end
    # its run manifest (F3) hashes both summaries' points and cases
    m = TOML.parsefile(joinpath(dirname(sd2), "cmp2", "manifest.toml"))
    @test length(m["inputs"]) == 4length(pts) && m["points"] == sort(first.(pts))
    # an other run that stops inside the reference's window is refused with a message
    rec = JSON.parsefile(joinpath(sd2, "n0128.json"))
    s = rec["series"]["growth_rate"]
    s["t"], s["U"] = s["t"][1:100], s["U"][1:100]
    open(io -> JSON.print(io, rec), joinpath(sd2, "n0128.json"), "w")
    @test_throws "ends before the reference's saturation" R.compare_codes(sd, sd2, joinpath(dirname(sd2), "cmp3"))
end

# The ParticleMesh scan driver (convergence_pm.jl), in a module of its own like PMCross in runtests.jl.
module PMScan
include(joinpath(dirname(@__DIR__), "convergence_pm.jl"))
end

@testset "T-417 the ParticleMesh scan driver writes the record format of summarize_scan [A5, WP-031]" begin
    # a tiny point: 32 cells, 10 particles per cell, a scalar cadence of 2 steps, 1/ω₀ of time
    base = load_case("two-stream")
    case = derive_case(base; name = "tiny", refine = 1 // 4, dt_divisor = 2, ppc_factor = 1 // 100)
    case["time"]["t_end"] = 1.0
    @test case["diagnostics"]["scalars_every"] == 2 && case["diagnostics"]["fields_every"] == 16
    root = mktempdir()
    sumdir = joinpath(root, "summary")
    mkpath(joinpath(sumdir, "cases"))
    open(io -> TOML.print(io, case; sorted = true), joinpath(sumdir, "cases", "tiny.toml"), "w")
    write(joinpath(sumdir, "scan.toml"), "name = \"tiny-scan\"\ncase = \"two-stream\"\n")
    out = joinpath(root, "pm")
    PMScan.run_points(sumdir, out, ["tiny"])
    @test isfile(joinpath(out, "scan.toml")) && isfile(joinpath(out, "cases", "tiny.toml")) && isfile(joinpath(out, "manifest.toml"))
    m = TOML.parsefile(joinpath(out, "manifest.toml"))
    @test m["points"] == ["tiny"] && m["mode"] == "Reference" && m["precision"] == "Float64"
    @test !any(isabspath, keys(m["inputs"]))
    rec = JSON.parsefile(joinpath(out, "tiny.json"))
    # the keys of a point of summarize_scan (T-414), less the ones only a Smilei run has, plus the
    # time of the loaded momenta (#282), in the record and in the manifest
    @test sort(collect(keys(rec))) == sort(["point", "case", "derived", "code", "dt", "cells", "steps", "wall_seconds",
                                            "threads", "cpu", "energy", "series", "momenta_at", "density_at"])
    @test rec["momenta_at"] == m["momenta_at"] == string(PMScan.MOMENTA_AT) == "t_minus_half"
    @test rec["density_at"] == m["density_at"] == string(PMScan.DENSITY_AT) == "cell_centre"
    @test rec["point"] == "tiny" && rec["code"] == "particlemesh" && rec["case"] == case["name"] && rec["derived"] == case["derived"]
    @test sort(collect(keys(rec["series"]))) == ["growth_rate", "mode"]
    dt = case["time"]["dt"]
    @test rec["dt"] == dt && rec["cells"] == 32 && rec["steps"] == round(Int, 1.0 / dt)
    # the run again, directly: the driver's series are the run's, at the cadences of the case (the
    # Reference mode is bitwise reproducible, A4), so an offset of the energy cadence or a |c| in place of |c|² shows
    sim = PMScan.simulate(case)
    @test sim.nsteps == rec["steps"]
    every = case["diagnostics"]["scalars_every"]
    iE = 1:every:length(sim.total)
    en = rec["energy"]
    @test Float64.(en["t"]) == (iE .- 1) .* dt && Float64.(en["Utot"]) == sim.total[iE]
    @test length(en["t"]) == cld(sim.nsteps, every)
    @test en["t"][2] - en["t"][1] ≈ every * dt rtol = 1e-12
    @test sim.total[1] != sim.total[2]                     # the cadence offset is visible in the energy
    g = rec["series"]["growth_rate"]
    @test Float64.(g["t"]) == sim.tU && Float64.(g["U"]) == sim.U && g["c_re"] === nothing && g["c_im"] === nothing
    md = rec["series"]["mode"]
    c = complex.(Float64.(md["c_re"]), Float64.(md["c_im"]))
    @test c == sim.C && Float64.(md["t"]) == sim.tC
    @test Float64.(md["U"]) == abs2.(sim.C) && any(>(0), md["U"])
    @test rec["wall_seconds"] > 0 && rec["threads"] == string(Threads.nthreads())
end
