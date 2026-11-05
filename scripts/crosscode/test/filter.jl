# Tests of the matched binomial current filter in the cross-code tooling (WP-038). Included by
# runtests.jl after picongpu.jl, which loads SmileiDeck, PIConGPUDeck and Observables.
using Random: Xoshiro

# The Smilei namelist of a case with its filter pass count replaced.
with_passes(case, p) = (c = deepcopy(case); c["numerics"]["filter_passes"] = p; c)

@testset "T-550 Smilei: the case's pass count becomes CurrentFilter, refused where Smilei cannot match it [A2, A5, WP-038]" begin
    for name in ("landau-oblique-2d-filter1", "landau-oblique-2d-filter2")
        case = load_case(name)
        p = case["numerics"]["filter_passes"]
        text = namelist(case)
        @test smilei_parameters(case).filter_passes == p
        # one block, binomial, the count along every axis (Params.cpp copies one entry to all axes)
        @test count("CurrentFilter(", text) == 1
        m = match(r"CurrentFilter\(\s*model = \"(\w+)\",\s*passes = \[(\d+)\],\s*\)", text)
        @test m !== nothing && m.captures[1] == "binomial" && parse(Int, m.captures[2]) == p
        # the filter is the only difference from the same case unfiltered, whose namelist has none
        plain = namelist(with_passes(case, 0))
        @test !occursin("CurrentFilter", plain)
        @test replace(text, SmileiDeck.current_filter_block(smilei_parameters(case)) => "") == plain
    end
    # the committed cases keep their namelists (zero passes write no block; T-270 checks the hashes)
    for name in ("landau", "two-stream", "two-stream-256", "landau-oblique-2d", "weibel-2d")
        @test !occursin("CurrentFilter", namelist(load_case(name)))
    end
    # schema 1 takes the same block
    l1 = with_passes(load_case("landau"), 3)
    @test occursin("CurrentFilter(\n    model = \"binomial\",\n    passes = [3],\n)", namelist(l1))
    # refusals: Smilei's filter has no GPU code path; counts must be non-negative integers
    f1 = load_case("landau-oblique-2d-filter1")
    @test_throws ErrorException namelist(f1; gpu = true)
    @test !occursin("CurrentFilter", namelist(with_passes(f1, 0); gpu = true))
    @test_throws ErrorException smilei_parameters(with_passes(f1, -1))
    @test_throws ErrorException smilei_parameters(with_passes(f1, 1.5))
    @test_throws ErrorException smilei_parameters(with_passes(load_case("landau"), -2))
end

@testset "T-551 PIConGPU: one pass becomes the binomial current interpolation, more are refused [A2, A5, WP-038]" begin
    f1 = load_case("landau-oblique-2d-filter1")
    p1 = picongpu_parameters(f1)
    @test p1.filter_passes == 1
    args = run_arguments(p1)
    @test occursin("--currentInterpolation binomial", args) && !occursin("--currentInterpolation none", args)
    @test occursin("--currentInterpolation none", run_arguments(picongpu_parameters(with_passes(f1, 0))))
    # PIConGPU refuses binomial with an activated current background; the generator never activates it
    @test occursin("static constexpr bool activated = false;",
                   match(r"class FieldBackgroundJ.*?};"s, param_files(f1)["fieldBackground.param"]).match)
    # a schema-1 case with one pass is matched too (the quasi-1D box filters k_y = 0 by 1 along y)
    @test occursin("--currentInterpolation binomial", run_arguments(picongpu_parameters(with_passes(load_case("landau"), 1))))
    for p in (2, 3)
        err = try picongpu_parameters(with_passes(f1, p)); nothing catch e; e end
        @test err isa ErrorException && occursin("one fixed pass", err.msg)
    end
    @test_throws ErrorException picongpu_parameters(load_case("landau-oblique-2d-filter2"))
    @test_throws ErrorException picongpu_parameters(with_passes(f1, -1))
end

# Fields that keep the filtered Gauss law exactly: for every mode, the E change of a node-centred
# divergence with E points at i + 1 − s and i − s (`gauss_residual_nodes`) solves
# D⁻·ΔE = F Δρ in Fourier space, with F `passes` binomial passes, plus a residual r⁰ at the start.
function filtered_fields(rng, N, h, shift, passes)
    ρ0 = randn(rng, N...)
    Δρ = 0.1 .* randn(rng, N...)
    Δρ .-= sum(Δρ) / length(Δρ)                    # charge is conserved
    ρ1 = ρ0 .+ Δρ
    E0x, E0y = randn(rng, N...), randn(rng, N...)
    FΔρ = dft2(Δρ) .* binomial_transfer_grid(N, passes)
    θ(m, n) = 2π * (m - 1) / n
    sym(m, n, d, s) = (cis(θ(m, n) * (1 - s)) - cis(-θ(m, n) * s)) / h[d]    # symbol of the difference
    ΔÊx, ΔÊy = zeros(ComplexF64, N...), zeros(ComplexF64, N...)
    for m1 in 1:N[1], m2 in 1:N[2]
        a, b = sym(m1, N[1], 1, shift[1]), sym(m2, N[2], 2, shift[2])
        den = abs2(a) + abs2(b)
        den == 0 && continue
        ΔÊx[m1, m2] = conj(a) * FΔρ[m1, m2] / den
        ΔÊy[m1, m2] = conj(b) * FΔρ[m1, m2] / den
    end
    idft(Â) = real.(conj.(dft2(conj.(Â))) .* length(Â))
    return (; ρ0, ρ1, E0 = (E0x, E0y), E1 = (E0x .+ idft(ΔÊx), E0y .+ idft(ΔÊy)))
end

@testset "T-552 the filtered Gauss check is roundoff for the run's pass count and not for its neighbours [A2, A7, WP-038]" begin
    rng = Xoshiro(38)
    N, h = (16, 12), (0.3, 0.25)
    @test binomial_transfer_grid(N, 0) == ones(N...)
    H = binomial_transfer_grid(N, 2)
    @test H[1, 1] == 1 && all(abs.(H[N[1] ÷ 2 + 1, :]) .< 1e-30) && H[2, 3] ≈ cospi(1 / 16)^4 * cospi(2 / 12)^4
    for shift in ((0, 0), (1, 1), (0, 1)), passes in (0, 1, 2)
        f = filtered_fields(rng, N, h, shift, passes)
        r0 = gauss_residual_nodes(f.E0..., f.ρ0, h, shift)
        r1 = gauss_residual_nodes(f.E1..., f.ρ1, h, shift)
        g = filtered_gauss(r0, f.ρ0, r1, f.ρ1, passes)
        @test g.deviation < 1e-13
        @test all(>(1e-2), values(g.deviation_other))
        @test sort(collect(keys(g.deviation_other))) == filter(>=(0), [passes - 1, passes + 1])
        @test maximum(abs, g.ratio .- g.one_minus_H) < 1e-10
        # the wrong stagger is seen: the same fields read with the other shift on axis 1
        wrong = (1 - shift[1], shift[2])
        gw = filtered_gauss(gauss_residual_nodes(f.E0..., f.ρ0, h, wrong), f.ρ0,
                            gauss_residual_nodes(f.E1..., f.ρ1, h, wrong), f.ρ1, passes)
        @test gw.deviation > 1e-2
    end
end

@testset "T-553 the filtered Gauss series reads each code's staggers from the dump positions [A5, WP-038]" begin
    rng = Xoshiro(553)
    N, L = (16, 12), (4.8, 3.0)
    h = L ./ N
    case = Dict("grid" => Dict("cells" => collect(N), "length" => collect(L)))
    # with the periodic duplicate the readers append, at the positions each reader reports:
    # Smilei's dual point i at (i − ½)Δ (s = 0), PIConGPU's E at (i + ½)Δ (s = 1); ρ at the nodes
    dup(a) = (b = vcat(a, a[1:1, :]); hcat(b, b[:, 1:1]))
    nodes(d) = [(i - 1) * h[d] for i in 1:N[d] + 1]
    for (s, off) in ((0, -0.5), (1, 0.5))
        f = filtered_fields(rng, N, h, (s, s), 1)
        x = Dict("Rho" => (nodes(1), nodes(2)), "Ex" => (nodes(1) .+ off * h[1], nodes(2)),
                 "Ey" => (nodes(1), nodes(2) .+ off * h[2]))
        fields = Dict("Rho" => cat(dup(f.ρ0), dup(f.ρ1); dims = 3),
                      "Ex" => cat(dup(f.E0[1]), dup(f.E1[1]); dims = 3), "Ey" => cat(dup(f.E0[2]), dup(f.E1[2]); dims = 3))
        ser = filtered_gauss_series((; fields, x, time = [0.0, 1.0]), case)
        @test length(ser.r) == 2 && ser.t == [0.0, 1.0]
        @test filtered_gauss(ser.r[1], ser.ρ[1], ser.r[2], ser.ρ[2], 1).deviation < 1e-13
    end
end
