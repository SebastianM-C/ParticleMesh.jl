# WP-021, the periodic spectral inverse of the discrete Laplacian and the forward-difference
# gradient at N = 1 and N = 2 on the CPU backend: T-310 to T-316.

const INPROCESS_SPECTRAL = spectral_digests()
spectral_worker(nt) = [chopprefix(l, "SPECTRAL ") for l in worker_lines(nt) if startswith(l, "SPECTRAL ")]

"T-310's grids: N = 2 at three sizes (non-square, Δx ≠ Δy) and N = 1 at two, so that the ratio's growth is visible."
const T310_GRIDS_2D = ((16, 12), (64, 48), (128, 96))
const T310_GRIDS_1D = (64, 2 * ParticleMeshCore.SCAN_TILE + 37)

@testset "T-310 the divergence of the gradient of the solve is s - mean(s) at every node [A7, A8, WP-021]" begin
    # Scale: (Σ_d n_d)·(max|s| + ‖L‖·max|φ|) with ‖L‖ = Σ_d 4/Δ_d², the size of the residual that
    # an error of eps·Σ_d n_d relative to φ (the transforms' sums of n_d terms) leaves after the
    # Laplacian, plus the same relative error of s.
    C, Cm = tolC("T-310"), tolC("T-310", "C_mean")
    for T in TYPES, grid in (map(n -> grid2(T, 2; n), T310_GRIDS_2D)..., map(n -> PeriodicGrid(n, basetype(T)(3 // 8), 2), T310_GRIDS_1D)...)
        ε = base_eps(T)
        s = solve_input(T, grid)
        φ, G = solved(s, grid)
        φmax = maximum(absmax, interior_of(φ, grid))
        smax = maximum(absmax, interior_of(s, grid))
        nsum = sum(grid.n)
        r = maximum(absmax, divergence_residual(G, s, grid)) / (ε * nsum * (smax + laplacian_norm(grid) * φmax))
        record!("T-310", T, r)
        @test r <= C
        # the mean of φ is zero: the k = 0 mode is set to zero
        rm = absmax(sum(interior_of(φ, grid)) / prod(grid.n)) / (ε * nsum * φmax)
        record!("T-310 mean", T, rm)
        @test rm <= Cm
        # outputs have their guards filled; the NaN guards of s were not read
        @test guards_are_images(φ, grid) && all(Gd -> guards_are_images(Gd, grid), G)
        @test all(x -> all(isfinite, components(x)), φ) && all(Gd -> all(x -> all(isfinite, components(x)), Gd), G)
        T === D3 && @test any(x -> !iszero(partials(x)), φ)               # the partials are exercised
    end
end

"`2 sin(π m/n)/Δ` in BigFloat (the sign of `m` kept), one axis of the Yee-consistent symbol."
yee_axis(m, n, Δ) = 2 * sinpi(BigFloat(m // n)) / BigFloat(Δ)

@testset "T-311 an oblique Fourier mode against the closed form with the Yee-consistent symbol [A7, A8, WP-021]" begin
    # s = A·cos θ_j, θ_j = 2π(m₁(j₁-1)/n₁ + m₂(j₂-1)/n₂), on 16 × 12 cells of 1/4 × 3/8 (non-square,
    # Δx ≠ Δy). Closed form of the discrete problem: φ = -s/K² with K² = Σ_d (2 sin(π m_d/n_d)/Δ_d)²,
    # and G₁[j] = (φ[j + e₁] - φ[j])/Δ₁ = 2 sin(π m₁/n₁)/(Δ₁K²)·A·sin(θ_j + π m₁/n₁), likewise G₂.
    # References in BigFloat; scale (n₁ + n₂)·max|reference| per output, per value and partial.
    C = tolC("T-311")
    for T in TYPES, (m1, m2) in ((1, 2), (3, -5))
        ε = base_eps(T)
        grid = grid2(T, 2)
        (n1, n2), (h1, h2), g = grid.n, grid.dx, guards(grid)
        A = seeded(T, 1)
        cA = Float64.(components(A))
        K2 = yee_axis(m1, n1, h1)^2 + yee_axis(m2, n2, h2)^2
        r(j1, j2) = 2 * (m1 * (j1 - 1) // n1 + m2 * (j2 - 1) // n2)          # θ/π, exact
        s = field(backend, T, grid)
        for j2 in 1:n2, j1 in 1:n1
            s[g + j1, g + j2] = A * basetype(T)(cospi(BigFloat(r(j1, j2))))
        end
        φ, G = solved(s, grid)
        φref = [Float64(-cospi(BigFloat(r(j1, j2))) / K2) for j1 in 1:n1, j2 in 1:n2]
        G1ref = [Float64(yee_axis(m1, n1, h1) / K2 * sinpi(BigFloat(r(j1, j2) + m1 // n1))) for j1 in 1:n1, j2 in 1:n2]
        G2ref = [Float64(yee_axis(m2, n2, h2) / K2 * sinpi(BigFloat(r(j1, j2) + m2 // n2))) for j1 in 1:n1, j2 in 1:n2]
        for (out, ref) in ((φ, φref), (G[1], G1ref), (G[2], G2ref))
            oi = interior_of(out, grid)
            err = maximum(maximum(abs.(Float64.(components(oi[j])) .- cA .* ref[j])) for j in eachindex(ref))
            rr = err / (ε * (n1 + n2) * maximum(cA) * maximum(abs, ref))
            record!("T-311", T, rr)
            @test rr <= C
        end
    end
end

@testset "T-312 the periodic solve is bitwise across reruns, thread counts and launch shapes [A4, WP-021]" begin
    @test spectral_digests() == INPROCESS_SPECTRAL                      # rerun in process
    groups = Dict{String,Vector{String}}()
    for l in INPROCESS_SPECTRAL
        f = split(l, '|')
        push!(get!(groups, join(f[1:3], '|'), String[]), f[5])
    end
    @test length(groups) == length(TYPES) * (length(SOLVE_GRIDS_1D) + length(SOLVE_GRIDS_2D))
    @test all(v -> length(v) == length(WG_SIZES) && allequal(v), values(groups))   # every launch shape
    for nt in (1, 4)
        @test spectral_worker(nt) == INPROCESS_SPECTRAL                 # 1 and 4 threads
    end
    # A reused workspace and an in-place solve (φ === s) give the same bits as a fresh one.
    for T in TYPES, grid in (grid2(T, 2), PeriodicGrid(64, basetype(T)(3 // 8), 2))
        s = solve_input(T, grid)
        φ, G = solved(s, grid)
        ws = SpectralWorkspace(backend, T, grid)
        solved(solve_input(T, grid), grid; ws)
        φ2, G2 = solved(s, grid; ws)
        @test same_bits(φ, φ2) && all(same_bits.(G, G2))
        s3 = copy(s)
        inverse_laplacian_periodic!(s3, s3, grid, ws)
        @test same_bits(s3, φ)
        @test same_bits(s, solve_input(T, grid))                         # s is not written
    end
end

@testset "T-313 at N = 1 the gradient of the solve agrees with cumsum_periodic! [A7, A8, WP-021]" begin
    # Both are zero-mean periodic antiderivatives of a - mean(a) with the same staggering
    # (`cumsum_periodic!`'s out[j] - out[j-1] and the gradient's G[j] - G[j-1] are backward
    # differences), so they agree up to the roundoff of each. Scale n·(max|a|·Δ + ‖L‖·Δ·max|φ|):
    # T-172's n·max|a|·Δ for the prefix sum, and for the gradient the error eps·n relative to φ
    # differenced over a cell.
    C = tolC("T-313")
    for T in TYPES, n in SCAN_GRIDS
        ε = base_eps(T)
        grid, a = scan_input(T, n)
        out = cumsum_periodic!(field(backend, T, grid), a, grid)
        φ, G = solved(a, grid)
        dx = cellsize(grid)
        amax = maximum(absmax, interior_of(a, grid))
        φmax = maximum(absmax, interior_of(φ, grid))
        scale = n * (amax * dx + laplacian_norm(grid) * dx * φmax)
        r = maximum(absdiff.(G[1], out)) / (ε * scale)                  # guards included
        record!("T-313", T, r)
        @test r <= C
        T === D3 && @test any(x -> !iszero(partials(x)), G[1])
    end
end

@testset "T-314 Dual: partials of parameters in s are the solve of the partials; the primal agrees with Float32 [A4, A8, WP-021]" begin
    # s = p₁·f₁ + p₂·f₂ + p₃ with three Dual parameters seeded along the three partials: f₁ smooth,
    # f₂ of high frequency, and p₃ a constant, whose solve is zero (the mean is removed). The solve
    # is linear in s, so partial i of (φ, G) is the Float32 solve of partial i of s, to roundoff;
    # the primal is the Float32 run within a measured tolerance, not bitwise by contract (A4: the
    # compiler may fuse differently on the two paths). Whether it happened to be bitwise is
    # recorded. Scale (Σ_d n_d)·max|reference| per output and component.
    C = tolC("T-314")
    bitwise = true
    for grid in (grid2(D3, 2), grid2(D3, 2; n = (64, 48)), PeriodicGrid(64, 3.0f0 / 8, 2))
        p = (mkdual(7 / 10, 1, 0, 0), mkdual(13 / 10, 0, 1, 0), mkdual(1 / 2, 0, 0, 1))
        s = fill!(field(backend, D3, grid), D3(NaN))
        for J in CartesianIndices(interior_view(s, grid))
            j = Tuple(J)
            f1 = Float32(sin(0.37j[1]) + cos(1.1j[end]) / 5)
            f2 = Float32(sin(0.9 * prod(j)) / 2)
            s[(j .+ guards(grid))...] = p[1] * f1 + p[2] * f2 + p[3]
        end
        φ, G = solved(s, grid)
        nsum = sum(grid.n)
        for c in 1:4
            comp(x) = components(x)[c]
            sc = map(comp, s)
            φc, Gc = solved(sc, grid)
            for (dual_out, ref) in ((φ, φc), zip(G, Gc)...)
                r = maximum(abs.(map(comp, dual_out) .- ref)) / (eps(Float32) * nsum * max(maximum(abs, ref), floatmin(Float32)))
                if c == 1
                    record!("T-314 primal", D3, r)
                    bitwise &= same_bits(map(comp, dual_out), ref)
                else
                    record!("T-314 partials", D3, r)
                end
                @test r <= C
            end
        end
        # the constant parameter p₃ leaves φ and G unchanged: its partial is at roundoff of zero
        @test maximum(x -> abs(partials(x)[3]), φ) <= C * eps(Float32) * nsum * maximum(x -> abs(partials(x)[1]), φ)
    end
    get!(MEASURED, "T-314", Dict{String,Float64}())["primal bitwise with Float32 (1 = yes)"] = bitwise
    @info "T-314: the Dual primal is $(bitwise ? "" : "not ")bitwise the Float32 run (recorded, not asserted; A4)"
end

@testset "T-315 allocation of the periodic solve does not grow with the grid size [A7, WP-021]" begin
    # Bytes per call after warm-up at two grid sizes per N, as T-144 and T-268 measure them: a
    # constant per launch on the CPU backend, no growth with the grid.
    for T in (Float32, Float64), sizes in (((16, 12), (32, 24)), (64, 256))
        bytes = Dict{Tuple{String,Any},Int}()
        for n in sizes
            grid = n isa Tuple ? grid2(T, 2; n) : PeriodicGrid(n, T(3 // 8), 2)
            s = solve_input(T, grid)
            ws = SpectralWorkspace(backend, T, grid)
            φ = field(backend, T, grid)
            G = ntuple(_ -> field(backend, T, grid), length(grid.n))
            ops = (solve = () -> inverse_laplacian_periodic!(φ, s, grid, ws), gradient = () -> gradient_periodic!(G, φ, grid))
            for (name, op) in pairs(ops)
                op(); op()
                bytes[(string(name), n)] = @allocated op()
            end
        end
        @info "T-315 bytes per call" T bytes
        for name in ("solve", "gradient")
            @test bytes[(name, sizes[2])] <= bytes[(name, sizes[1])]
        end
    end
end

@testset "T-316 the periodic solve checks its inputs [A7, WP-021]" begin
    grid = grid2(Float64, 2)
    s = solve_input(Float64, grid)
    φ = field(backend, Float64, grid)
    ws = SpectralWorkspace(backend, Float64, grid)
    @test_throws ArgumentError inverse_laplacian_periodic!(φ, s, grid, SpectralWorkspace(backend, Float64, grid2(Float64, 2; n = (12, 16))))
    @test_throws ArgumentError inverse_laplacian_periodic!(φ, s, grid, SpectralWorkspace(backend, Float32, grid))
    @test_throws ArgumentError inverse_laplacian_periodic!(φ, s[1:(end - 1), :], grid, ws)
    @test_throws ArgumentError inverse_laplacian_periodic!(φ[1:(end - 1), :], s, grid, ws)
    g0 = PeriodicGrid((16, 12), (0.25, 0.375), 0)
    @test_throws ArgumentError inverse_laplacian_periodic!(field(backend, Float64, g0), field(backend, Float64, g0), g0,
                                                           SpectralWorkspace(backend, Float64, g0))
    G = (field(backend, Float64, grid), field(backend, Float64, grid))
    @test_throws ArgumentError gradient_periodic!((φ, G[2]), φ, grid)
    @test_throws ArgumentError gradient_periodic!(G, φ[:, 1:(end - 1)], grid)
    @test_throws ArgumentError gradient_periodic!((G[1][1:(end - 1), :], G[2]), φ, grid)
    @test_throws ArgumentError gradient_periodic!((G[1], G[2][:, 1:(end - 1)]), φ, grid)
    @test_throws ArgumentError SpectralWorkspace(backend, Float64, PeriodicGrid((4, 4, 4), (1.0, 1.0, 1.0), 1))
end
