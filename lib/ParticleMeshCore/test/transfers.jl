# Shapes, gather, deposition, guards: T-130 to T-135 and T-140.

@testset "T-130 partition of unity of the shape weights [A7, A8, WP-010]" begin
    C = tolC("T-130")
    for T in TYPES, shape in SHAPES, stagger in STAGGERS
        ε = base_eps(T)
        worst = 0.0
        for δ0 in (0.0, 0.25, 0.5, 0.7, Float64(prevfloat(one(basetype(T)))), (k / 97 for k in 0:96)...)
            δ = seeded(T, δ0)
            _, w = shape_weights(shape, stagger, δ)
            worst = max(worst, absdiff(sum(w), one(T)) / ε)
            @test all(x -> value_of(x) >= 0, w)
        end
        record!("T-130", T, worst)
        @test worst <= C
    end
end

@testset "T-131 gather reproduces constant and linear fields [A7, A8, WP-010]" begin
    # Fill the whole array, guards included, with a + b·x at the node positions: no wrap, so
    # every particle is "away from the seam" for the polynomial.
    C = tolC("T-131")
    for T in TYPES, shape in SHAPES, stagger in STAGGERS
        ε = base_eps(T)
        grid = PeriodicGrid(16, basetype(T)(1 // 4), guard_width(shape))
        s = test_species(T, grid, 200; seed = 3)
        a0, b0 = basetype(T)(3 // 2), basetype(T)(-5 // 4)
        sv = ParticleMeshCore.stagger_offset(stagger, basetype(T))
        g = guards(grid)
        xnode(idx) = (idx - g - 1 + sv) * cellsize(grid)
        f = field(backend, T, grid)
        worst = 0.0
        for (a, b) in ((a0, zero(a0)), (a0, b0))
            for idx in eachindex(f)
                f[idx] = a + b * xnode(idx)
            end
            out = similar(s.w)
            gather!(out, f, s, shape, stagger, grid)
            for i in 1:200
                x = (s.cell[1][i] - 1 + s.offset[1][i]) * cellsize(grid)
                exact = a + b * x
                worst = max(worst, absdiff(out[i], exact) / (ε * (abs(a) + abs(b) * length(grid))))
            end
        end
        record!("T-131", T, worst)
        @test worst <= C
    end
end

@testset "T-132 deposition is the adjoint of gather with fill and fold [A7, A8, WP-010]" begin
    # Σᵢ qᵢwᵢ gather(fill f)ᵢ = ⟨fold(deposit(q w)), f⟩ over the interior.
    C = tolC("T-132")
    for T in TYPES, shape in SHAPES, stagger in STAGGERS, mode in (Reference(), Fast())
        ε = base_eps(T)
        grid = PeriodicGrid(12, basetype(T)(1 // 2), guard_width(shape))
        s = test_species(T, grid, 300; seed = 5)
        f = test_field(T, grid, j -> seeded(T, sin(0.7j) + 0.3))
        out = similar(s.w)
        gather!(out, f, s, shape, stagger, grid)
        lhs = sum(s.q * s.w[i] * out[i] for i in 1:300)
        ρ = fold_guards!(deposited(mode, s, shape, stagger, grid), grid)
        rhs = sum(ρ[guards(grid) + j] * f[guards(grid) + j] for j in 1:ncells(grid))
        scale = sum(absmax(s.w[i]) for i in 1:300) * maximum(absmax, f)
        r = record!("T-132", T, absdiff(lhs, rhs) / (ε * scale))
        @test r <= C
    end
end

@testset "T-133 fill and fold are adjoints [A7, A8, WP-010]" begin
    C = tolC("T-133")
    for T in TYPES, g in (1, 2), n in (4, 9)
        ε = base_eps(T)
        grid = PeriodicGrid(n, one(basetype(T)), g)
        x = field(backend, T, grid)
        y = field(backend, T, grid)
        for i in eachindex(x)
            x[i] = seeded(T, cos(1.3i))
            y[i] = seeded(T, sin(2.1i) + 0.1)
        end
        fx = fill_guards!(copy(x), grid)
        fy = fold_guards!(copy(y), grid)
        lhs = sum(fx .* y)
        rhs = sum(x .* fy)
        scale = sum(absmax.(x)) * maximum(absmax, y)
        r = record!("T-133", T, absdiff(lhs, rhs) / (ε * scale))
        @test r <= C
        @test all(iszero, fy[1:g]) && all(iszero, fy[(end - g + 1):end])
    end
end

@testset "T-134 deposited total equals the sum of q·w [A7, A8, WP-010]" begin
    C = tolC("T-134")
    for T in TYPES, shape in SHAPES, stagger in STAGGERS, mode in (Reference(), Fast()), np in (100, 2000)
        ε = base_eps(T)
        grid = PeriodicGrid(10, one(basetype(T)), guard_width(shape))
        s = test_species(T, grid, np; seed = 7)
        ρ = fold_guards!(deposited(mode, s, shape, stagger, grid), grid)
        total = sum(ρ)
        expect = sum(s.q * s.w[i] for i in 1:np)
        r = record!("T-134", T, absdiff(total, expect) / (ε * np * maximum(i -> absmax(s.w[i]), 1:np)))
        @test r <= C
    end
end

@testset "T-135 Reference and Fast deposition agree per node [A4, A7, WP-010]" begin
    # Bound per node: C·eps(T)·Σ|contributions to the node|. Every test weight is positive and the
    # species has one sign of charge, so the sum of magnitudes is |ρR| and the code divides by it.
    C = tolC("T-135")
    for T in (Float32, Float64), shape in SHAPES, stagger in STAGGERS
        ε = eps(T)
        grid = PeriodicGrid(16, one(T), guard_width(shape))
        s = test_species(T, grid, 16 * 400; seed = 11)
        ρF = deposited(Fast(), s, shape, stagger, grid)
        ρR = deposited(Reference(), s, shape, stagger, grid)
        worst = maximum(abs(ρF[a] - ρR[a]) / (ε * abs(ρR[a])) for a in eachindex(ρR) if ρR[a] != 0)
        record!("T-135", T, worst)
        @test worst <= C
    end
end

@testset "T-140 Dual weights: exact partials, primal values bitwise [A3, A4, A8, WP-010]" begin
    # Weights from the density callable with Dual parameters (n₀, α, k) seeded along the first
    # two directions. Deposition is linear in the weights, so ∂ρ/∂α = deposit(n₀ cos(kx)/ppc) and
    # ∂ρ/∂n₀ = deposit((1 + α cos kx)/ppc); both computed independently in Float64.
    C = tolC("T-140")
    ppc = 37
    for shape in SHAPES
        grid = PeriodicGrid(24, 1.0f0 / 3, guard_width(shape))
        stagger = natural_stagger(shape)
        n0, α, k = 1.0f0, 0.05f0, 2.0f0 * Float32(π) / length(grid)
        pD = (mkdual(n0, 0, 1, 0), mkdual(α, 1, 0, 0), mkdual(k, 0, 0, 0))
        sD = Species(backend, D3, 24 * ppc; q = -1, m = 1)
        sP = Species(backend, Float32, 24 * ppc; q = -1, m = 1)
        load_quiet!(sD, grid, CosineDensity(), pD, ppc; seed = 9, stream = 3)
        load_quiet!(sP, grid, CosineDensity(), (n0, α, k), ppc; seed = 9, stream = 3)
        # The primal run deposits the value parts of the Dual weights: loading itself is not
        # bitwise between Float32 and Dual{Float32} (`cos` inlined into the Dual kernel can round
        # differently in the last bit; see T-139 and the write-up), and the claim here is about
        # deposition.
        @test maximum(abs.(value.(sD.w) .- sP.w) ./ sP.w) <= 2eps(Float32)
        sP.w .= value.(sD.w)
        ρD = deposited(Reference(), sD, shape, stagger, grid)
        ρP = deposited(Reference(), sP, shape, stagger, grid)
        @test value.(ρD) == ρP                                      # bitwise
        # exact derivatives in Float64 at the same (Float32) positions
        s64 = Species(backend, Float64, 24 * ppc; q = -1, m = 1)
        for (dir, dw) in ((1, x -> n0 * cos(k * x) / ppc), (2, x -> (1 + α * cos(k * x)) / ppc))
            for i in 1:(24 * ppc)
                s64.cell[1][i] = sP.cell[1][i]
                s64.offset[1][i] = sP.offset[1][i]
                x = (Float64(sP.cell[1][i]) - 1 + Float64(sP.offset[1][i])) * Float64(cellsize(grid))
                s64.w[i] = dw(x)
                s64.status[i] = ParticleMeshCore.ALIVE
            end
            fill!(s64.count, Int32(24 * ppc))
            g64 = PeriodicGrid(24, Float64(cellsize(grid)), guard_width(shape))
            ρexact = deposited(Reference(), s64, shape, stagger, g64)
            ∂ = [partials(x)[dir] for x in ρD]
            scale = maximum(abs, ρexact)
            r = record!("T-140", D3, maximum(abs.(∂ .- ρexact)) / (eps(Float32) * scale))
            @test r <= C
        end
        @test all(iszero(partials(x)[3]) for x in ρD)
    end
end
