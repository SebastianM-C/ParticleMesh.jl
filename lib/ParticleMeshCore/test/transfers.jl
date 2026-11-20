# Shapes, gather, deposition, guards: T-130 to T-135 and T-140. Each body is a function of the
# backend (gpu_support.jl): the testsets here run it on the CPU, gpu.jl on CUDA (T-190).

"Sum of the shape weights at `δ` and whether all are non-negative, as a device-callable."
struct WeightSum{S,G}
    stagger::G
end
WeightSum(shape, stagger) = WeightSum{shape,typeof(stagger)}(stagger)
function (c::WeightSum{S})(δ) where {S}
    _, w = shape_weights(S, c.stagger, δ)
    return (sum(w), all(x -> value_of(x) >= 0, w))
end

function body_T130(be)
    C = tolC("T-130")
    for T in TYPES, shape in SHAPES, stagger in STAGGERS
        ε = base_eps(T)
        δs = [seeded(T, δ0) for δ0 in (0.0, 0.25, 0.5, 0.7, Float64(prevfloat(one(basetype(T)))), (k / 97 for k in 0:96)...)]
        res = devmap(be, WeightSum(shape, stagger), δs)
        worst = maximum(r -> absdiff(r[1], one(T)) / ε, res)
        @test all(r -> r[2], res)
        record!("T-130", T, worst, be)
        @test worst <= C
    end
end

@testset "T-130 partition of unity of the shape weights [A7, A8, WP-010]" begin
    body_T130(CPU())
end

function body_T131(be)
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
            on(be, gather!, out, f, s, shape, stagger, grid)
            for i in 1:200
                x = (s.cell[1][i] - 1 + s.offset[1][i]) * cellsize(grid)
                exact = a + b * x
                worst = max(worst, absdiff(out[i], exact) / (ε * (abs(a) + abs(b) * length(grid))))
            end
        end
        record!("T-131", T, worst, be)
        @test worst <= C
    end
end

@testset "T-131 gather reproduces constant and linear fields [A7, A8, WP-010]" begin
    body_T131(CPU())
end

function body_T132(be)
    # Σᵢ qᵢwᵢ gather(fill f)ᵢ = ⟨fold(deposit(q w)), f⟩ over the interior.
    C = tolC("T-132")
    for T in TYPES, shape in SHAPES, stagger in STAGGERS, mode in (Reference(), Fast())
        ε = base_eps(T)
        grid = PeriodicGrid(12, basetype(T)(1 // 2), guard_width(shape))
        s = test_species(T, grid, 300; seed = 5)
        f = test_field(T, grid, j -> seeded(T, sin(0.7j) + 0.3))
        out = similar(s.w)
        on(be, gather!, out, f, s, shape, stagger, grid)
        lhs = sum(s.q * s.w[i] * out[i] for i in 1:300)
        ρ = on(be, fold_guards!, deposited(mode, s, shape, stagger, grid; be), grid)
        rhs = sum(ρ[guards(grid) + j] * f[guards(grid) + j] for j in 1:ncells(grid))
        scale = sum(absmax(s.w[i]) for i in 1:300) * maximum(absmax, f)
        r = record!("T-132", T, absdiff(lhs, rhs) / (ε * scale), be)
        @test r <= C
    end
end

@testset "T-132 deposition is the adjoint of gather with fill and fold [A7, A8, WP-010]" begin
    body_T132(CPU())
end

function body_T133(be)
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
        fx = on(be, fill_guards!, copy(x), grid)
        fy = on(be, fold_guards!, copy(y), grid)
        lhs = sum(fx .* y)
        rhs = sum(x .* fy)
        scale = sum(absmax.(x)) * maximum(absmax, y)
        r = record!("T-133", T, absdiff(lhs, rhs) / (ε * scale), be)
        @test r <= C
        @test all(iszero, fy[1:g]) && all(iszero, fy[(end - g + 1):end])
    end
end

@testset "T-133 fill and fold are adjoints [A7, A8, WP-010]" begin
    body_T133(CPU())
end

function body_T134(be)
    C = tolC("T-134")
    for T in TYPES, shape in SHAPES, stagger in STAGGERS, mode in (Reference(), Fast()), np in (100, 2000)
        ε = base_eps(T)
        grid = PeriodicGrid(10, one(basetype(T)), guard_width(shape))
        s = test_species(T, grid, np; seed = 7)
        ρ = on(be, fold_guards!, deposited(mode, s, shape, stagger, grid; be), grid)
        total = sum(ρ)
        expect = sum(s.q * s.w[i] for i in 1:np)
        r = record!("T-134", T, absdiff(total, expect) / (ε * np * maximum(i -> absmax(s.w[i]), 1:np)), be)
        @test r <= C
    end
end

@testset "T-134 deposited total equals the sum of q·w [A7, A8, WP-010]" begin
    body_T134(CPU())
end

function body_T135(be)
    # Bound per node: C·eps(T)·Σ|contributions to the node|. Every test weight is positive and the
    # species has one sign of charge, so the sum of magnitudes is |ρR| and the code divides by it.
    C = tolC("T-135")
    for T in (Float32, Float64), shape in SHAPES, stagger in STAGGERS
        ε = eps(T)
        grid = PeriodicGrid(16, one(T), guard_width(shape))
        s = test_species(T, grid, 16 * 400; seed = 11)
        ρF = deposited(Fast(), s, shape, stagger, grid; be)
        ρR = deposited(Reference(), s, shape, stagger, grid; be)
        worst = maximum(abs(ρF[a] - ρR[a]) / (ε * abs(ρR[a])) for a in eachindex(ρR) if ρR[a] != 0)
        record!("T-135", T, worst, be)
        @test worst <= C
    end
end

@testset "T-135 Reference and Fast deposition agree per node [A4, A7, WP-010]" begin
    body_T135(CPU())
end

function body_T140(be)
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
        on(be, load_quiet!, sD, grid, CosineDensity(), pD, ppc; seed = 9, stream = 3)
        on(be, load_quiet!, sP, grid, CosineDensity(), (n0, α, k), ppc; seed = 9, stream = 3)
        # The primal run deposits the value parts of the Dual weights: loading itself is not
        # bitwise between Float32 and Dual{Float32} (`cos` inlined into the Dual kernel can round
        # differently in the last bit; see T-139 and the write-up), and the claim here is about
        # deposition.
        @test maximum(abs.(value.(sD.w) .- sP.w) ./ sP.w) <= 2eps(Float32)
        sP.w .= value.(sD.w)
        ρD = deposited(Reference(), sD, shape, stagger, grid; be)
        ρP = deposited(Reference(), sP, shape, stagger, grid; be)
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
            ρexact = deposited(Reference(), s64, shape, stagger, g64; be)
            ∂ = [partials(x)[dir] for x in ρD]
            scale = maximum(abs, ρexact)
            r = record!("T-140", D3, maximum(abs.(∂ .- ρexact)) / (eps(Float32) * scale), be)
            @test r <= C
        end
        @test all(iszero(partials(x)[3]) for x in ρD)
    end
end

@testset "T-140 Dual weights: exact partials, primal values bitwise [A3, A4, A8, WP-010]" begin
    body_T140(CPU())
end

# --- T-990 (WP-070) -------------------------------------------------------------------------

# A gather with one shape per axis, for the Galerkin gather of the electromagnetic module (a field
# component read one order lower along its staggered axes). Order 0 is the nearest grid point:
# weight one on the cell's own centre (`Center()`) or on the nearer cell edge (`Node()`). With the
# same shape on every axis the per-axis gather is the one-shape gather bit for bit, at the 2D and
# 3D test species' particles (offsets 0 and just below 1 included), every stagger tuple, orders 1
# and 2, all three types, through `gather_at` and through the kernel of `gather!`. With mixed
# orders 0, 1 and 2 the gather of a separable field a(i, j[, l]) = f(i)·g(j)[·h(l)] is the product
# of the 1D gathers of the factors with each axis' shape, to roundoff.
using ParticleMeshCore: gather_at

function body_T990(be = CPU())
    worst = 0.0
    for T in TYPES
        for (N, mk, sp) in ((2, g -> grid2(T, g), test_species_2d), (3, g -> grid3(T, g), test_species_3d))
            grid = mk(2)
            s = sp(T, grid, 40)
            a = field(backend, T, grid)
            A = Array(a)
            for I in eachindex(A)
                A[I] = seeded(T, sin(0.37 * I + 0.1))
            end
            copyto!(a, A)
            stags = Iterators.product(ntuple(_ -> (Node(), Center()), N)...)
            hs = s                                   # host arrays (the CPU backend)
            g = Int32(guards(grid))
            for P in 1:2, st in stags
                shape = Shape{P}()
                o1 = on(be, gather!, fill!(similar(s.w), zero(T)), a, s, shape, st, grid)
                o2 = on(be, gather!, fill!(similar(s.w), zero(T)), a, s, ntuple(_ -> shape, N), st, grid)
                @test reinterpret(UInt8, o1) == reinterpret(UInt8, o2)
                ok = all(1:alive_count(hs)) do i
                    c = ntuple(d -> hs.cell[d][i], N)
                    δ = ntuple(d -> hs.offset[d][i], N)
                    gather_at(A, c, δ, shape, st, g) === gather_at(A, c, δ, ntuple(_ -> shape, N), st, g)
                end
                @test ok
            end
            # mixed orders on a separable field, Float64 and Float32 only (the products' roundoff)
            T <: Dual && continue
            fs = [x -> sin(0.7x + 0.2), x -> cos(0.45x - 0.3), x -> 0.5 + sin(0.3x)]
            for st in stags, P in Iterators.product(ntuple(_ -> 0:2, N)...)
                shapes = ntuple(d -> Shape{P[d]}(), N)
                F = [prod(d -> T(fs[d](I[d])), 1:N) for I in CartesianIndices(A)]
                for i in 1:alive_count(hs)
                    c = ntuple(d -> hs.cell[d][i], N)
                    δ = ntuple(d -> hs.offset[d][i], N)
                    v = gather_at(F, c, δ, shapes, st, g)
                    ref = prod(1:N) do d
                        f1 = [T(fs[d](j)) for j in 1:size(F, d)]
                        gather_at(f1, c[d], δ[d], shapes[d], st[d], g)
                    end
                    err = abs(v - ref) / eps(T)
                    worst = max(worst, err)
                    @test err <= 8
                end
            end
        end
    end
    @info "T-990 mixed-order gather err/eps" worst
end

@testset "T-990 the gather with one shape per axis: order 0 the nearest grid point, the same shape on every axis bitwise the one-shape gather, mixed orders the product of the 1D gathers on a separable field [A4, A7, A8, WP-070]" begin
    for T in (Float32, Float64), δ in (0, 0.25, 0.49, 0.5, 0.75, 0.999)
        @test shape_weights(Shape{0}(), Center(), T(δ)) === (Int32(0), (one(T),))
        @test shape_weights(Shape{0}(), Node(), T(δ)) === (Int32(δ >= 0.5), (one(T),))
        a = T.(collect(1:10))
        @test gather_at(a, Int32(4), T(δ), Shape{0}(), Center(), Int32(2)) === T(6)                 # centre 4, two guards
        @test gather_at(a, Int32(4), T(δ), Shape{0}(), Node(), Int32(2)) === T(δ >= 0.5 ? 7 : 6)    # edge 4 or 5
    end
    @test ParticleMeshCore.stencil_reach(Shape{0}(), Center()) == (0, 0) && ParticleMeshCore.stencil_reach(Shape{0}(), Node()) == (0, 1)
    body_T990(CPU())
end
