# #226, item 2: where `load_quiet!` evaluates the density, T-510 and T-511. With
# `density_at = :particle` (the default) the weight of a particle is the density at its own
# position; with `:cell_centre` it is the density at the centre of its cell, so the loaded profile
# is piecewise constant per cell, as Smilei and PIConGPU sample it. Positions, draws and status
# are the same in both.

"The density at the centre of cell `c` (1D) or `(c1, c2)` (2D), divided by `K`, evaluated on the host."
centre_weight(density, p, grid::PeriodicGrid{1}, ::Type{T}, c, K) where {T} =
    density((T(c - 1) + one(T) / 2) * cellsize(grid), p) / T(K)
centre_weight(density, p, grid::PeriodicGrid{2}, ::Type{T}, c, K) where {T} =
    density(ntuple(d -> (T(c[d] - 1) + one(T) / 2) * cellsize(grid)[d], Val(2)), p) / T(K)

"Weights equal to the host values: bitwise for Float32 and Float64, within 2 ulp for `Dual` (the kernel may contract to FMAs, as in T-139)."
weights_match(::Type{T}, w, ref) where {T} =
    T === D3 ? all(i -> absdiff(w[i], ref[i]) <= 2eps(Float32) * absmax(ref[i]), eachindex(w, ref)) : all(i -> w[i] == ref[i], eachindex(w, ref))

function body_T510()
    for T in TYPES
        # N = 1
        grid = PeriodicGrid(16, basetype(T)(1 // 8), 2)
        p = (seeded(T, 1), seeded(T, 1 // 20), T(2π / length(grid)))
        vth = (T(1 // 20), T(1 // 20), T(1 // 20))
        load(; kw...) = load_quiet!(Species(backend, T, 16 * 8; q = -1, m = 1), grid, CosineDensity(), p, 8;
                                    vth, seed = 3, stream = 2, kw...)
        a, b, c = load(), load(; density_at = :particle), load(; density_at = :cell_centre)
        for f in (s -> s.cell[1], s -> s.offset[1], s -> s.u[1], s -> s.u[2], s -> s.u[3], s -> s.status)
            @test same_bits(f(a), f(b)) && same_bits(f(a), f(c))
        end
        @test same_bits(a.w, b.w)                          # the default is :particle, bit for bit
        @test !same_bits(a.w, c.w)
        @test weights_match(T, c.w, [centre_weight(CosineDensity(), p, grid, T, c.cell[1][i], 8) for i in 1:(16 * 8)])
        @test alive_count(c) == 16 * 8
        # N = 2
        g2 = PeriodicGrid((8, 6), (basetype(T)(1 // 4), basetype(T)(1 // 3)), 2)
        p2 = oblique_params(T, g2)
        ppc = (3, 2)
        load2(; kw...) = load_quiet!(Species(backend, T, 48 * 6; q = -1, m = 1, N = 2), g2, ObliqueDensity(), p2, ppc;
                                     vth, seed = 3, stream = 2, kw...)
        a2, b2, c2 = load2(), load2(; density_at = :particle), load2(; density_at = :cell_centre)
        for f in (s -> s.cell[1], s -> s.cell[2], s -> s.offset[1], s -> s.offset[2], s -> s.u[1], s -> s.u[2], s -> s.u[3], s -> s.status)
            @test same_bits(f(a2), f(b2)) && same_bits(f(a2), f(c2))
        end
        @test same_bits(a2.w, b2.w)
        @test !same_bits(a2.w, c2.w)
        @test weights_match(T, c2.w, [centre_weight(ObliqueDensity(), p2, g2, T, (c2.cell[1][i], c2.cell[2][i]), 6) for i in 1:(48 * 6)])
    end
    s = Species(backend, Float64, 16; q = -1, m = 1)
    @test_throws ArgumentError load_quiet!(s, PeriodicGrid(16, 1.0, 2), CosineDensity(), (1.0, 0.1, 1.0), 1;
                                           seed = 1, stream = 1, density_at = :cell_center)
    @test_throws ArgumentError load_quiet!(Species(backend, Float64, 16; q = -1, m = 1, N = 2), PeriodicGrid((4, 4), (1.0, 1.0), 2),
                                           ObliqueDensity(), (1.0, 0.1, 1.0, 1.0), (1, 1); seed = 1, stream = 1, density_at = :node)
end

# The cell sums of the weights of a cosine profile on the quiet lattice, Float64. Cell-centred,
# a cell's weights add up to the profile at its centre x_c. Per particle, the lattice offsets
# (δₖ − 1/2)Δx are symmetric about x_c, so Σₖ cos(k(x_c + (δₖ − 1/2)Δx)) = cos(k·x_c)·Σₖ cos(k(δₖ − 1/2)Δx):
# the cell sum is the centre value with the cosine scaled by the Dirichlet factor
# sin(y)/(ppc·sin(y/ppc)), y = kΔx/2, which is 1 − (kΔx)²/24 + O((kΔx)⁴, (kΔx)²/ppc²).
function body_T511()
    n, ppc, α = 32, 16, 0.3
    for mode in (1, 3)
        grid = PeriodicGrid(n, 0.25, 2)
        k = 2π * mode / length(grid)
        p = (1.0, α, k)
        load(at) = load_quiet!(Species(backend, Float64, n * ppc; q = -1, m = 1), grid, CosineDensity(), p, ppc;
                               seed = 1, stream = 1, density_at = at)
        a, c = load(:particle), load(:cell_centre)
        xc = [(j - 0.5) * cellsize(grid) for j in 1:n]
        cellsum(s) = [sum(s.w[(j - 1) * ppc + 1:j * ppc]) for j in 1:n]
        y = k * cellsize(grid) / 2
        dirichlet = sin(y) / (ppc * sin(y / ppc))
        @test isapprox(cellsum(c), [1 + α * cos(k * x) for x in xc]; rtol = 0, atol = 64eps())
        @test isapprox(cellsum(a), [1 + α * dirichlet * cos(k * x) for x in xc]; rtol = 0, atol = 64eps())
        @test abs(dirichlet - (1 - (2y)^2 / 24)) <= (2y)^4 / 1920 + (2y)^2 / (24 * ppc^2) + 4eps()
    end
end

@testset "T-510 load_quiet!'s density_at: :particle (default) keeps the weights bit for bit, :cell_centre samples the cell centre, the rest unchanged, N = 1 and 2 [A3, A4, A8]" begin
    body_T510()
end

@testset "T-511 the cell sums of a cosine's weights: the profile at the cell centre (cell-centred), the centre value with the Dirichlet factor on the cosine (per particle) [A3]" begin
    body_T511()
end
