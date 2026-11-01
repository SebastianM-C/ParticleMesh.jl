# Input checks and the Float32 uniform draw, from a review of b714331 (#194, #195, #201).

@testset "T-340 the 1D grid constructor applies the n ≥ max(1, 2g) check of the tuple constructor [A4, A7]" begin
    # For g < n < 2g the left and right guards fold into the same interior nodes, a race (#194).
    @test_throws ArgumentError PeriodicGrid(4, 0.25, 3)
    @test_throws ArgumentError PeriodicGrid(1, 1.0, 1)
    @test_throws ArgumentError PeriodicGrid(8, 0.125, -1)
    g = PeriodicGrid(6, 0.5f0, 3)
    @test g isa PeriodicGrid{1,Float32}
    @test g === PeriodicGrid((6,), (0.5f0,), 3)
    @test ncells(g) == 6 && cellsize(g) === 0.5f0 && guards(g) == 3
end

@testset "T-341 a workspace is checked against the species capacity and the grid [A4, A7]" begin
    # The kernels reading the workspace run under @inbounds; a mismatch wrote out of bounds (#195).
    g4 = PeriodicGrid(4, 0.25, 1)
    g8 = PeriodicGrid(8, 0.125, 1)
    s64 = Species(backend, Float64, 64; q = -1, m = 1)
    s128 = Species(backend, Float64, 128; q = -1, m = 1)
    ws = Workspace(s64, g4)
    ρ8 = field(backend, Float64, g8)
    @test_throws ArgumentError sort_by_cell!(s64, g8, ws)
    @test_throws ArgumentError sort_by_cell!(s128, g4, ws)
    @test_throws ArgumentError deposit!(Reference(), ρ8, s64, Shape{2}(), Node(), g8, ws)
    @test_throws ArgumentError deposit!(Reference(), field(backend, Float64, g4), s128, Shape{2}(), Node(), g4, ws)
    @test (sort_by_cell!(s64, g4, ws); true)
    g2 = PeriodicGrid((4, 3), (0.25, 0.5), 1)
    s2 = Species(backend, Float64, 24; q = -1, m = 1, N = 2)
    ws2 = Workspace(s2, g2)
    @test_throws ArgumentError sort_by_cell!(s2, PeriodicGrid((4, 4), (0.25, 0.5), 1), ws2)
end

@testset "T-342 the Float32 uniform draw stays below one, so Box–Muller gives finite Dual partials [A8]" begin
    # (x + 1/2)/2³² rounds to 1f0 for the top 128 draws; log(1) = 0 and sqrt'(0) = Inf gave 0·Inf = NaN (#201).
    top = typemax(UInt32)
    old(T, x) = (T(x) + one(T) / 2) / T(4294967296)
    @test old(Float32, top) == 1f0                       # the hazard the fix removes
    @test uniform_open(Float32, top) == prevfloat(1f0)
    @test all(x -> 0f0 < uniform_open(Float32, x) < 1f0, (top - UInt32(k) for k in 0:300))
    # Every draw that did not round to one keeps its bits, in every element type of the matrix.
    for T in (Float32, Float64), x in (UInt32(0), UInt32(1), UInt32(12345), top - UInt32(200), top)
        o = old(T, x)
        o < one(T) && @test uniform_open(T, x) === o
    end
    @test all(x -> uniform_open(Float64, x) === old(Float64, x), (top - UInt32(k) for k in 0:300))
    z1, z2 = box_muller(D3, top, UInt32(77))
    @test all(isfinite, components(z1)) && all(isfinite, components(z2))
    @test value(z1) === box_muller(Float32, top, UInt32(77))[1]
end
