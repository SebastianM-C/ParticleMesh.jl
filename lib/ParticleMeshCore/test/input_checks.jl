# Input checks, the Float32 uniform draw and the sort of the Fast() fallback, from a review of
# b714331 (#194, #195, #196, #201).

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

@testset "T-344 a loaded Dual species has finite partials at a top draw [A8]" begin
    # Seed 24485953, stream 1, cell 1, k = 1 gives the Philox draw r[3] = 0xffffffa3, one of the top 128
    # that rounded to 1f0 before the fix; the loader turns r[3] into u[3]. Found by scanning seeds (#201).
    r = philox4x32((UInt32(1), UInt32(0), UInt32(0), UInt32(1) << 16), (UInt32(24485953), UInt32(1)))
    @test r[3] >= typemax(UInt32) - 127
    grid = PeriodicGrid(4, 0.25, 1)
    for T in (Float32, D3)
        s = load_quiet!(Species(backend, T, 4; q = -1, m = 1), grid, (x, p) -> p, one(T), 1;
                        vth = (one(T), one(T), one(T)), seed = 24485953, stream = 1)
        @test all(isfinite, components(Array(s.u[3])[1]))
    end
end

"`ρ` of `s` in `Reference()` on a copy of `s`, sorted first: the result `Fast()` must reproduce for `Dual`."
function sorted_reference(s, ρ0, shape, stagger, grid)
    c = deepcopy(s)
    ws = Workspace(c, grid)
    sort_by_cell!(c, grid, ws)
    return deposit!(Reference(), copy(ρ0), c, shape, stagger, grid, ws)
end

@testset "T-345 Fast() needs no prior sort, also where it falls back to Reference() [A4, A8]" begin
    # Fast() on a Dual species ran the sorted kernel on the segment bounds of the last sort: with
    # none it deposited zero, after a move it used stale bounds, silently (#196). The fallback now
    # sorts first, so it equals Reference() after a sort bitwise, partials included.
    n, np = 12, 200
    for shape in SHAPES, stagger in STAGGERS
        grid = PeriodicGrid(n, 0.5f0, guard_width(shape))
        # never sorted: the slots of test_species are not in cell order
        s = test_species(D3, grid, np; seed = 7)
        expect = sorted_reference(s, field(backend, D3, grid), shape, stagger, grid)
        ρ = deposit!(Fast(), field(backend, D3, grid), s, shape, stagger, grid, Workspace(s, grid))
        @test same_bits(ρ, expect)
        @test absdiff(sum(ρ), s.q * sum(s.w[1:np])) <= 4np * eps(Float32) * absmax(s.q * sum(s.w[1:np]))
        # sorted, then moved: the bounds of that sort are stale
        s = test_species(D3, grid, np; seed = 8)
        ws = Workspace(s, grid)
        sort_by_cell!(s, grid, ws)
        for i in 1:np
            s.status[i] == ParticleMeshCore.ALIVE || continue
            c, δ = advance(s.cell[1][i], s.offset[1][i], seeded(D3, 0.9 * cos(2.3i)), Int32(n))
            s.cell[1][i], s.offset[1][i] = c, δ
        end
        expect = sorted_reference(s, field(backend, D3, grid), shape, stagger, grid)
        @test same_bits(deposit!(Fast(), field(backend, D3, grid), s, shape, stagger, grid, ws), expect)
    end
    # Float32 keeps its atomic path, which neither sorts nor needs a sort
    grid = PeriodicGrid(n, 0.5f0, 2)
    s = test_species(Float32, grid, np; seed = 7)
    cells = copy(s.cell[1])
    ρ = deposit!(Fast(), field(backend, Float32, grid), s, Quadratic, Center(), grid, Workspace(s, grid))
    @test s.cell[1] == cells
    @test abs(sum(ρ) - s.q * sum(s.w[1:np])) <= 4np * eps(Float32) * abs(s.q * sum(s.w[1:np]))
    # N = 2
    grid2 = PeriodicGrid((6, 5), (0.5f0, 0.25f0), 2)
    for st in ((Node(), Node()), (Center(), Node()))
        s = test_species_2d(D3, grid2, 150; seed = 4)
        expect = sorted_reference(s, field(backend, D3, grid2), Quadratic, st, grid2)
        @test same_bits(deposit!(Fast(), field(backend, D3, grid2), s, Quadratic, st, grid2, Workspace(s, grid2)), expect)
    end
    # deposit_flux!, never sorted
    for shape in SHAPES
        grid = PeriodicGrid(10, 0.5f0, flux_guard_width(shape, Node()))
        s = moved_species(D3, grid, 300; seed = 5, dead = (40, 41))
        c = deepcopy(s)
        wc = Workspace(c, grid)
        sort_by_cell!(c, grid, wc)
        J(mode, sp, ws) = deposit_flux!(mode, field(backend, D3, grid), field(backend, D3, grid), field(backend, D3, grid),
                                        sp, shape, Node(), grid, ws; dt = 0.25f0, quantity_y = ChargeUy(), quantity_z = ChargeUz())
        @test all(same_bits.(J(Fast(), s, Workspace(s, grid)), J(Reference(), c, wc)))
    end
    # The fallback sort moves the species' extras with the particles; an array the caller keeps
    # outside the species follows by indexing it with ws.perm (slot i now holds the particle of slot ws.perm[i])
    grid = PeriodicGrid(n, 0.5f0, 2)
    s = test_species(D3, grid, np; seed = 10, extras = (old_slot = Int32,))
    for i in 1:capacity(s)
        s.extras.old_slot[i] = Int32(i)
    end
    held = collect(1:capacity(s))
    ws = Workspace(s, grid)
    deposit!(Fast(), field(backend, D3, grid), s, Quadratic, Center(), grid, ws)
    alive(i) = s.status[i] == ParticleMeshCore.ALIVE
    @test ws.perm != 1:capacity(s)
    @test all(i -> s.extras.old_slot[i] == ws.perm[i], 1:capacity(s))
    @test all(i -> s.extras.old_slot[i] > np || Int(value_of(s.u[2][i])) == s.extras.old_slot[i], 1:capacity(s))
    @test all(i -> !alive(i) || Int(value_of(s.u[2][i])) == held[ws.perm[i]], 1:capacity(s))
    @test any(i -> alive(i) && Int(value_of(s.u[2][i])) != held[i], 1:capacity(s))
    # A caller that has sorted passes the effective mode, which does not sort again; Fast() sorts,
    # and on a sorted species that is the identity permutation
    grid = PeriodicGrid(n, 0.5f0, 2)
    s = test_species(D3, grid, np; seed = 9)
    ws = Workspace(s, grid)
    sort_by_cell!(s, grid, ws)
    perm = copy(ws.perm)
    @test perm != 1:capacity(s)
    before = deepcopy(s)
    ρ = deposit!(effective_mode(Fast(), D3), field(backend, D3, grid), s, Quadratic, Center(), grid, ws)
    @test ws.perm == perm
    @test same_bits(deposit!(Fast(), field(backend, D3, grid), s, Quadratic, Center(), grid, ws), ρ)
    @test ws.perm == 1:capacity(s)
    @test all(same_bits(getfield(s, f)[d], getfield(before, f)[d]) for f in (:cell, :offset, :u) for d in eachindex(getfield(s, f)))
end
