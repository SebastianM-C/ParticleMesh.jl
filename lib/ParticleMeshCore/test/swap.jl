# WP-041, the sort's gather into the workspace's spare arrays, swapped with the species' own
# (`Workspace(; swap = true)`, the default): T-920 to T-924. Uses resort.jl's helpers.

"A species over the arrays of `s` that it does not own (the copy-back form), as a state view builds one."
borrowed(s::Species) = Species(s.cell, s.offset, s.u, s.w, s.status, s.count, s.q, s.m, s.extras)

"An adaptor that copies every array: new storage of the same type, as a move to a device gives."
struct CopyingAdaptor end
ParticleMeshCore.Adapt.adapt_storage(::CopyingAdaptor, x::AbstractArray) = copy(x)

"""
    swap_run(T, n; swap_full, swap_resort, steps, wgs, bins) -> (Vector{Bool}, Int)

`resort_run`'s rounds of moves, frees and appends on three copies of one species: `a` sorted by
`sort_by_cell!` with a workspace without `swap` (the gather array by array with its copy back,
the form before WP-041's swap), `b` by `sort_by_cell!` with `swap = swap_full` and `c` by
`resort_by_cell!` with `swap = swap_resort`; per round whether all three hold the same bits, and
how many of `c`'s sorts started from the order of the last (all but the first, when the swapped
arrays are recognised as those of the last sort).
"""
function swap_run(::Type{T}, n::NTuple{N,Int}; swap_full = true, swap_resort = true, steps = 5, wgs = nothing, bins = CellBins(),
                  append = 23, seed = 2) where {T,N}
    cap = 3 * prod(n) * 4
    grid = PeriodicGrid(n, ntuple(_ -> 1 / 2, N), 1)
    a = resort_species(T, n, cap, 8 * prod(n); seed)
    b, c = deepcopy(a), deepcopy(a)
    wa = Workspace(a, grid; bins, swap = false)
    wb = Workspace(b, grid; bins, swap = swap_full)
    wc = Workspace(c, grid; bins, append, swap = swap_resort)
    same = Bool[]
    for k in 0:steps
        if k > 0
            for s in (a, b, c)
                rng = ResortLCG(seed + 100k)
                resort_move!(s, n, rng)
                iseven(k) && resort_append!(s, n, append, rng; tag0 = 1000k)
            end
        end
        sort_by_cell!(a, grid, wa; workgroupsize = wgs)
        sort_by_cell!(b, grid, wb; workgroupsize = wgs)
        resort_by_cell!(c, grid, wc; workgroupsize = wgs)
        push!(same, same_sort(a, wa, b, wb) && same_sort(a, wa, c, wc))
    end
    return same, wc.resort.resorts[]
end

@testset "T-920 the gather into swapped spare arrays gives the bits of the gather with its copy back, every array, the permutation, the segments and the count, for sort_by_cell! and resort_by_cell! at N = 1, 2, 3 [A4, WP-041]" begin
    for T in TYPES, n in ((7,), (16, 12), (3, 4, 3)), wgs in WG_SIZES
        same, resorts = swap_run(T, n; wgs)
        @test all(same) && resorts == 5
    end
    for bins in (NodeBins(Quadratic, (Node(), Node())), NodeBins(Linear, (Center(), Node())))
        same, resorts = swap_run(Float64, (16, 12); bins)
        @test all(same) && resorts == 5
    end
    # each sort form with and without swap against each other
    for (swap_full, swap_resort) in ((false, true), (true, false))
        same, resorts = swap_run(Float32, (16, 12); swap_full, swap_resort)
        @test all(same) && resorts == 5
    end
    # the permutation is Int32, read once per slot
    s = resort_species(Float32, (16, 12), 600, 500)
    @test eltype(Workspace(s, PeriodicGrid((16, 12), (1 / 2, 1 / 2), 1)).perm) === Int32
end

@testset "T-921 a species that owns its arrays takes the workspace's spare set at a sort and leaves its old set as the spare; the count and the scratch stay [WP-041]" begin
    grid = PeriodicGrid((16, 12), (1 / 2, 1 / 2), 1)
    s = resort_species(Float64, (16, 12), 600, 500)
    ws = Workspace(s, grid)
    @test ParticleMeshCore.owns_arrays(s) && ws.spare !== nothing
    old, spare = ParticleMeshCore.species_arrays(s), ParticleMeshCore.flat_arrays(ws.spare[])
    count = s.count
    copy_of = deepcopy(s)
    sort_by_cell!(s, grid, ws)
    now = ParticleMeshCore.species_arrays(s)
    @test all(map(===, now, spare)) && all(map(===, ParticleMeshCore.flat_arrays(ws.spare[]), old))
    @test s.count === count                                  # the count is not swapped
    # the slots of the old set are those before the sort: the gather reads, never writes them
    @test all(same_bits(x, y) for (x, y) in zip(old, ParticleMeshCore.species_arrays(copy_of)))
    # a second sort swaps back: the two sets alternate, nothing is allocated
    sort_by_cell!(s, grid, ws)
    @test all(map(===, ParticleMeshCore.species_arrays(s), old))
    # a copy of the species value shares the reference, so it reads the current arrays too
    t = s
    sort_by_cell!(s, grid, ws)
    @test all(map(===, ParticleMeshCore.species_arrays(t), spare))
    # adapted to new storage (as `adapt(CUDABackend(), s)` moves a species to a device) it keeps a
    # reference of its own and swaps; adapted to the storage it already has, it shares the
    # arrays and borrows them, so it cannot swap them away from `s`
    same = ParticleMeshCore.Adapt.adapt(Array, s)
    @test getfield(same, :data) isa Base.RefValue && !ParticleMeshCore.owns_arrays(same)
    @test all(map(===, ParticleMeshCore.species_arrays(same), ParticleMeshCore.species_arrays(s)))
    c = ParticleMeshCore.Adapt.adapt(CopyingAdaptor(), s)
    @test getfield(c, :data) isa Base.RefValue && getfield(c, :data) !== getfield(s, :data) && ParticleMeshCore.owns_arrays(c)
    @test !any(map(===, ParticleMeshCore.species_arrays(c), ParticleMeshCore.species_arrays(s)))
    wc = Workspace(c, grid)
    cs = ParticleMeshCore.flat_arrays(wc.spare[])
    sort_by_cell!(c, grid, wc)
    @test all(map(===, ParticleMeshCore.species_arrays(c), cs))
    # the scratch counts the spare set
    @test ParticleMeshCore.scratch_bytes(ws) - ParticleMeshCore.scratch_bytes(Workspace(s, grid; swap = false)) ==
          sum(sizeof, ParticleMeshCore.species_arrays(s))
end

@testset "T-922 a species over arrays it does not own keeps them: the sort gathers into the spare set and copies back, with the swap's bits [A4, WP-041]" begin
    grid = PeriodicGrid((16, 12), (1 / 2, 1 / 2), 1)
    for T in TYPES, n in ((7,), (16, 12), (3, 4, 3))
        g = PeriodicGrid(n, ntuple(_ -> 1 / 2, length(n)), 1)
        a = resort_species(T, n, 12 * prod(n), 8 * prod(n); seed = 4)
        b = deepcopy(a)
        v = borrowed(b)
        @test !ParticleMeshCore.owns_arrays(v)
        mine = ParticleMeshCore.species_arrays(b)
        wa, wv = Workspace(a, g), Workspace(v, g; append = 5)
        for k in 1:3
            for s in (a, v)
                resort_move!(s, n, ResortLCG(10k))
            end
            sort_by_cell!(a, g, wa)
            resort_by_cell!(v, g, wv)
            @test same_sort(a, wa, v, wv)
            @test all(map(===, ParticleMeshCore.species_arrays(v), mine))   # still the borrowed arrays
            @test all(map(===, ParticleMeshCore.species_arrays(b), mine))   # which their owner reads
        end
    end
end

@testset "T-923 a workspace lent to a species of other array types (borrowed or owned) gathers array by array, with the same bits [A4, WP-041]" begin
    n = (16, 12)
    grid = PeriodicGrid(n, (1 / 2, 1 / 2), 1)
    a = resort_species(Float32, n, 600, 500; seed = 6)
    b = deepcopy(a)
    # views are another array type than the workspace's spare set (vectors)
    vw(x) = view(x, 1:length(x))
    v = Species(map(vw, b.cell), map(vw, b.offset), map(vw, b.u), vw(b.w), vw(b.status), b.count, b.q, b.m, map(vw, b.extras))
    wa, wv = Workspace(a, grid), Workspace(b, grid)
    resort_move!(a, n, ResortLCG(7)); resort_move!(v, n, ResortLCG(7))
    sort_by_cell!(a, grid, wa)
    sort_by_cell!(v, grid, wv)
    @test same_sort(a, wa, v, wv) && same_sort(a, wa, b, wv)
    # a species that owns arrays of another type than the spare set's cannot take that set: it
    # gathers array by array and keeps its arrays
    c = deepcopy(b)
    o = Species(Ref(ParticleMeshCore.SpeciesArrays(map(vw, c.cell), map(vw, c.offset), map(vw, c.u), vw(c.w), vw(c.status), map(vw, c.extras))),
                copy(c.count), c.q, c.m, true)
    @test ParticleMeshCore.owns_arrays(o)
    held = ParticleMeshCore.species_arrays(o)
    wo = Workspace(c, grid)
    resort_move!(a, n, ResortLCG(8)); resort_move!(o, n, ResortLCG(8))
    sort_by_cell!(a, grid, wa)
    sort_by_cell!(o, grid, wo)
    @test same_sort(a, wa, o, wo) && all(map(===, ParticleMeshCore.species_arrays(o), held))
    @test all(map((x, y) -> parent(x) === y, ParticleMeshCore.species_arrays(v), ParticleMeshCore.species_arrays(b)))
end

@testset "T-924 kernels read a swapped species' current arrays: Reference deposits after a swapping sort equal those after the copy-back sort, bit for bit [A4, WP-041]" begin
    for T in TYPES, n in ((7,), (16, 12), (5, 4, 4))
        N = length(n)
        g = PeriodicGrid(n, ntuple(_ -> 1 / 2, N), 2)
        a = resort_species(T, n, 12 * prod(n), 8 * prod(n); seed = 8)
        b = deepcopy(a)
        wa, wb = Workspace(a, g; swap = false), Workspace(b, g)
        ρa, ρb = field(backend, T, g), field(backend, T, g)
        st = ntuple(_ -> Node(), N)
        for k in 1:2
            for s in (a, b)
                resort_move!(s, n, ResortLCG(20k))
            end
            sort_by_cell!(a, g, wa)
            sort_by_cell!(b, g, wb)
            fill!(ρa, zero(T)); fill!(ρb, zero(T))
            deposit!(Reference(), ρa, a, Quadratic, N == 1 ? Node() : st, g, wa)
            deposit!(Reference(), ρb, b, Quadratic, N == 1 ? Node() : st, g, wb)
            @test same_bits(ρa, ρb) && any(!iszero, ρb)
        end
    end
end

@testset "T-925 a borrowed species that shares arrays with the workspace's spare set (a view over an owner's extras after the owner's swap) sorts array by array, with the bits of a lone copy [A4, WP-041]" begin
    n = (16, 12)
    g = PeriodicGrid(n, (1 / 2, 1 / 2), 1)
    for T in TYPES, f in (sort_by_cell!, resort_by_cell!)
        s = resort_species(T, n, 600, 500; seed = 9)
        ws = Workspace(s, g; append = 5)
        sort_by_cell!(s, g, ws)                          # the owner's old set is now the spare set
        old = ws.spare[]
        v = Species(map(copy, s.cell), map(copy, s.offset), map(copy, s.u), copy(s.w), copy(s.status), copy(s.count), s.q, s.m,
                    old.extras)                          # its extras are the spare set's
        r = Species(map(copy, v.cell), map(copy, v.offset), map(copy, v.u), copy(v.w), copy(v.status), copy(v.count), v.q, v.m,
                    map(copy, v.extras))
        @test ParticleMeshCore.shares_array(ParticleMeshCore.species_arrays(v), ParticleMeshCore.flat_arrays(ws.spare[]))
        wr = Workspace(r, g; swap = false, append = 5)
        resort_move!(v, n, ResortLCG(31)); resort_move!(r, n, ResortLCG(31))
        f(v, g, ws)
        f(r, g, wr)
        @test same_sort(r, wr, v, ws)
        @test all(map(===, values(v.extras), values(old.extras)))   # a borrowed species keeps its arrays
    end
end
