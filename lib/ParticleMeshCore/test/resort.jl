# WP-041, the sort from the order of the last sort (`resort_by_cell!`): T-850 to T-853; the key sort's
# algorithm: T-890 (its CUDA rerun T-891 is in gpu.jl).

"A fixed LCG on [0, 1) (the tests' own, as `test_species_2d`'s), as a callable."
mutable struct ResortLCG
    state::UInt64
end
ResortLCG(seed::Integer) = ResortLCG(UInt64(seed) * 0x9e3779b97f4a7c15 + 0x2545f4914f6cdd1d)
(r::ResortLCG)() = (r.state = r.state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(r.state >> 11) / 2.0^53)
(r::ResortLCG)(m::Integer) = 1 + floor(Int, r() * m)          # uniform on 1:m

"""
    resort_species(T, n, cap, np; seed)

`np` alive particles of element type `T` on `n` cells per axis in a species of capacity `cap`, in
random cells and slots (the slots are not in cell order), three distinct momentum components, with the extras `tag::Int32` (the
particle's id) and `x::T`, and the slots past `np` free.
"""
function resort_species(::Type{T}, n::NTuple{N,Int}, cap, np; seed = 1) where {T,N}
    rng = ResortLCG(seed)
    s = Species(backend, T, cap; q = -1, m = 1, N, extras = (tag = Int32, x = T))
    for i in 1:np
        for d in 1:N
            s.cell[d][i] = rng(n[d])
            s.offset[d][i] = seeded(T, rng())
        end
        s.w[i] = seeded(T, 1 / 2 + rng())
        for d in 1:3
            s.u[d][i] = seeded(T, 2rng() - 1)
        end
        s.status[i] = ParticleMeshCore.ALIVE
        s.extras.tag[i] = Int32(i)
        s.extras.x[i] = seeded(T, i / 7)
    end
    fill!(s.count, Int32(np))
    return s
end

# Move every alive particle by less than one cell per axis through the engine's `advance` (by up
# to `0.98` cells, so particles cross both seams and the corners), and free a fraction of them. A
# particle that crosses the seam of an axis in `open` is freed, as the moving window retires one.
function resort_move!(s::Species{N}, n, rng; pfree = 1 / 20, open = ntuple(_ -> false, N)) where {N}
    T = eltype(s)
    for i in 1:capacity(s)
        s.status[i] == ParticleMeshCore.ALIVE || continue
        rng() < pfree && (s.status[i] = ParticleMeshCore.FREE; continue)
        Δ = ntuple(_ -> seeded(T, 49 / 50 * (2rng() - 1)), N)
        c, δ = advance(ntuple(d -> s.cell[d][i], N), ntuple(d -> s.offset[d][i], N), Δ, Int32.(n))
        any(d -> open[d] && abs(c[d] - s.cell[d][i]) > 1, 1:N) && (s.status[i] = ParticleMeshCore.FREE; continue)
        for d in 1:N
            s.cell[d][i], s.offset[d][i] = c[d], δ[d]
        end
    end
    return s
end

# Add particles behind the count, as a refill or a birth does: up to `k` slots, every `gap`-th one
# left free (none for `gap = 0`), in random cells, and the count advanced by `k` as
# `bump_count_kernel!` does.
function resort_append!(s::Species{N}, n, k, rng; tag0, gap = 3) where {N}
    T = eltype(s)
    c0 = Int(s.count[1])
    for j in 1:k
        i = c0 + j
        i <= capacity(s) || break
        gap > 0 && j % gap == 0 && continue
        for d in 1:N
            s.cell[d][i] = rng(n[d])
            s.offset[d][i] = seeded(T, rng())
        end
        s.status[i] = ParticleMeshCore.ALIVE
        s.w[i] = seeded(T, 1 / 2 + rng())
        s.extras.tag[i] = Int32(tag0 + j)
    end
    s.count[1] = min(c0 + k, capacity(s))
    return s
end

"Whether two species and their workspaces hold the same bits: every array, the permutation, the segments."
same_sort(a, wa, b, wb) = all(same_bits(x, y) for (x, y) in zip(species_arrays(a), species_arrays(b))) &&
                          Array(a.count) == Array(b.count) &&
                          wa.perm == wb.perm && wa.seg_start == wb.seg_start && wa.seg_stop == wb.seg_stop

"""
    resort_run(T, n; steps, append, wgs, seed, np, pfree, gap, bins, open) -> Vector{Bool}

Two copies of one species of `np` particles in `12·prod(n)` slots, one sorted by `sort_by_cell!`
and one by `resort_by_cell!` with a workspace of `append`, both with `bins`, through `steps` rounds
of moves, frees and (every second round) `append` particles added behind the count (every `gap`-th
slot left free, none for `gap = 0`), a fraction `pfree` freed per round, and the particles that
cross the seam of an axis in `open` freed (`resort_move!`); per round whether the two hold the same bits.
"""
function resort_run(::Type{T}, n::NTuple{N,Int}; steps = 6, append = 23, wgs = nothing, seed = 1, np = 8 * prod(n), pfree = 1 / 20, gap = 3,
                    bins = CellBins(), open = ntuple(_ -> false, N)) where {T,N}
    cap = 3 * prod(n) * 4
    grid = PeriodicGrid(n, ntuple(_ -> 1 / 2, N), 1)
    a = resort_species(T, n, cap, np; seed)
    b = deepcopy(a)
    wa, wb = Workspace(a, grid; bins), Workspace(b, grid; append, bins)
    sort_by_cell!(a, grid, wa; workgroupsize = wgs)
    resort_by_cell!(b, grid, wb; workgroupsize = wgs)          # no sort before: the full sort
    same = [same_sort(a, wa, b, wb)]
    for k in 1:steps
        for (s, w) in ((a, wa), (b, wb))
            rng = ResortLCG(seed + 100k)
            resort_move!(s, n, rng; pfree, open)
            iseven(k) && resort_append!(s, n, append, rng; tag0 = 1000k, gap)
        end
        sort_by_cell!(a, grid, wa; workgroupsize = wgs)
        resort_by_cell!(b, grid, wb; workgroupsize = wgs)
        push!(same, same_sort(a, wa, b, wb))
    end
    return same
end

@testset "T-850 resort_by_cell! gives sort_by_cell!'s bits, every array, the permutation, the segments and the count, after moves across the seams, frees and particles added behind the count, at N = 1, 2, 3 [A4, WP-041]" begin
    for T in TYPES, n in ((7,), (16, 12), (5, 4), (3, 4, 3)), wgs in WG_SIZES
        same = resort_run(T, n; wgs)
        @test length(same) == 7 && all(same)
    end
    # a full species: the tail of `append` slots behind the count runs past the capacity
    for n in ((7,), (16, 12), (3, 4, 3))
        @test all(resort_run(Float32, n; np = 12 * prod(n) - 10, pfree = 0))
    end
    # a tail with every slot alive, inside the capacity: its last particle closes its cell's segment
    for n in ((7,), (16, 12), (3, 4, 3))
        @test all(resort_run(Float64, n; gap = 0))
    end
    # fewer particles than cells: empty cells, whose segments are (1, 0) in both sorts
    for n in ((7,), (16, 12), (3, 4, 3))
        @test all(resort_run(Float32, n; np = prod(n) ÷ 2))
    end
    # The slots are in the order of the full sort, and the count is the alive count.
    grid = PeriodicGrid((16, 12), (1 / 2, 1 / 2), 1)
    s = resort_species(Float64, (16, 12), 2304, 1536)
    # the one-launch gather moves every array of the species: the cells, offsets, momenta, weight, status and extras
    a = ParticleMeshCore.species_arrays(s)
    @test length(a) == 2 * 2 + 5 + length(s.extras) &&
          all(x -> count(y -> x === y, a) == 1, (s.cell..., s.offset..., s.u..., s.w, s.status, values(s.extras)...))
    ws = Workspace(s, grid; append = 40)
    sort_by_cell!(s, grid, ws)
    resort_move!(s, (16, 12), ResortLCG(5))
    resort_append!(s, (16, 12), 40, ResortLCG(6); tag0 = 5000)
    alive = count(==(ParticleMeshCore.ALIVE), s.status)
    resort_by_cell!(s, grid, ws)
    L(i) = linear_cell((s.cell[1][i], s.cell[2][i]), Int32.((16, 12)))
    @test s.count[1] == alive && all(==(ParticleMeshCore.ALIVE), s.status[1:alive]) && all(!=(ParticleMeshCore.ALIVE), s.status[(alive + 1):end])
    @test issorted([(L(i), ws.perm[i]) for i in 1:alive])
    @test issorted(ws.perm[(alive + 1):end])
    @test count(>(5000), s.extras.tag) > 20                    # the added particles are in
end

@testset "T-851 resort_by_cell! is the full sort without append, before a first sort, after forget_order!, on arrays other than its last sort's and below 3 cells per axis [A4, WP-041]" begin
    T = Float32
    # without append: the full sort at every call, so a species changed in any way is still sorted
    grid = PeriodicGrid((16, 12), (1 / 2, 1 / 2), 1)
    a = resort_species(T, (16, 12), 600, 500)
    b = deepcopy(a)
    wa, wb = Workspace(a, grid), Workspace(b, grid)
    @test wb.resort === nothing
    for (s, w) in ((a, wa), (b, wb))
        sort_by_cell!(s, grid, w)
        s.cell[1] .= Int32.(mod1.(7 .* (1:600), 16))         # moves of many cells, outside resort's precondition
    end
    sort_by_cell!(a, grid, wa)
    resort_by_cell!(b, grid, wb)
    @test same_sort(a, wa, b, wb)
    # with append but no sort before, and on 2 cells: the full sort
    x = resort_species(T, (16, 12), 600, 500; seed = 3)
    y = deepcopy(x)
    wx, wy = Workspace(x, grid), Workspace(y, grid; append = 4)
    @test !wy.resort.ordered[]
    sort_by_cell!(x, grid, wx)
    resort_by_cell!(y, grid, wy)
    @test wy.resort.ordered[] && same_sort(x, wx, y, wy)
    # after forget_order!: the full sort, so arrays written from elsewhere are sorted
    for (s, w) in ((x, wx), (y, wy))
        s.cell[1] .= Int32.(mod1.(7 .* (1:600), 16))         # moves of many cells, as a restart from other arrays
    end
    @test forget_order!(wy) === wy && !wy.resort.ordered[] && forget_order!(wx) === wx
    sort_by_cell!(x, grid, wx)
    resort_by_cell!(y, grid, wy)
    @test same_sort(x, wx, y, wy)
    # a workspace whose last sort was of other arrays: the full sort
    z = deepcopy(y)
    z.cell[1] .= Int32.(mod1.(5 .* (1:600), 16))
    zc, wz = deepcopy(z), Workspace(z, grid)
    resort_by_cell!(z, grid, wy)
    sort_by_cell!(zc, grid, wz)
    @test same_sort(z, wy, zc, wz)
    # outside the precondition (a full buffer, every particle moved 2 or 3 cells): a wrong order,
    # but every write within the arrays (the suite runs with bounds checks; without the bound on
    # the permutation's writes, 6 of these 24 cases write past its end)
    g75 = PeriodicGrid((7, 5), (1 / 2, 1 / 2), 1)
    for seed in 1:6, step in (-3, -2, 2, 3)
        f = resort_species(Float64, (7, 5), 280, 280; seed)
        wf = Workspace(f, g75; append = 0)
        resort_by_cell!(f, g75, wf)
        f.cell[1] .= Int32.(mod1.(f.cell[1] .+ step, 7))
        resort_by_cell!(f, g75, wf)
        @test all(in(1:280), wf.perm)
    end
    g2 = PeriodicGrid((2, 12), (1 / 2, 1 / 2), 1)
    for k in 1:2
        c = resort_species(T, (2, 12), 300, 200; seed = k)
        d = deepcopy(c)
        wc, wd = Workspace(c, g2), Workspace(d, g2; append = 0)
        sort_by_cell!(c, g2, wc); resort_by_cell!(d, g2, wd)
        for (s, w) in ((c, wc), (d, wd))
            s.cell[2] .= Int32.(mod1.(5 .* (1:300), 12))
        end
        sort_by_cell!(c, g2, wc); resort_by_cell!(d, g2, wd)
        @test same_sort(c, wc, d, wd)
    end
end

"A non-CPU backend type for T-890: `key_sort_alg` reads only the backend's type."
struct KeySortFakeGPU <: KernelAbstractions.Backend end

@testset "T-890 the key sort's algorithm is radix sort on a GPU backend from 2^20 slots and AcceleratedKernels' Auto otherwise, and an explicit one gives the default's bits [A4, WP-041]" begin
    AK = ParticleMeshCore.AK
    @test ParticleMeshCore.key_sort_alg(CPU(), 1 << 24) isa AK.Auto
    @test ParticleMeshCore.key_sort_alg(KeySortFakeGPU(), 1 << 20) isa AK.RadixSort
    @test ParticleMeshCore.key_sort_alg(KeySortFakeGPU(), (1 << 20) - 1) isa AK.Auto
    n = (16, 12)
    grid = PeriodicGrid(n, (1 / 2, 1 / 2), 1)
    for T in TYPES
        a = resort_species(T, n, 2000, 1500; seed = 5)
        b = resort_species(T, n, 2000, 1500; seed = 5)
        wa = Workspace(a, grid)
        wb = Workspace(b, grid; key_sort = AK.MergeSort())
        @test !(wa.ak.plan.alg isa AK.MergeSort) && wb.ak.plan.alg isa AK.MergeSort
        sort_by_cell!(a, grid, wa); sort_by_cell!(b, grid, wb)
        @test same_sort(a, wa, b, wb)
        @test issorted(wa.key) && wa.perm != collect(1:2000)
    end
end

"""
    body_T1190(be)

T-1190 on `be` (gpu.jl reruns it on CUDA, T-1192): `resort_by_cell!(...; check = true)` through
T-850's moves, frees and appends at N = 1, 2, 3 gives the bits of the full sort and of the
unchecked resort, and it throws, with the arrays not moved and the order forgotten, when the
precondition breaks: a particle moved by two cells, more particles appended than `append`. The
moves and appends are made on host copies (moved to the host and back around them), as in T-892.
"""
function body_T1190(be)
    dev(h) = be isa CPU ? h : todev(be, h)
    sync!(dst, src) = dst === src || foreach(copyto!, species_arrays(dst), species_arrays(src))
    hostbits(s) = map(x -> Array(x), species_arrays(s))
    same(a, wa, b, wb) = all(same_bits(x, y) for (x, y) in zip(hostbits(a), hostbits(b))) && Array(wa.perm) == Array(wb.perm) &&
                         Array(wa.seg_start) == Array(wb.seg_start) && Array(wa.seg_stop) == Array(wb.seg_stop)
    for (T, n) in ((Float32, (7,)), (Float64, (16, 12)), (Float32, (5, 4, 3)))
        N = length(n)
        cap = 20 * prod(n)                 # room behind the count for the overflowing append below
        append = 23
        grid = PeriodicGrid(n, ntuple(_ -> 1 / 2, N), 1)
        # within the precondition: the checked resort, the unchecked one and the full sort agree
        hs = [resort_species(T, n, cap, 8 * prod(n); seed = 3) for _ in 1:3]
        a, b, c = map(dev, hs)
        wa, wb, wc = Workspace(a, grid), Workspace(b, grid; append), Workspace(c, grid; append)
        sort_by_cell!(a, grid, wa); resort_by_cell!(b, grid, wb); resort_by_cell!(c, grid, wc; check = true)
        ok = Bool[]
        for k in 1:4
            for (h, d) in zip(hs, (a, b, c))
                sync!(h, d)
                rng = ResortLCG(3 + 100k)
                resort_move!(h, n, rng)
                iseven(k) && resort_append!(h, n, append, rng; tag0 = 1000k)
                sync!(d, h)
            end
            sort_by_cell!(a, grid, wa); resort_by_cell!(b, grid, wb); resort_by_cell!(c, grid, wc; check = true)
            push!(ok, same(a, wa, c, wc) && same(b, wb, c, wc))
        end
        @test all(ok)
        @test wc.resort.resorts[] == 4
        # broken: one particle moved by two cells on axis 1
        h = hs[3]
        sync!(h, c)
        i = findfirst(==(ParticleMeshCore.ALIVE), h.status)
        h.cell[1][i] = Int32(mod1(h.cell[1][i] + 2, n[1]))
        sync!(c, h)
        before = hostbits(c)
        @test_throws ArgumentError resort_by_cell!(c, grid, wc; check = true)
        @test !wc.resort.ordered[] && wc.resort.resorts[] == 4
        @test all(same_bits(x, y) for (x, y) in zip(hostbits(c), before))    # the arrays are not moved
        # the next sort with the workspace is the full one, which sorts these arrays
        r = dev(deepcopy(h))
        wr = Workspace(r, grid)
        sort_by_cell!(r, grid, wr)
        @test resort_by_cell!(c, grid, wc; check = true) === c
        @test same(r, wr, c, wc) && Array(r.count) == Array(c.count) && wc.resort.resorts[] == 4
        # broken: more particles appended behind the count than the workspace's `append`
        sync!(h, c)
        resort_append!(h, n, append + 5, ResortLCG(77); tag0 = 9000, gap = 0)
        @test h.status[Int(Array(wc.resort.sorted_count)[1]) + append + 1] == ParticleMeshCore.ALIVE   # alive past the tail
        sync!(c, h)
        before = hostbits(c)                      # the count included (the tests' species_arrays)
        @test_throws ArgumentError resort_by_cell!(c, grid, wc; check = true)
        @test !wc.resort.ordered[] && wc.resort.resorts[] == 4
        @test all(same_bits(x, y) for (x, y) in zip(hostbits(c), before))    # the count put back: this resort had changed it
        # without check the same broken state is not detected: no throw, the order kept
        sort_by_cell!(c, grid, wc)
        sync!(h, c)
        resort_append!(h, n, append + 5, ResortLCG(78); tag0 = 9500, gap = 0)
        sync!(c, h)
        @test resort_by_cell!(c, grid, wc) === c && wc.resort.ordered[] && wc.resort.resorts[] == 5
    end
    # without append scratch resort_by_cell! is the full sort, and check does nothing
    grid = PeriodicGrid((16, 12), (1 / 2, 1 / 2), 1)
    h = resort_species(Float32, (16, 12), 600, 500)
    s = dev(h)
    w = Workspace(s, grid)
    sort_by_cell!(s, grid, w)
    sync!(h, s)
    h.cell[1] .= Int32.(mod1.(7 .* (1:600), 16))
    sync!(s, h)
    @test resort_by_cell!(s, grid, w; check = true) === s
end

@testset "T-1190 resort_by_cell!(...; check = true) gives the full sort's bits within its precondition and throws when a particle moved two cells or more particles were appended than its append, at N = 1, 2, 3 [A4, WP-041]" begin
    body_T1190(CPU())
end

@testset "T-1193 each comparison of resort_by_cell!'s check counts on its own: one permutation entry, one segment start, one segment stop, the count [A4, WP-041]" begin
    # The two broken cases of T-1190 differ from the full sort in the permutation and the segments
    # both, so there each comparison hides the others; here each one sees a single difference.
    PMC = ParticleMeshCore
    grid = PeriodicGrid((16, 12), (1 / 2, 1 / 2), 1)
    s = resort_species(Float64, (16, 12), 600, 500)
    ws = Workspace(s, grid)
    sort_by_cell!(s, grid, ws)                       # ws.key sorted, ws.perm, the segments and the count from it
    cap, ncell = 600, prod(grid.n)
    function bad_of(; perm = copy(ws.perm), start = copy(ws.seg_start), stop = copy(ws.seg_stop), count = copy(s.count))
        bad = zeros(Int32, 3)
        PMC.launcher(PMC.resort_perm_check_kernel!, backend, nothing)(bad, perm, ws.key, Int64(cap); ndrange = cap)
        PMC.launcher(PMC.resort_segments_check_kernel!, backend, nothing)(bad, start, stop, ws.seg_start, ws.seg_stop; ndrange = ncell)
        PMC.launcher(PMC.resort_count_check_kernel!, backend, nothing)(bad, count, s.count; ndrange = 1)
        return bad
    end
    @test bad_of() == [0, 0, 0]
    p = copy(ws.perm); p[cap] = p[1]
    @test bad_of(perm = p) == [1, 0, 0]
    p = copy(ws.perm); p[1], p[2] = p[2], p[1]
    @test bad_of(perm = p) == [2, 0, 0]
    L = findlast(>(0), ws.seg_stop)                  # the last non-empty bin: every slot and bin is compared
    a = copy(ws.seg_start); a[L] += 1
    @test bad_of(start = a) == [0, 1, 0]
    b = copy(ws.seg_stop); b[L] -= 1
    @test bad_of(stop = b) == [0, 1, 0]
    @test bad_of(count = s.count .- Int32(1)) == [0, 0, 1]
end
