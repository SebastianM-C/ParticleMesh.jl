# The sort by cell from the order of the last sort (WP-041): the permutation `sort_by_cell!`
# would find, built per cell from the segments of the last sort instead of by a comparison sort
# of every key, and applied by the same gather. Every particle moves by less than one cell per
# step, so the particles of a cell come from the old segments of its 3^N neighbours; the
# particles added since the last sort sit in a short tail behind its count and are sorted on
# their own.

"""
    ResortScratch

The scratch of [`resort_by_cell!`](@ref) in a [`Workspace`](@ref) built with `append`: per linear
cell the `3^N` move counts `hist` (later the ranks), the new counts `cnt` and their running sums
`cum`, the new segment bounds; per slot the free flags and their running sums (the free slots go
last in slot order); the tail of up to `append` slots behind the last sort's count, with its keys,
its permutation, the sortperm workspace and its per-cell bounds; the scan workspaces; the alive
count of the last sort (`sorted_count`, on the device), whether there was one (`ordered`, on
the host), the status array of the species it sorted (`sorted_on`, compared by `===`) and the
number of calls that sorted from the last order instead of in full (`resorts`, on the host); and `spare`, a second copy of every particle array, into which both sorts gather the
whole species in one launch and from which one launch copies it back (two launches where the
gather array by array through the shared buffers takes two per array).
"""
struct ResortScratch{MI,VI,VK,VP,AT,AC,AS,SP}
    hist::MI
    cnt::VI
    cum::VI
    new_start::VI
    new_stop::VI
    free::VI
    free_cum::VI
    sorted_count::VI
    tail_key::VK
    tail_perm::VP
    tail_ak::AT
    tail_start::VI
    tail_stop::VI
    scan_cells::AC
    scan_slots::AS
    spare::SP
    append::Int
    ordered::Base.RefValue{Bool}
    sorted_on::Base.RefValue{Any}
    resorts::Base.RefValue{Int}
end

function ResortScratch(s::Species{N}, ncell::Integer, append::Integer) where {N}
    backend = get_backend(s)
    cap = capacity(s)
    append >= 0 || throw(ArgumentError("append must be ≥ 0, got $append"))
    z(dims...) = KernelAbstractions.zeros(backend, Int32, dims...)
    cnt, cum = z(ncell), z(ncell)
    free, free_cum = z(cap), z(cap)
    tail_key = KernelAbstractions.zeros(backend, Int64, append)
    tail_perm = KernelAbstractions.zeros(backend, Int, append)
    tail_ak = append > 0 ? AK.workspace(AK.sortperm!, tail_perm, tail_key) : nothing
    return ResortScratch(z(3^N, ncell), cnt, cum, KernelAbstractions.ones(backend, Int32, ncell), z(ncell), free, free_cum, z(1),
                         tail_key, tail_perm, tail_ak, KernelAbstractions.ones(backend, Int32, ncell), z(ncell),
                         AK.workspace(AK.accumulate!, +, cum, cnt), AK.workspace(AK.accumulate!, +, free_cum, free),
                         map(similar, species_arrays(s)), Int(append), Ref(false), Ref{Any}(nothing), Ref(0))
end

resort_bytes(::Nothing) = 0
function resort_bytes(r::ResortScratch)
    arrays = (r.hist, r.cnt, r.cum, r.new_start, r.new_stop, r.free, r.free_cum, r.sorted_count, r.tail_key, r.tail_perm,
              r.tail_start, r.tail_stop)
    plans = (r.scan_cells.plan, r.scan_slots.plan, r.tail_ak === nothing ? nothing : r.tail_ak.plan)
    return sum(sizeof, arrays) + sum(sizeof, r.spare) + sum(p -> p === nothing ? 0 : AK.nbytes(p), plans)
end

# After a full sort: the order is known, of these arrays, and its count is the start of the next tail.
note_sorted!(::Nothing, s) = nothing
function note_sorted!(r::ResortScratch, s)
    copyto!(r.sorted_count, s.count)
    r.ordered[] = true
    r.sorted_on[] = s.status
    return nothing
end

# Whether the workspace holds the order of its last sort of these arrays (host values only).
knows_order(r::ResortScratch, s) = r.ordered[] && r.sorted_on[] === s.status

"""
    forget_order!(ws::Workspace) -> ws

Forget the order of the workspace's last sort, so that its next [`resort_by_cell!`](@ref) is the
full sort: for a caller that wrote the species' arrays other than by the moves and appends that
`resort_by_cell!` takes (a restart from saved arrays into the same arrays, a reload). A host
flag; nothing for a workspace without append scratch. Arrays other than those of its last sort
need no call: `resort_by_cell!` compares them (`===`) and sorts them in full.
"""
forget_order!(ws::Workspace) = (forget_order!(ws.resort); ws)
forget_order!(::Nothing) = nothing
forget_order!(r::ResortScratch) = (r.ordered[] = false; nothing)

"""
    resort_neighbours(c, n)

The cells within one cell of `c` per axis on a periodic grid of `n ≥ 3` cells per axis, `3^N` of
them with `c` itself, in increasing linear index (`linear_cell`): per axis the three cells in
increasing order, axis 1 fastest.
"""
@inline function resort_neighbours(c::NTuple{N,Int32}, n::NTuple{N,Int32}) where {N}
    t = map(c, n) do a, m
        lo, hi = a == Int32(1), a == m
        (ifelse(lo | hi, Int32(1), a - Int32(1)), ifelse(lo, Int32(2), ifelse(hi, m - Int32(1), a)), ifelse(lo | hi, m, a + Int32(1)))
    end
    return ntuple(k -> ntuple(d -> t[d][((k - 1) ÷ 3^(d - 1)) % 3 + 1], Val(N)), Val(3^N))
end

# The bin `1:3^N` of a move from cell `c` to cell `cn`, each axis' step -1, 0, 1 as 0, 1, 2 (axis 1
# fastest). A larger move is outside the precondition; it is clamped into the bins, and the
# permutation's writes are bounded, so that it cannot write out of bounds (it gives a wrong order).
@inline function resort_bin(c::NTuple{N,Int32}, cn::NTuple{N,Int32}, n::NTuple{N,Int32}) where {N}
    k = Int32(1)
    for d in 1:N
        k += min(mod(cn[d] - c[d] + Int32(1), n[d]), Int32(2)) * Int32(3)^(d - 1)
    end
    return k
end

# One work item per source cell: its alive particles per move.
@kernel function resort_count_kernel!(hist, s, bins, @Const(seg_start), @Const(seg_stop), n::NTuple{N,Int32}) where {N}
    Ls = Int32(@index(Global, Linear))
    c = cartesian_cell(Ls, n)
    @inbounds for k in axes(hist, 1)
        hist[k, Ls] = Int32(0)
    end
    @inbounds for i in seg_start[Ls]:seg_stop[Ls]
        if s.status[i] == ALIVE
            k = resort_bin(c, particle_bin(bins, s, i, n), n)
            hist[k, Ls] += Int32(1)
        end
    end
end

# One work item per tail slot `sc + j`: the key `(cell, j)` of an alive particle, free slots after
# all of them by `j`; unique, so the tail's permutation is fixed.
@kernel function tail_key_kernel!(key, s, bins, @Const(sorted_count), n, ncell::Int64, append::Int64, cap::Int32)
    j = @index(Global, Linear)
    @inbounds begin
        i = sorted_count[1] + Int32(j)
        alive = i <= cap && s.status[min(i, cap)] == ALIVE
        L = alive ? Int64(linear_cell(particle_bin(bins, s, min(i, cap), n), n)) : ncell + 1
        key[j] = (L - 1) * append + (j - 1)
    end
end

# The per-cell bounds of the sorted tail, as positions in its permutation.
@kernel function tail_segments_kernel!(tail_start, tail_stop, @Const(key), @Const(perm), ncell::Int64, append::Int64)
    q = @index(Global, Linear)
    @inbounds begin
        L = key[perm[q]] ÷ append + 1
        if L <= ncell
            if q == 1 || key[perm[q - 1]] ÷ append + 1 != L
                tail_start[L] = Int32(q)
            end
            if q == length(perm) || key[perm[q + 1]] ÷ append + 1 != L
                tail_stop[L] = Int32(q)
            end
        end
    end
end

@kernel function reset_bounds_kernel!(start, stop)
    L = @index(Global, Linear)
    @inbounds begin
        start[L] = Int32(1)
        stop[L] = Int32(0)
    end
end

# One work item per destination cell: its new count, and in every bin that points at it the
# number of its particles from the sources before that one (sources in increasing linear index,
# each in slot order); its tail particles come after them.
@kernel function resort_rank_kernel!(cnt, hist, @Const(tail_start), @Const(tail_stop), n::NTuple{N,Int32}) where {N}
    L = Int32(@index(Global, Linear))
    c = cartesian_cell(L, n)
    r = Int32(0)
    @inbounds for cs in resort_neighbours(c, n)
        Ls = linear_cell(cs, n)
        k = resort_bin(cs, c, n)
        m = hist[k, Ls]
        hist[k, Ls] = r
        r += m
    end
    @inbounds cnt[L] = r + tail_stop[L] - tail_start[L] + Int32(1)
end

@kernel function resort_segments_kernel!(seg_start, seg_stop, count, @Const(cum), @Const(cnt), ncell::Int32)
    L = Int32(@index(Global, Linear))
    @inbounds begin
        k, e = cnt[L], cum[L]
        seg_start[L] = ifelse(k == Int32(0), Int32(1), e - k + Int32(1))
        seg_stop[L] = ifelse(k == Int32(0), Int32(0), e)
        L == ncell && (count[1] = e)
    end
end

@kernel function free_flag_kernel!(free, @Const(status))
    i = @index(Global, Linear)
    @inbounds free[i] = Int32(status[i] != ALIVE)
end

# One work item per source cell: its alive particles in slot order, each to the next rank of its
# bin, as an entry of the permutation (`perm[j]` is the slot that slot `j` holds before).
@kernel function resort_perm_kernel!(perm, s, bins, hist, @Const(seg_start), @Const(seg_stop), @Const(new_start),
                                     n::NTuple{N,Int32}) where {N}
    Ls = Int32(@index(Global, Linear))
    c = cartesian_cell(Ls, n)
    @inbounds for i in seg_start[Ls]:seg_stop[Ls]
        if s.status[i] == ALIVE
            cn = particle_bin(bins, s, i, n)
            k = resort_bin(c, cn, n)
            r = hist[k, Ls]
            hist[k, Ls] = r + Int32(1)
            p = new_start[linear_cell(cn, n)] + r
            1 <= p <= length(perm) && (perm[p] = Int(i))   # always, within the precondition
        end
    end
end

# One work item per position of the sorted tail: an alive particle after the particles that come
# to its cell from the old segments.
@kernel function tail_perm_kernel!(perm, @Const(key), @Const(tperm), @Const(tail_start), @Const(tail_stop), @Const(new_start),
                                   @Const(cnt), @Const(sorted_count), ncell::Int64, append::Int64)
    q = @index(Global, Linear)
    @inbounds begin
        j = tperm[q]
        L = key[j] ÷ append + 1
        if L <= ncell
            ntail = tail_stop[L] - tail_start[L] + Int32(1)
            p = new_start[L] + cnt[L] - ntail + (Int32(q) - tail_start[L])
            1 <= p <= length(perm) && (perm[p] = Int(sorted_count[1]) + j)
        end
    end
end

# One work item per slot: a free slot after all alive ones, in slot order.
@kernel function free_perm_kernel!(perm, @Const(status), @Const(free_cum), @Const(count))
    i = @index(Global, Linear)
    @inbounds if status[i] != ALIVE
        perm[Int(count[1]) + Int(free_cum[i])] = i
    end
end

"""
    resort_by_cell!(species, grid, ws; workgroupsize = nothing)

What `sort_by_cell!(species, grid, ws)` does, the same bits in every array, the permutation
`ws.perm`, the segment bounds and the count, from the order of the last sort of `species` with
`ws`: every cell counts where its particles went, ranks them per destination and writes their
entries of the permutation, and the particles added since the last sort are sorted on their own
and placed after each cell's others; the arrays then move by `sort_by_cell!`'s gather. The work
is per cell and per tail slot, where `sort_by_cell!` sorts a key per slot.

With `NodeBins` (`Workspace(species, grid; bins)`) the bins take the cells' part throughout: a
particle that moves by less than one cell moves by at most one bin per axis.

It is `sort_by_cell!` itself for a workspace built without `append` (`Workspace(species, grid;
append)`), without a sort of `species` with `ws` as the last sort of `ws` (`sort_by_cell!` or
`resort_by_cell!`; the same arrays, compared by `===`, so a workspace shared by two states over
different arrays that step in turn sorts each in full), after [`forget_order!`](@ref), and with
fewer than 3 cells on an axis; that choice is made on the host from host values and
reads no device data (§0.4).

Preconditions, not detected: since the last sort, every particle alive then has moved by less
than one cell per axis (periodically, the seams and the corners included) or been freed, and
every particle made alive since then sits in the `append` slots behind that sort's alive count
(as `load_window_columns!`-style loads and births behind `species.count` place them), in any
order and at any place. Nothing else may write the arrays in between: whoever does (a restart,
arrays copied in from elsewhere) calls [`forget_order!`](@ref) on the workspace, and the next
sort is the full one.
"""
function resort_by_cell!(s::Species{N}, grid::PeriodicGrid{N}, ws::Workspace; workgroupsize = nothing) where {N}
    r = ws.resort
    (r !== nothing && knows_order(r, s) && all(>=(3), grid.n)) || return sort_by_cell!(s, grid, ws; workgroupsize)
    check_workspace(ws, s, grid)
    backend = get_backend(s)
    cap = capacity(s)
    n = Int32.(grid.n)
    ncell = prod(grid.n)
    launch(k) = launcher(k, backend, workgroupsize)
    launch(resort_count_kernel!)(r.hist, s, ws.bins, ws.seg_start, ws.seg_stop, n; ndrange = ncell)
    if r.append > 0
        launch(tail_key_kernel!)(r.tail_key, s, ws.bins, r.sorted_count, n, Int64(ncell), Int64(r.append), Int32(cap);
                                 ndrange = r.append)
        AK.sortperm!(r.tail_perm, r.tail_key; workspace = r.tail_ak)
        launch(reset_bounds_kernel!)(r.tail_start, r.tail_stop; ndrange = ncell)
        launch(tail_segments_kernel!)(r.tail_start, r.tail_stop, r.tail_key, r.tail_perm, Int64(ncell), Int64(r.append); ndrange = r.append)
    end
    launch(resort_rank_kernel!)(r.cnt, r.hist, r.tail_start, r.tail_stop, n; ndrange = ncell)
    AK.accumulate!(+, r.cum, r.cnt; workspace = r.scan_cells)
    launch(resort_segments_kernel!)(r.new_start, r.new_stop, s.count, r.cum, r.cnt, Int32(ncell); ndrange = ncell)
    launch(free_flag_kernel!)(r.free, s.status; ndrange = cap)
    AK.accumulate!(+, r.free_cum, r.free; workspace = r.scan_slots)
    launch(resort_perm_kernel!)(ws.perm, s, ws.bins, r.hist, ws.seg_start, ws.seg_stop, r.new_start, n; ndrange = ncell)
    r.append > 0 &&
        launch(tail_perm_kernel!)(ws.perm, r.tail_key, r.tail_perm, r.tail_start, r.tail_stop, r.new_start, r.cnt, r.sorted_count,
                                  Int64(ncell), Int64(r.append); ndrange = r.append)
    launch(free_perm_kernel!)(ws.perm, s.status, r.free_cum, s.count; ndrange = cap)
    copyto!(ws.seg_start, r.new_start)
    copyto!(ws.seg_stop, r.new_stop)
    permute_species!(s, ws, workgroupsize)
    copyto!(r.sorted_count, s.count)
    r.resorts[] += 1
    return s
end
