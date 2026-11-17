"""
    Workspace(species, grid; flux = nothing, flux_layout = flux_patch_layout(backend), append = nothing,
              bins = CellBins())

Scratch for one species on one grid, allocated once: the sort keys and permutation, the
AcceleratedKernels sort workspace, one permutation buffer per element type of the core SoA
arrays (`Int32`, `T`, `UInt8`), the buffers of the species' extras, and the per-cell segment
bounds `seg_start`, `seg_stop` that `sort_by_cell!` writes and the `Reference` deposition reads,
one entry per linear bin (`prod(n)` of them; an empty bin has `seg_start = 1`, `seg_stop = 0`).
The bins are `bins`: the cells (`CellBins()`, the default) or the first nodes of the particles'
stencils (`NodeBins(shape, stagger)`, WP-041), which the sort, `resort_by_cell!` and the
`Reference()` deposits all use.

`tmp_extras` holds one permutation buffer per extra, in the extras' order: the core buffer of
the same element type where there is one (so `Int32`- and `T`-valued extras cost no scratch),
otherwise one new buffer per further element type, shared by the extras of that type. An empty
extras tuple costs nothing.

`append = k` (an integer ≥ 0) adds the scratch of [`resort_by_cell!`](@ref) (`resort`, a
[`ResortScratch`](@ref)), which sorts from the order of the last sort with a tail of up to `k`
particles added behind its count; without it `resort` is `nothing` and costs nothing.

`flux = (shape, stagger)`, for `N = 2`, adds the per-cell patches of the `Reference()` flux
deposition with that shape and ρ's stagger (`deposit_flux!`): `3·M₁·M₂` values of `T` per linear
bin, `M_d` the flux stencil width of axis `d` in the bin's frame (`flux_bin_reach`; 5 or 6 at
order 2 by cell, 5 at order 2 on nodes by `NodeBins` of that shape and stagger), in the array `flux_patch`,
stored in the order `flux_layout`: `StencilMajor()`, `3·M₁·M₂ × prod(n)` with one column per
cell, or `CellMajor()`, `prod(n) × 3·M₁·M₂`; the default is the backend's
(`flux_patch_layout`). Without `flux`, `flux_patch` is empty and costs nothing. `N = 1` needs no
flux scratch.
"""
struct Workspace{VK,VP,AW,VI,VT,VS,XT<:Tuple,FP,FS,FL,RS,B}
    key::VK
    perm::VP
    ak::AW
    tmp_i32::VI
    tmp_T::VT
    tmp_u8::VS
    tmp_extras::XT
    seg_start::VI
    seg_stop::VI
    flux_patch::FP
    flux::FS
    flux_layout::FL
    resort::RS
    bins::B
end

function Workspace(s::Species{N,T}, grid::PeriodicGrid{N}; flux = nothing,
                   flux_layout = flux_patch_layout(get_backend(s)), append = nothing, bins = CellBins()) where {N,T}
    flux === nothing || N == 2 ||
        throw(ArgumentError("flux scratch is for N = 2; the N = 1 deposit_flux! needs none"))
    backend = get_backend(s)
    cap = capacity(s)
    key = KernelAbstractions.zeros(backend, Int64, cap)
    perm = KernelAbstractions.zeros(backend, Int, cap)
    ak = AK.workspace(AK.sort!, key)
    tmp_i32 = KernelAbstractions.zeros(backend, Int32, cap)
    tmp_T = KernelAbstractions.zeros(backend, T, cap)
    tmp_u8 = KernelAbstractions.zeros(backend, UInt8, cap)
    tmp_extras = extras_buffers(backend, cap, Tuple(values(s.extras)), (tmp_i32, tmp_T, tmp_u8))
    n = prod(grid.n)
    seg_start = KernelAbstractions.ones(backend, Int32, n)
    seg_stop = KernelAbstractions.zeros(backend, Int32, n)
    bins isa CellBins || bins isa NodeBins && length(bins.stagger) == N ||
        throw(ArgumentError("bins: CellBins() or NodeBins(shape, stagger) with one stagger per axis (N = $N), got $bins"))
    patch_size = flux === nothing ? (0, 0) : flux_patch_size(flux_layout, flux_patch_values(flux..., bins), n)
    flux_patch = KernelAbstractions.zeros(backend, T, patch_size...)
    resort = append === nothing ? nothing : ResortScratch(s, n, append)
    return Workspace(key, perm, ak, tmp_i32, tmp_T, tmp_u8, tmp_extras, seg_start, seg_stop, flux_patch, flux, flux_layout, resort,
                     bins)
end

# One buffer per extra: the first of `pool` with the extra's element type, else a new one,
# which joins the pool for the extras after it.
extras_buffers(backend, cap, ::Tuple{}, pool) = ()
function extras_buffers(backend, cap, extras::Tuple, pool)
    E = eltype(first(extras))
    i = findfirst(b -> eltype(b) === E, pool)
    buf = i === nothing ? KernelAbstractions.zeros(backend, E, cap) : pool[i]
    return (buf, extras_buffers(backend, cap, Base.tail(extras), i === nothing ? (pool..., buf) : pool)...)
end

"""
    check_workspace(ws, species, grid)

Throw an `ArgumentError` unless `ws` was built for a species of this capacity and a grid of this
many cells. The kernels that read the workspace run under `@inbounds`, so a mismatch would write
out of bounds. Lengths are static shapes, so the check is host-side and allowed by §0.4.
"""
function check_workspace(ws::Workspace, s::Species, grid::PeriodicGrid)
    length(ws.key) == capacity(s) ||
        throw(ArgumentError("workspace built for capacity $(length(ws.key)), species has capacity $(capacity(s))"))
    length(ws.seg_start) == prod(grid.n) ||
        throw(ArgumentError("workspace built for $(length(ws.seg_start)) cells, grid has $(prod(grid.n))"))
    return nothing
end

"Bytes of scratch held by a workspace (for the write-up's scratch-per-particle figure)."
function scratch_bytes(ws::Workspace)
    core = (ws.tmp_i32, ws.tmp_T, ws.tmp_u8)
    own = unique(objectid, Iterators.filter(b -> !any(c -> c === b, core), ws.tmp_extras))
    return sizeof(ws.key) + sizeof(ws.perm) + sizeof(ws.tmp_i32) + sizeof(ws.tmp_T) + sizeof(ws.tmp_u8) +
           sum(sizeof, own; init = 0) + sizeof(ws.seg_start) + sizeof(ws.seg_stop) + sizeof(ws.flux_patch) + AK.nbytes(ws.ak.plan) +
           resort_bytes(ws.resort)
end

# Total order: alive particles by (cell, slot), free slots after all of them by slot. Keys are
# unique, so every correct sort gives the same permutation on every backend and thread count.
@kernel function sort_key_kernel!(key, s, bins, n::Int64, cap::Int64)
    i = @index(Global, Linear)
    @inbounds begin
        c = ifelse(s.status[i] == ALIVE, Int64(particle_bin(bins, s, i, (Int32(n),))[1]) - 1, n)
        key[i] = c * cap + (i - 1)
    end
end

# The permutation from the sorted keys: each key ends in its slot (`key % cap` is `slot - 1`), so
# sorting the keys alone gives `sortperm`'s permutation, the keys being unique.
@kernel function perm_from_keys_kernel!(perm, @Const(key), cap::Int64)
    j = @index(Global, Linear)
    @inbounds perm[j] = Int(key[j] % cap) + 1
end

# Sort the keys in place with AcceleratedKernels' `sort!` (the algorithm its `Auto()` picks for the
# backend and length) and decode the permutation; `ws.key` is left sorted.
function sort_keys!(ws, cap, backend, workgroupsize)
    AK.sort!(ws.key; workspace = ws.ak)
    launcher(perm_from_keys_kernel!, backend, workgroupsize)(ws.perm, ws.key, Int64(cap); ndrange = cap)
    return nothing
end

@kernel function gather_perm_kernel!(dst, @Const(src), @Const(perm))
    i = @index(Global, Linear)
    @inbounds dst[i] = src[perm[i]]
end

@kernel function reset_segments_kernel!(seg_start, seg_stop, count)
    c = @index(Global, Linear)
    @inbounds begin
        seg_start[c] = Int32(1)
        seg_stop[c] = Int32(0)
        c == 1 && (count[1] = Int32(0))
    end
end

@kernel function segments_kernel!(seg_start, seg_stop, count, s, bins, n::Int32, cap::Int32)
    i32 = Int32(@index(Global, Linear))
    status = s.status
    bin(i) = particle_bin(bins, s, i, (n,))[1]
    @inbounds if status[i32] == ALIVE
        c = bin(i32)
        last_alive = i32 == cap || status[i32 + Int32(1)] != ALIVE
        if i32 == Int32(1) || bin(i32 - Int32(1)) != c
            seg_start[c] = i32
        end
        if last_alive || bin(i32 + Int32(1)) != c
            seg_stop[c] = i32
        end
        last_alive && (count[1] = i32)
    end
end

function permute!(a::AbstractVector, tmp::AbstractVector, perm, workgroupsize)
    launcher(gather_perm_kernel!, get_backend(a), workgroupsize)(tmp, a, perm; ndrange = length(a))
    copyto!(a, tmp)
    return a
end

permute_all!(::Tuple{}, ::Tuple{}, perm, workgroupsize) = nothing
function permute_all!(as::Tuple, tmps::Tuple, perm, workgroupsize)
    permute!(first(as), first(tmps), perm, workgroupsize)
    return permute_all!(Base.tail(as), Base.tail(tmps), perm, workgroupsize)
end

"Every SoA array of a species, the extras last, in the order `Species` lists them."
species_arrays(s::Species) = (s.cell..., s.offset..., s.u..., s.w, s.status, values(s.extras)...)

@inline copy_each!(::Tuple{}, ::Tuple{}, j, i) = nothing
@inline function copy_each!(d::Tuple, a::Tuple, j, i)
    @inbounds first(d)[j] = first(a)[i]
    return copy_each!(Base.tail(d), Base.tail(a), j, i)
end

# One work item per slot: slot `i` of every array of `dst` takes slot `perm[i]` of `src`.
@kernel function gather_all_kernel!(dst, @Const(src), @Const(perm))
    i = @index(Global, Linear)
    @inbounds copy_each!(dst, src, i, perm[i])
end

@kernel function copy_all_kernel!(dst, @Const(src))
    i = @index(Global, Linear)
    copy_each!(dst, src, i, i)
end

# Every SoA array of the species, the extras included, gathered by `ws.perm` (slot `i` takes what
# slot `ws.perm[i]` held) and copied back: in one launch each through the spare arrays of a
# workspace built with `append`, otherwise array by array through the shared buffers.
permute_species!(s::Species, ws::Workspace, workgroupsize) = permute_species!(s, ws, ws.resort, workgroupsize)
function permute_species!(s::Species, ws::Workspace, r, workgroupsize)
    a = species_arrays(s)
    launcher(gather_all_kernel!, get_backend(s), workgroupsize)(r.spare, a, ws.perm; ndrange = capacity(s))
    launcher(copy_all_kernel!, get_backend(s), workgroupsize)(a, r.spare; ndrange = capacity(s))
    return s
end
function permute_species!(s::Species{N}, ws::Workspace, ::Nothing, workgroupsize) where {N}
    for d in 1:N
        permute!(s.cell[d], ws.tmp_i32, ws.perm, workgroupsize)
        permute!(s.offset[d], ws.tmp_T, ws.perm, workgroupsize)
    end
    for d in 1:3
        permute!(s.u[d], ws.tmp_T, ws.perm, workgroupsize)
    end
    permute!(s.w, ws.tmp_T, ws.perm, workgroupsize)
    permute!(s.status, ws.tmp_u8, ws.perm, workgroupsize)
    permute_all!(Tuple(values(s.extras)), ws.tmp_extras, ws.perm, workgroupsize)
    return s
end

"""
    sort_by_cell!(species, grid, ws; workgroupsize = nothing)

Sort the particles in place by (cell, slot), with free slots last (for N > 1 by the column-major
linear cell of `linear_cell`, then slot); apply the one permutation to
every SoA array, the module's extras included (`ws.perm[i]` is the slot that slot `i` held
before); write the per-cell segment bounds into `ws` and the alive count into `species.count`.
Deterministic: the keys are unique, so the result does not depend on the sort algorithm, the
backend, the thread count or `workgroupsize` (A4; T-136, T-173, T-177). `workgroupsize`
applies to the engine's own kernels; AcceleratedKernels' sort launches as its workspace plan
says.
"""
function sort_by_cell!(s::Species{1}, grid::PeriodicGrid{1}, ws::Workspace; workgroupsize = nothing)
    check_workspace(ws, s, grid)
    backend = get_backend(s)
    cap = capacity(s)
    n = ncells(grid)
    launcher(sort_key_kernel!, backend, workgroupsize)(ws.key, s, ws.bins, Int64(n), Int64(cap); ndrange = cap)
    sort_keys!(ws, cap, backend, workgroupsize)
    permute_species!(s, ws, workgroupsize)
    launcher(reset_segments_kernel!, backend, workgroupsize)(ws.seg_start, ws.seg_stop, s.count; ndrange = n)
    launcher(segments_kernel!, backend, workgroupsize)(ws.seg_start, ws.seg_stop, s.count, s, ws.bins, Int32(n),
                                                       Int32(cap); ndrange = cap)
    note_sorted!(ws.resort, s)
    return s
end

# N axes: the key is `(linear cell - 1)·capacity + slot - 1`, free slots `prod(n)·capacity + slot - 1`,
# with the column-major linear cell of `linear_cell`; unique, so the permutation is fixed (A4).
@kernel function sort_key_nd_kernel!(key, s, bins, n, ncell::Int64, cap::Int64)
    i = @index(Global, Linear)
    @inbounds begin
        L = linear_cell(map(Int64, particle_bin(bins, s, i, map(Int32, n))), n)
        c = ifelse(s.status[i] == ALIVE, L - 1, ncell)
        key[i] = c * cap + (i - 1)
    end
end

@kernel function segments_nd_kernel!(seg_start, seg_stop, count, s, bins, n, cap::Int32)
    i32 = Int32(@index(Global, Linear))
    status = s.status
    bin(i) = linear_cell(particle_bin(bins, s, i, n), n)
    @inbounds if status[i32] == ALIVE
        c = bin(i32)
        last_alive = i32 == cap || status[i32 + Int32(1)] != ALIVE
        if i32 == Int32(1) || bin(i32 - Int32(1)) != c
            seg_start[c] = i32
        end
        if last_alive || bin(i32 + Int32(1)) != c
            seg_stop[c] = i32
        end
        last_alive && (count[1] = i32)
    end
end

function sort_by_cell!(s::Species{N}, grid::PeriodicGrid{N}, ws::Workspace; workgroupsize = nothing) where {N}
    check_workspace(ws, s, grid)
    backend = get_backend(s)
    cap = capacity(s)
    ncell = prod(grid.n)
    ncell <= typemax(Int32) || throw(ArgumentError("$ncell cells exceed the Int32 segment index"))
    (Int128(ncell) + 1) * cap <= typemax(Int64) || throw(ArgumentError("sort key overflows Int64: $ncell cells × capacity $cap"))
    launcher(sort_key_nd_kernel!, backend, workgroupsize)(ws.key, s, ws.bins, Int64.(grid.n), Int64(ncell), Int64(cap);
                                                          ndrange = cap)
    sort_keys!(ws, cap, backend, workgroupsize)
    permute_species!(s, ws, workgroupsize)
    launcher(reset_segments_kernel!, backend, workgroupsize)(ws.seg_start, ws.seg_stop, s.count; ndrange = ncell)
    launcher(segments_nd_kernel!, backend, workgroupsize)(ws.seg_start, ws.seg_stop, s.count, s, ws.bins,
                                                          Int32.(grid.n), Int32(cap); ndrange = cap)
    note_sorted!(ws.resort, s)
    return s
end
