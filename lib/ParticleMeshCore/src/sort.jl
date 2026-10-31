"""
    Workspace(species, grid)

Scratch for one species on one grid, allocated once: the sort keys and permutation, the
AcceleratedKernels sort workspace, one permutation buffer per element type of the SoA arrays,
and the per-cell segment bounds `seg_start`, `seg_stop` that `sort_by_cell!` writes and the
`Reference` deposition reads (an empty cell has `seg_start = 1`, `seg_stop = 0`).
"""
struct Workspace{VK,VP,AW,VI,VT,VS}
    key::VK
    perm::VP
    ak::AW
    tmp_i32::VI
    tmp_T::VT
    tmp_u8::VS
    seg_start::VI
    seg_stop::VI
end

function Workspace(s::Species{1,T}, grid::PeriodicGrid{1}) where {T}
    backend = get_backend(s)
    cap = capacity(s)
    key = KernelAbstractions.zeros(backend, Int64, cap)
    perm = KernelAbstractions.zeros(backend, Int, cap)
    ak = AK.workspace(AK.sortperm!, perm, key)
    tmp_i32 = KernelAbstractions.zeros(backend, Int32, cap)
    tmp_T = KernelAbstractions.zeros(backend, T, cap)
    tmp_u8 = KernelAbstractions.zeros(backend, UInt8, cap)
    n = ncells(grid)
    seg_start = KernelAbstractions.ones(backend, Int32, n)
    seg_stop = KernelAbstractions.zeros(backend, Int32, n)
    return Workspace(key, perm, ak, tmp_i32, tmp_T, tmp_u8, seg_start, seg_stop)
end

"Bytes of scratch held by a workspace (for the write-up's scratch-per-particle figure)."
scratch_bytes(ws::Workspace) =
    sizeof(ws.key) + sizeof(ws.perm) + sizeof(ws.tmp_i32) + sizeof(ws.tmp_T) + sizeof(ws.tmp_u8) +
    sizeof(ws.seg_start) + sizeof(ws.seg_stop) + AK.nbytes(ws.ak.plan)

# Total order: alive particles by (cell, slot), free slots after all of them by slot. Keys are
# unique, so every correct sort gives the same permutation on every backend and thread count.
@kernel function sort_key_kernel!(key, cell, status, n::Int64, cap::Int64)
    i = @index(Global, Linear)
    @inbounds begin
        c = ifelse(status[i] == ALIVE, Int64(cell[i]) - 1, n)
        key[i] = c * cap + (i - 1)
    end
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

@kernel function segments_kernel!(seg_start, seg_stop, count, @Const(cell), @Const(status), cap::Int32)
    i32 = Int32(@index(Global, Linear))
    @inbounds if status[i32] == ALIVE
        c = cell[i32]
        last_alive = i32 == cap || status[i32 + Int32(1)] != ALIVE
        if i32 == Int32(1) || cell[i32 - Int32(1)] != c
            seg_start[c] = i32
        end
        if last_alive || cell[i32 + Int32(1)] != c
            seg_stop[c] = i32
        end
        last_alive && (count[1] = i32)
    end
end

function permute!(a::AbstractVector, tmp::AbstractVector, perm)
    backend = get_backend(a)
    gather_perm_kernel!(backend)(tmp, a, perm; ndrange = length(a))
    copyto!(a, tmp)
    return a
end

"""
    sort_by_cell!(species, grid, ws)

Sort the particles in place by (cell, slot), with free slots last; apply the one permutation to
every SoA array; write the per-cell segment bounds into `ws` and the alive count into
`species.count`. Deterministic: the keys are unique, so the result does not depend on the sort
algorithm, the backend or the thread count (A4).
"""
function sort_by_cell!(s::Species{1}, grid::PeriodicGrid{1}, ws::Workspace)
    backend = get_backend(s)
    cap = capacity(s)
    n = ncells(grid)
    sort_key_kernel!(backend)(ws.key, s.cell[1], s.status, Int64(n), Int64(cap); ndrange = cap)
    AK.sortperm!(ws.perm, ws.key; workspace = ws.ak)
    permute!(s.cell[1], ws.tmp_i32, ws.perm)
    permute!(s.offset[1], ws.tmp_T, ws.perm)
    for d in 1:3
        permute!(s.u[d], ws.tmp_T, ws.perm)
    end
    permute!(s.w, ws.tmp_T, ws.perm)
    permute!(s.status, ws.tmp_u8, ws.perm)
    reset_segments_kernel!(backend)(ws.seg_start, ws.seg_stop, s.count; ndrange = n)
    segments_kernel!(backend)(ws.seg_start, ws.seg_stop, s.count, s.cell[1], s.status, Int32(cap);
                              ndrange = cap)
    return s
end
