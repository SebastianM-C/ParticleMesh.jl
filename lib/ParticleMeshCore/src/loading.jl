"The values of `load_quiet!`'s `density_at`; anything else is an `ArgumentError` (a host keyword, §0.4)."
check_density_at(a::Symbol) = a in (:particle, :cell_centre) ? a :
    throw(ArgumentError("density_at must be :particle or :cell_centre, got :$a"))

# The kernel argument of a checked `density_at` (a small Union, so the launch is union-split, not dispatched).
density_val(a::Symbol) = a === :cell_centre ? Val(:cell_centre) : Val(:particle)

# The in-cell offset at which a particle's density is evaluated: its own, or the cell centre.
@inline density_offset(::Val{:particle}, δ) = δ
@inline density_offset(::Val{:cell_centre}, δ) = one(δ) / 2

@kernel function load_quiet_kernel!(s, density, p, dx, ppc::Int32, first::Int32, vdrift, vth, key, at)
    i = @index(Global, Linear)
    T = eltype(s.w)
    c = Int32((i - 1) ÷ ppc + 1)
    k = Int32((i - 1) % ppc + 1)
    slot = first - Int32(1) + Int32(i)
    δ = (T(k) - one(T) / 2) / T(ppc)
    x = (T(c - Int32(1)) + density_offset(at, δ)) * dx
    # counter: (global cell, 0, 0, in-cell index k in the high 16 bits | draw index in the low 16 bits); `first` is not part of it
    ctr = (UInt32(c), UInt32(0), UInt32(0), UInt32(k) << 16)
    r = philox4x32(ctr, key)
    z1, z2 = box_muller(T, r[1], r[2])
    z3, _ = box_muller(T, r[3], r[4])
    @inbounds begin
        s.cell[1][slot] = c
        s.offset[1][slot] = δ
        s.u[1][slot] = vdrift[1] + vth[1] * z1
        s.u[2][slot] = vdrift[2] + vth[2] * z2
        s.u[3][slot] = vdrift[3] + vth[3] * z3
        s.w[slot] = density(x, p) / T(ppc)
        s.status[slot] = ALIVE
    end
end

"""
    load_quiet!(species, grid, density, p, ppc; vdrift, vth, seed, stream, first = 1, density_at = :particle, workgroupsize = nothing)

Quiet-start loading: `ppc` particles per cell on the lattice `δₖ = (k - 1/2)/ppc`, written into
slots `first:first-1+n·ppc`. The weight of a particle at `x` is `density(x, p)/ppc` (A3: the
parameters `p` enter through the weights only; `density` is an isbits callable). With
`density_at = :cell_centre` the density is evaluated at the centre of the particle's cell
instead, `density((c - 1/2)·dx, p)/ppc`, so the loaded profile is piecewise constant per cell, as
Smilei and PIConGPU sample it; positions and draws are the same either way, and the default
`:particle` keeps the weights bit for bit. Anything else is an `ArgumentError`. Each velocity
component is `vdrift[d] + vth[d]·z` with `z` standard normal from Box–Muller over Philox4x32-10
draws, keyed by `(seed, stream)` with the counter `(cell, 0, 0, k << 16)` (A4), so a seed means
the same thing on every backend and thread count. Sets the alive count to the last loaded slot.
"""
function load_quiet!(s::Species{1,T}, grid::PeriodicGrid{1}, density, p, ppc::Integer;
                     vdrift = (zero(T), zero(T), zero(T)), vth = (zero(T), zero(T), zero(T)),
                     seed::Integer, stream::Integer, first::Integer = 1, density_at::Symbol = :particle,
                     workgroupsize = nothing) where {T}
    check_density_at(density_at)
    n = ncells(grid)
    0 < ppc < 2^16 || throw(ArgumentError("ppc must be in 1:65535 for the counter layout"))
    last = first - 1 + n * ppc
    last <= capacity(s) || throw(ArgumentError("capacity $(capacity(s)) < $last particles"))
    key = (UInt32(seed), UInt32(stream))
    launcher(load_quiet_kernel!, get_backend(s), workgroupsize)(s, density, p, cellsize(grid), Int32(ppc), Int32(first),
                                                                T.(vdrift), T.(vth), key, density_val(density_at); ndrange = n * ppc)
    fill!(s.count, Int32(last))
    return s
end

# N ≥ 2: the per-axis product lattice, one set of device functions and one kernel for every N.

# One Philox counter word per axis for the particle's global cell, zero for an axis the grid does
# not have, and the in-cell index in the high 16 bits of the last word (its low 16 bits are the
# draw index, `with_draw`). At N = 1 it is the counter of the 1D load.
@inline cell_counter(c::NTuple{1,Int32}, k::Int32) = (UInt32(c[1]), UInt32(0), UInt32(0), UInt32(k) << 16)
@inline cell_counter(c::NTuple{2,Int32}, k::Int32) = (UInt32(c[1]), UInt32(c[2]), UInt32(0), UInt32(k) << 16)
@inline cell_counter(c::NTuple{3,Int32}, k::Int32) = (UInt32(c[1]), UInt32(c[2]), UInt32(c[3]), UInt32(k) << 16)

"""
    quiet_lattice(T, c::NTuple{N,Int32}, k, ppc::NTuple{N,Int32}, dx, at) -> (δ, x, ctr)

Where the quiet start puts the particle with the in-cell index `k` (`1:prod(ppc)`, axis 1 fastest)
of the cell `c`: its offsets on the per-axis lattice `δ_d = (k_d − 1/2)/ppc_d`, the position `x`
at which its density is evaluated (`((c_d − 1 + δ_d)·dx_d)`, or the cell centre with
`at = Val(:cell_centre)`) and its Philox counter `(c₁, c₂, c₃, k << 16)`, with zero for an axis
the grid does not have. A device function, in the element type `T`.
"""
@inline function quiet_lattice(::Type{T}, c::NTuple{N,Int32}, k::Int32, ppc::NTuple{N,Int32}, dx, at) where {T,N}
    kd = cartesian_cell(k, ppc)
    δ = ntuple(d -> (T(kd[d]) - one(T) / 2) / T(ppc[d]), Val(N))
    x = ntuple(d -> (T(c[d] - Int32(1)) + density_offset(at, δ[d])) * dx[d], Val(N))
    return δ, x, cell_counter(c, k)
end

"""
    quiet_particle(c::NTuple{N,Int32}, k, ppc::NTuple{N,Int32}, dx, density, p, vdrift, vth, key, at = Val(:particle)) -> (δ, u, w)

One particle of the quiet start on N ≥ 2 axes (a device function): the in-cell index `k` (axis 1
fastest) of the cell `c` on the per-axis lattice `δ_d = (k_d − 1/2)/ppc_d`, its velocity
`vdrift[d] + vth[d]·z_d` from Box–Muller over Philox4x32-10 with the counter
`(c₁, c₂, c₃, k << 16)` (`c₃ = 0` at N = 2) and `key`, and its weight `density(x, p)/prod(ppc)` at
the position `x = ((c_d − 1 + δ_d)·dx_d)`, or with `at = Val(:cell_centre)` at the cell centre
(`load_quiet!`'s `density_at`). The cell is the global one: `load_quiet!` passes the grid's cells,
and a module that loads a part of a larger domain (the moving window's refill) passes the cell in
that domain, so the draws and the weight of a cell do not depend on which call loads it (A3, A4).
"""
@inline function quiet_particle(c::NTuple{N,Int32}, k::Int32, ppc::NTuple{N,Int32}, dx, density, p, vdrift, vth, key,
                                at = Val(:particle)) where {N}
    T = eltype(vth)
    δ, x, ctr = quiet_lattice(T, c, k, ppc, dx, at)
    r = philox4x32(ctr, key)
    z1, z2 = box_muller(T, r[1], r[2])
    z3, _ = box_muller(T, r[3], r[4])
    u = (vdrift[1] + vth[1] * z1, vdrift[2] + vth[2] * z2, vdrift[3] + vth[3] * z3)
    return δ, u, density(x, p) / T(prod(ppc))
end

"""
    quiet_particle_2d(c1, c2, k, ppc, dx, density, p, vdrift, vth, key, at = Val(:particle)) -> (δ, u, w)

[`quiet_particle`](@ref) of the cell `(c1, c2)` at N = 2.
"""
@inline quiet_particle_2d(c1::Int32, c2::Int32, k::Int32, ppc::NTuple{2,Int32}, dx, density, p, vdrift, vth, key, at = Val(:particle)) =
    quiet_particle((c1, c2), k, ppc, dx, density, p, vdrift, vth, key, at)

# Entry `slot` of each array of a tuple, from the tuple of values.
@inline function store_axes!(arrays::NTuple{N,Any}, slot, values::NTuple{N,Any}) where {N}
    ntuple(d -> (@inbounds arrays[d][slot] = values[d]; nothing), Val(N))
    return nothing
end

# The linear cell `L` and the in-cell index `k` of work item `i`, and its slot: particle `i` is the
# in-cell index `k` (axis 1 fastest) of linear cell `L`, so the loaded slots are in sorted order.
@inline function quiet_item(i, n::NTuple{N,Int32}, ppc::NTuple{N,Int32}, first::Int32) where {N}
    K = prod(ppc)
    L = Int32((i - 1) ÷ K + 1)
    k = Int32((i - 1) % K + 1)
    return cartesian_cell(L, n), k, first - Int32(1) + Int32(i)
end

@kernel function load_quiet_nd_kernel!(s, density, p, dx, n::NTuple{N,Int32}, ppc::NTuple{N,Int32}, first::Int32, vdrift, vth, key,
                                       at) where {N}
    i = @index(Global, Linear)
    c, k, slot = quiet_item(i, n, ppc, first)
    δ, u, w = quiet_particle(c, k, ppc, dx, density, p, vdrift, vth, key, at)
    store_axes!(s.cell, slot, c)
    store_axes!(s.offset, slot, δ)
    store_axes!(s.u, slot, u)
    @inbounds begin
        s.w[slot] = w
        s.status[slot] = ALIVE
    end
end

# The host checks of the N ≥ 2 load: the last slot it writes.
function check_quiet_load(s::Species, grid::PeriodicGrid{N}, ppc::NTuple{N,Integer}, first::Integer) where {N}
    all(>(0), ppc) && prod(ppc) < 2^16 || throw(ArgumentError("ppc = $ppc: each ≥ 1 and prod(ppc) < 2^16 for the counter layout"))
    total = prod(ncells(grid)) * prod(ppc)
    last = first - 1 + total
    last <= capacity(s) || throw(ArgumentError("capacity $(capacity(s)) < $last particles"))
    return total, last
end

"""
    load_quiet!(species::Species{N}, grid::PeriodicGrid{N}, density, p, ppc::NTuple{N}; vdrift, vth, seed, stream, first = 1, density_at = :particle, workgroupsize = nothing)

The quiet start on N = 2 or 3 axes: `prod(ppc)` particles per cell on the per-axis product lattice
`δ_d = (k_d - 1/2)/ppc_d`, written cell by cell in column-major linear order (axis 1 fastest)
and within a cell by the in-cell index `k = k₁ + ppc₁(k₂ - 1) + ppc₁ppc₂(k₃ - 1)`, also axis 1
fastest (`prod(ppc) < 2^16`). The weight is `density(x, p)/prod(ppc)` with `x` an `NTuple{N}`
position, so the parameters enter through the weights only (A3); the Philox counter is
`(cell₁, cell₂, cell₃, k << 16)`, with `cell₃ = 0` at N = 2 (A4). `density_at = :cell_centre`
evaluates the density at the cell centre `((c_d - 1/2)·dx_d)` instead, as in 1D. Sets the alive
count to the last loaded slot.
"""
function load_quiet!(s::Species{N,T}, grid::PeriodicGrid{N}, density, p, ppc::NTuple{N,Integer};
                     vdrift = (zero(T), zero(T), zero(T)), vth = (zero(T), zero(T), zero(T)),
                     seed::Integer, stream::Integer, first::Integer = 1, density_at::Symbol = :particle,
                     workgroupsize = nothing) where {N,T}
    check_density_at(density_at)
    total, last = check_quiet_load(s, grid, ppc, first)
    key = (UInt32(seed), UInt32(stream))
    launcher(load_quiet_nd_kernel!, get_backend(s), workgroupsize)(s, density, p, cellsize(grid), Int32.(ncells(grid)), Int32.(ppc),
                                                                   Int32(first), T.(vdrift), T.(vth), key, density_val(density_at);
                                                                   ndrange = total)
    fill!(s.count, Int32(last))
    return s
end
